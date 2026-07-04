/*
	BitonicPixelSorter_PathGeometry.cpp

	Path mode treats the selected Bezier mask as a direction field and a local
	coordinate system:

	  1. The Bezier is flattened into an arc-length polyline at sufficient
	     precision (BpsPathSample: position, arc length s, unit tangent T).
	  2. For every pixel p we find the closest point q on the polyline, its arc
	     length s, the tangent T and normal N at q, and the signed distance
	     d = dot(p - q, N) (left of the tangent is positive).
	  3. Open paths extend their first/last segments as half-lines so s < 0 and
	     s > pathLength are defined out to the frame extent. Closed paths are not
	     extended; s wraps modulo pathLength.
	  4. Tangent mode builds a sort space parallel to the path:
	         lane = quantize(d),  order = s
	     Normal mode builds a sort space perpendicular to the path:
	         lane = quantize(s),  order = d
	  5. Each lane's slots are laid out in `order`, the pixels that fall in the
	     lane are sorted by the chosen key, and written back into those slots.

	Steps 1-4 live here and produce the pixel-owned records consumed identically
	by the CPU (SortMappedPixels) and GPU (mapped_sort_kernel) back ends.
*/

#include "BitonicPixelSorter_PathGeometry.h"

#include "AEFX_SuiteHelper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEps = 1.0e-6f;

inline A_long RoundToLong(float value)
{
	return static_cast<A_long>(std::floor(value + 0.5f));
}

inline std::uint32_t NextPow2U32(std::uint32_t value)
{
	if (value <= 1u) {
		return 1u;
	}
	value--;
	value |= value >> 1;
	value |= value >> 2;
	value |= value >> 4;
	value |= value >> 8;
	value |= value >> 16;
	return value + 1u;
}

inline float WrapPositiveRadians(float value)
{
	value = std::fmod(value, 2.0f * kPi);
	if (value < 0.0f) {
		value += 2.0f * kPi;
	}
	return value;
}

// Wrap an arc length into [0, length) for closed paths.
inline float WrapArcLength(float s, float length)
{
	if (length <= kEps) {
		return s;
	}
	float w = std::fmod(s, length);
	if (w < 0.0f) {
		w += length;
	}
	return w;
}

inline void NormaliseTangent(float *tx, float *ty)
{
	const float len = std::sqrt((*tx) * (*tx) + (*ty) * (*ty));
	if (len > kEps) {
		*tx /= len;
		*ty /= len;
	} else {
		*tx = 1.0f;
		*ty = 0.0f;
	}
}

void AppendSample(std::vector<BpsPathSample> *samplesP,
				  float x, float y, float s, float tx, float ty)
{
	NormaliseTangent(&tx, &ty);
	if (!samplesP->empty()) {
		const BpsPathSample &prev = samplesP->back();
		const float dx = x - prev.x;
		const float dy = y - prev.y;
		if (dx * dx + dy * dy < 1.0e-8f && std::fabs(s - prev.s) < kEps) {
			return;
		}
	}
	BpsPathSample sample = {x, y, s, tx, ty};
	samplesP->push_back(sample);
}

//-----------------------------------------------------------------------------
// (s, n) -> (lane, order) mapping. This single definition keeps the domain
// bounds (ResolvePathDomainBounds) and the per-pixel classification consistent.
//-----------------------------------------------------------------------------
inline bool PathLaneOrder(const BitonicSorterParams &prm,
						  float s,
						  float n,
						  A_long *lane_out,
						  float *order_out)
{
	const bool closed = (prm.pathClosed != 0) && (prm.pathLength > kEps);
	const float s_local = closed ? WrapArcLength(s, prm.pathLength) : s;

	A_long lane = 0;
	float order = 0.0f;

	if (prm.pathDirection == BPS_PATH_DIR_TANGENT) {
		// Parallel to the path: one lane per quantised normal distance,
		// ordered along the path by arc length.
		lane = RoundToLong(n) - prm.pathNMin;
		order = s_local;
	} else {
		// Perpendicular to the path: one lane per quantised arc length,
		// ordered across the path by signed distance.
		if (closed) {
			const A_long bins = (prm.domainLineCount > 0) ? prm.domainLineCount : 1;
			A_long q = RoundToLong(s_local) % bins;
			if (q < 0) {
				q += bins;
			}
			lane = q;
		} else {
			lane = RoundToLong(s) - prm.pathSMin;
		}
		order = n;
	}

	if (lane < 0 || lane >= prm.domainLineCount) {
		return false;
	}
	*lane_out = lane;
	*order_out = order;
	return true;
}

//-----------------------------------------------------------------------------
// Domain bounds. pathLength / pathClosed must already be set on paramsP.
//-----------------------------------------------------------------------------
void ResolvePathDomainBounds(BitonicSorterParams *paramsP, float diag)
{
	if (!paramsP || paramsP->pathSampleCount < 2 || !paramsP->pathSamples) {
		if (paramsP) {
			paramsP->pathSMin = 0;
			paramsP->pathNMin = 0;
			paramsP->domainLineCount = 0;
			paramsP->domainMaxLineLength = 0;
			paramsP->maxLineLength = 0;
		}
		return;
	}

	const BpsPathSample *samples = paramsP->pathSamples;
	float s_min = samples[0].s;
	float s_max = samples[0].s;
	for (A_long i = 1; i < paramsP->pathSampleCount; ++i) {
		s_min = (std::min)(s_min, samples[i].s);
		s_max = (std::max)(s_max, samples[i].s);
	}

	// Signed-distance domain: symmetric, large enough to cover the whole frame
	// so every pixel has a defined (lane, order) in either direction.
	const float n_extent = (std::max)(diag, 1.0f);
	paramsP->pathNMin = RoundToLong(std::floor(-n_extent));
	const A_long path_n_max = RoundToLong(std::ceil(n_extent));
	const A_long n_len = path_n_max - paramsP->pathNMin + 1;

	A_long s_len = 0;
	if (paramsP->pathClosed && paramsP->pathLength > kEps) {
		// Closed: arc length wraps, so lanes span exactly one perimeter.
		paramsP->pathSMin = 0;
		s_len = (std::max)(static_cast<A_long>(1), RoundToLong(paramsP->pathLength));
	} else {
		// Open: extended samples carry s below 0 and above pathLength.
		paramsP->pathSMin = RoundToLong(std::floor(s_min));
		const A_long path_s_max = RoundToLong(std::ceil(s_max));
		s_len = path_s_max - paramsP->pathSMin + 1;
	}

	if (paramsP->pathDirection == BPS_PATH_DIR_NORMAL) {
		paramsP->domainLineCount = s_len;
		paramsP->domainMaxLineLength = n_len;
	} else {
		paramsP->domainLineCount = n_len;
		paramsP->domainMaxLineLength = s_len;
	}

	if (paramsP->domainLineCount < 1) {
		paramsP->domainLineCount = 1;
	}
	if (paramsP->domainMaxLineLength < 1) {
		paramsP->domainMaxLineLength = 1;
	}
	paramsP->maxLineLength = paramsP->domainMaxLineLength;
}

// Extend both open ends as straight half-lines by `extend` along the end
// tangents, so pixels beyond the path endpoints still project onto a segment.
void ExtendOpenEnds(std::vector<BpsPathSample> *samplesP, float extend)
{
	if (!samplesP || samplesP->size() < 2u || extend <= kEps) {
		return;
	}

	const BpsPathSample first = samplesP->front();
	BpsPathSample start_ext = first;
	start_ext.x = first.x - first.tx * extend;
	start_ext.y = first.y - first.ty * extend;
	start_ext.s = first.s - extend;
	start_ext.tx = first.tx;
	start_ext.ty = first.ty;
	samplesP->insert(samplesP->begin(), start_ext);

	const BpsPathSample last = samplesP->back();
	BpsPathSample end_ext = last;
	end_ext.x = last.x + last.tx * extend;
	end_ext.y = last.y + last.ty * extend;
	end_ext.s = last.s + extend;
	end_ext.tx = last.tx;
	end_ext.ty = last.ty;
	samplesP->push_back(end_ext);
}

// Flatten one cubic Bezier segment into arc-length samples.
void SampleCubicSegment(std::vector<BpsPathSample> *samplesP,
						float *cumulative_s,
						float *prev_x,
						float *prev_y,
						bool *have_prev,
						float p0x, float p0y,
						float p1x, float p1y,
						float p2x, float p2y,
						float p3x, float p3y,
						bool skip_first)
{
	const float chord = std::sqrt((p3x - p0x) * (p3x - p0x) +
								 (p3y - p0y) * (p3y - p0y));
	const float ctrl = std::sqrt((p1x - p0x) * (p1x - p0x) +
								(p1y - p0y) * (p1y - p0y)) +
					  std::sqrt((p2x - p1x) * (p2x - p1x) +
								(p2y - p1y) * (p2y - p1y)) +
					  std::sqrt((p3x - p2x) * (p3x - p2x) +
								(p3y - p2y) * (p3y - p2y));
	// One sample per pixel of estimated length, with a floor for tight curves.
	const float est_len = (std::max)(chord, ctrl * 0.5f);
	const A_long steps = (std::max)(
		static_cast<A_long>(8),
		static_cast<A_long>(std::ceil(est_len)));

	for (A_long step = 0; step <= steps; ++step) {
		if (skip_first && step == 0) {
			continue;
		}
		const float t = static_cast<float>(step) / static_cast<float>(steps);
		const float u = 1.0f - t;
		const float uu = u * u;
		const float tt = t * t;
		const float x = uu * u * p0x + 3.0f * uu * t * p1x +
						3.0f * u * tt * p2x + tt * t * p3x;
		const float y = uu * u * p0y + 3.0f * uu * t * p1y +
						3.0f * u * tt * p2y + tt * t * p3y;
		float tx = 3.0f * uu * (p1x - p0x) +
				   6.0f * u * t * (p2x - p1x) +
				   3.0f * tt * (p3x - p2x);
		float ty = 3.0f * uu * (p1y - p0y) +
				   6.0f * u * t * (p2y - p1y) +
				   3.0f * tt * (p3y - p2y);
		NormaliseTangent(&tx, &ty);
		if (*have_prev) {
			const float dx = x - *prev_x;
			const float dy = y - *prev_y;
			*cumulative_s += std::sqrt(dx * dx + dy * dy);
		}
		AppendSample(samplesP, x, y, *cumulative_s, tx, ty);
		*prev_x = x;
		*prev_y = y;
		*have_prev = true;
	}
}

// Sample from PF_PathOutline (PathMaster pattern: always pass effect_ref).
bool SamplePathFromPfOutline(PF_ProgPtr effect_ref,
							 PF_PathDataSuite1 *path_data,
							 PF_PathOutlinePtr pathP,
							 std::vector<BpsPathSample> *samplesP,
							 PF_Boolean *open_out)
{
	samplesP->clear();
	if (!path_data || !pathP) {
		return false;
	}

	PF_Boolean openB = TRUE;
	(void)path_data->PF_PathIsOpen(effect_ref, pathP, &openB);
	if (open_out) {
		*open_out = openB;
	}

	A_long num_segments = 0;
	if (path_data->PF_PathNumSegments(effect_ref, pathP, &num_segments) ||
		num_segments <= 0) {
		return false;
	}

	float cumulative_s = 0.0f;
	float prev_x = 0.0f;
	float prev_y = 0.0f;
	bool have_prev = false;

	for (A_long seg = 0; seg < num_segments; ++seg) {
		PF_PathVertex v0 = {};
		PF_PathVertex v1 = {};
		if (path_data->PF_PathVertexInfo(effect_ref, pathP, seg, &v0) ||
			path_data->PF_PathVertexInfo(effect_ref, pathP, seg + 1, &v1)) {
			continue;
		}

		const float p0x = static_cast<float>(v0.x);
		const float p0y = static_cast<float>(v0.y);
		const float p3x = static_cast<float>(v1.x);
		const float p3y = static_cast<float>(v1.y);
		const float p1x = p0x + static_cast<float>(v0.tan_out_x);
		const float p1y = p0y + static_cast<float>(v0.tan_out_y);
		const float p2x = p3x + static_cast<float>(v1.tan_in_x);
		const float p2y = p3y + static_cast<float>(v1.tan_in_y);

		SampleCubicSegment(samplesP, &cumulative_s, &prev_x, &prev_y, &have_prev,
						   p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, seg > 0);
	}

	return samplesP->size() >= 2u;
}

// PathMaster/Projector pattern: layer masks via AEGP, outline vertices via
// MaskOutlineSuite (tangents relative to position).
bool SamplePathFromAegpMask(PF_InData *in_data,
							AEGP_SuiteHandler &suites,
							AEGP_PluginID plugin_id,
							PF_PathID path_id,
							std::vector<BpsPathSample> *samplesP,
							PF_Boolean *open_out)
{
	samplesP->clear();
	if (!in_data || !plugin_id || path_id == 0 ||
		!suites.PFInterfaceSuite1() || !suites.MaskSuite6() ||
		!suites.StreamSuite5() || !suites.MaskOutlineSuite3()) {
		return false;
	}

	AEGP_LayerH layerH = NULL;
	if (suites.PFInterfaceSuite1()->AEGP_GetEffectLayer(
			in_data->effect_ref, &layerH) || !layerH) {
		return false;
	}

	A_long num_masks = 0;
	if (suites.MaskSuite6()->AEGP_GetLayerNumMasks(layerH, &num_masks) ||
		num_masks <= 0) {
		return false;
	}

	AEGP_MaskRefH maskH = NULL;
	// Prefer 1-based mask index ("Mask 1" => index 0), then match by mask ID.
	if (path_id >= 1 && path_id <= num_masks) {
		(void)suites.MaskSuite6()->AEGP_GetLayerMaskByIndex(
			layerH, path_id - 1, &maskH);
	}
	if (!maskH) {
		for (A_long i = 0; i < num_masks; ++i) {
			AEGP_MaskRefH candidate = NULL;
			if (suites.MaskSuite6()->AEGP_GetLayerMaskByIndex(
					layerH, i, &candidate) || !candidate) {
				continue;
			}
			AEGP_MaskIDVal mask_id = 0;
			if (suites.MaskSuite6()->AEGP_GetMaskID(candidate, &mask_id) ==
					A_Err_NONE &&
				static_cast<PF_PathID>(mask_id) == path_id) {
				maskH = candidate;
				break;
			}
			(void)suites.MaskSuite6()->AEGP_DisposeMask(candidate);
		}
	}
	if (!maskH) {
		return false;
	}

	AEGP_StreamRefH streamH = NULL;
	A_Err aerr = suites.StreamSuite5()->AEGP_GetNewMaskStream(
		plugin_id, maskH, AEGP_MaskStream_OUTLINE, &streamH);
	if (aerr || !streamH) {
		(void)suites.MaskSuite6()->AEGP_DisposeMask(maskH);
		return false;
	}

	A_Time timeT;
	timeT.value = in_data->current_time;
	timeT.scale = in_data->time_scale;

	AEGP_StreamValue2 outline_val;
	AEFX_CLR_STRUCT(outline_val);
	aerr = suites.StreamSuite5()->AEGP_GetNewStreamValue(
		plugin_id, streamH, AEGP_LTimeMode_LayerTime, &timeT, TRUE, &outline_val);

	bool ok = false;
	if (!aerr && outline_val.val.mask) {
		AEGP_MaskOutlineValH outlineH = outline_val.val.mask;
		A_Boolean openB = TRUE;
		(void)suites.MaskOutlineSuite3()->AEGP_IsMaskOutlineOpen(outlineH, &openB);
		if (open_out) {
			*open_out = openB ? TRUE : FALSE;
		}

		A_long num_segments = 0;
		(void)suites.MaskOutlineSuite3()->AEGP_GetMaskOutlineNumSegments(
			outlineH, &num_segments);

		float cumulative_s = 0.0f;
		float prev_x = 0.0f;
		float prev_y = 0.0f;
		bool have_prev = false;

		for (A_long seg = 0; seg < num_segments; ++seg) {
			AEGP_MaskVertex v0 = {};
			AEGP_MaskVertex v1 = {};
			if (suites.MaskOutlineSuite3()->AEGP_GetMaskOutlineVertexInfo(
					outlineH, seg, &v0) ||
				suites.MaskOutlineSuite3()->AEGP_GetMaskOutlineVertexInfo(
					outlineH, seg + 1, &v1)) {
				continue;
			}

			// Tangents are relative to position (AEGP docs / PathMaster family).
			const float p0x = static_cast<float>(v0.x);
			const float p0y = static_cast<float>(v0.y);
			const float p3x = static_cast<float>(v1.x);
			const float p3y = static_cast<float>(v1.y);
			const float p1x = p0x + static_cast<float>(v0.tan_out_x);
			const float p1y = p0y + static_cast<float>(v0.tan_out_y);
			const float p2x = p3x + static_cast<float>(v1.tan_in_x);
			const float p2y = p3y + static_cast<float>(v1.tan_in_y);

			SampleCubicSegment(samplesP, &cumulative_s, &prev_x, &prev_y,
							   &have_prev, p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y,
							   seg > 0);
		}
		ok = samplesP->size() >= 2u;
	}

	(void)suites.StreamSuite5()->AEGP_DisposeStreamValue(&outline_val);
	(void)suites.StreamSuite5()->AEGP_DisposeStream(streamH);
	(void)suites.MaskSuite6()->AEGP_DisposeMask(maskH);
	return ok;
}

} // namespace

bool BPS_PathEvalAtS(
	const BpsPathSample *samples,
	A_long sample_count,
	float s,
	float *x_out,
	float *y_out,
	float *tx_out,
	float *ty_out)
{
	if (!samples || sample_count < 2 || !x_out || !y_out) {
		return false;
	}

	if (s <= samples[0].s) {
		*x_out = samples[0].x;
		*y_out = samples[0].y;
		if (tx_out) *tx_out = samples[0].tx;
		if (ty_out) *ty_out = samples[0].ty;
		return true;
	}
	if (s >= samples[sample_count - 1].s) {
		*x_out = samples[sample_count - 1].x;
		*y_out = samples[sample_count - 1].y;
		if (tx_out) *tx_out = samples[sample_count - 1].tx;
		if (ty_out) *ty_out = samples[sample_count - 1].ty;
		return true;
	}

	for (A_long i = 0; i < sample_count - 1; ++i) {
		const BpsPathSample &a = samples[i];
		const BpsPathSample &b = samples[i + 1];
		if (s < a.s || s > b.s) {
			continue;
		}
		const float seg = b.s - a.s;
		const float t = (seg > kEps) ? ((s - a.s) / seg) : 0.0f;
		*x_out = a.x + (b.x - a.x) * t;
		*y_out = a.y + (b.y - a.y) * t;
		float tx = a.tx + (b.tx - a.tx) * t;
		float ty = a.ty + (b.ty - a.ty) * t;
		NormaliseTangent(&tx, &ty);
		if (tx_out) *tx_out = tx;
		if (ty_out) *ty_out = ty;
		return true;
	}
	return false;
}

bool BPS_PathClosestPoint(
	const BpsPathSample *samples,
	A_long sample_count,
	float px,
	float py,
	float *s_out,
	float *n_out)
{
	if (!samples || sample_count < 2 || !s_out || !n_out) {
		return false;
	}

	float best_dist2 = std::numeric_limits<float>::infinity();
	float best_s = samples[0].s;
	float best_n = 0.0f;

	for (A_long i = 0; i < sample_count - 1; ++i) {
		const BpsPathSample &a = samples[i];
		const BpsPathSample &b = samples[i + 1];
		const float abx = b.x - a.x;
		const float aby = b.y - a.y;
		const float ab_len2 = abx * abx + aby * aby;
		float t = 0.0f;
		if (ab_len2 > kEps) {
			t = ((px - a.x) * abx + (py - a.y) * aby) / ab_len2;
			t = (std::max)(0.0f, (std::min)(1.0f, t));
		}
		const float cx = a.x + abx * t;
		const float cy = a.y + aby * t;
		const float dx = px - cx;
		const float dy = py - cy;
		const float dist2 = dx * dx + dy * dy;
		const float candidate_s = a.s + (b.s - a.s) * t;
		const bool closer = dist2 < best_dist2 - 1.0e-5f;
		const bool tie_by_s =
			std::fabs(dist2 - best_dist2) <= 1.0e-5f &&
			candidate_s < best_s - 1.0e-5f;
		if (closer || tie_by_s) {
			best_dist2 = dist2;
			best_s = candidate_s;
			// Unit tangent at the closest point; N = rotate(T, +90) = (-ty, tx).
			// Signed distance d = dot(p - q, N) (left of the tangent positive).
			float tx = a.tx + (b.tx - a.tx) * t;
			float ty = a.ty + (b.ty - a.ty) * t;
			NormaliseTangent(&tx, &ty);
			best_n = -ty * dx + tx * dy;
		}
	}

	*s_out = best_s;
	*n_out = best_n;
	return true;
}

bool BPS_PathCoordForPos(
	const BitonicSorterParams &prm,
	A_long line,
	A_long pos,
	A_long *x_out,
	A_long *y_out)
{
	if (!x_out || !y_out || !prm.pathSamples || prm.pathSampleCount < 2) {
		return false;
	}

	float s = 0.0f;
	float n = 0.0f;
	if (prm.pathDirection == BPS_PATH_DIR_NORMAL) {
		s = static_cast<float>(prm.pathSMin + line);
		n = static_cast<float>(prm.pathNMin + pos);
	} else {
		n = static_cast<float>(prm.pathNMin + line);
		s = static_cast<float>(prm.pathSMin + pos);
	}

	float x = 0.0f;
	float y = 0.0f;
	float tx = 1.0f;
	float ty = 0.0f;
	if (!BPS_PathEvalAtS(prm.pathSamples, prm.pathSampleCount, s, &x, &y, &tx, &ty)) {
		return false;
	}
	const float nx = -ty;
	const float ny = tx;
	*x_out = RoundToLong(x + n * nx);
	*y_out = RoundToLong(y + n * ny);
	return true;
}

bool BPS_PathDomainPosForPixel(
	const BitonicSorterParams &prm,
	A_long x,
	A_long y,
	A_long *line_out,
	A_long *pos_out)
{
	if (!line_out || !pos_out || !prm.pathSamples || prm.pathSampleCount < 2) {
		return false;
	}

	float s = 0.0f;
	float n = 0.0f;
	if (!BPS_PathClosestPoint(prm.pathSamples, prm.pathSampleCount,
							  static_cast<float>(x), static_cast<float>(y),
							  &s, &n)) {
		return false;
	}

	A_long line = 0;
	float order = 0.0f;
	if (!PathLaneOrder(prm, s, n, &line, &order)) {
		return false;
	}

	const A_long pos = (prm.pathDirection == BPS_PATH_DIR_NORMAL)
		? RoundToLong(order) - prm.pathNMin
		: RoundToLong(order) - prm.pathSMin;
	if (pos < 0 || pos >= prm.domainMaxLineLength) {
		return false;
	}

	*line_out = line;
	*pos_out = pos;
	return true;
}

bool BPS_ClassifyMappedPixel(
	const BitonicSorterParams &prm,
	A_long frameW,
	A_long frameH,
	A_long x,
	A_long y,
	A_long *line_out,
	float *pos_key_out)
{
	if (!line_out || !pos_key_out || frameW <= 0 || frameH <= 0 ||
		x < 0 || y < 0 || x >= frameW || y >= frameH) {
		return false;
	}

	if (prm.mode == BPS_MODE_SWIRL) {
		if (prm.domainLineCount <= 0) {
			return false;
		}
		const float dx = static_cast<float>(x) - prm.centerX;
		const float dy = static_cast<float>(y) - prm.centerY;
		const float radius = std::sqrt(dx * dx + dy * dy);
		float theta = std::atan2(dy, dx) - prm.angleRadians;
		theta = WrapPositiveRadians(theta);
		const float phase = WrapPositiveRadians(theta - prm.swirlK * radius);
		A_long line = prm.radialLineCount <= 1
			? 0
			: RoundToLong((phase / (2.0f * kPi)) *
						  static_cast<float>(prm.radialLineCount));
		if (line >= prm.radialLineCount) {
			line -= prm.radialLineCount;
		}
		if (line < 0 || line >= prm.domainLineCount) {
			return false;
		}
		*line_out = line;
		*pos_key_out = radius;
		return true;
	}

	if (prm.mode == BPS_MODE_PATH) {
		if (!prm.pathSamples || prm.pathSampleCount < 2 ||
			prm.domainLineCount <= 0) {
			return false;
		}

		float s = 0.0f;
		float n = 0.0f;
		if (!BPS_PathClosestPoint(prm.pathSamples, prm.pathSampleCount,
								  static_cast<float>(x),
								  static_cast<float>(y),
								  &s, &n)) {
			return false;
		}

		A_long line = 0;
		float order = 0.0f;
		if (!PathLaneOrder(prm, s, n, &line, &order)) {
			return false;
		}
		*line_out = line;
		*pos_key_out = order;
		return true;
	}

	return false;
}

namespace {

// Nearest-point field: for every frame pixel, the closest polyline point q
// (qx, qy), its arc length (s) and unit tangent (tx, ty), plus the squared
// distance (d2) to it. Computed with the Jump Flooding Algorithm so the cost is
// O(W*H * log(max(W,H))) instead of O(W*H * sampleCount).
struct JfaField {
	std::vector<float> qx;
	std::vector<float> qy;
	std::vector<float> s;
	std::vector<float> tx;
	std::vector<float> ty;
	std::vector<float> d2;

	void resize(size_t count) {
		const float inf = std::numeric_limits<float>::infinity();
		qx.assign(count, 0.0f);
		qy.assign(count, 0.0f);
		s.assign(count, 0.0f);
		tx.assign(count, 1.0f);
		ty.assign(count, 0.0f);
		d2.assign(count, inf);
	}
};

inline unsigned int JfaWorkerCount(A_long rows)
{
	if (rows < 64) {
		return 1u;
	}
	unsigned int hw = std::thread::hardware_concurrency();
	if (hw == 0) {
		hw = 1u;
	}
	unsigned int workers = (std::min)(hw, 8u);
	if (static_cast<A_long>(workers) > rows) {
		workers = static_cast<unsigned int>(rows);
	}
	return workers < 1u ? 1u : workers;
}

// One JFA pass at the given step over a grid whose cell (x, y) sits at the
// full-resolution anchor (x*scale, y*scale). Each cell adopts the nearest seed
// among its eight step-offset neighbours (read from `src`, written to `dst`).
void JfaPass(const JfaField &src, JfaField &dst,
			 A_long gridW, A_long gridH, A_long step, float scale)
{
	static const A_long kDX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
	static const A_long kDY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

	auto run_rows = [&](A_long y0, A_long y1) {
		for (A_long y = y0; y < y1; ++y) {
			const float ay = static_cast<float>(y) * scale;
			for (A_long x = 0; x < gridW; ++x) {
				const float ax = static_cast<float>(x) * scale;
				const size_t idx = static_cast<size_t>(y) *
					static_cast<size_t>(gridW) + static_cast<size_t>(x);
				float best_d2 = src.d2[idx];
				float best_qx = src.qx[idx];
				float best_qy = src.qy[idx];
				float best_s = src.s[idx];
				float best_tx = src.tx[idx];
				float best_ty = src.ty[idx];

				for (int k = 0; k < 8; ++k) {
					const A_long nx = x + kDX[k] * step;
					const A_long ny = y + kDY[k] * step;
					if (nx < 0 || ny < 0 || nx >= gridW || ny >= gridH) {
						continue;
					}
					const size_t nidx = static_cast<size_t>(ny) *
						static_cast<size_t>(gridW) + static_cast<size_t>(nx);
					if (!std::isfinite(src.d2[nidx])) {
						continue;
					}
					const float dx = ax - src.qx[nidx];
					const float dy = ay - src.qy[nidx];
					const float cand = dx * dx + dy * dy;
					if (cand < best_d2) {
						best_d2 = cand;
						best_qx = src.qx[nidx];
						best_qy = src.qy[nidx];
						best_s = src.s[nidx];
						best_tx = src.tx[nidx];
						best_ty = src.ty[nidx];
					}
				}

				dst.d2[idx] = best_d2;
				dst.qx[idx] = best_qx;
				dst.qy[idx] = best_qy;
				dst.s[idx] = best_s;
				dst.tx[idx] = best_tx;
				dst.ty[idx] = best_ty;
			}
		}
	};

	const unsigned int workers = JfaWorkerCount(gridH);
	if (workers <= 1u) {
		run_rows(0, gridH);
		return;
	}
	std::vector<std::thread> pool;
	pool.reserve(workers);
	for (unsigned int w = 0; w < workers; ++w) {
		const A_long y0 = static_cast<A_long>(
			(static_cast<long long>(gridH) * w) / workers);
		const A_long y1 = static_cast<A_long>(
			(static_cast<long long>(gridH) * (w + 1)) / workers);
		if (y0 < y1) {
			pool.emplace_back(run_rows, y0, y1);
		}
	}
	for (std::thread &t : pool) {
		t.join();
	}
}

// Downscale factor for the JFA grid. The nearest-point field is smooth away
// from the medial axis, so solving it on a coarser grid and classifying at full
// resolution keeps the visible lanes intact while cutting the flood cost by F^2.
inline A_long JfaGridFactor(A_long frameW, A_long frameH)
{
	const A_long max_dim = (std::max)(frameW, frameH);
	A_long f = (max_dim + 450) / 900;	// ~1 per 900px, rounded
	if (f < 1) {
		f = 1;
	}
	if (f > 3) {
		f = 3;
	}
	return f;
}

// Run `fn(threadIndex)` across `workers` threads and join.
template <typename Fn>
void RunParallel(unsigned int workers, Fn fn)
{
	if (workers <= 1u) {
		fn(0u);
		return;
	}
	std::vector<std::thread> pool;
	pool.reserve(workers);
	for (unsigned int t = 0; t < workers; ++t) {
		pool.emplace_back(fn, t);
	}
	for (std::thread &th : pool) {
		th.join();
	}
}

// Compute the low-resolution nearest-point field (grid `field`, cell (x,y) at
// anchor (x*factor, y*factor)) from the flattened/extended polyline.
void ComputePathField(
	A_long frameW,
	A_long frameH,
	const BitonicSorterParams &prm,
	JfaField *fieldOut,
	A_long *gridWOut,
	A_long *gridHOut,
	A_long *factorOut)
{
	const A_long factor = JfaGridFactor(frameW, frameH);
	const float scale = static_cast<float>(factor);
	const A_long gridW = (frameW + factor - 1) / factor;
	const A_long gridH = (frameH + factor - 1) / factor;
	*gridWOut = gridW;
	*gridHOut = gridH;
	*factorOut = factor;

	const size_t grid_count = static_cast<size_t>(gridW) *
							  static_cast<size_t>(gridH);
	JfaField a;
	JfaField b;
	a.resize(grid_count);
	b.resize(grid_count);

	// Seed: walk every polyline segment at <= 1px spacing and splat each sample
	// into the grid cell it lands in, keeping the anchor-closest seed per cell.
	auto splat = [&](float fx, float fy, float sv, float tvx, float tvy) {
		const A_long cx = RoundToLong(fx / scale);
		const A_long cy = RoundToLong(fy / scale);
		if (cx < 0 || cy < 0 || cx >= gridW || cy >= gridH) {
			return;
		}
		const size_t idx = static_cast<size_t>(cy) *
			static_cast<size_t>(gridW) + static_cast<size_t>(cx);
		const float dx = static_cast<float>(cx) * scale - fx;
		const float dy = static_cast<float>(cy) * scale - fy;
		const float dd = dx * dx + dy * dy;
		if (dd < a.d2[idx]) {
			a.d2[idx] = dd;
			a.qx[idx] = fx;
			a.qy[idx] = fy;
			a.s[idx] = sv;
			a.tx[idx] = tvx;
			a.ty[idx] = tvy;
		}
	};

	const BpsPathSample *samples = prm.pathSamples;
	const A_long n = prm.pathSampleCount;
	for (A_long i = 0; i < n - 1; ++i) {
		const BpsPathSample &p0 = samples[i];
		const BpsPathSample &p1 = samples[i + 1];
		const float dx = p1.x - p0.x;
		const float dy = p1.y - p0.y;
		const float len = std::sqrt(dx * dx + dy * dy);
		const A_long steps = (std::max)(static_cast<A_long>(1),
									   static_cast<A_long>(std::ceil(len)));
		for (A_long k = (i > 0) ? 1 : 0; k <= steps; ++k) {
			const float t = static_cast<float>(k) / static_cast<float>(steps);
			const float fx = p0.x + dx * t;
			const float fy = p0.y + dy * t;
			const float sv = p0.s + (p1.s - p0.s) * t;
			float tvx = p0.tx + (p1.tx - p0.tx) * t;
			float tvy = p0.ty + (p1.ty - p0.ty) * t;
			NormaliseTangent(&tvx, &tvy);
			splat(fx, fy, sv, tvx, tvy);
		}
	}

	// Jump flood: step from ~gridMax/2 down to 1, then one refinement pass.
	const A_long grid_max = (std::max)(gridW, gridH);
	A_long step = 1;
	while (step < (grid_max + 1) / 2) {
		step <<= 1;
	}
	JfaField *src = &a;
	JfaField *dst = &b;
	for (; step >= 1; step >>= 1) {
		JfaPass(*src, *dst, gridW, gridH, step, scale);
		std::swap(src, dst);
	}
	JfaPass(*src, *dst, gridW, gridH, 1, scale);
	std::swap(src, dst);

	*fieldOut = std::move(*src);
}

// Build the Path-mode pixel map: low-res Jump Flooding for the nearest-point
// field, then a parallel counting sort (per-thread histograms -> disjoint
// scatter -> parallel per-lane sort). No per-lane heap vectors, all cores used.
void BuildPathMap(
	A_long frameW,
	A_long frameH,
	const BitonicSorterParams &prm,
	BpsPathMap *mapP)
{
	mapP->records.clear();
	mapP->lineOffsets.assign(1u, 0u);
	mapP->workOffsets.assign(1u, 0u);
	mapP->maxLineLength = 0;
	mapP->domainMaxLineLength = 0;
	mapP->mappedRecordCount = 0;
	mapP->mappedWorkItemCount = 0;

	if (!prm.pathSamples || prm.pathSampleCount < 2 ||
		frameW <= 0 || frameH <= 0 || prm.domainLineCount <= 0) {
		return;
	}
	const A_long lane_count = prm.domainLineCount;

	JfaField field;
	A_long gridW = 0;
	A_long gridH = 0;
	A_long factor = 1;
	ComputePathField(frameW, frameH, prm, &field, &gridW, &gridH, &factor);

	const size_t pixel_count = static_cast<size_t>(frameW) *
							   static_cast<size_t>(frameH);
	std::vector<std::int32_t> laneOf(pixel_count);
	std::vector<float> keyOf(pixel_count);

	const unsigned int workers = JfaWorkerCount(frameH);
	auto row_range = [&](unsigned int t, A_long &y0, A_long &y1) {
		y0 = static_cast<A_long>((static_cast<long long>(frameH) * t) / workers);
		y1 = static_cast<A_long>((static_cast<long long>(frameH) * (t + 1)) / workers);
	};

	// Phase 1: classify each pixel to (lane, order) and count per (thread, lane).
	std::vector<std::vector<std::uint32_t>> hist(
		workers, std::vector<std::uint32_t>(static_cast<size_t>(lane_count), 0u));
	RunParallel(workers, [&](unsigned int t) {
		A_long y0 = 0, y1 = 0;
		row_range(t, y0, y1);
		std::vector<std::uint32_t> &h = hist[t];
		for (A_long y = y0; y < y1; ++y) {
			const A_long cy = (std::min)(y / factor, gridH - 1);
			for (A_long x = 0; x < frameW; ++x) {
				const size_t pidx = static_cast<size_t>(y) *
					static_cast<size_t>(frameW) + static_cast<size_t>(x);
				const A_long cx = (std::min)(x / factor, gridW - 1);
				const size_t idx = static_cast<size_t>(cy) *
					static_cast<size_t>(gridW) + static_cast<size_t>(cx);
				if (!std::isfinite(field.d2[idx])) {
					laneOf[pidx] = -1;
					continue;
				}
				const float dxp = static_cast<float>(x) - field.qx[idx];
				const float dyp = static_cast<float>(y) - field.qy[idx];
				// Signed distance d = dot(p - q, N), N = (-ty, tx).
				const float dsigned =
					-field.ty[idx] * dxp + field.tx[idx] * dyp;
				A_long line = 0;
				float order = 0.0f;
				if (!PathLaneOrder(prm, field.s[idx], dsigned, &line, &order)) {
					laneOf[pidx] = -1;
					continue;
				}
				laneOf[pidx] = static_cast<std::int32_t>(line);
				keyOf[pidx] = order;
				++h[static_cast<size_t>(line)];
			}
		}
	});

	// Prefix sum: lane record offsets and each thread's disjoint write base.
	mapP->lineOffsets.assign(static_cast<size_t>(lane_count) + 1u, 0u);
	std::vector<std::uint32_t> base(
		static_cast<size_t>(workers) * static_cast<size_t>(lane_count), 0u);
	std::uint32_t running = 0;
	for (A_long lane = 0; lane < lane_count; ++lane) {
		mapP->lineOffsets[static_cast<size_t>(lane)] = running;
		for (unsigned int t = 0; t < workers; ++t) {
			base[static_cast<size_t>(t) * static_cast<size_t>(lane_count) +
				 static_cast<size_t>(lane)] = running;
			running += hist[t][static_cast<size_t>(lane)];
		}
	}
	mapP->lineOffsets[static_cast<size_t>(lane_count)] = running;
	mapP->records.resize(running);

	// Phase 2: scatter into each thread's disjoint per-lane region (no locks).
	RunParallel(workers, [&](unsigned int t) {
		A_long y0 = 0, y1 = 0;
		row_range(t, y0, y1);
		std::vector<std::uint32_t> cursor(static_cast<size_t>(lane_count));
		for (A_long lane = 0; lane < lane_count; ++lane) {
			cursor[static_cast<size_t>(lane)] =
				base[static_cast<size_t>(t) * static_cast<size_t>(lane_count) +
					 static_cast<size_t>(lane)];
		}
		for (A_long y = y0; y < y1; ++y) {
			for (A_long x = 0; x < frameW; ++x) {
				const size_t pidx = static_cast<size_t>(y) *
					static_cast<size_t>(frameW) + static_cast<size_t>(x);
				const std::int32_t lane = laneOf[pidx];
				if (lane < 0) {
					continue;
				}
				const std::uint32_t dst = cursor[static_cast<size_t>(lane)]++;
				mapP->records[dst] = BpsMappedPixelRecord{
					keyOf[pidx], static_cast<std::uint32_t>(pidx)};
			}
		}
	});

	// Work offsets (padded to the next power of two) and the longest lane.
	mapP->workOffsets.assign(static_cast<size_t>(lane_count) + 1u, 0u);
	std::uint32_t work_offset = 0;
	std::uint32_t max_line_length = 0;
	for (A_long lane = 0; lane < lane_count; ++lane) {
		mapP->workOffsets[static_cast<size_t>(lane)] = work_offset;
		const std::uint32_t len =
			mapP->lineOffsets[static_cast<size_t>(lane) + 1u] -
			mapP->lineOffsets[static_cast<size_t>(lane)];
		max_line_length = (std::max)(max_line_length, len);
		work_offset += len + NextPow2U32(len);
	}
	mapP->workOffsets[static_cast<size_t>(lane_count)] = work_offset;

	// Phase 3: lay each lane's slots out in `order` (parallel, contiguous sort).
	RunParallel(workers, [&](unsigned int t) {
		for (A_long lane = static_cast<A_long>(t); lane < lane_count;
			 lane += static_cast<A_long>(workers)) {
			const std::uint32_t b0 = mapP->lineOffsets[static_cast<size_t>(lane)];
			const std::uint32_t b1 =
				mapP->lineOffsets[static_cast<size_t>(lane) + 1u];
			std::sort(mapP->records.begin() + b0, mapP->records.begin() + b1,
				[](const BpsMappedPixelRecord &a, const BpsMappedPixelRecord &b) {
					if (a.posKey < b.posKey) {
						return true;
					}
					if (a.posKey > b.posKey) {
						return false;
					}
					return a.pixelIndex < b.pixelIndex;
				});
		}
	});

	mapP->mappedRecordCount = static_cast<A_long>(running);
	mapP->mappedWorkItemCount = static_cast<A_long>(work_offset);
	mapP->maxLineLength = static_cast<A_long>(max_line_length);
	mapP->domainMaxLineLength = static_cast<A_long>(max_line_length);
}

// FNV-1a 64-bit over raw bytes.
inline std::uint64_t HashBytes(std::uint64_t seed, const void *data, size_t len)
{
	const unsigned char *p = static_cast<const unsigned char *>(data);
	std::uint64_t h = seed;
	for (size_t i = 0; i < len; ++i) {
		h ^= static_cast<std::uint64_t>(p[i]);
		h *= 1099511628211ull;
	}
	return h;
}

// Geometry-only key: identical key => identical pixel map, regardless of the
// image, time, or sort parameters. Covers frame size, direction, closure, the
// resolved domain, and the flattened/extended polyline itself.
std::uint64_t PathMapKey(A_long frameW, A_long frameH, const BitonicSorterParams &prm)
{
	std::uint64_t h = 1469598103934665603ull;
	const A_long header[] = {
		frameW, frameH, prm.pathDirection, prm.pathClosed,
		prm.pathSMin, prm.pathNMin, prm.domainLineCount, prm.pathSampleCount
	};
	h = HashBytes(h, header, sizeof(header));
	h = HashBytes(h, &prm.pathLength, sizeof(prm.pathLength));
	if (prm.pathSamples && prm.pathSampleCount > 0) {
		h = HashBytes(h, prm.pathSamples,
					  static_cast<size_t>(prm.pathSampleCount) *
						  sizeof(BpsPathSample));
	}
	return h;
}

// Tiny most-recently-used cache of built maps, shared across effect instances.
struct PathMapCacheEntry {
	std::uint64_t key = 0;
	std::shared_ptr<const BpsPathMap> map;
};

std::mutex g_pathMapCacheMutex;
std::vector<PathMapCacheEntry> g_pathMapCache;	// front = most recent
constexpr size_t kPathMapCacheCapacity = 4;

} // namespace

std::shared_ptr<const BpsPathMap> BPS_AcquirePathMap(
	A_long frameW,
	A_long frameH,
	const BitonicSorterParams &prm)
{
	if (prm.mode != BPS_MODE_PATH || frameW <= 0 || frameH <= 0 ||
		!prm.pathSamples || prm.pathSampleCount < 2 ||
		prm.domainLineCount <= 0) {
		return nullptr;
	}

	const std::uint64_t key = PathMapKey(frameW, frameH, prm);

	{
		std::lock_guard<std::mutex> lock(g_pathMapCacheMutex);
		for (size_t i = 0; i < g_pathMapCache.size(); ++i) {
			if (g_pathMapCache[i].key == key && g_pathMapCache[i].map) {
				std::shared_ptr<const BpsPathMap> hit = g_pathMapCache[i].map;
				if (i != 0) {	// promote to most-recently-used
					const PathMapCacheEntry entry = g_pathMapCache[i];
					g_pathMapCache.erase(g_pathMapCache.begin() +
										 static_cast<std::ptrdiff_t>(i));
					g_pathMapCache.insert(g_pathMapCache.begin(), entry);
				}
				return hit;
			}
		}
	}

	// Build outside the lock (the expensive step). A duplicate concurrent build
	// for the same key is possible but harmless — both produce identical maps.
	std::shared_ptr<BpsPathMap> built = std::make_shared<BpsPathMap>();
	BuildPathMap(frameW, frameH, prm, built.get());

	{
		std::lock_guard<std::mutex> lock(g_pathMapCacheMutex);
		for (const PathMapCacheEntry &entry : g_pathMapCache) {
			if (entry.key == key && entry.map) {
				return entry.map;	// another thread won the race
			}
		}
		PathMapCacheEntry entry;
		entry.key = key;
		entry.map = built;
		g_pathMapCache.insert(g_pathMapCache.begin(), entry);
		if (g_pathMapCache.size() > kPathMapCacheCapacity) {
			g_pathMapCache.pop_back();
		}
	}
	return built;
}

PF_Err BPS_BuildPathGeometry(
	PF_InData *in_data,
	PF_OutData *out_data,
	PF_PathID path_id,
	A_long path_direction,
	BitonicSorterParams *paramsP,
	std::vector<BpsPathSample> *samplesP)
{
	if (!paramsP || !samplesP) {
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	samplesP->clear();
	paramsP->pathDirection = path_direction;
	paramsP->pathClosed = 0;
	paramsP->pathLength = 0.0f;
	paramsP->pathSamples = nullptr;
	paramsP->pathSampleCount = 0;
	paramsP->pathSMin = 0;
	paramsP->pathNMin = 0;
	paramsP->domainLineCount = 0;
	paramsP->domainMaxLineLength = 0;
	paramsP->maxLineLength = 0;

	const A_long frameW = in_data ? in_data->width : 0;
	const A_long frameH = in_data ? in_data->height : 0;
	const float diag = (frameW > 0 && frameH > 0)
		? std::sqrt(static_cast<float>(frameW) * static_cast<float>(frameW) +
					static_cast<float>(frameH) * static_cast<float>(frameH))
		: 0.0f;

	if (!in_data || !in_data->pica_basicP || path_id == 0 || frameW <= 0 || frameH <= 0) {
		ResolvePathDomainBounds(paramsP, diag);
		return PF_Err_NONE;
	}

	AEGP_SuiteHandler suites(in_data->pica_basicP);

	AEGP_PluginID plugin_id = 0;
	if (in_data->global_data && suites.HandleSuite1()) {
		BPS_GlobalData *globalP = reinterpret_cast<BPS_GlobalData *>(
			suites.HandleSuite1()->host_lock_handle(in_data->global_data));
		if (globalP) {
			plugin_id = globalP->plugin_id;
		}
		suites.HandleSuite1()->host_unlock_handle(in_data->global_data);
	}

	PF_Boolean openB = TRUE;

	// 1) PathMaster / Projector pattern: AEGP layer mask outline (preferred).
	if (!SamplePathFromAegpMask(in_data, suites, plugin_id, path_id,
								samplesP, &openB)) {
		// 2) PathMaster Render pattern: PF_CheckoutPath + PathDataSuite with
		//    effect_ref (never NULL — PathMaster always passes effect_ref).
		samplesP->clear();
		if (suites.PathQuerySuite1() && suites.PathDataSuite1()) {
			PF_PathOutlinePtr pathP = nullptr;
			PF_PathID checkout_id = path_id;
			PF_Err path_err = suites.PathQuerySuite1()->PF_CheckoutPath(
				in_data->effect_ref,
				checkout_id,
				in_data->current_time,
				in_data->time_step,
				in_data->time_scale,
				&pathP);

			if ((path_err || !pathP) && path_id > 0) {
				path_err = PF_Err_NONE;
				pathP = nullptr;
				for (A_long index_try = 0; index_try < 2 && !pathP; ++index_try) {
					const A_long index = (index_try == 0) ? (path_id - 1) : path_id;
					if (index < 0) {
						continue;
					}
					PF_PathID unique_id = 0;
					if (suites.PathQuerySuite1()->PF_PathInfo(
							in_data->effect_ref, index, &unique_id) != PF_Err_NONE ||
						unique_id == 0) {
						continue;
					}
					checkout_id = unique_id;
					path_err = suites.PathQuerySuite1()->PF_CheckoutPath(
						in_data->effect_ref,
						checkout_id,
						in_data->current_time,
						in_data->time_step,
						in_data->time_scale,
						&pathP);
					if (path_err) {
						pathP = nullptr;
					}
				}
			}

			if (pathP) {
				(void)SamplePathFromPfOutline(
					in_data->effect_ref, suites.PathDataSuite1(), pathP,
					samplesP, &openB);
				(void)suites.PathQuerySuite1()->PF_CheckinPath(
					in_data->effect_ref, checkout_id, FALSE, pathP);
			}
		}
	}

	if (samplesP->size() < 2u) {
		ResolvePathDomainBounds(paramsP, diag);
		(void)out_data;
		return PF_Err_NONE;
	}

	// Mask geometry is full-resolution; render space may be downsampled. Scale
	// the polyline into render space only when the samples clearly sit outside
	// the current frame (matches the PathMaster feather-scaling convention).
	{
		const float dsx = paramsP->downsampleX > 0.0f ? paramsP->downsampleX : 1.0f;
		const float dsy = paramsP->downsampleY > 0.0f ? paramsP->downsampleY : 1.0f;
		float max_x = samplesP->front().x;
		float max_y = samplesP->front().y;
		for (const BpsPathSample &sample : *samplesP) {
			max_x = (std::max)(max_x, sample.x);
			max_y = (std::max)(max_y, sample.y);
		}
		const bool looks_full_res =
			max_x > static_cast<float>(frameW) * 1.25f ||
			max_y > static_cast<float>(frameH) * 1.25f;
		if (looks_full_res && (dsx < 0.999f || dsy < 0.999f)) {
			const float s_scale = (std::min)(dsx, dsy);
			for (BpsPathSample &sample : *samplesP) {
				sample.x *= dsx;
				sample.y *= dsy;
				sample.s *= s_scale;
				sample.tx *= dsx;
				sample.ty *= dsy;
				NormaliseTangent(&sample.tx, &sample.ty);
			}
		}
	}

	const bool closed = (openB == FALSE);
	paramsP->pathClosed = closed ? 1 : 0;

	if (closed) {
		// Close the polyline back to the first vertex so the seam has a segment
		// and pathLength is the full perimeter. Arc length then wraps modulo it.
		const BpsPathSample first = samplesP->front();
		const float dx = first.x - samplesP->back().x;
		const float dy = first.y - samplesP->back().y;
		if (dx * dx + dy * dy > 1.0e-6f) {
			BpsPathSample loop = first;
			loop.s = samplesP->back().s + std::sqrt(dx * dx + dy * dy);
			loop.tx = samplesP->back().tx;
			loop.ty = samplesP->back().ty;
			samplesP->push_back(loop);
		}
		paramsP->pathLength = samplesP->back().s;
	} else {
		// pathLength is the raw arc length; open ends are then extended so the
		// local coordinate system covers the whole frame (s < 0 and s > length).
		paramsP->pathLength = samplesP->back().s;
		ExtendOpenEnds(samplesP, diag > kEps ? diag : 1.0f);
	}

	paramsP->pathSamples = samplesP->data();
	paramsP->pathSampleCount = static_cast<A_long>(samplesP->size());

	ResolvePathDomainBounds(paramsP, diag);
	(void)out_data;
	(void)kPi;
	return PF_Err_NONE;
}
