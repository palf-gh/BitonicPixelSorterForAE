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
#include "BitonicPixelSorter_GpuEligibility.h"

#include "AEFX_SuiteHelper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

inline std::uint32_t SortableFloatKey(float value)
{
	// Match the previous comparator for ordinary finite values while giving the
	// radix pass a monotonic unsigned representation. Canonicalise signed zero so
	// -0 and +0 still tie and fall through to pixelIndex.
	if (value == 0.0f) {
		value = 0.0f;
	}
	std::uint32_t bits = 0;
	std::memcpy(&bits, &value, sizeof(bits));
	return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

inline std::uint64_t MappedRecordSortKey(const BpsMappedPixelRecord &record)
{
	return (static_cast<std::uint64_t>(SortableFloatKey(record.posKey)) << 32) |
		static_cast<std::uint64_t>(record.pixelIndex);
}

void RadixSortMappedRecordSpan(
	BpsMappedPixelRecord *first,
	std::uint32_t count,
	std::vector<BpsMappedPixelRecord> *scratchP)
{
	if (!first || !scratchP || count <= 1u) {
		return;
	}

	scratchP->resize(count);
	BpsMappedPixelRecord *src = first;
	BpsMappedPixelRecord *dst = scratchP->data();

	for (unsigned int shift = 0; shift < 64u; shift += 8u) {
		std::uint32_t buckets[256] = {};
		for (std::uint32_t i = 0; i < count; ++i) {
			const std::uint64_t key = MappedRecordSortKey(src[i]);
			++buckets[static_cast<unsigned int>((key >> shift) & 0xffu)];
		}

		std::uint32_t running = 0;
		for (std::uint32_t &bucket : buckets) {
			const std::uint32_t size = bucket;
			bucket = running;
			running += size;
		}

		for (std::uint32_t i = 0; i < count; ++i) {
			const std::uint64_t key = MappedRecordSortKey(src[i]);
			const unsigned int bucket =
				static_cast<unsigned int>((key >> shift) & 0xffu);
			dst[buckets[bucket]++] = src[i];
		}

		std::swap(src, dst);
	}
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

// Distance along the ray P + t*D (unit D) to exit the inclusive frame box.
// Returns a small positive epsilon if the ray does not exit (degenerate).
float RayExitDistance(float px, float py, float dx, float dy,
					  float x0, float y0, float x1, float y1)
{
	float t_exit = std::numeric_limits<float>::infinity();
	if (dx > kEps) {
		t_exit = (std::min)(t_exit, (x1 - px) / dx);
	} else if (dx < -kEps) {
		t_exit = (std::min)(t_exit, (x0 - px) / dx);
	}
	if (dy > kEps) {
		t_exit = (std::min)(t_exit, (y1 - py) / dy);
	} else if (dy < -kEps) {
		t_exit = (std::min)(t_exit, (y0 - py) / dy);
	}
	if (!(t_exit > kEps) || t_exit > 1.0e12f) {
		return kEps;
	}
	return t_exit;
}

// Extend both open ends along the end tangents to the frame edge. The extended
// rays are part of the path so pixels past the authored endpoints still belong
// to the path's local (s, n) coordinate system. End is extended first, then
// start, so indices stay simple.
void ExtendOpenEndsToFrameEdges(std::vector<BpsPathSample> *samplesP,
								float frame_w, float frame_h)
{
	if (!samplesP || samplesP->size() < 2u || frame_w <= 1.0f || frame_h <= 1.0f) {
		return;
	}

	const float x0 = 0.0f;
	const float y0 = 0.0f;
	const float x1 = frame_w - 1.0f;
	const float y1 = frame_h - 1.0f;

	// --- End ray (append) ---
	{
		const BpsPathSample last = samplesP->back();
		// Extend along the path's actual arrival direction (the last polyline
		// segment). The stored per-sample tangent can be stale or degenerate,
		// which would open a corner at the endpoint and fan the sort; the segment
		// direction is always collinear with how the path reaches its end.
		float tx = last.tx;
		float ty = last.ty;
		if (samplesP->size() >= 2u) {
			const BpsPathSample &prev = (*samplesP)[samplesP->size() - 2u];
			tx = last.x - prev.x;
			ty = last.y - prev.y;
		}
		NormaliseTangent(&tx, &ty);
		const float t_exit = RayExitDistance(
			last.x, last.y, tx, ty, x0, y0, x1, y1);
		if (t_exit > kEps) {
			const float to_x = last.x + tx * t_exit;
			const float to_y = last.y + ty * t_exit;
			const float dx = to_x - last.x;
			const float dy = to_y - last.y;
			const float len = std::sqrt(dx * dx + dy * dy);
			const A_long steps = (std::max)(
				static_cast<A_long>(1), static_cast<A_long>(std::ceil(len)));
			for (A_long k = 1; k <= steps; ++k) {
				const float t = static_cast<float>(k) / static_cast<float>(steps);
				BpsPathSample s = last;
				s.x = last.x + dx * t;
				s.y = last.y + dy * t;
				s.s = last.s + len * t;
				s.tx = tx;
				s.ty = ty;
				samplesP->push_back(s);
			}
		}
	}

	// --- Start ray (prepend), outward = -T ---
	{
		// Original start is still the first sample (we only appended).
		const BpsPathSample first = samplesP->front();
		// Departure direction from the first polyline segment; outward is -T.
		float tx = first.tx;
		float ty = first.ty;
		if (samplesP->size() >= 2u) {
			const BpsPathSample &second = (*samplesP)[1];
			tx = second.x - first.x;
			ty = second.y - first.y;
		}
		NormaliseTangent(&tx, &ty);
		const float t_exit = RayExitDistance(
			first.x, first.y, -tx, -ty, x0, y0, x1, y1);
		if (t_exit > kEps) {
			const float to_x = first.x - tx * t_exit;
			const float to_y = first.y - ty * t_exit;
			const float dx = to_x - first.x;
			const float dy = to_y - first.y;
			const float len = std::sqrt(dx * dx + dy * dy);
			const A_long steps = (std::max)(
				static_cast<A_long>(1), static_cast<A_long>(std::ceil(len)));
			std::vector<BpsPathSample> prefix;
			prefix.reserve(static_cast<size_t>(steps));
			for (A_long k = steps; k >= 1; --k) {
				const float t = static_cast<float>(k) / static_cast<float>(steps);
				BpsPathSample s = first;
				s.x = first.x + dx * t;
				s.y = first.y + dy * t;
				// Temporary s; renumbered below.
				s.s = -len * t;
				s.tx = tx;
				s.ty = ty;
				prefix.push_back(s);
			}
			samplesP->insert(samplesP->begin(), prefix.begin(), prefix.end());
		}
	}

	// Renumber arc length so the path is continuous and starts at s = 0.
	if (!samplesP->empty()) {
		float s = 0.0f;
		samplesP->front().s = 0.0f;
		for (size_t i = 1u; i < samplesP->size(); ++i) {
			const BpsPathSample &prev = (*samplesP)[i - 1u];
			BpsPathSample &sample = (*samplesP)[i];
			const float dx = sample.x - prev.x;
			const float dy = sample.y - prev.y;
			s += std::sqrt(dx * dx + dy * dy);
			sample.s = s;
		}
	}
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
		// The cubic derivative vanishes at the ends of a straight or degenerate
		// segment (p1==p0, p2==p3), which happens at every corner vertex. Fall
		// back to the chord so endpoint/corner tangents keep the real segment
		// direction rather than NormaliseTangent's generic (1,0): a stray (1,0)
		// here makes the open-end extension shoot off horizontally and fan the
		// sort around the corner it opens at the endpoint.
		if (tx * tx + ty * ty < kEps) {
			tx = p3x - p0x;
			ty = p3y - p0y;
		}
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

// Effect path-parameter checkout (PF_PathQuerySuite). Coordinates are already
// in the effect layer's top-left space for both raster and vector layers.
bool SamplePathFromEffectParam(PF_InData *in_data,
							   AEGP_SuiteHandler &suites,
							   PF_PathID path_id,
							   std::vector<BpsPathSample> *samplesP,
							   PF_Boolean *open_out)
{
	samplesP->clear();
	if (!in_data || path_id == 0 ||
		!suites.PathQuerySuite1() || !suites.PathDataSuite1()) {
		return false;
	}

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

	if (!pathP) {
		return false;
	}

	const bool ok = SamplePathFromPfOutline(
		in_data->effect_ref, suites.PathDataSuite1(), pathP, samplesP, open_out);
	(void)suites.PathQuerySuite1()->PF_CheckinPath(
		in_data->effect_ref, checkout_id, FALSE, pathP);
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

	if (prm.mode == BPS_MODE_FREE_ANGLE) {
		if (prm.domainLineCount <= 0) {
			return false;
		}
		const double p = static_cast<double>(x) * prm.angleCos +
						 static_cast<double>(y) * prm.angleSin;
		const double q = -static_cast<double>(x) * prm.angleSin +
						  static_cast<double>(y) * prm.angleCos;
		const A_long line = RoundToLong(q) - prm.freeQMin;
		if (line < 0 || line >= prm.domainLineCount) {
			return false;
		}
		*line_out = line;
		*pos_key_out = static_cast<float>(p);
		return true;
	}

	if (prm.mode == BPS_MODE_ROTATION) {
		if (prm.domainLineCount <= 0) {
			return false;
		}
		const double dx = static_cast<double>(x) - prm.centerX;
		const double dy = static_cast<double>(y) - prm.centerY;
		const A_long line = RoundToLong(std::sqrt(dx * dx + dy * dy));
		if (line < 0 || line >= prm.domainLineCount) {
			return false;
		}
		const A_long lineLen = (line <= 0)
			? 1
			: static_cast<A_long>(std::ceil(2.0 * kPi * static_cast<double>(line)));
		A_long pos = 0;
		if (lineLen > 1) {
			const double rx = dx * prm.angleCos + dy * prm.angleSin;
			const double ry = -dx * prm.angleSin + dy * prm.angleCos;
			double theta = std::atan2(rx, -ry);
			if (theta < 0.0) {
				theta += 2.0 * kPi;
			}
			pos = RoundToLong((theta / (2.0 * kPi)) * static_cast<double>(lineLen));
			if (pos >= lineLen) {
				pos -= lineLen;
			}
		}
		*line_out = line;
		*pos_key_out = static_cast<float>(pos);
		return true;
	}

	if (prm.mode == BPS_MODE_RADIAL) {
		if (prm.domainLineCount <= 0) {
			return false;
		}
		const double dx = static_cast<double>(x) - prm.centerX;
		const double dy = static_cast<double>(y) - prm.centerY;
		const A_long pos = RoundToLong(std::sqrt(dx * dx + dy * dy));
		if (pos < 0 || pos >= prm.radialLength) {
			return false;
		}
		double theta = std::atan2(dy, dx);
		if (theta < 0.0) {
			theta += 2.0 * kPi;
		}
		A_long line = (prm.radialLineCount <= 1)
			? 0
			: RoundToLong((theta / (2.0 * kPi)) *
						  static_cast<double>(prm.radialLineCount));
		if (line >= prm.radialLineCount) {
			line -= prm.radialLineCount;
		}
		if (line < 0 || line >= prm.domainLineCount) {
			return false;
		}
		*line_out = line;
		*pos_key_out = static_cast<float>(pos);
		return true;
	}

	return false;
}


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

A_long BPS_JfaGridFactor(A_long frame_w, A_long frame_h)
{
	return JfaGridFactor(frame_w, frame_h);
}

static void NormaliseTangentHost(float *tx, float *ty)
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

std::vector<BpsPathSample> BPS_BuildGpuPathSeeds(
	const BpsPathSample *samples,
	A_long sample_count)
{
	std::vector<BpsPathSample> seeds;
	if (!samples || sample_count < 2) {
		return seeds;
	}
	for (A_long i = 0; i < sample_count - 1; ++i) {
		const BpsPathSample p0 = samples[i];
		const BpsPathSample p1 = samples[i + 1];
		const float dx = p1.x - p0.x;
		const float dy = p1.y - p0.y;
		const float len = std::sqrt(dx * dx + dy * dy);
		const int steps = (std::max)(1, static_cast<int>(std::ceil(len)));
		for (int k = (i > 0) ? 1 : 0; k <= steps; ++k) {
			const float t = static_cast<float>(k) / static_cast<float>(steps);
			BpsPathSample seed;
			seed.x = p0.x + dx * t;
			seed.y = p0.y + dy * t;
			seed.s = p0.s + (p1.s - p0.s) * t;
			seed.tx = p0.tx + (p1.tx - p0.tx) * t;
			seed.ty = p0.ty + (p1.ty - p0.ty) * t;
			NormaliseTangentHost(&seed.tx, &seed.ty);
			seeds.push_back(seed);
		}
	}
	return seeds;
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
	// Open-end extensions meet the frame edge; clamp so boundary seeds are kept.
	auto splat = [&](float fx, float fy, float sv, float tvx, float tvy) {
		A_long cx = RoundToLong(fx / scale);
		A_long cy = RoundToLong(fy / scale);
		if (cx < 0) {
			cx = 0;
		} else if (cx >= gridW) {
			cx = gridW - 1;
		}
		if (cy < 0) {
			cy = 0;
		} else if (cy >= gridH) {
			cy = gridH - 1;
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

// Build a pixel-owned map for analytic transform modes (Free Angle / Rotation /
// Radial / Swirl). Each pixel is classified once; lanes are sorted by posKey.
void BuildTransformMap(
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

	if (frameW <= 0 || frameH <= 0 || prm.domainLineCount <= 0) {
		return;
	}
	const A_long lane_count = prm.domainLineCount;

	const size_t pixel_count = static_cast<size_t>(frameW) *
							   static_cast<size_t>(frameH);
	std::vector<std::int32_t> laneOf(pixel_count);
	std::vector<float> keyOf(pixel_count);

	const unsigned int workers = JfaWorkerCount(frameH);
	auto row_range = [&](unsigned int t, A_long &y0, A_long &y1) {
		y0 = static_cast<A_long>((static_cast<long long>(frameH) * t) / workers);
		y1 = static_cast<A_long>((static_cast<long long>(frameH) * (t + 1)) / workers);
	};

	std::vector<std::vector<std::uint32_t>> hist(
		workers, std::vector<std::uint32_t>(static_cast<size_t>(lane_count), 0u));
	RunParallel(workers, [&](unsigned int t) {
		A_long y0 = 0, y1 = 0;
		row_range(t, y0, y1);
		std::vector<std::uint32_t> &h = hist[t];
		for (A_long y = y0; y < y1; ++y) {
			for (A_long x = 0; x < frameW; ++x) {
				const size_t pidx = static_cast<size_t>(y) *
					static_cast<size_t>(frameW) + static_cast<size_t>(x);
				A_long line = 0;
				float posKey = 0.0f;
				if (!BPS_ClassifyMappedPixel(prm, frameW, frameH, x, y, &line, &posKey)) {
					laneOf[pidx] = -1;
					continue;
				}
				laneOf[pidx] = static_cast<std::int32_t>(line);
				keyOf[pidx] = posKey;
				++h[static_cast<size_t>(line)];
			}
		}
	});

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

	RunParallel(workers, [&](unsigned int t) {
		std::vector<BpsMappedPixelRecord> scratch;
		for (A_long lane = static_cast<A_long>(t); lane < lane_count;
			 lane += static_cast<A_long>(workers)) {
			const std::uint32_t b0 = mapP->lineOffsets[static_cast<size_t>(lane)];
			const std::uint32_t b1 =
				mapP->lineOffsets[static_cast<size_t>(lane) + 1u];
			const std::uint32_t len = b1 - b0;
			if (len > 1u) {
				RadixSortMappedRecordSpan(mapP->records.data() + b0, len, &scratch);
			}
		}
	});

	mapP->mappedRecordCount = static_cast<A_long>(running);
	mapP->mappedWorkItemCount = static_cast<A_long>(work_offset);
	mapP->maxLineLength = static_cast<A_long>(max_line_length);
	mapP->domainMaxLineLength = static_cast<A_long>(max_line_length);
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

	// Phase 3: lay each lane's slots out in `order` (parallel linear radix sort).
	RunParallel(workers, [&](unsigned int t) {
		std::vector<BpsMappedPixelRecord> scratch;
		for (A_long lane = static_cast<A_long>(t); lane < lane_count;
			 lane += static_cast<A_long>(workers)) {
			const std::uint32_t b0 = mapP->lineOffsets[static_cast<size_t>(lane)];
			const std::uint32_t b1 =
				mapP->lineOffsets[static_cast<size_t>(lane) + 1u];
			const std::uint32_t len = b1 - b0;
			if (len > 1u) {
				RadixSortMappedRecordSpan(mapP->records.data() + b0, len, &scratch);
			}
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
std::uint64_t ComputePathMapKey(A_long frameW, A_long frameH, const BitonicSorterParams &prm)
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
constexpr size_t kPathMapCacheCapacity = 16;

bool BPS_ComputeGpuJfaField(
	A_long frame_w,
	A_long frame_h,
	const BitonicSorterParams &prm,
	BpsGpuJfaField *field_out)
{
	if (!field_out || frame_w <= 0 || frame_h <= 0 ||
		!prm.pathSamples || prm.pathSampleCount < 2) {
		return false;
	}
	JfaField field;
	A_long grid_w = 0;
	A_long grid_h = 0;
	A_long factor = 1;
	ComputePathField(frame_w, frame_h, prm, &field, &grid_w, &grid_h, &factor);
	field_out->qx = std::move(field.qx);
	field_out->qy = std::move(field.qy);
	field_out->s = std::move(field.s);
	field_out->tx = std::move(field.tx);
	field_out->ty = std::move(field.ty);
	field_out->d2 = std::move(field.d2);
	field_out->gridW = grid_w;
	field_out->gridH = grid_h;
	field_out->factor = factor;
	return !field_out->d2.empty();
}

std::vector<BpsJfaCellGpu> BPS_PackGpuJfaCells(const BpsGpuJfaField &field)
{
	const size_t count =
		static_cast<size_t>(field.gridW) * static_cast<size_t>(field.gridH);
	std::vector<BpsJfaCellGpu> cells(count);
	for (size_t i = 0; i < count; ++i) {
		cells[i].qx = field.qx[i];
		cells[i].qy = field.qy[i];
		cells[i].s = field.s[i];
		cells[i].tx = field.tx[i];
		cells[i].ty = field.ty[i];
		cells[i].d2 = field.d2[i];
	}
	return cells;
}

std::uint64_t BPS_PathMapKey(
	A_long frameW,
	A_long frameH,
	const BitonicSorterParams &prm)
{
	return ComputePathMapKey(frameW, frameH, prm);
}

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

	const std::uint64_t key = BPS_PathMapKey(frameW, frameH, prm);

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

std::uint64_t ComputeTransformMapKey(A_long frameW, A_long frameH, const BitonicSorterParams &prm)
{
	std::uint64_t h = 1469598103934665603ull;
	const A_long header[] = {
		frameW, frameH, prm.mode, prm.direction,
		prm.freePMin, prm.freeQMin, prm.freeLineLength, prm.freeLineCount,
		prm.radialLength, prm.radialLineCount, prm.swirlLineMin,
		prm.domainLineCount, prm.domainMaxLineLength
	};
	h = HashBytes(h, header, sizeof(header));
	const float floats[] = {
		prm.angleCos, prm.angleSin, prm.angleRadians,
		prm.centerX, prm.centerY, prm.swirlK
	};
	h = HashBytes(h, floats, sizeof(floats));
	return h;
}

std::uint64_t BPS_TransformMapKey(
	A_long frameW,
	A_long frameH,
	const BitonicSorterParams &prm)
{
	return ComputeTransformMapKey(frameW, frameH, prm);
}

std::shared_ptr<const BpsPathMap> BPS_AcquireTransformMap(
	A_long frameW,
	A_long frameH,
	const BitonicSorterParams &prm)
{
	if (!BPS_ModeUsesTransformMap(prm.mode) || frameW <= 0 || frameH <= 0 ||
		prm.domainLineCount <= 0) {
		return nullptr;
	}

	const std::uint64_t key = BPS_TransformMapKey(frameW, frameH, prm);

	{
		std::lock_guard<std::mutex> lock(g_pathMapCacheMutex);
		for (size_t i = 0; i < g_pathMapCache.size(); ++i) {
			if (g_pathMapCache[i].key == key && g_pathMapCache[i].map) {
				std::shared_ptr<const BpsPathMap> hit = g_pathMapCache[i].map;
				if (i != 0) {
					const PathMapCacheEntry entry = g_pathMapCache[i];
					g_pathMapCache.erase(g_pathMapCache.begin() +
										 static_cast<std::ptrdiff_t>(i));
					g_pathMapCache.insert(g_pathMapCache.begin(), entry);
				}
				return hit;
			}
		}
	}

	std::shared_ptr<BpsPathMap> built = std::make_shared<BpsPathMap>();
	BuildTransformMap(frameW, frameH, prm, built.get());

	{
		std::lock_guard<std::mutex> lock(g_pathMapCacheMutex);
		for (const PathMapCacheEntry &entry : g_pathMapCache) {
			if (entry.key == key && entry.map) {
				return entry.map;
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

	const A_long frameW = BPS_RenderWidth(in_data);
	const A_long frameH = BPS_RenderHeight(in_data);
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

	// Prefer the AEGP layer mask outline (correct drawn-mask geometry, tangents
	// relative to position), then fall back to the effect path parameter. This
	// matches the cfc4bcd ordering; preferring the PF param sampled a different
	// path and produced a completely wrong shape.
	const char *dbgSrc = "none";
	if (SamplePathFromAegpMask(in_data, suites, plugin_id, path_id,
							   samplesP, &openB)) {
		dbgSrc = "AEGP_mask";
	} else {
		samplesP->clear();
		if (SamplePathFromEffectParam(in_data, suites, path_id, samplesP, &openB)) {
			dbgSrc = "PF_param";
		}
	}

	if (samplesP->size() < 2u) {
		ResolvePathDomainBounds(paramsP, diag);
		(void)out_data;
		return PF_Err_NONE;
	}

	// Map mask vertices into effect-buffer space (top-left origin).
	// Footage layers: anchor is typically (W/2, H/2) ⇒ identity.
	// Shape/text layers: default anchor is (0, 0) at the layer centre ⇒ + (W/2, H/2).
	// buffer = mask - anchor + (W/2, H/2)
	if (plugin_id && suites.PFInterfaceSuite1() && suites.StreamSuite5() &&
		suites.LayerSuite5()) {
		AEGP_LayerH layerH = NULL;
		if (suites.PFInterfaceSuite1()->AEGP_GetEffectLayer(
				in_data->effect_ref, &layerH) == A_Err_NONE &&
			layerH) {
			AEGP_StreamRefH anchor_streamH = NULL;
			if (suites.StreamSuite5()->AEGP_GetNewLayerStream(
					plugin_id, layerH, AEGP_LayerStream_ANCHORPOINT,
					&anchor_streamH) == A_Err_NONE &&
				anchor_streamH) {
				A_Time timeT;
				timeT.value = in_data->current_time;
				timeT.scale = in_data->time_scale;
				AEGP_StreamValue2 anchor_val;
				AEFX_CLR_STRUCT(anchor_val);
				if (suites.StreamSuite5()->AEGP_GetNewStreamValue(
						plugin_id, anchor_streamH, AEGP_LTimeMode_LayerTime,
						&timeT, TRUE, &anchor_val) == A_Err_NONE) {
					const float anchor_x =
						static_cast<float>(anchor_val.val.two_d.x);
					const float anchor_y =
						static_cast<float>(anchor_val.val.two_d.y);
					const float ox =
						0.5f * static_cast<float>(in_data->width) - anchor_x;
					const float oy =
						0.5f * static_cast<float>(in_data->height) - anchor_y;
					if (std::fabs(ox) > 1.0e-3f || std::fabs(oy) > 1.0e-3f) {
						for (BpsPathSample &sample : *samplesP) {
							sample.x += ox;
							sample.y += oy;
						}
					}
					(void)suites.StreamSuite5()->AEGP_DisposeStreamValue(
						&anchor_val);
				}
				(void)suites.StreamSuite5()->AEGP_DisposeStream(anchor_streamH);
			}
		}
	}

	// Mask geometry is authored in full-resolution layer space. Path mode sorts
	// in the current render space, so apply AE's downsample ratio unconditionally
	// and rebuild arc length after any non-uniform scale.
	const float dsx = paramsP->downsampleX > 0.0f ? paramsP->downsampleX : 1.0f;
	const float dsy = paramsP->downsampleY > 0.0f ? paramsP->downsampleY : 1.0f;
	if (std::fabs(dsx - 1.0f) > 1.0e-6f ||
		std::fabs(dsy - 1.0f) > 1.0e-6f) {
		for (BpsPathSample &sample : *samplesP) {
			sample.x *= dsx;
			sample.y *= dsy;
			sample.tx *= dsx;
			sample.ty *= dsy;
			NormaliseTangent(&sample.tx, &sample.ty);
		}
		float s = 0.0f;
		samplesP->front().s = 0.0f;
		for (size_t i = 1u; i < samplesP->size(); ++i) {
			const BpsPathSample &prev = (*samplesP)[i - 1u];
			BpsPathSample &sample = (*samplesP)[i];
			const float dx = sample.x - prev.x;
			const float dy = sample.y - prev.y;
			s += std::sqrt(dx * dx + dy * dy);
			sample.s = s;
		}
	}

	const bool closed = (openB == FALSE);
	paramsP->pathClosed = closed ? 1 : 0;

	const float frame_w = static_cast<float>(frameW);
	const float frame_h = static_cast<float>(frameH);

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
		// Open ends: extend along end tangents to the frame edge, then the
		// whole polyline (authored path + rays) is the sort path.
		// --- TEMP DEBUG DUMP (Path open-end diagnosis) ---
		const BpsPathSample dbgAuthFirst = samplesP->front();
		const BpsPathSample dbgAuthSecond = (*samplesP)[1];
		const BpsPathSample dbgAuthLast = samplesP->back();
		const BpsPathSample dbgAuthPrev = (*samplesP)[samplesP->size() - 2u];
		const A_long dbgAuthCount = static_cast<A_long>(samplesP->size());
		ExtendOpenEndsToFrameEdges(samplesP, frame_w, frame_h);
		paramsP->pathLength = samplesP->back().s;
		{
			const char *tmp = std::getenv("TEMP");
			char path[1024];
			std::snprintf(path, sizeof(path), "%s\\bps_pathdump.txt",
						  tmp ? tmp : ".");
			FILE *fp = std::fopen(path, "w");
			if (fp) {
				std::fprintf(fp, "frame WxH = %ld x %ld  dir=%ld  source=%s\n",
							 (long)frameW, (long)frameH, (long)paramsP->pathDirection, dbgSrc);
				std::fprintf(fp, "authored count=%ld\n", (long)dbgAuthCount);
				std::fprintf(fp, "auth FIRST  (%.1f,%.1f) tan(%.3f,%.3f)\n",
							 dbgAuthFirst.x, dbgAuthFirst.y, dbgAuthFirst.tx, dbgAuthFirst.ty);
				std::fprintf(fp, "auth SECOND (%.1f,%.1f)  dir(second-first)=(%.3f,%.3f)\n",
							 dbgAuthSecond.x, dbgAuthSecond.y,
							 dbgAuthSecond.x - dbgAuthFirst.x, dbgAuthSecond.y - dbgAuthFirst.y);
				std::fprintf(fp, "auth PREV   (%.1f,%.1f)  dir(last-prev)=(%.3f,%.3f)\n",
							 dbgAuthPrev.x, dbgAuthPrev.y,
							 dbgAuthLast.x - dbgAuthPrev.x, dbgAuthLast.y - dbgAuthPrev.y);
				std::fprintf(fp, "auth LAST   (%.1f,%.1f) tan(%.3f,%.3f)\n",
							 dbgAuthLast.x, dbgAuthLast.y, dbgAuthLast.tx, dbgAuthLast.ty);
				std::fprintf(fp, "-- after extend: count=%ld --\n",
							 (long)samplesP->size());
				std::fprintf(fp, "ext  FIRST  (%.1f,%.1f)  <- start extension target\n",
							 samplesP->front().x, samplesP->front().y);
				std::fprintf(fp, "ext  LAST   (%.1f,%.1f)  <- end extension target\n",
							 samplesP->back().x, samplesP->back().y);
				std::fclose(fp);
			}
		}
		// --- END TEMP DEBUG DUMP ---
	}

	paramsP->pathSamples = samplesP->data();
	paramsP->pathSampleCount = static_cast<A_long>(samplesP->size());

	ResolvePathDomainBounds(paramsP, diag);
	(void)out_data;
	(void)kPi;
	return PF_Err_NONE;
}
