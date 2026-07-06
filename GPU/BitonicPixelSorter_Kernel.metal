/*
	BitonicPixelSorter_Kernel.metal

	Metal Shading Language port of BitonicPixelSorter_Kernel.cl.
	Axis mode: one threadgroup sorts one line into the destination.
	Non-axis modes use exact multi-pass (domain index buffer + inverse map).
	AE GPU worlds are BGRA float4 buffers, with row pitches in float4 units.

	Buffer bindings (axis BitonicSortKernel):
	  buffer(0) - srcTex     : device const float4*   (source image)
	  buffer(1) - sortTex    : device float4*          (destination image)
	  buffer(2) - params     : constant BitonicSortParams&

	Buffer bindings (BitonicSortDomainKernel):
	  buffer(0) - srcTex, buffer(1) - domain (uint*), buffer(2) - keys (float*),
	  buffer(3) - params

	Buffer bindings (BitonicApplyDomainKernel):
	  buffer(0) - srcTex, buffer(1) - sortTex, buffer(2) - domain, buffer(3) - params

	Threadgroup memory budget:
	  scratchKey[4096]   = 4096 * 4 bytes = 16 384 bytes
	  scratchIndex[4096] = 4096 * 4 bytes = 16 384 bytes
	  Total              =                  32 768 bytes  (32 KB)
	The host validates that the device supports at least 32 KB of threadgroup memory
	before building the pipeline; if not, it returns PF_Err_INTERNAL_STRUCT_DAMAGED
	so AE falls back to the CPU path.
*/

#include <metal_stdlib>
using namespace metal;

#define MAX_THREADS 256u
#define MAX_SIZE    4096u
#define BPS_FLOAT_MAX 3.402823466e+38F
#define BPS_TWO_PI 6.28318530717958647692f

#define BPS_MODE_AXIS 1
#define BPS_MODE_FREE_ANGLE 2
#define BPS_MODE_ROTATION 3
#define BPS_MODE_RADIAL 4
#define BPS_MODE_SWIRL 5
#define BPS_MODE_PATH 6
#define BPS_PATH_DIR_NORMAL 1
#define BPS_PATH_DIR_TANGENT 2
#define BPS_SWIRL_K_EPS 1.0e-4f

#define BPS_CRITERION_RGB_AVERAGE 2
#define BPS_CRITERION_RGB_PRODUCT 3
#define BPS_CRITERION_RGB_MINIMUM 4
#define BPS_CRITERION_RGB_MAXIMUM 5
#define BPS_CRITERION_RED_CHANNEL 6
#define BPS_CRITERION_GREEN_CHANNEL 7
#define BPS_CRITERION_BLUE_CHANNEL 8
#define BPS_CRITERION_ALPHA_CHANNEL 9
#define BPS_CRITERION_HUE 10
#define BPS_CRITERION_SATURATION 11

#define BPS_AFFECT_INSIDE_THRESHOLDS 1
#define BPS_AFFECT_OUTSIDE_THRESHOLDS 2

// ---------------------------------------------------------------------------
// Parameter struct - must match BitonicSortParams in BPS_MetalBackend.mm
// exactly (same field order, same types) so the host can cast &struct to a
// byte pointer and upload it directly.
// ---------------------------------------------------------------------------
struct BitonicSortParams {
	int srcPitch;
	int dstPitch;
	int width;
	int height;
	int inputOriginX;
	int inputOriginY;
	int inputWidth;
	int inputHeight;
	int criterionPitch;
	int criterionOriginX;
	int criterionOriginY;
	int criterionWidth;
	int criterionHeight;
	int triggerPitch;
	int triggerOriginX;
	int triggerOriginY;
	int triggerWidth;
	int triggerHeight;
	int outputOriginX;
	int outputOriginY;
	int outputWidth;
	int outputHeight;
	int mode;
	int direction;
	int ordering;
	int criterion;
	int trigger;
	int affect;
	float cycleDegrees;
	int lineCount;
	int freePMin;
	int freeQMin;
	int freeLineLength;
	int radialLength;
	int domainStride;
	float thresholdMin;
	float thresholdMax;
	float angleCos;
	float angleSin;
	float centerX;
	float centerY;
	float swirlK;
	int   swirlLineMin;
	int   pathDirection;
	int   pathClosed;
	float pathLength;
	int   pathSMin;
	int   pathNMin;
	int   pathSampleCount;
};

struct BpsPathSampleGpu {
	float x;
	float y;
	float s;
	float tx;
	float ty;
};

struct BpsMappedPixelRecordGpu {
	float posKey;
	uint pixelIndex;
};

// ---------------------------------------------------------------------------
// Helper functions - direct ports of the OpenCL inline helpers.
// ---------------------------------------------------------------------------
inline float bps_sort_key(float4 c, int criterion)
{
	// BGRA: R=.z, G=.y, B=.x
	if (criterion == BPS_CRITERION_RGB_AVERAGE) {
		return clamp((c.z + c.y + c.x) * (1.0f / 3.0f), 0.0f, 1.0f);
	}
	if (criterion == BPS_CRITERION_RGB_PRODUCT) {
		return clamp(c.z * c.y * c.x, 0.0f, 1.0f);
	}
	if (criterion == BPS_CRITERION_RGB_MINIMUM) {
		return clamp(min(min(c.z, c.y), c.x), 0.0f, 1.0f);
	}
	if (criterion == BPS_CRITERION_RGB_MAXIMUM) {
		return clamp(max(max(c.z, c.y), c.x), 0.0f, 1.0f);
	}
	if (criterion == BPS_CRITERION_RED_CHANNEL) {
		return clamp(c.z, 0.0f, 1.0f);
	}
	if (criterion == BPS_CRITERION_GREEN_CHANNEL) {
		return clamp(c.y, 0.0f, 1.0f);
	}
	if (criterion == BPS_CRITERION_BLUE_CHANNEL) {
		return clamp(c.x, 0.0f, 1.0f);
	}
	if (criterion == BPS_CRITERION_ALPHA_CHANNEL) {
		return clamp(c.w, 0.0f, 1.0f);
	}
	if (criterion == BPS_CRITERION_HUE || criterion == BPS_CRITERION_SATURATION) {
		const float r = clamp(c.z, 0.0f, 1.0f);
		const float g = clamp(c.y, 0.0f, 1.0f);
		const float b = clamp(c.x, 0.0f, 1.0f);
		const float maxRGB = max(max(r, g), b);
		const float minRGB = min(min(r, g), b);
		const float delta = maxRGB - minRGB;
		if (criterion == BPS_CRITERION_SATURATION) {
			return maxRGB <= 0.0f ? 0.0f : clamp(delta / maxRGB, 0.0f, 1.0f);
		}
		if (delta <= 0.0f) {
			return 0.0f;
		}
		float hue = 0.0f;
		if (maxRGB == r) {
			hue = (g - b) / delta;
			if (hue < 0.0f) hue += 6.0f;
		} else if (maxRGB == g) {
			hue = ((b - r) / delta) + 2.0f;
		} else {
			hue = ((r - g) / delta) + 4.0f;
		}
		return clamp(hue * (1.0f / 6.0f), 0.0f, 1.0f);
	}
	return clamp(0.298912f * c.z + 0.586611f * c.y + 0.114478f * c.x, 0.0f, 1.0f);
}

inline bool bps_is_affected(float triggerKey, float thresholdMin, float thresholdMax, int affect)
{
	const bool inside = triggerKey >= thresholdMin && triggerKey <= thresholdMax;
	return affect == BPS_AFFECT_OUTSIDE_THRESHOLDS ? !inside : inside;
}

struct BpsKeySource {
	device const float4 *tex;
	int pitch;
	int originX;
	int originY;
	int width;
	int height;
};

inline bool bps_key_in_world(BpsKeySource src, int x, int y)
{
	return x >= src.originX && y >= src.originY &&
		   x < src.originX + src.width && y < src.originY + src.height;
}

inline float bps_sample_key(BpsKeySource src, int x, int y, int keyCriterion)
{
	if (!bps_key_in_world(src, x, y)) {
		return -1.0f;
	}
	const uint idx = (uint)((x - src.originX) + (y - src.originY) * src.pitch);
	return bps_sort_key(src.tex[idx], keyCriterion);
}

inline float bps_sample_key_from_src_index(
	BpsKeySource src,
	int srcPitch,
	int inputOriginX,
	int inputOriginY,
	uint srcIndex,
	int keyCriterion)
{
	if (srcIndex == 0xffffffffu) {
		return -1.0f;
	}
	const int x = (int)(srcIndex % (uint)srcPitch) + inputOriginX;
	const int y = (int)(srcIndex / (uint)srcPitch) + inputOriginY;
	return bps_sample_key(src, x, y, keyCriterion);
}

inline BpsKeySource bps_criterion_src(device const float4 *tex, constant BitonicSortParams &p)
{
	BpsKeySource src;
	src.tex = tex;
	src.pitch = p.criterionPitch;
	src.originX = p.criterionOriginX;
	src.originY = p.criterionOriginY;
	src.width = p.criterionWidth;
	src.height = p.criterionHeight;
	return src;
}

inline BpsKeySource bps_trigger_src(device const float4 *tex, constant BitonicSortParams &p)
{
	BpsKeySource src;
	src.tex = tex;
	src.pitch = p.triggerPitch;
	src.originX = p.triggerOriginX;
	src.originY = p.triggerOriginY;
	src.width = p.triggerWidth;
	src.height = p.triggerHeight;
	return src;
}


inline uint bps_cycle_shift(uint count, float cycleDegrees)
{
	if (count <= 1u) return 0u;
	int shift = (int)floor((cycleDegrees / 360.0f) * (float)count + 0.5f);
	const int n = (int)count;
	shift %= n;
	if (shift < 0) shift += n;
	return (uint)shift;
}

inline uint bps_next_pow2(uint value)
{
	if (value <= 1u) return 1u;
	value--;
	value |= value >> 1;
	value |= value >> 2;
	value |= value >> 4;
	value |= value >> 8;
	value |= value >> 16;
	return value + 1u;
}

inline bool bps_before(float keyA, uint indexA, float keyB, uint indexB)
{
	if (keyA < keyB) return true;
	if (keyA > keyB) return false;
	return indexA < indexB;
}

inline bool bps_record_before(const BpsMappedPixelRecordGpu a,
							  const BpsMappedPixelRecordGpu b)
{
	if (a.posKey < b.posKey) return true;
	if (a.posKey > b.posKey) return false;
	return a.pixelIndex < b.pixelIndex;
}

inline int bps_round_to_int(float value)
{
	return (int)floor(value + 0.5f);
}

inline uint bps_rotation_line_length(uint radius)
{
	if (radius == 0u) return 1u;
	const uint length = (uint)ceil(BPS_TWO_PI * (float)radius);
	return length == 0u ? 1u : length;
}

inline void bps_normalise_tangent(thread float *tx, thread float *ty)
{
	const float len = sqrt((*tx) * (*tx) + (*ty) * (*ty));
	if (len > 1.0e-6f) {
		*tx /= len;
		*ty /= len;
	} else {
		*tx = 1.0f;
		*ty = 0.0f;
	}
}

inline bool bps_path_eval_at_s(device const BpsPathSampleGpu *samples, int sampleCount,
							   float s, thread float *x, thread float *y,
							   thread float *tx, thread float *ty)
{
	if (!samples || sampleCount < 2) return false;
	if (s <= samples[0].s) {
		*x = samples[0].x; *y = samples[0].y;
		*tx = samples[0].tx; *ty = samples[0].ty;
		return true;
	}
	if (s >= samples[sampleCount - 1].s) {
		*x = samples[sampleCount - 1].x; *y = samples[sampleCount - 1].y;
		*tx = samples[sampleCount - 1].tx; *ty = samples[sampleCount - 1].ty;
		return true;
	}
	for (int i = 0; i < sampleCount - 1; ++i) {
		const BpsPathSampleGpu a = samples[i];
		const BpsPathSampleGpu b = samples[i + 1];
		if (s < a.s || s > b.s) continue;
		const float seg = b.s - a.s;
		const float t = seg > 1.0e-6f ? ((s - a.s) / seg) : 0.0f;
		*x = a.x + (b.x - a.x) * t;
		*y = a.y + (b.y - a.y) * t;
		*tx = a.tx + (b.tx - a.tx) * t;
		*ty = a.ty + (b.ty - a.ty) * t;
		bps_normalise_tangent(tx, ty);
		return true;
	}
	return false;
}

inline bool bps_path_closest(device const BpsPathSampleGpu *samples, int sampleCount,
							 float px, float py, thread float *sOut, thread float *nOut)
{
	if (!samples || sampleCount < 2) return false;
	float bestDist2 = 1.0e30f;
	float bestS = samples[0].s;
	float bestN = 0.0f;
	for (int i = 0; i < sampleCount - 1; ++i) {
		const BpsPathSampleGpu a = samples[i];
		const BpsPathSampleGpu b = samples[i + 1];
		const float abx = b.x - a.x;
		const float aby = b.y - a.y;
		const float abLen2 = abx * abx + aby * aby;
		float t = 0.0f;
		if (abLen2 > 1.0e-6f) {
			t = clamp(((px - a.x) * abx + (py - a.y) * aby) / abLen2, 0.0f, 1.0f);
		}
		const float cx = a.x + abx * t;
		const float cy = a.y + aby * t;
		const float dx = px - cx;
		const float dy = py - cy;
		const float dist2 = dx * dx + dy * dy;
		if (dist2 < bestDist2) {
			bestDist2 = dist2;
			bestS = a.s + (b.s - a.s) * t;
			float tx = a.tx + (b.tx - a.tx) * t;
			float ty = a.ty + (b.ty - a.ty) * t;
			bps_normalise_tangent(&tx, &ty);
			bestN = -ty * dx + tx * dy;
		}
	}
	*sOut = bestS;
	*nOut = bestN;
	return true;
}

inline float bps_wrap_arc_length(float s, float length)
{
	if (length <= 1.0e-6f) return s;
	float w = fmod(s, length);
	if (w < 0.0f) w += length;
	return w;
}

inline bool bps_path_lane_order(constant BitonicSortParams &p,
								float s, float n,
								thread int *lineP,
								thread float *orderP)
{
	const bool closed = (p.pathClosed != 0) && p.pathLength > 1.0e-6f;
	const float sLocal = closed ? bps_wrap_arc_length(s, p.pathLength) : s;
	int line = 0;
	float order = 0.0f;
	if (p.pathDirection == BPS_PATH_DIR_TANGENT) {
		line = bps_round_to_int(n) - p.pathNMin;
		order = sLocal;
	} else {
		if (closed) {
			const int bins = p.lineCount > 0 ? p.lineCount : 1;
			int q = bps_round_to_int(sLocal) % bins;
			if (q < 0) q += bins;
			line = q;
		} else {
			line = bps_round_to_int(s) - p.pathSMin;
		}
		order = n;
	}
	if (line < 0 || line >= p.lineCount) {
		return false;
	}
	*lineP = line;
	*orderP = order;
	return true;
}

inline uint bps_line_size(constant BitonicSortParams &p, uint gid)
{
	if (p.mode == BPS_MODE_FREE_ANGLE) return (uint)p.freeLineLength;
	if (p.mode == BPS_MODE_ROTATION) return bps_rotation_line_length(gid);
	if (p.mode == BPS_MODE_RADIAL) return (uint)p.radialLength;
	if (p.mode == BPS_MODE_SWIRL) {
		return (uint)(p.radialLength > 0 ? p.radialLength : 1);
	}
	if (p.mode == BPS_MODE_PATH) return (uint)(p.radialLength > 0 ? p.radialLength : 1);
	return p.direction ? (uint)p.width : (uint)p.height;
}

inline bool bps_coord_for_pos(constant BitonicSortParams &p,
							  device const BpsPathSampleGpu *pathSamples,
							  uint gid, uint pos,
							  thread int *x, thread int *y)
{
	if (p.mode == BPS_MODE_FREE_ANGLE) {
		const float pp = (float)(p.freePMin + (int)pos);
		const float q = (float)(p.freeQMin + (int)gid);
		*x = bps_round_to_int(pp * p.angleCos - q * p.angleSin);
		*y = bps_round_to_int(pp * p.angleSin + q * p.angleCos);
	} else if (p.mode == BPS_MODE_ROTATION) {
		const uint lineLen = bps_rotation_line_length(gid);
		const float theta = (lineLen <= 1u) ? 0.0f :
			(BPS_TWO_PI * (float)pos) / (float)lineLen;
		const float c = cos(theta);
		const float s = sin(theta);
		const float radius = (float)gid;
		*x = bps_round_to_int(p.centerX + radius * (s * p.angleCos + c * p.angleSin));
		*y = bps_round_to_int(p.centerY + radius * (s * p.angleSin - c * p.angleCos));
	} else if (p.mode == BPS_MODE_RADIAL) {
		const float denom = p.lineCount <= 0 ? 1.0f : (float)p.lineCount;
		const float theta = (BPS_TWO_PI * (float)gid) / denom;
		*x = bps_round_to_int(p.centerX + (float)pos * cos(theta));
		*y = bps_round_to_int(p.centerY + (float)pos * sin(theta));
	} else if (p.mode == BPS_MODE_SWIRL) {
		const float r = (float)pos;
		const float denom = p.lineCount <= 0 ? 1.0f : (float)p.lineCount;
		const float phase = (BPS_TWO_PI * (float)gid) / denom;
		const float angle = atan2(p.angleSin, p.angleCos);
		const float theta = phase + p.swirlK * r + angle;
		*x = bps_round_to_int(p.centerX + r * cos(theta));
		*y = bps_round_to_int(p.centerY + r * sin(theta));
	} else if (p.mode == BPS_MODE_PATH) {
		float s = 0.0f;
		float n = 0.0f;
		if (p.pathDirection == BPS_PATH_DIR_NORMAL) {
			s = (float)(p.pathSMin + (int)gid);
			n = (float)(p.pathNMin + (int)pos);
		} else {
			n = (float)(p.pathNMin + (int)gid);
			s = (float)(p.pathSMin + (int)pos);
		}
		float px = 0.0f, py = 0.0f, tx = 1.0f, ty = 0.0f;
		if (!bps_path_eval_at_s(pathSamples, p.pathSampleCount, s, &px, &py, &tx, &ty)) {
			return false;
		}
		*x = bps_round_to_int(px + n * (-ty));
		*y = bps_round_to_int(py + n * tx);
	} else {
		const int lineLayer = p.direction
			? (p.outputOriginY + (int)gid)
			: (p.outputOriginX + (int)gid);
		*x = p.direction ? (int)pos : lineLayer;
		*y = p.direction ? lineLayer : (int)pos;
	}
	return *x >= 0 && *y >= 0 && *x < p.width && *y < p.height;
}

inline bool bps_src_in_world(constant BitonicSortParams &p, int x, int y)
{
	return x >= p.inputOriginX && y >= p.inputOriginY &&
		x < p.inputOriginX + p.inputWidth && y < p.inputOriginY + p.inputHeight;
}

inline bool bps_dst_in_world(constant BitonicSortParams &p, int x, int y)
{
	return x >= p.outputOriginX && y >= p.outputOriginY &&
		x < p.outputOriginX + p.outputWidth && y < p.outputOriginY + p.outputHeight;
}

inline uint bps_src_index_xy(constant BitonicSortParams &p, int x, int y)
{
	return (uint)((x - p.inputOriginX) + (y - p.inputOriginY) * p.srcPitch);
}

inline uint bps_dst_index_xy(constant BitonicSortParams &p, int x, int y)
{
	return (uint)((x - p.outputOriginX) + (y - p.outputOriginY) * p.dstPitch);
}

inline bool bps_domain_pos_for_pixel(constant BitonicSortParams &p,
									 device const BpsPathSampleGpu *pathSamples,
									 int x, int y,
									 thread int *lineP, thread int *posP)
{
	int line = 0;
	int pos = 0;

	if (p.mode == BPS_MODE_FREE_ANGLE) {
		const float pp = (float)x * p.angleCos + (float)y * p.angleSin;
		const float q = -(float)x * p.angleSin + (float)y * p.angleCos;
		pos = bps_round_to_int(pp) - p.freePMin;
		line = bps_round_to_int(q) - p.freeQMin;
	} else if (p.mode == BPS_MODE_ROTATION) {
		const float dx = (float)x - p.centerX;
		const float dy = (float)y - p.centerY;
		line = bps_round_to_int(sqrt(dx * dx + dy * dy));
		const uint lineLen = bps_rotation_line_length((uint)max(line, 0));
		if (lineLen <= 1u) {
			pos = 0;
		} else {
			const float rx = dx * p.angleCos + dy * p.angleSin;
			const float ry = -dx * p.angleSin + dy * p.angleCos;
			float theta = atan2(rx, -ry);
			if (theta < 0.0f) theta += BPS_TWO_PI;
			pos = bps_round_to_int((theta / BPS_TWO_PI) * (float)lineLen);
			if (pos >= (int)lineLen) pos -= (int)lineLen;
		}
	} else if (p.mode == BPS_MODE_RADIAL) {
		const float dx = (float)x - p.centerX;
		const float dy = (float)y - p.centerY;
		pos = bps_round_to_int(sqrt(dx * dx + dy * dy));
		float theta = atan2(dy, dx);
		if (theta < 0.0f) theta += BPS_TWO_PI;
		line = (p.lineCount <= 1) ? 0 :
			bps_round_to_int((theta / BPS_TWO_PI) * (float)p.lineCount);
		if (line >= p.lineCount) line -= p.lineCount;
	} else if (p.mode == BPS_MODE_SWIRL) {
		const float dx = (float)x - p.centerX;
		const float dy = (float)y - p.centerY;
		const float r = sqrt(dx * dx + dy * dy);
		const float angle = atan2(p.angleSin, p.angleCos);
		float phase = atan2(dy, dx) - angle - p.swirlK * r;
		phase = fmod(phase, BPS_TWO_PI);
		if (phase < 0.0f) phase += BPS_TWO_PI;
		line = (p.lineCount <= 1) ? 0 :
			bps_round_to_int((phase / BPS_TWO_PI) * (float)p.lineCount);
		if (line >= p.lineCount) line -= p.lineCount;
		pos = bps_round_to_int(r);
	} else if (p.mode == BPS_MODE_PATH) {
		float s = 0.0f;
		float n = 0.0f;
		if (!bps_path_closest(pathSamples, p.pathSampleCount, (float)x, (float)y, &s, &n)) {
			return false;
		}
		if (p.pathDirection == BPS_PATH_DIR_NORMAL) {
			line = bps_round_to_int(s) - p.pathSMin;
			pos = bps_round_to_int(n) - p.pathNMin;
		} else {
			line = bps_round_to_int(n) - p.pathNMin;
			pos = bps_round_to_int(s) - p.pathSMin;
		}
	} else {
		return false;
	}

	if (line < 0 || line >= p.lineCount) return false;

	int lineLen = 0;
	if (p.mode == BPS_MODE_FREE_ANGLE) lineLen = p.freeLineLength;
	else if (p.mode == BPS_MODE_ROTATION) lineLen = (int)bps_rotation_line_length((uint)line);
	else lineLen = p.radialLength;

	if (pos < 0 || pos >= lineLen || pos >= p.domainStride) return false;

	*lineP = line;
	*posP = pos;
	return true;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
kernel void BitonicSortKernel(
	device const float4       *srcTex  [[buffer(0)]],
	device float4             *sortTex [[buffer(1)]],
	device const float4       *criterionTex [[buffer(2)]],
	device const float4       *triggerTex [[buffer(3)]],
	constant BitonicSortParams &p      [[buffer(4)]],
	device const BpsPathSampleGpu *pathSamples [[buffer(5)]],
	uint gid  [[threadgroup_position_in_grid]],
	uint gtid [[thread_position_in_threadgroup]])
{
	// scratchKey + scratchIndex together consume the full 32 KB threadgroup
	// budget validated by the host. The span metadata reuses scratchIndex[0..3]
	// rather than dedicated threadgroup variables; thread 0 publishes it; every
	// thread copies it into thread-local registers; a second barrier guarantees
	// those copies complete before the load loop overwrites scratchIndex,
	// removing the data race while keeping spanSize/sortSize uniform (no
	// divergent barriers).
	threadgroup float scratchKey[MAX_SIZE];
	threadgroup uint  scratchIndex[MAX_SIZE];

	const uint size = bps_line_size(p, gid);
	if (size == 0u || size > MAX_SIZE) {
		return;
	}

	// Always write every destination pixel. Partial input worlds (common with
	// alpha / adjustment layers) must not leave stale frame data behind.
	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		int x = 0;
		int y = 0;
		if (bps_coord_for_pos(p, pathSamples, gid, pos, &x, &y) &&
			bps_dst_in_world(p, x, y)) {
			sortTex[bps_dst_index_xy(p, x, y)] = bps_src_in_world(p, x, y)
				? srcTex[bps_src_index_xy(p, x, y)]
				: float4(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	uint cursor = 0u;
	while (cursor < size) {
		// Thread 0 finds the next in-threshold span and publishes [start, end,
		// size, sortSize] into scratchIndex[0..3].
		if (gtid == 0u) {
			uint spanStart = cursor;
			while (spanStart < size) {
				int x = 0;
				int y = 0;
				if (bps_coord_for_pos(p, pathSamples, gid, spanStart, &x, &y) &&
					bps_src_in_world(p, x, y)) {
					float br = bps_sample_key(bps_trigger_src(triggerTex, p), x, y, p.trigger);
					if (bps_is_affected(br, p.thresholdMin, p.thresholdMax, p.affect)) break;
				}
				spanStart++;
			}

			uint spanEnd = spanStart;
			while (spanEnd < size) {
				int x = 0;
				int y = 0;
				if (!bps_coord_for_pos(p, pathSamples, gid, spanEnd, &x, &y) ||
					!bps_src_in_world(p, x, y)) break;
				float br = bps_sample_key(bps_trigger_src(triggerTex, p), x, y, p.trigger);
				if (!bps_is_affected(br, p.thresholdMin, p.thresholdMax, p.affect)) break;
				spanEnd++;
			}

			scratchIndex[0] = spanStart;
			scratchIndex[1] = spanEnd;
			scratchIndex[2] = spanEnd - spanStart;
			scratchIndex[3] = bps_next_pow2(scratchIndex[2]);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		const uint spanStart = scratchIndex[0];
		const uint spanEnd   = scratchIndex[1];
		const uint spanSize  = scratchIndex[2];
		const uint sortSize  = scratchIndex[3];

		// Publish/consume fence: every thread has now copied the metadata into
		// thread-local registers, so the load loop below may safely overwrite
		// scratchIndex without racing the reads above.
		threadgroup_barrier(mem_flags::mem_threadgroup);

		if (spanStart >= size || spanSize == 0u) {
			break;
		}

		const bool ascending = p.ordering != 0;
		for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
			if (i < spanSize) {
				const uint pos = spanStart + i;
				int x = 0;
				int y = 0;
				const bool valid = bps_coord_for_pos(p, pathSamples, gid, pos, &x, &y) &&
					bps_src_in_world(p, x, y);
				const uint srcIndex = valid ? bps_src_index_xy(p, x, y) : 0xffffffffu;
				scratchKey[i] = valid
					? bps_sample_key(bps_criterion_src(criterionTex, p), x, y, p.criterion)
					: (ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX);
				scratchIndex[i] = srcIndex;
			} else {
				scratchKey[i]   = ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX;
				scratchIndex[i] = 0xffffffffu;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		for (uint k = 2u; k <= sortSize; k <<= 1) {
			for (uint j = k >> 1; j > 0u; j >>= 1) {
				for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
					const uint partner = i ^ j;
					if (partner > i) {
						bool stageAscending = (i & k) == 0u;
						if (!ascending) stageAscending = !stageAscending;

						const float keyA  = scratchKey[i];
						const float keyB  = scratchKey[partner];
						const uint indexA = scratchIndex[i];
						const uint indexB = scratchIndex[partner];
						const bool before = bps_before(keyA, indexA, keyB, indexB);
						if (before != stageAscending) {
							scratchKey[i]       = keyB;
							scratchKey[partner] = keyA;
							scratchIndex[i]       = indexB;
							scratchIndex[partner] = indexA;
						}
					}
				}
				threadgroup_barrier(mem_flags::mem_threadgroup);
			}
		}

		for (uint i = gtid; i < spanSize; i += MAX_THREADS) {
			const uint pos = spanStart + i;
			const uint shift = bps_cycle_shift(spanSize, p.cycleDegrees);
			const uint sortedIndex = (i + spanSize - shift) % spanSize;
			int x = 0;
			int y = 0;
			if (bps_coord_for_pos(p, pathSamples, gid, pos, &x, &y) &&
				bps_dst_in_world(p, x, y) && scratchIndex[sortedIndex] != 0xffffffffu) {
				sortTex[bps_dst_index_xy(p, x, y)] = srcTex[scratchIndex[sortedIndex]];
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		cursor = spanEnd + 1u;
	}

}

kernel void BitonicSortDomainKernel(
	device const float4       *srcTex  [[buffer(0)]],
	device const float4       *criterionTex [[buffer(1)]],
	device const float4       *triggerTex [[buffer(2)]],
	device uint               *domain  [[buffer(3)]],
	device float              *keys    [[buffer(4)]],
	constant BitonicSortParams &p      [[buffer(5)]],
	device const BpsPathSampleGpu *pathSamples [[buffer(6)]],
	uint gid  [[threadgroup_position_in_grid]],
	uint gtid [[thread_position_in_threadgroup]])
{
	threadgroup uint s_spanStart;
	threadgroup uint s_spanEnd;
	threadgroup uint s_spanSize;
	threadgroup uint s_sortSize;
	threadgroup uint s_pathLen;

	const uint size = bps_line_size(p, gid);
	if (size == 0u || p.domainStride <= 0 || (int)size > p.domainStride) {
		return;
	}

	const int stride = p.domainStride;

	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		int x = 0;
		int y = 0;
		const bool valid = bps_coord_for_pos(p, pathSamples, gid, pos, &x, &y) &&
			bps_src_in_world(p, x, y);
		domain[(uint)((int)gid * stride + (int)pos)] =
			valid ? bps_src_index_xy(p, x, y) : 0xffffffffu;
	}
	threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

	if (gtid == 0u) {
		uint pathLen = 0u;
		for (uint pos = 0u; pos < size; ++pos) {
			if (domain[(uint)((int)gid * stride + (int)pos)] != 0xffffffffu) {
				keys[(uint)((int)gid * stride + (int)pathLen)] = as_type<float>(pos);
				pathLen++;
			}
		}
		s_pathLen = pathLen;
	}
	threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

	const uint pathLen = s_pathLen;
	uint cursor = 0u;
	while (cursor < pathLen) {
		if (gtid == 0u) {
			uint runStart = cursor;
			while (runStart < pathLen) {
				const uint pos = as_type<uint>(keys[(uint)((int)gid * stride + (int)runStart)]);
				const uint srcIndex = domain[(uint)((int)gid * stride + (int)pos)];
				const float br = bps_sample_key_from_src_index(bps_trigger_src(triggerTex, p), p.srcPitch, p.inputOriginX, p.inputOriginY, srcIndex, p.trigger);
				if (bps_is_affected(br, p.thresholdMin, p.thresholdMax, p.affect)) break;
				runStart++;
			}
			uint runEnd = runStart;
			while (runEnd < pathLen) {
				const uint pos = as_type<uint>(keys[(uint)((int)gid * stride + (int)runEnd)]);
				const uint srcIndex = domain[(uint)((int)gid * stride + (int)pos)];
				const float br = bps_sample_key_from_src_index(bps_trigger_src(triggerTex, p), p.srcPitch, p.inputOriginX, p.inputOriginY, srcIndex, p.trigger);
				if (!bps_is_affected(br, p.thresholdMin, p.thresholdMax, p.affect)) break;
				runEnd++;
			}
			s_spanStart = runStart;
			s_spanEnd = runEnd;
			s_spanSize = runEnd - runStart;
			s_sortSize = bps_next_pow2(s_spanSize);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		const uint runStart = s_spanStart;
		const uint runEnd = s_spanEnd;
		const uint spanSize = s_spanSize;
		const uint sortSize = s_sortSize;

		if (runStart >= pathLen || spanSize == 0u) {
			break;
		}

		const bool ascending = p.ordering != 0;
		for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
			const uint work = (uint)((int)gid * stride + (int)(size + i));
			if (i < spanSize) {
				const uint pos = as_type<uint>(
					keys[(uint)((int)gid * stride + (int)(runStart + i))]);
				const uint srcIndex = domain[(uint)((int)gid * stride + (int)pos)];
				domain[work] = srcIndex;
				keys[work] = bps_sample_key_from_src_index(bps_criterion_src(criterionTex, p), p.srcPitch, p.inputOriginX, p.inputOriginY, srcIndex, p.criterion);
			} else {
				domain[work] = 0xffffffffu;
				keys[work] = ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX;
			}
		}
		threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

		for (uint k = 2u; k <= sortSize; k <<= 1) {
			for (uint j = k >> 1; j > 0u; j >>= 1) {
				for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
					const uint partner = i ^ j;
					if (partner > i) {
						bool stageAscending = (i & k) == 0u;
						if (!ascending) stageAscending = !stageAscending;

						const uint slotA = (uint)((int)gid * stride + (int)(size + i));
						const uint slotB = (uint)((int)gid * stride + (int)(size + partner));
						const float keyA = keys[slotA];
						const float keyB = keys[slotB];
						const uint indexA = domain[slotA];
						const uint indexB = domain[slotB];
						const bool before = bps_before(keyA, indexA, keyB, indexB);
						if (before != stageAscending) {
							keys[slotA] = keyB;
							keys[slotB] = keyA;
							domain[slotA] = indexB;
							domain[slotB] = indexA;
						}
					}
				}
				threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
			}
		}

		const uint shift = bps_cycle_shift(spanSize, p.cycleDegrees);
		// Flag reordered slots (run length >= 2) so ApplyDomain composites only
		// those over the source and leaves unsorted areas pristine.
		const uint affectedBit = (spanSize >= 2u) ? 0x80000000u : 0u;
		for (uint i = gtid; i < spanSize; i += MAX_THREADS) {
			const uint sortedIndex = (i + spanSize - shift) % spanSize;
			const uint pos = as_type<uint>(
				keys[(uint)((int)gid * stride + (int)(runStart + i))]);
			const uint srcIndex =
				domain[(uint)((int)gid * stride + (int)(size + sortedIndex))];
			domain[(uint)((int)gid * stride + (int)pos)] = srcIndex | affectedBit;
		}
		threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

		cursor = runEnd + 1u;
	}
}

kernel void BitonicApplyDomainKernel(
	device const float4       *srcTex  [[buffer(0)]],
	device float4             *sortTex [[buffer(1)]],
	device const uint         *domain  [[buffer(2)]],
	constant BitonicSortParams &p      [[buffer(3)]],
	device const BpsPathSampleGpu *pathSamples [[buffer(4)]],
	uint2 gid [[thread_position_in_grid]])
{
	const int ox = (int)gid.x;
	const int oy = (int)gid.y;
	if (ox >= p.outputWidth || oy >= p.outputHeight) {
		return;
	}

	const int x = p.outputOriginX + ox;
	const int y = p.outputOriginY + oy;
	const uint dstIndex = (uint)(ox + oy * p.dstPitch);

	float4 pixel = float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (bps_src_in_world(p, x, y)) {
		pixel = srcTex[bps_src_index_xy(p, x, y)];
	}

	int line = 0;
	int pos = 0;
	if (bps_domain_pos_for_pixel(p, pathSamples, x, y, &line, &pos)) {
		const uint raw = domain[(uint)(line * p.domainStride + pos)];
		// High bit flags slots a sort reordered; composite only those over the
		// source so unsorted areas keep the exact original (no resample loss).
		if (raw != 0xffffffffu && (raw & 0x80000000u) != 0u) {
			pixel = srcTex[raw & 0x7fffffffu];
		}
	}

	sortTex[dstIndex] = pixel;
}

kernel void BitonicCopyInputKernel(
	device const float4       *srcTex  [[buffer(0)]],
	device float4             *sortTex [[buffer(1)]],
	constant BitonicSortParams &p      [[buffer(2)]],
	uint2 gid [[thread_position_in_grid]])
{
	const int ox = (int)gid.x;
	const int oy = (int)gid.y;
	if (ox >= p.outputWidth || oy >= p.outputHeight) {
		return;
	}

	const int x = p.outputOriginX + ox;
	const int y = p.outputOriginY + oy;
	const uint dstIndex = (uint)(ox + oy * p.dstPitch);
	float4 pixel = float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (bps_src_in_world(p, x, y)) {
		pixel = srcTex[bps_src_index_xy(p, x, y)];
	}
	sortTex[dstIndex] = pixel;
}

struct BpsJfaCellGpu {
	float qx;
	float qy;
	float s;
	float tx;
	float ty;
	float d2;
};

kernel void BitonicBuildPathClassifyCountKernel(
	device const BpsJfaCellGpu *jfaField [[buffer(0)]],
	device int               *laneOf     [[buffer(1)]],
	device float             *keyOf      [[buffer(2)]],
	device atomic_uint       *laneCounts [[buffer(3)]],
	constant BitonicSortParams &p        [[buffer(4)]],
	uint2 gid [[thread_position_in_grid]])
{
	const int x = (int)gid.x;
	const int y = (int)gid.y;
	if (x >= p.width || y >= p.height) {
		return;
	}

	const int gridW = p.freePMin;
	const int gridH = p.freeQMin;
	const int jfaFactor = p.swirlLineMin;

	const uint pidx = (uint)(y * p.width + x);
	laneOf[pidx] = -1;

	const int cx = min(x / jfaFactor, gridW - 1);
	const int cy = min(y / jfaFactor, gridH - 1);
	const BpsJfaCellGpu cell = jfaField[(uint)(cy * gridW + cx)];
	if (!isfinite(cell.d2)) {
		return;
	}

	const float dxp = (float)x - cell.qx;
	const float dyp = (float)y - cell.qy;
	const float n = -cell.ty * dxp + cell.tx * dyp;
	int lane = 0;
	float order = 0.0f;
	if (!bps_path_lane_order(p, cell.s, n, &lane, &order)) {
		return;
	}
	laneOf[pidx] = lane;
	keyOf[pidx] = order;
	atomic_fetch_add_explicit(&laneCounts[lane], 1u, memory_order_relaxed);
}

kernel void BitonicBuildPathScatterRecordsKernel(
	device const int         *laneOf      [[buffer(0)]],
	device const float       *keyOf       [[buffer(1)]],
	device atomic_uint       *laneCursors [[buffer(2)]],
	device BpsMappedPixelRecordGpu *records [[buffer(3)]],
	constant BitonicSortParams &p         [[buffer(4)]],
	uint2 gid [[thread_position_in_grid]])
{
	const int x = (int)gid.x;
	const int y = (int)gid.y;
	if (x >= p.width || y >= p.height) {
		return;
	}

	const uint pidx = (uint)(y * p.width + x);
	const int lane = laneOf[pidx];
	if (lane < 0) {
		return;
	}
	const uint dst = atomic_fetch_add_explicit(
		&laneCursors[lane], 1u, memory_order_relaxed);
	records[dst].posKey = keyOf[pidx];
	records[dst].pixelIndex = pidx;
}

kernel void BitonicBuildPathSortRecordsKernel(
	device BpsMappedPixelRecordGpu *records [[buffer(0)]],
	device BpsMappedPixelRecordGpu *workRecords [[buffer(1)]],
	device const uint         *lineOffsets [[buffer(2)]],
	device const uint         *workOffsets [[buffer(3)]],
	constant BitonicSortParams &p          [[buffer(4)]],
	uint mapLine [[threadgroup_position_in_grid]],
	uint gtid [[thread_position_in_threadgroup]])
{
	if ((int)mapLine >= p.lineCount) {
		return;
	}
	const uint begin = lineOffsets[mapLine];
	const uint end = lineOffsets[mapLine + 1u];
	const uint lineSize = end - begin;
	if (lineSize <= 1u) {
		return;
	}

	const uint sortSize = bps_next_pow2(lineSize);
	const uint sortBase = workOffsets[mapLine] + lineSize;
	for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
		if (i < lineSize) {
			workRecords[sortBase + i] = records[begin + i];
		} else {
			workRecords[sortBase + i].posKey = BPS_FLOAT_MAX;
			workRecords[sortBase + i].pixelIndex = 0xffffffffu;
		}
	}
	threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

	for (uint k = 2u; k <= sortSize; k <<= 1) {
		for (uint j = k >> 1; j > 0u; j >>= 1) {
			for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
				const uint partner = i ^ j;
				if (partner > i) {
					const uint slotA = sortBase + i;
					const uint slotB = sortBase + partner;
					const BpsMappedPixelRecordGpu a = workRecords[slotA];
					const BpsMappedPixelRecordGpu b = workRecords[slotB];
					const bool stageAscending = (i & k) == 0u;
					const bool before = bps_record_before(a, b);
					if (before != stageAscending) {
						workRecords[slotA] = b;
						workRecords[slotB] = a;
					}
				}
			}
			threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
		}
	}

	for (uint i = gtid; i < lineSize; i += MAX_THREADS) {
		records[begin + i] = workRecords[sortBase + i];
	}
}

kernel void BitonicSortMappedKernel(
	device const float4       *srcTex  [[buffer(0)]],
	device float4             *sortTex [[buffer(1)]],
	device const float4       *criterionTex [[buffer(2)]],
	device const float4       *triggerTex [[buffer(3)]],
	device uint               *domain  [[buffer(4)]],
	device float              *keys    [[buffer(5)]],
	constant BitonicSortParams &p      [[buffer(6)]],
	device const BpsMappedPixelRecordGpu *records [[buffer(7)]],
	device const uint         *lineOffsets [[buffer(8)]],
	device const uint         *workOffsets [[buffer(9)]],
	uint mappedLine [[threadgroup_position_in_grid]],
	uint gtid [[thread_position_in_threadgroup]])
{
	threadgroup uint s_spanStart;
	threadgroup uint s_spanEnd;
	threadgroup uint s_spanSize;
	threadgroup uint s_sortSize;

	if ((int)mappedLine >= p.lineCount) {
		return;
	}

	const uint begin = lineOffsets[mappedLine];
	const uint end = lineOffsets[mappedLine + 1u];
	const uint lineSize = end - begin;
	if (lineSize == 0u) {
		return;
	}

	const uint workBase = workOffsets[mappedLine];
	const bool ascending = p.ordering != 0;

	for (uint i = gtid; i < lineSize; i += MAX_THREADS) {
		const uint pixelIndex = records[begin + i].pixelIndex;
		const int x = (int)(pixelIndex % (uint)p.width);
		const int y = (int)(pixelIndex / (uint)p.width);
		domain[workBase + i] = bps_src_in_world(p, x, y)
			? bps_src_index_xy(p, x, y)
			: 0xffffffffu;
	}
	threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

	uint cursor = 0u;
	while (cursor < lineSize) {
		if (gtid == 0u) {
			uint runStart = cursor;
			while (runStart < lineSize) {
				const uint srcIndex = domain[workBase + runStart];
				if (srcIndex != 0xffffffffu) {
					const float br = bps_sample_key_from_src_index(bps_trigger_src(triggerTex, p), p.srcPitch, p.inputOriginX, p.inputOriginY, srcIndex, p.trigger);
					if (bps_is_affected(br, p.thresholdMin, p.thresholdMax, p.affect)) break;
				}
				runStart++;
			}
			uint runEnd = runStart;
			while (runEnd < lineSize) {
				const uint srcIndex = domain[workBase + runEnd];
				if (srcIndex == 0xffffffffu) break;
				const float br = bps_sample_key_from_src_index(bps_trigger_src(triggerTex, p), p.srcPitch, p.inputOriginX, p.inputOriginY, srcIndex, p.trigger);
				if (!bps_is_affected(br, p.thresholdMin, p.thresholdMax, p.affect)) break;
				runEnd++;
			}
			s_spanStart = runStart;
			s_spanEnd = runEnd;
			s_spanSize = runEnd - runStart;
			s_sortSize = bps_next_pow2(s_spanSize);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		const uint runStart = s_spanStart;
		const uint runEnd = s_spanEnd;
		const uint spanSize = s_spanSize;
		const uint sortSize = s_sortSize;
		if (runStart >= lineSize || spanSize == 0u) {
			break;
		}

		const uint sortBase = workBase + lineSize;
		for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
			if (i < spanSize) {
				const uint srcIndex = domain[workBase + runStart + i];
				domain[sortBase + i] = srcIndex;
				keys[sortBase + i] = bps_sample_key_from_src_index(bps_criterion_src(criterionTex, p), p.srcPitch, p.inputOriginX, p.inputOriginY, srcIndex, p.criterion);
			} else {
				domain[sortBase + i] = 0xffffffffu;
				keys[sortBase + i] = ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX;
			}
		}
		threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

		for (uint k = 2u; k <= sortSize; k <<= 1) {
			for (uint j = k >> 1; j > 0u; j >>= 1) {
				for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
					const uint partner = i ^ j;
					if (partner > i) {
						bool stageAscending = (i & k) == 0u;
						if (!ascending) stageAscending = !stageAscending;
						const float keyA = keys[sortBase + i];
						const float keyB = keys[sortBase + partner];
						const uint indexA = domain[sortBase + i];
						const uint indexB = domain[sortBase + partner];
						const bool before = bps_before(keyA, indexA, keyB, indexB);
						if (before != stageAscending) {
							keys[sortBase + i] = keyB;
							keys[sortBase + partner] = keyA;
							domain[sortBase + i] = indexB;
							domain[sortBase + partner] = indexA;
						}
					}
				}
				threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
			}
		}

		const uint shift = bps_cycle_shift(spanSize, p.cycleDegrees);
		for (uint i = gtid; i < spanSize; i += MAX_THREADS) {
			const uint sortedIndex = (i + spanSize - shift) % spanSize;
			const uint srcIndex = domain[sortBase + sortedIndex];
			const uint pixelIndex = records[begin + runStart + i].pixelIndex;
			const int x = (int)(pixelIndex % (uint)p.width);
			const int y = (int)(pixelIndex / (uint)p.width);
			if (srcIndex != 0xffffffffu && bps_dst_in_world(p, x, y)) {
				sortTex[bps_dst_index_xy(p, x, y)] = srcTex[srcIndex];
			}
		}
		threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

		cursor = runEnd + 1u;
	}
}

inline float bps_luma(float4 c)
{
	return clamp(0.298912f * c.z + 0.586611f * c.y + 0.114478f * c.x, 0.0f, 1.0f);
}

inline uint bps_axis_src_index_pos(constant BitonicSortParams &p, uint gid, uint pos)
{
	const int x = p.direction ? (int)pos : (p.inputOriginX + (int)gid);
	const int y = p.direction ? (p.inputOriginY + (int)gid) : (int)pos;
	return bps_src_index_xy(p, x, y);
}

inline uint bps_axis_dst_index_pos(constant BitonicSortParams &p, uint gid, uint pos)
{
	const int x = p.direction ? (int)pos : (p.outputOriginX + (int)gid);
	const int y = p.direction ? (p.outputOriginY + (int)gid) : (int)pos;
	return bps_dst_index_xy(p, x, y);
}

kernel void BitonicSortAxisLumaKernel(
	device const float4       *srcTex  [[buffer(0)]],
	device float4             *sortTex [[buffer(1)]],
	constant BitonicSortParams &p      [[buffer(2)]],
	uint gid  [[threadgroup_position_in_grid]],
	uint gtid [[thread_position_in_threadgroup]])
{
	threadgroup float scratchKey[MAX_SIZE];
	threadgroup uint  scratchIndex[MAX_SIZE];

	const uint size = bps_line_size(p, gid);
	if (size == 0u || size > MAX_SIZE) {
		return;
	}

	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		int x = 0;
		int y = 0;
		if (bps_coord_for_pos(p, nullptr, gid, pos, &x, &y) &&
			bps_dst_in_world(p, x, y)) {
			sortTex[bps_dst_index_xy(p, x, y)] = bps_src_in_world(p, x, y)
				? srcTex[bps_src_index_xy(p, x, y)]
				: float4(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	uint cursor = 0u;
	while (cursor < size) {
		if (gtid == 0u) {
			uint spanStart = cursor;
			while (spanStart < size) {
				int x = 0;
				int y = 0;
				if (bps_coord_for_pos(p, nullptr, gid, spanStart, &x, &y) &&
					bps_src_in_world(p, x, y)) {
					const float br = bps_luma(srcTex[bps_src_index_xy(p, x, y)]);
					if (br >= p.thresholdMin && br <= p.thresholdMax) {
						break;
					}
				}
				spanStart++;
			}

			uint spanEnd = spanStart;
			while (spanEnd < size) {
				int x = 0;
				int y = 0;
				if (!bps_coord_for_pos(p, nullptr, gid, spanEnd, &x, &y) ||
					!bps_src_in_world(p, x, y)) {
					break;
				}
				const float br = bps_luma(srcTex[bps_src_index_xy(p, x, y)]);
				if (br < p.thresholdMin || br > p.thresholdMax) {
					break;
				}
				spanEnd++;
			}

			scratchIndex[0] = spanStart;
			scratchIndex[1] = spanEnd;
			scratchIndex[2] = spanEnd - spanStart;
			scratchIndex[3] = bps_next_pow2(scratchIndex[2]);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		const uint spanStart = scratchIndex[0];
		const uint spanEnd = scratchIndex[1];
		const uint spanSize = scratchIndex[2];
		const uint sortSize = scratchIndex[3];
		threadgroup_barrier(mem_flags::mem_threadgroup);

		if (spanStart >= size || spanSize == 0u) {
			break;
		}

		const bool ascending = p.ordering != 0;
		for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
			if (i < spanSize) {
				const uint pos = spanStart + i;
				int x = 0;
				int y = 0;
				const bool valid = bps_coord_for_pos(p, nullptr, gid, pos, &x, &y) &&
					bps_src_in_world(p, x, y);
				const uint srcIndex = valid ? bps_src_index_xy(p, x, y) : 0xffffffffu;
				scratchKey[i] = valid
					? bps_luma(srcTex[srcIndex])
					: (ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX);
				scratchIndex[i] = srcIndex;
			} else {
				scratchKey[i] = ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX;
				scratchIndex[i] = 0xffffffffu;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		for (uint k = 2u; k <= sortSize; k <<= 1) {
			for (uint j = k >> 1; j > 0u; j >>= 1) {
				for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
					const uint partner = i ^ j;
					if (partner > i) {
						bool stageAscending = (i & k) == 0u;
						if (!ascending) {
							stageAscending = !stageAscending;
						}
						const float keyA = scratchKey[i];
						const float keyB = scratchKey[partner];
						const uint indexA = scratchIndex[i];
						const uint indexB = scratchIndex[partner];
						const bool before = bps_before(keyA, indexA, keyB, indexB);
						if (before != stageAscending) {
							scratchKey[i] = keyB;
							scratchKey[partner] = keyA;
							scratchIndex[i] = indexB;
							scratchIndex[partner] = indexA;
						}
					}
				}
				threadgroup_barrier(mem_flags::mem_threadgroup);
			}
		}

		for (uint i = gtid; i < spanSize; i += MAX_THREADS) {
			const uint pos = spanStart + i;
			int x = 0;
			int y = 0;
			if (bps_coord_for_pos(p, nullptr, gid, pos, &x, &y) &&
				bps_dst_in_world(p, x, y) && scratchIndex[i] != 0xffffffffu) {
				sortTex[bps_dst_index_xy(p, x, y)] = srcTex[scratchIndex[i]];
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		cursor = spanEnd + 1u;
	}
}

kernel void BitonicSortAxisLumaFullKernel(
	device const float4       *srcTex  [[buffer(0)]],
	device float4             *sortTex [[buffer(1)]],
	constant BitonicSortParams &p      [[buffer(2)]],
	uint gid  [[threadgroup_position_in_grid]],
	uint gtid [[thread_position_in_threadgroup]])
{
	threadgroup float scratchKey[MAX_SIZE];
	threadgroup uint  scratchIndex[MAX_SIZE];

	const uint size = bps_line_size(p, gid);
	if (size == 0u || size > MAX_SIZE) {
		return;
	}

	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		sortTex[bps_axis_dst_index_pos(p, gid, pos)] =
			srcTex[bps_axis_src_index_pos(p, gid, pos)];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	uint cursor = 0u;
	while (cursor < size) {
		if (gtid == 0u) {
			uint spanStart = cursor;
			while (spanStart < size) {
				const float br = bps_luma(srcTex[bps_axis_src_index_pos(p, gid, spanStart)]);
				if (br >= p.thresholdMin && br <= p.thresholdMax) {
					break;
				}
				spanStart++;
			}

			uint spanEnd = spanStart;
			while (spanEnd < size) {
				const float br = bps_luma(srcTex[bps_axis_src_index_pos(p, gid, spanEnd)]);
				if (br < p.thresholdMin || br > p.thresholdMax) {
					break;
				}
				spanEnd++;
			}

			scratchIndex[0] = spanStart;
			scratchIndex[1] = spanEnd;
			scratchIndex[2] = spanEnd - spanStart;
			scratchIndex[3] = bps_next_pow2(scratchIndex[2]);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		const uint spanStart = scratchIndex[0];
		const uint spanEnd = scratchIndex[1];
		const uint spanSize = scratchIndex[2];
		const uint sortSize = scratchIndex[3];
		threadgroup_barrier(mem_flags::mem_threadgroup);

		if (spanStart >= size || spanSize == 0u) {
			break;
		}

		const bool ascending = p.ordering != 0;
		for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
			if (i < spanSize) {
				const uint pos = spanStart + i;
				const uint srcIndex = bps_axis_src_index_pos(p, gid, pos);
				scratchKey[i] = bps_luma(srcTex[srcIndex]);
				scratchIndex[i] = srcIndex;
			} else {
				scratchKey[i] = ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX;
				scratchIndex[i] = 0xffffffffu;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		for (uint k = 2u; k <= sortSize; k <<= 1) {
			for (uint j = k >> 1; j > 0u; j >>= 1) {
				for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
					const uint partner = i ^ j;
					if (partner > i) {
						bool stageAscending = (i & k) == 0u;
						if (!ascending) {
							stageAscending = !stageAscending;
						}
						const float keyA = scratchKey[i];
						const float keyB = scratchKey[partner];
						const uint indexA = scratchIndex[i];
						const uint indexB = scratchIndex[partner];
						const bool before = bps_before(keyA, indexA, keyB, indexB);
						if (before != stageAscending) {
							scratchKey[i] = keyB;
							scratchKey[partner] = keyA;
							scratchIndex[i] = indexB;
							scratchIndex[partner] = indexA;
						}
					}
				}
				threadgroup_barrier(mem_flags::mem_threadgroup);
			}
		}

		for (uint i = gtid; i < spanSize; i += MAX_THREADS) {
			const uint pos = spanStart + i;
			const uint srcIndex = scratchIndex[i];
			if (srcIndex != 0xffffffffu) {
				sortTex[bps_axis_dst_index_pos(p, gid, pos)] = srcTex[srcIndex];
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		cursor = spanEnd + 1u;
	}
}
