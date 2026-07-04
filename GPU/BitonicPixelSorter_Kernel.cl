/*
	BitonicPixelSorter_Kernel.cl

	OpenCL port of the upstream single-dispatch bitonic pixel sort.
	Axis mode: one work-group sorts one line and gathers into the destination.
	Non-axis modes use exact multi-pass (sort paths into a domain of source
	indices, then inverse-map each output pixel). AE GPU worlds are BGRA float4
	buffers, with row pitches expressed in float4 units by the host code.
*/

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

#define BPS_CRITERION_LUMINANCE 1
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
		return clamp(fmin(fmin(c.z, c.y), c.x), 0.0f, 1.0f);
	}
	if (criterion == BPS_CRITERION_RGB_MAXIMUM) {
		return clamp(fmax(fmax(c.z, c.y), c.x), 0.0f, 1.0f);
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
		const float maxRGB = fmax(fmax(r, g), b);
		const float minRGB = fmin(fmin(r, g), b);
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

typedef struct BpsKeySource {
	__global const float4 *tex;
	int pitch;
	int originX;
	int originY;
	int width;
	int height;
} BpsKeySource;

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

inline int bps_round_to_int(float value)
{
	return (int)floor(value + 0.5f);
}

inline uint bps_rotation_line_length(uint radius)
{
	if (radius == 0u) return 1u;
	uint length = (uint)ceil(BPS_TWO_PI * (float)radius);
	return length == 0u ? 1u : length;
}

typedef struct {
	float x;
	float y;
	float s;
	float tx;
	float ty;
} BpsPathSampleGpu;

typedef struct {
	float posKey;
	uint pixelIndex;
} BpsMappedPixelRecordGpu;

inline void bps_normalise_tangent(float *tx, float *ty)
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

inline bool bps_path_eval_at_s(__global const BpsPathSampleGpu *samples, int sampleCount,
	float s, float *x, float *y, float *tx, float *ty)
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

inline bool bps_path_closest(__global const BpsPathSampleGpu *samples, int sampleCount,
	float px, float py, float *sOut, float *nOut)
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
			t = ((px - a.x) * abx + (py - a.y) * aby) / abLen2;
			t = clamp(t, 0.0f, 1.0f);
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

inline bool bps_path_lane_order(int pathDirection, int pathClosed, float pathLength,
	int pathSMin, int pathNMin, int lineCount,
	float s, float n, int *line, float *order)
{
	const bool closed = (pathClosed != 0) && pathLength > 1.0e-6f;
	const float sLocal = closed ? bps_wrap_arc_length(s, pathLength) : s;
	int lane = 0;
	float posKey = 0.0f;
	if (pathDirection == BPS_PATH_DIR_TANGENT) {
		lane = bps_round_to_int(n) - pathNMin;
		posKey = sLocal;
	} else {
		if (closed) {
			const int bins = lineCount > 0 ? lineCount : 1;
			int q = bps_round_to_int(sLocal) % bins;
			if (q < 0) q += bins;
			lane = q;
		} else {
			lane = bps_round_to_int(s) - pathSMin;
		}
		posKey = n;
	}
	if (lane < 0 || lane >= lineCount) {
		return false;
	}
	*line = lane;
	*order = posKey;
	return true;
}

inline uint bps_line_size(int mode, int direction, uint gid,
						  int width, int height,
						  int freeLineLength, int radialLength, float swirlK)
{
	if (mode == BPS_MODE_FREE_ANGLE) return (uint)freeLineLength;
	if (mode == BPS_MODE_ROTATION) return bps_rotation_line_length(gid);
	if (mode == BPS_MODE_RADIAL) return (uint)radialLength;
	if (mode == BPS_MODE_SWIRL) {
		return (uint)(radialLength > 0 ? radialLength : 1);
	}
	if (mode == BPS_MODE_PATH) return (uint)(radialLength > 0 ? radialLength : 1);
	return direction ? (uint)width : (uint)height;
}

inline bool bps_coord_for_pos(int mode, int direction, uint gid, uint pos,
							  int width, int height,
							  int outputOriginX, int outputOriginY,
							  int lineCount,
							  int freePMin, int freeQMin, int freeLineLength,
							  float angleCos, float angleSin,
							  float centerX, float centerY,
							  float swirlK, int swirlLineMin,
							  int pathDirection, int pathSMin, int pathNMin,
							  int pathSampleCount,
							  __global const BpsPathSampleGpu *pathSamples,
							  int *x, int *y)
{
	if (mode == BPS_MODE_FREE_ANGLE) {
		const float p = (float)(freePMin + (int)pos);
		const float q = (float)(freeQMin + (int)gid);
		*x = bps_round_to_int(p * angleCos - q * angleSin);
		*y = bps_round_to_int(p * angleSin + q * angleCos);
		(void)freeLineLength;
	} else if (mode == BPS_MODE_ROTATION) {
		const uint lineLen = bps_rotation_line_length(gid);
		const float theta = (lineLen <= 1u) ? 0.0f :
			(BPS_TWO_PI * (float)pos) / (float)lineLen;
		const float c = cos(theta);
		const float s = sin(theta);
		const float radius = (float)gid;
		*x = bps_round_to_int(centerX + radius * (s * angleCos + c * angleSin));
		*y = bps_round_to_int(centerY + radius * (s * angleSin - c * angleCos));
	} else if (mode == BPS_MODE_RADIAL) {
		const float denom = lineCount <= 0 ? 1.0f : (float)lineCount;
		const float theta = (BPS_TWO_PI * (float)gid) / denom;
		*x = bps_round_to_int(centerX + (float)pos * cos(theta));
		*y = bps_round_to_int(centerY + (float)pos * sin(theta));
	} else if (mode == BPS_MODE_SWIRL) {
		const float r = (float)pos;
		const float denom = lineCount <= 0 ? 1.0f : (float)lineCount;
		const float phase = (BPS_TWO_PI * (float)gid) / denom;
		const float angle = atan2(angleSin, angleCos);
		const float theta = phase + swirlK * r + angle;
		*x = bps_round_to_int(centerX + r * cos(theta));
		*y = bps_round_to_int(centerY + r * sin(theta));
		(void)swirlLineMin;
	} else if (mode == BPS_MODE_PATH) {
		float s = 0.0f;
		float n = 0.0f;
		if (pathDirection == BPS_PATH_DIR_NORMAL) {
			s = (float)(pathSMin + (int)gid);
			n = (float)(pathNMin + (int)pos);
		} else {
			n = (float)(pathNMin + (int)gid);
			s = (float)(pathSMin + (int)pos);
		}
		float px = 0.0f, py = 0.0f, tx = 1.0f, ty = 0.0f;
		if (!bps_path_eval_at_s(pathSamples, pathSampleCount, s, &px, &py, &tx, &ty)) {
			return false;
		}
		*x = bps_round_to_int(px + n * (-ty));
		*y = bps_round_to_int(py + n * tx);
	} else {
		const int lineLayer = direction ? (outputOriginY + (int)gid) : (outputOriginX + (int)gid);
		*x = direction ? (int)pos : lineLayer;
		*y = direction ? lineLayer : (int)pos;
	}
	return *x >= 0 && *y >= 0 && *x < width && *y < height;
}

inline bool bps_domain_pos_for_pixel(int mode, int x, int y,
									 int lineCount,
									 int freePMin, int freeQMin, int freeLineLength,
									 int radialLength, int domainStride,
									 float angleCos, float angleSin,
									 float centerX, float centerY,
									 float swirlK, int swirlLineMin,
									 int pathDirection, int pathSMin, int pathNMin,
									 int pathSampleCount,
									 __global const BpsPathSampleGpu *pathSamples,
									 int *lineP, int *posP)
{
	int line = 0;
	int pos = 0;

	if (mode == BPS_MODE_FREE_ANGLE) {
		const float p = (float)x * angleCos + (float)y * angleSin;
		const float q = -(float)x * angleSin + (float)y * angleCos;
		pos = bps_round_to_int(p) - freePMin;
		line = bps_round_to_int(q) - freeQMin;
	} else if (mode == BPS_MODE_ROTATION) {
		const float dx = (float)x - centerX;
		const float dy = (float)y - centerY;
		line = bps_round_to_int(sqrt(dx * dx + dy * dy));
		const uint lineLen = bps_rotation_line_length((uint)(line < 0 ? 0 : line));
		if (lineLen <= 1u) {
			pos = 0;
		} else {
			const float rx = dx * angleCos + dy * angleSin;
			const float ry = -dx * angleSin + dy * angleCos;
			float theta = atan2(rx, -ry);
			if (theta < 0.0f) theta += BPS_TWO_PI;
			pos = bps_round_to_int((theta / BPS_TWO_PI) * (float)lineLen);
			if (pos >= (int)lineLen) pos -= (int)lineLen;
		}
	} else if (mode == BPS_MODE_RADIAL) {
		const float dx = (float)x - centerX;
		const float dy = (float)y - centerY;
		pos = bps_round_to_int(sqrt(dx * dx + dy * dy));
		float theta = atan2(dy, dx);
		if (theta < 0.0f) theta += BPS_TWO_PI;
		line = (lineCount <= 1) ? 0 :
			bps_round_to_int((theta / BPS_TWO_PI) * (float)lineCount);
		if (line >= lineCount) line -= lineCount;
	} else if (mode == BPS_MODE_SWIRL) {
		const float dx = (float)x - centerX;
		const float dy = (float)y - centerY;
		const float r = sqrt(dx * dx + dy * dy);
		const float angle = atan2(angleSin, angleCos);
		float phase = atan2(dy, dx) - angle - swirlK * r;
		phase = fmod(phase, BPS_TWO_PI);
		if (phase < 0.0f) phase += BPS_TWO_PI;
		line = (lineCount <= 1) ? 0 :
			bps_round_to_int((phase / BPS_TWO_PI) * (float)lineCount);
		if (line >= lineCount) line -= lineCount;
		pos = bps_round_to_int(r);
		(void)swirlLineMin;
	} else if (mode == BPS_MODE_PATH) {
		float s = 0.0f;
		float n = 0.0f;
		if (!bps_path_closest(pathSamples, pathSampleCount, (float)x, (float)y, &s, &n)) {
			return false;
		}
		if (pathDirection == BPS_PATH_DIR_NORMAL) {
			line = bps_round_to_int(s) - pathSMin;
			pos = bps_round_to_int(n) - pathNMin;
		} else {
			line = bps_round_to_int(n) - pathNMin;
			pos = bps_round_to_int(s) - pathSMin;
		}
	} else {
		return false;
	}

	if (line < 0 || line >= lineCount) return false;

	int lineLen = 0;
	if (mode == BPS_MODE_FREE_ANGLE) lineLen = freeLineLength;
	else if (mode == BPS_MODE_ROTATION) lineLen = (int)bps_rotation_line_length((uint)line);
	else lineLen = radialLength;

	if (pos < 0 || pos >= lineLen || pos >= domainStride) return false;

	*lineP = line;
	*posP = pos;
	return true;
}

__kernel void BitonicSortKernel(
	__global const float4 *srcTex,
	__global float4       *sortTex,
	__global const float4 *criterionTex,
	__global const float4 *triggerTex,
	int                    criterionPitch,
	int                    criterionOriginX,
	int                    criterionOriginY,
	int                    criterionWidth,
	int                    criterionHeight,
	int                    triggerPitch,
	int                    triggerOriginX,
	int                    triggerOriginY,
	int                    triggerWidth,
	int                    triggerHeight,
	int                    srcPitch,
	int                    dstPitch,
	int                    width,
	int                    height,
	int                    inputOriginX,
	int                    inputOriginY,
	int                    inputWidth,
	int                    inputHeight,
	int                    outputOriginX,
	int                    outputOriginY,
	int                    outputWidth,
	int                    outputHeight,
	int                    mode,
	int                    direction,
	int                    ordering,
	int                    criterion,
	int                    trigger,
	int                    affect,
	float                  cycleDegrees,
	int                    lineCount,
	int                    freePMin,
	int                    freeQMin,
	int                    freeLineLength,
	int                    radialLength,
	float                  thresholdMin,
	float                  thresholdMax,
	float                  angleCos,
	float                  angleSin,
	float                  centerX,
	float                  centerY,
	float                  swirlK,
	int                    swirlLineMin,
	int                    pathDirection,
	int                    pathSMin,
	int                    pathNMin,
	int                    pathSampleCount,
	__global const BpsPathSampleGpu *pathSamples)
{

	BpsKeySource criterionSrc;
	criterionSrc.tex = criterionTex;
	criterionSrc.pitch = criterionPitch;
	criterionSrc.originX = criterionOriginX;
	criterionSrc.originY = criterionOriginY;
	criterionSrc.width = criterionWidth;
	criterionSrc.height = criterionHeight;
	BpsKeySource triggerSrc;
	triggerSrc.tex = triggerTex;
	triggerSrc.pitch = triggerPitch;
	triggerSrc.originX = triggerOriginX;
	triggerSrc.originY = triggerOriginY;
	triggerSrc.width = triggerWidth;
	triggerSrc.height = triggerHeight;
	// scratchKey + scratchIndex already use the full 32 KB local-memory budget
	// guaranteed by the OpenCL spec, so the span metadata reuses scratchIndex[0..3]
	// rather than dedicated locals. Local-id 0 publishes it; every work-item copies
	// it into private locals; a second barrier then guarantees those copies finish
	// before the load loop overwrites scratchIndex, removing the data race while
	// keeping spanSize/sortSize uniform (no divergent barriers).
	__local float scratchKey[MAX_SIZE];
	__local uint scratchIndex[MAX_SIZE];

	const uint gid = get_group_id(0);
	const uint gtid = get_local_id(0);
	const uint size = bps_line_size(mode, direction, gid, width, height,
									freeLineLength, radialLength, swirlK);
	if (size == 0u || size > MAX_SIZE) {
		return;
	}

	#define BPS_SRC_IN_WORLD(x, y) ((x) >= inputOriginX && (y) >= inputOriginY && \
									(x) < inputOriginX + inputWidth && (y) < inputOriginY + inputHeight)
	#define BPS_DST_IN_WORLD(x, y) ((x) >= outputOriginX && (y) >= outputOriginY && \
									(x) < outputOriginX + outputWidth && (y) < outputOriginY + outputHeight)
	#define BPS_SRC_INDEX_XY(x, y) ((uint)(((x) - inputOriginX) + ((y) - inputOriginY) * srcPitch))
	#define BPS_DST_INDEX_XY(x, y) ((uint)(((x) - outputOriginX) + ((y) - outputOriginY) * dstPitch))

	// Always write every destination pixel. Partial input worlds (common with
	// alpha / adjustment layers) must not leave stale frame data behind.
	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		int x = 0, y = 0;
		if (bps_coord_for_pos(mode, direction, gid, pos, width, height,
							  outputOriginX, outputOriginY, lineCount,
							  freePMin, freeQMin, freeLineLength,
							  angleCos, angleSin, centerX, centerY,
							  swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
							  pathSampleCount, pathSamples, &x, &y) &&
			BPS_DST_IN_WORLD(x, y)) {
			sortTex[BPS_DST_INDEX_XY(x, y)] = BPS_SRC_IN_WORLD(x, y)
				? srcTex[BPS_SRC_INDEX_XY(x, y)]
				: (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	uint cursor = 0u;
	while (cursor < size) {
		if (gtid == 0u) {
			uint spanStart = cursor;
			while (spanStart < size) {
				int x = 0, y = 0;
				if (bps_coord_for_pos(mode, direction, gid, spanStart, width, height,
									  outputOriginX, outputOriginY, lineCount,
									  freePMin, freeQMin, freeLineLength,
									  angleCos, angleSin, centerX, centerY,
							 swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
							 pathSampleCount, pathSamples, &x, &y) &&
					BPS_SRC_IN_WORLD(x, y)) {
					float br = bps_sample_key(triggerSrc, x, y, trigger);
					if (bps_is_affected(br, thresholdMin, thresholdMax, affect)) break;
				}
				spanStart++;
			}

			uint spanEnd = spanStart;
			while (spanEnd < size) {
				int x = 0, y = 0;
				if (!bps_coord_for_pos(mode, direction, gid, spanEnd, width, height,
									   outputOriginX, outputOriginY, lineCount,
									   freePMin, freeQMin, freeLineLength,
									   angleCos, angleSin, centerX, centerY,
							 swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
							 pathSampleCount, pathSamples, &x, &y) ||
					!BPS_SRC_IN_WORLD(x, y)) break;
				float br = bps_sample_key(triggerSrc, x, y, trigger);
				if (!bps_is_affected(br, thresholdMin, thresholdMax, affect)) break;
				spanEnd++;
			}

			scratchIndex[0] = spanStart;
			scratchIndex[1] = spanEnd;
			scratchIndex[2] = spanEnd - spanStart;
			scratchIndex[3] = bps_next_pow2(scratchIndex[2]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		const uint spanStart = scratchIndex[0];
		const uint spanEnd = scratchIndex[1];
		const uint spanSize = scratchIndex[2];
		const uint sortSize = scratchIndex[3];

		// Publish/consume fence: every work-item has now copied the metadata into
		// private locals, so the load loop below may safely overwrite scratchIndex.
		barrier(CLK_LOCAL_MEM_FENCE);

		if (spanStart >= size || spanSize == 0u) {
			break;
		}

		const bool ascending = ordering != 0;
		for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
			if (i < spanSize) {
				const uint pos = spanStart + i;
				int x = 0, y = 0;
				const bool valid = bps_coord_for_pos(mode, direction, gid, pos, width, height,
													 outputOriginX, outputOriginY, lineCount,
													 freePMin, freeQMin, freeLineLength,
													 angleCos, angleSin, centerX, centerY,
							 swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
							 pathSampleCount, pathSamples, &x, &y) &&
								   BPS_SRC_IN_WORLD(x, y);
				const uint srcIndex = valid ? BPS_SRC_INDEX_XY(x, y) : 0xffffffffu;
				scratchKey[i] = valid ? bps_sample_key(criterionSrc, x, y, criterion) :
					(ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX);
				scratchIndex[i] = srcIndex;
			} else {
				scratchKey[i] = ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX;
				scratchIndex[i] = 0xffffffffu;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint k = 2u; k <= sortSize; k <<= 1) {
			for (uint j = k >> 1; j > 0u; j >>= 1) {
				for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
					const uint partner = i ^ j;
					if (partner > i) {
						bool stageAscending = (i & k) == 0u;
						if (!ascending) stageAscending = !stageAscending;

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
				barrier(CLK_LOCAL_MEM_FENCE);
			}
		}

		for (uint i = gtid; i < spanSize; i += MAX_THREADS) {
			const uint pos = spanStart + i;
			const uint shift = bps_cycle_shift(spanSize, cycleDegrees);
			const uint sortedIndex = (i + spanSize - shift) % spanSize;
			int x = 0, y = 0;
			if (bps_coord_for_pos(mode, direction, gid, pos, width, height,
								  outputOriginX, outputOriginY, lineCount,
								  freePMin, freeQMin, freeLineLength,
								  angleCos, angleSin, centerX, centerY,
							 swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
							 pathSampleCount, pathSamples, &x, &y) &&
				BPS_DST_IN_WORLD(x, y) && scratchIndex[sortedIndex] != 0xffffffffu) {
				sortTex[BPS_DST_INDEX_XY(x, y)] = srcTex[scratchIndex[sortedIndex]];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		cursor = spanEnd + 1u;
	}

	#undef BPS_SRC_IN_WORLD
	#undef BPS_DST_IN_WORLD
	#undef BPS_SRC_INDEX_XY
	#undef BPS_DST_INDEX_XY
}

__kernel void BitonicSortDomainKernel(
	__global const float4 *srcTex,
	__global const float4 *criterionTex,
	__global const float4 *triggerTex,
	int                    criterionPitch,
	int                    criterionOriginX,
	int                    criterionOriginY,
	int                    criterionWidth,
	int                    criterionHeight,
	int                    triggerPitch,
	int                    triggerOriginX,
	int                    triggerOriginY,
	int                    triggerWidth,
	int                    triggerHeight,
	__global uint         *domain,
	__global float        *keys,
	int                    srcPitch,
	int                    width,
	int                    height,
	int                    inputOriginX,
	int                    inputOriginY,
	int                    inputWidth,
	int                    inputHeight,
	int                    mode,
	int                    ordering,
	int                    criterion,
	int                    trigger,
	int                    affect,
	float                  cycleDegrees,
	int                    lineCount,
	int                    freePMin,
	int                    freeQMin,
	int                    freeLineLength,
	int                    radialLength,
	int                    domainStride,
	float                  thresholdMin,
	float                  thresholdMax,
	float                  angleCos,
	float                  angleSin,
	float                  centerX,
	float                  centerY,
	float                  swirlK,
	int                    swirlLineMin,
	int                    pathDirection,
	int                    pathSMin,
	int                    pathNMin,
	int                    pathSampleCount,
	__global const BpsPathSampleGpu *pathSamples)
{

	BpsKeySource criterionSrc;
	criterionSrc.tex = criterionTex;
	criterionSrc.pitch = criterionPitch;
	criterionSrc.originX = criterionOriginX;
	criterionSrc.originY = criterionOriginY;
	criterionSrc.width = criterionWidth;
	criterionSrc.height = criterionHeight;
	BpsKeySource triggerSrc;
	triggerSrc.tex = triggerTex;
	triggerSrc.pitch = triggerPitch;
	triggerSrc.originX = triggerOriginX;
	triggerSrc.originY = triggerOriginY;
	triggerSrc.width = triggerWidth;
	triggerSrc.height = triggerHeight;
	__local uint s_spanStart;
	__local uint s_spanEnd;
	__local uint s_spanSize;
	__local uint s_sortSize;
	__local uint s_pathLen;

	const uint gid = get_group_id(0);
	const uint gtid = get_local_id(0);
	const uint size = bps_line_size(mode, 0, gid, width, height,
									freeLineLength, radialLength, swirlK);
	if (size == 0u || domainStride <= 0 || (int)size > domainStride) {
		return;
	}

	#define BPS_SRC_IN_WORLD(x, y) ((x) >= inputOriginX && (y) >= inputOriginY && \
									(x) < inputOriginX + inputWidth && (y) < inputOriginY + inputHeight)
	#define BPS_SRC_INDEX_XY(x, y) ((uint)(((x) - inputOriginX) + ((y) - inputOriginY) * srcPitch))
	#define BPS_DOMAIN_INDEX(pos) ((uint)((int)gid * domainStride + (int)(pos)))

	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		int x = 0, y = 0;
		const bool valid = bps_coord_for_pos(mode, 0, gid, pos, width, height,
											 0, 0, lineCount,
											 freePMin, freeQMin, freeLineLength,
											 angleCos, angleSin, centerX, centerY,
							 swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
							 pathSampleCount, pathSamples, &x, &y) &&
						   BPS_SRC_IN_WORLD(x, y);
		domain[BPS_DOMAIN_INDEX(pos)] = valid ? BPS_SRC_INDEX_XY(x, y) : 0xffffffffu;
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if (gtid == 0u) {
		uint pathLen = 0u;
		for (uint pos = 0u; pos < size; ++pos) {
			if (domain[BPS_DOMAIN_INDEX(pos)] != 0xffffffffu) {
				keys[BPS_DOMAIN_INDEX(pathLen)] = as_float(pos);
				pathLen++;
			}
		}
		s_pathLen = pathLen;
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	const uint pathLen = s_pathLen;
	uint cursor = 0u;
	while (cursor < pathLen) {
		if (gtid == 0u) {
			uint runStart = cursor;
			while (runStart < pathLen) {
				const uint pos = as_uint(keys[BPS_DOMAIN_INDEX(runStart)]);
				const uint srcIndex = domain[BPS_DOMAIN_INDEX(pos)];
				const float br = bps_sample_key_from_src_index(triggerSrc, srcPitch, inputOriginX, inputOriginY, srcIndex, trigger);
				if (bps_is_affected(br, thresholdMin, thresholdMax, affect)) break;
				runStart++;
			}
			uint runEnd = runStart;
			while (runEnd < pathLen) {
				const uint pos = as_uint(keys[BPS_DOMAIN_INDEX(runEnd)]);
				const uint srcIndex = domain[BPS_DOMAIN_INDEX(pos)];
				const float br = bps_sample_key_from_src_index(triggerSrc, srcPitch, inputOriginX, inputOriginY, srcIndex, trigger);
				if (!bps_is_affected(br, thresholdMin, thresholdMax, affect)) break;
				runEnd++;
			}
			s_spanStart = runStart;
			s_spanEnd = runEnd;
			s_spanSize = runEnd - runStart;
			s_sortSize = bps_next_pow2(s_spanSize);
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		const uint runStart = s_spanStart;
		const uint runEnd = s_spanEnd;
		const uint spanSize = s_spanSize;
		const uint sortSize = s_sortSize;

		if (runStart >= pathLen || spanSize == 0u) {
			break;
		}

		const bool ascending = ordering != 0;
		for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
			const uint work = BPS_DOMAIN_INDEX(size + i);
			if (i < spanSize) {
				const uint pos = as_uint(keys[BPS_DOMAIN_INDEX(runStart + i)]);
				const uint srcIndex = domain[BPS_DOMAIN_INDEX(pos)];
				domain[work] = srcIndex;
				keys[work] = bps_sample_key_from_src_index(criterionSrc, srcPitch, inputOriginX, inputOriginY, srcIndex, criterion);
			} else {
				domain[work] = 0xffffffffu;
				keys[work] = ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		for (uint k = 2u; k <= sortSize; k <<= 1) {
			for (uint j = k >> 1; j > 0u; j >>= 1) {
				for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
					const uint partner = i ^ j;
					if (partner > i) {
						bool stageAscending = (i & k) == 0u;
						if (!ascending) stageAscending = !stageAscending;

						const uint slotA = BPS_DOMAIN_INDEX(size + i);
						const uint slotB = BPS_DOMAIN_INDEX(size + partner);
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
				barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
			}
		}

		const uint shift = bps_cycle_shift(spanSize, cycleDegrees);
		// Flag reordered slots (run length >= 2) so ApplyDomain composites only
		// those over the source and leaves unsorted areas pristine.
		const uint affectedBit = (spanSize >= 2u) ? 0x80000000u : 0u;
		for (uint i = gtid; i < spanSize; i += MAX_THREADS) {
			const uint sortedIndex = (i + spanSize - shift) % spanSize;
			const uint pos = as_uint(keys[BPS_DOMAIN_INDEX(runStart + i)]);
			const uint srcIndex = domain[BPS_DOMAIN_INDEX(size + sortedIndex)];
			domain[BPS_DOMAIN_INDEX(pos)] = srcIndex | affectedBit;
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		cursor = runEnd + 1u;
	}

	#undef BPS_SRC_IN_WORLD
	#undef BPS_SRC_INDEX_XY
	#undef BPS_DOMAIN_INDEX
}

__kernel void BitonicApplyDomainKernel(
	__global const float4 *srcTex,
	__global float4       *dstTex,
	__global const uint   *domain,
	int                    srcPitch,
	int                    dstPitch,
	int                    width,
	int                    height,
	int                    inputOriginX,
	int                    inputOriginY,
	int                    inputWidth,
	int                    inputHeight,
	int                    outputOriginX,
	int                    outputOriginY,
	int                    outputWidth,
	int                    outputHeight,
	int                    mode,
	int                    lineCount,
	int                    freePMin,
	int                    freeQMin,
	int                    freeLineLength,
	int                    radialLength,
	int                    domainStride,
	float                  angleCos,
	float                  angleSin,
	float                  centerX,
	float                  centerY,
	float                  swirlK,
	int                    swirlLineMin,
	int                    pathDirection,
	int                    pathSMin,
	int                    pathNMin,
	int                    pathSampleCount,
	__global const BpsPathSampleGpu *pathSamples)
{
	const int ox = (int)get_global_id(0);
	const int oy = (int)get_global_id(1);
	if (ox >= outputWidth || oy >= outputHeight) {
		return;
	}

	const int x = outputOriginX + ox;
	const int y = outputOriginY + oy;
	const uint dstIndex = (uint)(ox + oy * dstPitch);

	float4 pixel = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	if (x >= inputOriginX && y >= inputOriginY &&
		x < inputOriginX + inputWidth && y < inputOriginY + inputHeight) {
		const uint srcIndex =
			(uint)((x - inputOriginX) + (y - inputOriginY) * srcPitch);
		pixel = srcTex[srcIndex];
	}

	int line = 0;
	int pos = 0;
	if (bps_domain_pos_for_pixel(mode, x, y, lineCount,
								 freePMin, freeQMin, freeLineLength, radialLength,
								 domainStride, angleCos, angleSin, centerX, centerY,
								 swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
								 pathSampleCount, pathSamples,
								 &line, &pos)) {
		const uint raw = domain[(uint)(line * domainStride + pos)];
		// High bit flags slots a sort reordered; composite only those over the
		// source so unsorted areas keep the exact original (no resample loss).
		if (raw != 0xffffffffu && (raw & 0x80000000u) != 0u) {
			pixel = srcTex[raw & 0x7fffffffu];
		}
	}

	dstTex[dstIndex] = pixel;
}

__kernel void BitonicCopyInputKernel(
	__global const float4 *srcTex,
	__global float4       *dstTex,
	int                    srcPitch,
	int                    dstPitch,
	int                    width,
	int                    height,
	int                    inputOriginX,
	int                    inputOriginY,
	int                    inputWidth,
	int                    inputHeight,
	int                    outputOriginX,
	int                    outputOriginY,
	int                    outputWidth,
	int                    outputHeight)
{
	const int ox = (int)get_global_id(0);
	const int oy = (int)get_global_id(1);
	if (ox >= outputWidth || oy >= outputHeight) {
		return;
	}

	const int x = outputOriginX + ox;
	const int y = outputOriginY + oy;
	const uint dstIndex = (uint)(ox + oy * dstPitch);
	float4 pixel = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	if (x >= inputOriginX && y >= inputOriginY &&
		x < inputOriginX + inputWidth && y < inputOriginY + inputHeight) {
		pixel = srcTex[(uint)((x - inputOriginX) + (y - inputOriginY) * srcPitch)];
	}
	dstTex[dstIndex] = pixel;
}

__kernel void BitonicBuildPathClassifyCountKernel(
	__global const BpsPathSampleGpu *pathSamples,
	int                    pathSampleCount,
	int                    width,
	int                    height,
	int                    lineCount,
	int                    pathDirection,
	int                    pathClosed,
	float                  pathLength,
	int                    pathSMin,
	int                    pathNMin,
	__global int          *laneOf,
	__global float        *keyOf,
	volatile __global uint *laneCounts)
{
	const int x = (int)get_global_id(0);
	const int y = (int)get_global_id(1);
	if (x >= width || y >= height) {
		return;
	}
	const uint pidx = (uint)(y * width + x);
	laneOf[pidx] = -1;

	float s = 0.0f;
	float n = 0.0f;
	if (!bps_path_closest(pathSamples, pathSampleCount, (float)x, (float)y, &s, &n)) {
		return;
	}

	int lane = 0;
	float order = 0.0f;
	if (!bps_path_lane_order(pathDirection, pathClosed, pathLength,
							 pathSMin, pathNMin, lineCount,
							 s, n, &lane, &order)) {
		return;
	}
	laneOf[pidx] = lane;
	keyOf[pidx] = order;
	atomic_add(&laneCounts[lane], 1u);
}

__kernel void BitonicBuildPathScatterRecordsKernel(
	__global const int    *laneOf,
	__global const float  *keyOf,
	volatile __global uint *laneCursors,
	__global BpsMappedPixelRecordGpu *records,
	int                    width,
	int                    height)
{
	const int x = (int)get_global_id(0);
	const int y = (int)get_global_id(1);
	if (x >= width || y >= height) {
		return;
	}
	const uint pidx = (uint)(y * width + x);
	const int lane = laneOf[pidx];
	if (lane < 0) {
		return;
	}
	const uint dst = atomic_add(&laneCursors[lane], 1u);
	records[dst].posKey = keyOf[pidx];
	records[dst].pixelIndex = pidx;
}

inline bool bps_record_before(BpsMappedPixelRecordGpu a, BpsMappedPixelRecordGpu b)
{
	if (a.posKey < b.posKey) return true;
	if (a.posKey > b.posKey) return false;
	return a.pixelIndex < b.pixelIndex;
}

__kernel void BitonicBuildPathSortRecordsKernel(
	__global BpsMappedPixelRecordGpu *records,
	__global BpsMappedPixelRecordGpu *workRecords,
	__global const uint   *lineOffsets,
	__global const uint   *workOffsets,
	int                    lineCount)
{
	const uint line = get_group_id(0);
	const uint gtid = get_local_id(0);
	if ((int)line >= lineCount) {
		return;
	}
	const uint begin = lineOffsets[line];
	const uint end = lineOffsets[line + 1u];
	const uint lineSize = end - begin;
	if (lineSize <= 1u) {
		return;
	}

	const uint sortSize = bps_next_pow2(lineSize);
	const uint sortBase = workOffsets[line] + lineSize;
	for (uint i = gtid; i < sortSize; i += MAX_THREADS) {
		BpsMappedPixelRecordGpu record;
		if (i < lineSize) {
			record = records[begin + i];
		} else {
			record.posKey = BPS_FLOAT_MAX;
			record.pixelIndex = 0xffffffffu;
		}
		workRecords[sortBase + i] = record;
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

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
			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		}
	}

	for (uint i = gtid; i < lineSize; i += MAX_THREADS) {
		records[begin + i] = workRecords[sortBase + i];
	}
}

__kernel void BitonicSortMappedKernel(
	__global const float4 *srcTex,
	__global float4       *dstTex,
	__global const float4 *criterionTex,
	__global const float4 *triggerTex,
	int                    criterionPitch,
	int                    criterionOriginX,
	int                    criterionOriginY,
	int                    criterionWidth,
	int                    criterionHeight,
	int                    triggerPitch,
	int                    triggerOriginX,
	int                    triggerOriginY,
	int                    triggerWidth,
	int                    triggerHeight,
	__global uint         *domain,
	__global float        *keys,
	__global const BpsMappedPixelRecordGpu *records,
	__global const uint   *lineOffsets,
	__global const uint   *workOffsets,
	int                    srcPitch,
	int                    dstPitch,
	int                    width,
	int                    height,
	int                    inputOriginX,
	int                    inputOriginY,
	int                    inputWidth,
	int                    inputHeight,
	int                    outputOriginX,
	int                    outputOriginY,
	int                    outputWidth,
	int                    outputHeight,
	int                    ordering,
	int                    criterion,
	int                    trigger,
	int                    affect,
	float                  cycleDegrees,
	float                  thresholdMin,
	float                  thresholdMax,
	int                    lineCount)
{

	BpsKeySource criterionSrc;
	criterionSrc.tex = criterionTex;
	criterionSrc.pitch = criterionPitch;
	criterionSrc.originX = criterionOriginX;
	criterionSrc.originY = criterionOriginY;
	criterionSrc.width = criterionWidth;
	criterionSrc.height = criterionHeight;
	BpsKeySource triggerSrc;
	triggerSrc.tex = triggerTex;
	triggerSrc.pitch = triggerPitch;
	triggerSrc.originX = triggerOriginX;
	triggerSrc.originY = triggerOriginY;
	triggerSrc.width = triggerWidth;
	triggerSrc.height = triggerHeight;
	__local uint s_spanStart;
	__local uint s_spanEnd;
	__local uint s_spanSize;
	__local uint s_sortSize;

	const uint line = get_group_id(0);
	const uint gtid = get_local_id(0);
	if ((int)line >= lineCount) {
		return;
	}

	const uint begin = lineOffsets[line];
	const uint end = lineOffsets[line + 1u];
	const uint lineSize = end - begin;
	if (lineSize == 0u) {
		return;
	}
	const uint workBase = workOffsets[line];
	const bool ascending = ordering != 0;

	#define BPS_SRC_IN_WORLD(x, y) ((x) >= inputOriginX && (y) >= inputOriginY && \
									(x) < inputOriginX + inputWidth && (y) < inputOriginY + inputHeight)
	#define BPS_DST_IN_WORLD(x, y) ((x) >= outputOriginX && (y) >= outputOriginY && \
									(x) < outputOriginX + outputWidth && (y) < outputOriginY + outputHeight)
	#define BPS_SRC_INDEX_XY(x, y) ((uint)(((x) - inputOriginX) + ((y) - inputOriginY) * srcPitch))
	#define BPS_DST_INDEX_XY(x, y) ((uint)(((x) - outputOriginX) + ((y) - outputOriginY) * dstPitch))

	for (uint i = gtid; i < lineSize; i += MAX_THREADS) {
		const uint pixelIndex = records[begin + i].pixelIndex;
		const int x = (int)(pixelIndex % (uint)width);
		const int y = (int)(pixelIndex / (uint)width);
		domain[workBase + i] = BPS_SRC_IN_WORLD(x, y)
			? BPS_SRC_INDEX_XY(x, y)
			: 0xffffffffu;
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	uint cursor = 0u;
	while (cursor < lineSize) {
		if (gtid == 0u) {
			uint runStart = cursor;
			while (runStart < lineSize) {
				const uint srcIndex = domain[workBase + runStart];
				if (srcIndex != 0xffffffffu) {
					const float br = bps_sample_key_from_src_index(triggerSrc, srcPitch, inputOriginX, inputOriginY, srcIndex, trigger);
					if (bps_is_affected(br, thresholdMin, thresholdMax, affect)) break;
				}
				runStart++;
			}
			uint runEnd = runStart;
			while (runEnd < lineSize) {
				const uint srcIndex = domain[workBase + runEnd];
				if (srcIndex == 0xffffffffu) break;
				const float br = bps_sample_key_from_src_index(triggerSrc, srcPitch, inputOriginX, inputOriginY, srcIndex, trigger);
				if (!bps_is_affected(br, thresholdMin, thresholdMax, affect)) break;
				runEnd++;
			}
			s_spanStart = runStart;
			s_spanEnd = runEnd;
			s_spanSize = runEnd - runStart;
			s_sortSize = bps_next_pow2(s_spanSize);
		}
		barrier(CLK_LOCAL_MEM_FENCE);

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
				keys[sortBase + i] = bps_sample_key_from_src_index(criterionSrc, srcPitch, inputOriginX, inputOriginY, srcIndex, criterion);
			} else {
				domain[sortBase + i] = 0xffffffffu;
				keys[sortBase + i] = ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

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
				barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
			}
		}

		const uint shift = bps_cycle_shift(spanSize, cycleDegrees);
		for (uint i = gtid; i < spanSize; i += MAX_THREADS) {
			const uint sortedIndex = (i + spanSize - shift) % spanSize;
			const uint srcIndex = domain[sortBase + sortedIndex];
			const uint pixelIndex = records[begin + runStart + i].pixelIndex;
			const int x = (int)(pixelIndex % (uint)width);
			const int y = (int)(pixelIndex / (uint)width);
			if (srcIndex != 0xffffffffu && BPS_DST_IN_WORLD(x, y)) {
				dstTex[BPS_DST_INDEX_XY(x, y)] = srcTex[srcIndex];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		cursor = runEnd + 1u;
	}

	#undef BPS_SRC_IN_WORLD
	#undef BPS_DST_IN_WORLD
	#undef BPS_SRC_INDEX_XY
	#undef BPS_DST_INDEX_XY
}
