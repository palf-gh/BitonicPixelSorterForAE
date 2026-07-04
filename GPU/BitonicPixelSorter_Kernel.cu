/*
	BitonicPixelSorter_Kernel.cu

	CUDA port of the upstream single-dispatch bitonic pixel sort
	(Packages/com.ruccho.bitonicpixelsorter/Runtime/BitonicPixelSorter.compute).

	Axis mode uses a single pass: one thread block sorts one line and gathers
	full-colour pixels into the destination via sorted *indices*, so colours
	never pass through shared memory.

	Non-axis modes (Free Angle / Rotation / Radial) use exact multi-pass:
	  1. Sort each path into a padded domain of source indices.
	  2. Inverse-map every output pixel and gather from that domain.
	Forward scatter would race where multiple path samples map to one pixel.

	After Effects GPU worlds are PF_PixelFormat_GPU_BGRA128: linear, row-pitched
	float4 buffers in BGRA order (x=B, y=G, z=R, w=A). Pitches below are in float4
	units (rowbytes / 16).
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Single configuration covering lines up to 4096 px (the host falls back to CPU
// beyond that). 256 threads, 32 KiB of shared sort keys and source indices.
#define MAX_THREADS 256
#define MAX_SIZE    4096
#define BPS_FLOAT_MAX 3.402823466e+38F
#define BPS_PI 3.14159265358979323846f
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

struct BpsPathSampleGpu {
	float x;
	float y;
	float s;
	float tx;
	float ty;
};

struct BpsMappedPixelRecordGpu {
	float posKey;
	unsigned int pixelIndex;
};

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

__device__ __forceinline__ float bps_sort_key(float4 c, int criterion)
{
	// BGRA: R=.z, G=.y, B=.x
	if (criterion == BPS_CRITERION_RGB_AVERAGE) {
		return __saturatef((c.z + c.y + c.x) * (1.0f / 3.0f));
	}
	if (criterion == BPS_CRITERION_RGB_PRODUCT) {
		return __saturatef(c.z * c.y * c.x);
	}
	if (criterion == BPS_CRITERION_RGB_MINIMUM) {
		return __saturatef(fminf(fminf(c.z, c.y), c.x));
	}
	if (criterion == BPS_CRITERION_RGB_MAXIMUM) {
		return __saturatef(fmaxf(fmaxf(c.z, c.y), c.x));
	}
	if (criterion == BPS_CRITERION_RED_CHANNEL) {
		return __saturatef(c.z);
	}
	if (criterion == BPS_CRITERION_GREEN_CHANNEL) {
		return __saturatef(c.y);
	}
	if (criterion == BPS_CRITERION_BLUE_CHANNEL) {
		return __saturatef(c.x);
	}
	if (criterion == BPS_CRITERION_ALPHA_CHANNEL) {
		return __saturatef(c.w);
	}
	if (criterion == BPS_CRITERION_HUE || criterion == BPS_CRITERION_SATURATION) {
		const float r = __saturatef(c.z);
		const float g = __saturatef(c.y);
		const float b = __saturatef(c.x);
		const float maxRGB = fmaxf(fmaxf(r, g), b);
		const float minRGB = fminf(fminf(r, g), b);
		const float delta = maxRGB - minRGB;
		if (criterion == BPS_CRITERION_SATURATION) {
			return maxRGB <= 0.0f ? 0.0f : __saturatef(delta / maxRGB);
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
		return __saturatef(hue * (1.0f / 6.0f));
	}
	return __saturatef(0.298912f * c.z + 0.586611f * c.y + 0.114478f * c.x);
}

__device__ __forceinline__ bool bps_is_affected(
	float triggerKey,
	float thresholdMin,
	float thresholdMax,
	int affect)
{
	const bool inside = triggerKey >= thresholdMin && triggerKey <= thresholdMax;
	return affect == BPS_AFFECT_OUTSIDE_THRESHOLDS ? !inside : inside;
}

__device__ __forceinline__ unsigned int bps_cycle_shift(unsigned int count, float cycleDegrees)
{
	if (count <= 1u) return 0u;
	int shift = (int)floorf((cycleDegrees / 360.0f) * (float)count + 0.5f);
	const int n = (int)count;
	shift %= n;
	if (shift < 0) shift += n;
	return (unsigned int)shift;
}

__device__ __forceinline__ unsigned int bps_next_pow2(unsigned int value)
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

__device__ __forceinline__ bool bps_before(
	float keyA,
	unsigned int indexA,
	float keyB,
	unsigned int indexB)
{
	if (keyA < keyB) return true;
	if (keyA > keyB) return false;
	return indexA < indexB;
}

__device__ __forceinline__ int bps_round_to_int(float value)
{
	return (int)floorf(value + 0.5f);
}

__device__ __forceinline__ unsigned int bps_rotation_line_length(unsigned int radius)
{
	if (radius == 0u) return 1u;
	const unsigned int length = (unsigned int)ceilf(BPS_TWO_PI * (float)radius);
	return length == 0u ? 1u : length;
}

__device__ __forceinline__ void bps_normalise_tangent(float *tx, float *ty)
{
	const float len = sqrtf((*tx) * (*tx) + (*ty) * (*ty));
	if (len > 1.0e-6f) {
		*tx /= len;
		*ty /= len;
	} else {
		*tx = 1.0f;
		*ty = 0.0f;
	}
}

__device__ __forceinline__ bool bps_path_eval_at_s(
	const BpsPathSampleGpu *samples,
	int sampleCount,
	float s,
	float *x,
	float *y,
	float *tx,
	float *ty)
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

__device__ __forceinline__ bool bps_path_closest(
	const BpsPathSampleGpu *samples,
	int sampleCount,
	float px,
	float py,
	float *sOut,
	float *nOut)
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
			t = fminf(1.0f, fmaxf(0.0f, t));
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

__device__ __forceinline__ unsigned int bps_line_size(
	int mode,
	int direction,
	unsigned int gid,
	int width,
	int height,
	int freeLineLength,
	int radialLength,
	int domainStride,
	float swirlK)
{
	if (mode == BPS_MODE_FREE_ANGLE) return (unsigned int)freeLineLength;
	if (mode == BPS_MODE_ROTATION) return bps_rotation_line_length(gid);
	if (mode == BPS_MODE_RADIAL) return (unsigned int)radialLength;
	if (mode == BPS_MODE_SWIRL) {
		return (unsigned int)(radialLength > 0 ? radialLength : 1);
	}
	if (mode == BPS_MODE_PATH) {
		return (unsigned int)(radialLength > 0 ? radialLength : 1);
	}
	(void)domainStride;
	return direction ? (unsigned int)width : (unsigned int)height;
}

__device__ __forceinline__ bool bps_coord_for_pos(
	int mode,
	int direction,
	unsigned int gid,
	unsigned int pos,
	int width,
	int height,
	int outputOriginX,
	int outputOriginY,
	int lineCount,
	int freePMin,
	int freeQMin,
	int freeLineLength,
	float angleCos,
	float angleSin,
	float centerX,
	float centerY,
	float swirlK,
	int swirlLineMin,
	int pathDirection,
	int pathSMin,
	int pathNMin,
	int pathSampleCount,
	const BpsPathSampleGpu *pathSamples,
	int *x,
	int *y)
{
	if (mode == BPS_MODE_FREE_ANGLE) {
		const float p = (float)(freePMin + (int)pos);
		const float q = (float)(freeQMin + (int)gid);
		*x = bps_round_to_int(p * angleCos - q * angleSin);
		*y = bps_round_to_int(p * angleSin + q * angleCos);
		(void)freeLineLength;
	} else if (mode == BPS_MODE_ROTATION) {
		const unsigned int lineLen = bps_rotation_line_length(gid);
		const float theta = (lineLen <= 1u) ? 0.0f :
			(BPS_TWO_PI * (float)pos) / (float)lineLen;
		const float c = cosf(theta);
		const float s = sinf(theta);
		const float radius = (float)gid;
		*x = bps_round_to_int(centerX + radius * (s * angleCos + c * angleSin));
		*y = bps_round_to_int(centerY + radius * (s * angleSin - c * angleCos));
	} else if (mode == BPS_MODE_RADIAL) {
		const float denom = lineCount <= 0 ? 1.0f : (float)lineCount;
		const float theta = (BPS_TWO_PI * (float)gid) / denom;
		*x = bps_round_to_int(centerX + (float)pos * cosf(theta));
		*y = bps_round_to_int(centerY + (float)pos * sinf(theta));
	} else if (mode == BPS_MODE_SWIRL) {
		// phase = atan2 - angle - k*r; line indexes phase, pos is radius.
		const float r = (float)pos;
		const float denom = lineCount <= 0 ? 1.0f : (float)lineCount;
		const float phase = (BPS_TWO_PI * (float)gid) / denom;
		// Recover absolute angle: atan2 = phase + k*r + angle, with
		// angle from (angleCos, angleSin) as angleRadians.
		const float angle = atan2f(angleSin, angleCos);
		const float theta = phase + swirlK * r + angle;
		*x = bps_round_to_int(centerX + r * cosf(theta));
		*y = bps_round_to_int(centerY + r * sinf(theta));
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

// Inverse of bps_coord_for_pos for non-axis modes. Each output pixel maps to
// exactly one (line, pos) domain slot - the exact-mapping contract.
__device__ __forceinline__ bool bps_domain_pos_for_pixel(
	int mode,
	int x,
	int y,
	int lineCount,
	int freePMin,
	int freeQMin,
	int freeLineLength,
	int radialLength,
	int domainStride,
	float angleCos,
	float angleSin,
	float centerX,
	float centerY,
	float swirlK,
	int swirlLineMin,
	int pathDirection,
	int pathSMin,
	int pathNMin,
	int pathSampleCount,
	const BpsPathSampleGpu *pathSamples,
	int *lineP,
	int *posP)
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
		line = bps_round_to_int(sqrtf(dx * dx + dy * dy));
		const unsigned int lineLen = bps_rotation_line_length((unsigned int)(line < 0 ? 0 : line));
		if (lineLen <= 1u) {
			pos = 0;
		} else {
			const float rx = dx * angleCos + dy * angleSin;
			const float ry = -dx * angleSin + dy * angleCos;
			float theta = atan2f(rx, -ry);
			if (theta < 0.0f) theta += BPS_TWO_PI;
			pos = bps_round_to_int((theta / BPS_TWO_PI) * (float)lineLen);
			if (pos >= (int)lineLen) pos -= (int)lineLen;
		}
	} else if (mode == BPS_MODE_RADIAL) {
		const float dx = (float)x - centerX;
		const float dy = (float)y - centerY;
		pos = bps_round_to_int(sqrtf(dx * dx + dy * dy));
		float theta = atan2f(dy, dx);
		if (theta < 0.0f) theta += BPS_TWO_PI;
		line = (lineCount <= 1) ? 0 :
			bps_round_to_int((theta / BPS_TWO_PI) * (float)lineCount);
		if (line >= lineCount) line -= lineCount;
	} else if (mode == BPS_MODE_SWIRL) {
		const float dx = (float)x - centerX;
		const float dy = (float)y - centerY;
		const float r = sqrtf(dx * dx + dy * dy);
		const float angle = atan2f(angleSin, angleCos);
		float phase = atan2f(dy, dx) - angle - swirlK * r;
		if (phase < 0.0f) phase += BPS_TWO_PI;
		phase = fmodf(phase, BPS_TWO_PI);
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
	else if (mode == BPS_MODE_ROTATION) lineLen = (int)bps_rotation_line_length((unsigned int)line);
	else lineLen = radialLength;

	if (pos < 0 || pos >= lineLen || pos >= domainStride) return false;

	*lineP = line;
	*posP = pos;
	return true;
}

__global__ void BitonicSortKernel(
	const float4 *srcTex,
	float4       *sortTex,
	int           srcPitch,
	int           dstPitch,
	int           width,
	int           height,
	int           inputOriginX,
	int           inputOriginY,
	int           outputOriginX,
	int           outputOriginY,
	int           outputWidth,
	int           outputHeight,
	int           mode,
	int           direction,     // 1 = horizontal (sort along X), 0 = vertical
	int           ordering,      // 1 = ascending
	int           criterion,
	int           trigger,
	int           affect,
	float         cycleDegrees,
	int           lineCount,
	int           freePMin,
	int           freeQMin,
	int           freeLineLength,
	int           radialLength,
	float         thresholdMin,
	float         thresholdMax,
	float         angleCos,
	float         angleSin,
	float         centerX,
	float         centerY,
	float         swirlK,
	int           swirlLineMin,
	int           pathDirection,
	int           pathSMin,
	int           pathNMin,
	int           pathSampleCount,
	const BpsPathSampleGpu *pathSamples)
{
	__shared__ float scratchKey[MAX_SIZE];
	__shared__ unsigned int scratchIndex[MAX_SIZE];
	// Dedicated span metadata, kept separate from the sort scratch so the load
	// loop never overwrites it. Thread 0 writes it; a barrier publishes it to
	// every thread, keeping spanSize/sortSize uniform across the block (the
	// per-span break and the bitonic-network barriers must stay non-divergent).
	__shared__ unsigned int s_spanStart;
	__shared__ unsigned int s_spanEnd;
	__shared__ unsigned int s_spanSize;
	__shared__ unsigned int s_sortSize;

	const unsigned int gid  = blockIdx.x;
	const unsigned int gtid = threadIdx.x;

	const unsigned int size = bps_line_size(
		mode, direction, gid, width, height, freeLineLength, radialLength, 0, swirlK);
	if (size == 0u || size > MAX_SIZE) {
		return;
	}

	// Map layer coordinates into AE's possibly partial GPU worlds.
	#define BPS_SRC_IN_WORLD(x, y) ((x) >= inputOriginX && (y) >= inputOriginY && \
									(x) < inputOriginX + width && (y) < inputOriginY + height)
	#define BPS_DST_IN_WORLD(x, y) ((x) >= outputOriginX && (y) >= outputOriginY && \
									(x) < outputOriginX + outputWidth && (y) < outputOriginY + outputHeight)
	#define BPS_SRC_INDEX_XY(x, y) ((unsigned int)(((x) - inputOriginX) + ((y) - inputOriginY) * srcPitch))
	#define BPS_DST_INDEX_XY(x, y) ((unsigned int)(((x) - outputOriginX) + ((y) - outputOriginY) * dstPitch))

	for (unsigned int pos = gtid; pos < size; pos += MAX_THREADS) {
		int x = 0, y = 0;
		if (bps_coord_for_pos(mode, direction, gid, pos, width, height,
							  outputOriginX, outputOriginY, lineCount,
							  freePMin, freeQMin, freeLineLength,
							  angleCos, angleSin, centerX, centerY,
							  swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
							  pathSampleCount, pathSamples, &x, &y) &&
			BPS_SRC_IN_WORLD(x, y) && BPS_DST_IN_WORLD(x, y)) {
			sortTex[BPS_DST_INDEX_XY(x, y)] = srcTex[BPS_SRC_INDEX_XY(x, y)];
		}
	}
	__syncthreads();

	unsigned int cursor = 0;
	while (cursor < size) {
		if (gtid == 0) {
			unsigned int spanStart = cursor;
			while (spanStart < size) {
				int x = 0, y = 0;
				if (bps_coord_for_pos(mode, direction, gid, spanStart, width, height,
									  outputOriginX, outputOriginY, lineCount,
									  freePMin, freeQMin, freeLineLength,
									  angleCos, angleSin, centerX, centerY,
									  swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
									  pathSampleCount, pathSamples, &x, &y) &&
					BPS_SRC_IN_WORLD(x, y)) {
					float br = bps_sort_key(srcTex[BPS_SRC_INDEX_XY(x, y)], trigger);
					if (bps_is_affected(br, thresholdMin, thresholdMax, affect)) break;
				}
				spanStart++;
			}

			unsigned int spanEnd = spanStart;
			while (spanEnd < size) {
				int x = 0, y = 0;
				if (!bps_coord_for_pos(mode, direction, gid, spanEnd, width, height,
									   outputOriginX, outputOriginY, lineCount,
									   freePMin, freeQMin, freeLineLength,
									   angleCos, angleSin, centerX, centerY,
									   swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
									   pathSampleCount, pathSamples, &x, &y) ||
					!BPS_SRC_IN_WORLD(x, y)) break;
				float br = bps_sort_key(srcTex[BPS_SRC_INDEX_XY(x, y)], trigger);
				if (!bps_is_affected(br, thresholdMin, thresholdMax, affect)) break;
				spanEnd++;
			}

			const unsigned int spanSize = spanEnd - spanStart;
			s_spanStart = spanStart;
			s_spanEnd = spanEnd;
			s_spanSize = spanSize;
			s_sortSize = bps_next_pow2(spanSize);
		}
		__syncthreads();

		const unsigned int spanStart = s_spanStart;
		const unsigned int spanEnd = s_spanEnd;
		const unsigned int spanSize = s_spanSize;
		const unsigned int sortSize = s_sortSize;

		if (spanStart >= size || spanSize == 0u) {
			break;
		}

		const bool ascending = ordering != 0;
		for (unsigned int i = gtid; i < sortSize; i += MAX_THREADS) {
			if (i < spanSize) {
				const unsigned int pos = spanStart + i;
				int x = 0, y = 0;
				const bool valid = bps_coord_for_pos(mode, direction, gid, pos, width, height,
													 outputOriginX, outputOriginY, lineCount,
													 freePMin, freeQMin, freeLineLength,
													 angleCos, angleSin, centerX, centerY,
													 swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
													 pathSampleCount, pathSamples, &x, &y) &&
								   BPS_SRC_IN_WORLD(x, y);
				const unsigned int srcIndex = valid ? BPS_SRC_INDEX_XY(x, y) : 0xffffffffu;
				scratchKey[i] = valid ? bps_sort_key(srcTex[srcIndex], criterion) :
					(ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX);
				scratchIndex[i] = srcIndex;
			} else {
				scratchKey[i] = ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX;
				scratchIndex[i] = 0xFFFFFFFFu;
			}
		}
		__syncthreads();

		for (unsigned int k = 2u; k <= sortSize; k <<= 1) {
			for (unsigned int j = k >> 1; j > 0u; j >>= 1) {
				for (unsigned int i = gtid; i < sortSize; i += MAX_THREADS) {
					const unsigned int partner = i ^ j;
					if (partner > i) {
						bool stageAscending = (i & k) == 0u;
						if (!ascending) stageAscending = !stageAscending;

						const float keyA = scratchKey[i];
						const float keyB = scratchKey[partner];
						const unsigned int indexA = scratchIndex[i];
						const unsigned int indexB = scratchIndex[partner];
						const bool before = bps_before(keyA, indexA, keyB, indexB);
						if (before != stageAscending) {
							scratchKey[i] = keyB;
							scratchKey[partner] = keyA;
							scratchIndex[i] = indexB;
							scratchIndex[partner] = indexA;
						}
					}
				}
				__syncthreads();
			}
		}

		for (unsigned int i = gtid; i < spanSize; i += MAX_THREADS) {
			const unsigned int pos = spanStart + i;
			const unsigned int shift = bps_cycle_shift(spanSize, cycleDegrees);
			const unsigned int sortedIndex = (i + spanSize - shift) % spanSize;
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
		__syncthreads();

		cursor = spanEnd + 1u;
	}

	#undef BPS_SRC_IN_WORLD
	#undef BPS_DST_IN_WORLD
	#undef BPS_SRC_INDEX_XY
	#undef BPS_DST_INDEX_XY
}

// Non-axis pass 1: sort each path into a padded domain of source indices.
// Keys live in global memory so Rotation/Radial paths may exceed MAX_SIZE.
__global__ void BitonicSortDomainKernel(
	const float4     *srcTex,
	unsigned int     *domain,
	float            *keys,
	int               srcPitch,
	int               width,
	int               height,
	int               inputOriginX,
	int               inputOriginY,
	int               mode,
	int               ordering,
	int               criterion,
	int               trigger,
	int               affect,
	float             cycleDegrees,
	int               lineCount,
	int               freePMin,
	int               freeQMin,
	int               freeLineLength,
	int               radialLength,
	int               domainStride,
	float             thresholdMin,
	float             thresholdMax,
	float             angleCos,
	float             angleSin,
	float             centerX,
	float             centerY,
	float             swirlK,
	int               swirlLineMin,
	int               pathDirection,
	int               pathSMin,
	int               pathNMin,
	int               pathSampleCount,
	const BpsPathSampleGpu *pathSamples)
{
	__shared__ unsigned int s_spanStart;
	__shared__ unsigned int s_spanEnd;
	__shared__ unsigned int s_spanSize;
	__shared__ unsigned int s_sortSize;

	const unsigned int gid  = blockIdx.x;
	const unsigned int gtid = threadIdx.x;

	const unsigned int size = bps_line_size(
		mode, 0, gid, width, height, freeLineLength, radialLength, domainStride, swirlK);
	if (size == 0u || domainStride <= 0 || (int)size > domainStride) {
		return;
	}

	#define BPS_SRC_IN_WORLD(x, y) ((x) >= inputOriginX && (y) >= inputOriginY && \
									(x) < inputOriginX + width && (y) < inputOriginY + height)
	#define BPS_SRC_INDEX_XY(x, y) ((unsigned int)(((x) - inputOriginX) + ((y) - inputOriginY) * srcPitch))
	#define BPS_DOMAIN_INDEX(pos) ((unsigned int)((int)gid * domainStride + (int)(pos)))

	// Forward path sampling into domain[0..size).
	for (unsigned int pos = gtid; pos < size; pos += MAX_THREADS) {
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
	__syncthreads();

	// Build in-bounds path in keys[0..pathLen) as uint bitcast, starting at
	// pos = 0 (angle start). Out-of-frame samples are omitted (not run breaks)
	// so the visible arc sorts as one continuous sequence.
	__shared__ unsigned int s_pathLen;
	if (gtid == 0) {
		unsigned int pathLen = 0;
		for (unsigned int pos = 0; pos < size; ++pos) {
			if (domain[BPS_DOMAIN_INDEX(pos)] != 0xffffffffu) {
				keys[BPS_DOMAIN_INDEX(pathLen)] = __uint_as_float(pos);
				pathLen++;
			}
		}
		s_pathLen = pathLen;
	}
	__syncthreads();

	const unsigned int pathLen = s_pathLen;
	unsigned int cursor = 0;
	while (cursor < pathLen) {
		if (gtid == 0) {
			unsigned int runStart = cursor;
			while (runStart < pathLen) {
				const unsigned int pos =
					__float_as_uint(keys[BPS_DOMAIN_INDEX(runStart)]);
				const unsigned int srcIndex = domain[BPS_DOMAIN_INDEX(pos)];
				const float br = bps_sort_key(srcTex[srcIndex], trigger);
				if (bps_is_affected(br, thresholdMin, thresholdMax, affect)) break;
				runStart++;
			}

			unsigned int runEnd = runStart;
			while (runEnd < pathLen) {
				const unsigned int pos =
					__float_as_uint(keys[BPS_DOMAIN_INDEX(runEnd)]);
				const unsigned int srcIndex = domain[BPS_DOMAIN_INDEX(pos)];
				const float br = bps_sort_key(srcTex[srcIndex], trigger);
				if (!bps_is_affected(br, thresholdMin, thresholdMax, affect)) break;
				runEnd++;
			}

			s_spanStart = runStart;
			s_spanEnd = runEnd;
			s_spanSize = runEnd - runStart;
			s_sortSize = bps_next_pow2(s_spanSize);
		}
		__syncthreads();

		const unsigned int runStart = s_spanStart;
		const unsigned int runEnd = s_spanEnd;
		const unsigned int spanSize = s_spanSize;
		const unsigned int sortSize = s_sortSize;

		if (runStart >= pathLen || spanSize == 0u) {
			break;
		}

		// Compact the (possibly non-contiguous) run into workspace at offset
		// `size`, bitonic-sort there, then scatter back to path positions.
		const bool ascending = ordering != 0;
		for (unsigned int i = gtid; i < sortSize; i += MAX_THREADS) {
			const unsigned int work = BPS_DOMAIN_INDEX(size + i);
			if (i < spanSize) {
				const unsigned int pos =
					__float_as_uint(keys[BPS_DOMAIN_INDEX(runStart + i)]);
				const unsigned int srcIndex = domain[BPS_DOMAIN_INDEX(pos)];
				domain[work] = srcIndex;
				keys[work] = bps_sort_key(srcTex[srcIndex], criterion);
			} else {
				domain[work] = 0xffffffffu;
				keys[work] = ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX;
			}
		}
		__syncthreads();

		for (unsigned int k = 2u; k <= sortSize; k <<= 1) {
			for (unsigned int j = k >> 1; j > 0u; j >>= 1) {
				for (unsigned int i = gtid; i < sortSize; i += MAX_THREADS) {
					const unsigned int partner = i ^ j;
					if (partner > i) {
						bool stageAscending = (i & k) == 0u;
						if (!ascending) stageAscending = !stageAscending;

						const unsigned int slotA = BPS_DOMAIN_INDEX(size + i);
						const unsigned int slotB = BPS_DOMAIN_INDEX(size + partner);
						const float keyA = keys[slotA];
						const float keyB = keys[slotB];
						const unsigned int indexA = domain[slotA];
						const unsigned int indexB = domain[slotB];
						const bool before = bps_before(keyA, indexA, keyB, indexB);
						if (before != stageAscending) {
							keys[slotA] = keyB;
							keys[slotB] = keyA;
							domain[slotA] = indexB;
							domain[slotB] = indexA;
						}
					}
				}
				__syncthreads();
			}
		}

		const unsigned int shift = bps_cycle_shift(spanSize, cycleDegrees);
		for (unsigned int i = gtid; i < spanSize; i += MAX_THREADS) {
			const unsigned int sortedIndex = (i + spanSize - shift) % spanSize;
			const unsigned int pos =
				__float_as_uint(keys[BPS_DOMAIN_INDEX(runStart + i)]);
			domain[BPS_DOMAIN_INDEX(pos)] =
				domain[BPS_DOMAIN_INDEX(size + sortedIndex)];
		}
		__syncthreads();

		cursor = runEnd + 1u;
	}

	#undef BPS_SRC_IN_WORLD
	#undef BPS_SRC_INDEX_XY
	#undef BPS_DOMAIN_INDEX
}

// Non-axis pass 2: copy source, then overwrite via inverse domain lookup.
__global__ void BitonicApplyDomainKernel(
	const float4         *srcTex,
	float4               *dstTex,
	const unsigned int   *domain,
	int                   srcPitch,
	int                   dstPitch,
	int                   width,
	int                   height,
	int                   inputOriginX,
	int                   inputOriginY,
	int                   outputOriginX,
	int                   outputOriginY,
	int                   outputWidth,
	int                   outputHeight,
	int                   mode,
	int                   lineCount,
	int                   freePMin,
	int                   freeQMin,
	int                   freeLineLength,
	int                   radialLength,
	int                   domainStride,
	float                 angleCos,
	float                 angleSin,
	float                 centerX,
	float                 centerY,
	float                 swirlK,
	int                   swirlLineMin,
	int                   pathDirection,
	int                   pathSMin,
	int                   pathNMin,
	int                   pathSampleCount,
	const BpsPathSampleGpu *pathSamples)
{
	const int ox = (int)(blockIdx.x * blockDim.x + threadIdx.x);
	const int oy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
	if (ox >= outputWidth || oy >= outputHeight) {
		return;
	}

	const int x = outputOriginX + ox;
	const int y = outputOriginY + oy;
	const unsigned int dstIndex = (unsigned int)(ox + oy * dstPitch);

	float4 pixel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (x >= inputOriginX && y >= inputOriginY &&
		x < inputOriginX + width && y < inputOriginY + height) {
		const unsigned int srcIndex =
			(unsigned int)((x - inputOriginX) + (y - inputOriginY) * srcPitch);
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
		const unsigned int srcIndex = domain[(unsigned int)(line * domainStride + pos)];
		if (srcIndex != 0xffffffffu) {
			pixel = srcTex[srcIndex];
		}
	}

	dstTex[dstIndex] = pixel;
}

__global__ void BitonicCopyInputKernel(
	const float4 *srcTex,
	float4       *dstTex,
	int           srcPitch,
	int           dstPitch,
	int           width,
	int           height,
	int           inputOriginX,
	int           inputOriginY,
	int           outputOriginX,
	int           outputOriginY,
	int           outputWidth,
	int           outputHeight)
{
	const int ox = (int)(blockIdx.x * blockDim.x + threadIdx.x);
	const int oy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
	if (ox >= outputWidth || oy >= outputHeight) {
		return;
	}

	const int x = outputOriginX + ox;
	const int y = outputOriginY + oy;
	const unsigned int dstIndex = (unsigned int)(ox + oy * dstPitch);
	float4 pixel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (x >= inputOriginX && y >= inputOriginY &&
		x < inputOriginX + width && y < inputOriginY + height) {
		pixel = srcTex[(unsigned int)((x - inputOriginX) + (y - inputOriginY) * srcPitch)];
	}
	dstTex[dstIndex] = pixel;
}

__global__ void BitonicSortMappedKernel(
	const float4 *srcTex,
	float4       *dstTex,
	unsigned int *domain,
	float        *keys,
	const BpsMappedPixelRecordGpu *records,
	const unsigned int *lineOffsets,
	const unsigned int *workOffsets,
	int           srcPitch,
	int           dstPitch,
	int           width,
	int           height,
	int           inputOriginX,
	int           inputOriginY,
	int           outputOriginX,
	int           outputOriginY,
	int           outputWidth,
	int           outputHeight,
	int           ordering,
	int           criterion,
	int           trigger,
	int           affect,
	float         cycleDegrees,
	float         thresholdMin,
	float         thresholdMax,
	int           lineCount)
{
	__shared__ unsigned int s_spanStart;
	__shared__ unsigned int s_spanEnd;
	__shared__ unsigned int s_spanSize;
	__shared__ unsigned int s_sortSize;

	const unsigned int line = blockIdx.x;
	const unsigned int gtid = threadIdx.x;
	if ((int)line >= lineCount) {
		return;
	}

	const unsigned int begin = lineOffsets[line];
	const unsigned int end = lineOffsets[line + 1u];
	const unsigned int lineSize = end - begin;
	if (lineSize == 0u) {
		return;
	}
	const unsigned int workBase = workOffsets[line];
	const bool ascending = ordering != 0;

	#define BPS_SRC_IN_WORLD(x, y) ((x) >= inputOriginX && (y) >= inputOriginY && \
									(x) < inputOriginX + width && (y) < inputOriginY + height)
	#define BPS_DST_IN_WORLD(x, y) ((x) >= outputOriginX && (y) >= outputOriginY && \
									(x) < outputOriginX + outputWidth && (y) < outputOriginY + outputHeight)
	#define BPS_SRC_INDEX_XY(x, y) ((unsigned int)(((x) - inputOriginX) + ((y) - inputOriginY) * srcPitch))
	#define BPS_DST_INDEX_XY(x, y) ((unsigned int)(((x) - outputOriginX) + ((y) - outputOriginY) * dstPitch))

	for (unsigned int i = gtid; i < lineSize; i += MAX_THREADS) {
		const unsigned int pixelIndex = records[begin + i].pixelIndex;
		const int x = (int)(pixelIndex % (unsigned int)width);
		const int y = (int)(pixelIndex / (unsigned int)width);
		domain[workBase + i] = BPS_SRC_IN_WORLD(x, y)
			? BPS_SRC_INDEX_XY(x, y)
			: 0xffffffffu;
	}
	__syncthreads();

	unsigned int cursor = 0u;
	while (cursor < lineSize) {
		if (gtid == 0u) {
			unsigned int runStart = cursor;
			while (runStart < lineSize) {
				const unsigned int srcIndex = domain[workBase + runStart];
				if (srcIndex != 0xffffffffu) {
					const float br = bps_sort_key(srcTex[srcIndex], trigger);
					if (bps_is_affected(br, thresholdMin, thresholdMax, affect)) break;
				}
				runStart++;
			}
			unsigned int runEnd = runStart;
			while (runEnd < lineSize) {
				const unsigned int srcIndex = domain[workBase + runEnd];
				if (srcIndex == 0xffffffffu) break;
				const float br = bps_sort_key(srcTex[srcIndex], trigger);
				if (!bps_is_affected(br, thresholdMin, thresholdMax, affect)) break;
				runEnd++;
			}
			s_spanStart = runStart;
			s_spanEnd = runEnd;
			s_spanSize = runEnd - runStart;
			s_sortSize = bps_next_pow2(s_spanSize);
		}
		__syncthreads();

		const unsigned int runStart = s_spanStart;
		const unsigned int runEnd = s_spanEnd;
		const unsigned int spanSize = s_spanSize;
		const unsigned int sortSize = s_sortSize;
		if (runStart >= lineSize || spanSize == 0u) {
			break;
		}

		const unsigned int sortBase = workBase + lineSize;
		for (unsigned int i = gtid; i < sortSize; i += MAX_THREADS) {
			if (i < spanSize) {
				const unsigned int srcIndex = domain[workBase + runStart + i];
				domain[sortBase + i] = srcIndex;
				keys[sortBase + i] = bps_sort_key(srcTex[srcIndex], criterion);
			} else {
				domain[sortBase + i] = 0xffffffffu;
				keys[sortBase + i] = ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX;
			}
		}
		__syncthreads();

		for (unsigned int k = 2u; k <= sortSize; k <<= 1) {
			for (unsigned int j = k >> 1; j > 0u; j >>= 1) {
				for (unsigned int i = gtid; i < sortSize; i += MAX_THREADS) {
					const unsigned int partner = i ^ j;
					if (partner > i) {
						bool stageAscending = (i & k) == 0u;
						if (!ascending) stageAscending = !stageAscending;
						const float keyA = keys[sortBase + i];
						const float keyB = keys[sortBase + partner];
						const unsigned int indexA = domain[sortBase + i];
						const unsigned int indexB = domain[sortBase + partner];
						const bool before = bps_before(keyA, indexA, keyB, indexB);
						if (before != stageAscending) {
							keys[sortBase + i] = keyB;
							keys[sortBase + partner] = keyA;
							domain[sortBase + i] = indexB;
							domain[sortBase + partner] = indexA;
						}
					}
				}
				__syncthreads();
			}
		}

		const unsigned int shift = bps_cycle_shift(spanSize, cycleDegrees);
		for (unsigned int i = gtid; i < spanSize; i += MAX_THREADS) {
			const unsigned int sortedIndex = (i + spanSize - shift) % spanSize;
			const unsigned int srcIndex = domain[sortBase + sortedIndex];
			const unsigned int pixelIndex = records[begin + runStart + i].pixelIndex;
			const int x = (int)(pixelIndex % (unsigned int)width);
			const int y = (int)(pixelIndex / (unsigned int)width);
			if (srcIndex != 0xffffffffu && BPS_DST_IN_WORLD(x, y)) {
				dstTex[BPS_DST_INDEX_XY(x, y)] = srcTex[srcIndex];
			}
		}
		__syncthreads();

		cursor = runEnd + 1u;
	}

	#undef BPS_SRC_IN_WORLD
	#undef BPS_DST_IN_WORLD
	#undef BPS_SRC_INDEX_XY
	#undef BPS_DST_INDEX_XY
}

// Host launch wrapper, called from SmartRenderGPU (BitonicPixelSorter_GPU.cpp).
// extern "C" to keep a stable, unmangled symbol across the nvcc/MSVC boundary.
//
// Follow SDK_Invert_ProcAmp: launch on the default stream and synchronise with
// cudaDeviceSynchronize(). The host-provided CUstream in command_queuePV is not
// passed through the CUDA runtime launch API in the Adobe sample.
extern "C" cudaError_t BitonicSort_CUDA(
	const void *src,
	void       *dst,
	int         srcPitch,
	int         dstPitch,
	int         width,
	int         height,
	int         inputOriginX,
	int         inputOriginY,
	int         outputOriginX,
	int         outputOriginY,
	int         outputWidth,
	int         outputHeight,
	int         mode,
	int         direction,
	int         ordering,
	int         criterion,
	int         trigger,
	int         affect,
	float       cycleDegrees,
	int         lineCount,
	int         freePMin,
	int         freeQMin,
	int         freeLineLength,
	int         radialLength,
	int         domainStride,
	float       thresholdMin,
	float       thresholdMax,
	float       angleCos,
	float       angleSin,
	float       centerX,
	float       centerY,
	float       swirlK,
	int         swirlLineMin,
	int         pathDirection,
	int         pathSMin,
	int         pathNMin,
	int         pathSampleCount,
	const void *pathSamplesHost,
	int         mappedRecordCount,
	int         mappedWorkItemCount,
	const void *mappedRecordsHost,
	const void *mappedLineOffsetsHost,
	const void *mappedWorkOffsetsHost)
{
	if (lineCount <= 0 && mode != BPS_MODE_PATH) {
		return cudaSuccess;
	}

	BpsPathSampleGpu *pathSamplesDev = nullptr;
	if (pathSampleCount > 0 && pathSamplesHost) {
		const size_t pathBytes =
			(size_t)pathSampleCount * sizeof(BpsPathSampleGpu);
		cudaError_t path_result =
			cudaMalloc((void **)&pathSamplesDev, pathBytes);
		if (path_result != cudaSuccess) {
			return path_result;
		}
		path_result = cudaMemcpy(pathSamplesDev, pathSamplesHost, pathBytes,
								 cudaMemcpyHostToDevice);
		if (path_result != cudaSuccess) {
			cudaFree(pathSamplesDev);
			return path_result;
		}
	}

	if (mode == BPS_MODE_AXIS) {
		BitonicSortKernel<<<lineCount, MAX_THREADS, 0>>>(
			(const float4 *)src, (float4 *)dst,
			srcPitch, dstPitch, width, height,
			inputOriginX, inputOriginY, outputOriginX, outputOriginY, outputWidth, outputHeight,
			mode, direction, ordering, criterion, trigger, affect, cycleDegrees, lineCount,
			freePMin, freeQMin, freeLineLength, radialLength,
			thresholdMin, thresholdMax, angleCos, angleSin, centerX, centerY,
			swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
			pathSampleCount, pathSamplesDev);

		cudaError_t launch_result = cudaPeekAtLastError();
		if (launch_result != cudaSuccess) {
			cudaFree(pathSamplesDev);
			return launch_result;
		}
		launch_result = cudaDeviceSynchronize();
		cudaFree(pathSamplesDev);
		return launch_result;
	}

	if (mode == BPS_MODE_PATH) {
		dim3 copyBlock(16, 16, 1);
		dim3 copyGrid(
			(unsigned int)((outputWidth + 15) / 16),
			(unsigned int)((outputHeight + 15) / 16),
			1u);
		BitonicCopyInputKernel<<<copyGrid, copyBlock, 0>>>(
			(const float4 *)src, (float4 *)dst,
			srcPitch, dstPitch, width, height,
			inputOriginX, inputOriginY, outputOriginX, outputOriginY,
			outputWidth, outputHeight);
		cudaError_t result = cudaPeekAtLastError();
		if (result != cudaSuccess) {
			cudaFree(pathSamplesDev);
			return result;
		}
		result = cudaDeviceSynchronize();
		if (result != cudaSuccess) {
			cudaFree(pathSamplesDev);
			return result;
		}

		if (mappedRecordCount <= 0 || mappedWorkItemCount <= 0 ||
			!mappedRecordsHost || !mappedLineOffsetsHost || !mappedWorkOffsetsHost) {
			cudaFree(pathSamplesDev);
			return cudaSuccess;
		}

		BpsMappedPixelRecordGpu *recordsDev = nullptr;
		unsigned int *lineOffsetsDev = nullptr;
		unsigned int *workOffsetsDev = nullptr;
		unsigned int *domain = nullptr;
		float *keys = nullptr;
		const size_t recordsBytes =
			(size_t)mappedRecordCount * sizeof(BpsMappedPixelRecordGpu);
		const size_t offsetsBytes =
			(size_t)(lineCount + 1) * sizeof(unsigned int);
		const size_t workUintBytes =
			(size_t)mappedWorkItemCount * sizeof(unsigned int);
		const size_t workFloatBytes =
			(size_t)mappedWorkItemCount * sizeof(float);

		result = cudaMalloc((void **)&recordsDev, recordsBytes);
		if (result == cudaSuccess) result = cudaMalloc((void **)&lineOffsetsDev, offsetsBytes);
		if (result == cudaSuccess) result = cudaMalloc((void **)&workOffsetsDev, offsetsBytes);
		if (result == cudaSuccess) result = cudaMalloc((void **)&domain, workUintBytes);
		if (result == cudaSuccess) result = cudaMalloc((void **)&keys, workFloatBytes);
		if (result == cudaSuccess) {
			result = cudaMemcpy(recordsDev, mappedRecordsHost, recordsBytes,
								cudaMemcpyHostToDevice);
		}
		if (result == cudaSuccess) {
			result = cudaMemcpy(lineOffsetsDev, mappedLineOffsetsHost, offsetsBytes,
								cudaMemcpyHostToDevice);
		}
		if (result == cudaSuccess) {
			result = cudaMemcpy(workOffsetsDev, mappedWorkOffsetsHost, offsetsBytes,
								cudaMemcpyHostToDevice);
		}
		if (result != cudaSuccess) {
			cudaFree(keys);
			cudaFree(domain);
			cudaFree(workOffsetsDev);
			cudaFree(lineOffsetsDev);
			cudaFree(recordsDev);
			cudaFree(pathSamplesDev);
			return result;
		}

		BitonicSortMappedKernel<<<lineCount, MAX_THREADS, 0>>>(
			(const float4 *)src, (float4 *)dst, domain, keys,
			recordsDev, lineOffsetsDev, workOffsetsDev,
			srcPitch, dstPitch, width, height,
			inputOriginX, inputOriginY, outputOriginX, outputOriginY,
			outputWidth, outputHeight,
			ordering, criterion, trigger, affect, cycleDegrees,
			thresholdMin, thresholdMax, lineCount);

		result = cudaPeekAtLastError();
		if (result == cudaSuccess) {
			result = cudaDeviceSynchronize();
		}
		cudaFree(keys);
		cudaFree(domain);
		cudaFree(workOffsetsDev);
		cudaFree(lineOffsetsDev);
		cudaFree(recordsDev);
		cudaFree(pathSamplesDev);
		return result;
	}

	if (domainStride <= 0 || outputWidth <= 0 || outputHeight <= 0) {
		cudaFree(pathSamplesDev);
		return cudaSuccess;
	}

	// Pad stride to the next power of two so bitonic padding never collides
	// with live samples on the longest path.
	int paddedStride = domainStride;
	if (paddedStride > 1) {
		unsigned int v = (unsigned int)paddedStride - 1u;
		v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
		paddedStride = (int)(v + 1u);
	}

	const size_t domainCount = (size_t)lineCount * (size_t)paddedStride;
	const size_t domainBytes = domainCount * sizeof(unsigned int);
	const size_t keysBytes = domainCount * sizeof(float);
	unsigned int *domain = nullptr;
	float *keys = nullptr;
	cudaError_t result = cudaMalloc((void **)&domain, domainBytes);
	if (result != cudaSuccess) {
		cudaFree(pathSamplesDev);
		return result;
	}
	result = cudaMalloc((void **)&keys, keysBytes);
	if (result != cudaSuccess) {
		cudaFree(domain);
		cudaFree(pathSamplesDev);
		return result;
	}

	result = cudaMemset(domain, 0xFF, domainBytes);
	if (result != cudaSuccess) {
		cudaFree(keys);
		cudaFree(domain);
		cudaFree(pathSamplesDev);
		return result;
	}

	BitonicSortDomainKernel<<<lineCount, MAX_THREADS, 0>>>(
		(const float4 *)src, domain, keys,
		srcPitch, width, height,
		inputOriginX, inputOriginY,
		mode, ordering, criterion, trigger, affect, cycleDegrees, lineCount,
		freePMin, freeQMin, freeLineLength, radialLength, paddedStride,
		thresholdMin, thresholdMax, angleCos, angleSin, centerX, centerY,
		swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
		pathSampleCount, pathSamplesDev);

	result = cudaPeekAtLastError();
	if (result != cudaSuccess) {
		cudaFree(keys);
		cudaFree(domain);
		cudaFree(pathSamplesDev);
		return result;
	}

	result = cudaDeviceSynchronize();
	if (result != cudaSuccess) {
		cudaFree(keys);
		cudaFree(domain);
		cudaFree(pathSamplesDev);
		return result;
	}

	dim3 applyBlock(16, 16, 1);
	dim3 applyGrid(
		(unsigned int)((outputWidth + 15) / 16),
		(unsigned int)((outputHeight + 15) / 16),
		1u);
	BitonicApplyDomainKernel<<<applyGrid, applyBlock, 0>>>(
		(const float4 *)src, (float4 *)dst, domain,
		srcPitch, dstPitch, width, height,
		inputOriginX, inputOriginY, outputOriginX, outputOriginY, outputWidth, outputHeight,
		mode, lineCount,
		freePMin, freeQMin, freeLineLength, radialLength, paddedStride,
		angleCos, angleSin, centerX, centerY,
		swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
		pathSampleCount, pathSamplesDev);

	result = cudaPeekAtLastError();
	if (result != cudaSuccess) {
		cudaFree(keys);
		cudaFree(domain);
		cudaFree(pathSamplesDev);
		return result;
	}

	result = cudaDeviceSynchronize();
	cudaFree(keys);
	cudaFree(domain);
	cudaFree(pathSamplesDev);
	return result;
}
