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

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

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
#define BPS_MODE_USES_MAPPED_SORT(m) \
	((m) >= BPS_MODE_FREE_ANGLE && (m) <= BPS_MODE_PATH)

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

struct BpsCudaPathMap {
	unsigned long long key = 0ull;
	int device = -1;
	int width = 0;
	int height = 0;
	int lineCount = 0;
	int mappedRecordCount = 0;
	int mappedWorkItemCount = 0;
	BpsMappedPixelRecordGpu *records = nullptr;
	unsigned int *lineOffsets = nullptr;
	unsigned int *workOffsets = nullptr;
};

namespace {
std::mutex g_cudaPathMapMutex;
BpsCudaPathMap g_cudaPathMap;

struct BpsCudaDomainPool {
	unsigned int *domain = nullptr;
	float *keys = nullptr;
	size_t capacityBytes = 0;
};

static BpsCudaDomainPool g_cudaDomainPool;

static cudaError_t bps_acquire_cuda_domain_buffers(
	size_t domainBytes,
	size_t keysBytes,
	unsigned int **domainOut,
	float **keysOut)
{
	const size_t needed = domainBytes + keysBytes;
	if (g_cudaDomainPool.capacityBytes < needed) {
		cudaFree(g_cudaDomainPool.keys);
		cudaFree(g_cudaDomainPool.domain);
		g_cudaDomainPool.domain = nullptr;
		g_cudaDomainPool.keys = nullptr;
		g_cudaDomainPool.capacityBytes = 0;

		cudaError_t result = cudaMalloc((void **)&g_cudaDomainPool.domain, domainBytes);
		if (result != cudaSuccess) {
			return result;
		}
		result = cudaMalloc((void **)&g_cudaDomainPool.keys, keysBytes);
		if (result != cudaSuccess) {
			cudaFree(g_cudaDomainPool.domain);
			g_cudaDomainPool.domain = nullptr;
			return result;
		}
		g_cudaDomainPool.capacityBytes = needed;
	}

	*domainOut = g_cudaDomainPool.domain;
	*keysOut = g_cudaDomainPool.keys;
	return cudaSuccess;
}
}

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

struct BpsKeySource {
	const float4 *tex;
	int pitch;
	int originX;
	int originY;
	int width;
	int height;
};

__device__ __forceinline__ bool bps_key_in_world(const BpsKeySource &src, int x, int y)
{
	return x >= src.originX && y >= src.originY &&
		   x < src.originX + src.width && y < src.originY + src.height;
}

__device__ __forceinline__ float bps_sample_key(const BpsKeySource &src, int x, int y, int keyCriterion)
{
	if (!bps_key_in_world(src, x, y)) {
		return -1.0f;
	}
	const unsigned int idx =
		(unsigned int)((x - src.originX) + (y - src.originY) * src.pitch);
	return bps_sort_key(src.tex[idx], keyCriterion);
}

__device__ __forceinline__ bool bps_unified_keys(
	const BpsKeySource &criterionSrc,
	const BpsKeySource &triggerSrc,
	int criterion,
	int trigger)
{
	return criterion == trigger &&
		   criterionSrc.tex == triggerSrc.tex &&
		   criterionSrc.pitch == triggerSrc.pitch &&
		   criterionSrc.originX == triggerSrc.originX &&
		   criterionSrc.originY == triggerSrc.originY &&
		   criterionSrc.width == triggerSrc.width &&
		   criterionSrc.height == triggerSrc.height;
}

__device__ __forceinline__ float bps_sample_trigger_key(
	const BpsKeySource &criterionSrc,
	const BpsKeySource &triggerSrc,
	int x,
	int y,
	int criterion,
	int trigger)
{
	if (bps_unified_keys(criterionSrc, triggerSrc, criterion, trigger)) {
		return bps_sample_key(criterionSrc, x, y, criterion);
	}
	return bps_sample_key(triggerSrc, x, y, trigger);
}

// Recover layer coordinates from a source-world linear index.
__device__ __forceinline__ float bps_sample_key_from_src_index(
	const BpsKeySource &src,
	int srcPitch,
	int inputOriginX,
	int inputOriginY,
	unsigned int srcIndex,
	int keyCriterion)
{
	if (srcIndex == 0xffffffffu) {
		return -1.0f;
	}
	const int x = (int)(srcIndex % (unsigned int)srcPitch) + inputOriginX;
	const int y = (int)(srcIndex / (unsigned int)srcPitch) + inputOriginY;
	return bps_sample_key(src, x, y, keyCriterion);
}

__device__ __forceinline__ float bps_sample_trigger_key_from_src_index(
	const BpsKeySource &criterionSrc,
	const BpsKeySource &triggerSrc,
	int srcPitch,
	int inputOriginX,
	int inputOriginY,
	unsigned int srcIndex,
	int criterion,
	int trigger)
{
	if (bps_unified_keys(criterionSrc, triggerSrc, criterion, trigger)) {
		return bps_sample_key_from_src_index(criterionSrc, srcPitch, inputOriginX,
											 inputOriginY, srcIndex, criterion);
	}
	return bps_sample_key_from_src_index(triggerSrc, srcPitch, inputOriginX,
										 inputOriginY, srcIndex, trigger);
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

static unsigned int bps_next_pow2_host(unsigned int value)
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

static void bps_release_cuda_path_map(BpsCudaPathMap *map)
{
	if (!map) return;
	cudaFree(map->workOffsets);
	cudaFree(map->lineOffsets);
	cudaFree(map->records);
	map->workOffsets = nullptr;
	map->lineOffsets = nullptr;
	map->records = nullptr;
	map->key = 0ull;
	map->device = -1;
	map->width = 0;
	map->height = 0;
	map->lineCount = 0;
	map->mappedRecordCount = 0;
	map->mappedWorkItemCount = 0;
}

extern "C" void BitonicClearCudaPathMapCache()
{
	std::lock_guard<std::mutex> lock(g_cudaPathMapMutex);
	bps_release_cuda_path_map(&g_cudaPathMap);
}

static int bps_cuda_jfa_grid_factor(int width, int height)
{
	const int maxDim = (std::max)(width, height);
	int factor = (maxDim + 450) / 900;
	if (factor < 1) factor = 1;
	if (factor > 3) factor = 3;
	return factor;
}

static void bps_normalise_tangent_host(float *tx, float *ty)
{
	const float len = std::sqrt((*tx) * (*tx) + (*ty) * (*ty));
	if (len > 1.0e-6f) {
		*tx /= len;
		*ty /= len;
	} else {
		*tx = 1.0f;
		*ty = 0.0f;
	}
}

static std::vector<BpsPathSampleGpu> bps_build_cuda_path_seeds(
	const BpsPathSampleGpu *samples,
	int sampleCount)
{
	std::vector<BpsPathSampleGpu> seeds;
	if (!samples || sampleCount < 2) {
		return seeds;
	}
	for (int i = 0; i < sampleCount - 1; ++i) {
		const BpsPathSampleGpu p0 = samples[i];
		const BpsPathSampleGpu p1 = samples[i + 1];
		const float dx = p1.x - p0.x;
		const float dy = p1.y - p0.y;
		const float len = std::sqrt(dx * dx + dy * dy);
		const int steps = (std::max)(1, (int)std::ceil(len));
		for (int k = (i > 0) ? 1 : 0; k <= steps; ++k) {
			const float t = (float)k / (float)steps;
			BpsPathSampleGpu seed;
			seed.x = p0.x + dx * t;
			seed.y = p0.y + dy * t;
			seed.s = p0.s + (p1.s - p0.s) * t;
			seed.tx = p0.tx + (p1.tx - p0.tx) * t;
			seed.ty = p0.ty + (p1.ty - p0.ty) * t;
			bps_normalise_tangent_host(&seed.tx, &seed.ty);
			seeds.push_back(seed);
		}
	}
	return seeds;
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

__device__ __forceinline__ float bps_wrap_arc_length(float s, float length)
{
	if (length <= 1.0e-6f) return s;
	float w = fmodf(s, length);
	if (w < 0.0f) w += length;
	return w;
}

__device__ __forceinline__ bool bps_path_lane_order(
	int pathDirection,
	int pathClosed,
	float pathLength,
	int pathSMin,
	int pathNMin,
	int lineCount,
	float s,
	float n,
	int *line,
	float *order)
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
	const float4 *criterionTex,
	const float4 *triggerTex,
	int           criterionPitch,
	int           criterionOriginX,
	int           criterionOriginY,
	int           criterionWidth,
	int           criterionHeight,
	int           triggerPitch,
	int           triggerOriginX,
	int           triggerOriginY,
	int           triggerWidth,
	int           triggerHeight,
	int           srcPitch,
	int           dstPitch,
	int           width,
	int           height,
	int           inputOriginX,
	int           inputOriginY,
	int           inputWidth,
	int           inputHeight,
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


	const BpsKeySource criterionSrc = {
		criterionTex, criterionPitch, criterionOriginX, criterionOriginY,
		criterionWidth, criterionHeight};
	const BpsKeySource triggerSrc = {
		triggerTex, triggerPitch, triggerOriginX, triggerOriginY,
		triggerWidth, triggerHeight};
	const unsigned int gid  = blockIdx.x;
	const unsigned int gtid = threadIdx.x;

	const unsigned int size = bps_line_size(
		mode, direction, gid, width, height, freeLineLength, radialLength, 0, swirlK);
	if (size == 0u || size > MAX_SIZE) {
		return;
	}

	// Map layer coordinates into AE's possibly partial GPU worlds.
	#define BPS_SRC_IN_WORLD(x, y) ((x) >= inputOriginX && (y) >= inputOriginY && \
									(x) < inputOriginX + inputWidth && (y) < inputOriginY + inputHeight)
	#define BPS_DST_IN_WORLD(x, y) ((x) >= outputOriginX && (y) >= outputOriginY && \
									(x) < outputOriginX + outputWidth && (y) < outputOriginY + outputHeight)
	#define BPS_SRC_INDEX_XY(x, y) ((unsigned int)(((x) - inputOriginX) + ((y) - inputOriginY) * srcPitch))
	#define BPS_DST_INDEX_XY(x, y) ((unsigned int)(((x) - outputOriginX) + ((y) - outputOriginY) * dstPitch))

	// Always write every destination pixel. Partial input worlds (common with
	// alpha / adjustment layers) must not leave stale frame data behind.
	for (unsigned int pos = gtid; pos < size; pos += MAX_THREADS) {
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
				: make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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
					float br = bps_sample_trigger_key(
						criterionSrc, triggerSrc, x, y, criterion, trigger);
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
				float br = bps_sample_trigger_key(
					criterionSrc, triggerSrc, x, y, criterion, trigger);
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
				scratchKey[i] = valid ? bps_sample_key(criterionSrc, x, y, criterion) :
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
	const float4     *criterionTex,
	const float4     *triggerTex,
	int               criterionPitch,
	int               criterionOriginX,
	int               criterionOriginY,
	int               criterionWidth,
	int               criterionHeight,
	int               triggerPitch,
	int               triggerOriginX,
	int               triggerOriginY,
	int               triggerWidth,
	int               triggerHeight,
	unsigned int     *domain,
	float            *keys,
	int               srcPitch,
	int               width,
	int               height,
	int               inputOriginX,
	int               inputOriginY,
	int               inputWidth,
	int               inputHeight,
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


	const BpsKeySource criterionSrc = {
		criterionTex, criterionPitch, criterionOriginX, criterionOriginY,
		criterionWidth, criterionHeight};
	const BpsKeySource triggerSrc = {
		triggerTex, triggerPitch, triggerOriginX, triggerOriginY,
		triggerWidth, triggerHeight};
	const unsigned int gid  = blockIdx.x;
	const unsigned int gtid = threadIdx.x;

	const unsigned int size = bps_line_size(
		mode, 0, gid, width, height, freeLineLength, radialLength, domainStride, swirlK);
	if (size == 0u || domainStride <= 0 || (int)size > domainStride) {
		return;
	}

	#define BPS_SRC_IN_WORLD(x, y) ((x) >= inputOriginX && (y) >= inputOriginY && \
									(x) < inputOriginX + inputWidth && (y) < inputOriginY + inputHeight)
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
				const float br = bps_sample_trigger_key_from_src_index(
					criterionSrc, triggerSrc, srcPitch, inputOriginX, inputOriginY,
					srcIndex, criterion, trigger);
				if (bps_is_affected(br, thresholdMin, thresholdMax, affect)) break;
				runStart++;
			}

			unsigned int runEnd = runStart;
			while (runEnd < pathLen) {
				const unsigned int pos =
					__float_as_uint(keys[BPS_DOMAIN_INDEX(runEnd)]);
				const unsigned int srcIndex = domain[BPS_DOMAIN_INDEX(pos)];
				const float br = bps_sample_trigger_key_from_src_index(
					criterionSrc, triggerSrc, srcPitch, inputOriginX, inputOriginY,
					srcIndex, criterion, trigger);
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
				keys[work] = bps_sample_key_from_src_index(criterionSrc, srcPitch, inputOriginX, inputOriginY, srcIndex, criterion);
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
		// Flag reordered slots (run length >= 2) so ApplyDomain composites only
		// those over the source and leaves unsorted areas pristine.
		const unsigned int affectedBit = (spanSize >= 2u) ? 0x80000000u : 0u;
		for (unsigned int i = gtid; i < spanSize; i += MAX_THREADS) {
			const unsigned int sortedIndex = (i + spanSize - shift) % spanSize;
			const unsigned int pos =
				__float_as_uint(keys[BPS_DOMAIN_INDEX(runStart + i)]);
			const unsigned int srcIndex =
				domain[BPS_DOMAIN_INDEX(size + sortedIndex)];
			domain[BPS_DOMAIN_INDEX(pos)] = srcIndex | affectedBit;
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
	int                   inputWidth,
	int                   inputHeight,
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
		x < inputOriginX + inputWidth && y < inputOriginY + inputHeight) {
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
		const unsigned int raw = domain[(unsigned int)(line * domainStride + pos)];
		// High bit flags slots a sort reordered; composite only those over the
		// source so unsorted areas keep the exact original (no resample loss).
		if (raw != 0xffffffffu && (raw & 0x80000000u) != 0u) {
			pixel = srcTex[raw & 0x7fffffffu];
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
	int           inputWidth,
	int           inputHeight,
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
		x < inputOriginX + inputWidth && y < inputOriginY + inputHeight) {
		pixel = srcTex[(unsigned int)((x - inputOriginX) + (y - inputOriginY) * srcPitch)];
	}
	dstTex[dstIndex] = pixel;
}

__device__ __forceinline__ unsigned long long bps_pack_jfa_key(float d2, unsigned int seedIndex)
{
	return ((unsigned long long)__float_as_uint(d2) << 32) |
		(unsigned long long)seedIndex;
}

__device__ __forceinline__ unsigned int bps_jfa_seed_index(unsigned long long key)
{
	return (unsigned int)(key & 0xffffffffull);
}

__device__ __forceinline__ bool bps_jfa_key_valid(unsigned long long key)
{
	return bps_jfa_seed_index(key) != 0xffffffffu;
}

__global__ void CudaPathSeedKernel(
	const BpsPathSampleGpu *seeds,
	int seedCount,
	unsigned long long *field,
	int gridW,
	int gridH,
	float scale)
{
	const unsigned int seedIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if ((int)seedIndex >= seedCount) return;
	const BpsPathSampleGpu seed = seeds[seedIndex];
	int cx = bps_round_to_int(seed.x / scale);
	int cy = bps_round_to_int(seed.y / scale);
	// Open-end extensions meet the frame edge; keep boundary seeds.
	if (cx < 0) cx = 0;
	else if (cx >= gridW) cx = gridW - 1;
	if (cy < 0) cy = 0;
	else if (cy >= gridH) cy = gridH - 1;
	const float ax = (float)cx * scale;
	const float ay = (float)cy * scale;
	const float dx = ax - seed.x;
	const float dy = ay - seed.y;
	const float d2 = dx * dx + dy * dy;
	const unsigned long long key = bps_pack_jfa_key(d2, seedIndex);
	atomicMin(&field[(unsigned int)(cy * gridW + cx)], key);
}

__global__ void CudaPathJfaPassKernel(
	const BpsPathSampleGpu *seeds,
	const unsigned long long *src,
	unsigned long long *dst,
	int gridW,
	int gridH,
	int step,
	float scale)
{
	static const int kDX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
	static const int kDY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
	const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
	const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= gridW || y >= gridH) return;

	const unsigned int idx = (unsigned int)(y * gridW + x);
	const float ax = (float)x * scale;
	const float ay = (float)y * scale;
	unsigned long long best = src[idx];

	for (int k = 0; k < 8; ++k) {
		const int nx = x + kDX[k] * step;
		const int ny = y + kDY[k] * step;
		if (nx < 0 || ny < 0 || nx >= gridW || ny >= gridH) continue;
		const unsigned long long nkey = src[(unsigned int)(ny * gridW + nx)];
		if (!bps_jfa_key_valid(nkey)) continue;
		const unsigned int seedIndex = bps_jfa_seed_index(nkey);
		const BpsPathSampleGpu seed = seeds[seedIndex];
		const float dx = ax - seed.x;
		const float dy = ay - seed.y;
		const unsigned long long cand =
			bps_pack_jfa_key(dx * dx + dy * dy, seedIndex);
		if (cand < best) {
			best = cand;
		}
	}
	dst[idx] = best;
}

__global__ void CudaPathClassifyCountKernel(
	const BpsPathSampleGpu *seeds,
	const unsigned long long *field,
	int gridW,
	int gridH,
	int factor,
	int width,
	int height,
	int lineCount,
	int pathDirection,
	int pathClosed,
	float pathLength,
	int pathSMin,
	int pathNMin,
	int *laneOf,
	float *keyOf,
	unsigned int *laneCounts)
{
	const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
	const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	const unsigned int pidx = (unsigned int)(y * width + x);
	laneOf[pidx] = -1;

	const int cx = min(x / factor, gridW - 1);
	const int cy = min(y / factor, gridH - 1);
	const unsigned long long key = field[(unsigned int)(cy * gridW + cx)];
	if (!bps_jfa_key_valid(key)) return;

	const BpsPathSampleGpu seed = seeds[bps_jfa_seed_index(key)];
	const float dx = (float)x - seed.x;
	const float dy = (float)y - seed.y;
	const float n = -seed.ty * dx + seed.tx * dy;
	int lane = 0;
	float order = 0.0f;
	if (!bps_path_lane_order(pathDirection, pathClosed, pathLength,
							 pathSMin, pathNMin, lineCount,
							 seed.s, n, &lane, &order)) {
		return;
	}
	laneOf[pidx] = lane;
	keyOf[pidx] = order;
	atomicAdd(&laneCounts[lane], 1u);
}

__global__ void CudaPathScatterRecordsKernel(
	const int *laneOf,
	const float *keyOf,
	unsigned int *laneCursors,
	BpsMappedPixelRecordGpu *records,
	int width,
	int height)
{
	const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
	const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	const unsigned int pidx = (unsigned int)(y * width + x);
	const int lane = laneOf[pidx];
	if (lane < 0) return;
	const unsigned int dst = atomicAdd(&laneCursors[lane], 1u);
	records[dst].posKey = keyOf[pidx];
	records[dst].pixelIndex = pidx;
}

__device__ __forceinline__ bool bps_record_before(
	const BpsMappedPixelRecordGpu &a,
	const BpsMappedPixelRecordGpu &b)
{
	if (a.posKey < b.posKey) return true;
	if (a.posKey > b.posKey) return false;
	return a.pixelIndex < b.pixelIndex;
}

__global__ void CudaPathSortRecordsKernel(
	BpsMappedPixelRecordGpu *records,
	BpsMappedPixelRecordGpu *workRecords,
	const unsigned int *lineOffsets,
	const unsigned int *workOffsets,
	int lineCount)
{
	const unsigned int line = blockIdx.x;
	const unsigned int gtid = threadIdx.x;
	if ((int)line >= lineCount) return;
	const unsigned int begin = lineOffsets[line];
	const unsigned int end = lineOffsets[line + 1u];
	const unsigned int lineSize = end - begin;
	if (lineSize <= 1u) return;

	const unsigned int sortSize = bps_next_pow2(lineSize);
	const unsigned int sortBase = workOffsets[line] + lineSize;
	for (unsigned int i = gtid; i < sortSize; i += MAX_THREADS) {
		BpsMappedPixelRecordGpu record;
		if (i < lineSize) {
			record = records[begin + i];
		} else {
			record.posKey = BPS_FLOAT_MAX;
			record.pixelIndex = 0xffffffffu;
		}
		workRecords[sortBase + i] = record;
	}
	__syncthreads();

	for (unsigned int k = 2u; k <= sortSize; k <<= 1) {
		for (unsigned int j = k >> 1; j > 0u; j >>= 1) {
			for (unsigned int i = gtid; i < sortSize; i += MAX_THREADS) {
				const unsigned int partner = i ^ j;
				if (partner > i) {
					const unsigned int slotA = sortBase + i;
					const unsigned int slotB = sortBase + partner;
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
			__syncthreads();
		}
	}

	for (unsigned int i = gtid; i < lineSize; i += MAX_THREADS) {
		records[begin + i] = workRecords[sortBase + i];
	}
}

__global__ void BitonicSortMappedKernel(
	const float4 *srcTex,
	float4       *dstTex,
	const float4 *criterionTex,
	const float4 *triggerTex,
	int           criterionPitch,
	int           criterionOriginX,
	int           criterionOriginY,
	int           criterionWidth,
	int           criterionHeight,
	int           triggerPitch,
	int           triggerOriginX,
	int           triggerOriginY,
	int           triggerWidth,
	int           triggerHeight,
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
	int           inputWidth,
	int           inputHeight,
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


	const BpsKeySource criterionSrc = {
		criterionTex, criterionPitch, criterionOriginX, criterionOriginY,
		criterionWidth, criterionHeight};
	const BpsKeySource triggerSrc = {
		triggerTex, triggerPitch, triggerOriginX, triggerOriginY,
		triggerWidth, triggerHeight};
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
									(x) < inputOriginX + inputWidth && (y) < inputOriginY + inputHeight)
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
					const float br = bps_sample_trigger_key_from_src_index(
					criterionSrc, triggerSrc, srcPitch, inputOriginX, inputOriginY,
					srcIndex, criterion, trigger);
					if (bps_is_affected(br, thresholdMin, thresholdMax, affect)) break;
				}
				runStart++;
			}
			unsigned int runEnd = runStart;
			while (runEnd < lineSize) {
				const unsigned int srcIndex = domain[workBase + runEnd];
				if (srcIndex == 0xffffffffu) break;
				const float br = bps_sample_trigger_key_from_src_index(
					criterionSrc, triggerSrc, srcPitch, inputOriginX, inputOriginY,
					srcIndex, criterion, trigger);
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
				keys[sortBase + i] = bps_sample_key_from_src_index(criterionSrc, srcPitch, inputOriginX, inputOriginY, srcIndex, criterion);
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

static cudaError_t bps_build_cuda_path_map(
	int width,
	int height,
	int lineCount,
	int pathDirection,
	int pathClosed,
	float pathLength,
	int pathSMin,
	int pathNMin,
	int pathSampleCount,
	const void *pathSamplesHost,
	unsigned long long pathMapKey,
	BpsCudaPathMap *cache)
{
	if (!cache || width <= 0 || height <= 0 || lineCount <= 0 ||
		pathSampleCount < 2 || !pathSamplesHost || pathMapKey == 0ull) {
		return cudaErrorInvalidValue;
	}

	int device = -1;
	cudaError_t result = cudaGetDevice(&device);
	if (result != cudaSuccess) return result;

	if (cache->records &&
		cache->key == pathMapKey &&
		cache->device == device &&
		cache->width == width &&
		cache->height == height &&
		cache->lineCount == lineCount) {
		return cudaSuccess;
	}

	bps_release_cuda_path_map(cache);

	const BpsPathSampleGpu *pathSamples =
		static_cast<const BpsPathSampleGpu *>(pathSamplesHost);
	std::vector<BpsPathSampleGpu> seeds =
		bps_build_cuda_path_seeds(pathSamples, pathSampleCount);
	if (seeds.empty() || seeds.size() > 0xfffffffeull) {
		return cudaErrorInvalidValue;
	}

	const int factor = bps_cuda_jfa_grid_factor(width, height);
	const int gridW = (width + factor - 1) / factor;
	const int gridH = (height + factor - 1) / factor;
	const size_t gridCount = (size_t)gridW * (size_t)gridH;
	const size_t pixelCount = (size_t)width * (size_t)height;
	const size_t offsetsCount = (size_t)lineCount + 1u;

	BpsPathSampleGpu *seedsDev = nullptr;
	unsigned long long *fieldA = nullptr;
	unsigned long long *fieldB = nullptr;
	int *laneOf = nullptr;
	float *keyOf = nullptr;
	unsigned int *laneCountsDev = nullptr;
	unsigned int *lineOffsetsDev = nullptr;
	unsigned int *workOffsetsDev = nullptr;
	unsigned int *laneCursorsDev = nullptr;
	BpsMappedPixelRecordGpu *recordsDev = nullptr;
	BpsMappedPixelRecordGpu *workRecordsDev = nullptr;

	const unsigned long long invalidKey = 0xffffffffffffffffull;
	result = cudaMalloc((void **)&seedsDev, seeds.size() * sizeof(BpsPathSampleGpu));
	if (result == cudaSuccess) {
		result = cudaMemcpy(seedsDev, seeds.data(),
							seeds.size() * sizeof(BpsPathSampleGpu),
							cudaMemcpyHostToDevice);
	}
	if (result == cudaSuccess) result = cudaMalloc((void **)&fieldA, gridCount * sizeof(unsigned long long));
	if (result == cudaSuccess) result = cudaMalloc((void **)&fieldB, gridCount * sizeof(unsigned long long));
	if (result == cudaSuccess) result = cudaMalloc((void **)&laneOf, pixelCount * sizeof(int));
	if (result == cudaSuccess) result = cudaMalloc((void **)&keyOf, pixelCount * sizeof(float));
	if (result == cudaSuccess) result = cudaMalloc((void **)&laneCountsDev, (size_t)lineCount * sizeof(unsigned int));

	if (result != cudaSuccess) {
		cudaFree(laneCountsDev);
		cudaFree(keyOf);
		cudaFree(laneOf);
		cudaFree(fieldB);
		cudaFree(fieldA);
		cudaFree(seedsDev);
		return result;
	}

	result = cudaMemset(fieldA, 0xff, gridCount * sizeof(unsigned long long));
	if (result == cudaSuccess) {
		result = cudaMemset(laneCountsDev, 0, (size_t)lineCount * sizeof(unsigned int));
	}
	if (result != cudaSuccess) {
		cudaFree(laneCountsDev);
		cudaFree(keyOf);
		cudaFree(laneOf);
		cudaFree(fieldB);
		cudaFree(fieldA);
		cudaFree(seedsDev);
		return result;
	}
	(void)invalidKey;

	const unsigned int seedBlock = 256u;
	const unsigned int seedGrid =
		(unsigned int)((seeds.size() + seedBlock - 1u) / seedBlock);
	CudaPathSeedKernel<<<seedGrid, seedBlock>>>(
		seedsDev, (int)seeds.size(), fieldA, gridW, gridH, (float)factor);
	result = cudaPeekAtLastError();
	if (result == cudaSuccess) result = cudaDeviceSynchronize();
	if (result != cudaSuccess) {
		cudaFree(laneCountsDev);
		cudaFree(keyOf);
		cudaFree(laneOf);
		cudaFree(fieldB);
		cudaFree(fieldA);
		cudaFree(seedsDev);
		return result;
	}

	const dim3 jfaBlock(16, 16, 1);
	const dim3 jfaGrid(
		(unsigned int)((gridW + 15) / 16),
		(unsigned int)((gridH + 15) / 16),
		1u);
	int step = 1;
	const int gridMax = (std::max)(gridW, gridH);
	while (step < (gridMax + 1) / 2) {
		step <<= 1;
	}
	unsigned long long *srcField = fieldA;
	unsigned long long *dstField = fieldB;
	for (; step >= 1; step >>= 1) {
		CudaPathJfaPassKernel<<<jfaGrid, jfaBlock>>>(
			seedsDev, srcField, dstField, gridW, gridH, step, (float)factor);
		result = cudaPeekAtLastError();
		if (result == cudaSuccess) result = cudaDeviceSynchronize();
		if (result != cudaSuccess) {
			cudaFree(laneCountsDev);
			cudaFree(keyOf);
			cudaFree(laneOf);
			cudaFree(fieldB);
			cudaFree(fieldA);
			cudaFree(seedsDev);
			return result;
		}
		std::swap(srcField, dstField);
	}
	CudaPathJfaPassKernel<<<jfaGrid, jfaBlock>>>(
		seedsDev, srcField, dstField, gridW, gridH, 1, (float)factor);
	result = cudaPeekAtLastError();
	if (result == cudaSuccess) result = cudaDeviceSynchronize();
	if (result != cudaSuccess) {
		cudaFree(laneCountsDev);
		cudaFree(keyOf);
		cudaFree(laneOf);
		cudaFree(fieldB);
		cudaFree(fieldA);
		cudaFree(seedsDev);
		return result;
	}
	srcField = dstField;

	const dim3 classifyBlock(16, 16, 1);
	const dim3 classifyGrid(
		(unsigned int)((width + 15) / 16),
		(unsigned int)((height + 15) / 16),
		1u);
	CudaPathClassifyCountKernel<<<classifyGrid, classifyBlock>>>(
		seedsDev, srcField, gridW, gridH, factor, width, height, lineCount,
		pathDirection, pathClosed, pathLength, pathSMin, pathNMin,
		laneOf, keyOf, laneCountsDev);
	result = cudaPeekAtLastError();
	if (result == cudaSuccess) result = cudaDeviceSynchronize();
	if (result != cudaSuccess) {
		cudaFree(laneCountsDev);
		cudaFree(keyOf);
		cudaFree(laneOf);
		cudaFree(fieldB);
		cudaFree(fieldA);
		cudaFree(seedsDev);
		return result;
	}

	std::vector<unsigned int> laneCounts((size_t)lineCount);
	result = cudaMemcpy(laneCounts.data(), laneCountsDev,
						(size_t)lineCount * sizeof(unsigned int),
						cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) {
		cudaFree(laneCountsDev);
		cudaFree(keyOf);
		cudaFree(laneOf);
		cudaFree(fieldB);
		cudaFree(fieldA);
		cudaFree(seedsDev);
		return result;
	}

	std::vector<unsigned int> lineOffsets(offsetsCount, 0u);
	std::vector<unsigned int> workOffsets(offsetsCount, 0u);
	unsigned int recordCount = 0u;
	unsigned int workCount = 0u;
	for (int line = 0; line < lineCount; ++line) {
		lineOffsets[(size_t)line] = recordCount;
		workOffsets[(size_t)line] = workCount;
		const unsigned int len = laneCounts[(size_t)line];
		recordCount += len;
		workCount += len + bps_next_pow2_host(len);
	}
	lineOffsets[(size_t)lineCount] = recordCount;
	workOffsets[(size_t)lineCount] = workCount;

	if (recordCount == 0u || workCount == 0u) {
		cudaFree(laneCountsDev);
		cudaFree(keyOf);
		cudaFree(laneOf);
		cudaFree(fieldB);
		cudaFree(fieldA);
		cudaFree(seedsDev);
		return cudaSuccess;
	}

	result = cudaMalloc((void **)&lineOffsetsDev, offsetsCount * sizeof(unsigned int));
	if (result == cudaSuccess) result = cudaMalloc((void **)&workOffsetsDev, offsetsCount * sizeof(unsigned int));
	if (result == cudaSuccess) result = cudaMalloc((void **)&laneCursorsDev, offsetsCount * sizeof(unsigned int));
	if (result == cudaSuccess) result = cudaMalloc((void **)&recordsDev, (size_t)recordCount * sizeof(BpsMappedPixelRecordGpu));
	if (result == cudaSuccess) result = cudaMalloc((void **)&workRecordsDev, (size_t)workCount * sizeof(BpsMappedPixelRecordGpu));
	if (result == cudaSuccess) {
		result = cudaMemcpy(lineOffsetsDev, lineOffsets.data(),
							offsetsCount * sizeof(unsigned int),
							cudaMemcpyHostToDevice);
	}
	if (result == cudaSuccess) {
		result = cudaMemcpy(workOffsetsDev, workOffsets.data(),
							offsetsCount * sizeof(unsigned int),
							cudaMemcpyHostToDevice);
	}
	if (result == cudaSuccess) {
		result = cudaMemcpy(laneCursorsDev, lineOffsets.data(),
							offsetsCount * sizeof(unsigned int),
							cudaMemcpyHostToDevice);
	}
	if (result != cudaSuccess) {
		cudaFree(workRecordsDev);
		cudaFree(recordsDev);
		cudaFree(laneCursorsDev);
		cudaFree(workOffsetsDev);
		cudaFree(lineOffsetsDev);
		cudaFree(laneCountsDev);
		cudaFree(keyOf);
		cudaFree(laneOf);
		cudaFree(fieldB);
		cudaFree(fieldA);
		cudaFree(seedsDev);
		return result;
	}

	CudaPathScatterRecordsKernel<<<classifyGrid, classifyBlock>>>(
		laneOf, keyOf, laneCursorsDev, recordsDev, width, height);
	result = cudaPeekAtLastError();
	if (result == cudaSuccess) result = cudaDeviceSynchronize();
	if (result != cudaSuccess) {
		cudaFree(workRecordsDev);
		cudaFree(recordsDev);
		cudaFree(laneCursorsDev);
		cudaFree(workOffsetsDev);
		cudaFree(lineOffsetsDev);
		cudaFree(laneCountsDev);
		cudaFree(keyOf);
		cudaFree(laneOf);
		cudaFree(fieldB);
		cudaFree(fieldA);
		cudaFree(seedsDev);
		return result;
	}

	CudaPathSortRecordsKernel<<<lineCount, MAX_THREADS>>>(
		recordsDev, workRecordsDev, lineOffsetsDev, workOffsetsDev, lineCount);
	result = cudaPeekAtLastError();
	if (result == cudaSuccess) result = cudaDeviceSynchronize();
	if (result != cudaSuccess) {
		cudaFree(workRecordsDev);
		cudaFree(recordsDev);
		cudaFree(laneCursorsDev);
		cudaFree(workOffsetsDev);
		cudaFree(lineOffsetsDev);
		cudaFree(laneCountsDev);
		cudaFree(keyOf);
		cudaFree(laneOf);
		cudaFree(fieldB);
		cudaFree(fieldA);
		cudaFree(seedsDev);
		return result;
	}

	cache->key = pathMapKey;
	cache->device = device;
	cache->width = width;
	cache->height = height;
	cache->lineCount = lineCount;
	cache->mappedRecordCount = (int)recordCount;
	cache->mappedWorkItemCount = (int)workCount;
	cache->records = recordsDev;
	cache->lineOffsets = lineOffsetsDev;
	cache->workOffsets = workOffsetsDev;

	cudaFree(workRecordsDev);
	cudaFree(laneCursorsDev);
	cudaFree(laneCountsDev);
	cudaFree(keyOf);
	cudaFree(laneOf);
	cudaFree(fieldB);
	cudaFree(fieldA);
	cudaFree(seedsDev);
	return cudaSuccess;
}

// Host launch wrapper, called from SmartRenderGPU (BitonicPixelSorter_GPU.cpp).
// extern "C" to keep a stable, unmangled symbol across the nvcc/MSVC boundary.
//
// Follow SDK_Invert_ProcAmp: launch on the default stream and synchronise with
// cudaDeviceSynchronize(). The host-provided CUstream in command_queuePV is not
// passed through the CUDA runtime launch API in the Adobe sample.
extern "C" cudaError_t BitonicSort_CUDA(
	const void *src,
	const void *criterionMem,
	const void *triggerMem,
	void       *dst,
	int         srcPitch,
	int         dstPitch,
	int         width,
	int         height,
	int         inputOriginX,
	int         inputOriginY,
	int         inputWidth,
	int         inputHeight,
	int         criterionPitch,
	int         criterionOriginX,
	int         criterionOriginY,
	int         criterionWidth,
	int         criterionHeight,
	int         triggerPitch,
	int         triggerOriginX,
	int         triggerOriginY,
	int         triggerWidth,
	int         triggerHeight,
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
	int         pathClosed,
	float       pathLength,
	int         pathSMin,
	int         pathNMin,
	int         pathSampleCount,
	const void *pathSamplesHost,
	unsigned long long pathMapKey,
	int         useGpuPathMapBuild,
	int         mappedRecordCount,
	int         mappedWorkItemCount,
	const void *mappedRecordsHost,
	const void *mappedLineOffsetsHost,
	const void *mappedWorkOffsetsHost)
{
	if (lineCount <= 0 && !BPS_MODE_USES_MAPPED_SORT(mode)) {
		return cudaSuccess;
	}

	BpsPathSampleGpu *pathSamplesDev = nullptr;
	if (mode != BPS_MODE_PATH && pathSampleCount > 0 && pathSamplesHost) {
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
			(const float4 *)criterionMem, (const float4 *)triggerMem,
			criterionPitch, criterionOriginX, criterionOriginY, criterionWidth, criterionHeight,
			triggerPitch, triggerOriginX, triggerOriginY, triggerWidth, triggerHeight,
			srcPitch, dstPitch, width, height,
			inputOriginX, inputOriginY, inputWidth, inputHeight, outputOriginX, outputOriginY, outputWidth, outputHeight,
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

	if (BPS_MODE_USES_MAPPED_SORT(mode)) {
		dim3 copyBlock(16, 16, 1);
		dim3 copyGrid(
			(unsigned int)((outputWidth + 15) / 16),
			(unsigned int)((outputHeight + 15) / 16),
			1u);
		BitonicCopyInputKernel<<<copyGrid, copyBlock, 0>>>(
			(const float4 *)src, (float4 *)dst,
			srcPitch, dstPitch, width, height,
			inputOriginX, inputOriginY, inputWidth, inputHeight, outputOriginX, outputOriginY,
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

		BpsMappedPixelRecordGpu *recordsDev = nullptr;
		unsigned int *lineOffsetsDev = nullptr;
		unsigned int *workOffsetsDev = nullptr;
		unsigned int *domain = nullptr;
		float *keys = nullptr;
		bool ownsMapBuffers = true;
		std::unique_lock<std::mutex> cacheLock;
		if (useGpuPathMapBuild && mode == BPS_MODE_PATH) {
			cacheLock = std::unique_lock<std::mutex>(g_cudaPathMapMutex);
			result = bps_build_cuda_path_map(
				width, height, lineCount, pathDirection, pathClosed, pathLength,
				pathSMin, pathNMin, pathSampleCount, pathSamplesHost, pathMapKey,
				&g_cudaPathMap);
			if (result != cudaSuccess) {
				cudaFree(pathSamplesDev);
				return result;
			}
			mappedRecordCount = g_cudaPathMap.mappedRecordCount;
			mappedWorkItemCount = g_cudaPathMap.mappedWorkItemCount;
			recordsDev = g_cudaPathMap.records;
			lineOffsetsDev = g_cudaPathMap.lineOffsets;
			workOffsetsDev = g_cudaPathMap.workOffsets;
			ownsMapBuffers = false;
		} else {
			if (mappedRecordCount <= 0 || mappedWorkItemCount <= 0 ||
				!mappedRecordsHost || !mappedLineOffsetsHost || !mappedWorkOffsetsHost) {
				cudaFree(pathSamplesDev);
				return cudaSuccess;
			}
			const size_t recordsBytes =
				(size_t)mappedRecordCount * sizeof(BpsMappedPixelRecordGpu);
			const size_t offsetsBytes =
				(size_t)(lineCount + 1) * sizeof(unsigned int);
			result = cudaMalloc((void **)&recordsDev, recordsBytes);
			if (result == cudaSuccess) result = cudaMalloc((void **)&lineOffsetsDev, offsetsBytes);
			if (result == cudaSuccess) result = cudaMalloc((void **)&workOffsetsDev, offsetsBytes);
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
		}
		if (mappedRecordCount <= 0 || mappedWorkItemCount <= 0 ||
			!recordsDev || !lineOffsetsDev || !workOffsetsDev) {
			if (ownsMapBuffers) {
				cudaFree(workOffsetsDev);
				cudaFree(lineOffsetsDev);
				cudaFree(recordsDev);
			}
			cudaFree(pathSamplesDev);
			return cudaSuccess;
		}
		const size_t workUintBytes =
			(size_t)mappedWorkItemCount * sizeof(unsigned int);
		const size_t workFloatBytes =
			(size_t)mappedWorkItemCount * sizeof(float);

		if (result == cudaSuccess) result = cudaMalloc((void **)&domain, workUintBytes);
		if (result == cudaSuccess) result = cudaMalloc((void **)&keys, workFloatBytes);
		if (result != cudaSuccess) {
			cudaFree(keys);
			cudaFree(domain);
			if (ownsMapBuffers) {
				cudaFree(workOffsetsDev);
				cudaFree(lineOffsetsDev);
				cudaFree(recordsDev);
			}
			cudaFree(pathSamplesDev);
			return result;
		}

		BitonicSortMappedKernel<<<lineCount, MAX_THREADS, 0>>>(
			(const float4 *)src, (float4 *)dst,
			(const float4 *)criterionMem, (const float4 *)triggerMem,
			criterionPitch, criterionOriginX, criterionOriginY, criterionWidth, criterionHeight,
			triggerPitch, triggerOriginX, triggerOriginY, triggerWidth, triggerHeight,
			domain, keys,
			recordsDev, lineOffsetsDev, workOffsetsDev,
			srcPitch, dstPitch, width, height,
			inputOriginX, inputOriginY, inputWidth, inputHeight, outputOriginX, outputOriginY,
			outputWidth, outputHeight,
			ordering, criterion, trigger, affect, cycleDegrees,
			thresholdMin, thresholdMax, lineCount);

		result = cudaPeekAtLastError();
		if (result == cudaSuccess) {
			result = cudaDeviceSynchronize();
		}
		cudaFree(keys);
		cudaFree(domain);
		if (ownsMapBuffers) {
			cudaFree(workOffsetsDev);
			cudaFree(lineOffsetsDev);
			cudaFree(recordsDev);
		}
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
	cudaError_t result = bps_acquire_cuda_domain_buffers(domainBytes, keysBytes, &domain, &keys);
	if (result != cudaSuccess) {
		cudaFree(pathSamplesDev);
		return result;
	}

	result = cudaMemset(domain, 0xFF, domainBytes);
	if (result != cudaSuccess) {
		cudaFree(pathSamplesDev);
		return result;
	}

	BitonicSortDomainKernel<<<lineCount, MAX_THREADS, 0>>>(
		(const float4 *)src,
		(const float4 *)criterionMem, (const float4 *)triggerMem,
		criterionPitch, criterionOriginX, criterionOriginY, criterionWidth, criterionHeight,
		triggerPitch, triggerOriginX, triggerOriginY, triggerWidth, triggerHeight,
		domain, keys,
		srcPitch, width, height,
		inputOriginX, inputOriginY, inputWidth, inputHeight,
		mode, ordering, criterion, trigger, affect, cycleDegrees, lineCount,
		freePMin, freeQMin, freeLineLength, radialLength, paddedStride,
		thresholdMin, thresholdMax, angleCos, angleSin, centerX, centerY,
		swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
		pathSampleCount, pathSamplesDev);

	result = cudaPeekAtLastError();
	if (result != cudaSuccess) {
		cudaFree(pathSamplesDev);
		return result;
	}

	result = cudaDeviceSynchronize();
	if (result != cudaSuccess) {
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
		inputOriginX, inputOriginY, inputWidth, inputHeight, outputOriginX, outputOriginY, outputWidth, outputHeight,
		mode, lineCount,
		freePMin, freeQMin, freeLineLength, radialLength, paddedStride,
		angleCos, angleSin, centerX, centerY,
		swirlK, swirlLineMin, pathDirection, pathSMin, pathNMin,
		pathSampleCount, pathSamplesDev);

	result = cudaPeekAtLastError();
	if (result != cudaSuccess) {
		cudaFree(pathSamplesDev);
		return result;
	}

	result = cudaDeviceSynchronize();
	cudaFree(pathSamplesDev);
	return result;
}
