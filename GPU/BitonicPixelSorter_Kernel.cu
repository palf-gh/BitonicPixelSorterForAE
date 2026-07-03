/*
	BitonicPixelSorter_Kernel.cu

	CUDA port of the upstream single-dispatch bitonic pixel sort
	(Packages/com.ruccho.bitonicpixelsorter/Runtime/BitonicPixelSorter.compute).

	One thread block sorts one line. Pixels are read once from the source buffer
	and the sorted *indices* gather full-colour pixels into the destination, so
	colours never pass through shared memory.

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

#define BPS_CRITERION_LUMINANCE 1
#define BPS_CRITERION_RGB_AVERAGE 2
#define BPS_CRITERION_RGB_PRODUCT 3
#define BPS_CRITERION_RGB_MINIMUM 4
#define BPS_CRITERION_RGB_MAXIMUM 5

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
	return __saturatef(0.298912f * c.z + 0.586611f * c.y + 0.114478f * c.x);
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

__device__ __forceinline__ unsigned int bps_line_size(
	int mode,
	int direction,
	unsigned int gid,
	int width,
	int height,
	int freeLineLength,
	int radialLength)
{
	if (mode == BPS_MODE_FREE_ANGLE) return (unsigned int)freeLineLength;
	if (mode == BPS_MODE_ROTATION) return bps_rotation_line_length(gid);
	if (mode == BPS_MODE_RADIAL) return (unsigned int)radialLength;
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
		*x = bps_round_to_int(centerX + (float)gid * cosf(theta));
		*y = bps_round_to_int(centerY + (float)gid * sinf(theta));
	} else if (mode == BPS_MODE_RADIAL) {
		const float denom = lineCount <= 0 ? 1.0f : (float)lineCount;
		const float theta = (BPS_TWO_PI * (float)gid) / denom;
		*x = bps_round_to_int(centerX + (float)pos * cosf(theta));
		*y = bps_round_to_int(centerY + (float)pos * sinf(theta));
	} else {
		const int lineLayer = direction ? (outputOriginY + (int)gid) : (outputOriginX + (int)gid);
		*x = direction ? (int)pos : lineLayer;
		*y = direction ? lineLayer : (int)pos;
	}
	return *x >= 0 && *y >= 0 && *x < width && *y < height;
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
	float         centerY)
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
		mode, direction, gid, width, height, freeLineLength, radialLength);
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
							  angleCos, angleSin, centerX, centerY, &x, &y) &&
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
									  angleCos, angleSin, centerX, centerY, &x, &y) &&
					BPS_SRC_IN_WORLD(x, y)) {
					float br = bps_sort_key(srcTex[BPS_SRC_INDEX_XY(x, y)], criterion);
					if (thresholdMin <= br && br <= thresholdMax) break;
				}
				spanStart++;
			}

			unsigned int spanEnd = spanStart;
			while (spanEnd < size) {
				int x = 0, y = 0;
				if (!bps_coord_for_pos(mode, direction, gid, spanEnd, width, height,
									   outputOriginX, outputOriginY, lineCount,
									   freePMin, freeQMin, freeLineLength,
									   angleCos, angleSin, centerX, centerY, &x, &y) ||
					!BPS_SRC_IN_WORLD(x, y)) break;
				float br = bps_sort_key(srcTex[BPS_SRC_INDEX_XY(x, y)], criterion);
				if (br < thresholdMin || br > thresholdMax) break;
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
													 angleCos, angleSin, centerX, centerY, &x, &y) &&
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
			int x = 0, y = 0;
			if (bps_coord_for_pos(mode, direction, gid, pos, width, height,
								  outputOriginX, outputOriginY, lineCount,
								  freePMin, freeQMin, freeLineLength,
								  angleCos, angleSin, centerX, centerY, &x, &y) &&
				BPS_DST_IN_WORLD(x, y) && scratchIndex[i] != 0xffffffffu) {
				sortTex[BPS_DST_INDEX_XY(x, y)] = srcTex[scratchIndex[i]];
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
	int         lineCount,
	int         freePMin,
	int         freeQMin,
	int         freeLineLength,
	int         radialLength,
	float       thresholdMin,
	float       thresholdMax,
	float       angleCos,
	float       angleSin,
	float       centerX,
	float       centerY)
{
	if (lineCount <= 0) {
		return cudaSuccess;
	}

	BitonicSortKernel<<<lineCount, MAX_THREADS, 0>>>(
		(const float4 *)src, (float4 *)dst,
		srcPitch, dstPitch, width, height,
		inputOriginX, inputOriginY, outputOriginX, outputOriginY, outputWidth, outputHeight,
		mode, direction, ordering, criterion, lineCount,
		freePMin, freeQMin, freeLineLength, radialLength,
		thresholdMin, thresholdMax, angleCos, angleSin, centerX, centerY);

	cudaError_t launch_result = cudaPeekAtLastError();
	if (launch_result != cudaSuccess) {
		return launch_result;
	}

	return cudaDeviceSynchronize();
}
