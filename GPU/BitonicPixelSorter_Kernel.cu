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

__device__ __forceinline__ float bps_brightness(float4 c)
{
	// BGRA: R=.z, G=.y, B=.x
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
	int           direction,     // 1 = horizontal (sort along X), 0 = vertical
	int           ordering,      // 1 = ascending
	float         thresholdMin,
	float         thresholdMax)
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

	const unsigned int size = direction ? (unsigned int)width : (unsigned int)height;
	const int lineLayer = direction ? (outputOriginY + (int)gid) : (outputOriginX + (int)gid);
	const int outputAxisStart = direction ? outputOriginX : outputOriginY;
	const int outputAxisEnd = outputAxisStart + (direction ? outputWidth : outputHeight);

	// Map layer coordinates into AE's possibly partial GPU worlds.
	#define BPS_SRC_INDEX(pos) ( (unsigned int)(((direction ? (int)(pos) : lineLayer) - inputOriginX) + \
								((direction ? lineLayer : (int)(pos)) - inputOriginY) * srcPitch) )
	#define BPS_DST_INDEX(pos) ( (unsigned int)(((direction ? (int)(pos) - outputOriginX : (int)gid)) + \
								((direction ? (int)gid : (int)(pos) - outputOriginY) * dstPitch)) )
	#define BPS_SHOULD_WRITE(pos) ((int)(pos) >= outputAxisStart && (int)(pos) < outputAxisEnd)

	for (unsigned int pos = gtid; pos < size; pos += MAX_THREADS) {
		if (BPS_SHOULD_WRITE(pos)) {
			sortTex[BPS_DST_INDEX(pos)] = srcTex[BPS_SRC_INDEX(pos)];
		}
	}
	__syncthreads();

	unsigned int cursor = 0;
	while (cursor < size) {
		if (gtid == 0) {
			unsigned int spanStart = cursor;
			while (spanStart < size) {
				float br = bps_brightness(srcTex[BPS_SRC_INDEX(spanStart)]);
				if (thresholdMin <= br && br <= thresholdMax) break;
				spanStart++;
			}

			unsigned int spanEnd = spanStart;
			while (spanEnd < size) {
				float br = bps_brightness(srcTex[BPS_SRC_INDEX(spanEnd)]);
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
				scratchKey[i] = bps_brightness(srcTex[BPS_SRC_INDEX(pos)]);
				scratchIndex[i] = pos;
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
			if (BPS_SHOULD_WRITE(pos)) {
				sortTex[BPS_DST_INDEX(pos)] = srcTex[BPS_SRC_INDEX(scratchIndex[i])];
			}
		}
		__syncthreads();

		cursor = spanEnd + 1u;
	}

	#undef BPS_SRC_INDEX
	#undef BPS_DST_INDEX
	#undef BPS_SHOULD_WRITE
}

// Host launch wrapper, called from SmartRenderGPU (BitonicPixelSorter_GPU.cpp).
// extern "C" to keep a stable, unmangled symbol across the nvcc/MSVC boundary.
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
	int         direction,
	int         ordering,
	float       thresholdMin,
	float       thresholdMax,
	void       *streamPV)
{
	const int lineCount = direction ? outputHeight : outputWidth;
	if (lineCount <= 0) return cudaSuccess;

	cudaStream_t stream = reinterpret_cast<cudaStream_t>(streamPV);
	BitonicSortKernel<<<lineCount, MAX_THREADS, 0, stream>>>(
		(const float4 *)src, (float4 *)dst,
		srcPitch, dstPitch, width, height,
		inputOriginX, inputOriginY, outputOriginX, outputOriginY, outputWidth, outputHeight,
		direction, ordering, thresholdMin, thresholdMax);

	cudaError_t launch_result = cudaPeekAtLastError();
	if (launch_result != cudaSuccess) {
		return launch_result;
	}
	return cudaStreamSynchronize(stream);
}
