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
// beyond that). 256 threads, 16 KiB of shared sort keys.
#define MAX_THREADS 256
#define MAX_SIZE    4096
#define MAX_PAIRS   (MAX_SIZE / MAX_THREADS / 2)   // = 8 comparator pairs/thread

__device__ __forceinline__ float bps_brightness(float4 c)
{
	// BGRA: R=.z, G=.y, B=.x
	return __saturatef(0.298912f * c.z + 0.586611f * c.y + 0.114478f * c.x);
}

__device__ __forceinline__ unsigned int bps_packKey(unsigned int index, float bright)
{
	unsigned int h = __half_as_ushort(__float2half_rn(bright));
	return (index << 16) | (h & 0xFFFFu);
}

__device__ __forceinline__ float bps_unpackBrightness(unsigned int packed)
{
	return __half2float(__ushort_as_half((unsigned short)(packed & 0xFFFFu)));
}

__global__ void BitonicSortKernel(
	const float4 *srcTex,
	float4       *sortTex,
	int           srcPitch,
	int           dstPitch,
	int           width,
	int           height,
	int           direction,     // 1 = horizontal (sort along X), 0 = vertical
	int           ordering,      // 1 = ascending
	float         thresholdMin,
	float         thresholdMax)
{
	__shared__ unsigned int groupCache[MAX_SIZE];
	__shared__ unsigned int scanCache[MAX_THREADS];

	const unsigned int gid  = blockIdx.x;
	const unsigned int gtid = threadIdx.x;

	const unsigned int size = direction ? (unsigned int)width : (unsigned int)height;

	const unsigned int reducedSize = (size + 1u) >> 1;
	const unsigned int ops = (reducedSize + MAX_THREADS - 1u) / MAX_THREADS;
	const unsigned int pairBase = gtid * ops;

	// Maps a position along the line to a linear buffer index.
	#define BPS_SRC_INDEX(pos) ( (direction ? (gid)        : (unsigned int)(pos)) \
								+ (direction ? (unsigned int)(pos) : (gid)) * (unsigned int)srcPitch )

	// load pixels: build sort keys and an in-threshold mask for this thread's chunk
	unsigned int rangeMask = 0;
	{
		for (unsigned int t = 0; t < ops; t++)
		{
			unsigned int xL = (pairBase + t) << 1;
			unsigned int xR = xL + 1;

			// Guard out-of-line reads: with linear buffers (unlike clamped
			// textures) reading past the line length would touch other rows or
			// run off the buffer. xL/xR can exceed `size` for short lines.
			float brL = (xL < size) ? bps_brightness(srcTex[BPS_SRC_INDEX(xL)]) : 0.0f;
			float brR = (xR < size) ? bps_brightness(srcTex[BPS_SRC_INDEX(xR)]) : 0.0f;

			bool inL = xL < size && thresholdMin <= brL && brL <= thresholdMax;
			bool inR = xR < size && thresholdMin <= brR && brR <= thresholdMax;

			rangeMask |= (inL ? 1u : 0u) << (t * 2);
			rangeMask |= (inR ? 1u : 0u) << (t * 2 + 1);

			groupCache[xL] = bps_packKey(xL, brL);
			groupCache[xR] = bps_packKey(xR, brR);
		}
	}

	unsigned int preMeta[MAX_PAIRS];

	// forward pass: resolve the start index of the span containing each pair
	{
		unsigned int seed = 0;
		for (unsigned int t = 0; t < ops; t++)
		{
			unsigned int xL = (pairBase + t) << 1;
			if (!(rangeMask & (1u << (t * 2)))) seed = xL + 1;
			if (!(rangeMask & (2u << (t * 2)))) seed = xL + 2;
		}

		scanCache[gtid] = seed;
		__syncthreads();

		for (unsigned int offset = 1; offset < MAX_THREADS; offset <<= 1)
		{
			unsigned int own = scanCache[gtid];
			unsigned int other = scanCache[max((unsigned int)gtid, offset) - offset];
			__syncthreads();
			scanCache[gtid] = max(own, gtid >= offset ? other : 0u);
			__syncthreads();
		}

		unsigned int carry = scanCache[max((unsigned int)gtid, 1u) - 1];
		carry = gtid > 0 ? carry : 0u;

		for (unsigned int t = 0; t < ops; t++)
		{
			unsigned int xL = (pairBase + t) << 1;
			bool inL = (rangeMask & (1u << (t * 2))) != 0;
			bool inR = (rangeMask & (2u << (t * 2))) != 0;

			unsigned int start;
			if (inL) {
				start = carry;
			} else {
				carry = xL + 1;
				start = inR ? carry : xL;
			}
			if (!inR) carry = xL + 2;

			preMeta[t] = start << 16;
		}

		__syncthreads();
	}

	unsigned int lineLevels;

	// backward pass: resolve the end index of each span and the longest span
	{
		unsigned int seed = 0xFFFF;
		for (unsigned int t = ops; t > 0;)
		{
			t--;
			unsigned int xL = (pairBase + t) << 1;
			if (!(rangeMask & (2u << (t * 2)))) seed = xL + 1;
			if (!(rangeMask & (1u << (t * 2)))) seed = xL;
		}

		scanCache[gtid] = seed;
		__syncthreads();

		for (unsigned int offset = 1; offset < MAX_THREADS; offset <<= 1)
		{
			unsigned int own = scanCache[gtid];
			unsigned int other = scanCache[min((unsigned int)gtid + offset, (unsigned int)(MAX_THREADS - 1))];
			__syncthreads();
			scanCache[gtid] = min(own, (unsigned int)gtid + offset < MAX_THREADS ? other : 0xFFFFu);
			__syncthreads();
		}

		unsigned int carry = scanCache[min((unsigned int)gtid + 1, (unsigned int)(MAX_THREADS - 1))];
		carry = (unsigned int)gtid + 1 < MAX_THREADS ? carry : 0xFFFFu;

		unsigned int maxLen = 1;
		for (unsigned int t = ops; t > 0;)
		{
			t--;
			unsigned int xL = (pairBase + t) << 1;
			bool inL = (rangeMask & (1u << (t * 2))) != 0;
			bool inR = (rangeMask & (2u << (t * 2))) != 0;

			if (!inR) carry = xL + 1;

			unsigned int start = preMeta[t] >> 16;
			unsigned int end = (inL || inR) ? min(carry, size) - 1 : xL;

			if (!inL) carry = xL;

			unsigned int xSlot = xL + (start & 1);
			bool valid = end > start && xSlot <= end;
			start = valid ? start : xSlot;
			end = valid ? end : xSlot;

			preMeta[t] = (start << 16) | end;
			maxLen = max(maxLen, end - start + 1);
		}

		__syncthreads();

		scanCache[gtid] = maxLen;
		__syncthreads();

		for (unsigned int offset = MAX_THREADS >> 1; offset > 0; offset >>= 1)
		{
			if (gtid < offset) scanCache[gtid] = max(scanCache[gtid], scanCache[gtid + offset]);
			__syncthreads();
		}

		maxLen = scanCache[0];

		lineLevels = maxLen <= 1 ? 0 : (unsigned int)(31 - __clz((int)maxLen)) + 1;
	}

	for (unsigned int phase = 0; (phase & 0xFFFF) < lineLevels; phase++)
	{
		for (phase = (phase << 16) + (phase & 0xFFFF); (unsigned int)(phase >> 16) <= 0x7FFF; phase -= (1 << 16))
		{
			__syncthreads();

			for (unsigned int t = 0; t < ops; t++)
			{
				unsigned int metaPacked = preMeta[t];
				unsigned int rangeStart = metaPacked >> 16;
				unsigned int rangeEnd = metaPacked & 0xFFFF;

				unsigned int x = (pairBase + t) << 1;

				unsigned int useR = rangeStart & 1;
				unsigned int posInRange = x - rangeStart + useR;
				unsigned int swapIndex = posInRange >> 1;
				unsigned int comparatorSize = 1u << (phase >> 16);
				unsigned int a = rangeStart + (swapIndex & (comparatorSize - 1))
					+ (swapIndex >> (phase >> 16)) * (comparatorSize << 1);

				unsigned int b = a + comparatorSize;

				b = b <= rangeEnd ? b : a;

				unsigned int block = posInRange >> 1 >> (phase >> 16);
				unsigned int n = rangeEnd - rangeStart + 1;
				unsigned int endBlock = n >> ((phase >> 16) + 1);
				bool ascPattern = ((endBlock & 1) == 0) == (ordering != 0);
				bool asc = ((block & 1) == 0) == ascPattern;

				unsigned int val_a = groupCache[a];
				unsigned int val_b = groupCache[b];

				float br_a = bps_unpackBrightness(val_a);
				float br_b = bps_unpackBrightness(val_b);

				bool comp = br_a < br_b;

				unsigned int left = (asc == comp) ? val_a : val_b;
				unsigned int right = (asc == comp) ? val_b : val_a;

				groupCache[a] = left;
				groupCache[b] = right;
			}
		}
	}

	__syncthreads();

	#define BPS_DST_INDEX(pos) ( (direction ? (gid)        : (unsigned int)(pos)) \
								+ (direction ? (unsigned int)(pos) : (gid)) * (unsigned int)dstPitch )

	for (unsigned int t = 0; t < ops; t++)
	{
		unsigned int xL = (pairBase + t) << 1;
		unsigned int xR = xL + 1;

		unsigned int idx_left  = groupCache[xL] >> 16;
		unsigned int idx_right = groupCache[xR] >> 16;

		if (xL < size) sortTex[BPS_DST_INDEX(xL)] = srcTex[BPS_SRC_INDEX(idx_left)];
		if (xR < size) sortTex[BPS_DST_INDEX(xR)] = srcTex[BPS_SRC_INDEX(idx_right)];
	}

	#undef BPS_SRC_INDEX
	#undef BPS_DST_INDEX
}

// Host launch wrapper, called from SmartRenderGPU (BitonicPixelSorter_GPU.cpp).
// extern "C" to keep a stable, unmangled symbol across the nvcc/MSVC boundary.
extern "C" void BitonicSort_CUDA(
	const void *src,
	void       *dst,
	int         srcPitch,
	int         dstPitch,
	int         width,
	int         height,
	int         direction,
	int         ordering,
	float       thresholdMin,
	float       thresholdMax)
{
	const int lineCount = direction ? height : width;
	if (lineCount <= 0) return;

	BitonicSortKernel<<<lineCount, MAX_THREADS>>>(
		(const float4 *)src, (float4 *)dst,
		srcPitch, dstPitch, width, height,
		direction, ordering, thresholdMin, thresholdMax);

	cudaDeviceSynchronize();
}
