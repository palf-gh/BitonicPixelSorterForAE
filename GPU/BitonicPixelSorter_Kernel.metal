/*
	BitonicPixelSorter_Kernel.metal

	Metal Shading Language port of BitonicPixelSorter_Kernel.cl.
	One threadgroup sorts one line. AE GPU worlds are BGRA float4 buffers, with
	row pitches expressed in float4 units by the host code.

	Buffer bindings:
	  buffer(0) - srcTex     : device const float4*   (source image)
	  buffer(1) - sortTex    : device float4*          (destination image)
	  buffer(2) - params     : constant BitonicSortParams& (14 scalar params packed)

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
	int outputOriginX;
	int outputOriginY;
	int outputWidth;
	int outputHeight;
	int direction;
	int ordering;
	float thresholdMin;
	float thresholdMax;
};

// ---------------------------------------------------------------------------
// Helper functions - direct ports of the OpenCL inline helpers.
// ---------------------------------------------------------------------------
inline float bps_brightness(float4 c)
{
	// BGRA: R=.z, G=.y, B=.x
	return clamp(0.298912f * c.z + 0.586611f * c.y + 0.114478f * c.x, 0.0f, 1.0f);
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

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
kernel void BitonicSortKernel(
	device const float4       *srcTex  [[buffer(0)]],
	device float4             *sortTex [[buffer(1)]],
	constant BitonicSortParams &p      [[buffer(2)]],
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

	const uint size = p.direction ? (uint)p.width : (uint)p.height;
	const int lineLayer = p.direction
		? (p.outputOriginY + (int)gid)
		: (p.outputOriginX + (int)gid);
	const int outputAxisStart = p.direction ? p.outputOriginX : p.outputOriginY;
	const int outputAxisEnd   = outputAxisStart + (p.direction ? p.outputWidth : p.outputHeight);

#define BPS_SRC_INDEX(pos) \
	((uint)(((p.direction ? (int)(pos) : lineLayer) - p.inputOriginX) + \
	        ((p.direction ? lineLayer : (int)(pos)) - p.inputOriginY) * p.srcPitch))

#define BPS_DST_INDEX(pos) \
	((uint)(((p.direction ? (int)(pos) - p.outputOriginX : (int)gid)) + \
	        ((p.direction ? (int)gid : (int)(pos) - p.outputOriginY) * p.dstPitch)))

#define BPS_SHOULD_WRITE(pos) \
	((int)(pos) >= outputAxisStart && (int)(pos) < outputAxisEnd)

	// Copy source pixels that belong to this line's output range.
	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		if (BPS_SHOULD_WRITE(pos)) {
			sortTex[BPS_DST_INDEX(pos)] = srcTex[BPS_SRC_INDEX(pos)];
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
				float br = bps_brightness(srcTex[BPS_SRC_INDEX(spanStart)]);
				if (p.thresholdMin <= br && br <= p.thresholdMax) break;
				spanStart++;
			}

			uint spanEnd = spanStart;
			while (spanEnd < size) {
				float br = bps_brightness(srcTex[BPS_SRC_INDEX(spanEnd)]);
				if (br < p.thresholdMin || br > p.thresholdMax) break;
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
				scratchKey[i]   = bps_brightness(srcTex[BPS_SRC_INDEX(pos)]);
				scratchIndex[i] = pos;
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
			if (BPS_SHOULD_WRITE(pos)) {
				sortTex[BPS_DST_INDEX(pos)] = srcTex[BPS_SRC_INDEX(scratchIndex[i])];
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		cursor = spanEnd + 1u;
	}

#undef BPS_SRC_INDEX
#undef BPS_DST_INDEX
#undef BPS_SHOULD_WRITE
}
