/*
	BitonicPixelSorter_Kernel.metal

	Metal Shading Language port of BitonicPixelSorter_Kernel.cl.
	One threadgroup sorts one line. AE GPU worlds are BGRA float4 buffers, with
	row pitches expressed in float4 units by the host code.

	Buffer bindings:
	  buffer(0) - srcTex     : device const float4*   (source image)
	  buffer(1) - sortTex    : device float4*          (destination image)
	  buffer(2) - params     : constant BitonicSortParams& (host-packed scalar params)

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

#define BPS_CRITERION_RGB_AVERAGE 2
#define BPS_CRITERION_RGB_PRODUCT 3
#define BPS_CRITERION_RGB_MINIMUM 4
#define BPS_CRITERION_RGB_MAXIMUM 5

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
	int mode;
	int direction;
	int ordering;
	int criterion;
	int lineCount;
	int freePMin;
	int freeQMin;
	int freeLineLength;
	int radialLength;
	float thresholdMin;
	float thresholdMax;
	float angleCos;
	float angleSin;
	float centerX;
	float centerY;
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

inline uint bps_line_size(constant BitonicSortParams &p, uint gid)
{
	if (p.mode == BPS_MODE_FREE_ANGLE) return (uint)p.freeLineLength;
	if (p.mode == BPS_MODE_ROTATION) return bps_rotation_line_length(gid);
	if (p.mode == BPS_MODE_RADIAL) return (uint)p.radialLength;
	return p.direction ? (uint)p.width : (uint)p.height;
}

inline bool bps_coord_for_pos(constant BitonicSortParams &p, uint gid, uint pos,
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
		*x = bps_round_to_int(p.centerX + (float)gid * cos(theta));
		*y = bps_round_to_int(p.centerY + (float)gid * sin(theta));
	} else if (p.mode == BPS_MODE_RADIAL) {
		const float denom = p.lineCount <= 0 ? 1.0f : (float)p.lineCount;
		const float theta = (BPS_TWO_PI * (float)gid) / denom;
		*x = bps_round_to_int(p.centerX + (float)pos * cos(theta));
		*y = bps_round_to_int(p.centerY + (float)pos * sin(theta));
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
		x < p.inputOriginX + p.width && y < p.inputOriginY + p.height;
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

	const uint size = bps_line_size(p, gid);
	if (size == 0u || size > MAX_SIZE) {
		return;
	}

	// Copy source pixels that belong to this line's output range.
	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		int x = 0;
		int y = 0;
		if (bps_coord_for_pos(p, gid, pos, &x, &y) &&
			bps_src_in_world(p, x, y) && bps_dst_in_world(p, x, y)) {
			sortTex[bps_dst_index_xy(p, x, y)] = srcTex[bps_src_index_xy(p, x, y)];
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
				if (bps_coord_for_pos(p, gid, spanStart, &x, &y) &&
					bps_src_in_world(p, x, y)) {
					float br = bps_sort_key(srcTex[bps_src_index_xy(p, x, y)], p.criterion);
					if (p.thresholdMin <= br && br <= p.thresholdMax) break;
				}
				spanStart++;
			}

			uint spanEnd = spanStart;
			while (spanEnd < size) {
				int x = 0;
				int y = 0;
				if (!bps_coord_for_pos(p, gid, spanEnd, &x, &y) ||
					!bps_src_in_world(p, x, y)) break;
				float br = bps_sort_key(srcTex[bps_src_index_xy(p, x, y)], p.criterion);
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
				int x = 0;
				int y = 0;
				const bool valid = bps_coord_for_pos(p, gid, pos, &x, &y) &&
					bps_src_in_world(p, x, y);
				const uint srcIndex = valid ? bps_src_index_xy(p, x, y) : 0xffffffffu;
				scratchKey[i] = valid
					? bps_sort_key(srcTex[srcIndex], p.criterion)
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
			int x = 0;
			int y = 0;
			if (bps_coord_for_pos(p, gid, pos, &x, &y) &&
				bps_dst_in_world(p, x, y) && scratchIndex[i] != 0xffffffffu) {
				sortTex[bps_dst_index_xy(p, x, y)] = srcTex[scratchIndex[i]];
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);

		cursor = spanEnd + 1u;
	}

}
