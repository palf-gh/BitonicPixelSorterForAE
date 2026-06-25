/*
	BitonicPixelSorter_Kernel.cl

	OpenCL port of the upstream single-dispatch bitonic pixel sort.
	One work-group sorts one line. AE GPU worlds are BGRA float4 buffers, with
	row pitches expressed in float4 units by the host code.
*/

#define MAX_THREADS 256u
#define MAX_SIZE    4096u
#define BPS_FLOAT_MAX 3.402823466e+38F

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

__kernel void BitonicSortKernel(
	__global const float4 *srcTex,
	__global float4       *sortTex,
	int                    srcPitch,
	int                    dstPitch,
	int                    width,
	int                    height,
	int                    inputOriginX,
	int                    inputOriginY,
	int                    outputOriginX,
	int                    outputOriginY,
	int                    outputWidth,
	int                    outputHeight,
	int                    direction,
	int                    ordering,
	float                  thresholdMin,
	float                  thresholdMax)
{
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
	const uint size = direction ? (uint)width : (uint)height;
	const int lineLayer = direction ? (outputOriginY + (int)gid) : (outputOriginX + (int)gid);
	const int outputAxisStart = direction ? outputOriginX : outputOriginY;
	const int outputAxisEnd = outputAxisStart + (direction ? outputWidth : outputHeight);

	#define BPS_SRC_INDEX(pos) ((uint)(((direction ? (int)(pos) : lineLayer) - inputOriginX) + \
								((direction ? lineLayer : (int)(pos)) - inputOriginY) * srcPitch))
	#define BPS_DST_INDEX(pos) ((uint)(((direction ? (int)(pos) - outputOriginX : (int)gid)) + \
								((direction ? (int)gid : (int)(pos) - outputOriginY) * dstPitch)))
	#define BPS_SHOULD_WRITE(pos) ((int)(pos) >= outputAxisStart && (int)(pos) < outputAxisEnd)

	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		if (BPS_SHOULD_WRITE(pos)) {
			sortTex[BPS_DST_INDEX(pos)] = srcTex[BPS_SRC_INDEX(pos)];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	uint cursor = 0u;
	while (cursor < size) {
		if (gtid == 0u) {
			uint spanStart = cursor;
			while (spanStart < size) {
				float br = bps_brightness(srcTex[BPS_SRC_INDEX(spanStart)]);
				if (thresholdMin <= br && br <= thresholdMax) break;
				spanStart++;
			}

			uint spanEnd = spanStart;
			while (spanEnd < size) {
				float br = bps_brightness(srcTex[BPS_SRC_INDEX(spanEnd)]);
				if (br < thresholdMin || br > thresholdMax) break;
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
				scratchKey[i] = bps_brightness(srcTex[BPS_SRC_INDEX(pos)]);
				scratchIndex[i] = pos;
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
			if (BPS_SHOULD_WRITE(pos)) {
				sortTex[BPS_DST_INDEX(pos)] = srcTex[BPS_SRC_INDEX(scratchIndex[i])];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		cursor = spanEnd + 1u;
	}

	#undef BPS_SRC_INDEX
	#undef BPS_DST_INDEX
	#undef BPS_SHOULD_WRITE
}
