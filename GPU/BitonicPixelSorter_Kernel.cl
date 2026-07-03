/*
	BitonicPixelSorter_Kernel.cl

	OpenCL port of the upstream single-dispatch bitonic pixel sort.
	One work-group sorts one line. AE GPU worlds are BGRA float4 buffers, with
	row pitches expressed in float4 units by the host code.
*/

#define MAX_THREADS 256u
#define MAX_SIZE    4096u
#define BPS_FLOAT_MAX 3.402823466e+38F
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
	uint length = (uint)ceil(BPS_TWO_PI * (float)radius);
	return length == 0u ? 1u : length;
}

inline uint bps_line_size(int mode, int direction, uint gid,
						  int width, int height,
						  int freeLineLength, int radialLength)
{
	if (mode == BPS_MODE_FREE_ANGLE) return (uint)freeLineLength;
	if (mode == BPS_MODE_ROTATION) return bps_rotation_line_length(gid);
	if (mode == BPS_MODE_RADIAL) return (uint)radialLength;
	return direction ? (uint)width : (uint)height;
}

inline bool bps_coord_for_pos(int mode, int direction, uint gid, uint pos,
							  int width, int height,
							  int outputOriginX, int outputOriginY,
							  int lineCount,
							  int freePMin, int freeQMin, int freeLineLength,
							  float angleCos, float angleSin,
							  float centerX, float centerY,
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
		*x = bps_round_to_int(centerX + (float)gid * cos(theta));
		*y = bps_round_to_int(centerY + (float)gid * sin(theta));
	} else if (mode == BPS_MODE_RADIAL) {
		const float denom = lineCount <= 0 ? 1.0f : (float)lineCount;
		const float theta = (BPS_TWO_PI * (float)gid) / denom;
		*x = bps_round_to_int(centerX + (float)pos * cos(theta));
		*y = bps_round_to_int(centerY + (float)pos * sin(theta));
	} else {
		const int lineLayer = direction ? (outputOriginY + (int)gid) : (outputOriginX + (int)gid);
		*x = direction ? (int)pos : lineLayer;
		*y = direction ? lineLayer : (int)pos;
	}
	return *x >= 0 && *y >= 0 && *x < width && *y < height;
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
	int                    mode,
	int                    direction,
	int                    ordering,
	int                    criterion,
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
	float                  centerY)
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
	const uint size = bps_line_size(mode, direction, gid, width, height,
									freeLineLength, radialLength);
	if (size == 0u || size > MAX_SIZE) {
		return;
	}

	#define BPS_SRC_IN_WORLD(x, y) ((x) >= inputOriginX && (y) >= inputOriginY && \
									(x) < inputOriginX + width && (y) < inputOriginY + height)
	#define BPS_DST_IN_WORLD(x, y) ((x) >= outputOriginX && (y) >= outputOriginY && \
									(x) < outputOriginX + outputWidth && (y) < outputOriginY + outputHeight)
	#define BPS_SRC_INDEX_XY(x, y) ((uint)(((x) - inputOriginX) + ((y) - inputOriginY) * srcPitch))
	#define BPS_DST_INDEX_XY(x, y) ((uint)(((x) - outputOriginX) + ((y) - outputOriginY) * dstPitch))

	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		int x = 0, y = 0;
		if (bps_coord_for_pos(mode, direction, gid, pos, width, height,
							  outputOriginX, outputOriginY, lineCount,
							  freePMin, freeQMin, freeLineLength,
							  angleCos, angleSin, centerX, centerY, &x, &y) &&
			BPS_SRC_IN_WORLD(x, y) && BPS_DST_IN_WORLD(x, y)) {
			sortTex[BPS_DST_INDEX_XY(x, y)] = srcTex[BPS_SRC_INDEX_XY(x, y)];
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
									  angleCos, angleSin, centerX, centerY, &x, &y) &&
					BPS_SRC_IN_WORLD(x, y)) {
					float br = bps_sort_key(srcTex[BPS_SRC_INDEX_XY(x, y)], criterion);
					if (thresholdMin <= br && br <= thresholdMax) break;
				}
				spanStart++;
			}

			uint spanEnd = spanStart;
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
													 angleCos, angleSin, centerX, centerY, &x, &y) &&
								   BPS_SRC_IN_WORLD(x, y);
				const uint srcIndex = valid ? BPS_SRC_INDEX_XY(x, y) : 0xffffffffu;
				scratchKey[i] = valid ? bps_sort_key(srcTex[srcIndex], criterion) :
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
			int x = 0, y = 0;
			if (bps_coord_for_pos(mode, direction, gid, pos, width, height,
								  outputOriginX, outputOriginY, lineCount,
								  freePMin, freeQMin, freeLineLength,
								  angleCos, angleSin, centerX, centerY, &x, &y) &&
				BPS_DST_IN_WORLD(x, y) && scratchIndex[i] != 0xffffffffu) {
				sortTex[BPS_DST_INDEX_XY(x, y)] = srcTex[scratchIndex[i]];
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
