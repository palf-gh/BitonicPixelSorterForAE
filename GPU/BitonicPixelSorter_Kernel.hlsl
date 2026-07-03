/*
	BitonicPixelSorter_Kernel.hlsl

	DirectX 12 compute shader for the AE GPU BGRA128 path. The descriptor order
	matches Util/DirectXUtils: CBV(b0), UAV(u0), SRV(t0).
*/

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

cbuffer BitonicParams : register(b0)
{
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

RWByteAddressBuffer sortTex : register(u0);
ByteAddressBuffer srcTex : register(t0);

// scratchKey + scratchIndex already use the full 32 KB groupshared budget that
// Direct3D 12 hard-caps per group, so the span metadata reuses scratchIndex[0..3]
// rather than dedicated groupshared scalars (which would overflow the 32 KB cap).
// Thread 0 publishes it; every thread copies it into private locals; a second
// barrier then guarantees those copies finish before the load loop overwrites
// scratchIndex, removing the data race while keeping spanSize/sortSize uniform
// (no divergent barriers).
groupshared float scratchKey[MAX_SIZE];
groupshared uint scratchIndex[MAX_SIZE];

float4 LoadPixel(ByteAddressBuffer buf, uint index)
{
	return asfloat(buf.Load4(index * 16u));
}

void StorePixel(RWByteAddressBuffer buf, uint index, float4 value)
{
	buf.Store4(index * 16u, asuint(value));
}

float BpsSortKey(float4 c)
{
	// BGRA: R=.z, G=.y, B=.x
	if (criterion == BPS_CRITERION_RGB_AVERAGE) {
		return saturate((c.z + c.y + c.x) * (1.0f / 3.0f));
	}
	if (criterion == BPS_CRITERION_RGB_PRODUCT) {
		return saturate(c.z * c.y * c.x);
	}
	if (criterion == BPS_CRITERION_RGB_MINIMUM) {
		return saturate(min(min(c.z, c.y), c.x));
	}
	if (criterion == BPS_CRITERION_RGB_MAXIMUM) {
		return saturate(max(max(c.z, c.y), c.x));
	}
	return saturate(0.298912f * c.z + 0.586611f * c.y + 0.114478f * c.x);
}

uint BpsNextPow2(uint value)
{
	if (value <= 1u) {
		return 1u;
	}
	value--;
	value |= value >> 1;
	value |= value >> 2;
	value |= value >> 4;
	value |= value >> 8;
	value |= value >> 16;
	return value + 1u;
}

bool BpsBefore(float keyA, uint indexA, float keyB, uint indexB)
{
	if (keyA < keyB) {
		return true;
	}
	if (keyA > keyB) {
		return false;
	}
	return indexA < indexB;
}

int BpsRoundToInt(float value)
{
	return (int)floor(value + 0.5f);
}

uint BpsRotationLineLength(uint radius)
{
	if (radius == 0u) {
		return 1u;
	}
	uint length = (uint)ceil(BPS_TWO_PI * (float)radius);
	return length == 0u ? 1u : length;
}

uint BpsLineSize(uint gid)
{
	if (mode == BPS_MODE_FREE_ANGLE) {
		return (uint)freeLineLength;
	}
	if (mode == BPS_MODE_ROTATION) {
		return BpsRotationLineLength(gid);
	}
	if (mode == BPS_MODE_RADIAL) {
		return (uint)radialLength;
	}
	return direction != 0 ? (uint)width : (uint)height;
}

bool BpsCoordForPos(uint gid, uint pos, out int x, out int y)
{
	if (mode == BPS_MODE_FREE_ANGLE) {
		const float p = (float)(freePMin + (int)pos);
		const float q = (float)(freeQMin + (int)gid);
		x = BpsRoundToInt(p * angleCos - q * angleSin);
		y = BpsRoundToInt(p * angleSin + q * angleCos);
	} else if (mode == BPS_MODE_ROTATION) {
		const uint lineLen = BpsRotationLineLength(gid);
		const float theta = lineLen <= 1u ? 0.0f :
			(BPS_TWO_PI * (float)pos) / (float)lineLen;
		x = BpsRoundToInt(centerX + (float)gid * cos(theta));
		y = BpsRoundToInt(centerY + (float)gid * sin(theta));
	} else if (mode == BPS_MODE_RADIAL) {
		const float denom = lineCount <= 0 ? 1.0f : (float)lineCount;
		const float theta = (BPS_TWO_PI * (float)gid) / denom;
		x = BpsRoundToInt(centerX + (float)pos * cos(theta));
		y = BpsRoundToInt(centerY + (float)pos * sin(theta));
	} else {
		const int lineLayer = direction != 0 ? outputOriginY + (int)gid : outputOriginX + (int)gid;
		x = direction != 0 ? (int)pos : lineLayer;
		y = direction != 0 ? lineLayer : (int)pos;
	}
	return x >= 0 && y >= 0 && x < width && y < height;
}

bool SrcInWorld(int x, int y)
{
	return x >= inputOriginX && y >= inputOriginY &&
		x < inputOriginX + width && y < inputOriginY + height;
}

bool DstInWorld(int x, int y)
{
	return x >= outputOriginX && y >= outputOriginY &&
		x < outputOriginX + outputWidth && y < outputOriginY + outputHeight;
}

uint SrcIndexXY(int x, int y)
{
	return (uint)((x - inputOriginX) + (y - inputOriginY) * srcPitch);
}

uint DstIndexXY(int x, int y)
{
	return (uint)((x - outputOriginX) + (y - outputOriginY) * dstPitch);
}

[RootSignature("DescriptorTable(CBV(b0,numDescriptors=1)),DescriptorTable(UAV(u0,numDescriptors=1)),DescriptorTable(SRV(t0,numDescriptors=1))")]
[numthreads(256, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{
	const uint gid = groupID.x;
	const uint gtid = groupThreadID.x;
	const uint size = BpsLineSize(gid);
	if (size == 0u || size > MAX_SIZE) {
		return;
	}

	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		int x = 0;
		int y = 0;
		if (BpsCoordForPos(gid, pos, x, y) && SrcInWorld(x, y) && DstInWorld(x, y)) {
			StorePixel(sortTex, DstIndexXY(x, y), LoadPixel(srcTex, SrcIndexXY(x, y)));
		}
	}
	GroupMemoryBarrierWithGroupSync();

	uint cursor = 0u;
	while (cursor < size) {
		if (gtid == 0u) {
			uint spanStart = cursor;
			while (spanStart < size) {
				int x = 0;
				int y = 0;
				if (BpsCoordForPos(gid, spanStart, x, y) && SrcInWorld(x, y)) {
					float br = BpsSortKey(LoadPixel(srcTex, SrcIndexXY(x, y)));
					if (thresholdMin <= br && br <= thresholdMax) {
						break;
					}
				}
				spanStart++;
			}

			uint spanEnd = spanStart;
			while (spanEnd < size) {
				int x = 0;
				int y = 0;
				if (!BpsCoordForPos(gid, spanEnd, x, y) || !SrcInWorld(x, y)) {
					break;
				}
				float br = BpsSortKey(LoadPixel(srcTex, SrcIndexXY(x, y)));
				if (br < thresholdMin || br > thresholdMax) {
					break;
				}
				spanEnd++;
			}

			scratchIndex[0] = spanStart;
			scratchIndex[1] = spanEnd;
			scratchIndex[2] = spanEnd - spanStart;
			scratchIndex[3] = BpsNextPow2(scratchIndex[2]);
		}
		GroupMemoryBarrierWithGroupSync();

		const uint spanStart = scratchIndex[0];
		const uint spanEnd = scratchIndex[1];
		const uint spanSize = scratchIndex[2];
		const uint sortSize = scratchIndex[3];

		// Publish/consume fence: every thread has now copied the metadata into
		// private locals, so the load loop below may safely overwrite scratchIndex.
		GroupMemoryBarrierWithGroupSync();

		if (spanStart >= size || spanSize == 0u) {
			break;
		}

		const bool ascending = ordering != 0;
		for (uint loadIndex = gtid; loadIndex < sortSize; loadIndex += MAX_THREADS) {
			if (loadIndex < spanSize) {
				const uint pos = spanStart + loadIndex;
				int x = 0;
				int y = 0;
				const bool valid = BpsCoordForPos(gid, pos, x, y) && SrcInWorld(x, y);
				const uint srcIndex = valid ? SrcIndexXY(x, y) : 0xffffffffu;
				scratchKey[loadIndex] = valid
					? BpsSortKey(LoadPixel(srcTex, srcIndex))
					: (ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX);
				scratchIndex[loadIndex] = srcIndex;
			} else {
				scratchKey[loadIndex] = ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX;
				scratchIndex[loadIndex] = 0xffffffffu;
			}
		}
		GroupMemoryBarrierWithGroupSync();

		for (uint k = 2u; k <= sortSize; k <<= 1) {
			for (uint j = k >> 1; j > 0u; j >>= 1) {
				for (uint sortIndex = gtid; sortIndex < sortSize; sortIndex += MAX_THREADS) {
					const uint partner = sortIndex ^ j;
					if (partner > sortIndex) {
						bool stageAscending = (sortIndex & k) == 0u;
						if (!ascending) {
							stageAscending = !stageAscending;
						}

						const float keyA = scratchKey[sortIndex];
						const float keyB = scratchKey[partner];
						const uint indexA = scratchIndex[sortIndex];
						const uint indexB = scratchIndex[partner];
						const bool before = BpsBefore(keyA, indexA, keyB, indexB);
						if (before != stageAscending) {
							scratchKey[sortIndex] = keyB;
							scratchKey[partner] = keyA;
							scratchIndex[sortIndex] = indexB;
							scratchIndex[partner] = indexA;
						}
					}
				}
				GroupMemoryBarrierWithGroupSync();
			}
		}

		for (uint writeIndex = gtid; writeIndex < spanSize; writeIndex += MAX_THREADS) {
			const uint pos = spanStart + writeIndex;
			int x = 0;
			int y = 0;
			if (BpsCoordForPos(gid, pos, x, y) &&
				DstInWorld(x, y) && scratchIndex[writeIndex] != 0xffffffffu) {
				StorePixel(sortTex, DstIndexXY(x, y), LoadPixel(srcTex, scratchIndex[writeIndex]));
			}
		}
		GroupMemoryBarrierWithGroupSync();

		cursor = spanEnd + 1u;
	}
}
