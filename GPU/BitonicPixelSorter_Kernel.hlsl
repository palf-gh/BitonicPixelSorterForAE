/*
	BitonicPixelSorter_Kernel.hlsl

	DirectX 12 compute shader for the AE GPU BGRA128 path. The descriptor order
	matches Util/DirectXUtils: CBV(b0), UAV(u0), SRV(t0).
*/

#define MAX_THREADS 256u
#define MAX_SIZE    4096u
#define BPS_FLOAT_MAX 3.402823466e+38F

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
	int direction;
	int ordering;
	float thresholdMin;
	float thresholdMax;
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

float BpsBrightness(float4 c)
{
	// BGRA: R=.z, G=.y, B=.x
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

uint SrcIndex(uint gid, uint pos)
{
	const int lineLayer = direction != 0 ? outputOriginY + (int)gid : outputOriginX + (int)gid;
	const int layerX = direction != 0 ? (int)pos : lineLayer;
	const int layerY = direction != 0 ? lineLayer : (int)pos;
	return (uint)((layerX - inputOriginX) + (layerY - inputOriginY) * srcPitch);
}

uint DstIndex(uint gid, uint pos)
{
	const int localX = direction != 0 ? (int)pos - outputOriginX : (int)gid;
	const int localY = direction != 0 ? (int)gid : (int)pos - outputOriginY;
	return (uint)(localX + localY * dstPitch);
}

bool ShouldWrite(uint pos)
{
	const int outputAxisStart = direction != 0 ? outputOriginX : outputOriginY;
	const int outputAxisEnd = outputAxisStart + (direction != 0 ? outputWidth : outputHeight);
	return (int)pos >= outputAxisStart && (int)pos < outputAxisEnd;
}

[RootSignature("DescriptorTable(CBV(b0,numDescriptors=1)),DescriptorTable(UAV(u0,numDescriptors=1)),DescriptorTable(SRV(t0,numDescriptors=1))")]
[numthreads(256, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{
	const uint gid = groupID.x;
	const uint gtid = groupThreadID.x;
	const uint size = direction != 0 ? (uint)width : (uint)height;

	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		if (ShouldWrite(pos)) {
			StorePixel(sortTex, DstIndex(gid, pos), LoadPixel(srcTex, SrcIndex(gid, pos)));
		}
	}
	GroupMemoryBarrierWithGroupSync();

	uint cursor = 0u;
	while (cursor < size) {
		if (gtid == 0u) {
			uint spanStart = cursor;
			while (spanStart < size) {
				float br = BpsBrightness(LoadPixel(srcTex, SrcIndex(gid, spanStart)));
				if (thresholdMin <= br && br <= thresholdMax) {
					break;
				}
				spanStart++;
			}

			uint spanEnd = spanStart;
			while (spanEnd < size) {
				float br = BpsBrightness(LoadPixel(srcTex, SrcIndex(gid, spanEnd)));
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
				scratchKey[loadIndex] = BpsBrightness(LoadPixel(srcTex, SrcIndex(gid, pos)));
				scratchIndex[loadIndex] = pos;
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
			if (ShouldWrite(pos)) {
				StorePixel(sortTex, DstIndex(gid, pos), LoadPixel(srcTex, SrcIndex(gid, scratchIndex[writeIndex])));
			}
		}
		GroupMemoryBarrierWithGroupSync();

		cursor = spanEnd + 1u;
	}
}
