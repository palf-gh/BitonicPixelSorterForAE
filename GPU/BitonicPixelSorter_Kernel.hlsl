/*
	BitonicPixelSorter_Kernel.hlsl

	DirectX 12 compute shader for the AE GPU BGRA128 path. The descriptor order
	matches Util/DirectXUtils: CBV(b0), UAV(u0), SRV(t0).
*/

#define MAX_THREADS 256u
#define MAX_SIZE    4096u
#define MAX_PAIRS   (MAX_SIZE / MAX_THREADS / 2u)

cbuffer BitonicParams : register(b0)
{
	int srcPitch;
	int dstPitch;
	int width;
	int height;
	int direction;
	int ordering;
	float thresholdMin;
	float thresholdMax;
};

RWByteAddressBuffer sortTex : register(u0);
ByteAddressBuffer srcTex : register(t0);

groupshared uint groupCache[MAX_SIZE];
groupshared uint scanCache[MAX_THREADS];

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

uint BpsPackKey(uint index, float bright)
{
	return (index << 16) | (f32tof16(bright) & 0xffffu);
}

uint SrcIndex(uint gid, uint pos)
{
	return (direction != 0 ? gid : pos) + (direction != 0 ? pos : gid) * (uint)srcPitch;
}

uint DstIndex(uint gid, uint pos)
{
	return (direction != 0 ? gid : pos) + (direction != 0 ? pos : gid) * (uint)dstPitch;
}

[RootSignature("DescriptorTable(CBV(b0,numDescriptors=1)),DescriptorTable(UAV(u0,numDescriptors=1)),DescriptorTable(SRV(t0,numDescriptors=1))")]
[numthreads(256, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{
	const uint gid = groupID.x;
	const uint gtid = groupThreadID.x;
	const uint size = direction != 0 ? (uint)width : (uint)height;

	const uint reducedSize = (size + 1u) >> 1;
	const uint ops = (reducedSize + MAX_THREADS - 1u) / MAX_THREADS;
	const uint pairBase = gtid * ops;

	uint rangeMask = 0;
	for (uint loadT = 0; loadT < ops; loadT++) {
		uint xL = (pairBase + loadT) << 1;
		uint xR = xL + 1u;

		float brL = (xL < size) ? BpsBrightness(LoadPixel(srcTex, SrcIndex(gid, xL))) : 0.0f;
		float brR = (xR < size) ? BpsBrightness(LoadPixel(srcTex, SrcIndex(gid, xR))) : 0.0f;

		bool inL = xL < size && thresholdMin <= brL && brL <= thresholdMax;
		bool inR = xR < size && thresholdMin <= brR && brR <= thresholdMax;

		rangeMask |= (inL ? 1u : 0u) << (loadT * 2u);
		rangeMask |= (inR ? 1u : 0u) << (loadT * 2u + 1u);

		groupCache[xL] = BpsPackKey(xL, brL);
		groupCache[xR] = BpsPackKey(xR, brR);
	}

	uint preMeta[MAX_PAIRS];

	{
		uint seed = 0;
		for (uint forwardSeedT = 0; forwardSeedT < ops; forwardSeedT++) {
			uint xL = (pairBase + forwardSeedT) << 1;
			if (!(rangeMask & (1u << (forwardSeedT * 2u)))) seed = xL + 1u;
			if (!(rangeMask & (2u << (forwardSeedT * 2u)))) seed = xL + 2u;
		}

		scanCache[gtid] = seed;
		GroupMemoryBarrierWithGroupSync();

		for (uint forwardOffset = 1u; forwardOffset < MAX_THREADS; forwardOffset <<= 1) {
			uint own = scanCache[gtid];
			uint other = scanCache[max(gtid, forwardOffset) - forwardOffset];
			GroupMemoryBarrierWithGroupSync();
			scanCache[gtid] = max(own, gtid >= forwardOffset ? other : 0u);
			GroupMemoryBarrierWithGroupSync();
		}

		uint carry = scanCache[max(gtid, 1u) - 1u];
		carry = gtid > 0 ? carry : 0u;

		for (uint forwardMetaT = 0; forwardMetaT < ops; forwardMetaT++) {
			uint xL = (pairBase + forwardMetaT) << 1;
			bool inL = (rangeMask & (1u << (forwardMetaT * 2u))) != 0;
			bool inR = (rangeMask & (2u << (forwardMetaT * 2u))) != 0;

			uint start;
			if (inL) {
				start = carry;
			} else {
				carry = xL + 1u;
				start = inR ? carry : xL;
			}
			if (!inR) carry = xL + 2u;

			preMeta[forwardMetaT] = start << 16;
		}

		GroupMemoryBarrierWithGroupSync();
	}

	uint lineLevels;

	{
		uint seed = 0xffffu;
		for (uint backwardSeedT = ops; backwardSeedT > 0;) {
			backwardSeedT--;
			uint xL = (pairBase + backwardSeedT) << 1;
			if (!(rangeMask & (2u << (backwardSeedT * 2u)))) seed = xL + 1u;
			if (!(rangeMask & (1u << (backwardSeedT * 2u)))) seed = xL;
		}

		scanCache[gtid] = seed;
		GroupMemoryBarrierWithGroupSync();

		for (uint backwardOffset = 1u; backwardOffset < MAX_THREADS; backwardOffset <<= 1) {
			uint own = scanCache[gtid];
			uint other = scanCache[min(gtid + backwardOffset, MAX_THREADS - 1u)];
			GroupMemoryBarrierWithGroupSync();
			scanCache[gtid] = min(own, gtid + backwardOffset < MAX_THREADS ? other : 0xffffu);
			GroupMemoryBarrierWithGroupSync();
		}

		uint carry = scanCache[min(gtid + 1u, MAX_THREADS - 1u)];
		carry = gtid + 1u < MAX_THREADS ? carry : 0xffffu;

		uint maxLen = 1u;
		for (uint backwardMetaT = ops; backwardMetaT > 0;) {
			backwardMetaT--;
			uint xL = (pairBase + backwardMetaT) << 1;
			bool inL = (rangeMask & (1u << (backwardMetaT * 2u))) != 0;
			bool inR = (rangeMask & (2u << (backwardMetaT * 2u))) != 0;

			if (!inR) carry = xL + 1u;

			uint start = preMeta[backwardMetaT] >> 16;
			uint end = (inL || inR) ? min(carry, size) - 1u : xL;

			if (!inL) carry = xL;

			uint xSlot = xL + (start & 1u);
			bool valid = end > start && xSlot <= end;
			start = valid ? start : xSlot;
			end = valid ? end : xSlot;

			preMeta[backwardMetaT] = (start << 16) | end;
			maxLen = max(maxLen, end - start + 1u);
		}

		GroupMemoryBarrierWithGroupSync();

		scanCache[gtid] = maxLen;
		GroupMemoryBarrierWithGroupSync();

		for (uint reduceOffset = MAX_THREADS >> 1; reduceOffset > 0; reduceOffset >>= 1) {
			if (gtid < reduceOffset) {
				scanCache[gtid] = max(scanCache[gtid], scanCache[gtid + reduceOffset]);
			}
			GroupMemoryBarrierWithGroupSync();
		}

		maxLen = scanCache[0];
		lineLevels = maxLen <= 1u ? 0u : (uint)firstbithigh(maxLen) + 1u;
	}

	for (uint phase = 0; (phase & 0xffffu) < lineLevels; phase++) {
		for (phase = (phase << 16) + (phase & 0xffffu);
			 (phase >> 16) <= 0x7fffu;
			 phase -= (1u << 16)) {
			GroupMemoryBarrierWithGroupSync();

			for (uint t = 0; t < ops; t++) {
				uint metaPacked = preMeta[t];
				uint rangeStart = metaPacked >> 16;
				uint rangeEnd = metaPacked & 0xffffu;

				uint x = (pairBase + t) << 1;

				uint useR = rangeStart & 1u;
				uint posInRange = x - rangeStart + useR;
				uint swapIndex = posInRange >> 1;
				uint comparatorSize = 1u << (phase >> 16);
				uint a = rangeStart + (swapIndex & (comparatorSize - 1u))
					+ (swapIndex >> (phase >> 16)) * (comparatorSize << 1);
				uint b = a + comparatorSize;

				b = b <= rangeEnd ? b : a;

				uint block = posInRange >> 1 >> (phase >> 16);
				uint n = rangeEnd - rangeStart + 1u;
				uint endBlock = n >> ((phase >> 16) + 1u);
				bool ascPattern = ((endBlock & 1u) == 0) == (ordering != 0);
				bool asc = ((block & 1u) == 0) == ascPattern;

				uint valA = groupCache[a];
				uint valB = groupCache[b];

				bool comp = (valA & 0xffffu) < (valB & 0xffffu);

				uint left = (asc == comp) ? valA : valB;
				uint right = (asc == comp) ? valB : valA;

				groupCache[a] = left;
				groupCache[b] = right;
			}
		}
	}

	GroupMemoryBarrierWithGroupSync();

	for (uint writeT = 0; writeT < ops; writeT++) {
		uint xL = (pairBase + writeT) << 1;
		uint xR = xL + 1u;

		uint idxLeft = groupCache[xL] >> 16;
		uint idxRight = groupCache[xR] >> 16;

		if (xL < size) {
			StorePixel(sortTex, DstIndex(gid, xL), LoadPixel(srcTex, SrcIndex(gid, idxLeft)));
		}
		if (xR < size) {
			StorePixel(sortTex, DstIndex(gid, xR), LoadPixel(srcTex, SrcIndex(gid, idxRight)));
		}
	}
}
