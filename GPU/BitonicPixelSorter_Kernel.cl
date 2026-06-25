/*
	BitonicPixelSorter_Kernel.cl

	OpenCL port of the upstream single-dispatch bitonic pixel sort.
	One work-group sorts one line. AE GPU worlds are BGRA float4 buffers, with
	row pitches expressed in float4 units by the host code.
*/

#define MAX_THREADS 256u
#define MAX_SIZE    4096u
#define MAX_PAIRS   (MAX_SIZE / MAX_THREADS / 2u)

inline float bps_brightness(float4 c)
{
	// BGRA: R=.z, G=.y, B=.x
	return clamp(0.298912f * c.z + 0.586611f * c.y + 0.114478f * c.x, 0.0f, 1.0f);
}

inline uint bps_float_to_half_bits(float value)
{
	uint bits = as_uint(value);
	uint sign = (bits >> 16) & 0x8000u;
	int exp = (int)((bits >> 23) & 0xffu) - 127 + 15;
	uint mant = bits & 0x7fffffu;

	if (exp <= 0) {
		if (exp < -10) {
			return sign;
		}
		mant = (mant | 0x800000u) >> (uint)(1 - exp);
		return sign | ((mant + 0x1000u) >> 13);
	}

	if (exp >= 31) {
		return sign | 0x7c00u;
	}

	uint rounded = mant + 0x1000u;
	if (rounded & 0x800000u) {
		rounded = 0;
		exp += 1;
		if (exp >= 31) {
			return sign | 0x7c00u;
		}
	}

	return sign | ((uint)exp << 10) | ((rounded >> 13) & 0x03ffu);
}

inline uint bps_pack_key(uint index, float bright)
{
	return (index << 16) | (bps_float_to_half_bits(bright) & 0xffffu);
}

__kernel void BitonicSortKernel(
	__global const float4 *srcTex,
	__global float4       *sortTex,
	int                    srcPitch,
	int                    dstPitch,
	int                    width,
	int                    height,
	int                    direction,
	int                    ordering,
	float                  thresholdMin,
	float                  thresholdMax)
{
	__local uint groupCache[MAX_SIZE];
	__local uint scanCache[MAX_THREADS];

	const uint gid = get_group_id(0);
	const uint gtid = get_local_id(0);
	const uint size = direction ? (uint)width : (uint)height;

	const uint reducedSize = (size + 1u) >> 1;
	const uint ops = (reducedSize + MAX_THREADS - 1u) / MAX_THREADS;
	const uint pairBase = gtid * ops;

	#define BPS_SRC_INDEX(pos) ((direction ? gid : (uint)(pos)) + \
								(direction ? (uint)(pos) : gid) * (uint)srcPitch)
	#define BPS_DST_INDEX(pos) ((direction ? gid : (uint)(pos)) + \
								(direction ? (uint)(pos) : gid) * (uint)dstPitch)

	uint rangeMask = 0;
	for (uint t = 0; t < ops; t++) {
		uint xL = (pairBase + t) << 1;
		uint xR = xL + 1u;

		float brL = (xL < size) ? bps_brightness(srcTex[BPS_SRC_INDEX(xL)]) : 0.0f;
		float brR = (xR < size) ? bps_brightness(srcTex[BPS_SRC_INDEX(xR)]) : 0.0f;

		bool inL = xL < size && thresholdMin <= brL && brL <= thresholdMax;
		bool inR = xR < size && thresholdMin <= brR && brR <= thresholdMax;

		rangeMask |= (inL ? 1u : 0u) << (t * 2u);
		rangeMask |= (inR ? 1u : 0u) << (t * 2u + 1u);

		groupCache[xL] = bps_pack_key(xL, brL);
		groupCache[xR] = bps_pack_key(xR, brR);
	}

	uint preMeta[MAX_PAIRS];

	{
		uint seed = 0;
		for (uint t = 0; t < ops; t++) {
			uint xL = (pairBase + t) << 1;
			if (!(rangeMask & (1u << (t * 2u)))) seed = xL + 1u;
			if (!(rangeMask & (2u << (t * 2u)))) seed = xL + 2u;
		}

		scanCache[gtid] = seed;
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint offset = 1u; offset < MAX_THREADS; offset <<= 1) {
			uint own = scanCache[gtid];
			uint other = scanCache[max(gtid, offset) - offset];
			barrier(CLK_LOCAL_MEM_FENCE);
			scanCache[gtid] = max(own, gtid >= offset ? other : 0u);
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		uint carry = scanCache[max(gtid, 1u) - 1u];
		carry = gtid > 0 ? carry : 0u;

		for (uint t = 0; t < ops; t++) {
			uint xL = (pairBase + t) << 1;
			bool inL = (rangeMask & (1u << (t * 2u))) != 0;
			bool inR = (rangeMask & (2u << (t * 2u))) != 0;

			uint start;
			if (inL) {
				start = carry;
			} else {
				carry = xL + 1u;
				start = inR ? carry : xL;
			}
			if (!inR) carry = xL + 2u;

			preMeta[t] = start << 16;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	uint lineLevels;

	{
		uint seed = 0xffffu;
		for (uint t = ops; t > 0;) {
			t--;
			uint xL = (pairBase + t) << 1;
			if (!(rangeMask & (2u << (t * 2u)))) seed = xL + 1u;
			if (!(rangeMask & (1u << (t * 2u)))) seed = xL;
		}

		scanCache[gtid] = seed;
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint offset = 1u; offset < MAX_THREADS; offset <<= 1) {
			uint own = scanCache[gtid];
			uint other = scanCache[min(gtid + offset, MAX_THREADS - 1u)];
			barrier(CLK_LOCAL_MEM_FENCE);
			scanCache[gtid] = min(own, gtid + offset < MAX_THREADS ? other : 0xffffu);
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		uint carry = scanCache[min(gtid + 1u, MAX_THREADS - 1u)];
		carry = gtid + 1u < MAX_THREADS ? carry : 0xffffu;

		uint maxLen = 1u;
		for (uint t = ops; t > 0;) {
			t--;
			uint xL = (pairBase + t) << 1;
			bool inL = (rangeMask & (1u << (t * 2u))) != 0;
			bool inR = (rangeMask & (2u << (t * 2u))) != 0;

			if (!inR) carry = xL + 1u;

			uint start = preMeta[t] >> 16;
			uint end = (inL || inR) ? min(carry, size) - 1u : xL;

			if (!inL) carry = xL;

			uint xSlot = xL + (start & 1u);
			bool valid = end > start && xSlot <= end;
			start = valid ? start : xSlot;
			end = valid ? end : xSlot;

			preMeta[t] = (start << 16) | end;
			maxLen = max(maxLen, end - start + 1u);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		scanCache[gtid] = maxLen;
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint offset = MAX_THREADS >> 1; offset > 0; offset >>= 1) {
			if (gtid < offset) {
				scanCache[gtid] = max(scanCache[gtid], scanCache[gtid + offset]);
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		maxLen = scanCache[0];
		lineLevels = maxLen <= 1u ? 0u : (31u - clz(maxLen)) + 1u;
	}

	for (uint phase = 0; (phase & 0xffffu) < lineLevels; phase++) {
		for (phase = (phase << 16) + (phase & 0xffffu);
			 (phase >> 16) <= 0x7fffu;
			 phase -= (1u << 16)) {
			barrier(CLK_LOCAL_MEM_FENCE);

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

				uint val_a = groupCache[a];
				uint val_b = groupCache[b];

				bool comp = (val_a & 0xffffu) < (val_b & 0xffffu);

				uint left = (asc == comp) ? val_a : val_b;
				uint right = (asc == comp) ? val_b : val_a;

				groupCache[a] = left;
				groupCache[b] = right;
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (uint t = 0; t < ops; t++) {
		uint xL = (pairBase + t) << 1;
		uint xR = xL + 1u;

		uint idxLeft = groupCache[xL] >> 16;
		uint idxRight = groupCache[xR] >> 16;

		if (xL < size) sortTex[BPS_DST_INDEX(xL)] = srcTex[BPS_SRC_INDEX(idxLeft)];
		if (xR < size) sortTex[BPS_DST_INDEX(xR)] = srcTex[BPS_SRC_INDEX(idxRight)];
	}

	#undef BPS_SRC_INDEX
	#undef BPS_DST_INDEX
}
