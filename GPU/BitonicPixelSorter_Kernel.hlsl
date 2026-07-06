/*
	BitonicPixelSorter_Kernel.hlsl

	DirectX 12 compute shaders for the AE GPU BGRA128 path.
	  main         - axis single-pass sort (CBV, UAV dst, SRV src)
	  SortDomain   - non-axis pass 1 (CBV, UAV domain+keys, SRV src)
	  ApplyDomain  - non-axis pass 2 (CBV, UAV dst, SRV src, SRV domain)
	Descriptor order matches Util/DirectXUtils: CBV(b0), UAV(u0..), SRV(t0..).
*/

#define MAX_THREADS 256u
#define MAX_SIZE    4096u
#define BPS_FLOAT_MAX 3.402823466e+38F
#define BPS_TWO_PI 6.28318530717958647692f

#define BPS_MODE_AXIS 1
#define BPS_MODE_FREE_ANGLE 2
#define BPS_MODE_ROTATION 3
#define BPS_MODE_RADIAL 4
#define BPS_MODE_SWIRL 5
#define BPS_MODE_PATH 6
#define BPS_PATH_DIR_NORMAL 1
#define BPS_PATH_DIR_TANGENT 2
#define BPS_SWIRL_K_EPS 1.0e-4f

#define BPS_CRITERION_RGB_AVERAGE 2
#define BPS_CRITERION_RGB_PRODUCT 3
#define BPS_CRITERION_RGB_MINIMUM 4
#define BPS_CRITERION_RGB_MAXIMUM 5
#define BPS_CRITERION_RED_CHANNEL 6
#define BPS_CRITERION_GREEN_CHANNEL 7
#define BPS_CRITERION_BLUE_CHANNEL 8
#define BPS_CRITERION_ALPHA_CHANNEL 9
#define BPS_CRITERION_HUE 10
#define BPS_CRITERION_SATURATION 11

#define BPS_AFFECT_INSIDE_THRESHOLDS 1
#define BPS_AFFECT_OUTSIDE_THRESHOLDS 2

cbuffer BitonicParams : register(b0)
{
	int srcPitch;
	int dstPitch;
	int width;
	int height;
	int inputOriginX;
	int inputOriginY;
	int inputWidth;
	int inputHeight;
	int criterionPitch;
	int criterionOriginX;
	int criterionOriginY;
	int criterionWidth;
	int criterionHeight;
	int triggerPitch;
	int triggerOriginX;
	int triggerOriginY;
	int triggerWidth;
	int triggerHeight;
	int outputOriginX;
	int outputOriginY;
	int outputWidth;
	int outputHeight;
	int mode;
	int direction;
	int ordering;
	int criterion;
	int trigger;
	int affect;
	float cycleDegrees;
	int lineCount;
	int freePMin;
	int freeQMin;
	int freeLineLength;
	int radialLength;
	int domainStride;
	float thresholdMin;
	float thresholdMax;
	float angleCos;
	float angleSin;
	float centerX;
	float centerY;
	float swirlK;
	int swirlLineMin;
	int pathDirection;
	int pathClosed;
	float pathLength;
	int pathSMin;
	int pathNMin;
	int pathSampleCount;
};

struct BpsPathSampleGpu
{
	float x;
	float y;
	float s;
	float tx;
	float ty;
};

RWByteAddressBuffer sortTex : register(u0);
RWByteAddressBuffer keysTex : register(u1);
RWByteAddressBuffer mappedKeysTex : register(u2);
RWByteAddressBuffer pathRecordsTex : register(u3);
ByteAddressBuffer srcTex : register(t0);
ByteAddressBuffer criterionTex : register(t1);
ByteAddressBuffer triggerTex : register(t2);
ByteAddressBuffer pathTex : register(t3);
ByteAddressBuffer domainTex : register(t4);
ByteAddressBuffer mappedWorkOffsetsTex : register(t5);

BpsPathSampleGpu LoadPathSample(uint index)
{
	// Five tightly-packed floats (20 bytes) per sample.
	const uint base = index * 20u;
	BpsPathSampleGpu sample;
	sample.x = asfloat(pathTex.Load(base + 0u));
	sample.y = asfloat(pathTex.Load(base + 4u));
	sample.s = asfloat(pathTex.Load(base + 8u));
	sample.tx = asfloat(pathTex.Load(base + 12u));
	sample.ty = asfloat(pathTex.Load(base + 16u));
	return sample;
}

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

uint LoadMappedPixelIndex(uint index)
{
	return pathTex.Load(index * 8u + 4u);
}

uint LoadMappedLineOffset(uint index)
{
	return domainTex.Load(index * 4u);
}

uint LoadMappedWorkOffset(uint index)
{
	return mappedWorkOffsetsTex.Load(index * 4u);
}

float BpsSortKey(float4 c, int keyCriterion)
{
	// BGRA: R=.z, G=.y, B=.x
	if (keyCriterion == BPS_CRITERION_RGB_AVERAGE) {
		return saturate((c.z + c.y + c.x) * (1.0f / 3.0f));
	}
	if (keyCriterion == BPS_CRITERION_RGB_PRODUCT) {
		return saturate(c.z * c.y * c.x);
	}
	if (keyCriterion == BPS_CRITERION_RGB_MINIMUM) {
		return saturate(min(min(c.z, c.y), c.x));
	}
	if (keyCriterion == BPS_CRITERION_RGB_MAXIMUM) {
		return saturate(max(max(c.z, c.y), c.x));
	}
	if (keyCriterion == BPS_CRITERION_RED_CHANNEL) {
		return saturate(c.z);
	}
	if (keyCriterion == BPS_CRITERION_GREEN_CHANNEL) {
		return saturate(c.y);
	}
	if (keyCriterion == BPS_CRITERION_BLUE_CHANNEL) {
		return saturate(c.x);
	}
	if (keyCriterion == BPS_CRITERION_ALPHA_CHANNEL) {
		return saturate(c.w);
	}
	if (keyCriterion == BPS_CRITERION_HUE || keyCriterion == BPS_CRITERION_SATURATION) {
		const float r = saturate(c.z);
		const float g = saturate(c.y);
		const float b = saturate(c.x);
		const float maxRGB = max(max(r, g), b);
		const float minRGB = min(min(r, g), b);
		const float delta = maxRGB - minRGB;
		if (keyCriterion == BPS_CRITERION_SATURATION) {
			return maxRGB <= 0.0f ? 0.0f : saturate(delta / maxRGB);
		}
		if (delta <= 0.0f) {
			return 0.0f;
		}
		float hue = 0.0f;
		if (maxRGB == r) {
			hue = (g - b) / delta;
			if (hue < 0.0f) {
				hue += 6.0f;
			}
		} else if (maxRGB == g) {
			hue = ((b - r) / delta) + 2.0f;
		} else {
			hue = ((r - g) / delta) + 4.0f;
		}
		return saturate(hue * (1.0f / 6.0f));
	}
	return saturate(0.298912f * c.z + 0.586611f * c.y + 0.114478f * c.x);
}

bool BpsIsAffected(float triggerKey)
{
	const bool inside = triggerKey >= thresholdMin && triggerKey <= thresholdMax;
	return affect == BPS_AFFECT_OUTSIDE_THRESHOLDS ? !inside : inside;
}

bool CriterionInWorld(int x, int y)
{
	return x >= criterionOriginX && y >= criterionOriginY &&
		x < criterionOriginX + criterionWidth && y < criterionOriginY + criterionHeight;
}

bool TriggerInWorld(int x, int y)
{
	return x >= triggerOriginX && y >= triggerOriginY &&
		x < triggerOriginX + triggerWidth && y < triggerOriginY + triggerHeight;
}

uint CriterionIndexXY(int x, int y)
{
	return (uint)((x - criterionOriginX) + (y - criterionOriginY) * criterionPitch);
}

uint TriggerIndexXY(int x, int y)
{
	return (uint)((x - triggerOriginX) + (y - triggerOriginY) * triggerPitch);
}

float SampleCriterionKey(int x, int y)
{
	if (!CriterionInWorld(x, y)) {
		return -1.0f;
	}
	return BpsSortKey(LoadPixel(criterionTex, CriterionIndexXY(x, y)), criterion);
}

float SampleTriggerKey(int x, int y)
{
	if (!TriggerInWorld(x, y)) {
		return -1.0f;
	}
	return BpsSortKey(LoadPixel(triggerTex, TriggerIndexXY(x, y)), trigger);
}

float SampleCriterionKeyFromSrcIndex(uint srcIndex)
{
	if (srcIndex == 0xffffffffu) {
		return -1.0f;
	}
	const int x = (int)(srcIndex % (uint)srcPitch) + inputOriginX;
	const int y = (int)(srcIndex / (uint)srcPitch) + inputOriginY;
	return SampleCriterionKey(x, y);
}

float SampleTriggerKeyFromSrcIndex(uint srcIndex)
{
	if (srcIndex == 0xffffffffu) {
		return -1.0f;
	}
	const int x = (int)(srcIndex % (uint)srcPitch) + inputOriginX;
	const int y = (int)(srcIndex / (uint)srcPitch) + inputOriginY;
	return SampleTriggerKey(x, y);
}


uint BpsCycleShift(uint count)
{
	if (count <= 1u) {
		return 0u;
	}
	int shift = (int)floor((cycleDegrees / 360.0f) * (float)count + 0.5f);
	const int n = (int)count;
	shift %= n;
	if (shift < 0) {
		shift += n;
	}
	return (uint)shift;
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

void BpsNormaliseTangent(inout float tx, inout float ty)
{
	const float len = sqrt(tx * tx + ty * ty);
	if (len > 1.0e-6f) {
		tx /= len;
		ty /= len;
	} else {
		tx = 1.0f;
		ty = 0.0f;
	}
}

bool BpsPathEvalAtS(float s, out float x, out float y, out float tx, out float ty)
{
	if (pathSampleCount < 2) {
		x = 0.0f; y = 0.0f; tx = 1.0f; ty = 0.0f;
		return false;
	}
	const BpsPathSampleGpu first = LoadPathSample(0u);
	const BpsPathSampleGpu last = LoadPathSample((uint)(pathSampleCount - 1));
	if (s <= first.s) {
		x = first.x; y = first.y;
		tx = first.tx; ty = first.ty;
		return true;
	}
	if (s >= last.s) {
		x = last.x; y = last.y;
		tx = last.tx; ty = last.ty;
		return true;
	}
	for (int i = 0; i < pathSampleCount - 1; ++i) {
		const BpsPathSampleGpu a = LoadPathSample((uint)i);
		const BpsPathSampleGpu b = LoadPathSample((uint)(i + 1));
		if (s < a.s || s > b.s) {
			continue;
		}
		const float seg = b.s - a.s;
		const float t = seg > 1.0e-6f ? ((s - a.s) / seg) : 0.0f;
		x = a.x + (b.x - a.x) * t;
		y = a.y + (b.y - a.y) * t;
		tx = a.tx + (b.tx - a.tx) * t;
		ty = a.ty + (b.ty - a.ty) * t;
		BpsNormaliseTangent(tx, ty);
		return true;
	}
	x = 0.0f; y = 0.0f; tx = 1.0f; ty = 0.0f;
	return false;
}

bool BpsPathClosest(float px, float py, out float sOut, out float nOut)
{
	if (pathSampleCount < 2) {
		sOut = 0.0f;
		nOut = 0.0f;
		return false;
	}
	float bestDist2 = 1.0e30f;
	float bestS = LoadPathSample(0u).s;
	float bestN = 0.0f;
	for (int i = 0; i < pathSampleCount - 1; ++i) {
		const BpsPathSampleGpu a = LoadPathSample((uint)i);
		const BpsPathSampleGpu b = LoadPathSample((uint)(i + 1));
		const float abx = b.x - a.x;
		const float aby = b.y - a.y;
		const float abLen2 = abx * abx + aby * aby;
		float t = 0.0f;
		if (abLen2 > 1.0e-6f) {
			t = saturate(((px - a.x) * abx + (py - a.y) * aby) / abLen2);
		}
		const float cx = a.x + abx * t;
		const float cy = a.y + aby * t;
		const float dx = px - cx;
		const float dy = py - cy;
		const float dist2 = dx * dx + dy * dy;
		if (dist2 < bestDist2) {
			bestDist2 = dist2;
			bestS = a.s + (b.s - a.s) * t;
			float tx = a.tx + (b.tx - a.tx) * t;
			float ty = a.ty + (b.ty - a.ty) * t;
			BpsNormaliseTangent(tx, ty);
			bestN = -ty * dx + tx * dy;
		}
	}
	sOut = bestS;
	nOut = bestN;
	return true;
}

float BpsWrapArcLength(float s, float length)
{
	if (length <= 1.0e-6f) {
		return s;
	}
	float w = fmod(s, length);
	if (w < 0.0f) {
		w += length;
	}
	return w;
}

bool BpsPathLaneOrder(float s, float n, out int laneOut, out float order)
{
	const bool closed = pathClosed != 0 && pathLength > 1.0e-6f;
	const float sLocal = closed ? BpsWrapArcLength(s, pathLength) : s;
	if (pathDirection == BPS_PATH_DIR_TANGENT) {
		laneOut = BpsRoundToInt(n) - pathNMin;
		order = sLocal;
	} else {
		if (closed) {
			const int bins = lineCount > 0 ? lineCount : 1;
			int q = BpsRoundToInt(sLocal) % bins;
			if (q < 0) {
				q += bins;
			}
			laneOut = q;
		} else {
			laneOut = BpsRoundToInt(s) - pathSMin;
		}
		order = n;
	}
	return laneOut >= 0 && laneOut < lineCount;
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
	if (mode == BPS_MODE_SWIRL) {
		return (uint)(radialLength > 0 ? radialLength : 1);
	}
	if (mode == BPS_MODE_PATH) {
		return (uint)(radialLength > 0 ? radialLength : 1);
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
		const float c = cos(theta);
		const float s = sin(theta);
		const float radius = (float)gid;
		x = BpsRoundToInt(centerX + radius * (s * angleCos + c * angleSin));
		y = BpsRoundToInt(centerY + radius * (s * angleSin - c * angleCos));
	} else if (mode == BPS_MODE_RADIAL) {
		const float denom = lineCount <= 0 ? 1.0f : (float)lineCount;
		const float theta = (BPS_TWO_PI * (float)gid) / denom;
		x = BpsRoundToInt(centerX + (float)pos * cos(theta));
		y = BpsRoundToInt(centerY + (float)pos * sin(theta));
	} else if (mode == BPS_MODE_SWIRL) {
		const float r = (float)pos;
		const float denom = lineCount <= 0 ? 1.0f : (float)lineCount;
		const float phase = (BPS_TWO_PI * (float)gid) / denom;
		const float angle = atan2(angleSin, angleCos);
		const float theta = phase + swirlK * r + angle;
		x = BpsRoundToInt(centerX + r * cos(theta));
		y = BpsRoundToInt(centerY + r * sin(theta));
	} else if (mode == BPS_MODE_PATH) {
		float s = 0.0f;
		float n = 0.0f;
		if (pathDirection == BPS_PATH_DIR_NORMAL) {
			s = (float)(pathSMin + (int)gid);
			n = (float)(pathNMin + (int)pos);
		} else {
			n = (float)(pathNMin + (int)gid);
			s = (float)(pathSMin + (int)pos);
		}
		float px = 0.0f;
		float py = 0.0f;
		float tx = 1.0f;
		float ty = 0.0f;
		if (!BpsPathEvalAtS(s, px, py, tx, ty)) {
			x = 0;
			y = 0;
			return false;
		}
		x = BpsRoundToInt(px + n * (-ty));
		y = BpsRoundToInt(py + n * tx);
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
		x < inputOriginX + inputWidth && y < inputOriginY + inputHeight;
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

float BpsLuma(float4 c)
{
	return saturate(0.298912f * c.z + 0.586611f * c.y + 0.114478f * c.x);
}

int BpsAxisLineLayer(uint gid)
{
	return direction != 0 ? outputOriginY + (int)gid : outputOriginX + (int)gid;
}

void BpsAxisCoordForPos(uint gid, uint pos, out int x, out int y)
{
	const int lineLayer = BpsAxisLineLayer(gid);
	x = direction != 0 ? (int)pos : lineLayer;
	y = direction != 0 ? lineLayer : (int)pos;
}

uint BpsAxisSrcIndexPos(uint gid, uint pos)
{
	int x = 0;
	int y = 0;
	BpsAxisCoordForPos(gid, pos, x, y);
	return (uint)((x - inputOriginX) + (y - inputOriginY) * srcPitch);
}

uint BpsAxisDstIndexPos(uint gid, uint pos)
{
	int x = 0;
	int y = 0;
	BpsAxisCoordForPos(gid, pos, x, y);
	return (uint)((x - outputOriginX) + (y - outputOriginY) * dstPitch);
}

void BpsAxisLumaSortCore(
	uint gid,
	uint gtid,
	uint size,
	bool fullSpan)
{
	uint cursor = 0u;
	while (cursor < size) {
		if (gtid == 0u) {
			uint spanStart = cursor;
			while (spanStart < size) {
				float br = 0.0f;
				if (fullSpan) {
					br = BpsLuma(LoadPixel(srcTex, BpsAxisSrcIndexPos(gid, spanStart)));
				} else {
					int x = 0;
					int y = 0;
					BpsAxisCoordForPos(gid, spanStart, x, y);
					if (x >= 0 && y >= 0 && x < width && y < height && SrcInWorld(x, y)) {
						br = BpsLuma(LoadPixel(srcTex, SrcIndexXY(x, y)));
					} else {
						br = thresholdMax + 1.0f;
					}
				}
				if (br >= thresholdMin && br <= thresholdMax) {
					break;
				}
				spanStart++;
			}

			uint spanEnd = spanStart;
			while (spanEnd < size) {
				float br = 0.0f;
				if (fullSpan) {
					br = BpsLuma(LoadPixel(srcTex, BpsAxisSrcIndexPos(gid, spanEnd)));
				} else {
					int x = 0;
					int y = 0;
					BpsAxisCoordForPos(gid, spanEnd, x, y);
					if (x < 0 || y < 0 || x >= width || y >= height || !SrcInWorld(x, y)) {
						br = thresholdMax + 1.0f;
					} else {
						br = BpsLuma(LoadPixel(srcTex, SrcIndexXY(x, y)));
					}
				}
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

		GroupMemoryBarrierWithGroupSync();

		if (spanStart >= size || spanSize == 0u) {
			break;
		}

		const bool ascending = ordering != 0;
		for (uint loadIndex = gtid; loadIndex < sortSize; loadIndex += MAX_THREADS) {
			if (loadIndex < spanSize) {
				const uint pos = spanStart + loadIndex;
				uint srcIndex = 0xffffffffu;
				float key = ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX;
				if (fullSpan) {
					srcIndex = BpsAxisSrcIndexPos(gid, pos);
					key = BpsLuma(LoadPixel(srcTex, srcIndex));
				} else {
					int x = 0;
					int y = 0;
					BpsAxisCoordForPos(gid, pos, x, y);
					const bool valid =
						x >= 0 && y >= 0 && x < width && y < height && SrcInWorld(x, y);
					srcIndex = valid ? SrcIndexXY(x, y) : 0xffffffffu;
					key = valid
						? BpsLuma(LoadPixel(srcTex, srcIndex))
						: (ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX);
				}
				scratchKey[loadIndex] = key;
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
			const uint srcIndex = scratchIndex[writeIndex];
			if (srcIndex == 0xffffffffu) {
				continue;
			}
			if (fullSpan) {
				StorePixel(sortTex, BpsAxisDstIndexPos(gid, pos), LoadPixel(srcTex, srcIndex));
			} else {
				int x = 0;
				int y = 0;
				BpsAxisCoordForPos(gid, pos, x, y);
				if (x >= 0 && y >= 0 && x < width && y < height && DstInWorld(x, y)) {
					StorePixel(sortTex, DstIndexXY(x, y), LoadPixel(srcTex, srcIndex));
				}
			}
		}
		GroupMemoryBarrierWithGroupSync();

		cursor = spanEnd + 1u;
	}
}

[RootSignature("DescriptorTable(CBV(b0,numDescriptors=1)),DescriptorTable(UAV(u0,numDescriptors=1)),DescriptorTable(SRV(t0,numDescriptors=1))")]
[numthreads(256, 1, 1)]
void SortAxisLuma(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{
	const uint gid = groupID.x;
	const uint gtid = groupThreadID.x;
	const uint size = direction != 0u ? (uint)width : (uint)height;
	if (size == 0u || size > MAX_SIZE) {
		return;
	}

	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		int x = 0;
		int y = 0;
		BpsAxisCoordForPos(gid, pos, x, y);
		if (x >= 0 && y >= 0 && x < width && y < height && DstInWorld(x, y)) {
			StorePixel(sortTex, DstIndexXY(x, y),
				SrcInWorld(x, y) ? LoadPixel(srcTex, SrcIndexXY(x, y))
								 : float4(0.0f, 0.0f, 0.0f, 0.0f));
		}
	}
	GroupMemoryBarrierWithGroupSync();

	BpsAxisLumaSortCore(gid, gtid, size, false);
}

[RootSignature("DescriptorTable(CBV(b0,numDescriptors=1)),DescriptorTable(UAV(u0,numDescriptors=1)),DescriptorTable(SRV(t0,numDescriptors=1))")]
[numthreads(256, 1, 1)]
void SortAxisLumaFull(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{
	const uint gid = groupID.x;
	const uint gtid = groupThreadID.x;
	const uint size = direction != 0u ? (uint)width : (uint)height;
	if (size == 0u || size > MAX_SIZE) {
		return;
	}

	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		StorePixel(sortTex, BpsAxisDstIndexPos(gid, pos),
			LoadPixel(srcTex, BpsAxisSrcIndexPos(gid, pos)));
	}
	GroupMemoryBarrierWithGroupSync();

	BpsAxisLumaSortCore(gid, gtid, size, true);
}

uint DomainIndex(uint gid, uint pos)
{
	return (uint)((int)gid * domainStride + (int)pos);
}

void StoreUint(RWByteAddressBuffer buf, uint index, uint value)
{
	buf.Store(index * 4u, value);
}

uint LoadUint(ByteAddressBuffer buf, uint index)
{
	return buf.Load(index * 4u);
}

uint LoadUintUAV(RWByteAddressBuffer buf, uint index)
{
	return buf.Load(index * 4u);
}

void StoreFloat(RWByteAddressBuffer buf, uint index, float value)
{
	buf.Store(index * 4u, asuint(value));
}

float LoadFloat(RWByteAddressBuffer buf, uint index)
{
	return asfloat(buf.Load(index * 4u));
}

void StoreMappedRecord(RWByteAddressBuffer buf, uint index, float posKey, uint pixelIndex)
{
	const uint base = index * 8u;
	buf.Store(base + 0u, asuint(posKey));
	buf.Store(base + 4u, pixelIndex);
}

float LoadMappedRecordKey(RWByteAddressBuffer buf, uint index)
{
	return asfloat(buf.Load(index * 8u));
}

uint LoadMappedRecordPixel(RWByteAddressBuffer buf, uint index)
{
	return buf.Load(index * 8u + 4u);
}

bool BpsRecordBefore(float keyA, uint pixelA, float keyB, uint pixelB)
{
	if (keyA < keyB) {
		return true;
	}
	if (keyA > keyB) {
		return false;
	}
	return pixelA < pixelB;
}

bool BpsDomainPosForPixel(int x, int y, out int domainLine, out int domainPos)
{
	domainLine = 0;
	domainPos = 0;

	if (mode == BPS_MODE_FREE_ANGLE) {
		const float p = (float)x * angleCos + (float)y * angleSin;
		const float q = -(float)x * angleSin + (float)y * angleCos;
		domainPos = BpsRoundToInt(p) - freePMin;
		domainLine = BpsRoundToInt(q) - freeQMin;
	} else if (mode == BPS_MODE_ROTATION) {
		const float dx = (float)x - centerX;
		const float dy = (float)y - centerY;
		domainLine = BpsRoundToInt(sqrt(dx * dx + dy * dy));
		const uint lineLen = BpsRotationLineLength((uint)max(domainLine, 0));
		if (lineLen <= 1u) {
			domainPos = 0;
		} else {
			// Inverse of (sin, -cos) basis: theta = atan2(rx, -ry).
			const float rx = dx * angleCos + dy * angleSin;
			const float ry = -dx * angleSin + dy * angleCos;
			float theta = atan2(rx, -ry);
			if (theta < 0.0f) {
				theta += BPS_TWO_PI;
			}
			domainPos = BpsRoundToInt((theta / BPS_TWO_PI) * (float)lineLen);
			if (domainPos >= (int)lineLen) {
				domainPos -= (int)lineLen;
			}
		}
	} else if (mode == BPS_MODE_RADIAL) {
		const float dx = (float)x - centerX;
		const float dy = (float)y - centerY;
		domainPos = BpsRoundToInt(sqrt(dx * dx + dy * dy));
		float theta = atan2(dy, dx);
		if (theta < 0.0f) {
			theta += BPS_TWO_PI;
		}
		domainLine = lineCount <= 1 ? 0 :
			BpsRoundToInt((theta / BPS_TWO_PI) * (float)lineCount);
		if (domainLine >= lineCount) {
			domainLine -= lineCount;
		}
	} else if (mode == BPS_MODE_SWIRL) {
		const float dx = (float)x - centerX;
		const float dy = (float)y - centerY;
		const float r = sqrt(dx * dx + dy * dy);
		const float angle = atan2(angleSin, angleCos);
		float phase = atan2(dy, dx) - angle - swirlK * r;
		phase = fmod(phase, BPS_TWO_PI);
		if (phase < 0.0f) {
			phase += BPS_TWO_PI;
		}
		domainLine = lineCount <= 1 ? 0 :
			BpsRoundToInt((phase / BPS_TWO_PI) * (float)lineCount);
		if (domainLine >= lineCount) {
			domainLine -= lineCount;
		}
		domainPos = BpsRoundToInt(r);
	} else if (mode == BPS_MODE_PATH) {
		float s = 0.0f;
		float n = 0.0f;
		if (!BpsPathClosest((float)x, (float)y, s, n)) {
			return false;
		}
		if (pathDirection == BPS_PATH_DIR_NORMAL) {
			domainLine = BpsRoundToInt(s) - pathSMin;
			domainPos = BpsRoundToInt(n) - pathNMin;
		} else {
			domainLine = BpsRoundToInt(n) - pathNMin;
			domainPos = BpsRoundToInt(s) - pathSMin;
		}
	} else {
		return false;
	}

	if (domainLine < 0 || domainLine >= lineCount) {
		return false;
	}

	int lineLen = 0;
	if (mode == BPS_MODE_FREE_ANGLE) {
		lineLen = freeLineLength;
	} else if (mode == BPS_MODE_ROTATION) {
		lineLen = (int)BpsRotationLineLength((uint)domainLine);
	} else {
		lineLen = radialLength;
	}

	if (domainPos < 0 || domainPos >= lineLen || domainPos >= domainStride) {
		return false;
	}
	return true;
}

[RootSignature("DescriptorTable(CBV(b0,numDescriptors=1)),DescriptorTable(UAV(u0,numDescriptors=1)),DescriptorTable(SRV(t0,numDescriptors=4))")]
[numthreads(256, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{
	const uint gid = groupID.x;
	const uint gtid = groupThreadID.x;
	const uint size = BpsLineSize(gid);
	if (size == 0u || size > MAX_SIZE) {
		return;
	}

	// Always write every destination pixel. Partial input worlds (common with
	// alpha / adjustment layers) must not leave stale frame data behind.
	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		int x = 0;
		int y = 0;
		if (BpsCoordForPos(gid, pos, x, y) && DstInWorld(x, y)) {
			StorePixel(sortTex, DstIndexXY(x, y),
				SrcInWorld(x, y) ? LoadPixel(srcTex, SrcIndexXY(x, y))
								 : float4(0.0f, 0.0f, 0.0f, 0.0f));
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
					float br = SampleTriggerKey(x, y);
					if (BpsIsAffected(br)) {
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
				float br = SampleTriggerKey(x, y);
				if (!BpsIsAffected(br)) {
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
					? SampleCriterionKey(x, y)
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
		const uint shift = BpsCycleShift(spanSize);
		const uint sortedIndex = (writeIndex + spanSize - shift) % spanSize;
		int x = 0;
		int y = 0;
		if (BpsCoordForPos(gid, pos, x, y) &&
			DstInWorld(x, y) && scratchIndex[sortedIndex] != 0xffffffffu) {
			StorePixel(sortTex, DstIndexXY(x, y), LoadPixel(srcTex, scratchIndex[sortedIndex]));
		}
	}
		GroupMemoryBarrierWithGroupSync();

		cursor = spanEnd + 1u;
	}
}

// Non-axis pass 1: forward path sampling into domain, then sort runs.
groupshared uint s_spanStart;
groupshared uint s_spanEnd;
groupshared uint s_spanSize;
groupshared uint s_sortSize;
groupshared uint s_pathLen;

[RootSignature("DescriptorTable(CBV(b0,numDescriptors=1)),DescriptorTable(UAV(u0,numDescriptors=2)),DescriptorTable(SRV(t0,numDescriptors=4))")]
[numthreads(256, 1, 1)]
void SortDomain(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{
	const uint gid = groupID.x;
	const uint gtid = groupThreadID.x;
	const uint size = BpsLineSize(gid);
	if (size == 0u || domainStride <= 0 || (int)size > domainStride) {
		return;
	}

	for (uint pos = gtid; pos < size; pos += MAX_THREADS) {
		int x = 0;
		int y = 0;
		const bool valid = BpsCoordForPos(gid, pos, x, y) && SrcInWorld(x, y);
		StoreUint(sortTex, DomainIndex(gid, pos), valid ? SrcIndexXY(x, y) : 0xffffffffu);
	}
	AllMemoryBarrierWithGroupSync();

	if (gtid == 0u) {
		uint pathLen = 0u;
		for (uint pos = 0u; pos < size; ++pos) {
			if (LoadUintUAV(sortTex, DomainIndex(gid, pos)) != 0xffffffffu) {
				StoreFloat(keysTex, DomainIndex(gid, pathLen), asfloat(pos));
				pathLen++;
			}
		}
		s_pathLen = pathLen;
	}
	AllMemoryBarrierWithGroupSync();

	const uint pathLen = s_pathLen;
	uint cursor = 0u;
	while (cursor < pathLen) {
		if (gtid == 0u) {
			uint runStart = cursor;
			while (runStart < pathLen) {
				const uint pos = asuint(LoadFloat(keysTex, DomainIndex(gid, runStart)));
				const uint srcIndex = LoadUintUAV(sortTex, DomainIndex(gid, pos));
				const float br = SampleTriggerKeyFromSrcIndex(srcIndex);
				if (BpsIsAffected(br)) {
					break;
				}
				runStart++;
			}
			uint runEnd = runStart;
			while (runEnd < pathLen) {
				const uint pos = asuint(LoadFloat(keysTex, DomainIndex(gid, runEnd)));
				const uint srcIndex = LoadUintUAV(sortTex, DomainIndex(gid, pos));
				const float br = SampleTriggerKeyFromSrcIndex(srcIndex);
				if (!BpsIsAffected(br)) {
					break;
				}
				runEnd++;
			}
			s_spanStart = runStart;
			s_spanEnd = runEnd;
			s_spanSize = runEnd - runStart;
			s_sortSize = BpsNextPow2(s_spanSize);
		}
		GroupMemoryBarrierWithGroupSync();

		const uint runStart = s_spanStart;
		const uint runEnd = s_spanEnd;
		const uint spanSize = s_spanSize;
		const uint sortSize = s_sortSize;

		if (runStart >= pathLen || spanSize == 0u) {
			break;
		}

		const bool ascending = ordering != 0;
		for (uint loadIndex = gtid; loadIndex < sortSize; loadIndex += MAX_THREADS) {
			const uint work = DomainIndex(gid, size + loadIndex);
			if (loadIndex < spanSize) {
				const uint pos = asuint(LoadFloat(keysTex, DomainIndex(gid, runStart + loadIndex)));
				const uint srcIndex = LoadUintUAV(sortTex, DomainIndex(gid, pos));
				StoreUint(sortTex, work, srcIndex);
				StoreFloat(keysTex, work, SampleCriterionKeyFromSrcIndex(srcIndex));
			} else {
				StoreUint(sortTex, work, 0xffffffffu);
				StoreFloat(keysTex, work, ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX);
			}
		}
		AllMemoryBarrierWithGroupSync();

		for (uint k = 2u; k <= sortSize; k <<= 1) {
			for (uint j = k >> 1; j > 0u; j >>= 1) {
				for (uint sortIndex = gtid; sortIndex < sortSize; sortIndex += MAX_THREADS) {
					const uint partner = sortIndex ^ j;
					if (partner > sortIndex) {
						bool stageAscending = (sortIndex & k) == 0u;
						if (!ascending) {
							stageAscending = !stageAscending;
						}

						const uint slotA = DomainIndex(gid, size + sortIndex);
						const uint slotB = DomainIndex(gid, size + partner);
						const float keyA = LoadFloat(keysTex, slotA);
						const float keyB = LoadFloat(keysTex, slotB);
						const uint indexA = LoadUintUAV(sortTex, slotA);
						const uint indexB = LoadUintUAV(sortTex, slotB);
						const bool before = BpsBefore(keyA, indexA, keyB, indexB);
						if (before != stageAscending) {
							StoreFloat(keysTex, slotA, keyB);
							StoreFloat(keysTex, slotB, keyA);
							StoreUint(sortTex, slotA, indexB);
							StoreUint(sortTex, slotB, indexA);
						}
					}
				}
				AllMemoryBarrierWithGroupSync();
			}
		}

		const uint shift = BpsCycleShift(spanSize);
		// Flag reordered slots (run length >= 2) so ApplyDomain composites only
		// those over the source and leaves unsorted areas pristine.
		const uint affectedBit = (spanSize >= 2u) ? 0x80000000u : 0u;
		for (uint writeIndex = gtid; writeIndex < spanSize; writeIndex += MAX_THREADS) {
			const uint sortedIndex = (writeIndex + spanSize - shift) % spanSize;
			const uint pos = asuint(LoadFloat(keysTex, DomainIndex(gid, runStart + writeIndex)));
			const uint srcIndex = LoadUintUAV(sortTex, DomainIndex(gid, size + sortedIndex));
			StoreUint(sortTex, DomainIndex(gid, pos), srcIndex | affectedBit);
		}
		AllMemoryBarrierWithGroupSync();

		cursor = runEnd + 1u;
	}
}

// Non-axis pass 2: gather via inverse domain lookup.
[RootSignature("DescriptorTable(CBV(b0,numDescriptors=1)),DescriptorTable(UAV(u0,numDescriptors=1)),DescriptorTable(SRV(t0,numDescriptors=5))")]
[numthreads(16, 16, 1)]
void ApplyDomain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	const int ox = (int)dispatchThreadID.x;
	const int oy = (int)dispatchThreadID.y;
	if (ox >= outputWidth || oy >= outputHeight) {
		return;
	}

	const int x = outputOriginX + ox;
	const int y = outputOriginY + oy;
	const uint dstIndex = (uint)(ox + oy * dstPitch);

	float4 pixel = float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (SrcInWorld(x, y)) {
		pixel = LoadPixel(srcTex, SrcIndexXY(x, y));
	}

	int domainLine = 0;
	int domainPos = 0;
	if (BpsDomainPosForPixel(x, y, domainLine, domainPos)) {
		const uint raw = LoadUint(domainTex, (uint)(domainLine * domainStride + domainPos));
		// High bit flags slots a sort reordered; composite only those over the
		// source so unsorted areas keep the exact original (no resample loss).
		if (raw != 0xffffffffu && (raw & 0x80000000u) != 0u) {
			pixel = LoadPixel(srcTex, raw & 0x7fffffffu);
		}
	}

	StorePixel(sortTex, dstIndex, pixel);
}

[RootSignature("DescriptorTable(CBV(b0,numDescriptors=1)),DescriptorTable(UAV(u0,numDescriptors=1)),DescriptorTable(SRV(t0,numDescriptors=1))")]
[numthreads(16, 16, 1)]
void CopyInput(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	const int ox = (int)dispatchThreadID.x;
	const int oy = (int)dispatchThreadID.y;
	if (ox >= outputWidth || oy >= outputHeight) {
		return;
	}

	const int x = outputOriginX + ox;
	const int y = outputOriginY + oy;
	const uint dstIndex = (uint)(ox + oy * dstPitch);
	float4 pixel = float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (SrcInWorld(x, y)) {
		pixel = LoadPixel(srcTex, SrcIndexXY(x, y));
	}
	StorePixel(sortTex, dstIndex, pixel);
}

struct BpsJfaCellGpu
{
	float qx;
	float qy;
	float s;
	float tx;
	float ty;
	float d2;
};

BpsJfaCellGpu LoadJfaCell(uint index)
{
	const uint base = index * 24u;
	BpsJfaCellGpu cell;
	cell.qx = asfloat(pathTex.Load(base + 0u));
	cell.qy = asfloat(pathTex.Load(base + 4u));
	cell.s = asfloat(pathTex.Load(base + 8u));
	cell.tx = asfloat(pathTex.Load(base + 12u));
	cell.ty = asfloat(pathTex.Load(base + 16u));
	cell.d2 = asfloat(pathTex.Load(base + 20u));
	return cell;
}

[RootSignature("DescriptorTable(CBV(b0,numDescriptors=1)),DescriptorTable(UAV(u0,numDescriptors=3)),DescriptorTable(SRV(t0,numDescriptors=4))")]
[numthreads(16, 16, 1)]
void BuildPathClassifyCount(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	const int x = (int)dispatchThreadID.x;
	const int y = (int)dispatchThreadID.y;
	if (x >= width || y >= height) {
		return;
	}

	const int gridW = freePMin;
	const int gridH = freeQMin;
	const int jfaFactor = swirlLineMin;

	const uint pidx = (uint)(y * width + x);
	sortTex.Store(pidx * 4u, 0xffffffffu);

	const int cx = min(x / jfaFactor, gridW - 1);
	const int cy = min(y / jfaFactor, gridH - 1);
	const BpsJfaCellGpu cell = LoadJfaCell((uint)(cy * gridW + cx));
	if (!isfinite(cell.d2)) {
		return;
	}

	const float dxp = (float)x - cell.qx;
	const float dyp = (float)y - cell.qy;
	const float n = -cell.ty * dxp + cell.tx * dyp;
	int lane = 0;
	float order = 0.0f;
	if (!BpsPathLaneOrder(cell.s, n, lane, order)) {
		return;
	}

	sortTex.Store(pidx * 4u, asuint(lane));
	keysTex.Store(pidx * 4u, asuint(order));
	uint ignored = 0u;
	mappedKeysTex.InterlockedAdd((uint)lane * 4u, 1u, ignored);
}

[RootSignature("DescriptorTable(CBV(b0,numDescriptors=1)),DescriptorTable(UAV(u0,numDescriptors=4))")]
[numthreads(16, 16, 1)]
void BuildPathScatterRecords(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	const int x = (int)dispatchThreadID.x;
	const int y = (int)dispatchThreadID.y;
	if (x >= width || y >= height) {
		return;
	}

	const uint pidx = (uint)(y * width + x);
	const int lane = asint(sortTex.Load(pidx * 4u));
	if (lane < 0) {
		return;
	}

	uint dst = 0u;
	mappedKeysTex.InterlockedAdd((uint)lane * 4u, 1u, dst);
	StoreMappedRecord(pathRecordsTex, dst, asfloat(keysTex.Load(pidx * 4u)), pidx);
}

[RootSignature("DescriptorTable(CBV(b0,numDescriptors=1)),DescriptorTable(UAV(u0,numDescriptors=2)),DescriptorTable(SRV(t0,numDescriptors=4))")]
[numthreads(256, 1, 1)]
void BuildPathSortRecords(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{
	const uint mapLine = groupID.x;
	const uint gtid = groupThreadID.x;
	if ((int)mapLine >= lineCount) {
		return;
	}

	const uint begin = LoadUint(srcTex, mapLine);
	const uint end = LoadUint(srcTex, mapLine + 1u);
	const uint lineSize = end - begin;
	if (lineSize <= 1u) {
		return;
	}

	const uint sortSize = BpsNextPow2(lineSize);
	const uint sortBase = LoadUint(pathTex, mapLine) + lineSize;
	for (uint loadIndex = gtid; loadIndex < sortSize; loadIndex += MAX_THREADS) {
		if (loadIndex < lineSize) {
			StoreMappedRecord(keysTex, sortBase + loadIndex,
				LoadMappedRecordKey(sortTex, begin + loadIndex),
				LoadMappedRecordPixel(sortTex, begin + loadIndex));
		} else {
			StoreMappedRecord(keysTex, sortBase + loadIndex, BPS_FLOAT_MAX, 0xffffffffu);
		}
	}
	AllMemoryBarrierWithGroupSync();

	for (uint k = 2u; k <= sortSize; k <<= 1) {
		for (uint j = k >> 1; j > 0u; j >>= 1) {
			for (uint sortIndex = gtid; sortIndex < sortSize; sortIndex += MAX_THREADS) {
				const uint partner = sortIndex ^ j;
				if (partner > sortIndex) {
					const uint slotA = sortBase + sortIndex;
					const uint slotB = sortBase + partner;
					const float keyA = LoadMappedRecordKey(keysTex, slotA);
					const float keyB = LoadMappedRecordKey(keysTex, slotB);
					const uint pixelA = LoadMappedRecordPixel(keysTex, slotA);
					const uint pixelB = LoadMappedRecordPixel(keysTex, slotB);
					const bool stageAscending = (sortIndex & k) == 0u;
					const bool before = BpsRecordBefore(keyA, pixelA, keyB, pixelB);
					if (before != stageAscending) {
						StoreMappedRecord(keysTex, slotA, keyB, pixelB);
						StoreMappedRecord(keysTex, slotB, keyA, pixelA);
					}
				}
			}
			AllMemoryBarrierWithGroupSync();
		}
	}

	for (uint writeIndex = gtid; writeIndex < lineSize; writeIndex += MAX_THREADS) {
		StoreMappedRecord(sortTex, begin + writeIndex,
			LoadMappedRecordKey(keysTex, sortBase + writeIndex),
			LoadMappedRecordPixel(keysTex, sortBase + writeIndex));
	}
}

[RootSignature("DescriptorTable(CBV(b0,numDescriptors=1)),DescriptorTable(UAV(u0,numDescriptors=3)),DescriptorTable(SRV(t0,numDescriptors=6))")]
[numthreads(256, 1, 1)]
void SortMapped(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{
	const uint mappedLine = groupID.x;
	const uint gtid = groupThreadID.x;
	if ((int)mappedLine >= lineCount) {
		return;
	}

	const uint begin = LoadMappedLineOffset(mappedLine);
	const uint end = LoadMappedLineOffset(mappedLine + 1u);
	const uint lineSize = end - begin;
	if (lineSize == 0u) {
		return;
	}

	const uint workBase = LoadMappedWorkOffset(mappedLine);
	const bool ascending = ordering != 0;

	for (uint i = gtid; i < lineSize; i += MAX_THREADS) {
		const uint pixelIndex = LoadMappedPixelIndex(begin + i);
		const int x = (int)(pixelIndex % (uint)width);
		const int y = (int)(pixelIndex / (uint)width);
		StoreUint(keysTex, workBase + i,
			SrcInWorld(x, y) ? SrcIndexXY(x, y) : 0xffffffffu);
	}
	AllMemoryBarrierWithGroupSync();

	uint cursor = 0u;
	while (cursor < lineSize) {
		if (gtid == 0u) {
			uint runStart = cursor;
			while (runStart < lineSize) {
				const uint srcIndex = LoadUintUAV(keysTex, workBase + runStart);
				if (srcIndex != 0xffffffffu) {
					const float br = SampleTriggerKeyFromSrcIndex(srcIndex);
					if (BpsIsAffected(br)) {
						break;
					}
				}
				runStart++;
			}

			uint runEnd = runStart;
			while (runEnd < lineSize) {
				const uint srcIndex = LoadUintUAV(keysTex, workBase + runEnd);
				if (srcIndex == 0xffffffffu) {
					break;
				}
				const float br = SampleTriggerKeyFromSrcIndex(srcIndex);
				if (!BpsIsAffected(br)) {
					break;
				}
				runEnd++;
			}

			s_spanStart = runStart;
			s_spanEnd = runEnd;
			s_spanSize = runEnd - runStart;
			s_sortSize = BpsNextPow2(s_spanSize);
		}
		GroupMemoryBarrierWithGroupSync();

		const uint runStart = s_spanStart;
		const uint runEnd = s_spanEnd;
		const uint spanSize = s_spanSize;
		const uint sortSize = s_sortSize;
		if (runStart >= lineSize || spanSize == 0u) {
			break;
		}

		const uint sortBase = workBase + lineSize;
		for (uint loadIndex = gtid; loadIndex < sortSize; loadIndex += MAX_THREADS) {
			if (loadIndex < spanSize) {
				const uint srcIndex = LoadUintUAV(keysTex, workBase + runStart + loadIndex);
				StoreUint(keysTex, sortBase + loadIndex, srcIndex);
				StoreFloat(mappedKeysTex, sortBase + loadIndex,
					SampleCriterionKeyFromSrcIndex(srcIndex));
			} else {
				StoreUint(keysTex, sortBase + loadIndex, 0xffffffffu);
				StoreFloat(mappedKeysTex, sortBase + loadIndex,
					ascending ? BPS_FLOAT_MAX : -BPS_FLOAT_MAX);
			}
		}
		AllMemoryBarrierWithGroupSync();

		for (uint k = 2u; k <= sortSize; k <<= 1) {
			for (uint j = k >> 1; j > 0u; j >>= 1) {
				for (uint sortIndex = gtid; sortIndex < sortSize; sortIndex += MAX_THREADS) {
					const uint partner = sortIndex ^ j;
					if (partner > sortIndex) {
						bool stageAscending = (sortIndex & k) == 0u;
						if (!ascending) {
							stageAscending = !stageAscending;
						}
						const float keyA = LoadFloat(mappedKeysTex, sortBase + sortIndex);
						const float keyB = LoadFloat(mappedKeysTex, sortBase + partner);
						const uint indexA = LoadUintUAV(keysTex, sortBase + sortIndex);
						const uint indexB = LoadUintUAV(keysTex, sortBase + partner);
						const bool before = BpsBefore(keyA, indexA, keyB, indexB);
						if (before != stageAscending) {
							StoreFloat(mappedKeysTex, sortBase + sortIndex, keyB);
							StoreFloat(mappedKeysTex, sortBase + partner, keyA);
							StoreUint(keysTex, sortBase + sortIndex, indexB);
							StoreUint(keysTex, sortBase + partner, indexA);
						}
					}
				}
				AllMemoryBarrierWithGroupSync();
			}
		}

		const uint shift = BpsCycleShift(spanSize);
		for (uint writeIndex = gtid; writeIndex < spanSize; writeIndex += MAX_THREADS) {
			const uint sortedIndex = (writeIndex + spanSize - shift) % spanSize;
			const uint srcIndex = LoadUintUAV(keysTex, sortBase + sortedIndex);
			const uint pixelIndex = LoadMappedPixelIndex(begin + runStart + writeIndex);
			const int x = (int)(pixelIndex % (uint)width);
			const int y = (int)(pixelIndex / (uint)width);
			if (srcIndex != 0xffffffffu && DstInWorld(x, y)) {
				StorePixel(sortTex, DstIndexXY(x, y), LoadPixel(srcTex, srcIndex));
			}
		}
		AllMemoryBarrierWithGroupSync();

		cursor = runEnd + 1u;
	}
}
