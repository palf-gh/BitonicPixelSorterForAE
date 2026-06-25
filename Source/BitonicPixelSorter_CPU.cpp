/*
	BitonicPixelSorter_CPU.cpp

	CPU implementation of the pixel sort. This is the correctness oracle and the
	GPU fallback. It reproduces the upstream effect's semantics without the GPU's
	group-shared-memory line-length limit, and across 8/16/32-bit.

	The GPU uses a bitonic sorting network purely to parallelise the sort; the
	visible result is "sort each contiguous in-threshold span of a line by
	brightness", which on the CPU is a plain std::stable_sort per span.
*/

#include "BitonicPixelSorter.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

inline float Saturate(float v) {
	return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// Luma weights matching the upstream shader: 0.298912 R + 0.586611 G + 0.114478 B.
inline float Luma(float r, float g, float b) {
	return Saturate(0.298912f * r + 0.586611f * g + 0.114478f * b);
}

inline float BrightnessUnit(const PF_Pixel8 &p) {
	const float inv = 1.0f / 255.0f;
	return Luma(p.red * inv, p.green * inv, p.blue * inv);
}

inline float BrightnessUnit(const PF_Pixel16 &p) {
	// After Effects 16-bit channels are 0..32768.
	const float inv = 1.0f / 32768.0f;
	return Luma(p.red * inv, p.green * inv, p.blue * inv);
}

inline float BrightnessUnit(const PF_PixelFloat &p) {
	return Luma(p.red, p.green, p.blue);
}

template <typename T>
inline T *PixelAt(PF_EffectWorld *worldP, A_long x, A_long y) {
	return reinterpret_cast<T *>(reinterpret_cast<char *>(worldP->data) +
								 static_cast<size_t>(y) * worldP->rowbytes) + x;
}

template <typename T>
struct Entry {
	float	key;
	T		pix;
};

// Sort each contiguous in-threshold span of every line by brightness.
//
// Lines are rows when sorting horizontally and columns when sorting vertically.
// Phase 1 assumes the input world covers the full frame and the output world is
// the full frame aligned at the origin (PreRender requests the full input and
// reports a full result rect). Reads are bounds-checked against the input world.
template <typename T>
static void SortLines(PF_EffectWorld *inP, PF_EffectWorld *outP,
					  const BitonicSorterParams &prm) {
	const bool	horizontal = (prm.direction == BPS_DIR_HORIZONTAL);
	const bool	ascending  = (prm.ascending != 0);
	const float	tmin = prm.thresholdMin;
	const float	tmax = prm.thresholdMax;

	const A_long outW = outP->width;
	const A_long outH = outP->height;
	const A_long inW  = inP->width;
	const A_long inH  = inP->height;

	const A_long lineCount = horizontal ? outH : outW;
	const A_long lineLen   = horizontal ? outW : outH;

	std::vector<Entry<T>> run;
	run.reserve(static_cast<size_t>(lineLen));

	for (A_long line = 0; line < lineCount; ++line) {
		// Map line position k -> (x, y) for both orientations.
		auto coord = [&](A_long k, A_long &x, A_long &y) {
			if (horizontal) { x = k; y = line; }
			else            { x = line; y = k; }
		};

		// Pass-through copy first; spans overwrite with sorted order below.
		for (A_long k = 0; k < lineLen; ++k) {
			A_long x, y;
			coord(k, x, y);
			if (x < inW && y < inH) {
				*PixelAt<T>(outP, x, y) = *PixelAt<T>(inP, x, y);
			}
		}

		A_long k = 0;
		while (k < lineLen) {
			A_long x, y;
			coord(k, x, y);
			const bool inBounds = (x < inW && y < inH);
			float br = inBounds ? BrightnessUnit(*PixelAt<T>(inP, x, y)) : -1.0f;

			if (inBounds && br >= tmin && br <= tmax) {
				const A_long start = k;
				run.clear();
				while (k < lineLen) {
					coord(k, x, y);
					if (!(x < inW && y < inH)) break;
					T *src = PixelAt<T>(inP, x, y);
					float b = BrightnessUnit(*src);
					if (b < tmin || b > tmax) break;
					run.push_back(Entry<T>{b, *src});
					++k;
				}

				if (ascending) {
					std::stable_sort(run.begin(), run.end(),
						[](const Entry<T> &a, const Entry<T> &b) { return a.key < b.key; });
				} else {
					std::stable_sort(run.begin(), run.end(),
						[](const Entry<T> &a, const Entry<T> &b) { return a.key > b.key; });
				}

				for (size_t i = 0; i < run.size(); ++i) {
					A_long ox, oy;
					coord(start + static_cast<A_long>(i), ox, oy);
					*PixelAt<T>(outP, ox, oy) = run[i].pix;
				}
			} else {
				++k;
			}
		}
	}
}

} // namespace

PF_Err BPS_SortImageCPU(
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_PixelFormat		pixel_format,
	PF_EffectWorld		*input_worldP,
	PF_EffectWorld		*output_worldP,
	const BitonicSorterParams *paramsP)
{
	if (!input_worldP || !output_worldP || !paramsP) {
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	switch (pixel_format) {
	case PF_PixelFormat_ARGB128:
		SortLines<PF_PixelFloat>(input_worldP, output_worldP, *paramsP);
		break;
	case PF_PixelFormat_ARGB64:
		SortLines<PF_Pixel16>(input_worldP, output_worldP, *paramsP);
		break;
	case PF_PixelFormat_ARGB32:
		SortLines<PF_Pixel8>(input_worldP, output_worldP, *paramsP);
		break;
	default:
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	return PF_Err_NONE;
}
