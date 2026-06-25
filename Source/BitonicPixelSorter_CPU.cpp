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
#include <cstdint>
#include <thread>
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
inline T *PixelAtLayer(PF_EffectWorld *worldP, A_long layerX, A_long layerY) {
	return PixelAt<T>(worldP, layerX - worldP->origin_x, layerY - worldP->origin_y);
}

inline bool ContainsLayerPoint(const PF_EffectWorld *worldP, A_long layerX, A_long layerY) {
	return layerX >= worldP->origin_x &&
		   layerY >= worldP->origin_y &&
		   layerX < worldP->origin_x + worldP->width &&
		   layerY < worldP->origin_y + worldP->height;
}

inline A_long MaxLong(A_long a, A_long b) {
	return a > b ? a : b;
}

inline A_long MinLong(A_long a, A_long b) {
	return a < b ? a : b;
}

template <typename T>
struct Entry {
	float	key;
	uint32_t index;
};

template <typename T>
struct LineScratch {
	std::vector<T> sourceLine;
	std::vector<T> sortedLine;
	std::vector<float> keys;
	std::vector<unsigned char> validLine;
	std::vector<Entry<T>> run;
};

inline unsigned int WorkerCountFor(A_long lineCount) {
	if (lineCount < 32) {
		return 1;
	}

	unsigned int hw = std::thread::hardware_concurrency();
	if (hw == 0) {
		hw = 1;
	}

	// Keep per-frame fan-out small: AE's MFR provides frame-level parallelism.
	unsigned int workers = (hw >= 4) ? 2u : 1u;
	if (static_cast<A_long>(workers) > lineCount) {
		workers = static_cast<unsigned int>(lineCount);
	}
	return workers;
}

// Sort each contiguous in-threshold span of every line by brightness.
//
// Lines are rows when sorting horizontally and columns when sorting vertically;
// AE may hand Smart Render smaller worlds, so layer-coordinate origins must be
// honoured for both input dependency data and output writes.
template <typename T>
static void SortLines(PF_EffectWorld *inP, PF_EffectWorld *outP,
					  const BitonicSorterParams &prm,
					  A_long frameW,
					  A_long frameH) {
	const bool	horizontal = (prm.direction == BPS_DIR_HORIZONTAL);
	const bool	ascending  = (prm.ascending != 0);
	const float	tmin = prm.thresholdMin;
	const float	tmax = prm.thresholdMax;

	const A_long outW = outP->width;
	const A_long outH = outP->height;
	const A_long outLeft = outP->origin_x;
	const A_long outTop = outP->origin_y;
	const A_long outRight = outLeft + outW;
	const A_long outBottom = outTop + outH;

	const A_long lineStart = horizontal ? MaxLong(0, outTop)
										: MaxLong(0, outLeft);
	const A_long lineEnd = horizontal ? MinLong(frameH, outBottom)
									  : MinLong(frameW, outRight);
	const A_long lineLen = horizontal ? frameW : frameH;
	const A_long writeStart = horizontal ? MaxLong(0, outLeft)
										 : MaxLong(0, outTop);
	const A_long writeEnd = horizontal ? MinLong(frameW, outRight)
									   : MinLong(frameH, outBottom);

	const A_long lineCount = lineEnd - lineStart;
	if (lineCount <= 0 || lineLen <= 0 || writeEnd <= writeStart) {
		return;
	}

	auto processLines = [&](A_long chunkStart, A_long chunkEnd) {
		LineScratch<T> scratch;
		scratch.sourceLine.resize(static_cast<size_t>(lineLen));
		scratch.sortedLine.resize(static_cast<size_t>(lineLen));
		scratch.keys.resize(static_cast<size_t>(lineLen));
		scratch.validLine.resize(static_cast<size_t>(lineLen));
		scratch.run.reserve(static_cast<size_t>(lineLen));

		auto coord = [&](A_long k, A_long &x, A_long &y) {
			if (horizontal) { x = k; y = 0; }
			else            { x = 0; y = k; }
		};

		for (A_long line = chunkStart; line < chunkEnd; ++line) {
			// Snapshot the dependency line once, including its brightness key.
			for (A_long k = 0; k < lineLen; ++k) {
				A_long x, y;
				coord(k, x, y);
				if (horizontal) { y = line; }
				else            { x = line; }

				const size_t index = static_cast<size_t>(k);
				if (ContainsLayerPoint(inP, x, y)) {
					const T pixel = *PixelAtLayer<T>(inP, x, y);
					scratch.sourceLine[index] = pixel;
					scratch.sortedLine[index] = pixel;
					scratch.keys[index] = BrightnessUnit(pixel);
					scratch.validLine[index] = 1;
				} else {
					scratch.sourceLine[index] = T{};
					scratch.sortedLine[index] = T{};
					scratch.keys[index] = -1.0f;
					scratch.validLine[index] = 0;
				}
			}

			A_long k = 0;
			while (k < lineLen) {
				const size_t keyIndex = static_cast<size_t>(k);
				const bool inBounds = scratch.validLine[keyIndex] != 0;
				const float br = scratch.keys[keyIndex];

				if (inBounds && br >= tmin && br <= tmax) {
					const A_long start = k;
					scratch.run.clear();
					while (k < lineLen) {
						const size_t runIndex = static_cast<size_t>(k);
						if (scratch.validLine[runIndex] == 0) break;
						const float b = scratch.keys[runIndex];
						if (b < tmin || b > tmax) break;
						scratch.run.push_back(Entry<T>{b, static_cast<uint32_t>(k)});
						++k;
					}

					if (scratch.run.size() > 1) {
						if (ascending) {
							std::stable_sort(scratch.run.begin(), scratch.run.end(),
								[](const Entry<T> &a, const Entry<T> &b) { return a.key < b.key; });
						} else {
							std::stable_sort(scratch.run.begin(), scratch.run.end(),
								[](const Entry<T> &a, const Entry<T> &b) { return a.key > b.key; });
						}

						for (size_t i = 0; i < scratch.run.size(); ++i) {
							scratch.sortedLine[static_cast<size_t>(start) + i] =
								scratch.sourceLine[static_cast<size_t>(scratch.run[i].index)];
						}
					}
				} else {
					++k;
				}
			}

			for (A_long k = writeStart; k < writeEnd; ++k) {
				A_long x, y;
				coord(k, x, y);
				if (horizontal) { y = line; }
				else            { x = line; }
				if (ContainsLayerPoint(outP, x, y)) {
					*PixelAtLayer<T>(outP, x, y) = scratch.sortedLine[static_cast<size_t>(k)];
				}
			}
		}
	};

	const unsigned int workerCount = WorkerCountFor(lineCount);
	if (workerCount <= 1) {
		processLines(lineStart, lineEnd);
		return;
	}

	std::vector<std::thread> workers;
	workers.reserve(workerCount);
	for (unsigned int worker = 0; worker < workerCount; ++worker) {
		const A_long chunkBegin = lineStart +
			static_cast<A_long>((static_cast<long long>(lineCount) * worker) / workerCount);
		const A_long chunkEnd = lineStart +
			static_cast<A_long>((static_cast<long long>(lineCount) * (worker + 1)) / workerCount);
		if (chunkBegin < chunkEnd) {
			workers.emplace_back(processLines, chunkBegin, chunkEnd);
		}
	}

	for (std::thread &worker : workers) {
		worker.join();
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
		SortLines<PF_PixelFloat>(input_worldP, output_worldP, *paramsP,
								  in_data->width, in_data->height);
		break;
	case PF_PixelFormat_ARGB64:
		SortLines<PF_Pixel16>(input_worldP, output_worldP, *paramsP,
							   in_data->width, in_data->height);
		break;
	case PF_PixelFormat_ARGB32:
		SortLines<PF_Pixel8>(input_worldP, output_worldP, *paramsP,
							  in_data->width, in_data->height);
		break;
	default:
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	return PF_Err_NONE;
}
