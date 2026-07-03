/*
	BitonicPixelSorter_CPU.cpp

	CPU implementation of the pixel sort. This is the correctness oracle and the
	GPU fallback. It reproduces the upstream effect's semantics without the GPU's
	group-shared-memory line-length limit, and across 8/16/32-bit.

	The GPU uses a bitonic sorting network purely to parallelise the sort; the
	visible result is "sort each contiguous in-threshold span of a line by
	the selected key", which on the CPU is a plain std::stable_sort per span.
*/

#include "BitonicPixelSorter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

constexpr double BPS_CPU_PI = 3.14159265358979323846;

inline float Saturate(float v) {
	return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// Luma weights matching the upstream shader: 0.298912 R + 0.586611 G + 0.114478 B.
inline float SortKey(float r, float g, float b, A_long criterion) {
	switch (criterion) {
	case BPS_CRITERION_RGB_AVERAGE:
		return Saturate((r + g + b) * (1.0f / 3.0f));
	case BPS_CRITERION_RGB_PRODUCT:
		return Saturate(r * g * b);
	case BPS_CRITERION_RGB_MINIMUM: {
		const float rg = (r < g) ? r : g;
		return Saturate((rg < b) ? rg : b);
	}
	case BPS_CRITERION_RGB_MAXIMUM: {
		const float rg = (r > g) ? r : g;
		return Saturate((rg > b) ? rg : b);
	}
	case BPS_CRITERION_LUMINANCE:
	default:
		return Saturate(0.298912f * r + 0.586611f * g + 0.114478f * b);
	}
}

inline float SortKeyUnit(const PF_Pixel8 &p, A_long criterion) {
	const float inv = 1.0f / 255.0f;
	return SortKey(p.red * inv, p.green * inv, p.blue * inv, criterion);
}

inline float SortKeyUnit(const PF_Pixel16 &p, A_long criterion) {
	// After Effects 16-bit channels are 0..32768.
	const float inv = 1.0f / 32768.0f;
	return SortKey(p.red * inv, p.green * inv, p.blue * inv, criterion);
}

inline float SortKeyUnit(const PF_PixelFloat &p, A_long criterion) {
	return SortKey(p.red, p.green, p.blue, criterion);
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

struct PathCoord {
	A_long x;
	A_long y;
	bool valid;
};

template <typename T>
struct PathScratch {
	std::vector<T> sourceLine;
	std::vector<T> sortedLine;
	std::vector<float> keys;
	std::vector<unsigned char> validLine;
	std::vector<PathCoord> coords;
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

inline A_long RoundToLong(double value) {
	return static_cast<A_long>(std::lround(value));
}

inline A_long RotationLineLength(A_long radius) {
	if (radius <= 0) {
		return 1;
	}
	A_long length = static_cast<A_long>(std::ceil(2.0 * BPS_CPU_PI * radius));
	return length < 1 ? 1 : length;
}

inline A_long GenericLineCount(const BitonicSorterParams &prm) {
	switch (prm.mode) {
	case BPS_MODE_FREE_ANGLE:
		return prm.freeLineCount;
	case BPS_MODE_ROTATION:
		return prm.radialLength;
	case BPS_MODE_RADIAL:
		return prm.radialLineCount;
	default:
		return 0;
	}
}

inline A_long GenericLineLength(const BitonicSorterParams &prm, A_long line) {
	switch (prm.mode) {
	case BPS_MODE_FREE_ANGLE:
		return prm.freeLineLength;
	case BPS_MODE_ROTATION:
		return RotationLineLength(line);
	case BPS_MODE_RADIAL:
		return prm.radialLength;
	default:
		return 0;
	}
}

inline PathCoord GenericCoord(const BitonicSorterParams &prm,
							  A_long line,
							  A_long pos,
							  A_long frameW,
							  A_long frameH) {
	PathCoord coord = {0, 0, false};
	if (frameW <= 0 || frameH <= 0) {
		return coord;
	}

	if (prm.mode == BPS_MODE_FREE_ANGLE) {
		const double p = static_cast<double>(prm.freePMin + pos);
		const double q = static_cast<double>(prm.freeQMin + line);
		coord.x = RoundToLong(p * prm.angleCos - q * prm.angleSin);
		coord.y = RoundToLong(p * prm.angleSin + q * prm.angleCos);
	} else if (prm.mode == BPS_MODE_ROTATION) {
		const A_long lineLen = GenericLineLength(prm, line);
		const double theta = (lineLen <= 1)
			? 0.0
			: (2.0 * BPS_CPU_PI * static_cast<double>(pos)) /
			  static_cast<double>(lineLen);
		coord.x = RoundToLong(prm.centerX + static_cast<double>(line) * std::cos(theta));
		coord.y = RoundToLong(prm.centerY + static_cast<double>(line) * std::sin(theta));
	} else if (prm.mode == BPS_MODE_RADIAL) {
		const double theta = (prm.radialLineCount <= 0)
			? 0.0
			: (2.0 * BPS_CPU_PI * static_cast<double>(line)) /
			  static_cast<double>(prm.radialLineCount);
		coord.x = RoundToLong(prm.centerX + static_cast<double>(pos) * std::cos(theta));
		coord.y = RoundToLong(prm.centerY + static_cast<double>(pos) * std::sin(theta));
	}

	coord.valid = coord.x >= 0 && coord.y >= 0 && coord.x < frameW && coord.y < frameH;
	return coord;
}

// Sort each contiguous in-threshold span of every line by the selected key.
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
	const A_long criterion = prm.criterion;
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
			// Snapshot the dependency line once, including its selected key.
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
					scratch.keys[index] = SortKeyUnit(pixel, criterion);
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

template <typename T>
static void CopyInputToOutputRect(PF_EffectWorld *inP, PF_EffectWorld *outP) {
	const A_long outLeft = outP->origin_x;
	const A_long outTop = outP->origin_y;
	const A_long outRight = outLeft + outP->width;
	const A_long outBottom = outTop + outP->height;

	for (A_long y = outTop; y < outBottom; ++y) {
		for (A_long x = outLeft; x < outRight; ++x) {
			if (ContainsLayerPoint(outP, x, y)) {
				*PixelAtLayer<T>(outP, x, y) = ContainsLayerPoint(inP, x, y)
					? *PixelAtLayer<T>(inP, x, y)
					: T{};
			}
		}
	}
}

template <typename T>
static void SortGenericPaths(PF_EffectWorld *inP, PF_EffectWorld *outP,
							 const BitonicSorterParams &prm,
							 A_long frameW,
							 A_long frameH) {
	CopyInputToOutputRect<T>(inP, outP);

	const A_long lineCount = GenericLineCount(prm);
	if (lineCount <= 0) {
		return;
	}

	const bool ascending = (prm.ascending != 0);
	const A_long criterion = prm.criterion;
	const float tmin = prm.thresholdMin;
	const float tmax = prm.thresholdMax;

	auto processLines = [&](A_long chunkStart, A_long chunkEnd) {
		PathScratch<T> scratch;
		scratch.run.reserve(static_cast<size_t>(prm.maxLineLength > 0 ? prm.maxLineLength : 1));

		for (A_long line = chunkStart; line < chunkEnd; ++line) {
			const A_long lineLen = GenericLineLength(prm, line);
			if (lineLen <= 0) {
				continue;
			}
			const size_t lineSize = static_cast<size_t>(lineLen);
			scratch.sourceLine.resize(lineSize);
			scratch.sortedLine.resize(lineSize);
			scratch.keys.resize(lineSize);
			scratch.validLine.resize(lineSize);
			scratch.coords.resize(lineSize);

			for (A_long k = 0; k < lineLen; ++k) {
				const size_t index = static_cast<size_t>(k);
				const PathCoord coord = GenericCoord(prm, line, k, frameW, frameH);
				scratch.coords[index] = coord;
				if (coord.valid && ContainsLayerPoint(inP, coord.x, coord.y)) {
					const T pixel = *PixelAtLayer<T>(inP, coord.x, coord.y);
					scratch.sourceLine[index] = pixel;
					scratch.sortedLine[index] = pixel;
					scratch.keys[index] = SortKeyUnit(pixel, criterion);
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
				const float key = scratch.keys[keyIndex];

				if (inBounds && key >= tmin && key <= tmax) {
					const A_long start = k;
					scratch.run.clear();
					while (k < lineLen) {
						const size_t runIndex = static_cast<size_t>(k);
						if (scratch.validLine[runIndex] == 0) break;
						const float runKey = scratch.keys[runIndex];
						if (runKey < tmin || runKey > tmax) break;
						scratch.run.push_back(Entry<T>{runKey, static_cast<uint32_t>(k)});
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

			for (A_long writeIndex = 0; writeIndex < lineLen; ++writeIndex) {
				const PathCoord coord = scratch.coords[static_cast<size_t>(writeIndex)];
				if (coord.valid && ContainsLayerPoint(outP, coord.x, coord.y)) {
					*PixelAtLayer<T>(outP, coord.x, coord.y) =
						scratch.sortedLine[static_cast<size_t>(writeIndex)];
				}
			}
		}
	};

	processLines(0, lineCount);
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
		if (paramsP->mode == BPS_MODE_AXIS) {
			SortLines<PF_PixelFloat>(input_worldP, output_worldP, *paramsP,
									  in_data->width, in_data->height);
		} else {
			SortGenericPaths<PF_PixelFloat>(input_worldP, output_worldP, *paramsP,
											in_data->width, in_data->height);
		}
		break;
	case PF_PixelFormat_ARGB64:
		if (paramsP->mode == BPS_MODE_AXIS) {
			SortLines<PF_Pixel16>(input_worldP, output_worldP, *paramsP,
								   in_data->width, in_data->height);
		} else {
			SortGenericPaths<PF_Pixel16>(input_worldP, output_worldP, *paramsP,
										 in_data->width, in_data->height);
		}
		break;
	case PF_PixelFormat_ARGB32:
		if (paramsP->mode == BPS_MODE_AXIS) {
			SortLines<PF_Pixel8>(input_worldP, output_worldP, *paramsP,
								  in_data->width, in_data->height);
		} else {
			SortGenericPaths<PF_Pixel8>(input_worldP, output_worldP, *paramsP,
										in_data->width, in_data->height);
		}
		break;
	default:
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	return PF_Err_NONE;
}
