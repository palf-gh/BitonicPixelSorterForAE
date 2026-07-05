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
#include "BitonicPixelSorter_GpuEligibility.h"
#include "BitonicPixelSorter_PathGeometry.h"

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
inline float SortKey(float r, float g, float b, float a, A_long criterion) {
	r = Saturate(r);
	g = Saturate(g);
	b = Saturate(b);
	a = Saturate(a);

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
	case BPS_CRITERION_RED_CHANNEL:
		return r;
	case BPS_CRITERION_GREEN_CHANNEL:
		return g;
	case BPS_CRITERION_BLUE_CHANNEL:
		return b;
	case BPS_CRITERION_ALPHA_CHANNEL:
		return a;
	case BPS_CRITERION_HUE: {
		const float maxRGB = (r > g) ? ((r > b) ? r : b) : ((g > b) ? g : b);
		const float minRGB = (r < g) ? ((r < b) ? r : b) : ((g < b) ? g : b);
		const float delta = maxRGB - minRGB;
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
		return Saturate(hue * (1.0f / 6.0f));
	}
	case BPS_CRITERION_SATURATION: {
		const float maxRGB = (r > g) ? ((r > b) ? r : b) : ((g > b) ? g : b);
		const float minRGB = (r < g) ? ((r < b) ? r : b) : ((g < b) ? g : b);
		if (maxRGB <= 0.0f) {
			return 0.0f;
		}
		return Saturate((maxRGB - minRGB) / maxRGB);
	}
	case BPS_CRITERION_LUMINANCE:
	default:
		return Saturate(0.298912f * r + 0.586611f * g + 0.114478f * b);
	}
}

inline float SortKeyUnit(const PF_Pixel8 &p, A_long criterion) {
	const float inv = 1.0f / 255.0f;
	return SortKey(p.red * inv, p.green * inv, p.blue * inv, p.alpha * inv, criterion);
}

inline float SortKeyUnit(const PF_Pixel16 &p, A_long criterion) {
	// After Effects 16-bit channels are 0..32768.
	const float inv = 1.0f / 32768.0f;
	return SortKey(p.red * inv, p.green * inv, p.blue * inv, p.alpha * inv, criterion);
}

inline float SortKeyUnit(const PF_PixelFloat &p, A_long criterion) {
	return SortKey(p.red, p.green, p.blue, p.alpha, criterion);
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

// Sample a sort/trigger key from an optional source world at layer coordinates.
// Out-of-bounds samples return -1 so threshold runs treat them as unaffected.
template <typename T>
inline float SampleKeyAt(PF_EffectWorld *worldP, A_long x, A_long y, A_long keyCriterion) {
	if (!worldP || !ContainsLayerPoint(worldP, x, y)) {
		return -1.0f;
	}
	return SortKeyUnit(*PixelAtLayer<T>(worldP, x, y), keyCriterion);
}

inline bool UsesUnifiedKeys(PF_EffectWorld *criterionP,
							PF_EffectWorld *triggerP,
							A_long criterion,
							A_long trigger)
{
	return criterionP == triggerP && criterion == trigger;
}

template <typename T>
inline void FillSortKeys(PF_EffectWorld *criterionP,
						 PF_EffectWorld *triggerP,
						 A_long x,
						 A_long y,
						 A_long criterion,
						 A_long trigger,
						 float *criterionKeyOut,
						 float *triggerKeyOut)
{
	if (UsesUnifiedKeys(criterionP, triggerP, criterion, trigger)) {
		const float key = SampleKeyAt<T>(criterionP, x, y, criterion);
		*criterionKeyOut = key;
		*triggerKeyOut = key;
		return;
	}
	*criterionKeyOut = SampleKeyAt<T>(criterionP, x, y, criterion);
	*triggerKeyOut = SampleKeyAt<T>(triggerP, x, y, trigger);
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
	std::vector<float> triggerKeys;
	std::vector<unsigned char> validLine;
	std::vector<Entry<T>> run;
};

struct PathCoord {
	A_long x;
	A_long y;
	bool valid;
};

struct MappedPixelCoord {
	A_long x;
	A_long y;
	A_long pos;
};

template <typename T>
struct PathScratch {
	std::vector<T> sourceLine;
	std::vector<T> sortedLine;
	std::vector<float> keys;
	std::vector<float> triggerKeys;
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
	return static_cast<A_long>(std::floor(value + 0.5));
}

inline bool IsAffectedByThreshold(float triggerKey, const BitonicSorterParams &prm) {
	const bool inside = triggerKey >= prm.thresholdMin && triggerKey <= prm.thresholdMax;
	return (prm.affect == BPS_AFFECT_OUTSIDE_THRESHOLDS) ? !inside : inside;
}

inline size_t CycleShiftFor(size_t count, const BitonicSorterParams &prm) {
	if (count <= 1) {
		return 0;
	}

	const double turns = static_cast<double>(prm.cycleDegrees) / 360.0;
	long long shift = static_cast<long long>(std::llround(turns * static_cast<double>(count)));
	const long long n = static_cast<long long>(count);
	shift %= n;
	if (shift < 0) {
		shift += n;
	}
	return static_cast<size_t>(shift);
}

template <typename T, typename Scratch>
inline void SortRunIntoLine(Scratch &scratch,
							A_long start,
							const BitonicSorterParams &prm) {
	if (scratch.run.size() <= 1) {
		return;
	}

	if (prm.ascending) {
		std::stable_sort(scratch.run.begin(), scratch.run.end(),
			[](const Entry<T> &a, const Entry<T> &b) { return a.key < b.key; });
	} else {
		std::stable_sort(scratch.run.begin(), scratch.run.end(),
			[](const Entry<T> &a, const Entry<T> &b) { return a.key > b.key; });
	}

	const size_t shift = CycleShiftFor(scratch.run.size(), prm);
	for (size_t i = 0; i < scratch.run.size(); ++i) {
		const size_t dest = (i + shift) % scratch.run.size();
		scratch.sortedLine[static_cast<size_t>(start) + dest] =
			scratch.sourceLine[static_cast<size_t>(scratch.run[i].index)];
	}
}

// Sort a run whose members may be non-contiguous along the path (OOB positions
// are omitted). `positions` lists path indices in traversal order.
template <typename T, typename Scratch>
inline void SortRunAtPositions(Scratch &scratch,
							   const std::vector<A_long> &positions,
							   const BitonicSorterParams &prm) {
	if (positions.size() <= 1u) {
		return;
	}

	scratch.run.clear();
	scratch.run.reserve(positions.size());
	for (A_long pos : positions) {
		const size_t index = static_cast<size_t>(pos);
		scratch.run.push_back(Entry<T>{scratch.keys[index], static_cast<uint32_t>(pos)});
	}

	if (prm.ascending) {
		std::stable_sort(scratch.run.begin(), scratch.run.end(),
			[](const Entry<T> &a, const Entry<T> &b) { return a.key < b.key; });
	} else {
		std::stable_sort(scratch.run.begin(), scratch.run.end(),
			[](const Entry<T> &a, const Entry<T> &b) { return a.key > b.key; });
	}

	const size_t shift = CycleShiftFor(scratch.run.size(), prm);
	for (size_t i = 0; i < scratch.run.size(); ++i) {
		const size_t dest = (i + shift) % scratch.run.size();
		const size_t dest_pos = static_cast<size_t>(positions[dest]);
		scratch.sortedLine[dest_pos] =
			scratch.sourceLine[static_cast<size_t>(scratch.run[i].index)];
	}
}

// Build in-bounds path order from pos = 0 (angle start). Out-of-frame samples
// are omitted but do not break runs, so the visible arc sorts as one continuous
// sequence while the seam still follows the angle parameter.
inline void BuildPathOrder(const std::vector<unsigned char> &validLine,
						   A_long lineLen,
						   std::vector<A_long> *pathP) {
	pathP->clear();
	if (lineLen <= 0) {
		return;
	}

	for (A_long pos = 0; pos < lineLen; ++pos) {
		if (validLine[static_cast<size_t>(pos)] != 0u) {
			pathP->push_back(pos);
		}
	}
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
	case BPS_MODE_SWIRL:
	case BPS_MODE_PATH:
		return prm.domainLineCount;
	default:
		return 0;
	}
}

inline A_long GenericLineLength(const BitonicSorterParams &prm, A_long line) {
	(void)line;
	switch (prm.mode) {
	case BPS_MODE_FREE_ANGLE:
		return prm.freeLineLength;
	case BPS_MODE_ROTATION:
		return RotationLineLength(line);
	case BPS_MODE_RADIAL:
		return prm.radialLength;
	case BPS_MODE_SWIRL:
		return prm.radialLength;
	case BPS_MODE_PATH:
		return prm.domainMaxLineLength;
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
		// Path basis is (sin, -cos) so theta = 0 is top (North) at angle = 0.
		// angleCos/angleSin then rotate that start around the centre.
		const double theta = (lineLen <= 1)
			? 0.0
			: (2.0 * BPS_CPU_PI * static_cast<double>(pos)) /
			  static_cast<double>(lineLen);
		const double c = std::cos(theta);
		const double s = std::sin(theta);
		const double radius = static_cast<double>(line);
		coord.x = RoundToLong(prm.centerX +
			radius * (s * prm.angleCos + c * prm.angleSin));
		coord.y = RoundToLong(prm.centerY +
			radius * (s * prm.angleSin - c * prm.angleCos));
	} else if (prm.mode == BPS_MODE_RADIAL) {
		const double theta = (prm.radialLineCount <= 0)
			? 0.0
			: (2.0 * BPS_CPU_PI * static_cast<double>(line)) /
			  static_cast<double>(prm.radialLineCount);
		coord.x = RoundToLong(prm.centerX + static_cast<double>(pos) * std::cos(theta));
		coord.y = RoundToLong(prm.centerY + static_cast<double>(pos) * std::sin(theta));
	} else if (prm.mode == BPS_MODE_SWIRL) {
		// Phase lines: phase = atan2 - angle - k*r (same as pixel classification).
		// line indexes phase, pos is radius — same cost class as Radial.
		const double r = static_cast<double>(pos);
		const double phase = (prm.radialLineCount <= 1)
			? 0.0
			: (2.0 * BPS_CPU_PI * static_cast<double>(line)) /
			  static_cast<double>(prm.radialLineCount);
		const double theta =
			phase + static_cast<double>(prm.swirlK) * r + prm.angleRadians;
		coord.x = RoundToLong(prm.centerX + r * std::cos(theta));
		coord.y = RoundToLong(prm.centerY + r * std::sin(theta));
	} else if (prm.mode == BPS_MODE_PATH) {
		A_long px = 0;
		A_long py = 0;
		if (!BPS_PathCoordForPos(prm, line, pos, &px, &py)) {
			return coord;
		}
		coord.x = px;
		coord.y = py;
	}

	coord.valid = coord.x >= 0 && coord.y >= 0 && coord.x < frameW && coord.y < frameH;
	return coord;
}

inline bool GenericDomainPosForPixel(const BitonicSorterParams &prm,
									 A_long x,
									 A_long y,
									 A_long *lineP,
									 A_long *posP) {
	if (!lineP || !posP) {
		return false;
	}

	A_long line = 0;
	A_long pos = 0;
	if (prm.mode == BPS_MODE_FREE_ANGLE) {
		const double p = static_cast<double>(x) * prm.angleCos +
						 static_cast<double>(y) * prm.angleSin;
		const double q = -static_cast<double>(x) * prm.angleSin +
						  static_cast<double>(y) * prm.angleCos;
		pos = RoundToLong(p) - prm.freePMin;
		line = RoundToLong(q) - prm.freeQMin;
	} else if (prm.mode == BPS_MODE_ROTATION) {
		const double dx = static_cast<double>(x) - prm.centerX;
		const double dy = static_cast<double>(y) - prm.centerY;
		line = RoundToLong(std::sqrt(dx * dx + dy * dy));
		const A_long lineLen = GenericLineLength(prm, line);
		if (lineLen <= 1) {
			pos = 0;
		} else {
			// Undo the start-angle rotation; basis is (sin, -cos) so
			// theta = atan2(rx, -ry) with theta = 0 at top.
			const double rx = dx * prm.angleCos + dy * prm.angleSin;
			const double ry = -dx * prm.angleSin + dy * prm.angleCos;
			double theta = std::atan2(rx, -ry);
			if (theta < 0.0) {
				theta += 2.0 * BPS_CPU_PI;
			}
			pos = RoundToLong((theta / (2.0 * BPS_CPU_PI)) *
							  static_cast<double>(lineLen));
			if (pos >= lineLen) {
				pos -= lineLen;
			}
		}
	} else if (prm.mode == BPS_MODE_RADIAL) {
		const double dx = static_cast<double>(x) - prm.centerX;
		const double dy = static_cast<double>(y) - prm.centerY;
		const double radius = std::sqrt(dx * dx + dy * dy);
		pos = RoundToLong(radius);
		double theta = std::atan2(dy, dx);
		if (theta < 0.0) {
			theta += 2.0 * BPS_CPU_PI;
		}
		line = (prm.radialLineCount <= 1)
			? 0
			: RoundToLong((theta / (2.0 * BPS_CPU_PI)) *
						  static_cast<double>(prm.radialLineCount));
		if (line >= prm.radialLineCount) {
			line -= prm.radialLineCount;
		}
	} else if (prm.mode == BPS_MODE_SWIRL) {
		const double dx = static_cast<double>(x) - prm.centerX;
		const double dy = static_cast<double>(y) - prm.centerY;
		const double r = std::sqrt(dx * dx + dy * dy);
		double theta = std::atan2(dy, dx) - prm.angleRadians;
		theta -= static_cast<double>(prm.swirlK) * r;
		theta = std::fmod(theta, 2.0 * BPS_CPU_PI);
		if (theta < 0.0) {
			theta += 2.0 * BPS_CPU_PI;
		}
		line = (prm.radialLineCount <= 1)
			? 0
			: RoundToLong((theta / (2.0 * BPS_CPU_PI)) *
						  static_cast<double>(prm.radialLineCount));
		if (line >= prm.radialLineCount) {
			line -= prm.radialLineCount;
		}
		pos = RoundToLong(r);
	} else if (prm.mode == BPS_MODE_PATH) {
		return BPS_PathDomainPosForPixel(prm, x, y, lineP, posP);
	} else {
		return false;
	}

	if (line < 0 || line >= GenericLineCount(prm)) {
		return false;
	}
	const A_long lineLen = GenericLineLength(prm, line);
	if (pos < 0 || pos >= lineLen) {
		return false;
	}

	*lineP = line;
	*posP = pos;
	return true;
}

// Sort each contiguous in-threshold span of every line by the selected key.
//
// Lines are rows when sorting horizontally and columns when sorting vertically;
// AE may hand Smart Render smaller worlds, so layer-coordinate origins must be
// honoured for both input dependency data and output writes.
template <typename T>
static void SortLines(PF_EffectWorld *inP, PF_EffectWorld *outP,
					  PF_EffectWorld *criterionP, PF_EffectWorld *triggerP,
					  const BitonicSorterParams &prm,
					  A_long frameW,
					  A_long frameH) {
	const bool	horizontal = (prm.direction == BPS_DIR_HORIZONTAL);
	const A_long criterion = prm.criterion;
	const A_long trigger = prm.trigger;

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
		scratch.triggerKeys.resize(static_cast<size_t>(lineLen));
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
					FillSortKeys<T>(criterionP, triggerP, x, y, criterion, trigger,
									&scratch.keys[index], &scratch.triggerKeys[index]);
					scratch.validLine[index] = 1;
				} else {
					scratch.sourceLine[index] = T{};
					scratch.sortedLine[index] = T{};
					scratch.keys[index] = -1.0f;
					scratch.triggerKeys[index] = -1.0f;
					scratch.validLine[index] = 0;
				}
			}

			A_long k = 0;
			while (k < lineLen) {
				const size_t keyIndex = static_cast<size_t>(k);
				const bool inBounds = scratch.validLine[keyIndex] != 0;
				const float triggerKey = scratch.triggerKeys[keyIndex];

				if (inBounds && IsAffectedByThreshold(triggerKey, prm)) {
					const A_long start = k;
					scratch.run.clear();
					while (k < lineLen) {
						const size_t runIndex = static_cast<size_t>(k);
						if (scratch.validLine[runIndex] == 0) break;
						if (!IsAffectedByThreshold(scratch.triggerKeys[runIndex], prm)) break;
						scratch.run.push_back(Entry<T>{
							scratch.keys[runIndex], static_cast<uint32_t>(k)});
						++k;
					}

					SortRunIntoLine<T>(scratch, start, prm);
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
static void SortMappedPixels(PF_EffectWorld *inP, PF_EffectWorld *outP,
							 PF_EffectWorld *criterionP, PF_EffectWorld *triggerP,
							 const BitonicSorterParams &prm,
							 A_long frameW,
							 A_long frameH) {
	CopyInputToOutputRect<T>(inP, outP);

	const A_long lineCount = prm.domainLineCount;
	if (lineCount <= 0 || frameW <= 0 || frameH <= 0 ||
		!prm.mappedRecords || !prm.mappedLineOffsets ||
		prm.mappedRecordCount <= 0) {
		return;
	}

	const A_long criterion = prm.criterion;
	const A_long trigger = prm.trigger;

	auto processLines = [&](A_long chunkStart, A_long chunkEnd) {
		LineScratch<T> scratch;

		for (A_long line = chunkStart; line < chunkEnd; ++line) {
			const std::uint32_t begin = prm.mappedLineOffsets[line];
			const std::uint32_t end = prm.mappedLineOffsets[line + 1];
			if (end <= begin) {
				continue;
			}

			const size_t lineLen = static_cast<size_t>(end - begin);
			scratch.sourceLine.resize(lineLen);
			scratch.sortedLine.resize(lineLen);
			scratch.keys.resize(lineLen);
			scratch.triggerKeys.resize(lineLen);
			scratch.validLine.resize(lineLen);
			scratch.run.clear();
			scratch.run.reserve(lineLen);

			for (size_t i = 0; i < lineLen; ++i) {
				const BpsMappedPixelRecord &record =
					prm.mappedRecords[static_cast<size_t>(begin) + i];
				const A_long x = static_cast<A_long>(record.pixelIndex % frameW);
				const A_long y = static_cast<A_long>(record.pixelIndex / frameW);
				if (ContainsLayerPoint(inP, x, y)) {
					const T pixel = *PixelAtLayer<T>(inP, x, y);
					scratch.sourceLine[i] = pixel;
					scratch.sortedLine[i] = pixel;
					FillSortKeys<T>(criterionP, triggerP, x, y, criterion, trigger,
									&scratch.keys[i], &scratch.triggerKeys[i]);
					scratch.validLine[i] = 1u;
				} else {
					scratch.sourceLine[i] = T{};
					scratch.sortedLine[i] = T{};
					scratch.keys[i] = -1.0f;
					scratch.triggerKeys[i] = -1.0f;
					scratch.validLine[i] = 0u;
				}
			}

			A_long k = 0;
			const A_long lineLenLong = static_cast<A_long>(lineLen);
			while (k < lineLenLong) {
				const size_t keyIndex = static_cast<size_t>(k);
				if (scratch.validLine[keyIndex] != 0u &&
					IsAffectedByThreshold(scratch.triggerKeys[keyIndex], prm)) {
					const A_long start = k;
					scratch.run.clear();
					while (k < lineLenLong) {
						const size_t runIndex = static_cast<size_t>(k);
						if (scratch.validLine[runIndex] == 0u ||
							!IsAffectedByThreshold(scratch.triggerKeys[runIndex], prm)) {
							break;
						}
						scratch.run.push_back(Entry<T>{
							scratch.keys[runIndex], static_cast<uint32_t>(k)});
						++k;
					}
					SortRunIntoLine<T>(scratch, start, prm);
				} else {
					++k;
				}
			}

			for (size_t i = 0; i < lineLen; ++i) {
				const BpsMappedPixelRecord &record =
					prm.mappedRecords[static_cast<size_t>(begin) + i];
				const A_long x = static_cast<A_long>(record.pixelIndex % frameW);
				const A_long y = static_cast<A_long>(record.pixelIndex / frameW);
				if (scratch.validLine[i] != 0u && ContainsLayerPoint(outP, x, y)) {
					*PixelAtLayer<T>(outP, x, y) = scratch.sortedLine[i];
				}
			}
		}
	};

	const unsigned int workerCount = WorkerCountFor(lineCount);
	if (workerCount <= 1) {
		processLines(0, lineCount);
		return;
	}

	std::vector<std::thread> workers;
	workers.reserve(workerCount);
	for (unsigned int worker = 0; worker < workerCount; ++worker) {
		const A_long chunkBegin =
			static_cast<A_long>((static_cast<long long>(lineCount) * worker) / workerCount);
		const A_long chunkEnd =
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
static void SortGenericPaths(PF_EffectWorld *inP, PF_EffectWorld *outP,
							 PF_EffectWorld *criterionP, PF_EffectWorld *triggerP,
							 const BitonicSorterParams &prm,
							 A_long frameW,
							 A_long frameH) {
	CopyInputToOutputRect<T>(inP, outP);

	const A_long lineCount = GenericLineCount(prm);
	if (lineCount <= 0) {
		return;
	}

	const A_long criterion = prm.criterion;
	const A_long trigger = prm.trigger;
	std::vector<size_t> offsets(static_cast<size_t>(lineCount) + 1u, 0u);
	for (A_long line = 0; line < lineCount; ++line) {
		const A_long lineLen = GenericLineLength(prm, line);
		offsets[static_cast<size_t>(line) + 1u] =
			offsets[static_cast<size_t>(line)] +
			static_cast<size_t>(lineLen > 0 ? lineLen : 0);
	}

	const size_t domainSize = offsets.back();
	if (domainSize == 0u) {
		return;
	}

	std::vector<T> domain(domainSize);
	std::vector<unsigned char> domainValid(domainSize, 0u);
	// Only slots inside an in-threshold run (length >= 2) are composited over the
	// original. Everything else keeps its exact source pixel, so the domain
	// resample round-trip never degrades unsorted areas.
	std::vector<unsigned char> domainAffected(domainSize, 0u);
	PathScratch<T> scratch;
	scratch.run.reserve(static_cast<size_t>(prm.maxLineLength > 0 ? prm.maxLineLength : 1));

	// Forward path sampling. Out-of-frame samples are omitted from the path
	// (not treated as run breaks), so a circle that clips the frame still sorts
	// as one continuous arc — matching sample-plugin behaviour.
	std::vector<A_long> path;
	std::vector<A_long> run_positions;
	std::vector<unsigned char> affectedLine;
	path.reserve(static_cast<size_t>(prm.maxLineLength > 0 ? prm.maxLineLength : 1));
	run_positions.reserve(path.capacity());
	affectedLine.reserve(path.capacity());

	// Mark a run as affected only when it can actually reorder pixels (>= 2), then
	// sort it. Single in-threshold pixels are left as the exact original.
	auto flushRun = [&]() {
		if (run_positions.size() >= 2u) {
			for (A_long runPos : run_positions) {
				affectedLine[static_cast<size_t>(runPos)] = 1u;
			}
		}
		SortRunAtPositions<T>(scratch, run_positions, prm);
		run_positions.clear();
	};

	for (A_long line = 0; line < lineCount; ++line) {
		const A_long lineLen = GenericLineLength(prm, line);
		if (lineLen <= 0) {
			continue;
		}

		const size_t lineSize = static_cast<size_t>(lineLen);
		scratch.sourceLine.resize(lineSize);
		scratch.sortedLine.resize(lineSize);
		scratch.keys.resize(lineSize);
		scratch.triggerKeys.resize(lineSize);
		scratch.validLine.resize(lineSize);

		for (A_long k = 0; k < lineLen; ++k) {
			const size_t index = static_cast<size_t>(k);
			const PathCoord coord = GenericCoord(prm, line, k, frameW, frameH);
			if (coord.valid && ContainsLayerPoint(inP, coord.x, coord.y)) {
				const T pixel = *PixelAtLayer<T>(inP, coord.x, coord.y);
				scratch.sourceLine[index] = pixel;
				scratch.sortedLine[index] = pixel;
				FillSortKeys<T>(criterionP, triggerP, coord.x, coord.y, criterion, trigger,
								&scratch.keys[index], &scratch.triggerKeys[index]);
				scratch.validLine[index] = 1u;
			} else {
				scratch.sourceLine[index] = T{};
				scratch.sortedLine[index] = T{};
				scratch.keys[index] = -1.0f;
				scratch.triggerKeys[index] = -1.0f;
				scratch.validLine[index] = 0u;
			}
		}

		BuildPathOrder(scratch.validLine, lineLen, &path);
		affectedLine.assign(lineSize, 0u);

		run_positions.clear();
		for (A_long pos : path) {
			const size_t index = static_cast<size_t>(pos);
			if (IsAffectedByThreshold(scratch.triggerKeys[index], prm)) {
				run_positions.push_back(pos);
			} else {
				flushRun();
			}
		}
		flushRun();

		const size_t base = offsets[static_cast<size_t>(line)];
		for (A_long kWrite = 0; kWrite < lineLen; ++kWrite) {
			const size_t index = static_cast<size_t>(kWrite);
			const size_t dstIndex = base + index;
			domain[dstIndex] = scratch.sortedLine[index];
			domainValid[dstIndex] = scratch.validLine[index];
			domainAffected[dstIndex] = affectedLine[index];
		}
	}

	const A_long outLeft = outP->origin_x;
	const A_long outTop = outP->origin_y;
	const A_long outRight = outLeft + outP->width;
	const A_long outBottom = outTop + outP->height;
	for (A_long y = outTop; y < outBottom; ++y) {
		for (A_long x = outLeft; x < outRight; ++x) {
			if (!ContainsLayerPoint(outP, x, y)) {
				continue;
			}

			A_long line = 0;
			A_long pos = 0;
			if (!GenericDomainPosForPixel(prm, x, y, &line, &pos)) {
				continue;
			}

			const size_t domainIndex =
				offsets[static_cast<size_t>(line)] + static_cast<size_t>(pos);
			// Composite over the original: only overwrite pixels a sort actually
			// reordered; untouched areas keep the exact source (no resample loss).
			if (domainIndex < domain.size() && domainAffected[domainIndex] != 0u) {
				*PixelAtLayer<T>(outP, x, y) = domain[domainIndex];
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
	PF_EffectWorld		*criterion_worldP,
	PF_EffectWorld		*trigger_worldP,
	const BitonicSorterParams *paramsP)
{
	if (!input_worldP || !output_worldP || !paramsP) {
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	if (!criterion_worldP) {
		criterion_worldP = input_worldP;
	}
	if (!trigger_worldP) {
		trigger_worldP = input_worldP;
	}

	switch (pixel_format) {
	case PF_PixelFormat_ARGB128:
		if (paramsP->mode == BPS_MODE_AXIS) {
			SortLines<PF_PixelFloat>(input_worldP, output_worldP,
									  criterion_worldP, trigger_worldP, *paramsP,
									  BPS_RenderWidth(in_data), BPS_RenderHeight(in_data));
		} else if (paramsP->mappedRecords && paramsP->mappedRecordCount > 0) {
			SortMappedPixels<PF_PixelFloat>(input_worldP, output_worldP,
											criterion_worldP, trigger_worldP, *paramsP,
											BPS_RenderWidth(in_data), BPS_RenderHeight(in_data));
		} else {
			SortGenericPaths<PF_PixelFloat>(input_worldP, output_worldP,
											criterion_worldP, trigger_worldP, *paramsP,
											BPS_RenderWidth(in_data), BPS_RenderHeight(in_data));
		}
		break;
	case PF_PixelFormat_ARGB64:
		if (paramsP->mode == BPS_MODE_AXIS) {
			SortLines<PF_Pixel16>(input_worldP, output_worldP,
								   criterion_worldP, trigger_worldP, *paramsP,
								   BPS_RenderWidth(in_data), BPS_RenderHeight(in_data));
		} else if (paramsP->mappedRecords && paramsP->mappedRecordCount > 0) {
			SortMappedPixels<PF_Pixel16>(input_worldP, output_worldP,
										 criterion_worldP, trigger_worldP, *paramsP,
										 BPS_RenderWidth(in_data), BPS_RenderHeight(in_data));
		} else {
			SortGenericPaths<PF_Pixel16>(input_worldP, output_worldP,
										 criterion_worldP, trigger_worldP, *paramsP,
										 BPS_RenderWidth(in_data), BPS_RenderHeight(in_data));
		}
		break;
	case PF_PixelFormat_ARGB32:
		if (paramsP->mode == BPS_MODE_AXIS) {
			SortLines<PF_Pixel8>(input_worldP, output_worldP,
								  criterion_worldP, trigger_worldP, *paramsP,
								  BPS_RenderWidth(in_data), BPS_RenderHeight(in_data));
		} else if (paramsP->mappedRecords && paramsP->mappedRecordCount > 0) {
			SortMappedPixels<PF_Pixel8>(input_worldP, output_worldP,
										criterion_worldP, trigger_worldP, *paramsP,
										BPS_RenderWidth(in_data), BPS_RenderHeight(in_data));
		} else {
			SortGenericPaths<PF_Pixel8>(input_worldP, output_worldP,
										criterion_worldP, trigger_worldP, *paramsP,
										BPS_RenderWidth(in_data), BPS_RenderHeight(in_data));
		}
		break;
	default:
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	return PF_Err_NONE;
}
