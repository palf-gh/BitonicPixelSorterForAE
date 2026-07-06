#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Pixel {
	std::uint8_t r;
	std::uint8_t g;
	std::uint8_t b;
	std::uint8_t a;
};

struct Entry {
	float key;
	std::uint32_t index;
};

struct BenchOptions {
	std::string rawPath;
	int width = 0;
	int height = 0;
	bool horizontal = true;
	bool ascending = true;
	float thresholdMin = 0.4f;
	float thresholdMax = 0.6f;
	int iterations = 30;
};

struct Stats {
	double min = 0.0;
	double p50 = 0.0;
	double mean = 0.0;
	double p95 = 0.0;
	double max = 0.0;
};

float Saturate(float v)
{
	return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

float Luma(const Pixel &p)
{
	const float inv = 1.0f / 255.0f;
	const float r = static_cast<float>(p.r) * inv;
	const float g = static_cast<float>(p.g) * inv;
	const float b = static_cast<float>(p.b) * inv;
	return Saturate(0.298912f * r + 0.586611f * g + 0.114478f * b);
}

float SortKey(const Pixel &p, int criterion)
{
	const float inv = 1.0f / 255.0f;
	const float r = Saturate(static_cast<float>(p.r) * inv);
	const float g = Saturate(static_cast<float>(p.g) * inv);
	const float b = Saturate(static_cast<float>(p.b) * inv);
	const float a = Saturate(static_cast<float>(p.a) * inv);

	switch (criterion) {
	case 2:
		return Saturate((r + g + b) * (1.0f / 3.0f));
	case 3:
		return Saturate(r * g * b);
	case 4:
		return Saturate((std::min)(r, (std::min)(g, b)));
	case 5:
		return Saturate((std::max)(r, (std::max)(g, b)));
	case 6:
		return r;
	case 7:
		return g;
	case 8:
		return b;
	case 9:
		return a;
	case 10:
	case 11: {
		const float maxRGB = (std::max)(r, (std::max)(g, b));
		const float minRGB = (std::min)(r, (std::min)(g, b));
		const float delta = maxRGB - minRGB;
		if (criterion == 11) {
			return maxRGB <= 0.0f ? 0.0f : Saturate(delta / maxRGB);
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
		return Saturate(hue * (1.0f / 6.0f));
	}
	default:
		return Luma(p);
	}
}

bool Affected(float key, float thresholdMin, float thresholdMax, bool outside)
{
	const bool inside = key >= thresholdMin && key <= thresholdMax;
	return outside ? !inside : inside;
}

std::size_t PixelIndex(int width, bool horizontal, int line, int k)
{
	return horizontal
		? static_cast<std::size_t>(line) * static_cast<std::size_t>(width) +
			  static_cast<std::size_t>(k)
		: static_cast<std::size_t>(k) * static_cast<std::size_t>(width) +
			  static_cast<std::size_t>(line);
}

std::vector<Pixel> ReadRaw(const std::string &path, int width, int height)
{
	if (width <= 0 || height <= 0) {
		throw std::runtime_error("width and height must be positive");
	}
	const std::size_t pixelCount =
		static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
	std::vector<Pixel> pixels(pixelCount);
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		throw std::runtime_error("failed to open raw file: " + path);
	}
	in.read(reinterpret_cast<char *>(pixels.data()),
			static_cast<std::streamsize>(pixels.size() * sizeof(Pixel)));
	if (in.gcount() != static_cast<std::streamsize>(pixels.size() * sizeof(Pixel))) {
		throw std::runtime_error("raw file ended before width*height*4 bytes");
	}
	return pixels;
}

bool BeforeAsc(const Entry &a, const Entry &b)
{
	return a.key < b.key || (a.key == b.key && a.index < b.index);
}

bool BeforeDesc(const Entry &a, const Entry &b)
{
	return a.key > b.key || (a.key == b.key && a.index < b.index);
}

void SortRun(std::vector<Entry> &run, bool ascending)
{
	if (ascending) {
		std::stable_sort(run.begin(), run.end(), BeforeAsc);
	} else {
		std::stable_sort(run.begin(), run.end(), BeforeDesc);
	}
}

void SortAxisLuma(const std::vector<Pixel> &src,
				  std::vector<Pixel> &dst,
				  int width,
				  int height,
				  bool horizontal,
				  bool ascending,
				  float thresholdMin,
				  float thresholdMax)
{
	const int lineCount = horizontal ? height : width;
	const int lineLen = horizontal ? width : height;
	std::vector<Pixel> sourceLine(static_cast<std::size_t>(lineLen));
	std::vector<Pixel> sortedLine(static_cast<std::size_t>(lineLen));
	std::vector<float> keys(static_cast<std::size_t>(lineLen));
	std::vector<Entry> run;
	run.reserve(static_cast<std::size_t>(lineLen));

	for (int line = 0; line < lineCount; ++line) {
		for (int k = 0; k < lineLen; ++k) {
			const std::size_t pix = PixelIndex(width, horizontal, line, k);
			sourceLine[static_cast<std::size_t>(k)] = src[pix];
			sortedLine[static_cast<std::size_t>(k)] = src[pix];
			keys[static_cast<std::size_t>(k)] = Luma(src[pix]);
		}

		int cursor = 0;
		while (cursor < lineLen) {
			const float key = keys[static_cast<std::size_t>(cursor)];
			if (key >= thresholdMin && key <= thresholdMax) {
				const int start = cursor;
				run.clear();
				while (cursor < lineLen) {
					const float runKey = keys[static_cast<std::size_t>(cursor)];
					if (runKey < thresholdMin || runKey > thresholdMax) {
						break;
					}
					run.push_back(Entry{runKey, static_cast<std::uint32_t>(cursor)});
					++cursor;
				}
				if (run.size() > 1u) {
					SortRun(run, ascending);
					for (std::size_t i = 0; i < run.size(); ++i) {
						sortedLine[static_cast<std::size_t>(start) + i] =
							sourceLine[static_cast<std::size_t>(run[i].index)];
					}
				}
			} else {
				++cursor;
			}
		}

		for (int k = 0; k < lineLen; ++k) {
			dst[PixelIndex(width, horizontal, line, k)] =
				sortedLine[static_cast<std::size_t>(k)];
		}
	}
}

int CycleShift(std::size_t count, double cycleDegrees)
{
	if (count <= 1u) {
		return 0;
	}
	const double turns = cycleDegrees / 360.0;
	long long shift = static_cast<long long>(
		std::llround(turns * static_cast<double>(count)));
	const long long n = static_cast<long long>(count);
	shift %= n;
	if (shift < 0) {
		shift += n;
	}
	return static_cast<int>(shift);
}

void SortAxisGeneric(const std::vector<Pixel> &src,
					 std::vector<Pixel> &dst,
					 int width,
					 int height,
					 bool horizontal,
					 bool ascending,
					 float thresholdMin,
					 float thresholdMax,
					 int criterion,
					 int trigger,
					 bool outside,
					 double cycleDegrees,
					 bool zeroShiftFastPath)
{
	const int lineCount = horizontal ? height : width;
	const int lineLen = horizontal ? width : height;
	std::vector<Pixel> sourceLine(static_cast<std::size_t>(lineLen));
	std::vector<Pixel> sortedLine(static_cast<std::size_t>(lineLen));
	std::vector<float> keys(static_cast<std::size_t>(lineLen));
	std::vector<float> triggerKeys(static_cast<std::size_t>(lineLen));
	std::vector<Entry> run;
	run.reserve(static_cast<std::size_t>(lineLen));

	for (int line = 0; line < lineCount; ++line) {
		for (int k = 0; k < lineLen; ++k) {
			const std::size_t pix = PixelIndex(width, horizontal, line, k);
			sourceLine[static_cast<std::size_t>(k)] = src[pix];
			sortedLine[static_cast<std::size_t>(k)] = src[pix];
			if (criterion == trigger) {
				const float key = SortKey(src[pix], criterion);
				keys[static_cast<std::size_t>(k)] = key;
				triggerKeys[static_cast<std::size_t>(k)] = key;
			} else {
				keys[static_cast<std::size_t>(k)] = SortKey(src[pix], criterion);
				triggerKeys[static_cast<std::size_t>(k)] = SortKey(src[pix], trigger);
			}
		}

		int cursor = 0;
		while (cursor < lineLen) {
			if (Affected(triggerKeys[static_cast<std::size_t>(cursor)],
						 thresholdMin, thresholdMax, outside)) {
				const int start = cursor;
				run.clear();
				while (cursor < lineLen &&
					   Affected(triggerKeys[static_cast<std::size_t>(cursor)],
								thresholdMin, thresholdMax, outside)) {
					run.push_back(Entry{
						keys[static_cast<std::size_t>(cursor)],
						static_cast<std::uint32_t>(cursor)});
					++cursor;
				}
				if (run.size() > 1u) {
					SortRun(run, ascending);
					const int shift = CycleShift(run.size(), cycleDegrees);
					if (zeroShiftFastPath && shift == 0) {
						for (std::size_t i = 0; i < run.size(); ++i) {
							sortedLine[static_cast<std::size_t>(start) + i] =
								sourceLine[static_cast<std::size_t>(run[i].index)];
						}
					} else {
						for (std::size_t i = 0; i < run.size(); ++i) {
							const std::size_t dest =
								(i + static_cast<std::size_t>(shift)) % run.size();
							sortedLine[static_cast<std::size_t>(start) + dest] =
								sourceLine[static_cast<std::size_t>(run[i].index)];
						}
					}
				}
			} else {
				++cursor;
			}
		}

		for (int k = 0; k < lineLen; ++k) {
			dst[PixelIndex(width, horizontal, line, k)] =
				sortedLine[static_cast<std::size_t>(k)];
		}
	}
}

using SortFn = void (*)(const BenchOptions &,
						const std::vector<Pixel> &,
						std::vector<Pixel> &);

struct Variant {
	const char *name;
	SortFn fn;
};

struct VariantResult {
	const char *name;
	Stats stats;
};

void RunV1(const BenchOptions &opt, const std::vector<Pixel> &src, std::vector<Pixel> &dst)
{
	SortAxisLuma(src, dst, opt.width, opt.height, opt.horizontal, opt.ascending,
				 opt.thresholdMin, opt.thresholdMax);
}

void RunGenericBefore(const BenchOptions &opt,
					  const std::vector<Pixel> &src,
					  std::vector<Pixel> &dst)
{
	SortAxisGeneric(src, dst, opt.width, opt.height, opt.horizontal, opt.ascending,
					opt.thresholdMin, opt.thresholdMax,
					1, 1, false, 0.0, false);
}

void RunGenericCurrent(const BenchOptions &opt,
					   const std::vector<Pixel> &src,
					   std::vector<Pixel> &dst)
{
	SortAxisGeneric(src, dst, opt.width, opt.height, opt.horizontal, opt.ascending,
					opt.thresholdMin, opt.thresholdMax,
					1, 1, false, 0.0, true);
}

void RunOptimisedDispatch(const BenchOptions &opt,
						  const std::vector<Pixel> &src,
						  std::vector<Pixel> &dst)
{
	RunV1(opt, src, dst);
}

std::uint64_t HashPixels(const std::vector<Pixel> &pixels)
{
	std::uint64_t hash = 1469598103934665603ull;
	const auto *bytes = reinterpret_cast<const std::uint8_t *>(pixels.data());
	const std::size_t bytesLen = pixels.size() * sizeof(Pixel);
	for (std::size_t i = 0; i < bytesLen; ++i) {
		hash ^= bytes[i];
		hash *= 1099511628211ull;
	}
	return hash;
}

Stats ComputeStats(std::vector<double> values)
{
	std::sort(values.begin(), values.end());
	const double sum = std::accumulate(values.begin(), values.end(), 0.0);
	const std::size_t p95Index = (std::min)(
		values.size() - 1u,
		static_cast<std::size_t>(std::ceil(values.size() * 0.95) - 1.0));
	return Stats{
		values.front(),
		values[values.size() / 2u],
		sum / static_cast<double>(values.size()),
		values[p95Index],
		values.back()};
}

std::vector<VariantResult> MeasureVariants(const BenchOptions &opt,
										   const std::vector<Pixel> &src,
										   std::vector<Pixel> &dst,
										   const std::vector<Variant> &variants)
{
	std::vector<std::vector<double>> elapsed(variants.size());
	for (auto &values : elapsed) {
		values.reserve(static_cast<std::size_t>(opt.iterations));
	}

	for (std::size_t i = 0; i < variants.size(); ++i) {
		variants[i].fn(opt, src, dst);
	}

	for (int iteration = 0; iteration < opt.iterations; ++iteration) {
		const std::size_t first =
			static_cast<std::size_t>(iteration) % variants.size();
		for (std::size_t offset = 0; offset < variants.size(); ++offset) {
			const std::size_t index = (first + offset) % variants.size();
			const auto start = std::chrono::steady_clock::now();
			variants[index].fn(opt, src, dst);
			const auto end = std::chrono::steady_clock::now();
			elapsed[index].push_back(
				std::chrono::duration<double, std::milli>(end - start).count());
		}
	}

	std::vector<VariantResult> results;
	results.reserve(variants.size());
	for (std::size_t i = 0; i < variants.size(); ++i) {
		results.push_back(VariantResult{
			variants[i].name,
			ComputeStats(std::move(elapsed[i]))});
	}
	return results;
}

void PrintStats(const char *name, const Stats &stats)
{
	std::cout << std::left << std::setw(20) << name
			  << std::right << " min=" << std::setw(8) << std::fixed << std::setprecision(3) << stats.min
			  << " ms  p50=" << std::setw(8) << stats.p50
			  << " ms  mean=" << std::setw(8) << stats.mean
			  << " ms  p95=" << std::setw(8) << stats.p95
			  << " ms  max=" << std::setw(8) << stats.max
			  << " ms\n";
}

int ToInt(const char *value, const char *name)
{
	char *end = nullptr;
	const long parsed = std::strtol(value, &end, 10);
	if (!end || *end != '\0') {
		throw std::runtime_error(std::string("invalid integer for ") + name);
	}
	return static_cast<int>(parsed);
}

float ToFloat(const char *value, const char *name)
{
	char *end = nullptr;
	const float parsed = std::strtof(value, &end);
	if (!end || *end != '\0') {
		throw std::runtime_error(std::string("invalid float for ") + name);
	}
	return parsed;
}

BenchOptions ParseArgs(int argc, char **argv)
{
	BenchOptions opt;
	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		auto requireValue = [&](const char *name) -> const char * {
			if (i + 1 >= argc) {
				throw std::runtime_error(std::string("missing value for ") + name);
			}
			return argv[++i];
		};

		if (arg == "--raw") {
			opt.rawPath = requireValue("--raw");
		} else if (arg == "--width") {
			opt.width = ToInt(requireValue("--width"), "--width");
		} else if (arg == "--height") {
			opt.height = ToInt(requireValue("--height"), "--height");
		} else if (arg == "--direction") {
			const std::string value = requireValue("--direction");
			if (value == "horizontal" || value == "Horizontal") {
				opt.horizontal = true;
			} else if (value == "vertical" || value == "Vertical") {
				opt.horizontal = false;
			} else {
				throw std::runtime_error("direction must be horizontal or vertical");
			}
		} else if (arg == "--iterations") {
			opt.iterations = ToInt(requireValue("--iterations"), "--iterations");
		} else if (arg == "--threshold-min") {
			opt.thresholdMin = ToFloat(requireValue("--threshold-min"), "--threshold-min");
		} else if (arg == "--threshold-max") {
			opt.thresholdMax = ToFloat(requireValue("--threshold-max"), "--threshold-max");
		} else if (arg == "--descending") {
			opt.ascending = false;
		} else if (arg == "--help") {
			std::cout
				<< "Usage: bps_axis_bench --raw image.rgba --width W --height H "
				<< "[--direction horizontal|vertical] [--iterations N]\n";
			std::exit(0);
		} else {
			throw std::runtime_error("unknown argument: " + arg);
		}
	}

	if (opt.rawPath.empty()) {
		throw std::runtime_error("--raw is required");
	}
	if (opt.width <= 0 || opt.height <= 0) {
		throw std::runtime_error("--width and --height are required");
	}
	if (opt.iterations <= 0) {
		throw std::runtime_error("--iterations must be positive");
	}
	return opt;
}

} // namespace

int main(int argc, char **argv)
{
	try {
		const BenchOptions opt = ParseArgs(argc, argv);
		const std::vector<Pixel> src = ReadRaw(opt.rawPath, opt.width, opt.height);
		std::vector<Pixel> dst(src.size());

		RunV1(opt, src, dst);
		const std::uint64_t v1Hash = HashPixels(dst);
		RunGenericBefore(opt, src, dst);
		const std::uint64_t genericBeforeHash = HashPixels(dst);
		RunGenericCurrent(opt, src, dst);
		const std::uint64_t genericCurrentHash = HashPixels(dst);
		RunOptimisedDispatch(opt, src, dst);
		const std::uint64_t optimisedHash = HashPixels(dst);

		if (v1Hash != genericBeforeHash ||
			v1Hash != genericCurrentHash ||
			v1Hash != optimisedHash) {
			throw std::runtime_error("algorithm outputs diverged");
		}

		const std::vector<Variant> variants = {
			{"v1-luma", RunV1},
			{"generic-before", RunGenericBefore},
			{"generic-current", RunGenericCurrent},
			{"optimised-dispatch", RunOptimisedDispatch},
		};
		const std::vector<VariantResult> results =
			MeasureVariants(opt, src, dst, variants);

		std::cout << "image=" << opt.width << "x" << opt.height
				  << " direction=" << (opt.horizontal ? "Horizontal" : "Vertical")
				  << " iterations=" << opt.iterations
				  << " thresholds=" << std::fixed << std::setprecision(3)
				  << opt.thresholdMin << "-" << opt.thresholdMax
				  << " output_hash=0x" << std::hex << v1Hash << std::dec << "\n";
		for (const VariantResult &result : results) {
			PrintStats(result.name, result.stats);
		}
		const double beforeMean = results[1].stats.mean;
		const double optimisedMean = results[3].stats.mean;
		std::cout << "speedup_vs_before_mean="
				  << std::fixed << std::setprecision(3)
				  << (beforeMean / optimisedMean) << "x\n";
		return 0;
	} catch (const std::exception &ex) {
		std::cerr << "error: " << ex.what() << "\n";
		return 1;
	}
}
