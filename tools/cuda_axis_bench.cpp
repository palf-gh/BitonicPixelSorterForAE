#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" cudaError_t BitonicSortAxisLuma_CUDA_Bench(
	const void *src,
	void       *dst,
	int         srcPitch,
	int         dstPitch,
	int         width,
	int         height,
	int         inputOriginX,
	int         inputOriginY,
	int         inputWidth,
	int         inputHeight,
	int         outputOriginX,
	int         outputOriginY,
	int         outputWidth,
	int         outputHeight,
	int         direction,
	int         ordering,
	float       thresholdMin,
	float       thresholdMax,
	int         fastMode);

namespace {

struct Float4 {
	float x;
	float y;
	float z;
	float w;
};

struct Options {
	std::string rawPath;
	int width = 0;
	int height = 0;
	bool horizontal = true;
	bool ascending = true;
	float thresholdMin = 0.4f;
	float thresholdMax = 0.6f;
	int iterations = 60;
	int warmup = 6;
	int lineStart = 0;
	int lineCount = 0;
};

struct Stats {
	double min = 0.0;
	double p50 = 0.0;
	double mean = 0.0;
	double p95 = 0.0;
	double max = 0.0;
};

void Check(cudaError_t err, const char *what)
{
	if (err != cudaSuccess) {
		throw std::runtime_error(
			std::string(what) + " failed: " + cudaGetErrorString(err));
	}
}

std::vector<std::uint8_t> ReadRaw(const std::string &path, int width, int height)
{
	const std::size_t expected =
		static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u;
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		throw std::runtime_error("failed to open raw image: " + path);
	}
	std::vector<std::uint8_t> raw(expected);
	file.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(raw.size()));
	if (static_cast<std::size_t>(file.gcount()) != expected) {
		throw std::runtime_error("raw image size does not match width/height");
	}
	return raw;
}

std::vector<Float4> Rgba8ToBgra128(const std::vector<std::uint8_t> &rgba)
{
	std::vector<Float4> bgra(rgba.size() / 4u);
	for (std::size_t i = 0, p = 0; i < rgba.size(); i += 4u, ++p) {
		bgra[p] = Float4{
			static_cast<float>(rgba[i + 2u]) * (1.0f / 255.0f),
			static_cast<float>(rgba[i + 1u]) * (1.0f / 255.0f),
			static_cast<float>(rgba[i + 0u]) * (1.0f / 255.0f),
			static_cast<float>(rgba[i + 3u]) * (1.0f / 255.0f)};
	}
	return bgra;
}

std::uint64_t HashBytes(const std::vector<Float4> &pixels)
{
	std::uint64_t h = 1469598103934665603ull;
	const auto *bytes = reinterpret_cast<const std::uint8_t *>(pixels.data());
	const std::size_t byteCount = pixels.size() * sizeof(Float4);
	for (std::size_t i = 0; i < byteCount; ++i) {
		h ^= static_cast<std::uint64_t>(bytes[i]);
		h *= 1099511628211ull;
	}
	return h;
}

Stats ComputeStats(std::vector<double> values)
{
	if (values.empty()) {
		return {};
	}
	std::sort(values.begin(), values.end());
	const auto at = [&](double q) {
		const std::size_t idx = static_cast<std::size_t>(
			std::min<double>(
				values.size() - 1u,
				std::floor(q * static_cast<double>(values.size() - 1u))));
		return values[idx];
	};
	const double sum = std::accumulate(values.begin(), values.end(), 0.0);
	return {values.front(), at(0.50), sum / static_cast<double>(values.size()), at(0.95), values.back()};
}

void PrintStats(const char *name, const std::vector<double> &values)
{
	const Stats s = ComputeStats(values);
	std::cout << std::left << std::setw(18) << name
			  << " min=" << std::right << std::setw(8) << std::fixed << std::setprecision(3) << s.min
			  << " ms  p50=" << std::setw(8) << s.p50
			  << " ms  mean=" << std::setw(8) << s.mean
			  << " ms  p95=" << std::setw(8) << s.p95
			  << " ms  max=" << std::setw(8) << s.max << " ms\n";
}

Options ParseArgs(int argc, char **argv)
{
	Options opt;
	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		const auto value = [&]() -> std::string {
			if (i + 1 >= argc) {
				throw std::runtime_error("missing value after " + arg);
			}
			return argv[++i];
		};
		if (arg == "--raw") {
			opt.rawPath = value();
		} else if (arg == "--width") {
			opt.width = std::stoi(value());
		} else if (arg == "--height") {
			opt.height = std::stoi(value());
		} else if (arg == "--direction") {
			const std::string dir = value();
			if (dir == "horizontal") {
				opt.horizontal = true;
			} else if (dir == "vertical") {
				opt.horizontal = false;
			} else {
				throw std::runtime_error("direction must be horizontal or vertical");
			}
		} else if (arg == "--iterations") {
			opt.iterations = std::stoi(value());
		} else if (arg == "--warmup") {
			opt.warmup = std::stoi(value());
		} else if (arg == "--line-start") {
			opt.lineStart = std::stoi(value());
		} else if (arg == "--line-count") {
			opt.lineCount = std::stoi(value());
		} else if (arg == "--threshold-min") {
			opt.thresholdMin = std::stof(value());
		} else if (arg == "--threshold-max") {
			opt.thresholdMax = std::stof(value());
		} else if (arg == "--descending") {
			opt.ascending = false;
		} else if (arg == "--help" || arg == "-h") {
			std::cout
				<< "Usage: bps_cuda_axis_bench --raw image.rgba --width W --height H "
				<< "[--direction horizontal|vertical] [--iterations N] [--warmup N] "
				<< "[--line-start N --line-count N]\n";
			std::exit(0);
		} else {
			throw std::runtime_error("unknown argument: " + arg);
		}
	}
	if (opt.rawPath.empty() || opt.width <= 0 || opt.height <= 0 || opt.iterations <= 0) {
		throw std::runtime_error("pass --raw, --width, --height, and positive --iterations");
	}
	const int availableLines = opt.horizontal ? opt.height : opt.width;
	if (opt.lineCount == 0) {
		opt.lineCount = availableLines - opt.lineStart;
	}
	if (opt.lineStart < 0 || opt.lineCount <= 0 ||
		opt.lineStart + opt.lineCount > availableLines) {
		throw std::runtime_error("line window is outside the image");
	}
	return opt;
}

double DispatchTimed(
	const Options &opt,
	const int outputOriginX,
	const int outputOriginY,
	const int outputWidth,
	const int outputHeight,
	const int fastMode,
	const void *srcDev,
	void *dstDev,
	cudaEvent_t start,
	cudaEvent_t stop)
{
	Check(cudaEventRecord(start, 0), "cudaEventRecord(start)");
	Check(BitonicSortAxisLuma_CUDA_Bench(
		srcDev, dstDev,
		opt.width, outputWidth,
		opt.width, opt.height,
		0, 0, opt.width, opt.height,
		outputOriginX, outputOriginY, outputWidth, outputHeight,
		opt.horizontal ? 1 : 0,
		opt.ascending ? 1 : 0,
		opt.thresholdMin, opt.thresholdMax,
		fastMode), "BitonicSortAxisLuma_CUDA_Bench");
	Check(cudaEventRecord(stop, 0), "cudaEventRecord(stop)");
	Check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
	float ms = 0.0f;
	Check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
	return static_cast<double>(ms);
}

std::vector<Float4> Download(void *devicePtr, std::size_t count)
{
	std::vector<Float4> out(count);
	Check(cudaMemcpy(out.data(), devicePtr, count * sizeof(Float4), cudaMemcpyDeviceToHost),
		"cudaMemcpy device to host");
	return out;
}

int Main(int argc, char **argv)
{
	const Options opt = ParseArgs(argc, argv);
	const std::vector<std::uint8_t> raw = ReadRaw(opt.rawPath, opt.width, opt.height);
	const std::vector<Float4> src = Rgba8ToBgra128(raw);
	const int outputOriginX = opt.horizontal ? 0 : opt.lineStart;
	const int outputOriginY = opt.horizontal ? opt.lineStart : 0;
	const int outputWidth = opt.horizontal ? opt.width : opt.lineCount;
	const int outputHeight = opt.horizontal ? opt.lineCount : opt.height;
	const std::size_t srcCount = src.size();
	const std::size_t dstCount =
		static_cast<std::size_t>(outputWidth) * static_cast<std::size_t>(outputHeight);

	int device = 0;
	Check(cudaGetDevice(&device), "cudaGetDevice");
	cudaDeviceProp props = {};
	Check(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties");

	void *srcDev = nullptr;
	void *genericDev = nullptr;
	void *lumaDev = nullptr;
	void *lumaFullDev = nullptr;
	Check(cudaMalloc(&srcDev, srcCount * sizeof(Float4)), "cudaMalloc src");
	Check(cudaMalloc(&genericDev, dstCount * sizeof(Float4)), "cudaMalloc generic");
	Check(cudaMalloc(&lumaDev, dstCount * sizeof(Float4)), "cudaMalloc luma");
	Check(cudaMalloc(&lumaFullDev, dstCount * sizeof(Float4)), "cudaMalloc luma-full");
	Check(cudaMemcpy(srcDev, src.data(), srcCount * sizeof(Float4), cudaMemcpyHostToDevice),
		"cudaMemcpy host to device");

	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;
	Check(cudaEventCreate(&start), "cudaEventCreate(start)");
	Check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

	for (int i = 0; i < opt.warmup; ++i) {
		const int fastMode = i % 3;
		void *dst = fastMode == 0 ? genericDev : (fastMode == 1 ? lumaDev : lumaFullDev);
		(void)DispatchTimed(
			opt, outputOriginX, outputOriginY, outputWidth, outputHeight,
			fastMode, srcDev, dst, start, stop);
	}

	std::vector<double> genericTimes;
	std::vector<double> lumaTimes;
	std::vector<double> lumaFullTimes;
	genericTimes.reserve(static_cast<std::size_t>(opt.iterations));
	lumaTimes.reserve(static_cast<std::size_t>(opt.iterations));
	lumaFullTimes.reserve(static_cast<std::size_t>(opt.iterations));
	for (int i = 0; i < opt.iterations; ++i) {
		const int order = i % 3;
		if (order == 0) {
			genericTimes.push_back(DispatchTimed(opt, outputOriginX, outputOriginY, outputWidth, outputHeight, 0, srcDev, genericDev, start, stop));
			lumaTimes.push_back(DispatchTimed(opt, outputOriginX, outputOriginY, outputWidth, outputHeight, 1, srcDev, lumaDev, start, stop));
			lumaFullTimes.push_back(DispatchTimed(opt, outputOriginX, outputOriginY, outputWidth, outputHeight, 2, srcDev, lumaFullDev, start, stop));
		} else if (order == 1) {
			lumaTimes.push_back(DispatchTimed(opt, outputOriginX, outputOriginY, outputWidth, outputHeight, 1, srcDev, lumaDev, start, stop));
			lumaFullTimes.push_back(DispatchTimed(opt, outputOriginX, outputOriginY, outputWidth, outputHeight, 2, srcDev, lumaFullDev, start, stop));
			genericTimes.push_back(DispatchTimed(opt, outputOriginX, outputOriginY, outputWidth, outputHeight, 0, srcDev, genericDev, start, stop));
		} else {
			lumaFullTimes.push_back(DispatchTimed(opt, outputOriginX, outputOriginY, outputWidth, outputHeight, 2, srcDev, lumaFullDev, start, stop));
			genericTimes.push_back(DispatchTimed(opt, outputOriginX, outputOriginY, outputWidth, outputHeight, 0, srcDev, genericDev, start, stop));
			lumaTimes.push_back(DispatchTimed(opt, outputOriginX, outputOriginY, outputWidth, outputHeight, 1, srcDev, lumaDev, start, stop));
		}
	}

	const std::vector<Float4> genericOut = Download(genericDev, dstCount);
	const std::vector<Float4> lumaOut = Download(lumaDev, dstCount);
	const std::vector<Float4> lumaFullOut = Download(lumaFullDev, dstCount);
	if (genericOut.size() != lumaOut.size() ||
		genericOut.size() != lumaFullOut.size() ||
		std::memcmp(genericOut.data(), lumaOut.data(), genericOut.size() * sizeof(Float4)) != 0 ||
		std::memcmp(genericOut.data(), lumaFullOut.data(), genericOut.size() * sizeof(Float4)) != 0) {
		throw std::runtime_error("generic-axis, axis-luma, and axis-luma-full outputs differ");
	}

	std::cout << "device=" << props.name
			  << " image=" << opt.width << "x" << opt.height
			  << " direction=" << (opt.horizontal ? "Horizontal" : "Vertical")
			  << " line_window=" << opt.lineStart << "+" << opt.lineCount
			  << " iterations=" << opt.iterations
			  << " thresholds=" << std::fixed << std::setprecision(3)
			  << opt.thresholdMin << "-" << opt.thresholdMax
			  << " output_hash=0x" << std::hex << HashBytes(genericOut) << std::dec << "\n";
	PrintStats("generic-axis", genericTimes);
	PrintStats("axis-luma", lumaTimes);
	PrintStats("axis-luma-full", lumaFullTimes);
	const Stats genericStats = ComputeStats(genericTimes);
	const Stats lumaStats = ComputeStats(lumaTimes);
	const Stats lumaFullStats = ComputeStats(lumaFullTimes);
	if (lumaStats.mean > 0.0) {
		std::cout << "speedup_vs_generic_mean=" << std::fixed << std::setprecision(3)
				  << (genericStats.mean / lumaStats.mean) << "x\n";
	}
	if (lumaFullStats.mean > 0.0) {
		std::cout << "full_speedup_vs_generic_mean=" << std::fixed << std::setprecision(3)
				  << (genericStats.mean / lumaFullStats.mean) << "x\n";
		std::cout << "full_speedup_vs_axis_luma_mean=" << std::fixed << std::setprecision(3)
				  << (lumaStats.mean / lumaFullStats.mean) << "x\n";
	}

	Check(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
	Check(cudaEventDestroy(start), "cudaEventDestroy(start)");
	Check(cudaFree(lumaFullDev), "cudaFree luma-full");
	Check(cudaFree(lumaDev), "cudaFree luma");
	Check(cudaFree(genericDev), "cudaFree generic");
	Check(cudaFree(srcDev), "cudaFree src");
	return 0;
}

} // namespace

int main(int argc, char **argv)
{
	try {
		return Main(argc, argv);
	} catch (const std::exception &ex) {
		std::cerr << "error: " << ex.what() << "\n";
		return 1;
	}
}
