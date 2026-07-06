#include "BitonicSortAxisLuma.cso.h"
#include "BitonicSortAxisLuma.rs.h"
#include "BitonicSortAxisLumaFull.cso.h"
#include "BitonicSortAxisLumaFull.rs.h"
#include "BitonicSortKernel.cso.h"
#include "BitonicSortKernel.rs.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NOMINMAX
	#define NOMINMAX
#endif
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

namespace {

using Microsoft::WRL::ComPtr;

constexpr std::uint32_t kBpsModeAxis = 1;
constexpr std::uint32_t kBpsCriterionLuminance = 1;
constexpr std::uint32_t kBpsAffectInside = 1;

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

struct DirectXSortParams {
	std::int32_t srcPitch;
	std::int32_t dstPitch;
	std::int32_t width;
	std::int32_t height;
	std::int32_t inputOriginX;
	std::int32_t inputOriginY;
	std::int32_t inputWidth;
	std::int32_t inputHeight;
	std::int32_t criterionPitch;
	std::int32_t criterionOriginX;
	std::int32_t criterionOriginY;
	std::int32_t criterionWidth;
	std::int32_t criterionHeight;
	std::int32_t triggerPitch;
	std::int32_t triggerOriginX;
	std::int32_t triggerOriginY;
	std::int32_t triggerWidth;
	std::int32_t triggerHeight;
	std::int32_t outputOriginX;
	std::int32_t outputOriginY;
	std::int32_t outputWidth;
	std::int32_t outputHeight;
	std::int32_t mode;
	std::int32_t direction;
	std::int32_t ordering;
	std::int32_t criterion;
	std::int32_t trigger;
	std::int32_t affect;
	float cycleDegrees;
	std::int32_t lineCount;
	std::int32_t freePMin;
	std::int32_t freeQMin;
	std::int32_t freeLineLength;
	std::int32_t radialLength;
	std::int32_t domainStride;
	float thresholdMin;
	float thresholdMax;
	float angleCos;
	float angleSin;
	float centerX;
	float centerY;
	float swirlK;
	std::int32_t swirlLineMin;
	std::int32_t pathDirection;
	std::int32_t pathClosed;
	float pathLength;
	std::int32_t pathSMin;
	std::int32_t pathNMin;
	std::int32_t pathSampleCount;
};

struct ShaderCase {
	const char *name = nullptr;
	const std::uint8_t *rs = nullptr;
	std::size_t rsSize = 0;
	const std::uint8_t *cso = nullptr;
	std::size_t csoSize = 0;
	int srvCount = 1;
};

struct Pipeline {
	ComPtr<ID3D12RootSignature> rootSignature;
	ComPtr<ID3D12PipelineState> state;
	ComPtr<ID3D12DescriptorHeap> descriptorHeap;
};

void Check(HRESULT hr, const char *what)
{
	if (FAILED(hr)) {
		throw std::runtime_error(std::string(what) + " failed, HRESULT=0x" +
			[] (HRESULT value) {
				char buffer[16] = {};
				std::snprintf(buffer, sizeof(buffer), "%08x", static_cast<unsigned>(value));
				return std::string(buffer);
			}(hr));
	}
}

std::size_t Align(std::size_t value, std::size_t alignment)
{
	return (value + alignment - 1u) & ~(alignment - 1u);
}

D3D12_HEAP_PROPERTIES HeapProps(D3D12_HEAP_TYPE type)
{
	D3D12_HEAP_PROPERTIES props = {};
	props.Type = type;
	props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
	props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
	return props;
}

D3D12_RESOURCE_DESC BufferDesc(std::uint64_t bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
	D3D12_RESOURCE_DESC desc = {};
	desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	desc.Width = bytes;
	desc.Height = 1;
	desc.DepthOrArraySize = 1;
	desc.MipLevels = 1;
	desc.Format = DXGI_FORMAT_UNKNOWN;
	desc.SampleDesc.Count = 1;
	desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	desc.Flags = flags;
	return desc;
}

ComPtr<ID3D12Resource> CreateBuffer(
	ID3D12Device *device,
	std::uint64_t bytes,
	D3D12_HEAP_TYPE heapType,
	D3D12_RESOURCE_STATES state,
	D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
	ComPtr<ID3D12Resource> resource;
	const D3D12_HEAP_PROPERTIES heap = HeapProps(heapType);
	const D3D12_RESOURCE_DESC desc = BufferDesc(bytes, flags);
	Check(device->CreateCommittedResource(
		&heap,
		D3D12_HEAP_FLAG_NONE,
		&desc,
		state,
		nullptr,
		IID_PPV_ARGS(&resource)),
		"CreateCommittedResource");
	return resource;
}

std::vector<std::uint8_t> ReadRaw(const std::string &path, int width, int height)
{
	const std::size_t expected = static_cast<std::size_t>(width) *
		static_cast<std::size_t>(height) * 4u;
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

std::vector<std::uint8_t> Rgba8ToBgra128(const std::vector<std::uint8_t> &rgba)
{
	std::vector<std::uint8_t> bgra128((rgba.size() / 4u) * 16u);
	for (std::size_t i = 0, p = 0; i < rgba.size(); i += 4u, ++p) {
		const float r = static_cast<float>(rgba[i + 0u]) * (1.0f / 255.0f);
		const float g = static_cast<float>(rgba[i + 1u]) * (1.0f / 255.0f);
		const float b = static_cast<float>(rgba[i + 2u]) * (1.0f / 255.0f);
		const float a = static_cast<float>(rgba[i + 3u]) * (1.0f / 255.0f);
		float packed[4] = {b, g, r, a};
		std::memcpy(bgra128.data() + p * 16u, packed, sizeof(packed));
	}
	return bgra128;
}

std::uint64_t HashBytes(const std::vector<std::uint8_t> &bytes)
{
	std::uint64_t h = 1469598103934665603ull;
	for (std::uint8_t b : bytes) {
		h ^= static_cast<std::uint64_t>(b);
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
			std::min<double>(values.size() - 1u, std::floor(q * static_cast<double>(values.size() - 1u))));
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

std::string WideToUtf8(const wchar_t *text)
{
	if (!text || !*text) {
		return {};
	}
	const int needed = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
	if (needed <= 1) {
		return {};
	}
	std::string out(static_cast<std::size_t>(needed - 1), '\0');
	WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), needed, nullptr, nullptr);
	return out;
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
				<< "Usage: bps_dx_axis_bench --raw image.rgba --width W --height H "
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

ComPtr<IDXGIAdapter1> ChooseAdapter(IDXGIFactory6 *factory)
{
	ComPtr<IDXGIAdapter1> adapter;
	for (UINT i = 0; factory->EnumAdapterByGpuPreference(
			i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; ++i) {
		DXGI_ADAPTER_DESC1 desc = {};
		adapter->GetDesc1(&desc);
		if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
			SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
			return adapter;
		}
		adapter.Reset();
	}
	throw std::runtime_error("no hardware D3D12 adapter found");
}

Pipeline CreatePipeline(
	ID3D12Device *device,
	const ShaderCase &shader,
	const ComPtr<ID3D12Resource> &cb,
	const ComPtr<ID3D12Resource> &dst,
	const ComPtr<ID3D12Resource> &src,
	std::uint64_t dstBytes,
	std::uint64_t srcBytes)
{
	Pipeline pipeline;
	Check(device->CreateRootSignature(
		0,
		shader.rs,
		shader.rsSize,
		IID_PPV_ARGS(&pipeline.rootSignature)),
		"CreateRootSignature");

	D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
	psoDesc.pRootSignature = pipeline.rootSignature.Get();
	psoDesc.CS.pShaderBytecode = shader.cso;
	psoDesc.CS.BytecodeLength = shader.csoSize;
	Check(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pipeline.state)),
		"CreateComputePipelineState");

	const UINT descriptorCount = static_cast<UINT>(2 + shader.srvCount);
	D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
	heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	heapDesc.NumDescriptors = descriptorCount;
	heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	Check(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&pipeline.descriptorHeap)),
		"CreateDescriptorHeap");

	const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	D3D12_CPU_DESCRIPTOR_HANDLE cpu = pipeline.descriptorHeap->GetCPUDescriptorHandleForHeapStart();

	D3D12_CONSTANT_BUFFER_VIEW_DESC cbv = {};
	cbv.BufferLocation = cb->GetGPUVirtualAddress();
	cbv.SizeInBytes = static_cast<UINT>(Align(sizeof(DirectXSortParams), 256u));
	device->CreateConstantBufferView(&cbv, cpu);

	cpu.ptr += step;
	D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
	uav.Format = DXGI_FORMAT_R32_TYPELESS;
	uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uav.Buffer.NumElements = static_cast<UINT>(dstBytes / 4u);
	uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
	device->CreateUnorderedAccessView(dst.Get(), nullptr, &uav, cpu);

	for (int i = 0; i < shader.srvCount; ++i) {
		cpu.ptr += step;
		D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
		srv.Format = DXGI_FORMAT_R32_TYPELESS;
		srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srv.Buffer.NumElements = static_cast<UINT>(srcBytes / 4u);
		srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
		device->CreateShaderResourceView(src.Get(), &srv, cpu);
	}

	return pipeline;
}

void BindDescriptorTables(ID3D12Device *device, ID3D12GraphicsCommandList *list, const Pipeline &pipeline)
{
	const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	D3D12_GPU_DESCRIPTOR_HANDLE base = pipeline.descriptorHeap->GetGPUDescriptorHandleForHeapStart();
	D3D12_GPU_DESCRIPTOR_HANDLE cbv = base;
	D3D12_GPU_DESCRIPTOR_HANDLE uav = {base.ptr + step};
	D3D12_GPU_DESCRIPTOR_HANDLE srv = {base.ptr + step * 2u};
	list->SetComputeRootDescriptorTable(0, cbv);
	list->SetComputeRootDescriptorTable(1, uav);
	list->SetComputeRootDescriptorTable(2, srv);
}

void ExecuteAndWait(
	ID3D12CommandQueue *queue,
	ID3D12Fence *fence,
	HANDLE fenceEvent,
	std::uint64_t &fenceValue,
	ID3D12CommandList *list)
{
	queue->ExecuteCommandLists(1, &list);
	++fenceValue;
	Check(queue->Signal(fence, fenceValue), "Signal");
	if (fence->GetCompletedValue() < fenceValue) {
		Check(fence->SetEventOnCompletion(fenceValue, fenceEvent), "SetEventOnCompletion");
		WaitForSingleObject(fenceEvent, INFINITE);
	}
}

D3D12_RESOURCE_BARRIER Transition(
	ID3D12Resource *resource,
	D3D12_RESOURCE_STATES before,
	D3D12_RESOURCE_STATES after)
{
	D3D12_RESOURCE_BARRIER barrier = {};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Transition.pResource = resource;
	barrier.Transition.StateBefore = before;
	barrier.Transition.StateAfter = after;
	barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	return barrier;
}

D3D12_RESOURCE_BARRIER UavBarrier(ID3D12Resource *resource)
{
	D3D12_RESOURCE_BARRIER barrier = {};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	barrier.UAV.pResource = resource;
	return barrier;
}

std::vector<std::uint8_t> CopyToReadback(
	ID3D12Device *device,
	ID3D12CommandAllocator *allocator,
	ID3D12GraphicsCommandList *list,
	ID3D12CommandQueue *queue,
	ID3D12Fence *fence,
	HANDLE fenceEvent,
	std::uint64_t &fenceValue,
	ID3D12Resource *resource,
	std::uint64_t bytes)
{
	ComPtr<ID3D12Resource> readback = CreateBuffer(
		device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
	Check(allocator->Reset(), "allocator reset");
	Check(list->Reset(allocator, nullptr), "list reset");
	D3D12_RESOURCE_BARRIER barriers[2] = {
		UavBarrier(resource),
		Transition(resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)
	};
	list->ResourceBarrier(2, barriers);
	list->CopyBufferRegion(readback.Get(), 0, resource, 0, bytes);
	D3D12_RESOURCE_BARRIER back = Transition(
		resource, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	list->ResourceBarrier(1, &back);
	Check(list->Close(), "close copy list");
	ExecuteAndWait(queue, fence, fenceEvent, fenceValue, list);

	std::vector<std::uint8_t> bytesOut(static_cast<std::size_t>(bytes));
	void *mapped = nullptr;
	Check(readback->Map(0, nullptr, &mapped), "readback map");
	std::memcpy(bytesOut.data(), mapped, bytesOut.size());
	readback->Unmap(0, nullptr);
	return bytesOut;
}

double DispatchTimed(
	ID3D12Device *device,
	ID3D12CommandAllocator *allocator,
	ID3D12GraphicsCommandList *list,
	ID3D12CommandQueue *queue,
	ID3D12Fence *fence,
	HANDLE fenceEvent,
	std::uint64_t &fenceValue,
	ID3D12QueryHeap *queryHeap,
	ID3D12Resource *queryReadback,
	const Pipeline &pipeline,
	std::uint32_t lineCount,
	std::uint64_t timestampFrequency)
{
	Check(allocator->Reset(), "allocator reset");
	Check(list->Reset(allocator, nullptr), "list reset");
	ID3D12DescriptorHeap *heaps[] = {pipeline.descriptorHeap.Get()};
	list->SetDescriptorHeaps(1, heaps);
	list->SetComputeRootSignature(pipeline.rootSignature.Get());
	list->SetPipelineState(pipeline.state.Get());
	BindDescriptorTables(device, list, pipeline);
	list->EndQuery(queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
	list->Dispatch(lineCount, 1, 1);
	D3D12_RESOURCE_BARRIER uav = {};
	uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	list->ResourceBarrier(1, &uav);
	list->EndQuery(queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
	list->ResolveQueryData(queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, queryReadback, 0);
	Check(list->Close(), "close dispatch list");
	ExecuteAndWait(queue, fence, fenceEvent, fenceValue, list);

	std::uint64_t *timestamps = nullptr;
	Check(queryReadback->Map(0, nullptr, reinterpret_cast<void **>(&timestamps)), "timestamp map");
	const std::uint64_t start = timestamps[0];
	const std::uint64_t end = timestamps[1];
	queryReadback->Unmap(0, nullptr);
	return (static_cast<double>(end - start) * 1000.0) /
		static_cast<double>(timestampFrequency);
}

int Main(int argc, char **argv)
{
	const Options opt = ParseArgs(argc, argv);
	const std::vector<std::uint8_t> raw = ReadRaw(opt.rawPath, opt.width, opt.height);
	const std::vector<std::uint8_t> srcBytes = Rgba8ToBgra128(raw);
	const std::uint64_t srcBufferBytes = static_cast<std::uint64_t>(srcBytes.size());
	const int outputOriginX = opt.horizontal ? 0 : opt.lineStart;
	const int outputOriginY = opt.horizontal ? opt.lineStart : 0;
	const int outputWidth = opt.horizontal ? opt.width : opt.lineCount;
	const int outputHeight = opt.horizontal ? opt.lineCount : opt.height;
	const std::uint64_t dstBufferBytes =
		static_cast<std::uint64_t>(outputWidth) *
		static_cast<std::uint64_t>(outputHeight) * 16ull;

	ComPtr<IDXGIFactory6> factory;
	Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
	ComPtr<IDXGIAdapter1> adapter = ChooseAdapter(factory.Get());
	DXGI_ADAPTER_DESC1 adapterDesc = {};
	adapter->GetDesc1(&adapterDesc);

	ComPtr<ID3D12Device> device;
	Check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)),
		"D3D12CreateDevice");

	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
	ComPtr<ID3D12CommandQueue> queue;
	Check(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)), "CreateCommandQueue");

	ComPtr<ID3D12CommandAllocator> allocator;
	Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator)),
		"CreateCommandAllocator");
	ComPtr<ID3D12GraphicsCommandList> list;
	Check(device->CreateCommandList(
		0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator.Get(), nullptr, IID_PPV_ARGS(&list)),
		"CreateCommandList");
	Check(list->Close(), "initial close");

	ComPtr<ID3D12Fence> fence;
	Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
	HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	if (!fenceEvent) {
		throw std::runtime_error("CreateEvent failed");
	}
	std::uint64_t fenceValue = 0;

	ComPtr<ID3D12Resource> src = CreateBuffer(
		device.Get(), srcBufferBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
	ComPtr<ID3D12Resource> upload = CreateBuffer(
		device.Get(), srcBufferBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
	void *uploadPtr = nullptr;
	Check(upload->Map(0, nullptr, &uploadPtr), "upload map");
	std::memcpy(uploadPtr, srcBytes.data(), srcBytes.size());
	upload->Unmap(0, nullptr);

	ComPtr<ID3D12Resource> genericDst = CreateBuffer(
		device.Get(), dstBufferBytes, D3D12_HEAP_TYPE_DEFAULT,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	ComPtr<ID3D12Resource> lumaDst = CreateBuffer(
		device.Get(), dstBufferBytes, D3D12_HEAP_TYPE_DEFAULT,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	ComPtr<ID3D12Resource> lumaFullDst = CreateBuffer(
		device.Get(), dstBufferBytes, D3D12_HEAP_TYPE_DEFAULT,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	const std::size_t cbBytes = Align(sizeof(DirectXSortParams), 256u);
	ComPtr<ID3D12Resource> cb = CreateBuffer(
		device.Get(), cbBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
	DirectXSortParams params = {};
	params.srcPitch = opt.width;
	params.dstPitch = outputWidth;
	params.width = opt.width;
	params.height = opt.height;
	params.inputWidth = opt.width;
	params.inputHeight = opt.height;
	params.criterionPitch = opt.width;
	params.criterionWidth = opt.width;
	params.criterionHeight = opt.height;
	params.triggerPitch = opt.width;
	params.triggerWidth = opt.width;
	params.triggerHeight = opt.height;
	params.outputOriginX = outputOriginX;
	params.outputOriginY = outputOriginY;
	params.outputWidth = outputWidth;
	params.outputHeight = outputHeight;
	params.mode = static_cast<std::int32_t>(kBpsModeAxis);
	params.direction = opt.horizontal ? 1 : 0;
	params.ordering = opt.ascending ? 1 : 0;
	params.criterion = static_cast<std::int32_t>(kBpsCriterionLuminance);
	params.trigger = static_cast<std::int32_t>(kBpsCriterionLuminance);
	params.affect = static_cast<std::int32_t>(kBpsAffectInside);
	params.cycleDegrees = 0.0f;
	params.lineCount = opt.lineCount;
	params.domainStride = opt.horizontal ? opt.width : opt.height;
	params.thresholdMin = opt.thresholdMin;
	params.thresholdMax = opt.thresholdMax;
	params.angleCos = 1.0f;
	params.angleSin = 0.0f;
	void *cbPtr = nullptr;
	Check(cb->Map(0, nullptr, &cbPtr), "cb map");
	std::memcpy(cbPtr, &params, sizeof(params));
	cb->Unmap(0, nullptr);

	Check(allocator->Reset(), "allocator reset");
	Check(list->Reset(allocator.Get(), nullptr), "list reset");
	list->CopyBufferRegion(src.Get(), 0, upload.Get(), 0, srcBufferBytes);
	D3D12_RESOURCE_BARRIER srcReady = Transition(
		src.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	list->ResourceBarrier(1, &srcReady);
	Check(list->Close(), "upload close");
	ExecuteAndWait(queue.Get(), fence.Get(), fenceEvent, fenceValue, list.Get());

	const ShaderCase genericCase = {
		"generic-axis",
		bps_directx_embedded::kBitonicSortKernel_rs,
		bps_directx_embedded::kBitonicSortKernel_rs_size,
		bps_directx_embedded::kBitonicSortKernel_cso,
		bps_directx_embedded::kBitonicSortKernel_cso_size,
		4
	};
	const ShaderCase lumaCase = {
		"axis-luma",
		bps_directx_embedded::kBitonicSortAxisLuma_rs,
		bps_directx_embedded::kBitonicSortAxisLuma_rs_size,
		bps_directx_embedded::kBitonicSortAxisLuma_cso,
		bps_directx_embedded::kBitonicSortAxisLuma_cso_size,
		1
	};
	const ShaderCase lumaFullCase = {
		"axis-luma-full",
		bps_directx_embedded::kBitonicSortAxisLumaFull_rs,
		bps_directx_embedded::kBitonicSortAxisLumaFull_rs_size,
		bps_directx_embedded::kBitonicSortAxisLumaFull_cso,
		bps_directx_embedded::kBitonicSortAxisLumaFull_cso_size,
		1
	};
	Pipeline genericPipeline = CreatePipeline(
		device.Get(), genericCase, cb, genericDst, src, dstBufferBytes, srcBufferBytes);
	Pipeline lumaPipeline = CreatePipeline(
		device.Get(), lumaCase, cb, lumaDst, src, dstBufferBytes, srcBufferBytes);
	Pipeline lumaFullPipeline = CreatePipeline(
		device.Get(), lumaFullCase, cb, lumaFullDst, src, dstBufferBytes, srcBufferBytes);

	D3D12_QUERY_HEAP_DESC queryDesc = {};
	queryDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
	queryDesc.Count = 2;
	ComPtr<ID3D12QueryHeap> queryHeap;
	Check(device->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&queryHeap)), "CreateQueryHeap");
	ComPtr<ID3D12Resource> queryReadback = CreateBuffer(
		device.Get(), sizeof(std::uint64_t) * 2u,
		D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
	std::uint64_t timestampFrequency = 0;
	Check(queue->GetTimestampFrequency(&timestampFrequency), "GetTimestampFrequency");

	const std::uint32_t lineCount = static_cast<std::uint32_t>(params.lineCount);
	for (int i = 0; i < opt.warmup; ++i) {
		(void)DispatchTimed(device.Get(), allocator.Get(), list.Get(), queue.Get(), fence.Get(),
			fenceEvent, fenceValue, queryHeap.Get(), queryReadback.Get(),
			(i & 1) ? genericPipeline : lumaFullPipeline, lineCount, timestampFrequency);
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
			genericTimes.push_back(DispatchTimed(device.Get(), allocator.Get(), list.Get(), queue.Get(), fence.Get(),
				fenceEvent, fenceValue, queryHeap.Get(), queryReadback.Get(),
				genericPipeline, lineCount, timestampFrequency));
			lumaTimes.push_back(DispatchTimed(device.Get(), allocator.Get(), list.Get(), queue.Get(), fence.Get(),
				fenceEvent, fenceValue, queryHeap.Get(), queryReadback.Get(),
				lumaPipeline, lineCount, timestampFrequency));
			lumaFullTimes.push_back(DispatchTimed(device.Get(), allocator.Get(), list.Get(), queue.Get(), fence.Get(),
				fenceEvent, fenceValue, queryHeap.Get(), queryReadback.Get(),
				lumaFullPipeline, lineCount, timestampFrequency));
		} else if (order == 1) {
			lumaTimes.push_back(DispatchTimed(device.Get(), allocator.Get(), list.Get(), queue.Get(), fence.Get(),
				fenceEvent, fenceValue, queryHeap.Get(), queryReadback.Get(),
				lumaPipeline, lineCount, timestampFrequency));
			lumaFullTimes.push_back(DispatchTimed(device.Get(), allocator.Get(), list.Get(), queue.Get(), fence.Get(),
				fenceEvent, fenceValue, queryHeap.Get(), queryReadback.Get(),
				lumaFullPipeline, lineCount, timestampFrequency));
			genericTimes.push_back(DispatchTimed(device.Get(), allocator.Get(), list.Get(), queue.Get(), fence.Get(),
				fenceEvent, fenceValue, queryHeap.Get(), queryReadback.Get(),
				genericPipeline, lineCount, timestampFrequency));
		} else {
			lumaFullTimes.push_back(DispatchTimed(device.Get(), allocator.Get(), list.Get(), queue.Get(), fence.Get(),
				fenceEvent, fenceValue, queryHeap.Get(), queryReadback.Get(),
				lumaFullPipeline, lineCount, timestampFrequency));
			genericTimes.push_back(DispatchTimed(device.Get(), allocator.Get(), list.Get(), queue.Get(), fence.Get(),
				fenceEvent, fenceValue, queryHeap.Get(), queryReadback.Get(),
				genericPipeline, lineCount, timestampFrequency));
			lumaTimes.push_back(DispatchTimed(device.Get(), allocator.Get(), list.Get(), queue.Get(), fence.Get(),
				fenceEvent, fenceValue, queryHeap.Get(), queryReadback.Get(),
				lumaPipeline, lineCount, timestampFrequency));
		}
	}

	const std::vector<std::uint8_t> genericOut = CopyToReadback(
		device.Get(), allocator.Get(), list.Get(), queue.Get(), fence.Get(), fenceEvent,
		fenceValue, genericDst.Get(), dstBufferBytes);
	const std::vector<std::uint8_t> lumaOut = CopyToReadback(
		device.Get(), allocator.Get(), list.Get(), queue.Get(), fence.Get(), fenceEvent,
		fenceValue, lumaDst.Get(), dstBufferBytes);
	const std::vector<std::uint8_t> lumaFullOut = CopyToReadback(
		device.Get(), allocator.Get(), list.Get(), queue.Get(), fence.Get(), fenceEvent,
		fenceValue, lumaFullDst.Get(), dstBufferBytes);
	if (genericOut != lumaOut || genericOut != lumaFullOut) {
		throw std::runtime_error("generic-axis, axis-luma, and axis-luma-full outputs differ");
	}

	const std::string adapterUtf8 = WideToUtf8(adapterDesc.Description);
	std::cout << "adapter=" << adapterUtf8
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

	CloseHandle(fenceEvent);
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
