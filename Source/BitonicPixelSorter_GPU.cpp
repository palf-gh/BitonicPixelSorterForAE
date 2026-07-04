/*
	BitonicPixelSorter_GPU.cpp

	GPU command handling: device setup/setdown and the GPU smart-render dispatch.
	After Effects GPU worlds are PF_PixelFormat_GPU_BGRA128 (linear, row-pitched
	float4 buffers in BGRA order).

	Axis mode is a single pass (src → dst). Non-axis modes use exact multi-pass:
	sort each path into a temporary domain of source indices, then inverse-map
	every output pixel and gather — matching the CPU oracle and avoiding the
	forward-scatter races inherent to Free Angle / Rotation / Radial paths.

	Each framework backend is compiled only when its toolchain is available; the
	build system defines BPS_HAS_CUDA / BPS_HAS_OPENCL / BPS_HAS_HLSL /
	BPS_HAS_METAL accordingly. When no backend is built these entries are inert
	(and AE never reaches them, because GlobalSetup does not advertise GPU then).
*/

#if defined(BPS_HAS_CUDA)
	#include <cuda_runtime.h>
	#include <cuda.h>
#endif

#if defined(BPS_HAS_OPENCL)
	#if defined(_WIN32)
		#include <CL/cl.h>
	#else
		#include <OpenCL/cl.h>
	#endif
	#include "BitonicPixelSorter_Kernel.cl.h"
#endif

#include "BitonicPixelSorter.h"
#include "BitonicPixelSorter_GpuEligibility.h"
#include "BitonicPixelSorter_PathGeometry.h"

#include <atomic>
#include <algorithm>
#if defined(BPS_RENDER_DIAG)
	#include <cstdarg>
	#include <cstdio>
#endif
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#if defined(BPS_HAS_HLSL)
	#include "BPS_DirectXShaderLoad.h"
	#include "DirectXUtils.h"
	#include <dxgi1_6.h>
#endif

#if defined(BPS_HAS_METAL)
	#include "BPS_MetalBackend.h"
#endif

#if defined(BPS_HAS_CUDA)
	// Host launch wrapper, defined in GPU/BitonicPixelSorter_Kernel.cu.
	extern "C" cudaError_t BitonicSort_CUDA(
		const void *src, const void *criterionMem, const void *triggerMem, void *dst,
		int srcPitch, int dstPitch, int width, int height,
		int inputOriginX, int inputOriginY, int inputWidth, int inputHeight,
		int criterionPitch, int criterionOriginX, int criterionOriginY,
		int criterionWidth, int criterionHeight,
		int triggerPitch, int triggerOriginX, int triggerOriginY,
		int triggerWidth, int triggerHeight,
		int outputOriginX, int outputOriginY, int outputWidth, int outputHeight,
		int mode, int direction, int ordering, int criterion, int trigger, int affect,
		float cycleDegrees, int lineCount,
		int freePMin, int freeQMin, int freeLineLength, int radialLength,
		int domainStride,
		float thresholdMin, float thresholdMax,
		float angleCos, float angleSin, float centerX, float centerY,
		float swirlK, int swirlLineMin,
		int pathDirection, int pathClosed, float pathLength,
		int pathSMin, int pathNMin, int pathSampleCount,
		const void *pathSamplesHost,
		unsigned long long pathMapKey,
		int useGpuPathMapBuild,
		int mappedRecordCount, int mappedWorkItemCount,
		const void *mappedRecordsHost,
		const void *mappedLineOffsetsHost,
		const void *mappedWorkOffsetsHost);
	extern "C" void BitonicClearCudaPathMapCache();
#endif

#if defined(BPS_HAS_OPENCL)
namespace {

struct OpenCLGPUData {
	cl_program program;
	cl_kernel sort_kernel;
	cl_kernel domain_sort_kernel;
	cl_kernel apply_domain_kernel;
	cl_kernel copy_kernel;
	cl_kernel mapped_sort_kernel;
	cl_kernel path_classify_count_kernel;
	cl_kernel path_scatter_records_kernel;
	cl_kernel path_sort_records_kernel;
	cl_mem path_records_mem;
	cl_mem path_line_offsets_mem;
	cl_mem path_work_offsets_mem;
	unsigned long long path_map_key;
	int path_map_width;
	int path_map_height;
	int path_map_line_count;
	int path_mapped_record_count;
	int path_mapped_work_item_count;
};

inline PF_Err CL2Err(cl_int cl_result)
{
	return cl_result == CL_SUCCESS ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED;
}

#define BPS_CL_ERR(FUNC) ERR(CL2Err(FUNC))

static void ReleaseOpenCLPathMap(OpenCLGPUData *cl_dataP)
{
	if (!cl_dataP) {
		return;
	}
	if (cl_dataP->path_work_offsets_mem) {
		(void)clReleaseMemObject(cl_dataP->path_work_offsets_mem);
		cl_dataP->path_work_offsets_mem = 0;
	}
	if (cl_dataP->path_line_offsets_mem) {
		(void)clReleaseMemObject(cl_dataP->path_line_offsets_mem);
		cl_dataP->path_line_offsets_mem = 0;
	}
	if (cl_dataP->path_records_mem) {
		(void)clReleaseMemObject(cl_dataP->path_records_mem);
		cl_dataP->path_records_mem = 0;
	}
	cl_dataP->path_map_key = 0ull;
	cl_dataP->path_map_width = 0;
	cl_dataP->path_map_height = 0;
	cl_dataP->path_map_line_count = 0;
	cl_dataP->path_mapped_record_count = 0;
	cl_dataP->path_mapped_work_item_count = 0;
}

static void ReleaseOpenCLData(OpenCLGPUData *cl_dataP)
{
	if (cl_dataP) {
		ReleaseOpenCLPathMap(cl_dataP);
		if (cl_dataP->path_sort_records_kernel) {
			(void)clReleaseKernel(cl_dataP->path_sort_records_kernel);
			cl_dataP->path_sort_records_kernel = 0;
		}
		if (cl_dataP->path_scatter_records_kernel) {
			(void)clReleaseKernel(cl_dataP->path_scatter_records_kernel);
			cl_dataP->path_scatter_records_kernel = 0;
		}
		if (cl_dataP->path_classify_count_kernel) {
			(void)clReleaseKernel(cl_dataP->path_classify_count_kernel);
			cl_dataP->path_classify_count_kernel = 0;
		}
		if (cl_dataP->apply_domain_kernel) {
			(void)clReleaseKernel(cl_dataP->apply_domain_kernel);
			cl_dataP->apply_domain_kernel = 0;
		}
		if (cl_dataP->mapped_sort_kernel) {
			(void)clReleaseKernel(cl_dataP->mapped_sort_kernel);
			cl_dataP->mapped_sort_kernel = 0;
		}
		if (cl_dataP->copy_kernel) {
			(void)clReleaseKernel(cl_dataP->copy_kernel);
			cl_dataP->copy_kernel = 0;
		}
		if (cl_dataP->domain_sort_kernel) {
			(void)clReleaseKernel(cl_dataP->domain_sort_kernel);
			cl_dataP->domain_sort_kernel = 0;
		}
		if (cl_dataP->sort_kernel) {
			(void)clReleaseKernel(cl_dataP->sort_kernel);
			cl_dataP->sort_kernel = 0;
		}
		if (cl_dataP->program) {
			(void)clReleaseProgram(cl_dataP->program);
			cl_dataP->program = 0;
		}
	}
}

} // namespace
#endif

#if defined(BPS_HAS_HLSL)
namespace {

struct DirectXGPUData {
	DXContextPtr context;
	ShaderObjectPtr sort_shader;
	ShaderObjectPtr domain_sort_shader;
	ShaderObjectPtr apply_domain_shader;
	ShaderObjectPtr copy_shader;
	ShaderObjectPtr mapped_sort_shader;
	ShaderObjectPtr path_classify_count_shader;
	ShaderObjectPtr path_scatter_records_shader;
	ShaderObjectPtr path_sort_records_shader;
	Microsoft::WRL::ComPtr<ID3D12Resource> path_records_resource;
	Microsoft::WRL::ComPtr<ID3D12Resource> path_line_offsets_resource;
	Microsoft::WRL::ComPtr<ID3D12Resource> path_work_offsets_resource;
	unsigned long long path_map_key = 0ull;
	int path_map_width = 0;
	int path_map_height = 0;
	int path_map_line_count = 0;
	int path_mapped_record_count = 0;
	int path_mapped_work_item_count = 0;
};

struct DirectXSortParams {
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

inline PF_Err DXErr(bool success)
{
	return success ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED;
}

#define BPS_DX_ERR(FUNC) ERR(DXErr(FUNC))

static void ReleaseDirectXData(DirectXGPUData *dx_dataP)
{
	if (dx_dataP) {
		dx_dataP->path_work_offsets_resource.Reset();
		dx_dataP->path_line_offsets_resource.Reset();
		dx_dataP->path_records_resource.Reset();
		dx_dataP->path_map_key = 0ull;
		dx_dataP->path_map_width = 0;
		dx_dataP->path_map_height = 0;
		dx_dataP->path_map_line_count = 0;
		dx_dataP->path_mapped_record_count = 0;
		dx_dataP->path_mapped_work_item_count = 0;
		dx_dataP->path_sort_records_shader.reset();
		dx_dataP->path_scatter_records_shader.reset();
		dx_dataP->path_classify_count_shader.reset();
		dx_dataP->apply_domain_shader.reset();
		dx_dataP->mapped_sort_shader.reset();
		dx_dataP->copy_shader.reset();
		dx_dataP->domain_sort_shader.reset();
		dx_dataP->sort_shader.reset();
		dx_dataP->context.reset();
	}
}

} // namespace
#endif

namespace {

std::mutex &BPS_GpuRenderMutex()
{
	static std::mutex mutex;
	return mutex;
}

std::mutex &BPS_GpuDeviceStateMutex()
{
	static std::mutex mutex;
	return mutex;
}

std::string &BPS_GpuFrameworkStorage()
{
	static std::string value;
	return value;
}

std::string &BPS_GpuDeviceNameStorage()
{
	static std::string value;
	return value;
}

std::atomic<bool> &BPS_GpuDeviceReadyStorage()
{
	static std::atomic<bool> value(false);
	return value;
}

std::atomic<bool> &BPS_LastRenderUsedGpuStorage()
{
	static std::atomic<bool> value(false);
	return value;
}

const char *BPS_FrameworkName(PF_GPU_Framework framework)
{
	switch (framework) {
	case PF_GPU_Framework_CUDA: return "CUDA";
	case PF_GPU_Framework_OPENCL: return "OpenCL";
	case PF_GPU_Framework_DIRECTX: return "DirectX";
	case PF_GPU_Framework_METAL: return "Metal";
	default: return "";
	}
}

static std::uint32_t BPS_NextPow2U32(std::uint32_t value)
{
	if (value <= 1u) {
		return 1u;
	}
	--value;
	value |= value >> 1;
	value |= value >> 2;
	value |= value >> 4;
	value |= value >> 8;
	value |= value >> 16;
	return value + 1u;
}

struct BPS_GpuPathMapLayout {
	std::vector<std::uint32_t> lineOffsets;
	std::vector<std::uint32_t> workOffsets;
	std::uint32_t mappedRecordCount = 0;
	std::uint32_t mappedWorkItemCount = 0;
};

static bool BPS_ComputeGpuPathMapLayout(
	const std::vector<std::uint32_t> &laneCounts,
	BPS_GpuPathMapLayout *layoutP)
{
	if (!layoutP) {
		return false;
	}
	const size_t line_count = laneCounts.size();
	layoutP->lineOffsets.assign(line_count + 1u, 0u);
	layoutP->workOffsets.assign(line_count + 1u, 0u);
	layoutP->mappedRecordCount = 0u;
	layoutP->mappedWorkItemCount = 0u;

	std::uint64_t record_running = 0u;
	std::uint64_t work_running = 0u;
	for (size_t line = 0u; line < line_count; ++line) {
		if (record_running > 0xffffffffull ||
			work_running > 0xffffffffull) {
			return false;
		}
		layoutP->lineOffsets[line] = static_cast<std::uint32_t>(record_running);
		layoutP->workOffsets[line] = static_cast<std::uint32_t>(work_running);
		const std::uint32_t len = laneCounts[line];
		record_running += len;
		work_running += static_cast<std::uint64_t>(len) +
			static_cast<std::uint64_t>(BPS_NextPow2U32(len));
	}
	if (record_running > 0xffffffffull ||
		work_running > 0xffffffffull) {
		return false;
	}
	layoutP->lineOffsets[line_count] = static_cast<std::uint32_t>(record_running);
	layoutP->workOffsets[line_count] = static_cast<std::uint32_t>(work_running);
	layoutP->mappedRecordCount = static_cast<std::uint32_t>(record_running);
	layoutP->mappedWorkItemCount = static_cast<std::uint32_t>(work_running);
	return true;
}

#if defined(BPS_HAS_CUDA)
std::string BPS_QueryCudaDeviceName(CUdevice device)
{
	char name[256] = {};
	// Driver API only: safe during GPU_DEVICE_SETUP (no runtime context required).
	if (cuDeviceGetName(name, static_cast<int>(sizeof(name)), device) != CUDA_SUCCESS) {
		return std::string();
	}
	return std::string(name);
}
#endif

#if defined(BPS_HAS_OPENCL)
std::string BPS_QueryOpenCLDeviceName(cl_device_id device)
{
	if (!device) {
		return std::string();
	}

	size_t name_size = 0;
	if (clGetDeviceInfo(device, CL_DEVICE_NAME, 0, 0, &name_size) != CL_SUCCESS ||
		name_size == 0) {
		return std::string();
	}

	std::string name(name_size, '\0');
	if (clGetDeviceInfo(device, CL_DEVICE_NAME, name_size, &name[0], 0) != CL_SUCCESS) {
		return std::string();
	}
	if (!name.empty() && name.back() == '\0') {
		name.pop_back();
	}
	return name;
}

PF_Err BPS_BuildOpenCLPathMap(
	OpenCLGPUData *cl_dataP,
	cl_context context,
	cl_command_queue queue,
	cl_mem path_mem,
	int width,
	int height,
	int lineCount,
	int pathDirection,
	int pathClosed,
	float pathLength,
	int pathSMin,
	int pathNMin,
	int pathSampleCount,
	unsigned long long pathMapKey)
{
	if (!cl_dataP || !context || !queue || !path_mem ||
		width <= 0 || height <= 0 || lineCount <= 0 ||
		pathSampleCount < 2) {
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}

	if (cl_dataP->path_map_key == pathMapKey &&
		cl_dataP->path_map_width == width &&
		cl_dataP->path_map_height == height &&
		cl_dataP->path_map_line_count == lineCount &&
		cl_dataP->path_records_mem &&
		cl_dataP->path_line_offsets_mem &&
		cl_dataP->path_work_offsets_mem) {
		return PF_Err_NONE;
	}

	ReleaseOpenCLPathMap(cl_dataP);

	const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
	if (pixelCount == 0u || pixelCount > 0xffffffffull) {
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}
	const size_t laneBytes = pixelCount * sizeof(cl_int);
	const size_t keyBytes = pixelCount * sizeof(cl_float);
	const size_t countBytes = static_cast<size_t>(lineCount) * sizeof(cl_uint);

	PF_Err err = PF_Err_NONE;
	cl_int cl_result = CL_SUCCESS;
	cl_mem lane_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, laneBytes, 0, &cl_result);
	if (cl_result != CL_SUCCESS) err = CL2Err(cl_result);
	cl_mem key_mem = 0;
	if (!err) {
		key_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, keyBytes, 0, &cl_result);
		if (cl_result != CL_SUCCESS) err = CL2Err(cl_result);
	}
	cl_mem counts_mem = 0;
	if (!err) {
		counts_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, countBytes, 0, &cl_result);
		if (cl_result != CL_SUCCESS) err = CL2Err(cl_result);
	}

	const cl_uint zero = 0u;
	if (!err) {
		err = CL2Err(clEnqueueFillBuffer(queue, counts_mem, &zero, sizeof(zero),
										 0, countBytes, 0, 0, 0));
	}

	const size_t local2[2] = {16u, 16u};
	const size_t global2[2] = {
		((static_cast<size_t>(width) + 15u) / 16u) * 16u,
		((static_cast<size_t>(height) + 15u) / 16u) * 16u
	};
	if (!err) {
		cl_uint arg = 0;
		err = CL2Err(clSetKernelArg(cl_dataP->path_classify_count_kernel, arg++, sizeof(cl_mem), &path_mem));
		if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_classify_count_kernel, arg++, sizeof(int), &pathSampleCount));
		if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_classify_count_kernel, arg++, sizeof(int), &width));
		if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_classify_count_kernel, arg++, sizeof(int), &height));
		if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_classify_count_kernel, arg++, sizeof(int), &lineCount));
		if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_classify_count_kernel, arg++, sizeof(int), &pathDirection));
		if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_classify_count_kernel, arg++, sizeof(int), &pathClosed));
		if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_classify_count_kernel, arg++, sizeof(float), &pathLength));
		if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_classify_count_kernel, arg++, sizeof(int), &pathSMin));
		if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_classify_count_kernel, arg++, sizeof(int), &pathNMin));
		if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_classify_count_kernel, arg++, sizeof(cl_mem), &lane_mem));
		if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_classify_count_kernel, arg++, sizeof(cl_mem), &key_mem));
		if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_classify_count_kernel, arg++, sizeof(cl_mem), &counts_mem));
		if (!err) {
			err = CL2Err(clEnqueueNDRangeKernel(queue, cl_dataP->path_classify_count_kernel,
												2, 0, global2, local2, 0, 0, 0));
		}
	}

	std::vector<std::uint32_t> laneCounts(static_cast<size_t>(lineCount), 0u);
	if (!err) {
		err = CL2Err(clEnqueueReadBuffer(queue, counts_mem, CL_TRUE, 0, countBytes,
										 laneCounts.data(), 0, 0, 0));
	}

	BPS_GpuPathMapLayout layout;
	if (!err && !BPS_ComputeGpuPathMapLayout(laneCounts, &layout)) {
		err = PF_Err_INTERNAL_STRUCT_DAMAGED;
	}

	cl_mem records_mem = 0;
	cl_mem line_offsets_mem = 0;
	cl_mem work_offsets_mem = 0;
	cl_mem cursor_mem = 0;
	cl_mem work_records_mem = 0;
	if (!err && layout.mappedRecordCount > 0u &&
		layout.mappedWorkItemCount > 0u) {
		const size_t recordsBytes =
			static_cast<size_t>(layout.mappedRecordCount) * sizeof(BpsMappedPixelRecord);
		const size_t offsetsBytes =
			(static_cast<size_t>(lineCount) + 1u) * sizeof(cl_uint);
		const size_t workRecordsBytes =
			static_cast<size_t>(layout.mappedWorkItemCount) * sizeof(BpsMappedPixelRecord);

		records_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, recordsBytes, 0, &cl_result);
		if (cl_result != CL_SUCCESS) err = CL2Err(cl_result);
		if (!err) {
			line_offsets_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, offsetsBytes, 0, &cl_result);
			if (cl_result != CL_SUCCESS) err = CL2Err(cl_result);
		}
		if (!err) {
			work_offsets_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, offsetsBytes, 0, &cl_result);
			if (cl_result != CL_SUCCESS) err = CL2Err(cl_result);
		}
		if (!err) {
			cursor_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, offsetsBytes, 0, &cl_result);
			if (cl_result != CL_SUCCESS) err = CL2Err(cl_result);
		}
		if (!err) {
			work_records_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, workRecordsBytes, 0, &cl_result);
			if (cl_result != CL_SUCCESS) err = CL2Err(cl_result);
		}
		if (!err) {
			err = CL2Err(clEnqueueWriteBuffer(queue, line_offsets_mem, CL_TRUE, 0,
											 offsetsBytes, layout.lineOffsets.data(), 0, 0, 0));
		}
		if (!err) {
			err = CL2Err(clEnqueueWriteBuffer(queue, work_offsets_mem, CL_TRUE, 0,
											 offsetsBytes, layout.workOffsets.data(), 0, 0, 0));
		}
		if (!err) {
			err = CL2Err(clEnqueueWriteBuffer(queue, cursor_mem, CL_TRUE, 0,
											 offsetsBytes, layout.lineOffsets.data(), 0, 0, 0));
		}
		if (!err) {
			cl_uint arg = 0;
			err = CL2Err(clSetKernelArg(cl_dataP->path_scatter_records_kernel, arg++, sizeof(cl_mem), &lane_mem));
			if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_scatter_records_kernel, arg++, sizeof(cl_mem), &key_mem));
			if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_scatter_records_kernel, arg++, sizeof(cl_mem), &cursor_mem));
			if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_scatter_records_kernel, arg++, sizeof(cl_mem), &records_mem));
			if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_scatter_records_kernel, arg++, sizeof(int), &width));
			if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_scatter_records_kernel, arg++, sizeof(int), &height));
			if (!err) {
				err = CL2Err(clEnqueueNDRangeKernel(queue, cl_dataP->path_scatter_records_kernel,
													2, 0, global2, local2, 0, 0, 0));
			}
		}
		if (!err) {
			cl_uint arg = 0;
			err = CL2Err(clSetKernelArg(cl_dataP->path_sort_records_kernel, arg++, sizeof(cl_mem), &records_mem));
			if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_sort_records_kernel, arg++, sizeof(cl_mem), &work_records_mem));
			if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_sort_records_kernel, arg++, sizeof(cl_mem), &line_offsets_mem));
			if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_sort_records_kernel, arg++, sizeof(cl_mem), &work_offsets_mem));
			if (!err) err = CL2Err(clSetKernelArg(cl_dataP->path_sort_records_kernel, arg++, sizeof(int), &lineCount));
			const size_t local = 256u;
			const size_t global = static_cast<size_t>(lineCount) * local;
			if (!err) {
				err = CL2Err(clEnqueueNDRangeKernel(queue, cl_dataP->path_sort_records_kernel,
													1, 0, &global, &local, 0, 0, 0));
			}
		}
	}

	if (work_records_mem) (void)clReleaseMemObject(work_records_mem);
	if (cursor_mem) (void)clReleaseMemObject(cursor_mem);
	if (counts_mem) (void)clReleaseMemObject(counts_mem);
	if (key_mem) (void)clReleaseMemObject(key_mem);
	if (lane_mem) (void)clReleaseMemObject(lane_mem);

	if (err) {
		if (work_offsets_mem) (void)clReleaseMemObject(work_offsets_mem);
		if (line_offsets_mem) (void)clReleaseMemObject(line_offsets_mem);
		if (records_mem) (void)clReleaseMemObject(records_mem);
		ReleaseOpenCLPathMap(cl_dataP);
		return err;
	}

	cl_dataP->path_records_mem = records_mem;
	cl_dataP->path_line_offsets_mem = line_offsets_mem;
	cl_dataP->path_work_offsets_mem = work_offsets_mem;
	cl_dataP->path_map_key = pathMapKey;
	cl_dataP->path_map_width = width;
	cl_dataP->path_map_height = height;
	cl_dataP->path_map_line_count = lineCount;
	cl_dataP->path_mapped_record_count = static_cast<int>(layout.mappedRecordCount);
	cl_dataP->path_mapped_work_item_count = static_cast<int>(layout.mappedWorkItemCount);
	return PF_Err_NONE;
}
#endif

#if defined(BPS_HAS_HLSL)
std::string BPS_WideToUtf8(const wchar_t *text)
{
	if (!text || !text[0]) {
		return std::string();
	}

	const int required = WideCharToMultiByte(CP_UTF8, 0, text, -1, 0, 0, 0, 0);
	if (required <= 1) {
		return std::string();
	}

	std::string utf8(static_cast<size_t>(required - 1), '\0');
	WideCharToMultiByte(CP_UTF8, 0, text, -1, &utf8[0], required, 0, 0);
	return utf8;
}

std::string BPS_QueryDirectXDeviceName(ID3D12Device *device)
{
	if (!device) {
		return std::string();
	}

	Microsoft::WRL::ComPtr<IDXGIFactory6> factory;
	if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
		return std::string();
	}

	Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
	const LUID luid = device->GetAdapterLuid();
	if (FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)))) {
		return std::string();
	}

	DXGI_ADAPTER_DESC1 desc;
	if (FAILED(adapter->GetDesc1(&desc))) {
		return std::string();
	}
	return BPS_WideToUtf8(desc.Description);
}
#endif

#if defined(BPS_RENDER_DIAG)
static void
BPS_GpuDiagLog(const char *format, ...)
{
	char message[1024];
	va_list args;
	va_start(args, format);
	std::vsnprintf(message, sizeof(message), format, args);
	va_end(args);

#if defined(_WIN32)
	OutputDebugStringA("[BitonicPixelSorter] ");
	OutputDebugStringA(message);
	OutputDebugStringA("\n");
#else
	FILE *file = std::fopen("/tmp/BitonicPixelSorter_render_diag.log", "a");
	if (file) {
		std::fprintf(file, "[BitonicPixelSorter] %s\n", message);
		std::fclose(file);
	}
#endif
}
#endif

} // namespace
// BPS_RecordGpuDevice and BPS_ClearGpuDevice are defined at file scope (not
// inside the anonymous namespace) so that BPS_MetalBackend.mm can link to
// them.  They are declared in BPS_MetalBackend.h under BPS_HAS_METAL.

void BPS_RecordGpuDevice(PF_GPU_Framework framework, const std::string &device_name)
{
	std::lock_guard<std::mutex> lock(BPS_GpuDeviceStateMutex());
	BPS_GpuFrameworkStorage() = BPS_FrameworkName(framework);
	BPS_GpuDeviceNameStorage() = device_name;
	BPS_GpuDeviceReadyStorage().store(framework != PF_GPU_Framework_NONE, std::memory_order_release);
	BPS_LastRenderUsedGpuStorage().store(false, std::memory_order_release);
}

void BPS_ClearGpuDevice()
{
	std::lock_guard<std::mutex> lock(BPS_GpuDeviceStateMutex());
	BPS_GpuFrameworkStorage().clear();
	BPS_GpuDeviceNameStorage().clear();
	BPS_GpuDeviceReadyStorage().store(false, std::memory_order_release);
	BPS_LastRenderUsedGpuStorage().store(false, std::memory_order_release);
}

const char *BPS_ActiveGpuFrameworkName()
{
	thread_local std::string snapshot;
	std::lock_guard<std::mutex> lock(BPS_GpuDeviceStateMutex());
	snapshot = BPS_GpuFrameworkStorage();
	return snapshot.c_str();
}

const char *BPS_ActiveGpuDeviceName()
{
	thread_local std::string snapshot;
	std::lock_guard<std::mutex> lock(BPS_GpuDeviceStateMutex());
	snapshot = BPS_GpuDeviceNameStorage();
	return snapshot.c_str();
}

bool BPS_IsGpuDeviceReady()
{
	return BPS_GpuDeviceReadyStorage().load(std::memory_order_acquire);
}

bool BPS_LastRenderUsedGpu()
{
	return BPS_LastRenderUsedGpuStorage().load(std::memory_order_acquire);
}

void BPS_SetLastRenderUsedGpu(bool used_gpu)
{
	BPS_LastRenderUsedGpuStorage().store(used_gpu, std::memory_order_release);
}

//-----------------------------------------------------------------------------
PF_Err BPS_GPUDeviceSetup(
	PF_InData				*in_dataP,
	PF_OutData				*out_dataP,
	PF_GPUDeviceSetupExtra	*extraP)
{
	PF_Err err = PF_Err_NONE;

	if (!extraP || !extraP->input) {
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	// SDK opt-out: do not confirm GPU for this device/framework. The host falls
	// back to CPU rendering without changing GlobalSetup/PiPL outflags.
	if (!BPS_ShouldAcceptGpuDeviceSetup(in_dataP, extraP->input->what_gpu)) {
		return PF_Err_NONE;
	}

#if defined(BPS_HAS_CUDA)
	if (extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
		AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite =
			AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP, kPFGPUDeviceSuite,
												 kPFGPUDeviceSuiteVersion1, out_dataP);

		PF_GPUDeviceInfo device_info;
		AEFX_CLR_STRUCT(device_info);
		ERR(gpu_suite->GetDeviceInfo(in_dataP->effect_ref,
									  extraP->input->device_index,
									  &device_info));

		// CUDA kernels are statically linked; nothing to compile here. Do not call
		// the CUDA runtime during device setup — the SDK sample does not, and
		// cudaGetDevice on a host-owned context has been observed to crash Release
		// builds on older After Effects versions. Device name uses the Driver API.
		if (!err) {
			out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
			const CUdevice cu_device = static_cast<CUdevice>(
				reinterpret_cast<uintptr_t>(device_info.devicePV));
			BPS_RecordGpuDevice(device_info.device_framework,
								BPS_QueryCudaDeviceName(cu_device));
		}
	}
#endif

#if defined(BPS_HAS_OPENCL)
	if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL) {
		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite =
			AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite,
											 kPFHandleSuiteVersion1, out_dataP);

		AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite =
			AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP, kPFGPUDeviceSuite,
												 kPFGPUDeviceSuiteVersion1, out_dataP);

		PF_GPUDeviceInfo device_info;
		AEFX_CLR_STRUCT(device_info);
		ERR(gpu_suite->GetDeviceInfo(in_dataP->effect_ref,
									  extraP->input->device_index,
									  &device_info));

		PF_Handle gpu_dataH = 0;
		OpenCLGPUData *cl_dataP = 0;

		if (!err) {
			gpu_dataH = handle_suite->host_new_handle(sizeof(OpenCLGPUData));
			if (!gpu_dataH) {
				err = PF_Err_OUT_OF_MEMORY;
			} else {
				cl_dataP = reinterpret_cast<OpenCLGPUData *>(*gpu_dataH);
				std::memset(cl_dataP, 0, sizeof(OpenCLGPUData));
			}
		}

		cl_int result = CL_SUCCESS;
		cl_context context = reinterpret_cast<cl_context>(device_info.contextPV);
		cl_device_id device = reinterpret_cast<cl_device_id>(device_info.devicePV);

		if (!err && (!context || !device)) {
			err = PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

		if (!err) {
			const char *strings[] = { kBitonicPixelSorter_Kernel_OpenCLString };
			const size_t sizes[] = { std::strlen(kBitonicPixelSorter_Kernel_OpenCLString) };

			cl_dataP->program = clCreateProgramWithSource(context, 1, strings, sizes, &result);
			BPS_CL_ERR(result);
		}

		if (!err) {
			BPS_CL_ERR(clBuildProgram(cl_dataP->program, 1, &device,
									  "-cl-single-precision-constant -cl-fast-relaxed-math",
									  0, 0));
		}

		if (!err) {
			cl_dataP->sort_kernel = clCreateKernel(cl_dataP->program, "BitonicSortKernel", &result);
			BPS_CL_ERR(result);
		}

		if (!err) {
			cl_dataP->domain_sort_kernel =
				clCreateKernel(cl_dataP->program, "BitonicSortDomainKernel", &result);
			BPS_CL_ERR(result);
		}

		if (!err) {
			cl_dataP->apply_domain_kernel =
				clCreateKernel(cl_dataP->program, "BitonicApplyDomainKernel", &result);
			BPS_CL_ERR(result);
		}

		if (!err) {
			cl_dataP->copy_kernel =
				clCreateKernel(cl_dataP->program, "BitonicCopyInputKernel", &result);
			BPS_CL_ERR(result);
		}

		if (!err) {
			cl_dataP->mapped_sort_kernel =
				clCreateKernel(cl_dataP->program, "BitonicSortMappedKernel", &result);
			BPS_CL_ERR(result);
		}

		if (!err) {
			cl_dataP->path_classify_count_kernel =
				clCreateKernel(cl_dataP->program, "BitonicBuildPathClassifyCountKernel", &result);
			BPS_CL_ERR(result);
		}

		if (!err) {
			cl_dataP->path_scatter_records_kernel =
				clCreateKernel(cl_dataP->program, "BitonicBuildPathScatterRecordsKernel", &result);
			BPS_CL_ERR(result);
		}

		if (!err) {
			cl_dataP->path_sort_records_kernel =
				clCreateKernel(cl_dataP->program, "BitonicBuildPathSortRecordsKernel", &result);
			BPS_CL_ERR(result);
		}

		if (!err) {
			extraP->output->gpu_data = gpu_dataH;
			out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
			BPS_RecordGpuDevice(device_info.device_framework, BPS_QueryOpenCLDeviceName(device));
		} else {
			ReleaseOpenCLData(cl_dataP);
			if (gpu_dataH) {
				handle_suite->host_dispose_handle(gpu_dataH);
			}
		}
	}
#endif

#if defined(BPS_HAS_HLSL)
	if (!err && extraP->input->what_gpu == PF_GPU_Framework_DIRECTX) {
		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite =
			AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite,
											 kPFHandleSuiteVersion1, out_dataP);

		AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite =
			AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP, kPFGPUDeviceSuite,
												 kPFGPUDeviceSuiteVersion1, out_dataP);

		PF_GPUDeviceInfo device_info;
		AEFX_CLR_STRUCT(device_info);
		ERR(gpu_suite->GetDeviceInfo(in_dataP->effect_ref,
									  extraP->input->device_index,
									  &device_info));

		PF_Handle gpu_dataH = 0;
		DirectXGPUData *dx_dataP = 0;

		if (!err) {
			gpu_dataH = handle_suite->host_new_handle(sizeof(DirectXGPUData));
			if (!gpu_dataH) {
				err = PF_Err_OUT_OF_MEMORY;
			} else {
				dx_dataP = reinterpret_cast<DirectXGPUData *>(*gpu_dataH);
				new (dx_dataP) DirectXGPUData();
			}
		}

		if (!err) {
			dx_dataP->context = std::make_shared<DXContext>();
			dx_dataP->sort_shader = std::make_shared<ShaderObject>();
			dx_dataP->domain_sort_shader = std::make_shared<ShaderObject>();
			dx_dataP->apply_domain_shader = std::make_shared<ShaderObject>();
			dx_dataP->copy_shader = std::make_shared<ShaderObject>();
			dx_dataP->mapped_sort_shader = std::make_shared<ShaderObject>();
			dx_dataP->path_classify_count_shader = std::make_shared<ShaderObject>();
			dx_dataP->path_scatter_records_shader = std::make_shared<ShaderObject>();
			dx_dataP->path_sort_records_shader = std::make_shared<ShaderObject>();

			BPS_DX_ERR(dx_dataP->context->Initialize(
				reinterpret_cast<ID3D12Device *>(device_info.devicePV),
				reinterpret_cast<ID3D12CommandQueue *>(device_info.command_queuePV)));
		}

		if (!err) {
			BPS_DX_ERR(BPS_LoadEmbeddedDirectXSortShader(
				dx_dataP->context,
				dx_dataP->sort_shader));
		}

		if (!err) {
			BPS_DX_ERR(BPS_LoadEmbeddedDirectXDomainSortShader(
				dx_dataP->context,
				dx_dataP->domain_sort_shader));
		}

		if (!err) {
			BPS_DX_ERR(BPS_LoadEmbeddedDirectXApplyDomainShader(
				dx_dataP->context,
				dx_dataP->apply_domain_shader));
		}

		if (!err) {
			BPS_DX_ERR(BPS_LoadEmbeddedDirectXCopyShader(
				dx_dataP->context,
				dx_dataP->copy_shader));
		}

		if (!err) {
			BPS_DX_ERR(BPS_LoadEmbeddedDirectXMappedSortShader(
				dx_dataP->context,
				dx_dataP->mapped_sort_shader));
		}

		if (!err) {
			BPS_DX_ERR(BPS_LoadEmbeddedDirectXPathClassifyCountShader(
				dx_dataP->context,
				dx_dataP->path_classify_count_shader));
		}

		if (!err) {
			BPS_DX_ERR(BPS_LoadEmbeddedDirectXPathScatterRecordsShader(
				dx_dataP->context,
				dx_dataP->path_scatter_records_shader));
		}

		if (!err) {
			BPS_DX_ERR(BPS_LoadEmbeddedDirectXPathSortRecordsShader(
				dx_dataP->context,
				dx_dataP->path_sort_records_shader));
		}

		if (!err) {
			extraP->output->gpu_data = gpu_dataH;
			out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
			BPS_RecordGpuDevice(
				device_info.device_framework,
				BPS_QueryDirectXDeviceName(reinterpret_cast<ID3D12Device *>(device_info.devicePV)));
		} else {
			if (dx_dataP) {
				ReleaseDirectXData(dx_dataP);
				dx_dataP->~DirectXGPUData();
			}
			if (gpu_dataH) {
				handle_suite->host_dispose_handle(gpu_dataH);
			}
		}
	}
#endif

#if defined(BPS_HAS_METAL)
	if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL) {
		err = BPS_MetalDeviceSetup(in_dataP, out_dataP, extraP);
	}
#endif

#if !defined(BPS_HAS_CUDA) && !defined(BPS_HAS_OPENCL) && !defined(BPS_HAS_HLSL) && !defined(BPS_HAS_METAL)
	(void)in_dataP; (void)out_dataP; (void)extraP;
#endif

	return err;
}

//-----------------------------------------------------------------------------
PF_Err BPS_GPUDeviceSetdown(
	PF_InData					*in_dataP,
	PF_OutData					*out_dataP,
	PF_GPUDeviceSetdownExtra	*extraP)
{
	PF_Err err = PF_Err_NONE;

#if defined(BPS_HAS_OPENCL)
	if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL && extraP->input->gpu_data) {
		PF_Handle gpu_dataH =
			reinterpret_cast<PF_Handle>(const_cast<void *>(extraP->input->gpu_data));
		OpenCLGPUData *cl_dataP = reinterpret_cast<OpenCLGPUData *>(*gpu_dataH);
		ReleaseOpenCLData(cl_dataP);
		BPS_ClearGpuDevice();

		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite =
			AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite,
											 kPFHandleSuiteVersion1, out_dataP);
		handle_suite->host_dispose_handle(gpu_dataH);
	}
#endif

#if defined(BPS_HAS_HLSL)
	if (extraP->input->what_gpu == PF_GPU_Framework_DIRECTX && extraP->input->gpu_data) {
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		DirectXGPUData *dx_dataP = reinterpret_cast<DirectXGPUData *>(*gpu_dataH);
		ReleaseDirectXData(dx_dataP);
		dx_dataP->~DirectXGPUData();
		BPS_ClearGpuDevice();

		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite =
			AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite,
											 kPFHandleSuiteVersion1, out_dataP);
		handle_suite->host_dispose_handle(gpu_dataH);
	}
#endif

	// CUDA: nothing allocated at setup, so nothing to release.
	if (extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
#if defined(BPS_HAS_CUDA)
		BitonicClearCudaPathMapCache();
#endif
		BPS_ClearGpuDevice();
	}

#if defined(BPS_HAS_METAL)
	if (extraP->input->what_gpu == PF_GPU_Framework_METAL && extraP->input->gpu_data) {
		err = BPS_MetalDeviceSetdown(in_dataP, out_dataP, extraP);
	}
#endif

	(void)in_dataP; (void)out_dataP; (void)extraP;
	return err;
}

//-----------------------------------------------------------------------------
PF_Err BPS_SmartRenderGPU(
	PF_InData					*in_data,
	PF_OutData					*out_data,
	PF_PixelFormat				pixel_format,
	PF_EffectWorld				*input_worldP,
	PF_EffectWorld				*output_worldP,
	PF_EffectWorld				*criterion_worldP,
	PF_EffectWorld				*trigger_worldP,
	PF_SmartRenderExtra			*extraP,
	const BitonicSorterParams	*paramsP)
{
	PF_Err err = PF_Err_NONE;

	if (pixel_format != PF_PixelFormat_GPU_BGRA128) {
		return PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	if (!criterion_worldP) {
		criterion_worldP = input_worldP;
	}
	if (!trigger_worldP) {
		trigger_worldP = input_worldP;
	}

	const int mode = static_cast<int>(paramsP->mode);
	const int direction = (paramsP->direction == BPS_DIR_HORIZONTAL) ? 1 : 0;
	const int lineCount = (paramsP->mode == BPS_MODE_AXIS)
		? (direction ? output_worldP->height : output_worldP->width)
		: static_cast<int>(paramsP->domainLineCount);
	if (lineCount <= 0 && paramsP->mode != BPS_MODE_PATH) {
		return PF_Err_NONE;
	}

	AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite =
		AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_data, kPFGPUDeviceSuite,
											 kPFGPUDeviceSuiteVersion1, out_data);

	PF_GPUDeviceInfo device_info;
	AEFX_CLR_STRUCT(device_info);
	ERR(gpu_suite->GetDeviceInfo(in_data->effect_ref, extraP->input->device_index, &device_info));

	void *src_mem = 0;
	ERR(gpu_suite->GetGPUWorldData(in_data->effect_ref, input_worldP, &src_mem));
	void *criterion_mem = 0;
	ERR(gpu_suite->GetGPUWorldData(in_data->effect_ref, criterion_worldP, &criterion_mem));
	void *trigger_mem = 0;
	ERR(gpu_suite->GetGPUWorldData(in_data->effect_ref, trigger_worldP, &trigger_mem));
	void *dst_mem = 0;
	ERR(gpu_suite->GetGPUWorldData(in_data->effect_ref, output_worldP, &dst_mem));

	if (!criterion_mem) {
		criterion_mem = src_mem;
	}
	if (!trigger_mem) {
		trigger_mem = src_mem;
	}

	if (!err && (!src_mem || !dst_mem)) {
#if defined(BPS_RENDER_DIAG)
		BPS_GpuDiagLog(
			"SmartRenderGPU missing GPU world data framework=%s src=%p dst=%p "
			"frame=%ldx%ld output_origin=(%ld,%ld) output_size=%ldx%ld",
			BPS_FrameworkName(extraP->input->what_gpu),
			src_mem,
			dst_mem,
			static_cast<long>(in_data->width),
			static_cast<long>(in_data->height),
			static_cast<long>(output_worldP->origin_x),
			static_cast<long>(output_worldP->origin_y),
			static_cast<long>(output_worldP->width),
			static_cast<long>(output_worldP->height));
#endif
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}

	const int bytes_per_pixel = 16; // float4 BGRA
	// Frame size drives path/line geometry; input world size bounds buffer reads.
	// Smart Render often supplies a partial GPU world when alpha is present —
	// using frame size for buffer bounds causes OOB reads and corruption.
	const int width    = static_cast<int>(BPS_RenderWidth(in_data));
	const int height   = static_cast<int>(BPS_RenderHeight(in_data));
	const int srcPitch = input_worldP->rowbytes  / bytes_per_pixel;
	const int dstPitch = output_worldP->rowbytes / bytes_per_pixel;
	const int inputOriginX = input_worldP->origin_x;
	const int inputOriginY = input_worldP->origin_y;
	const int inputWidth = input_worldP->width;
	const int inputHeight = input_worldP->height;
	const int criterionPitch = criterion_worldP->rowbytes / bytes_per_pixel;
	const int criterionOriginX = criterion_worldP->origin_x;
	const int criterionOriginY = criterion_worldP->origin_y;
	const int criterionWidth = criterion_worldP->width;
	const int criterionHeight = criterion_worldP->height;
	const int triggerPitch = trigger_worldP->rowbytes / bytes_per_pixel;
	const int triggerOriginX = trigger_worldP->origin_x;
	const int triggerOriginY = trigger_worldP->origin_y;
	const int triggerWidth = trigger_worldP->width;
	const int triggerHeight = trigger_worldP->height;
	const int outputOriginX = output_worldP->origin_x;
	const int outputOriginY = output_worldP->origin_y;
	const int outputWidth = output_worldP->width;
	const int outputHeight = output_worldP->height;
	const int ordering  = paramsP->ascending ? 1 : 0;
	const int criterion = static_cast<int>(paramsP->criterion);
	const int trigger = static_cast<int>(paramsP->trigger);
	const int affect = static_cast<int>(paramsP->affect);
	const float cycleDegrees = paramsP->cycleDegrees;
	const int freePMin = static_cast<int>(paramsP->freePMin);
	const int freeQMin = static_cast<int>(paramsP->freeQMin);
	const int freeLineLength = static_cast<int>(paramsP->freeLineLength);
	// Path mode stores line length in domainMaxLineLength only.
	const int radialLength = (paramsP->mode == BPS_MODE_PATH)
		? static_cast<int>(paramsP->domainMaxLineLength)
		: static_cast<int>(paramsP->radialLength);
	const int domainStrideRaw = static_cast<int>(paramsP->domainMaxLineLength);
	const float swirlK = paramsP->swirlK;
	const int swirlLineMin = static_cast<int>(paramsP->swirlLineMin);
	const int pathDirection = static_cast<int>(paramsP->pathDirection);
	const int pathClosed = static_cast<int>(paramsP->pathClosed);
	const float pathLength = paramsP->pathLength;
	const int pathSMin = static_cast<int>(paramsP->pathSMin);
	const int pathNMin = static_cast<int>(paramsP->pathNMin);
	const int pathSampleCount = static_cast<int>(paramsP->pathSampleCount);
	const void *pathSamplesHost = paramsP->pathSamples;
	int mappedRecordCount = static_cast<int>(paramsP->mappedRecordCount);
	int mappedWorkItemCount = static_cast<int>(paramsP->mappedWorkItemCount);
	const void *mappedRecordsHost = paramsP->mappedRecords;
	const void *mappedLineOffsetsHost = paramsP->mappedLineOffsets;
	const void *mappedWorkOffsetsHost = paramsP->mappedWorkOffsets;
	std::shared_ptr<const BpsPathMap> fallbackPathMap;
	const bool needsPathMap =
		mode == BPS_MODE_PATH &&
		(mappedRecordCount <= 0 || mappedWorkItemCount <= 0 ||
		 !mappedRecordsHost || !mappedLineOffsetsHost || !mappedWorkOffsetsHost);
#if defined(BPS_HAS_CUDA)
	const bool cudaCanBuildPathMap =
		needsPathMap &&
		extraP->input->what_gpu == PF_GPU_Framework_CUDA &&
		pathSampleCount >= 2 &&
		pathSamplesHost;
#else
	const bool cudaCanBuildPathMap = false;
#endif
#if defined(BPS_HAS_OPENCL)
	const bool openclCanBuildPathMap =
		needsPathMap &&
		extraP->input->what_gpu == PF_GPU_Framework_OPENCL &&
		pathSampleCount >= 2 &&
		pathSamplesHost;
#else
	const bool openclCanBuildPathMap = false;
#endif
#if defined(BPS_HAS_HLSL)
	const bool directxCanBuildPathMap =
		needsPathMap &&
		extraP->input->what_gpu == PF_GPU_Framework_DIRECTX &&
		pathSampleCount >= 2 &&
		pathSamplesHost;
#else
	const bool directxCanBuildPathMap = false;
#endif
#if defined(BPS_HAS_METAL)
	const bool metalCanBuildPathMap =
		needsPathMap &&
		extraP->input->what_gpu == PF_GPU_Framework_METAL &&
		pathSampleCount >= 2 &&
		pathSamplesHost;
#else
	const bool metalCanBuildPathMap = false;
#endif
	const bool backendCanBuildPathMap =
		cudaCanBuildPathMap || openclCanBuildPathMap ||
		directxCanBuildPathMap || metalCanBuildPathMap;
	if (needsPathMap && !backendCanBuildPathMap) {
		fallbackPathMap = BPS_AcquirePathMap(width, height, *paramsP);
		if (fallbackPathMap) {
			mappedRecordCount = static_cast<int>(fallbackPathMap->mappedRecordCount);
			mappedWorkItemCount = static_cast<int>(fallbackPathMap->mappedWorkItemCount);
			mappedRecordsHost = fallbackPathMap->records.empty()
				? nullptr : fallbackPathMap->records.data();
			mappedLineOffsetsHost = fallbackPathMap->lineOffsets.empty()
				? nullptr : fallbackPathMap->lineOffsets.data();
			mappedWorkOffsetsHost = fallbackPathMap->workOffsets.empty()
				? nullptr : fallbackPathMap->workOffsets.data();
		}
	}
	const unsigned long long pathMapKey =
		(mode == BPS_MODE_PATH)
			? static_cast<unsigned long long>(BPS_PathMapKey(width, height, *paramsP))
			: 0ull;
	// Round up to the next power of two, then double it.  Doubling guarantees
	// that in-place bitonic padding for any span [spanStart, spanStart+sortSize)
	// never overflows the line's allocated region, even when the span starts
	// past the halfway point (e.g. when the leading arc of a Rotation circle
	// is entirely outside the frame and spanStart is large).
	int domainStride = domainStrideRaw;
	if (domainStride > 1) {
		unsigned int v = static_cast<unsigned int>(domainStride) - 1u;
		v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
		domainStride = static_cast<int>(v + 1u);
	}
	domainStride *= 2; // 2x ensures spanStart + sortSize <= domainStride always

	if (err) {
		return err;
	}

#if defined(BPS_HAS_OPENCL)
	if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL) {
		// OpenCL kernel arguments are stored on the shared cl_kernel object, so
		// keep argument setup and enqueue serialised under MFR.
		std::lock_guard<std::mutex> gpu_render_lock(BPS_GpuRenderMutex());

		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		if (!gpu_dataH) {
			return PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

		OpenCLGPUData *cl_dataP = reinterpret_cast<OpenCLGPUData *>(*gpu_dataH);
		cl_mem cl_src_mem = reinterpret_cast<cl_mem>(src_mem);
		cl_mem cl_criterion_mem = reinterpret_cast<cl_mem>(criterion_mem);
		cl_mem cl_trigger_mem = reinterpret_cast<cl_mem>(trigger_mem);
		cl_mem cl_dst_mem = reinterpret_cast<cl_mem>(dst_mem);

		auto set_key_source_args = [&](cl_kernel kernel, cl_uint &param_index) {
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(cl_mem), &cl_criterion_mem));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(cl_mem), &cl_trigger_mem));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &criterionPitch));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &criterionOriginX));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &criterionOriginY));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &criterionWidth));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &criterionHeight));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &triggerPitch));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &triggerOriginX));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &triggerOriginY));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &triggerWidth));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &triggerHeight));
		};
		cl_command_queue queue =
			reinterpret_cast<cl_command_queue>(device_info.command_queuePV);
		cl_context context = reinterpret_cast<cl_context>(device_info.contextPV);

		// Path samples buffer (dummy 1-sample when unused so the arg is valid).
		const size_t path_count =
			pathSampleCount > 0 ? static_cast<size_t>(pathSampleCount) : 1u;
		const size_t path_bytes = path_count * sizeof(BpsPathSample);
		cl_int path_cl_result = CL_SUCCESS;
		cl_mem path_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, path_bytes,
										 0, &path_cl_result);
		BPS_CL_ERR(path_cl_result);
		if (!err) {
			std::vector<BpsPathSample> path_upload(path_count);
			if (pathSampleCount > 0 && pathSamplesHost) {
				std::memcpy(path_upload.data(), pathSamplesHost,
							static_cast<size_t>(pathSampleCount) * sizeof(BpsPathSample));
			}
			BPS_CL_ERR(clEnqueueWriteBuffer(queue, path_mem, CL_TRUE, 0, path_bytes,
											path_upload.data(), 0, 0, 0));
		}
		if (err) {
			if (path_mem) {
				(void)clReleaseMemObject(path_mem);
			}
			return err;
		}

		auto set_path_tail = [&](cl_kernel kernel, cl_uint &param_index) {
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(float), &swirlK));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &swirlLineMin));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &pathDirection));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &pathSMin));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &pathNMin));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(int), &pathSampleCount));
			BPS_CL_ERR(clSetKernelArg(kernel, param_index++, sizeof(cl_mem), &path_mem));
		};

		if (mode == BPS_MODE_AXIS) {
			cl_uint param_index = 0;
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
			set_key_source_args(cl_dataP->sort_kernel, param_index);
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &srcPitch));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &dstPitch));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &width));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &height));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &inputOriginX));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &inputOriginY));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &inputWidth));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &inputHeight));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &outputOriginX));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &outputOriginY));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &outputWidth));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &outputHeight));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &mode));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &direction));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &ordering));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &criterion));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &trigger));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &affect));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(float), &cycleDegrees));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &lineCount));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &freePMin));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &freeQMin));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &freeLineLength));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &radialLength));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(float), &paramsP->thresholdMin));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(float), &paramsP->thresholdMax));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(float), &paramsP->angleCos));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(float), &paramsP->angleSin));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(float), &paramsP->centerX));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(float), &paramsP->centerY));
			set_path_tail(cl_dataP->sort_kernel, param_index);

			const size_t local = 256;
			const size_t global = static_cast<size_t>(lineCount) * local;
			BPS_CL_ERR(clEnqueueNDRangeKernel(queue, cl_dataP->sort_kernel, 1, 0,
											  &global, &local, 0, 0, 0));
			(void)clReleaseMemObject(path_mem);
			return err;
		}

		if (mode == BPS_MODE_PATH) {
			cl_uint copy_index = 0;
			BPS_CL_ERR(clSetKernelArg(cl_dataP->copy_kernel, copy_index++, sizeof(cl_mem), &cl_src_mem));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->copy_kernel, copy_index++, sizeof(cl_mem), &cl_dst_mem));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->copy_kernel, copy_index++, sizeof(int), &srcPitch));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->copy_kernel, copy_index++, sizeof(int), &dstPitch));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->copy_kernel, copy_index++, sizeof(int), &width));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->copy_kernel, copy_index++, sizeof(int), &height));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->copy_kernel, copy_index++, sizeof(int), &inputOriginX));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->copy_kernel, copy_index++, sizeof(int), &inputOriginY));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->copy_kernel, copy_index++, sizeof(int), &inputWidth));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->copy_kernel, copy_index++, sizeof(int), &inputHeight));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->copy_kernel, copy_index++, sizeof(int), &outputOriginX));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->copy_kernel, copy_index++, sizeof(int), &outputOriginY));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->copy_kernel, copy_index++, sizeof(int), &outputWidth));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->copy_kernel, copy_index++, sizeof(int), &outputHeight));
			const size_t copy_local[2] = {16, 16};
			const size_t copy_global[2] = {
				((static_cast<size_t>(outputWidth) + 15u) / 16u) * 16u,
				((static_cast<size_t>(outputHeight) + 15u) / 16u) * 16u
			};
			BPS_CL_ERR(clEnqueueNDRangeKernel(queue, cl_dataP->copy_kernel, 2, 0,
											  copy_global, copy_local, 0, 0, 0));

			bool using_gpu_path_map = false;
			cl_mem gpu_records_mem = 0;
			cl_mem gpu_line_offsets_mem = 0;
			cl_mem gpu_work_offsets_mem = 0;
			if (!err && openclCanBuildPathMap) {
				PF_Err build_err = BPS_BuildOpenCLPathMap(
					cl_dataP, context, queue, path_mem,
					width, height, lineCount, pathDirection, pathClosed,
					pathLength, pathSMin, pathNMin, pathSampleCount,
					pathMapKey);
				if (build_err == PF_Err_NONE &&
					cl_dataP->path_mapped_record_count > 0 &&
					cl_dataP->path_mapped_work_item_count > 0 &&
					cl_dataP->path_records_mem &&
					cl_dataP->path_line_offsets_mem &&
					cl_dataP->path_work_offsets_mem) {
					using_gpu_path_map = true;
					mappedRecordCount = cl_dataP->path_mapped_record_count;
					mappedWorkItemCount = cl_dataP->path_mapped_work_item_count;
					gpu_records_mem = cl_dataP->path_records_mem;
					gpu_line_offsets_mem = cl_dataP->path_line_offsets_mem;
					gpu_work_offsets_mem = cl_dataP->path_work_offsets_mem;
				} else if (build_err != PF_Err_NONE) {
					fallbackPathMap = BPS_AcquirePathMap(width, height, *paramsP);
					if (fallbackPathMap) {
						mappedRecordCount = static_cast<int>(fallbackPathMap->mappedRecordCount);
						mappedWorkItemCount = static_cast<int>(fallbackPathMap->mappedWorkItemCount);
						mappedRecordsHost = fallbackPathMap->records.empty()
							? nullptr : fallbackPathMap->records.data();
						mappedLineOffsetsHost = fallbackPathMap->lineOffsets.empty()
							? nullptr : fallbackPathMap->lineOffsets.data();
						mappedWorkOffsetsHost = fallbackPathMap->workOffsets.empty()
							? nullptr : fallbackPathMap->workOffsets.data();
					}
				}
			}

			if (err || mappedRecordCount <= 0 || mappedWorkItemCount <= 0 ||
				(!using_gpu_path_map &&
				 (!mappedRecordsHost || !mappedLineOffsetsHost || !mappedWorkOffsetsHost))) {
				(void)clReleaseMemObject(path_mem);
				return err;
			}

			cl_int cl_result = CL_SUCCESS;
			const size_t recordsBytes =
				static_cast<size_t>(mappedRecordCount) * sizeof(BpsMappedPixelRecord);
			const size_t offsetsBytes =
				static_cast<size_t>(lineCount + 1) * sizeof(cl_uint);
			const size_t workBytes =
				static_cast<size_t>(mappedWorkItemCount) * sizeof(cl_uint);
			cl_mem records_mem = 0;
			if (using_gpu_path_map) {
				records_mem = gpu_records_mem;
			} else {
				records_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
											 recordsBytes, 0, &cl_result);
				BPS_CL_ERR(cl_result);
			}
			cl_mem line_offsets_mem = 0;
			if (using_gpu_path_map) {
				line_offsets_mem = gpu_line_offsets_mem;
			} else if (!err) {
				line_offsets_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
												  offsetsBytes, 0, &cl_result);
				BPS_CL_ERR(cl_result);
			}
			cl_mem work_offsets_mem = 0;
			if (using_gpu_path_map) {
				work_offsets_mem = gpu_work_offsets_mem;
			} else if (!err) {
				work_offsets_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
												  offsetsBytes, 0, &cl_result);
				BPS_CL_ERR(cl_result);
			}
			cl_mem domain_mem = 0;
			if (!err) {
				domain_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
											workBytes, 0, &cl_result);
				BPS_CL_ERR(cl_result);
			}
			cl_mem keys_mem = 0;
			if (!err) {
				keys_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
										  static_cast<size_t>(mappedWorkItemCount) * sizeof(cl_float),
										  0, &cl_result);
				BPS_CL_ERR(cl_result);
			}
			if (!err && !using_gpu_path_map) {
				BPS_CL_ERR(clEnqueueWriteBuffer(queue, records_mem, CL_TRUE, 0,
												recordsBytes, mappedRecordsHost, 0, 0, 0));
				BPS_CL_ERR(clEnqueueWriteBuffer(queue, line_offsets_mem, CL_TRUE, 0,
												offsetsBytes, mappedLineOffsetsHost, 0, 0, 0));
				BPS_CL_ERR(clEnqueueWriteBuffer(queue, work_offsets_mem, CL_TRUE, 0,
												offsetsBytes, mappedWorkOffsetsHost, 0, 0, 0));
			}
			if (!err) {
				cl_uint param_index = 0;
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
				set_key_source_args(cl_dataP->mapped_sort_kernel, param_index);
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(cl_mem), &domain_mem));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(cl_mem), &keys_mem));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(cl_mem), &records_mem));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(cl_mem), &line_offsets_mem));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(cl_mem), &work_offsets_mem));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &srcPitch));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &dstPitch));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &width));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &height));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &inputOriginX));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &inputOriginY));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &inputWidth));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &inputHeight));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &outputOriginX));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &outputOriginY));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &outputWidth));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &outputHeight));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &ordering));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &criterion));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &trigger));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &affect));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(float), &cycleDegrees));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(float), &paramsP->thresholdMin));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(float), &paramsP->thresholdMax));
				BPS_CL_ERR(clSetKernelArg(cl_dataP->mapped_sort_kernel, param_index++, sizeof(int), &lineCount));

				const size_t local = 256;
				const size_t global = static_cast<size_t>(lineCount) * local;
				BPS_CL_ERR(clEnqueueNDRangeKernel(queue, cl_dataP->mapped_sort_kernel, 1, 0,
												  &global, &local, 0, 0, 0));
			}

			if (keys_mem) (void)clReleaseMemObject(keys_mem);
			if (domain_mem) (void)clReleaseMemObject(domain_mem);
			if (!using_gpu_path_map && work_offsets_mem) (void)clReleaseMemObject(work_offsets_mem);
			if (!using_gpu_path_map && line_offsets_mem) (void)clReleaseMemObject(line_offsets_mem);
			if (!using_gpu_path_map && records_mem) (void)clReleaseMemObject(records_mem);
			(void)clReleaseMemObject(path_mem);
			return err;
		}

		if (domainStride <= 0 || outputWidth <= 0 || outputHeight <= 0) {
			(void)clReleaseMemObject(path_mem);
			return PF_Err_NONE;
		}
		const size_t domainCount =
			static_cast<size_t>(lineCount) * static_cast<size_t>(domainStride);
		const size_t domainBytes = domainCount * sizeof(cl_uint);
		const size_t keysBytes = domainCount * sizeof(cl_float);
		cl_int cl_result = CL_SUCCESS;
		cl_mem domain_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, domainBytes, 0, &cl_result);
		BPS_CL_ERR(cl_result);
		cl_mem keys_mem = 0;
		if (!err) {
			keys_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, keysBytes, 0, &cl_result);
			BPS_CL_ERR(cl_result);
		}
		if (err) {
			if (domain_mem) {
				(void)clReleaseMemObject(domain_mem);
			}
			return err;
		}

		const cl_uint invalid_index = 0xffffffffu;
		BPS_CL_ERR(clEnqueueFillBuffer(queue, domain_mem, &invalid_index, sizeof(invalid_index),
									   0, domainBytes, 0, 0, 0));

		if (!err) {
			cl_uint param_index = 0;
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
			set_key_source_args(cl_dataP->domain_sort_kernel, param_index);
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(cl_mem), &domain_mem));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(cl_mem), &keys_mem));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &srcPitch));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &width));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &height));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &inputOriginX));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &inputOriginY));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &inputWidth));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &inputHeight));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &mode));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &ordering));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &criterion));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &trigger));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &affect));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(float), &cycleDegrees));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &lineCount));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &freePMin));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &freeQMin));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &freeLineLength));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &radialLength));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &domainStride));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(float), &paramsP->thresholdMin));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(float), &paramsP->thresholdMax));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(float), &paramsP->angleCos));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(float), &paramsP->angleSin));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(float), &paramsP->centerX));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(float), &paramsP->centerY));
			set_path_tail(cl_dataP->domain_sort_kernel, param_index);

			const size_t local = 256;
			const size_t global = static_cast<size_t>(lineCount) * local;
			BPS_CL_ERR(clEnqueueNDRangeKernel(queue, cl_dataP->domain_sort_kernel, 1, 0,
											  &global, &local, 0, 0, 0));
		}

		if (!err) {
			cl_uint param_index = 0;
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(cl_mem), &domain_mem));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &srcPitch));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &dstPitch));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &width));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &height));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &inputOriginX));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &inputOriginY));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &inputWidth));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &inputHeight));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &outputOriginX));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &outputOriginY));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &outputWidth));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &outputHeight));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &mode));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &lineCount));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &freePMin));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &freeQMin));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &freeLineLength));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &radialLength));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(int), &domainStride));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(float), &paramsP->angleCos));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(float), &paramsP->angleSin));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(float), &paramsP->centerX));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->apply_domain_kernel, param_index++, sizeof(float), &paramsP->centerY));
			set_path_tail(cl_dataP->apply_domain_kernel, param_index);

			const size_t local[2] = {16, 16};
			const size_t global[2] = {
				((static_cast<size_t>(outputWidth) + 15u) / 16u) * 16u,
				((static_cast<size_t>(outputHeight) + 15u) / 16u) * 16u
			};
			BPS_CL_ERR(clEnqueueNDRangeKernel(queue, cl_dataP->apply_domain_kernel, 2, 0,
											  global, local, 0, 0, 0));
		}

		(void)clReleaseMemObject(keys_mem);
		(void)clReleaseMemObject(domain_mem);
		(void)clReleaseMemObject(path_mem);
		return err;
	}
#endif

#if defined(BPS_HAS_CUDA)
	if (extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
		// Full GPU plugins launch CUDA work on AE's MFR render threads. AE's CUDA
		// context is not reliably current on those threads on older hosts (AE 23/24),
		// so make it current explicitly with the Driver API before launching — the
		// runtime kernel launch then uses this context, where AE's GPU buffers are
		// valid. This mirrors the OpenCL path, which already targets AE's context and
		// command queue explicitly. Balanced push/pop is safe even when AE already
		// made the context current.
		CUcontext ae_cuda_ctx = reinterpret_cast<CUcontext>(device_info.contextPV);
		bool cuda_ctx_pushed = false;
		if (ae_cuda_ctx) {
			cuda_ctx_pushed = (cuCtxPushCurrent(ae_cuda_ctx) == CUDA_SUCCESS);
		}

		cudaError_t cuda_result =
			BitonicSort_CUDA(src_mem, criterion_mem, trigger_mem, dst_mem,
							 srcPitch, dstPitch, width, height,
							 inputOriginX, inputOriginY, inputWidth, inputHeight,
							 criterionPitch, criterionOriginX, criterionOriginY,
							 criterionWidth, criterionHeight,
							 triggerPitch, triggerOriginX, triggerOriginY,
							 triggerWidth, triggerHeight,
							 outputOriginX, outputOriginY,
							 outputWidth, outputHeight,
							 mode, direction, ordering, criterion, trigger, affect,
							 cycleDegrees, lineCount,
							 freePMin, freeQMin, freeLineLength, radialLength,
							 domainStride,
							 paramsP->thresholdMin, paramsP->thresholdMax,
							 paramsP->angleCos, paramsP->angleSin,
							 paramsP->centerX, paramsP->centerY,
							 swirlK, swirlLineMin,
							 pathDirection, pathClosed, pathLength,
							 pathSMin, pathNMin, pathSampleCount,
							 pathSamplesHost,
							 pathMapKey,
							 cudaCanBuildPathMap ? 1 : 0,
							 mappedRecordCount, mappedWorkItemCount,
							 mappedRecordsHost,
							 mappedLineOffsetsHost,
							 mappedWorkOffsetsHost);

		if (cuda_result != cudaSuccess && cudaCanBuildPathMap) {
			(void)cudaGetLastError();
			fallbackPathMap = BPS_AcquirePathMap(width, height, *paramsP);
			if (fallbackPathMap) {
				mappedRecordCount = static_cast<int>(fallbackPathMap->mappedRecordCount);
				mappedWorkItemCount = static_cast<int>(fallbackPathMap->mappedWorkItemCount);
				mappedRecordsHost = fallbackPathMap->records.empty()
					? nullptr : fallbackPathMap->records.data();
				mappedLineOffsetsHost = fallbackPathMap->lineOffsets.empty()
					? nullptr : fallbackPathMap->lineOffsets.data();
				mappedWorkOffsetsHost = fallbackPathMap->workOffsets.empty()
					? nullptr : fallbackPathMap->workOffsets.data();
				cuda_result =
					BitonicSort_CUDA(src_mem, criterion_mem, trigger_mem, dst_mem,
									 srcPitch, dstPitch, width, height,
									 inputOriginX, inputOriginY, inputWidth, inputHeight,
									 criterionPitch, criterionOriginX, criterionOriginY,
									 criterionWidth, criterionHeight,
									 triggerPitch, triggerOriginX, triggerOriginY,
									 triggerWidth, triggerHeight,
									 outputOriginX, outputOriginY,
									 outputWidth, outputHeight,
									 mode, direction, ordering, criterion, trigger, affect,
									 cycleDegrees, lineCount,
									 freePMin, freeQMin, freeLineLength, radialLength,
									 domainStride,
									 paramsP->thresholdMin, paramsP->thresholdMax,
									 paramsP->angleCos, paramsP->angleSin,
									 paramsP->centerX, paramsP->centerY,
									 swirlK, swirlLineMin,
									 pathDirection, pathClosed, pathLength,
									 pathSMin, pathNMin, pathSampleCount,
									 pathSamplesHost,
									 pathMapKey,
									 0,
									 mappedRecordCount, mappedWorkItemCount,
									 mappedRecordsHost,
									 mappedLineOffsetsHost,
									 mappedWorkOffsetsHost);
			}
		}

		if (cuda_result != cudaSuccess) {
			(void)cudaGetLastError();
			err = PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

#if defined(BPS_RENDER_DIAG)
		BPS_GpuDiagLog(
			"SmartRenderGPU CUDA result cuda_error=%d frame=%ldx%ld "
			"output_origin=(%ld,%ld) output_size=%ldx%ld",
			static_cast<int>(cuda_result),
			static_cast<long>(in_data->width),
			static_cast<long>(in_data->height),
			static_cast<long>(outputOriginX),
			static_cast<long>(outputOriginY),
			static_cast<long>(outputWidth),
			static_cast<long>(outputHeight));
#endif

		if (cuda_ctx_pushed) {
			CUcontext popped_ctx = nullptr;
			(void)cuCtxPopCurrent(&popped_ctx);
		}

		return err;
	}
#endif

#if defined(BPS_HAS_HLSL)
	if (extraP->input->what_gpu == PF_GPU_Framework_DIRECTX) {
		// DirectXUtils owns one command list/allocator per AE gpu_data handle;
		// serialise use of that mutable context while leaving CUDA concurrent.
		std::lock_guard<std::mutex> gpu_render_lock(BPS_GpuRenderMutex());

		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		if (!gpu_dataH) {
			return PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

		DirectXGPUData *dx_dataP = reinterpret_cast<DirectXGPUData *>(*gpu_dataH);
		DirectXSortParams dx_params = {
			srcPitch,
			dstPitch,
			width,
			height,
			inputOriginX,
			inputOriginY,
			inputWidth,
			inputHeight,
			criterionPitch,
			criterionOriginX,
			criterionOriginY,
			criterionWidth,
			criterionHeight,
			triggerPitch,
			triggerOriginX,
			triggerOriginY,
			triggerWidth,
			triggerHeight,
			outputOriginX,
			outputOriginY,
			outputWidth,
			outputHeight,
			mode,
			direction,
			ordering,
			criterion,
			trigger,
			affect,
			cycleDegrees,
			lineCount,
			freePMin,
			freeQMin,
			freeLineLength,
			radialLength,
			domainStride,
			paramsP->thresholdMin,
			paramsP->thresholdMax,
			paramsP->angleCos,
			paramsP->angleSin,
			paramsP->centerX,
			paramsP->centerY,
			swirlK,
			swirlLineMin,
			pathDirection,
			pathClosed,
			pathLength,
			pathSMin,
			pathNMin,
			pathSampleCount
		};

		const UINT src_bytes =
			static_cast<UINT>(input_worldP->height * input_worldP->rowbytes);
		const UINT criterion_bytes =
			static_cast<UINT>(criterion_worldP->height * criterion_worldP->rowbytes);
		const UINT trigger_bytes =
			static_cast<UINT>(trigger_worldP->height * trigger_worldP->rowbytes);
		const UINT dst_bytes =
			static_cast<UINT>(output_worldP->height * output_worldP->rowbytes);
		auto create_uploaded_buffer =
			[&](const void *bytesP,
				UINT byte_count,
				Microsoft::WRL::ComPtr<ID3D12Resource> &default_resource,
				Microsoft::WRL::ComPtr<ID3D12Resource> &upload_resource) -> PF_Err {
				if (!bytesP || byte_count == 0) {
					return PF_Err_BAD_CALLBACK_PARAM;
				}
				D3D12_HEAP_PROPERTIES default_heap = {
					D3D12_HEAP_TYPE_DEFAULT,
					D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
					D3D12_MEMORY_POOL_UNKNOWN,
					0, 0};
				D3D12_HEAP_PROPERTIES upload_heap = {
					D3D12_HEAP_TYPE_UPLOAD,
					D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
					D3D12_MEMORY_POOL_UNKNOWN,
					0, 0};
				D3D12_RESOURCE_DESC desc = {
					D3D12_RESOURCE_DIMENSION_BUFFER, 0,
					byte_count, 1, 1, 1,
					DXGI_FORMAT_UNKNOWN, 1, 0,
					D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
					D3D12_RESOURCE_FLAG_NONE};
				if (FAILED(dx_dataP->context->mDevice->CreateCommittedResource(
						&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
						D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
						IID_PPV_ARGS(default_resource.GetAddressOf())))) {
					return PF_Err_OUT_OF_MEMORY;
				}
				if (FAILED(dx_dataP->context->mDevice->CreateCommittedResource(
						&upload_heap, D3D12_HEAP_FLAG_NONE, &desc,
						D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
						IID_PPV_ARGS(upload_resource.GetAddressOf())))) {
					return PF_Err_OUT_OF_MEMORY;
				}
				void *mapped = nullptr;
				if (FAILED(upload_resource->Map(0, nullptr, &mapped))) {
					return PF_Err_OUT_OF_MEMORY;
				}
				std::memcpy(mapped, bytesP, byte_count);
				upload_resource->Unmap(0, nullptr);
				dx_dataP->context->mCommandList->CopyResource(
					default_resource.Get(), upload_resource.Get());
				return PF_Err_NONE;
			};

		// Path samples as a RAW byte-address buffer (t1). Always bind at least
		// one sample so the SRV is valid when Path mode is inactive.
		const UINT path_count =
			pathSampleCount > 0 ? static_cast<UINT>(pathSampleCount) : 1u;
		const UINT path_bytes =
			path_count * static_cast<UINT>(sizeof(BpsPathSample));
		Microsoft::WRL::ComPtr<ID3D12Resource> path_resource;
		Microsoft::WRL::ComPtr<ID3D12Resource> path_upload;
		{
			D3D12_HEAP_PROPERTIES default_heap = {
				D3D12_HEAP_TYPE_DEFAULT,
				D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
				D3D12_MEMORY_POOL_UNKNOWN,
				0, 0};
			D3D12_HEAP_PROPERTIES upload_heap = {
				D3D12_HEAP_TYPE_UPLOAD,
				D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
				D3D12_MEMORY_POOL_UNKNOWN,
				0, 0};
			D3D12_RESOURCE_DESC path_desc = {
				D3D12_RESOURCE_DIMENSION_BUFFER, 0,
				path_bytes, 1, 1, 1,
				DXGI_FORMAT_UNKNOWN, 1, 0,
				D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
				D3D12_RESOURCE_FLAG_NONE};
			if (FAILED(dx_dataP->context->mDevice->CreateCommittedResource(
					&default_heap, D3D12_HEAP_FLAG_NONE, &path_desc,
					D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
					IID_PPV_ARGS(path_resource.GetAddressOf())))) {
				return PF_Err_OUT_OF_MEMORY;
			}
			if (FAILED(dx_dataP->context->mDevice->CreateCommittedResource(
					&upload_heap, D3D12_HEAP_FLAG_NONE, &path_desc,
					D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
					IID_PPV_ARGS(path_upload.GetAddressOf())))) {
				return PF_Err_OUT_OF_MEMORY;
			}
			void *mapped = nullptr;
			if (FAILED(path_upload->Map(0, nullptr, &mapped))) {
				return PF_Err_OUT_OF_MEMORY;
			}
			std::memset(mapped, 0, path_bytes);
			if (pathSampleCount > 0 && pathSamplesHost) {
				std::memcpy(mapped, pathSamplesHost,
							static_cast<size_t>(pathSampleCount) * sizeof(BpsPathSample));
			}
			path_upload->Unmap(0, nullptr);
		}

		if (mode == BPS_MODE_AXIS) {
			DXShaderExecution shader_execution(
				dx_dataP->context,
				dx_dataP->sort_shader,
				6);

			dx_dataP->context->mCommandList->CopyResource(
				path_resource.Get(), path_upload.Get());
			D3D12_RESOURCE_BARRIER path_barrier = {};
			path_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			path_barrier.Transition.pResource = path_resource.Get();
			path_barrier.Transition.Subresource = 0;
			path_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
			path_barrier.Transition.StateAfter =
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
			dx_dataP->context->mCommandList->ResourceBarrier(1, &path_barrier);

			BPS_DX_ERR(shader_execution.SetParamBuffer(&dx_params, sizeof(dx_params)));
			BPS_DX_ERR(shader_execution.SetUnorderedAccessView(
				reinterpret_cast<ID3D12Resource *>(dst_mem),
				dst_bytes));
			BPS_DX_ERR(shader_execution.SetShaderResourceView(
				reinterpret_cast<ID3D12Resource *>(src_mem),
				src_bytes));
			BPS_DX_ERR(shader_execution.SetShaderResourceView(
				reinterpret_cast<ID3D12Resource *>(criterion_mem),
				criterion_bytes));
			BPS_DX_ERR(shader_execution.SetShaderResourceView(
				reinterpret_cast<ID3D12Resource *>(trigger_mem),
				trigger_bytes));
			BPS_DX_ERR(shader_execution.SetShaderResourceView(
				path_resource.Get(), path_bytes));
			BPS_DX_ERR(shader_execution.Execute(static_cast<UINT>(lineCount), 1));
			return err;
		}

		if (mode == BPS_MODE_PATH) {
			{
				DXShaderExecution copy_execution(
					dx_dataP->context,
					dx_dataP->copy_shader,
					3);
				BPS_DX_ERR(copy_execution.SetParamBuffer(&dx_params, sizeof(dx_params)));
				BPS_DX_ERR(copy_execution.SetUnorderedAccessView(
					reinterpret_cast<ID3D12Resource *>(dst_mem),
					dst_bytes));
				BPS_DX_ERR(copy_execution.SetShaderResourceView(
					reinterpret_cast<ID3D12Resource *>(src_mem),
					src_bytes));
				const UINT apply_x = static_cast<UINT>((outputWidth + 15) / 16);
				const UINT apply_y = static_cast<UINT>((outputHeight + 15) / 16);
				BPS_DX_ERR(copy_execution.Execute(apply_x, apply_y));
			}

			bool using_gpu_path_map = false;
			Microsoft::WRL::ComPtr<ID3D12Resource> gpu_records_resource;
			Microsoft::WRL::ComPtr<ID3D12Resource> gpu_line_offsets_resource;
			Microsoft::WRL::ComPtr<ID3D12Resource> gpu_work_offsets_resource;
			if (!err && directxCanBuildPathMap) {
				if (dx_dataP->path_map_key == pathMapKey &&
					dx_dataP->path_map_width == width &&
					dx_dataP->path_map_height == height &&
					dx_dataP->path_map_line_count == lineCount &&
					dx_dataP->path_records_resource &&
					dx_dataP->path_line_offsets_resource &&
					dx_dataP->path_work_offsets_resource) {
					using_gpu_path_map = true;
				} else {
					dx_dataP->path_records_resource.Reset();
					dx_dataP->path_line_offsets_resource.Reset();
					dx_dataP->path_work_offsets_resource.Reset();
					dx_dataP->path_map_key = 0ull;
					dx_dataP->path_mapped_record_count = 0;
					dx_dataP->path_mapped_work_item_count = 0;

					const UINT pixel_count =
						static_cast<UINT>(static_cast<size_t>(width) *
										  static_cast<size_t>(height));
					const UINT lane_bytes = pixel_count * sizeof(UINT);
					const UINT key_bytes = pixel_count * sizeof(float);
					const UINT counts_bytes =
						static_cast<UINT>(static_cast<size_t>(lineCount) *
										  sizeof(UINT));

					D3D12_HEAP_PROPERTIES default_heap = {
						D3D12_HEAP_TYPE_DEFAULT,
						D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
						D3D12_MEMORY_POOL_UNKNOWN,
						0, 0};
					auto create_default_buffer =
						[&](UINT byte_count,
							D3D12_RESOURCE_FLAGS flags,
							D3D12_RESOURCE_STATES initial_state,
							Microsoft::WRL::ComPtr<ID3D12Resource> &resource) -> PF_Err {
							D3D12_RESOURCE_DESC desc = {
								D3D12_RESOURCE_DIMENSION_BUFFER, 0,
								byte_count, 1, 1, 1,
								DXGI_FORMAT_UNKNOWN, 1, 0,
								D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
								flags};
							if (FAILED(dx_dataP->context->mDevice->CreateCommittedResource(
									&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
									initial_state, nullptr,
									IID_PPV_ARGS(resource.GetAddressOf())))) {
								return PF_Err_OUT_OF_MEMORY;
							}
							return PF_Err_NONE;
						};

					Microsoft::WRL::ComPtr<ID3D12Resource> lane_resource;
					Microsoft::WRL::ComPtr<ID3D12Resource> key_resource;
					Microsoft::WRL::ComPtr<ID3D12Resource> counts_resource;
					Microsoft::WRL::ComPtr<ID3D12Resource> counts_upload;
					ERR(create_default_buffer(lane_bytes,
											  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
											  D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
											  lane_resource));
					ERR(create_default_buffer(key_bytes,
											  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
											  D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
											  key_resource));
					std::vector<std::uint32_t> zero_counts(
						static_cast<size_t>(lineCount), 0u);
					ERR(create_uploaded_buffer(zero_counts.data(), counts_bytes,
												counts_resource, counts_upload));
					if (!err) {
						dx_dataP->context->mCommandList->CopyResource(
							path_resource.Get(), path_upload.Get());
						D3D12_RESOURCE_BARRIER barriers[2] = {};
						barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
						barriers[0].Transition.pResource = path_resource.Get();
						barriers[0].Transition.Subresource = 0;
						barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
						barriers[0].Transition.StateAfter =
							D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
						barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
						barriers[1].Transition.pResource = counts_resource.Get();
						barriers[1].Transition.Subresource = 0;
						barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
						barriers[1].Transition.StateAfter =
							D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
						dx_dataP->context->mCommandList->ResourceBarrier(2, barriers);

						DXShaderExecution classify_execution(
							dx_dataP->context,
							dx_dataP->path_classify_count_shader,
							8);
						BPS_DX_ERR(classify_execution.SetParamBuffer(
							&dx_params, sizeof(dx_params)));
						BPS_DX_ERR(classify_execution.SetUnorderedAccessView(
							lane_resource.Get(), lane_bytes));
						BPS_DX_ERR(classify_execution.SetUnorderedAccessView(
							key_resource.Get(), key_bytes));
						BPS_DX_ERR(classify_execution.SetUnorderedAccessView(
							counts_resource.Get(), counts_bytes));
						BPS_DX_ERR(classify_execution.SetShaderResourceView(
							reinterpret_cast<ID3D12Resource *>(src_mem),
							src_bytes));
						BPS_DX_ERR(classify_execution.SetShaderResourceView(
							reinterpret_cast<ID3D12Resource *>(criterion_mem),
							criterion_bytes));
						BPS_DX_ERR(classify_execution.SetShaderResourceView(
							reinterpret_cast<ID3D12Resource *>(trigger_mem),
							trigger_bytes));
						BPS_DX_ERR(classify_execution.SetShaderResourceView(
							path_resource.Get(), path_bytes));
						const UINT classify_x = static_cast<UINT>((width + 15) / 16);
						const UINT classify_y = static_cast<UINT>((height + 15) / 16);
						BPS_DX_ERR(classify_execution.Execute(classify_x, classify_y));
					}

					std::vector<std::uint32_t> lane_counts(
						static_cast<size_t>(lineCount), 0u);
					if (!err) {
						D3D12_HEAP_PROPERTIES readback_heap = {
							D3D12_HEAP_TYPE_READBACK,
							D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
							D3D12_MEMORY_POOL_UNKNOWN,
							0, 0};
						D3D12_RESOURCE_DESC readback_desc = {
							D3D12_RESOURCE_DIMENSION_BUFFER, 0,
							counts_bytes, 1, 1, 1,
							DXGI_FORMAT_UNKNOWN, 1, 0,
							D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
							D3D12_RESOURCE_FLAG_NONE};
						Microsoft::WRL::ComPtr<ID3D12Resource> readback_resource;
						if (FAILED(dx_dataP->context->mDevice->CreateCommittedResource(
								&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
								D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
								IID_PPV_ARGS(readback_resource.GetAddressOf())))) {
							err = PF_Err_OUT_OF_MEMORY;
						} else {
							D3D12_RESOURCE_BARRIER barrier = {};
							barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
							barrier.Transition.pResource = counts_resource.Get();
							barrier.Transition.Subresource = 0;
							barrier.Transition.StateBefore =
								D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
							barrier.Transition.StateAfter =
								D3D12_RESOURCE_STATE_COPY_SOURCE;
							dx_dataP->context->mCommandList->ResourceBarrier(1, &barrier);
							dx_dataP->context->mCommandList->CopyResource(
								readback_resource.Get(), counts_resource.Get());
							dx_dataP->context->CloseWaitAndReset();

							void *mapped_counts = nullptr;
							D3D12_RANGE read_range = {0, counts_bytes};
							if (FAILED(readback_resource->Map(
									0, &read_range, &mapped_counts))) {
								err = PF_Err_OUT_OF_MEMORY;
							} else {
								std::memcpy(lane_counts.data(), mapped_counts, counts_bytes);
								D3D12_RANGE write_range = {0, 0};
								readback_resource->Unmap(0, &write_range);
							}
						}
					}

					BPS_GpuPathMapLayout layout;
					if (!err && !BPS_ComputeGpuPathMapLayout(lane_counts, &layout)) {
						err = PF_Err_INTERNAL_STRUCT_DAMAGED;
					}
					if (!err && layout.mappedRecordCount > 0u &&
						layout.mappedWorkItemCount > 0u) {
						const UINT records_bytes =
							static_cast<UINT>(static_cast<size_t>(layout.mappedRecordCount) *
											  sizeof(BpsMappedPixelRecord));
						const UINT offsets_bytes =
							static_cast<UINT>((static_cast<size_t>(lineCount) + 1u) *
											  sizeof(std::uint32_t));
						const UINT work_records_bytes =
							static_cast<UINT>(static_cast<size_t>(layout.mappedWorkItemCount) *
											  sizeof(BpsMappedPixelRecord));

						Microsoft::WRL::ComPtr<ID3D12Resource> cursor_resource;
						Microsoft::WRL::ComPtr<ID3D12Resource> cursor_upload;
						Microsoft::WRL::ComPtr<ID3D12Resource> line_offsets_upload;
						Microsoft::WRL::ComPtr<ID3D12Resource> work_offsets_upload;
						Microsoft::WRL::ComPtr<ID3D12Resource> work_records_resource;

						ERR(create_default_buffer(records_bytes,
												  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
												  D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
												  dx_dataP->path_records_resource));
						ERR(create_uploaded_buffer(layout.lineOffsets.data(), offsets_bytes,
													dx_dataP->path_line_offsets_resource,
													line_offsets_upload));
						ERR(create_uploaded_buffer(layout.workOffsets.data(), offsets_bytes,
													dx_dataP->path_work_offsets_resource,
													work_offsets_upload));
						ERR(create_uploaded_buffer(layout.lineOffsets.data(), offsets_bytes,
													cursor_resource,
													cursor_upload));
						ERR(create_default_buffer(work_records_bytes,
												  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
												  D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
												  work_records_resource));

						if (!err) {
							D3D12_RESOURCE_BARRIER barriers[3] = {};
							barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
							barriers[0].Transition.pResource =
								dx_dataP->path_line_offsets_resource.Get();
							barriers[0].Transition.Subresource = 0;
							barriers[0].Transition.StateBefore =
								D3D12_RESOURCE_STATE_COPY_DEST;
							barriers[0].Transition.StateAfter =
								D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
							barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
							barriers[1].Transition.pResource =
								dx_dataP->path_work_offsets_resource.Get();
							barriers[1].Transition.Subresource = 0;
							barriers[1].Transition.StateBefore =
								D3D12_RESOURCE_STATE_COPY_DEST;
							barriers[1].Transition.StateAfter =
								D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
							barriers[2].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
							barriers[2].Transition.pResource = cursor_resource.Get();
							barriers[2].Transition.Subresource = 0;
							barriers[2].Transition.StateBefore =
								D3D12_RESOURCE_STATE_COPY_DEST;
							barriers[2].Transition.StateAfter =
								D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
							dx_dataP->context->mCommandList->ResourceBarrier(3, barriers);

							DXShaderExecution scatter_execution(
								dx_dataP->context,
								dx_dataP->path_scatter_records_shader,
								5);
							BPS_DX_ERR(scatter_execution.SetParamBuffer(
								&dx_params, sizeof(dx_params)));
							BPS_DX_ERR(scatter_execution.SetUnorderedAccessView(
								lane_resource.Get(), lane_bytes));
							BPS_DX_ERR(scatter_execution.SetUnorderedAccessView(
								key_resource.Get(), key_bytes));
							BPS_DX_ERR(scatter_execution.SetUnorderedAccessView(
								cursor_resource.Get(), offsets_bytes));
							BPS_DX_ERR(scatter_execution.SetUnorderedAccessView(
								dx_dataP->path_records_resource.Get(), records_bytes));
							const UINT scatter_x = static_cast<UINT>((width + 15) / 16);
							const UINT scatter_y = static_cast<UINT>((height + 15) / 16);
							BPS_DX_ERR(scatter_execution.Execute(scatter_x, scatter_y));
						}

						if (!err) {
							// SRV layout is shared: t0=src, t1=criterion, t2=trigger,
							// t3=path. This pass only reads line/work offsets, so bind
							// dummies into the unused criterion/trigger slots.
							DXShaderExecution sort_records_execution(
								dx_dataP->context,
								dx_dataP->path_sort_records_shader,
								7);
							BPS_DX_ERR(sort_records_execution.SetParamBuffer(
								&dx_params, sizeof(dx_params)));
							BPS_DX_ERR(sort_records_execution.SetUnorderedAccessView(
								dx_dataP->path_records_resource.Get(), records_bytes));
							BPS_DX_ERR(sort_records_execution.SetUnorderedAccessView(
								work_records_resource.Get(), work_records_bytes));
							BPS_DX_ERR(sort_records_execution.SetShaderResourceView(
								dx_dataP->path_line_offsets_resource.Get(), offsets_bytes));
							BPS_DX_ERR(sort_records_execution.SetShaderResourceView(
								reinterpret_cast<ID3D12Resource *>(criterion_mem),
								criterion_bytes));
							BPS_DX_ERR(sort_records_execution.SetShaderResourceView(
								reinterpret_cast<ID3D12Resource *>(trigger_mem),
								trigger_bytes));
							BPS_DX_ERR(sort_records_execution.SetShaderResourceView(
								dx_dataP->path_work_offsets_resource.Get(), offsets_bytes));
							BPS_DX_ERR(sort_records_execution.Execute(
								static_cast<UINT>(lineCount), 1));
						}

						if (!err) {
							D3D12_RESOURCE_BARRIER barrier = {};
							barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
							barrier.Transition.pResource =
								dx_dataP->path_records_resource.Get();
							barrier.Transition.Subresource = 0;
							barrier.Transition.StateBefore =
								D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
							barrier.Transition.StateAfter =
								D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
							dx_dataP->context->mCommandList->ResourceBarrier(1, &barrier);
							dx_dataP->context->CloseWaitAndReset();

							dx_dataP->path_map_key = pathMapKey;
							dx_dataP->path_map_width = width;
							dx_dataP->path_map_height = height;
							dx_dataP->path_map_line_count = lineCount;
							dx_dataP->path_mapped_record_count =
								static_cast<int>(layout.mappedRecordCount);
							dx_dataP->path_mapped_work_item_count =
								static_cast<int>(layout.mappedWorkItemCount);
							using_gpu_path_map = true;
						}
					}
				}

				if (using_gpu_path_map) {
					mappedRecordCount = dx_dataP->path_mapped_record_count;
					mappedWorkItemCount = dx_dataP->path_mapped_work_item_count;
					gpu_records_resource = dx_dataP->path_records_resource;
					gpu_line_offsets_resource = dx_dataP->path_line_offsets_resource;
					gpu_work_offsets_resource = dx_dataP->path_work_offsets_resource;
				} else if (err) {
					err = PF_Err_NONE;
					fallbackPathMap = BPS_AcquirePathMap(width, height, *paramsP);
					if (fallbackPathMap) {
						mappedRecordCount = static_cast<int>(fallbackPathMap->mappedRecordCount);
						mappedWorkItemCount = static_cast<int>(fallbackPathMap->mappedWorkItemCount);
						mappedRecordsHost = fallbackPathMap->records.empty()
							? nullptr : fallbackPathMap->records.data();
						mappedLineOffsetsHost = fallbackPathMap->lineOffsets.empty()
							? nullptr : fallbackPathMap->lineOffsets.data();
						mappedWorkOffsetsHost = fallbackPathMap->workOffsets.empty()
							? nullptr : fallbackPathMap->workOffsets.data();
					}
				}
			}

			if (err || mappedRecordCount <= 0 || mappedWorkItemCount <= 0 ||
				(!using_gpu_path_map &&
				 (!mappedRecordsHost || !mappedLineOffsetsHost || !mappedWorkOffsetsHost))) {
				return err;
			}

			const UINT records_bytes =
				static_cast<UINT>(static_cast<size_t>(mappedRecordCount) *
								  sizeof(BpsMappedPixelRecord));
			const UINT offsets_bytes =
				static_cast<UINT>(static_cast<size_t>(lineCount + 1) *
								  sizeof(std::uint32_t));
			const UINT mapped_domain_bytes =
				static_cast<UINT>(static_cast<size_t>(mappedWorkItemCount) *
								  sizeof(UINT));
			const UINT mapped_keys_bytes =
				static_cast<UINT>(static_cast<size_t>(mappedWorkItemCount) *
								  sizeof(float));

			Microsoft::WRL::ComPtr<ID3D12Resource> records_resource;
			Microsoft::WRL::ComPtr<ID3D12Resource> records_upload;
			Microsoft::WRL::ComPtr<ID3D12Resource> line_offsets_resource;
			Microsoft::WRL::ComPtr<ID3D12Resource> line_offsets_upload;
			Microsoft::WRL::ComPtr<ID3D12Resource> work_offsets_resource;
			Microsoft::WRL::ComPtr<ID3D12Resource> work_offsets_upload;
			if (using_gpu_path_map) {
				records_resource = gpu_records_resource;
				line_offsets_resource = gpu_line_offsets_resource;
				work_offsets_resource = gpu_work_offsets_resource;
			} else {
				ERR(create_uploaded_buffer(mappedRecordsHost, records_bytes,
											records_resource, records_upload));
				ERR(create_uploaded_buffer(mappedLineOffsetsHost, offsets_bytes,
											line_offsets_resource, line_offsets_upload));
				ERR(create_uploaded_buffer(mappedWorkOffsetsHost, offsets_bytes,
											work_offsets_resource, work_offsets_upload));
			}
			if (err) {
				return err;
			}

			if (!using_gpu_path_map) {
				D3D12_RESOURCE_BARRIER srv_barriers[3] = {};
				ID3D12Resource *srv_resources[3] = {
					records_resource.Get(),
					line_offsets_resource.Get(),
					work_offsets_resource.Get()
				};
				for (int i = 0; i < 3; ++i) {
					srv_barriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
					srv_barriers[i].Transition.pResource = srv_resources[i];
					srv_barriers[i].Transition.Subresource = 0;
					srv_barriers[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
					srv_barriers[i].Transition.StateAfter =
						D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
				}
				dx_dataP->context->mCommandList->ResourceBarrier(3, srv_barriers);
			}

			Microsoft::WRL::ComPtr<ID3D12Resource> mapped_domain_resource;
			Microsoft::WRL::ComPtr<ID3D12Resource> mapped_keys_resource;
			D3D12_HEAP_PROPERTIES default_heap = {
				D3D12_HEAP_TYPE_DEFAULT,
				D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
				D3D12_MEMORY_POOL_UNKNOWN,
				0, 0};
			D3D12_RESOURCE_DESC mapped_domain_desc = {
				D3D12_RESOURCE_DIMENSION_BUFFER, 0,
				mapped_domain_bytes, 1, 1, 1,
				DXGI_FORMAT_UNKNOWN, 1, 0,
				D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
				D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
			D3D12_RESOURCE_DESC mapped_keys_desc = mapped_domain_desc;
			mapped_keys_desc.Width = mapped_keys_bytes;
			if (FAILED(dx_dataP->context->mDevice->CreateCommittedResource(
					&default_heap, D3D12_HEAP_FLAG_NONE,
					&mapped_domain_desc,
					D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
					nullptr,
					IID_PPV_ARGS(mapped_domain_resource.GetAddressOf())))) {
				return PF_Err_OUT_OF_MEMORY;
			}
			if (FAILED(dx_dataP->context->mDevice->CreateCommittedResource(
					&default_heap, D3D12_HEAP_FLAG_NONE,
					&mapped_keys_desc,
					D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
					nullptr,
					IID_PPV_ARGS(mapped_keys_resource.GetAddressOf())))) {
				return PF_Err_OUT_OF_MEMORY;
			}

			DXShaderExecution mapped_execution(
				dx_dataP->context,
				dx_dataP->mapped_sort_shader,
				10);
			BPS_DX_ERR(mapped_execution.SetParamBuffer(&dx_params, sizeof(dx_params)));
			BPS_DX_ERR(mapped_execution.SetUnorderedAccessView(
				reinterpret_cast<ID3D12Resource *>(dst_mem),
				dst_bytes));
			BPS_DX_ERR(mapped_execution.SetUnorderedAccessView(
				mapped_domain_resource.Get(), mapped_domain_bytes));
			BPS_DX_ERR(mapped_execution.SetUnorderedAccessView(
				mapped_keys_resource.Get(), mapped_keys_bytes));
			BPS_DX_ERR(mapped_execution.SetShaderResourceView(
				reinterpret_cast<ID3D12Resource *>(src_mem),
				src_bytes));
			BPS_DX_ERR(mapped_execution.SetShaderResourceView(
				reinterpret_cast<ID3D12Resource *>(criterion_mem),
				criterion_bytes));
			BPS_DX_ERR(mapped_execution.SetShaderResourceView(
				reinterpret_cast<ID3D12Resource *>(trigger_mem),
				trigger_bytes));
			BPS_DX_ERR(mapped_execution.SetShaderResourceView(
				records_resource.Get(), records_bytes));
			BPS_DX_ERR(mapped_execution.SetShaderResourceView(
				line_offsets_resource.Get(), offsets_bytes));
			BPS_DX_ERR(mapped_execution.SetShaderResourceView(
				work_offsets_resource.Get(), offsets_bytes));
			BPS_DX_ERR(mapped_execution.Execute(static_cast<UINT>(lineCount), 1));
			return err;
		}

		if (domainStride <= 0 || outputWidth <= 0 || outputHeight <= 0) {
			return PF_Err_NONE;
		}

		const UINT domain_bytes = static_cast<UINT>(
			static_cast<size_t>(lineCount) * static_cast<size_t>(domainStride) * sizeof(UINT));
		const UINT keys_bytes = static_cast<UINT>(
			static_cast<size_t>(lineCount) * static_cast<size_t>(domainStride) * sizeof(float));
		Microsoft::WRL::ComPtr<ID3D12Resource> domain_resource;
		Microsoft::WRL::ComPtr<ID3D12Resource> keys_resource;
		D3D12_HEAP_PROPERTIES default_heap = {
			D3D12_HEAP_TYPE_DEFAULT,
			D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
			D3D12_MEMORY_POOL_UNKNOWN,
			0, 0};
		D3D12_RESOURCE_DESC domain_desc = {
			D3D12_RESOURCE_DIMENSION_BUFFER, 0,
			domain_bytes, 1, 1, 1,
			DXGI_FORMAT_UNKNOWN, 1, 0,
			D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
			D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
		D3D12_RESOURCE_DESC keys_desc = domain_desc;
		keys_desc.Width = keys_bytes;
		if (FAILED(dx_dataP->context->mDevice->CreateCommittedResource(
				&default_heap,
				D3D12_HEAP_FLAG_NONE,
				&domain_desc,
				D3D12_RESOURCE_STATE_COPY_DEST,
				nullptr,
				IID_PPV_ARGS(domain_resource.GetAddressOf())))) {
			return PF_Err_OUT_OF_MEMORY;
		}
		if (FAILED(dx_dataP->context->mDevice->CreateCommittedResource(
				&default_heap,
				D3D12_HEAP_FLAG_NONE,
				&keys_desc,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
				nullptr,
				IID_PPV_ARGS(keys_resource.GetAddressOf())))) {
			return PF_Err_OUT_OF_MEMORY;
		}

		// Clear domain indices to 0xFFFFFFFF (invalid).
		Microsoft::WRL::ComPtr<ID3D12Resource> upload_resource;
		D3D12_HEAP_PROPERTIES upload_heap = {
			D3D12_HEAP_TYPE_UPLOAD,
			D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
			D3D12_MEMORY_POOL_UNKNOWN,
			0, 0};
		D3D12_RESOURCE_DESC upload_desc = domain_desc;
		upload_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
		if (FAILED(dx_dataP->context->mDevice->CreateCommittedResource(
				&upload_heap,
				D3D12_HEAP_FLAG_NONE,
				&upload_desc,
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(upload_resource.GetAddressOf())))) {
			return PF_Err_OUT_OF_MEMORY;
		}

		void *mapped = nullptr;
		if (FAILED(upload_resource->Map(0, nullptr, &mapped))) {
			return PF_Err_OUT_OF_MEMORY;
		}
		std::memset(mapped, 0xFF, domain_bytes);
		upload_resource->Unmap(0, nullptr);

		{
			DXShaderExecution sort_execution(
				dx_dataP->context,
				dx_dataP->domain_sort_shader,
				7);
			// Upload clear before the sort dispatch shares the command list.
			dx_dataP->context->mCommandList->CopyResource(
				domain_resource.Get(), upload_resource.Get());
			dx_dataP->context->mCommandList->CopyResource(
				path_resource.Get(), path_upload.Get());
			D3D12_RESOURCE_BARRIER barriers[2] = {};
			barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			barriers[0].Transition.pResource = domain_resource.Get();
			barriers[0].Transition.Subresource = 0;
			barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
			barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
			barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			barriers[1].Transition.pResource = path_resource.Get();
			barriers[1].Transition.Subresource = 0;
			barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
			barriers[1].Transition.StateAfter =
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
			dx_dataP->context->mCommandList->ResourceBarrier(2, barriers);

			BPS_DX_ERR(sort_execution.SetParamBuffer(&dx_params, sizeof(dx_params)));
			BPS_DX_ERR(sort_execution.SetUnorderedAccessView(
				domain_resource.Get(), domain_bytes));
			BPS_DX_ERR(sort_execution.SetUnorderedAccessView(
				keys_resource.Get(), keys_bytes));
			BPS_DX_ERR(sort_execution.SetShaderResourceView(
				reinterpret_cast<ID3D12Resource *>(src_mem),
				src_bytes));
			BPS_DX_ERR(sort_execution.SetShaderResourceView(
				reinterpret_cast<ID3D12Resource *>(criterion_mem),
				criterion_bytes));
			BPS_DX_ERR(sort_execution.SetShaderResourceView(
				reinterpret_cast<ID3D12Resource *>(trigger_mem),
				trigger_bytes));
			BPS_DX_ERR(sort_execution.SetShaderResourceView(
				path_resource.Get(), path_bytes));
			BPS_DX_ERR(sort_execution.Execute(static_cast<UINT>(lineCount), 1));
		}

		if (!err) {
			// Domain is now read-only for the apply pass.
			DXShaderExecution apply_execution(
				dx_dataP->context,
				dx_dataP->apply_domain_shader,
				7);
			D3D12_RESOURCE_BARRIER barrier = {};
			barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			barrier.Transition.pResource = domain_resource.Get();
			barrier.Transition.Subresource = 0;
			barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
			barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
			dx_dataP->context->mCommandList->ResourceBarrier(1, &barrier);

			BPS_DX_ERR(apply_execution.SetParamBuffer(&dx_params, sizeof(dx_params)));
			BPS_DX_ERR(apply_execution.SetUnorderedAccessView(
				reinterpret_cast<ID3D12Resource *>(dst_mem),
				dst_bytes));
			BPS_DX_ERR(apply_execution.SetShaderResourceView(
				reinterpret_cast<ID3D12Resource *>(src_mem),
				src_bytes));
			BPS_DX_ERR(apply_execution.SetShaderResourceView(
				reinterpret_cast<ID3D12Resource *>(criterion_mem),
				criterion_bytes));
			BPS_DX_ERR(apply_execution.SetShaderResourceView(
				reinterpret_cast<ID3D12Resource *>(trigger_mem),
				trigger_bytes));
			BPS_DX_ERR(apply_execution.SetShaderResourceView(
				path_resource.Get(), path_bytes));
			BPS_DX_ERR(apply_execution.SetShaderResourceView(
				domain_resource.Get(), domain_bytes));
			const UINT apply_x = static_cast<UINT>((outputWidth + 15) / 16);
			const UINT apply_y = static_cast<UINT>((outputHeight + 15) / 16);
			BPS_DX_ERR(apply_execution.Execute(apply_x, apply_y));
		}
		return err;
	}
#endif

#if defined(BPS_HAS_METAL)
	if (extraP->input->what_gpu == PF_GPU_Framework_METAL) {
		return BPS_MetalSmartRender(
			in_data, out_data,
			input_worldP, output_worldP, extraP, paramsP,
			src_mem, criterion_mem, trigger_mem, dst_mem,
			srcPitch, dstPitch,
			width, height,
			inputOriginX, inputOriginY, inputWidth, inputHeight,
			criterionPitch, criterionOriginX, criterionOriginY,
			criterionWidth, criterionHeight,
			triggerPitch, triggerOriginX, triggerOriginY,
			triggerWidth, triggerHeight,
			outputOriginX, outputOriginY,
			outputWidth, outputHeight,
			direction, ordering,
			lineCount);
	}
#endif

	// No backend matched — should not be reached when BPS_GPU_ENABLED is set.
	(void)src_mem; (void)dst_mem; (void)srcPitch; (void)dstPitch;
	(void)width; (void)height; (void)mode; (void)direction; (void)ordering;
	(void)criterion; (void)trigger; (void)affect; (void)cycleDegrees;
	(void)freePMin; (void)freeQMin; (void)freeLineLength;
	(void)radialLength; (void)domainStride; (void)lineCount;
	return PF_Err_UNRECOGNIZED_PARAM_TYPE;
}
