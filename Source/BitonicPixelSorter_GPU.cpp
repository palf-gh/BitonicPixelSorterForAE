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
		const void *src, void *dst,
		int srcPitch, int dstPitch, int width, int height,
		int inputOriginX, int inputOriginY,
		int outputOriginX, int outputOriginY, int outputWidth, int outputHeight,
		int mode, int direction, int ordering, int criterion, int trigger, int affect,
		float cycleDegrees, int lineCount,
		int freePMin, int freeQMin, int freeLineLength, int radialLength,
		int domainStride,
		float thresholdMin, float thresholdMax,
		float angleCos, float angleSin, float centerX, float centerY,
		float swirlK, int swirlLineMin,
		int pathDirection, int pathSMin, int pathNMin, int pathSampleCount,
		const void *pathSamplesHost,
		int mappedRecordCount, int mappedWorkItemCount,
		const void *mappedRecordsHost,
		const void *mappedLineOffsetsHost,
		const void *mappedWorkOffsetsHost);
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
};

inline PF_Err CL2Err(cl_int cl_result)
{
	return cl_result == CL_SUCCESS ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED;
}

#define BPS_CL_ERR(FUNC) ERR(CL2Err(FUNC))

static void ReleaseOpenCLData(OpenCLGPUData *cl_dataP)
{
	if (cl_dataP) {
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
};

struct DirectXSortParams {
	int srcPitch;
	int dstPitch;
	int width;
	int height;
	int inputOriginX;
	int inputOriginY;
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
	PF_SmartRenderExtra			*extraP,
	const BitonicSorterParams	*paramsP)
{
	PF_Err err = PF_Err_NONE;

	if (pixel_format != PF_PixelFormat_GPU_BGRA128) {
		return PF_Err_UNRECOGNIZED_PARAM_TYPE;
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
	void *dst_mem = 0;
	ERR(gpu_suite->GetGPUWorldData(in_data->effect_ref, output_worldP, &dst_mem));

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
	const int width    = in_data->width;
	const int height   = in_data->height;
	const int srcPitch = input_worldP->rowbytes  / bytes_per_pixel;
	const int dstPitch = output_worldP->rowbytes / bytes_per_pixel;
	const int inputOriginX = input_worldP->origin_x;
	const int inputOriginY = input_worldP->origin_y;
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
	const int pathSMin = static_cast<int>(paramsP->pathSMin);
	const int pathNMin = static_cast<int>(paramsP->pathNMin);
	const int pathSampleCount = static_cast<int>(paramsP->pathSampleCount);
	const void *pathSamplesHost = paramsP->pathSamples;
	const int mappedRecordCount = static_cast<int>(paramsP->mappedRecordCount);
	const int mappedWorkItemCount = static_cast<int>(paramsP->mappedWorkItemCount);
	const void *mappedRecordsHost = paramsP->mappedRecords;
	const void *mappedLineOffsetsHost = paramsP->mappedLineOffsets;
	const void *mappedWorkOffsetsHost = paramsP->mappedWorkOffsets;
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
		cl_mem cl_dst_mem = reinterpret_cast<cl_mem>(dst_mem);
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
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &srcPitch));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &dstPitch));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &width));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &height));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &inputOriginX));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &inputOriginY));
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

			if (err || mappedRecordCount <= 0 || mappedWorkItemCount <= 0 ||
				!mappedRecordsHost || !mappedLineOffsetsHost || !mappedWorkOffsetsHost) {
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
			cl_mem records_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
												recordsBytes, 0, &cl_result);
			BPS_CL_ERR(cl_result);
			cl_mem line_offsets_mem = 0;
			if (!err) {
				line_offsets_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
												  offsetsBytes, 0, &cl_result);
				BPS_CL_ERR(cl_result);
			}
			cl_mem work_offsets_mem = 0;
			if (!err) {
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
			if (!err) {
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
			if (work_offsets_mem) (void)clReleaseMemObject(work_offsets_mem);
			if (line_offsets_mem) (void)clReleaseMemObject(line_offsets_mem);
			if (records_mem) (void)clReleaseMemObject(records_mem);
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
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(cl_mem), &domain_mem));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(cl_mem), &keys_mem));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &srcPitch));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &width));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &height));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &inputOriginX));
			BPS_CL_ERR(clSetKernelArg(cl_dataP->domain_sort_kernel, param_index++, sizeof(int), &inputOriginY));
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
			BitonicSort_CUDA(src_mem, dst_mem, srcPitch, dstPitch, width, height,
							 inputOriginX, inputOriginY, outputOriginX, outputOriginY,
							 outputWidth, outputHeight,
							 mode, direction, ordering, criterion, trigger, affect,
							 cycleDegrees, lineCount,
							 freePMin, freeQMin, freeLineLength, radialLength,
							 domainStride,
							 paramsP->thresholdMin, paramsP->thresholdMax,
							 paramsP->angleCos, paramsP->angleSin,
							 paramsP->centerX, paramsP->centerY,
							 swirlK, swirlLineMin,
							 pathDirection, pathSMin, pathNMin, pathSampleCount,
							 pathSamplesHost,
							 mappedRecordCount, mappedWorkItemCount,
							 mappedRecordsHost,
							 mappedLineOffsetsHost,
							 mappedWorkOffsetsHost);

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
			pathSMin,
			pathNMin,
			pathSampleCount
		};

		const UINT src_bytes =
			static_cast<UINT>(input_worldP->height * input_worldP->rowbytes);
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
				4);

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

			if (err || mappedRecordCount <= 0 || mappedWorkItemCount <= 0 ||
				!mappedRecordsHost || !mappedLineOffsetsHost || !mappedWorkOffsetsHost) {
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
			ERR(create_uploaded_buffer(mappedRecordsHost, records_bytes,
										records_resource, records_upload));
			ERR(create_uploaded_buffer(mappedLineOffsetsHost, offsets_bytes,
										line_offsets_resource, line_offsets_upload));
			ERR(create_uploaded_buffer(mappedWorkOffsetsHost, offsets_bytes,
										work_offsets_resource, work_offsets_upload));
			if (err) {
				return err;
			}

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
				8);
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
				5);
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
				path_resource.Get(), path_bytes));
			BPS_DX_ERR(sort_execution.Execute(static_cast<UINT>(lineCount), 1));
		}

		if (!err) {
			// Domain is now read-only for the apply pass.
			DXShaderExecution apply_execution(
				dx_dataP->context,
				dx_dataP->apply_domain_shader,
				5);
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
			src_mem, dst_mem,
			srcPitch, dstPitch,
			width, height,
			inputOriginX, inputOriginY,
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
