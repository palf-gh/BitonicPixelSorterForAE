/*
	BitonicPixelSorter_GPU.cpp

	GPU command handling: device setup/setdown and the GPU smart-render dispatch.
	After Effects GPU worlds are PF_PixelFormat_GPU_BGRA128 (linear, row-pitched
	float4 buffers in BGRA order). A single pass reads the source and writes the
	sorted result to the destination — no intermediate buffer is needed.

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

#include <atomic>
#if defined(BPS_RENDER_DIAG)
	#include <cstdarg>
	#include <cstdio>
#endif
#include <cstring>
#include <mutex>
#include <new>
#include <string>

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
		int direction, int ordering, float thresholdMin, float thresholdMax);
#endif

#if defined(BPS_HAS_OPENCL)
namespace {

struct OpenCLGPUData {
	cl_program program;
	cl_kernel sort_kernel;
};

inline PF_Err CL2Err(cl_int cl_result)
{
	return cl_result == CL_SUCCESS ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED;
}

#define BPS_CL_ERR(FUNC) ERR(CL2Err(FUNC))

static void ReleaseOpenCLData(OpenCLGPUData *cl_dataP)
{
	if (cl_dataP) {
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
	int direction;
	int ordering;
	float thresholdMin;
	float thresholdMax;
};

inline PF_Err DXErr(bool success)
{
	return success ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED;
}

#define BPS_DX_ERR(FUNC) ERR(DXErr(FUNC))

static void ReleaseDirectXData(DirectXGPUData *dx_dataP)
{
	if (dx_dataP) {
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
		// builds on older After Effects versions.
		if (!err) {
			out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
			BPS_RecordGpuDevice(device_info.device_framework, std::string());
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

	const int direction = (paramsP->direction == BPS_DIR_HORIZONTAL) ? 1 : 0;
	const int lineCount = direction ? output_worldP->height : output_worldP->width;
	if (lineCount <= 0) {
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
		BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &direction));
		BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(int), &ordering));
		BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(float), &paramsP->thresholdMin));
		BPS_CL_ERR(clSetKernelArg(cl_dataP->sort_kernel, param_index++, sizeof(float), &paramsP->thresholdMax));

		const size_t local = 256;
		const size_t global = static_cast<size_t>(lineCount) * local;
		BPS_CL_ERR(clEnqueueNDRangeKernel(reinterpret_cast<cl_command_queue>(device_info.command_queuePV),
										  cl_dataP->sort_kernel,
										  1,
										  0,
										  &global,
										  &local,
										  0,
										  0,
										  0));
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
							 direction, ordering, paramsP->thresholdMin, paramsP->thresholdMax);

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
			direction,
			ordering,
			paramsP->thresholdMin,
			paramsP->thresholdMax
		};

		DXShaderExecution shader_execution(
			dx_dataP->context,
			dx_dataP->sort_shader,
			3);

		BPS_DX_ERR(shader_execution.SetParamBuffer(&dx_params, sizeof(dx_params)));
		BPS_DX_ERR(shader_execution.SetUnorderedAccessView(
			reinterpret_cast<ID3D12Resource *>(dst_mem),
			static_cast<UINT>(output_worldP->height * output_worldP->rowbytes)));
		BPS_DX_ERR(shader_execution.SetShaderResourceView(
			reinterpret_cast<ID3D12Resource *>(src_mem),
			static_cast<UINT>(input_worldP->height * input_worldP->rowbytes)));
		BPS_DX_ERR(shader_execution.Execute(static_cast<UINT>(lineCount), 1));
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
	(void)width; (void)height; (void)direction; (void)ordering;
	return PF_Err_UNRECOGNIZED_PARAM_TYPE;
}
