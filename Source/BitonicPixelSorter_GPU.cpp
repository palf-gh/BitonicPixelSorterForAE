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

#include <cstring>
#include <new>

#if defined(BPS_HAS_HLSL)
	#include "DirectXUtils.h"
#endif

#if defined(BPS_HAS_CUDA)
	// Host launch wrapper, defined in GPU/BitonicPixelSorter_Kernel.cu.
	extern "C" void BitonicSort_CUDA(
		const void *src, void *dst,
		int srcPitch, int dstPitch, int width, int height,
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

//-----------------------------------------------------------------------------
PF_Err BPS_GPUDeviceSetup(
	PF_InData				*in_dataP,
	PF_OutData				*out_dataP,
	PF_GPUDeviceSetupExtra	*extraP)
{
	PF_Err err = PF_Err_NONE;

#if defined(BPS_HAS_CUDA)
	if (extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
		// CUDA kernels are statically linked; nothing to compile here.
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
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
			std::wstring cso_path;
			std::wstring sig_path;
			BPS_DX_ERR(GetShaderPath(L"BitonicSortKernel", cso_path, sig_path));
			BPS_DX_ERR(dx_dataP->context->LoadShader(
				cso_path.c_str(),
				sig_path.c_str(),
				dx_dataP->sort_shader));
		}

		if (!err) {
			extraP->output->gpu_data = gpu_dataH;
			out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
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

#if !defined(BPS_HAS_CUDA) && !defined(BPS_HAS_OPENCL) && !defined(BPS_HAS_HLSL)
	(void)in_dataP; (void)out_dataP; (void)extraP;
#endif

	// Metal device setup is added in later phases.
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

		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite =
			AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite,
											 kPFHandleSuiteVersion1, out_dataP);
		handle_suite->host_dispose_handle(gpu_dataH);
	}
#endif

	// CUDA: nothing allocated at setup, so nothing to release.
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

	const int bytes_per_pixel = 16; // float4 BGRA
	const int width    = input_worldP->width;
	const int height   = input_worldP->height;
	const int srcPitch = input_worldP->rowbytes  / bytes_per_pixel;
	const int dstPitch = output_worldP->rowbytes / bytes_per_pixel;
	const int direction = (paramsP->direction == BPS_DIR_HORIZONTAL) ? 1 : 0;
	const int ordering  = paramsP->ascending ? 1 : 0;
	const int lineCount = direction ? height : width;

	if (err) {
		return err;
	}

	if (lineCount <= 0) {
		return PF_Err_NONE;
	}

#if defined(BPS_HAS_OPENCL)
	if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL) {
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
		BitonicSort_CUDA(src_mem, dst_mem, srcPitch, dstPitch, width, height,
						 direction, ordering, paramsP->thresholdMin, paramsP->thresholdMax);
		if (cudaPeekAtLastError() != cudaSuccess) {
			err = PF_Err_INTERNAL_STRUCT_DAMAGED;
		}
		return err;
	}
#endif

#if defined(BPS_HAS_HLSL)
	if (extraP->input->what_gpu == PF_GPU_Framework_DIRECTX) {
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

	// Metal dispatch is added in later phases.
	(void)src_mem; (void)dst_mem; (void)srcPitch; (void)dstPitch;
	(void)width; (void)height; (void)direction; (void)ordering;
	return PF_Err_UNRECOGNIZED_PARAM_TYPE;
}
