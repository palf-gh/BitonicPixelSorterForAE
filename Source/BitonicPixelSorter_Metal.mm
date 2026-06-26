/*
	BitonicPixelSorter_Metal.mm

	Objective-C++ Metal backend for BitonicPixelSorter. Wraps all Metal /
	Objective-C code so the cross-platform BitonicPixelSorter_GPU.cpp stays
	pure C++.

	Lifecycle:
	  BPS_MetalDeviceSetup   — compile kernel source, build pipeline, store handle.
	  BPS_MetalDeviceSetdown — release pipeline, dispose handle.
	  BPS_MetalSmartRender   — encode and commit one compute pass per frame.

	Bridging / retain strategy:
	  The MTLComputePipelineState is an Obj-C object managed by ARC inside this
	  .mm file.  It must survive inside a plain malloc'd PF_Handle that ARC does
	  NOT track.  We store it as a raw void* using CFBridgingRetain (equivalent
	  to __bridge_retained) which transfers ownership out of ARC and increments
	  the retain count.  Setdown calls CFBridgingRelease (equivalent to
	  __bridge_transfer into a local that immediately goes out of scope) to
	  balance the retain.  No other code retains or releases the object.

	Threadgroup memory:
	  The kernel uses 32 KB of threadgroup memory (two arrays of 4096 elements,
	  float and uint).  Before building the pipeline we validate that the device
	  supports at least 32 768 bytes; if not we return PF_Err_INTERNAL_STRUCT_DAMAGED
	  so AE falls back to the CPU path gracefully.

	Concurrency:
	  Each render call allocates its own MTLBuffer for params and obtains its own
	  MTLCommandBuffer from the queue — both are per-call, immutable after setup.
	  The pipeline state is also immutable.  No mutable shared state exists at
	  render time, so the BPS_GpuRenderMutex is NOT taken here (consistent with
	  Metal's concurrency model, which encourages parallel command buffer
	  construction).  If AE ever calls Setdown while a command buffer is in flight,
	  the Metal driver keeps the pipeline alive until GPU work completes; our
	  CFBridgingRelease in Setdown is therefore safe.
*/

#if defined(BPS_HAS_METAL)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "BPS_MetalBackend.h"
#include "BitonicPixelSorter_Kernel.metal.h"	// kBitonicPixelSorter_Kernel_MetalString

#include <cstring>
#include <new>

// ---------------------------------------------------------------------------
// Per-device GPU data stored in the PF_Handle allocated at setup.
// The pipeline is stored as a bridged void* (CFBridgingRetain) so it survives
// in plain malloc'd memory without ARC tracking.
// ---------------------------------------------------------------------------
struct MetalGPUData {
	void *sort_pipeline_bridge;	// CFBridgingRetain'd id<MTLComputePipelineState>
};

// ---------------------------------------------------------------------------
// Parameter struct — must match BitonicSortParams in the .metal kernel exactly
// (same field order, same types, same sizes) so the host can memcpy the struct
// into a MTLBuffer and the GPU reads the right bytes.
// ---------------------------------------------------------------------------
struct BitonicSortParams {
	int   srcPitch;
	int   dstPitch;
	int   width;
	int   height;
	int   inputOriginX;
	int   inputOriginY;
	int   outputOriginX;
	int   outputOriginY;
	int   outputWidth;
	int   outputHeight;
	int   direction;
	int   ordering;
	float thresholdMin;
	float thresholdMax;
};

// ---------------------------------------------------------------------------
// BPS_MetalDeviceSetup
// ---------------------------------------------------------------------------
PF_Err BPS_MetalDeviceSetup(
	PF_InData               *in_dataP,
	PF_OutData              *out_dataP,
	PF_GPUDeviceSetupExtra  *extraP)
{
	PF_Err err = PF_Err_NONE;

	@autoreleasepool {

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
		if (err) { return err; }

		id<MTLDevice> device = (__bridge id<MTLDevice>)device_info.devicePV;
		if (!device) {
			return PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

		// Validate that the device supports the 32 KB threadgroup memory budget
		// required by the kernel (two arrays of MAX_SIZE=4096 elements, float+uint).
		// 4096*4 + 4096*4 = 32 768 bytes.
		if ([device maxThreadgroupMemoryLength] < 32768u) {
			return PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

		// Compile the kernel from the embedded Metal source string.
		NSString *source = [NSString stringWithCString:kBitonicPixelSorter_Kernel_MetalString
		                                      encoding:NSUTF8StringEncoding];

		NSError *nsErr = nil;
		id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&nsErr];

		// A non-nil error with a non-nil library means warnings only; a nil library
		// is the real error signal (mirrors the SDK ProcAmp NSError2PFErr pattern).
		if (!library) {
			return PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

		id<MTLFunction> sortFunction =
			[library newFunctionWithName:@"BitonicSortKernel"];
		// library is ARC-managed; it is released when it goes out of scope.

		if (!sortFunction) {
			return PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

		nsErr = nil;
		id<MTLComputePipelineState> pipeline =
			[device newComputePipelineStateWithFunction:sortFunction error:&nsErr];
		// sortFunction is ARC-managed; released when it goes out of scope.

		if (!pipeline) {
			return PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

		// After building the pipeline, also validate its static threadgroup
		// memory consumption.  For this kernel the arrays are compile-time sized
		// (no runtime-dynamic allocation), so staticThreadgroupMemoryLength should
		// report 32 768.  Guard against future kernel changes that exceed the budget.
		if ([pipeline staticThreadgroupMemoryLength] > [device maxThreadgroupMemoryLength]) {
			return PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

		// Allocate the PF_Handle to hold MetalGPUData.
		PF_Handle gpu_dataH = handle_suite->host_new_handle(sizeof(MetalGPUData));
		if (!gpu_dataH) {
			return PF_Err_OUT_OF_MEMORY;
		}

		MetalGPUData *metal_dataP =
			reinterpret_cast<MetalGPUData *>(*gpu_dataH);
		std::memset(metal_dataP, 0, sizeof(MetalGPUData));

		// Transfer ownership of the pipeline out of ARC into the PF_Handle.
		// CFBridgingRetain moves the ARC-managed object into manual retain/release:
		// ARC no longer releases the object; we are now responsible.  The matching
		// CFBridgingRelease in setdown restores ARC ownership briefly (to a local
		// variable), which releases on exit.  Do NOT call [pipeline release] after
		// this — CFBridgingRetain already consumed the +1 from newComputePipeline…
		metal_dataP->sort_pipeline_bridge = const_cast<void *>(CFBridgingRetain(pipeline));

		extraP->output->gpu_data = gpu_dataH;
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;

		// Record the device for the GPU status readout in the Effect Controls UI.
		const char *device_name_cstr =
			[device.name UTF8String] ? [device.name UTF8String] : "";
		BPS_RecordGpuDevice(device_info.device_framework,
		                    std::string(device_name_cstr));

	} // @autoreleasepool

	return err;
}

// ---------------------------------------------------------------------------
// BPS_MetalDeviceSetdown
// ---------------------------------------------------------------------------
PF_Err BPS_MetalDeviceSetdown(
	PF_InData                 *in_dataP,
	PF_OutData                *out_dataP,
	PF_GPUDeviceSetdownExtra  *extraP)
{
	PF_Err err = PF_Err_NONE;

	if (extraP->input->what_gpu == PF_GPU_Framework_METAL &&
	    extraP->input->gpu_data)
	{
		@autoreleasepool {
			PF_Handle gpu_dataH =
				reinterpret_cast<PF_Handle>(
					const_cast<void *>(extraP->input->gpu_data));

			MetalGPUData *metal_dataP =
				reinterpret_cast<MetalGPUData *>(*gpu_dataH);

			// Release the pipeline: balance the CFBridgingRetain from setup.
			// CFBridgingRelease transfers the +1 into a local ARC variable that
			// immediately goes out of scope, decrementing the retain count.
			if (metal_dataP->sort_pipeline_bridge) {
				id<MTLComputePipelineState> pipeline =
					(id<MTLComputePipelineState>)CFBridgingRelease(
						metal_dataP->sort_pipeline_bridge);
				(void)pipeline; // ARC releases on scope exit
				metal_dataP->sort_pipeline_bridge = nullptr;
			}

			BPS_ClearGpuDevice();

			AEFX_SuiteScoper<PF_HandleSuite1> handle_suite =
				AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite,
				                                  kPFHandleSuiteVersion1, out_dataP);
			handle_suite->host_dispose_handle(gpu_dataH);
		}
	}

	return err;
}

// ---------------------------------------------------------------------------
// BPS_MetalSmartRender
// ---------------------------------------------------------------------------
PF_Err BPS_MetalSmartRender(
	PF_InData               *in_dataP,
	PF_OutData              *out_dataP,
	PF_EffectWorld          *input_worldP,
	PF_EffectWorld          *output_worldP,
	PF_SmartRenderExtra     *extraP,
	const BitonicSorterParams *paramsP,
	void                    *src_mem,
	void                    *dst_mem,
	int                      srcPitch,
	int                      dstPitch,
	int                      width,
	int                      height,
	int                      inputOriginX,
	int                      inputOriginY,
	int                      outputOriginX,
	int                      outputOriginY,
	int                      outputWidth,
	int                      outputHeight,
	int                      direction,
	int                      ordering,
	int                      lineCount)
{
	PF_Err err = PF_Err_NONE;

	@autoreleasepool {

		AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite =
			AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP, kPFGPUDeviceSuite,
			                                      kPFGPUDeviceSuiteVersion1, out_dataP);

		PF_GPUDeviceInfo device_info;
		AEFX_CLR_STRUCT(device_info);
		ERR(gpu_suite->GetDeviceInfo(in_dataP->effect_ref,
		                             extraP->input->device_index,
		                             &device_info));
		if (err) { return err; }

		// Recover the pipeline from the PF_Handle.
		PF_Handle gpu_dataH =
			reinterpret_cast<PF_Handle>(
				const_cast<void *>(extraP->input->gpu_data));
		if (!gpu_dataH) {
			return PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

		MetalGPUData *metal_dataP =
			reinterpret_cast<MetalGPUData *>(*gpu_dataH);

		// __bridge cast: we do NOT transfer ownership; the pipeline is still owned
		// by the CFBridgingRetain in the PF_Handle.
		id<MTLComputePipelineState> pipeline =
			(__bridge id<MTLComputePipelineState>)metal_dataP->sort_pipeline_bridge;
		if (!pipeline) {
			return PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

		id<MTLDevice>       device = (__bridge id<MTLDevice>)device_info.devicePV;
		id<MTLCommandQueue> queue  =
			(__bridge id<MTLCommandQueue>)device_info.command_queuePV;

		// Build the params struct.  Field order must match BitonicSortParams in
		// the .metal kernel exactly.
		BitonicSortParams metal_params;
		metal_params.srcPitch      = srcPitch;
		metal_params.dstPitch      = dstPitch;
		metal_params.width         = width;
		metal_params.height        = height;
		metal_params.inputOriginX  = inputOriginX;
		metal_params.inputOriginY  = inputOriginY;
		metal_params.outputOriginX = outputOriginX;
		metal_params.outputOriginY = outputOriginY;
		metal_params.outputWidth   = outputWidth;
		metal_params.outputHeight  = outputHeight;
		metal_params.direction     = direction;
		metal_params.ordering      = ordering;
		metal_params.thresholdMin  = paramsP->thresholdMin;
		metal_params.thresholdMax  = paramsP->thresholdMax;

		// Allocate param buffer with MTLResourceStorageModeShared.
		// Shared mode works on both Apple Silicon (unified memory) and Intel Macs
		// and avoids the need for a didModifyRange: call that Managed mode requires.
		id<MTLBuffer> paramBuffer =
			[device newBufferWithBytes:&metal_params
			                   length:sizeof(BitonicSortParams)
			                  options:MTLResourceStorageModeShared];
		if (!paramBuffer) {
			return PF_Err_OUT_OF_MEMORY;
		}

		// AE provides the GPU world data as id<MTLBuffer> cast to void*.
		id<MTLBuffer> src_buffer = (__bridge id<MTLBuffer>)src_mem;
		id<MTLBuffer> dst_buffer = (__bridge id<MTLBuffer>)dst_mem;

		// Encode and commit.
		id<MTLCommandBuffer>        commandBuffer  = [queue commandBuffer];
		id<MTLComputeCommandEncoder> computeEncoder =
			[commandBuffer computeCommandEncoder];

		[computeEncoder setComputePipelineState:pipeline];
		[computeEncoder setBuffer:src_buffer  offset:0 atIndex:0];	// buffer(0) srcTex
		[computeEncoder setBuffer:dst_buffer  offset:0 atIndex:1];	// buffer(1) sortTex
		[computeEncoder setBuffer:paramBuffer offset:0 atIndex:2];	// buffer(2) params

		// One threadgroup per line, 256 threads per threadgroup (matches MAX_THREADS
		// in the kernel and the OpenCL local size).
		MTLSize threadgroupsPerGrid  = MTLSizeMake((NSUInteger)lineCount, 1, 1);
		MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
		[computeEncoder dispatchThreadgroups:threadgroupsPerGrid
		               threadsPerThreadgroup:threadsPerThreadgroup];

		[computeEncoder endEncoding];
		[commandBuffer commit];
		// Do NOT call waitUntilCompleted — AE owns the queue and manages
		// synchronisation between the GPU and the host.  This mirrors the
		// SDK ProcAmp reference which commits without waiting.

		// paramBuffer is ARC-managed; released automatically at end of
		// @autoreleasepool scope.

	} // @autoreleasepool

	return err;
}

#endif // BPS_HAS_METAL
