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

#include <cstdint>
#include <cstring>
#include <new>

// ---------------------------------------------------------------------------
// Per-device GPU data stored in the PF_Handle allocated at setup.
// The pipeline is stored as a bridged void* (CFBridgingRetain) so it survives
// in plain malloc'd memory without ARC tracking.
// ---------------------------------------------------------------------------
struct MetalGPUData {
	void *sort_pipeline_bridge;			// CFBridgingRetain'd id<MTLComputePipelineState>
	void *domain_sort_pipeline_bridge;
	void *apply_domain_pipeline_bridge;
	void *copy_pipeline_bridge;
	void *mapped_sort_pipeline_bridge;
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
	int   mode;
	int   direction;
	int   ordering;
	int   criterion;
	int   trigger;
	int   affect;
	float cycleDegrees;
	int   lineCount;
	int   freePMin;
	int   freeQMin;
	int   freeLineLength;
	int   radialLength;
	int   domainStride;
	float thresholdMin;
	float thresholdMax;
	float angleCos;
	float angleSin;
	float centerX;
	float centerY;
	float swirlK;
	int   swirlLineMin;
	int   pathDirection;
	int   pathSMin;
	int   pathNMin;
	int   pathSampleCount;
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

		NSString *kernel_names[5] = {
			@"BitonicSortKernel",
			@"BitonicSortDomainKernel",
			@"BitonicApplyDomainKernel",
			@"BitonicCopyInputKernel",
			@"BitonicSortMappedKernel"
		};
		id<MTLComputePipelineState> pipelines[5] = {nil, nil, nil, nil, nil};
		for (int i = 0; i < 5; ++i) {
			id<MTLFunction> function = [library newFunctionWithName:kernel_names[i]];
			if (!function) {
				return PF_Err_INTERNAL_STRUCT_DAMAGED;
			}
			nsErr = nil;
			pipelines[i] = [device newComputePipelineStateWithFunction:function error:&nsErr];
			if (!pipelines[i]) {
				return PF_Err_INTERNAL_STRUCT_DAMAGED;
			}
			// Sort kernels use 32 KB of threadgroup memory; apply does not, but
			// the same device budget check is harmless.
			if ([pipelines[i] staticThreadgroupMemoryLength] >
				[device maxThreadgroupMemoryLength]) {
				return PF_Err_INTERNAL_STRUCT_DAMAGED;
			}
		}

		id<MTLComputePipelineState> sort_pipeline = pipelines[0];
		id<MTLComputePipelineState> domain_sort_pipeline = pipelines[1];
		id<MTLComputePipelineState> apply_domain_pipeline = pipelines[2];
		id<MTLComputePipelineState> copy_pipeline = pipelines[3];
		id<MTLComputePipelineState> mapped_sort_pipeline = pipelines[4];

		// Allocate the PF_Handle to hold MetalGPUData.
		PF_Handle gpu_dataH = handle_suite->host_new_handle(sizeof(MetalGPUData));
		if (!gpu_dataH) {
			return PF_Err_OUT_OF_MEMORY;
		}

		MetalGPUData *metal_dataP =
			reinterpret_cast<MetalGPUData *>(*gpu_dataH);
		std::memset(metal_dataP, 0, sizeof(MetalGPUData));

		// Transfer ownership of the pipelines out of ARC into the PF_Handle.
		// CFBridgingRetain moves the ARC-managed object into manual retain/release:
		// ARC no longer releases the object; we are now responsible.  The matching
		// CFBridgingRelease in setdown restores ARC ownership briefly (to a local
		// variable), which releases on exit.
		metal_dataP->sort_pipeline_bridge =
			const_cast<void *>(CFBridgingRetain(sort_pipeline));
		metal_dataP->domain_sort_pipeline_bridge =
			const_cast<void *>(CFBridgingRetain(domain_sort_pipeline));
		metal_dataP->apply_domain_pipeline_bridge =
			const_cast<void *>(CFBridgingRetain(apply_domain_pipeline));
		metal_dataP->copy_pipeline_bridge =
			const_cast<void *>(CFBridgingRetain(copy_pipeline));
		metal_dataP->mapped_sort_pipeline_bridge =
			const_cast<void *>(CFBridgingRetain(mapped_sort_pipeline));

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

			// Release pipelines: balance the CFBridgingRetain from setup.
			// CFBridgingRelease transfers the +1 into a local ARC variable that
			// immediately goes out of scope, decrementing the retain count.
			if (metal_dataP->mapped_sort_pipeline_bridge) {
				id<MTLComputePipelineState> pipeline =
					(id<MTLComputePipelineState>)CFBridgingRelease(
						metal_dataP->mapped_sort_pipeline_bridge);
				(void)pipeline;
				metal_dataP->mapped_sort_pipeline_bridge = nullptr;
			}
			if (metal_dataP->copy_pipeline_bridge) {
				id<MTLComputePipelineState> pipeline =
					(id<MTLComputePipelineState>)CFBridgingRelease(
						metal_dataP->copy_pipeline_bridge);
				(void)pipeline;
				metal_dataP->copy_pipeline_bridge = nullptr;
			}
			if (metal_dataP->apply_domain_pipeline_bridge) {
				id<MTLComputePipelineState> pipeline =
					(id<MTLComputePipelineState>)CFBridgingRelease(
						metal_dataP->apply_domain_pipeline_bridge);
				(void)pipeline;
				metal_dataP->apply_domain_pipeline_bridge = nullptr;
			}
			if (metal_dataP->domain_sort_pipeline_bridge) {
				id<MTLComputePipelineState> pipeline =
					(id<MTLComputePipelineState>)CFBridgingRelease(
						metal_dataP->domain_sort_pipeline_bridge);
				(void)pipeline;
				metal_dataP->domain_sort_pipeline_bridge = nullptr;
			}
			if (metal_dataP->sort_pipeline_bridge) {
				id<MTLComputePipelineState> pipeline =
					(id<MTLComputePipelineState>)CFBridgingRelease(
						metal_dataP->sort_pipeline_bridge);
				(void)pipeline;
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

		// Recover pipelines from the PF_Handle.
		PF_Handle gpu_dataH =
			reinterpret_cast<PF_Handle>(
				const_cast<void *>(extraP->input->gpu_data));
		if (!gpu_dataH) {
			return PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

		MetalGPUData *metal_dataP =
			reinterpret_cast<MetalGPUData *>(*gpu_dataH);

		// __bridge cast: we do NOT transfer ownership; pipelines remain owned
		// by the CFBridgingRetain in the PF_Handle.
		id<MTLComputePipelineState> sort_pipeline =
			(__bridge id<MTLComputePipelineState>)metal_dataP->sort_pipeline_bridge;
		id<MTLComputePipelineState> domain_sort_pipeline =
			(__bridge id<MTLComputePipelineState>)metal_dataP->domain_sort_pipeline_bridge;
		id<MTLComputePipelineState> apply_domain_pipeline =
			(__bridge id<MTLComputePipelineState>)metal_dataP->apply_domain_pipeline_bridge;
		id<MTLComputePipelineState> copy_pipeline =
			(__bridge id<MTLComputePipelineState>)metal_dataP->copy_pipeline_bridge;
		id<MTLComputePipelineState> mapped_sort_pipeline =
			(__bridge id<MTLComputePipelineState>)metal_dataP->mapped_sort_pipeline_bridge;
		if (!sort_pipeline || !domain_sort_pipeline || !apply_domain_pipeline ||
			!copy_pipeline || !mapped_sort_pipeline) {
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
		metal_params.mode          = (int)paramsP->mode;
		metal_params.direction     = direction;
		metal_params.ordering      = ordering;
		metal_params.criterion     = (int)paramsP->criterion;
		metal_params.trigger       = (int)paramsP->trigger;
		metal_params.affect        = (int)paramsP->affect;
		metal_params.cycleDegrees  = paramsP->cycleDegrees;
		metal_params.lineCount     = lineCount;
		metal_params.freePMin      = (int)paramsP->freePMin;
		metal_params.freeQMin      = (int)paramsP->freeQMin;
		metal_params.freeLineLength = (int)paramsP->freeLineLength;
		metal_params.radialLength  = (paramsP->mode == BPS_MODE_PATH)
			? (int)paramsP->domainMaxLineLength
			: (int)paramsP->radialLength;
		{
			int stride = (int)paramsP->domainMaxLineLength;
			if (stride > 1) {
				unsigned int v = (unsigned int)stride - 1u;
				v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
				stride = (int)(v + 1u);
			}
			stride *= 2; // 2x: guarantees spanStart + sortSize <= domainStride
			metal_params.domainStride = stride;
		}
		metal_params.thresholdMin  = paramsP->thresholdMin;
		metal_params.thresholdMax  = paramsP->thresholdMax;
		metal_params.angleCos      = paramsP->angleCos;
		metal_params.angleSin      = paramsP->angleSin;
		metal_params.centerX       = paramsP->centerX;
		metal_params.centerY       = paramsP->centerY;
		metal_params.swirlK        = paramsP->swirlK;
		metal_params.swirlLineMin  = (int)paramsP->swirlLineMin;
		metal_params.pathDirection = (int)paramsP->pathDirection;
		metal_params.pathSMin      = (int)paramsP->pathSMin;
		metal_params.pathNMin      = (int)paramsP->pathNMin;
		metal_params.pathSampleCount = (int)paramsP->pathSampleCount;

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

		const NSUInteger pathCount =
			paramsP->pathSampleCount > 0
				? (NSUInteger)paramsP->pathSampleCount
				: 1u;
		const NSUInteger pathBytes = pathCount * sizeof(float) * 5u;
		id<MTLBuffer> pathBuffer =
			[device newBufferWithLength:pathBytes
			                    options:MTLResourceStorageModeShared];
		if (!pathBuffer) {
			return PF_Err_OUT_OF_MEMORY;
		}
		std::memset([pathBuffer contents], 0, pathBytes);
		if (paramsP->pathSampleCount > 0 && paramsP->pathSamples) {
			std::memcpy([pathBuffer contents], paramsP->pathSamples,
						(size_t)paramsP->pathSampleCount * sizeof(float) * 5u);
		}

		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

		if (paramsP->mode == BPS_MODE_AXIS) {
			id<MTLComputeCommandEncoder> computeEncoder =
				[commandBuffer computeCommandEncoder];

			[computeEncoder setComputePipelineState:sort_pipeline];
			[computeEncoder setBuffer:src_buffer  offset:0 atIndex:0];
			[computeEncoder setBuffer:dst_buffer  offset:0 atIndex:1];
			[computeEncoder setBuffer:paramBuffer offset:0 atIndex:2];
			[computeEncoder setBuffer:pathBuffer  offset:0 atIndex:3];

			MTLSize threadgroupsPerGrid  = MTLSizeMake((NSUInteger)lineCount, 1, 1);
			MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
			[computeEncoder dispatchThreadgroups:threadgroupsPerGrid
			               threadsPerThreadgroup:threadsPerThreadgroup];
			[computeEncoder endEncoding];
		} else if (paramsP->mode == BPS_MODE_PATH) {
			id<MTLComputeCommandEncoder> copyEncoder =
				[commandBuffer computeCommandEncoder];
			[copyEncoder setComputePipelineState:copy_pipeline];
			[copyEncoder setBuffer:src_buffer  offset:0 atIndex:0];
			[copyEncoder setBuffer:dst_buffer  offset:0 atIndex:1];
			[copyEncoder setBuffer:paramBuffer offset:0 atIndex:2];
			const NSUInteger copyGroupsX =
				((NSUInteger)outputWidth + 15u) / 16u;
			const NSUInteger copyGroupsY =
				((NSUInteger)outputHeight + 15u) / 16u;
			[copyEncoder dispatchThreadgroups:MTLSizeMake(copyGroupsX, copyGroupsY, 1)
			            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
			[copyEncoder endEncoding];

			if (paramsP->mappedRecordCount > 0 &&
				paramsP->mappedWorkItemCount > 0 &&
				paramsP->mappedRecords &&
				paramsP->mappedLineOffsets &&
				paramsP->mappedWorkOffsets) {
				const NSUInteger recordsBytes =
					(NSUInteger)paramsP->mappedRecordCount *
					(NSUInteger)sizeof(BpsMappedPixelRecord);
				const NSUInteger offsetsBytes =
					((NSUInteger)lineCount + 1u) *
					(NSUInteger)sizeof(std::uint32_t);
				const NSUInteger domainBytes =
					(NSUInteger)paramsP->mappedWorkItemCount *
					(NSUInteger)sizeof(unsigned int);
				const NSUInteger keysBytes =
					(NSUInteger)paramsP->mappedWorkItemCount *
					(NSUInteger)sizeof(float);

				id<MTLBuffer> recordsBuffer =
					[device newBufferWithBytes:paramsP->mappedRecords
					                    length:recordsBytes
					                   options:MTLResourceStorageModeShared];
				id<MTLBuffer> lineOffsetsBuffer =
					[device newBufferWithBytes:paramsP->mappedLineOffsets
					                    length:offsetsBytes
					                   options:MTLResourceStorageModeShared];
				id<MTLBuffer> workOffsetsBuffer =
					[device newBufferWithBytes:paramsP->mappedWorkOffsets
					                    length:offsetsBytes
					                   options:MTLResourceStorageModeShared];
				id<MTLBuffer> domainBuffer =
					[device newBufferWithLength:domainBytes
					                    options:MTLResourceStorageModePrivate];
				id<MTLBuffer> keysBuffer =
					[device newBufferWithLength:keysBytes
					                    options:MTLResourceStorageModePrivate];
				if (!recordsBuffer || !lineOffsetsBuffer || !workOffsetsBuffer ||
					!domainBuffer || !keysBuffer) {
					return PF_Err_OUT_OF_MEMORY;
				}

				id<MTLComputeCommandEncoder> mappedEncoder =
					[commandBuffer computeCommandEncoder];
				[mappedEncoder setComputePipelineState:mapped_sort_pipeline];
				[mappedEncoder setBuffer:src_buffer         offset:0 atIndex:0];
				[mappedEncoder setBuffer:dst_buffer         offset:0 atIndex:1];
				[mappedEncoder setBuffer:domainBuffer      offset:0 atIndex:2];
				[mappedEncoder setBuffer:keysBuffer        offset:0 atIndex:3];
				[mappedEncoder setBuffer:paramBuffer       offset:0 atIndex:4];
				[mappedEncoder setBuffer:recordsBuffer     offset:0 atIndex:5];
				[mappedEncoder setBuffer:lineOffsetsBuffer offset:0 atIndex:6];
				[mappedEncoder setBuffer:workOffsetsBuffer offset:0 atIndex:7];
				[mappedEncoder dispatchThreadgroups:MTLSizeMake((NSUInteger)lineCount, 1, 1)
				              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
				[mappedEncoder endEncoding];
			}
		} else {
			const int domainStride = metal_params.domainStride;
			if (domainStride <= 0 || outputWidth <= 0 || outputHeight <= 0) {
				return PF_Err_NONE;
			}

			const NSUInteger domainCount =
				(NSUInteger)lineCount * (NSUInteger)domainStride;
			const NSUInteger domainBytes = domainCount * sizeof(unsigned int);
			const NSUInteger keysBytes = domainCount * sizeof(float);
			id<MTLBuffer> domainBuffer =
				[device newBufferWithLength:domainBytes
				                    options:MTLResourceStorageModePrivate];
			id<MTLBuffer> keysBuffer =
				[device newBufferWithLength:keysBytes
				                    options:MTLResourceStorageModePrivate];
			if (!domainBuffer || !keysBuffer) {
				return PF_Err_OUT_OF_MEMORY;
			}

			// Clear domain indices to 0xFFFFFFFF via a small shared staging buffer.
			id<MTLBuffer> clearBuffer =
				[device newBufferWithLength:domainBytes
				                    options:MTLResourceStorageModeShared];
			if (!clearBuffer) {
				return PF_Err_OUT_OF_MEMORY;
			}
			std::memset([clearBuffer contents], 0xFF, domainBytes);

			id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
			[blitEncoder copyFromBuffer:clearBuffer
			               sourceOffset:0
			                   toBuffer:domainBuffer
			          destinationOffset:0
			                       size:domainBytes];
			[blitEncoder endEncoding];

			id<MTLComputeCommandEncoder> sortEncoder =
				[commandBuffer computeCommandEncoder];
			[sortEncoder setComputePipelineState:domain_sort_pipeline];
			[sortEncoder setBuffer:src_buffer   offset:0 atIndex:0];
			[sortEncoder setBuffer:domainBuffer offset:0 atIndex:1];
			[sortEncoder setBuffer:keysBuffer   offset:0 atIndex:2];
			[sortEncoder setBuffer:paramBuffer  offset:0 atIndex:3];
			[sortEncoder setBuffer:pathBuffer   offset:0 atIndex:4];
			[sortEncoder dispatchThreadgroups:MTLSizeMake((NSUInteger)lineCount, 1, 1)
			            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
			[sortEncoder endEncoding];

			id<MTLComputeCommandEncoder> applyEncoder =
				[commandBuffer computeCommandEncoder];
			[applyEncoder setComputePipelineState:apply_domain_pipeline];
			[applyEncoder setBuffer:src_buffer   offset:0 atIndex:0];
			[applyEncoder setBuffer:dst_buffer   offset:0 atIndex:1];
			[applyEncoder setBuffer:domainBuffer offset:0 atIndex:2];
			[applyEncoder setBuffer:paramBuffer  offset:0 atIndex:3];
			[applyEncoder setBuffer:pathBuffer   offset:0 atIndex:4];
			const NSUInteger applyGroupsX =
				((NSUInteger)outputWidth + 15u) / 16u;
			const NSUInteger applyGroupsY =
				((NSUInteger)outputHeight + 15u) / 16u;
			[applyEncoder dispatchThreadgroups:MTLSizeMake(applyGroupsX, applyGroupsY, 1)
			             threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
			[applyEncoder endEncoding];
		}

		[commandBuffer commit];
		// Do NOT call waitUntilCompleted — AE owns the queue and manages
		// synchronisation between the GPU and the host.  This mirrors the
		// SDK ProcAmp reference which commits without waiting.

	} // @autoreleasepool

	return err;
}

#endif // BPS_HAS_METAL
