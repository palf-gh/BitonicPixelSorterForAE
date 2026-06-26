#pragma once

/*
	BPS_MetalBackend.h

	C++ interface declarations for the Objective-C++ Metal backend
	(Source/BitonicPixelSorter_Metal.mm).  Guarded by BPS_HAS_METAL so
	including this header on non-Apple / non-Metal builds is a no-op.

	Also declares BPS_RecordGpuDevice / BPS_ClearGpuDevice at file scope so
	the .mm translation unit can call them (their single definition lives in
	BitonicPixelSorter_GPU.cpp; moving them out of the anonymous namespace
	makes them linkable from Objective-C++ without duplication).
*/

#if defined(BPS_HAS_METAL)

#include "BitonicPixelSorter.h"		// PF_* types, BitonicSorterParams
#include "AE_EffectGPUSuites.h"		// PF_GPUDeviceSetupExtra / SetdownExtra

#include <string>

// ---------------------------------------------------------------------------
// GPU device state helpers — defined in BitonicPixelSorter_GPU.cpp at file
// scope (moved out of anonymous namespace for cross-TU linkage).
// ---------------------------------------------------------------------------
void BPS_RecordGpuDevice(PF_GPU_Framework framework, const std::string &device_name);
void BPS_ClearGpuDevice();

// ---------------------------------------------------------------------------
// Metal backend entry points — implemented in BitonicPixelSorter_Metal.mm.
// Called from BitonicPixelSorter_GPU.cpp under the BPS_HAS_METAL guard.
// ---------------------------------------------------------------------------
PF_Err BPS_MetalDeviceSetup(
	PF_InData               *in_dataP,
	PF_OutData              *out_dataP,
	PF_GPUDeviceSetupExtra  *extraP);

PF_Err BPS_MetalDeviceSetdown(
	PF_InData                 *in_dataP,
	PF_OutData                *out_dataP,
	PF_GPUDeviceSetdownExtra  *extraP);

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
	int                      lineCount);

#endif // BPS_HAS_METAL
