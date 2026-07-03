/*
	BitonicPixelSorter.h

	Self-contained After Effects port of ruccho/BitonicPixelSorter (Unity, MIT).
	https://github.com/ruccho/BitonicPixelSorter

	This plugin references only the Adobe After Effects SDK. Localisation uses
	the header-only AELocalise helper under Localise/, so the project builds
	anywhere the SDK example tree is present without depending on Palf_Plugins.
*/

#pragma once
#ifndef BITONIC_PIXEL_SORTER_H
#define BITONIC_PIXEL_SORTER_H

#include "BitonicPixelSorter_Target.h"

#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectSuites.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AEGP_SuiteHandler.h"
#include "String_Utils.h"
#include "AEFX_SuiteHelper.h"
#include "Smart_Utils.h"
#include "AE_EffectGPUSuites.h"
#include "PrSDKAESupport.h"

#include <cmath>

#ifdef AE_OS_WIN
	#include <Windows.h>
#endif

// A GPU render path is built when at least one framework backend is compiled in
// (the build system defines BPS_HAS_CUDA / BPS_HAS_OPENCL / BPS_HAS_HLSL /
// BPS_HAS_METAL as the toolchain allows). The CPU path is always present.
#if defined(BPS_HAS_CUDA) || defined(BPS_HAS_OPENCL) || defined(BPS_HAS_HLSL) || defined(BPS_HAS_METAL)
	#define BPS_GPU_ENABLED 1
#endif

// Maximum sort-axis length the GPU bitonic path supports (group-shared memory
// limit). Longer lines fall back to the CPU path.
#define BPS_GPU_MAX_LINE 4096

//-----------------------------------------------------------------------------
// Parameter IDs.
// These are persisted by After Effects: never renumber an existing ID and never
// reuse a retired ID. Append new IDs before BPS_NUM_PARAMS.
//-----------------------------------------------------------------------------
enum {
	BPS_INPUT = 0,
	BPS_GPU_STATUS,			// custom UI: GPU acceleration status (read-only)
	BPS_DIRECTION,			// popup: Horizontal / Vertical
	BPS_ORDER,				// popup: Ascending / Descending
	BPS_THRESHOLD_MIN,		// float slider, selected key lower bound (%)
	BPS_THRESHOLD_MAX,		// float slider, selected key upper bound (%)
	BPS_MODE,				// popup: Axis / Free Angle / Rotation / Radial
	BPS_ANGLE,				// angle: Free Angle direction
	BPS_CENTER,				// point: Rotation / Radial centre
	BPS_SORT_CRITERION,		// popup: sort/threshold key
	BPS_NUM_PARAMS
};

// Current UI/checkout indexes. These are deliberately separate from persisted
// parameter IDs so new controls can appear above older streams without
// renumbering the IDs stored in existing After Effects projects.
enum {
	BPS_UI_INPUT = 0,
	BPS_UI_MODE,
	BPS_UI_GPU_STATUS,
	BPS_UI_DIRECTION,
	BPS_UI_ANGLE,
	BPS_UI_CENTER,
	BPS_UI_SORT_CRITERION,
	BPS_UI_ORDER,
	BPS_UI_THRESHOLD_MIN,
	BPS_UI_THRESHOLD_MAX,
	BPS_UI_NUM_PARAMS
};

// Direction popup choices (1-based, as AE popups are).
enum {
	BPS_DIR_HORIZONTAL = 1,	// sort along X within each row
	BPS_DIR_VERTICAL		// sort along Y within each column
};

// Mode popup choices (1-based).
enum {
	BPS_MODE_AXIS = 1,
	BPS_MODE_FREE_ANGLE,
	BPS_MODE_ROTATION,
	BPS_MODE_RADIAL
};

// Order popup choices (1-based).
enum {
	BPS_ORDER_ASCENDING = 1,
	BPS_ORDER_DESCENDING
};

// Sort criterion popup choices (1-based).
enum {
	BPS_CRITERION_LUMINANCE = 1,
	BPS_CRITERION_RGB_AVERAGE,
	BPS_CRITERION_RGB_PRODUCT,
	BPS_CRITERION_RGB_MINIMUM,
	BPS_CRITERION_RGB_MAXIMUM
};

// Parameter defaults.
#define BPS_DIRECTION_DFLT		BPS_DIR_HORIZONTAL
#define BPS_MODE_DFLT			BPS_MODE_AXIS
#define BPS_ORDER_DFLT			BPS_ORDER_ASCENDING
#define BPS_ANGLE_DFLT			0.0
#define BPS_CENTER_X_DFLT		50.0
#define BPS_CENTER_Y_DFLT		50.0
#define BPS_SORT_CRITERION_DFLT	BPS_CRITERION_LUMINANCE
#define BPS_THRESHOLD_MIN_DFLT	40.0	// percent (upstream default 0.4)
#define BPS_THRESHOLD_MAX_DFLT	60.0	// percent (upstream default 0.6)

// Effect Controls custom UI for the GPU status readout.
#define BPS_GPU_STATUS_UI_WIDTH		280
// Two lines at the default Drawbot font size (status label + framework/device).
#define BPS_GPU_STATUS_UI_HEIGHT	40

//-----------------------------------------------------------------------------
// Resolved parameters, computed at PreRender and consumed at (Smart)Render.
//-----------------------------------------------------------------------------
typedef struct {
	A_long	mode;			// BPS_MODE_*
	A_long	direction;		// BPS_DIR_HORIZONTAL or BPS_DIR_VERTICAL
	A_long	ascending;		// 1 = ascending, 0 = descending
	A_long	criterion;		// BPS_CRITERION_*
	float	thresholdMin;	// normalised 0..1
	float	thresholdMax;	// normalised 0..1
	float	angleRadians;
	float	angleCos;
	float	angleSin;
	float	centerX;		// render-space layer coordinate
	float	centerY;		// render-space layer coordinate
	float	downsampleX;
	float	downsampleY;
	A_long	freePMin;
	A_long	freeQMin;
	A_long	freeLineLength;
	A_long	freeLineCount;
	A_long	radialLength;
	A_long	radialLineCount;
	A_long	maxLineLength;
} BitonicSorterParams;

//-----------------------------------------------------------------------------
// CPU render entry (implemented in BitonicPixelSorter_CPU.cpp).
// Sorts contiguous in-threshold spans of each line by the selected key. No size limit,
// supports 8/16/32-bit. Acts as the GPU fallback and the correctness oracle.
//-----------------------------------------------------------------------------
PF_Err BPS_SortImageCPU(
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_PixelFormat		pixel_format,
	PF_EffectWorld		*input_worldP,
	PF_EffectWorld		*output_worldP,
	const BitonicSorterParams *paramsP);

//-----------------------------------------------------------------------------
// GPU entries (implemented in BitonicPixelSorter_GPU.cpp). Always declared; the
// bodies are no-ops unless a backend is compiled in. GPU worlds are BGRA128.
//-----------------------------------------------------------------------------
PF_Err BPS_GPUDeviceSetup(
	PF_InData *in_dataP, PF_OutData *out_dataP, PF_GPUDeviceSetupExtra *extraP);

PF_Err BPS_GPUDeviceSetdown(
	PF_InData *in_dataP, PF_OutData *out_dataP, PF_GPUDeviceSetdownExtra *extraP);

PF_Err BPS_SmartRenderGPU(
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_PixelFormat		pixel_format,
	PF_EffectWorld		*input_worldP,
	PF_EffectWorld		*output_worldP,
	PF_SmartRenderExtra	*extraP,
	const BitonicSorterParams *paramsP);

//-----------------------------------------------------------------------------
// Custom UI (implemented in BitonicPixelSorter_UI.cpp).
//-----------------------------------------------------------------------------
PF_Err BPS_HandleEvent(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	PF_EventExtra	*event_extra);

PF_Err BPS_UpdateParamsUI(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output);

extern "C" {

	DllExport
	PF_Err
	EffectMain(
		PF_Cmd			cmd,
		PF_InData		*in_data,
		PF_OutData		*out_data,
		PF_ParamDef		*params[],
		PF_LayerDef		*output,
		void			*extra);

}

#endif // BITONIC_PIXEL_SORTER_H
