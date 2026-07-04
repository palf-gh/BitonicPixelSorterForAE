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
#include <cstdint>

#ifdef AE_OS_WIN
	#include <Windows.h>
#endif

// A GPU render path is built when at least one framework backend is compiled in
// (the build system defines BPS_HAS_CUDA / BPS_HAS_OPENCL / BPS_HAS_HLSL /
// BPS_HAS_METAL as the toolchain allows). The CPU path is always present.
#if defined(BPS_HAS_CUDA) || defined(BPS_HAS_OPENCL) || defined(BPS_HAS_HLSL) || defined(BPS_HAS_METAL)
	#define BPS_GPU_ENABLED 1
#endif

// Axis mode: maximum sort-axis length for the group-shared bitonic path.
// Non-axis modes gate on frame width/height instead (paths may be longer and
// use a global-memory domain sort).
#define BPS_GPU_MAX_LINE 4096

// Global data stored in out_data->global_data (AEGP registration for UI).
typedef struct {
	AEGP_PluginID plugin_id;
} BPS_GlobalData;

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
	BPS_THRESHOLD_MIN,		// float slider, trigger key lower bound (%)
	BPS_THRESHOLD_MAX,		// float slider, trigger key upper bound (%)
	BPS_MODE,				// popup: Axis / Free Angle / Rotation / Radial / Swirl / Path
	BPS_ANGLE,				// angle: Free Angle direction
	BPS_CENTER,				// point: Rotation / Radial / Swirl centre
	BPS_SORT_CRITERION,		// popup: sort key
	BPS_SORT_TRIGGER,		// popup: threshold trigger key
	BPS_AFFECT,				// popup: inside/outside thresholds
	BPS_CYCLE,				// angle: per-run cyclic shift
	BPS_SWIRL_AMOUNT,		// angle: Swirl twist (360° = one turn to frame corner)
	BPS_PATH,				// path: layer mask for Path mode
	BPS_PATH_DIRECTION,		// popup: Normal / Tangent
	BPS_CRITERION_SOURCE,	// layer: pixels for sort key (None = effect source)
	BPS_TRIGGER_SOURCE,		// layer: pixels for trigger key (None = effect source)
	BPS_NUM_PARAMS
};

// SmartFX checkout_id values for optional key-source layers. Distinct from
// param indexes so PreRender/SmartRender stay paired even if UI order moves.
enum {
	BPS_CHECKOUT_TRIGGER_SOURCE = 100,
	BPS_CHECKOUT_CRITERION_SOURCE = 101
};

// Retired before first release (was a CW/CCW popup; sign of BPS_SWIRL_AMOUNT
// now selects direction). Do not reuse this ID.
// BPS_SWIRL_DIRECTION = retired

// Current UI/checkout indexes. These are deliberately separate from persisted
// parameter IDs so new controls can appear above older streams without
// renumbering the IDs stored in existing After Effects projects.
//
// UI layout order:
//   GPU Status | Mode | Order | Threshold Min/Max
//   | Trigger | Trigger Source | Criterion | Criterion Source
//   | Affect | Cycle | Direction | Angle | Centre
//   | Swirl Amount | Path | Path Direction
enum {
	BPS_UI_INPUT = 0,
	BPS_UI_GPU_STATUS,
	BPS_UI_MODE,
	BPS_UI_ORDER,
	BPS_UI_THRESHOLD_MIN,
	BPS_UI_THRESHOLD_MAX,
	BPS_UI_SORT_TRIGGER,
	BPS_UI_TRIGGER_SOURCE,
	BPS_UI_SORT_CRITERION,
	BPS_UI_CRITERION_SOURCE,
	BPS_UI_AFFECT,
	BPS_UI_CYCLE,
	BPS_UI_DIRECTION,
	BPS_UI_ANGLE,
	BPS_UI_CENTER,
	BPS_UI_SWIRL_AMOUNT,
	BPS_UI_PATH,
	BPS_UI_PATH_DIRECTION,
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
	BPS_MODE_RADIAL,
	BPS_MODE_SWIRL,
	BPS_MODE_PATH
};

// Path sort-axis popup choices (1-based).
enum {
	BPS_PATH_DIR_NORMAL = 1,
	BPS_PATH_DIR_TANGENT
};

// Forward declaration for Path-mode polyline samples (defined in PathGeometry.h).
struct BpsPathSample;

// Pixel-owned transformed path record. `pixelIndex` is y * frameWidth + x in
// render-space layer coordinates; records are sorted by line, posKey, pixel.
struct BpsMappedPixelRecord {
	float posKey;
	std::uint32_t pixelIndex;
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
	BPS_CRITERION_RGB_MAXIMUM,
	BPS_CRITERION_RED_CHANNEL,
	BPS_CRITERION_GREEN_CHANNEL,
	BPS_CRITERION_BLUE_CHANNEL,
	BPS_CRITERION_ALPHA_CHANNEL,
	BPS_CRITERION_HUE,
	BPS_CRITERION_SATURATION
};

// Affect popup choices (1-based).
enum {
	BPS_AFFECT_INSIDE_THRESHOLDS = 1,
	BPS_AFFECT_OUTSIDE_THRESHOLDS
};

// Parameter defaults.
#define BPS_DIRECTION_DFLT		BPS_DIR_HORIZONTAL
#define BPS_MODE_DFLT			BPS_MODE_AXIS
#define BPS_ORDER_DFLT			BPS_ORDER_ASCENDING
#define BPS_ANGLE_DFLT			0.0
#define BPS_CENTER_X_DFLT		50.0
#define BPS_CENTER_Y_DFLT		50.0
#define BPS_SORT_CRITERION_DFLT	BPS_CRITERION_LUMINANCE
#define BPS_SORT_TRIGGER_DFLT	BPS_CRITERION_LUMINANCE
#define BPS_AFFECT_DFLT		BPS_AFFECT_INSIDE_THRESHOLDS
#define BPS_CYCLE_DFLT			0.0
#define BPS_THRESHOLD_MIN_DFLT	40.0	// percent (upstream default 0.4)
#define BPS_THRESHOLD_MAX_DFLT	60.0	// percent (upstream default 0.6)
// Swirl amount is an AE angle: 360° = one full turn to the farthest frame corner.
// Sign selects direction (positive = CW, negative = CCW).
#define BPS_SWIRL_AMOUNT_DFLT	360.0
#define BPS_PATH_DIRECTION_DFLT	BPS_PATH_DIR_TANGENT
// Internal |swirlK| below this is treated as pure Radial (no twist).
#define BPS_SWIRL_K_EPS			1.0e-8f

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
	A_long	trigger;		// BPS_CRITERION_*
	A_long	affect;			// BPS_AFFECT_*
	float	thresholdMin;	// normalised 0..1
	float	thresholdMax;	// normalised 0..1
	float	cycleRadians;
	float	cycleDegrees;
	float	angleRadians;
	float	angleCos;
	float	angleSin;
	float	centerX;		// render-space layer coordinate
	float	centerY;		// render-space layer coordinate
	float	downsampleX;
	float	downsampleY;
	float	swirlK;			// signed twist (radians per render-space pixel of radius)
	A_long	swirlLineMin;	// Reserved for old analytic swirl; pixel map uses phase lines
	A_long	pathDirection;	// BPS_PATH_DIR_*
	A_long	pathClosed;		// 1 = closed path (arc length wraps), 0 = open (extended)
	float	pathLength;		// arc length of the base path (before open-end extension)
	A_long	pathSMin;		// floor(min arc length) over the sampled polyline
	A_long	pathNMin;		// floor(-normal extent); signed-distance domain origin
	A_long	pathSampleCount;
	const BpsPathSample *pathSamples; // host-owned; not part of GPU constants
	A_long	mappedRecordCount;
	A_long	mappedWorkItemCount;
	const BpsMappedPixelRecord *mappedRecords; // host-owned; GPU uploads separately
	const std::uint32_t *mappedLineOffsets;
	const std::uint32_t *mappedWorkOffsets;
	A_long	freePMin;
	A_long	freeQMin;
	A_long	freeLineLength;
	A_long	freeLineCount;
	A_long	radialLength;
	A_long	radialLineCount;
	A_long	maxLineLength;
	A_long	domainLineCount;
	A_long	domainMaxLineLength;
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
	PF_EffectWorld		*criterion_worldP,
	PF_EffectWorld		*trigger_worldP,
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
	PF_EffectWorld		*criterion_worldP,
	PF_EffectWorld		*trigger_worldP,
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
