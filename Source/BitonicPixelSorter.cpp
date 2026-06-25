/*
	BitonicPixelSorter.cpp

	Effect entry point, global/param setup and the Smart Render dispatch for the
	Bitonic Pixel Sorter. Self-contained: references only the Adobe SDK plus the
	vendored AELocalise.h.

	Phase 1 implements the CPU render path (the correctness oracle and GPU
	fallback). The GPU path (CUDA / OpenCL / Metal / DirectX) is layered on in a
	later phase via the GPU command set.
*/

#include "BitonicPixelSorter.h"

#include <new>
#include <string>

// Localisation: include keys, then the per-language string tables, then the
// AELocalise engine (which expects the LocKey type and language namespaces).
#include "Localise/LocKeys.h"
#include "Localise/Strings_en_US.h"
#include "Localise/Strings_ja_JP.h"
#include "Localise/Strings_zh_CN.h"
#include "Localise/Strings_ko_KR.h"
#include "vendor/palf/AELocalise.h"

//-----------------------------------------------------------------------------
static PF_Err
About(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_SPRINTF(out_data->return_msg, "%s, v%d.%d\r%s",
			   NAME, MAJOR_VERSION, MINOR_VERSION, DESCRIPTION);
	return PF_Err_NONE;
}

//-----------------------------------------------------------------------------
static PF_Err
GlobalSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	out_data->my_version	= BPS_VERSION_PACKED;
	out_data->out_flags		= OUT_FLAGS;
	out_data->out_flags2	= OUT_FLAGS2;

#if defined(BPS_GPU_ENABLED)
	// Advertise GPU rendering for non-Premiere hosts (After Effects). Premiere
	// uses its own pixel-format negotiation and is not targeted here.
	if (in_data->appl_id != 'PrMr') {
		out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	#if defined(BPS_HAS_HLSL)
		out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING;
	#endif
	}
#endif

	return PF_Err_NONE;
}

//-----------------------------------------------------------------------------
static PF_Err
ParamsSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_Err		err = PF_Err_NONE;
	PF_ParamDef	def;

	// Direction (Horizontal / Vertical). Copy localised strings into std::string
	// first: AELocalise::GetStringForAE returns a pointer into a shared thread-
	// local buffer that the next call overwrites, so we must not hold two live.
	AEFX_CLR_STRUCT(def);
	{
		std::string name  = AELocalise::GetStringForAE(LocKey::STR_DIRECTION_NAME, in_data);
		std::string items = AELocalise::GetStringForAE(LocKey::STR_DIRECTION_ITEMS, in_data);
		PF_ADD_POPUP(name.c_str(), 2, BPS_DIRECTION_DFLT, items.c_str(), BPS_DIRECTION);
	}

	// Order (Ascending / Descending).
	AEFX_CLR_STRUCT(def);
	{
		std::string name  = AELocalise::GetStringForAE(LocKey::STR_ORDER_NAME, in_data);
		std::string items = AELocalise::GetStringForAE(LocKey::STR_ORDER_ITEMS, in_data);
		PF_ADD_POPUP(name.c_str(), 2, BPS_ORDER_DFLT, items.c_str(), BPS_ORDER);
	}

	// Threshold Min (brightness lower bound, shown as a percentage).
	AEFX_CLR_STRUCT(def);
	{
		std::string name = AELocalise::GetStringForAE(LocKey::STR_THRESHOLD_MIN, in_data);
		PF_ADD_FLOAT_SLIDERX(name.c_str(), 0, 100, 0, 100, BPS_THRESHOLD_MIN_DFLT,
							 1, PF_ValueDisplayFlag_PERCENT, 0, BPS_THRESHOLD_MIN);
	}

	// Threshold Max (brightness upper bound, shown as a percentage).
	AEFX_CLR_STRUCT(def);
	{
		std::string name = AELocalise::GetStringForAE(LocKey::STR_THRESHOLD_MAX, in_data);
		PF_ADD_FLOAT_SLIDERX(name.c_str(), 0, 100, 0, 100, BPS_THRESHOLD_MAX_DFLT,
							 1, PF_ValueDisplayFlag_PERCENT, 0, BPS_THRESHOLD_MAX);
	}

	out_data->num_params = BPS_NUM_PARAMS;
	return err;
}

//-----------------------------------------------------------------------------
static void
DisposePreRenderData(void *pre_render_dataPV)
{
	if (pre_render_dataPV) {
		BitonicSorterParams *infoP = reinterpret_cast<BitonicSorterParams *>(pre_render_dataPV);
		delete infoP;
	}
}

//-----------------------------------------------------------------------------
static PF_Err
PreRender(
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_PreRenderExtra	*extraP)
{
	PF_Err				err = PF_Err_NONE;
	PF_CheckoutResult	in_result;
	PF_RenderRequest	req = extraP->input->output_request;

	BitonicSorterParams *infoP = new (std::nothrow) BitonicSorterParams();
	if (!infoP) {
		return PF_Err_OUT_OF_MEMORY;
	}

	PF_ParamDef cur_param;

	AEFX_CLR_STRUCT(cur_param);
	ERR(PF_CHECKOUT_PARAM(in_data, BPS_DIRECTION, in_data->current_time,
						  in_data->time_step, in_data->time_scale, &cur_param));
	infoP->direction = cur_param.u.pd.value;	// 1 = Horizontal, 2 = Vertical

	AEFX_CLR_STRUCT(cur_param);
	ERR(PF_CHECKOUT_PARAM(in_data, BPS_ORDER, in_data->current_time,
						  in_data->time_step, in_data->time_scale, &cur_param));
	infoP->ascending = (cur_param.u.pd.value == BPS_ORDER_ASCENDING) ? 1 : 0;

	AEFX_CLR_STRUCT(cur_param);
	ERR(PF_CHECKOUT_PARAM(in_data, BPS_THRESHOLD_MIN, in_data->current_time,
						  in_data->time_step, in_data->time_scale, &cur_param));
	infoP->thresholdMin = (float)(cur_param.u.fs_d.value / 100.0);

	AEFX_CLR_STRUCT(cur_param);
	ERR(PF_CHECKOUT_PARAM(in_data, BPS_THRESHOLD_MAX, in_data->current_time,
						  in_data->time_step, in_data->time_scale, &cur_param));
	infoP->thresholdMax = (float)(cur_param.u.fs_d.value / 100.0);

#if defined(BPS_GPU_ENABLED)
	{
		// The GPU bitonic sort runs a whole line in group-shared memory, so only
		// offer the GPU path when the sort-axis length is within the limit;
		// otherwise After Effects falls back to the (unlimited) CPU path.
		A_long sortAxisLen = (infoP->direction == BPS_DIR_HORIZONTAL)
								  ? in_data->width : in_data->height;
		if (sortAxisLen <= BPS_GPU_MAX_LINE) {
			extraP->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
		}
	}
#endif

	// Pixel sorting needs whole lines, so request the full input frame rather
	// than just the requested output band.
	req.rect.left	= 0;
	req.rect.top	= 0;
	req.rect.right	= in_data->width;
	req.rect.bottom	= in_data->height;

	ERR(extraP->cb->checkout_layer(in_data->effect_ref,
								   BPS_INPUT,
								   BPS_INPUT,
								   &req,
								   in_data->current_time,
								   in_data->time_step,
								   in_data->time_scale,
								   &in_result));

	if (!err) {
		UnionLRect(&in_result.result_rect,     &extraP->output->result_rect);
		UnionLRect(&in_result.max_result_rect, &extraP->output->max_result_rect);

		extraP->output->pre_render_data = infoP;
		extraP->output->delete_pre_render_data_func = DisposePreRenderData;
	} else {
		delete infoP;
	}

	return err;
}

//-----------------------------------------------------------------------------
static PF_Err
SmartRender(
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_SmartRenderExtra	*extraP,
	bool				isGPU)
{
	PF_Err	err  = PF_Err_NONE;
	PF_Err	err2 = PF_Err_NONE;

	PF_EffectWorld	*input_worldP  = NULL;
	PF_EffectWorld	*output_worldP = NULL;

	BitonicSorterParams *infoP =
		reinterpret_cast<BitonicSorterParams *>(extraP->input->pre_render_data);

	if (!infoP) {
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}

	ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, BPS_INPUT, &input_worldP)));
	ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

	if (!err && input_worldP && output_worldP) {
		AEFX_SuiteScoper<PF_WorldSuite2> world_suite =
			AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite,
											 kPFWorldSuiteVersion2, out_data);
		PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
		ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

		if (!err) {
			if (isGPU) {
				ERR(BPS_SmartRenderGPU(in_data, out_data, pixel_format,
									   input_worldP, output_worldP, extraP, infoP));
			} else {
				ERR(BPS_SortImageCPU(in_data, out_data, pixel_format,
									 input_worldP, output_worldP, infoP));
			}
		}
	}

	ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, BPS_INPUT));
	return err;
}

//-----------------------------------------------------------------------------
extern "C" DllExport
PF_Err
PluginDataEntryFunction2(
	PF_PluginDataPtr		inPtr,
	PF_PluginDataCB2		inPluginDataCallBackPtr,
	SPBasicSuite			*inSPBasicSuitePtr,
	const char				*inHostName,
	const char				*inHostVersion)
{
	PF_Err result = PF_Err_INVALID_CALLBACK;

	result = PF_REGISTER_EFFECT_EXT2(
		inPtr,
		inPluginDataCallBackPtr,
		NAME,				// Name
		MATCHNAME,			// Match Name
		CATEGORY,			// Category
		0,					// Reserved Info
		"EffectMain",		// Entry point
		SUPPORT_URL);		// Support URL

	return result;
}

//-----------------------------------------------------------------------------
PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err err = PF_Err_NONE;

	try {
		switch (cmd) {
		case PF_Cmd_ABOUT:
			err = About(in_data, out_data, params, output);
			break;
		case PF_Cmd_GLOBAL_SETUP:
			err = GlobalSetup(in_data, out_data, params, output);
			break;
		case PF_Cmd_PARAMS_SETUP:
			err = ParamsSetup(in_data, out_data, params, output);
			break;
		case PF_Cmd_GPU_DEVICE_SETUP:
			err = BPS_GPUDeviceSetup(in_data, out_data, reinterpret_cast<PF_GPUDeviceSetupExtra *>(extra));
			break;
		case PF_Cmd_GPU_DEVICE_SETDOWN:
			err = BPS_GPUDeviceSetdown(in_data, out_data, reinterpret_cast<PF_GPUDeviceSetdownExtra *>(extra));
			break;
		case PF_Cmd_SMART_PRE_RENDER:
			err = PreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra *>(extra));
			break;
		case PF_Cmd_SMART_RENDER:
			err = SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra *>(extra), false);
			break;
		case PF_Cmd_SMART_RENDER_GPU:
			err = SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra *>(extra), true);
			break;
		default:
			break;
		}
	} catch (PF_Err &thrown_err) {
		// Never let an exception escape into After Effects.
		err = thrown_err;
	}

	return err;
}
