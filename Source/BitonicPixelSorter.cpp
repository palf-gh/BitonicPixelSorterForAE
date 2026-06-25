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
#include "BitonicPixelSorter_GpuEligibility.h"

#if defined(BPS_RENDER_DIAG)
	#include <chrono>
	#include <cstdarg>
	#include <cstdio>
#endif
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

namespace {

#if defined(BPS_RENDER_DIAG)
static const char *
GpuBlockReasonName(BpsGpuBlockReason reason)
{
	switch (reason) {
	case BpsGpuBlockReason::None: return "None";
	case BpsGpuBlockReason::NoBackendCompiled: return "NoBackendCompiled";
	case BpsGpuBlockReason::HostPremiere: return "HostPremiere";
	case BpsGpuBlockReason::SortAxisTooLong: return "SortAxisTooLong";
	default: return "Unknown";
	}
}

static const char *
PixelFormatName(PF_PixelFormat pixel_format)
{
	switch (pixel_format) {
	case PF_PixelFormat_ARGB32: return "ARGB32";
	case PF_PixelFormat_ARGB64: return "ARGB64";
	case PF_PixelFormat_ARGB128: return "ARGB128";
	case PF_PixelFormat_GPU_BGRA128: return "GPU_BGRA128";
	default: return "Unknown";
	}
}

static void
BPS_DiagLog(const char *format, ...)
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

	// GPU acceleration status (read-only custom UI) — first user control after input.
	AEFX_CLR_STRUCT(def);
	{
		std::string name =
			AELocalise::GetStringForAE(LocKey::STR_GPU_STATUS_NAME, in_data);
		def.ui_flags = PF_PUI_CONTROL | PF_PUI_DONT_ERASE_CONTROL;
		def.ui_width = BPS_GPU_STATUS_UI_WIDTH;
		def.ui_height = BPS_GPU_STATUS_UI_HEIGHT;

		// Premiere Pro does not support standard param types with custom UI.
		if (in_data->appl_id != kAppID_Premiere) {
			PF_ADD_NULL(name.c_str(), BPS_GPU_STATUS);
		} else {
			PF_ADD_ARBITRARY2(name.c_str(),
							  BPS_GPU_STATUS_UI_WIDTH,
							  BPS_GPU_STATUS_UI_HEIGHT,
							  0,
							  PF_PUI_CONTROL | PF_PUI_DONT_ERASE_CONTROL,
							  0,
							  BPS_GPU_STATUS,
							  0);
		}
	}

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

	if (!err) {
		PF_CustomUIInfo ci;
		AEFX_CLR_STRUCT(ci);

		ci.events = PF_CustomEFlag_EFFECT;

		ci.comp_ui_width = 0;
		ci.comp_ui_height = 0;
		ci.comp_ui_alignment = PF_UIAlignment_NONE;

		ci.layer_ui_width = 0;
		ci.layer_ui_height = 0;
		ci.layer_ui_alignment = PF_UIAlignment_NONE;

		ci.preview_ui_width = 0;
		ci.preview_ui_height = 0;
		ci.preview_ui_alignment = PF_UIAlignment_NONE;

		err = (*(in_data->inter.register_ui))(in_data->effect_ref, &ci);
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
	const PF_RenderRequest	raw_req = extraP->input->output_request;
	PF_RenderRequest		req = raw_req;
	PF_LRect				output_rect = BPS_ClipRectToFrame(req.rect, in_data);

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

	const BpsGpuEligibility gpu_eligibility =
		BPS_EvaluateGpuEligibility(in_data, infoP->direction, output_rect);
	if (gpu_eligibility.render_possible) {
		extraP->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
	}

#if defined(BPS_RENDER_DIAG)
	BPS_DiagLog(
		"PreRender output_request=(%ld,%ld,%ld,%ld) clipped=(%ld,%ld,%ld,%ld) "
		"gpu_possible=%d reason=%s frame=%ldx%ld direction=%ld",
		static_cast<long>(raw_req.rect.left),
		static_cast<long>(raw_req.rect.top),
		static_cast<long>(raw_req.rect.right),
		static_cast<long>(raw_req.rect.bottom),
		static_cast<long>(output_rect.left),
		static_cast<long>(output_rect.top),
		static_cast<long>(output_rect.right),
		static_cast<long>(output_rect.bottom),
		gpu_eligibility.render_possible ? 1 : 0,
		GpuBlockReasonName(gpu_eligibility.reason),
		static_cast<long>(in_data->width),
		static_cast<long>(in_data->height),
		static_cast<long>(infoP->direction));
#endif

	// Pixel sorting needs the complete sort axis for each requested output
	// pixel. Expand only the dependency axis, while keeping the produced result
	// within AE's requested output rectangle.
	req.rect = output_rect;
	if (infoP->direction == BPS_DIR_HORIZONTAL) {
		req.rect.left = 0;
		req.rect.right = in_data->width;
	} else {
		req.rect.top = 0;
		req.rect.bottom = in_data->height;
	}

	ERR(extraP->cb->checkout_layer(in_data->effect_ref,
								   BPS_INPUT,
								   BPS_INPUT,
								   &req,
								   in_data->current_time,
								   in_data->time_step,
								   in_data->time_scale,
								   &in_result));

	if (!err) {
		extraP->output->result_rect = output_rect;
		extraP->output->max_result_rect = BPS_FrameRect(in_data);

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

	BPS_SetLastRenderUsedGpu(isGPU);

	ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, BPS_INPUT, &input_worldP)));
	ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

	if (!err && input_worldP && output_worldP) {
		AEFX_SuiteScoper<PF_WorldSuite2> world_suite =
			AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite,
											 kPFWorldSuiteVersion2, out_data);
		PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
		ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

		if (!err) {
#if defined(BPS_RENDER_DIAG)
			const auto render_start = std::chrono::steady_clock::now();
#endif
			if (isGPU) {
				ERR(BPS_SmartRenderGPU(in_data, out_data, pixel_format,
									   input_worldP, output_worldP, extraP, infoP));
			} else {
				ERR(BPS_SortImageCPU(in_data, out_data, pixel_format,
									 input_worldP, output_worldP, infoP));
			}
#if defined(BPS_RENDER_DIAG)
			const auto render_end = std::chrono::steady_clock::now();
			const double elapsed_ms =
				std::chrono::duration<double, std::milli>(render_end - render_start).count();
			BPS_DiagLog(
				"SmartRender path=%s isGPU=%d pixel_format=%s frame=%ldx%ld "
				"output_origin=(%ld,%ld) output_size=%ldx%ld direction=%ld elapsed_ms=%.3f err=%ld",
				isGPU ? "GPU" : "CPU",
				isGPU ? 1 : 0,
				PixelFormatName(pixel_format),
				static_cast<long>(in_data->width),
				static_cast<long>(in_data->height),
				static_cast<long>(output_worldP->origin_x),
				static_cast<long>(output_worldP->origin_y),
				static_cast<long>(output_worldP->width),
				static_cast<long>(output_worldP->height),
				static_cast<long>(infoP->direction),
				elapsed_ms,
				static_cast<long>(err));
#endif
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
		case PF_Cmd_EVENT:
			err = BPS_HandleEvent(in_data, out_data, params, output,
								  reinterpret_cast<PF_EventExtra *>(extra));
			break;
		case PF_Cmd_UPDATE_PARAMS_UI:
			err = BPS_UpdateParamsUI(in_data, out_data, params, output);
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
