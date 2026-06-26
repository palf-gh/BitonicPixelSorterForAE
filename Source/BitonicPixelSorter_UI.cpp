/*
	BitonicPixelSorter_UI.cpp

	Effect Controls custom UI for the GPU acceleration status readout.
	Draws localised coloured text using the shared GPU eligibility predicate.
*/

#include "BitonicPixelSorter.h"
#include "BitonicPixelSorter_GpuEligibility.h"

#include "AE_EffectUI.h"
#include "AEFX_SuiteHelper.h"
#include <adobesdk/DrawbotSuite.h>

#include <cstring>
#include <string>
#include <vector>

#include "Localise/LocKeys.h"
#include "Localise/Strings_en_US.h"
#include "Localise/Strings_ja_JP.h"
#include "Localise/Strings_zh_CN.h"
#include "Localise/Strings_ko_KR.h"
#include "Localise/AELocalise.h"

namespace {

static void
BPS_CopyUtf8ToDrawbotUtf16(const char *utf8, std::vector<DRAWBOT_UTF16Char> *out)
{
	out->clear();
	if (!utf8) {
		return;
	}

#ifdef AE_OS_WIN
	const int wide_size =
		MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
	if (wide_size <= 0) {
		return;
	}

	std::vector<wchar_t> wide(static_cast<size_t>(wide_size), 0);
	if (MultiByteToWideChar(CP_UTF8, 0, utf8, -1, wide.data(), wide_size) <= 0) {
		return;
	}

	for (int i = 0; i < wide_size - 1; ++i) {
		out->push_back(static_cast<DRAWBOT_UTF16Char>(wide[static_cast<size_t>(i)]));
	}
	out->push_back(0);
#else
	CFStringRef cf_str = CFStringCreateWithBytes(
		kCFAllocatorDefault,
		reinterpret_cast<const UInt8 *>(utf8),
		static_cast<CFIndex>(std::strlen(utf8)),
		kCFStringEncodingUTF8,
		false);
	if (!cf_str) {
		return;
	}

	const CFIndex length = CFStringGetLength(cf_str);
	const CFIndex max_bytes =
		CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF16) + sizeof(UniChar);
	std::vector<UInt8> buffer(static_cast<size_t>(max_bytes), 0);

	CFIndex used_bytes = 0;
	const Boolean converted = CFStringGetBytes(
		cf_str,
		CFRangeMake(0, length),
		kCFStringEncodingUTF16,
		0,
		false,
		buffer.data(),
		max_bytes,
		&used_bytes);
	CFRelease(cf_str);

	if (!converted) {
		return;
	}

	const CFIndex char_count = used_bytes / static_cast<CFIndex>(sizeof(UniChar));
	for (CFIndex i = 0; i < char_count; ++i) {
		UniChar ch = 0;
		std::memcpy(&ch, buffer.data() + (i * sizeof(UniChar)), sizeof(UniChar));
		out->push_back(static_cast<DRAWBOT_UTF16Char>(ch));
	}
	out->push_back(0);
#endif
}

static LocKey::Key
BPS_GpuStatusLocKey(BpsGpuBlockReason reason)
{
	switch (reason) {
	case BpsGpuBlockReason::NoBackendCompiled:
		return LocKey::STR_GPU_STATUS_NO_BACKEND;
	case BpsGpuBlockReason::HostPremiere:
		return LocKey::STR_GPU_STATUS_HOST;
	case BpsGpuBlockReason::SortAxisTooLong:
		return LocKey::STR_GPU_STATUS_AXIS_TOO_LONG;
	case BpsGpuBlockReason::None:
	default:
		return LocKey::STR_GPU_STATUS_ACTIVE;
	}
}

static std::string
BPS_BuildGpuDetailLine()
{
	const char *framework = BPS_ActiveGpuFrameworkName();
	const char *device = BPS_ActiveGpuDeviceName();

	std::string detail;
	if (framework && framework[0] != '\0') {
		detail = framework;
		if (device && device[0] != '\0') {
			detail += " \xC2\xB7 "; // UTF-8 middle dot
			detail += device;
		}
	} else if (device && device[0] != '\0') {
		detail = device;
	}
	return detail;
}

static PF_Err
BPS_DrawStatusLine(
	const DRAWBOT_Suites &drawbot_suites,
	DRAWBOT_SurfaceRef surface_ref,
	DRAWBOT_BrushRef brush_ref,
	DRAWBOT_FontRef font_ref,
	const char *utf8_text,
	float x,
	float baseline_y,
	float max_width)
{
	if (!utf8_text || utf8_text[0] == '\0') {
		return PF_Err_NONE;
	}

	std::vector<DRAWBOT_UTF16Char> unicode_string;
	BPS_CopyUtf8ToDrawbotUtf16(utf8_text, &unicode_string);
	if (unicode_string.empty()) {
		return PF_Err_NONE;
	}

	DRAWBOT_PointF32 text_origin;
	text_origin.x = x;
	text_origin.y = baseline_y;

	return drawbot_suites.surface_suiteP->DrawString(
		surface_ref,
		brush_ref,
		font_ref,
		unicode_string.data(),
		&text_origin,
		kDRAWBOT_TextAlignment_Left,
		kDRAWBOT_TextTruncation_EndEllipsis,
		max_width);
}

static PF_Err
BPS_DrawGpuStatus(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_EventExtra	*event_extra)
{
	PF_Err err = PF_Err_NONE;
	PF_Err err2 = PF_Err_NONE;

	if (!params || !params[BPS_GPU_STATUS] || !params[BPS_DIRECTION]) {
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	DRAWBOT_DrawRef		drawing_ref = NULL;
	DRAWBOT_SurfaceRef	surface_ref = NULL;
	DRAWBOT_SupplierRef	supplier_ref = NULL;
	DRAWBOT_BrushRef	brush_ref = NULL;
	DRAWBOT_FontRef		font_ref = NULL;

	DRAWBOT_Suites		drawbot_suites;
	DRAWBOT_ColorRGBA	drawbot_colour;
	float				default_font_size = 0.0f;

	ERR(AEFX_AcquireDrawbotSuites(in_data, out_data, &drawbot_suites));

	PF_EffectCustomUISuite1 *effect_custom_ui_suiteP = NULL;
	ERR(AEFX_AcquireSuite(in_data,
						  out_data,
						  kPFEffectCustomUISuite,
						  kPFEffectCustomUISuiteVersion1,
						  NULL,
						  reinterpret_cast<void **>(&effect_custom_ui_suiteP)));

	if (!err && effect_custom_ui_suiteP) {
		ERR((*effect_custom_ui_suiteP->PF_GetDrawingReference)(
			event_extra->contextH, &drawing_ref));
		AEFX_ReleaseSuite(in_data, out_data, kPFEffectCustomUISuite,
						  kPFEffectCustomUISuiteVersion1, NULL);
	}

	ERR(drawbot_suites.drawbot_suiteP->GetSupplier(drawing_ref, &supplier_ref));
	ERR(drawbot_suites.drawbot_suiteP->GetSurface(drawing_ref, &surface_ref));

	if (!err && PF_EA_CONTROL == event_extra->effect_win.area) {
		const A_long direction = params[BPS_DIRECTION]->u.pd.value;
		const BpsGpuEligibility eligibility = BPS_EvaluateGpuEligibility(
			in_data, direction, BPS_FrameRect(in_data));

		const bool last_render_used_gpu = BPS_LastRenderUsedGpu();
		bool draw_second_line = false;
		std::string line1_text;
		std::string line2_text;

		if (!eligibility.render_possible) {
			drawbot_colour.red = 0.9f;
			drawbot_colour.green = 0.1f;
			drawbot_colour.blue = 0.1f;
			line1_text = AELocalise::GetStringForAE(
				BPS_GpuStatusLocKey(eligibility.reason), in_data);
		} else if (last_render_used_gpu) {
			drawbot_colour.red = 0.0f;
			drawbot_colour.green = 0.8f;
			drawbot_colour.blue = 0.0f;
			line1_text = AELocalise::GetStringForAE(LocKey::STR_GPU_STATUS_ACTIVE, in_data);
			line2_text = BPS_BuildGpuDetailLine();
			draw_second_line = !line2_text.empty();
		} else {
			drawbot_colour.red = 0.9f;
			drawbot_colour.green = 0.1f;
			drawbot_colour.blue = 0.1f;
			line1_text = AELocalise::GetStringForAE(
				LocKey::STR_GPU_STATUS_HOST_INACTIVE, in_data);
		}
		drawbot_colour.alpha = 1.0f;

		ERR(drawbot_suites.supplier_suiteP->GetDefaultFontSize(
			supplier_ref, &default_font_size));
		ERR(drawbot_suites.supplier_suiteP->NewDefaultFont(
			supplier_ref, default_font_size, &font_ref));
		ERR(drawbot_suites.supplier_suiteP->NewBrush(
			supplier_ref, &drawbot_colour, &brush_ref));

		const float frame_left =
			static_cast<float>(event_extra->effect_win.current_frame.left);
		const float frame_top =
			static_cast<float>(event_extra->effect_win.current_frame.top);
		const float text_x = frame_left + 4.0f;
		const float max_width = static_cast<float>(
			event_extra->effect_win.current_frame.right -
			event_extra->effect_win.current_frame.left);

		// DrawString origin is the baseline; place it below the control top.
		const float top_pad = 4.0f;
		const float line_step = default_font_size * 1.25f;
		const float line1_baseline = frame_top + top_pad + default_font_size;

		ERR(BPS_DrawStatusLine(
			drawbot_suites, surface_ref, brush_ref, font_ref,
			line1_text.c_str(), text_x, line1_baseline, max_width));

		if (!err && draw_second_line) {
			ERR(BPS_DrawStatusLine(
				drawbot_suites, surface_ref, brush_ref, font_ref,
				line2_text.c_str(), text_x, line1_baseline + line_step, max_width));
		}

		if (brush_ref) {
			ERR2(drawbot_suites.supplier_suiteP->ReleaseObject(
				reinterpret_cast<DRAWBOT_ObjectRef>(brush_ref)));
		}
		if (font_ref) {
			ERR2(drawbot_suites.supplier_suiteP->ReleaseObject(
				reinterpret_cast<DRAWBOT_ObjectRef>(font_ref)));
		}
	}

	ERR2(AEFX_ReleaseDrawbotSuites(in_data, out_data));

	if (!err) {
		event_extra->evt_out_flags |= PF_EO_HANDLED_EVENT;
	}

	return err;
}

} // namespace

PF_Err
BPS_HandleEvent(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	PF_EventExtra	*event_extra)
{
	(void)output;

	if (!event_extra) {
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	switch (event_extra->e_type) {
	case PF_Event_DRAW:
		return BPS_DrawGpuStatus(in_data, out_data, params, event_extra);
	default:
		break;
	}

	return PF_Err_NONE;
}

PF_Err
BPS_UpdateParamsUI(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	(void)in_data;
	(void)out_data;
	(void)params;
	(void)output;
	return PF_Err_NONE;
}
