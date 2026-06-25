/*
	BitonicPixelSorter_GpuEligibility.h

	Shared GPU eligibility checks for render and UI code. Keep this header light:
	it depends only on AE effect types and the plugin's public constants.
*/

#pragma once
#ifndef BITONIC_PIXEL_SORTER_GPU_ELIGIBILITY_H
#define BITONIC_PIXEL_SORTER_GPU_ELIGIBILITY_H

#include "BitonicPixelSorter.h"

enum class BpsGpuBlockReason {
	None,
	NoBackendCompiled,
	HostPremiere,
	SortAxisTooLong
};

struct BpsGpuEligibility {
	bool render_possible;
	BpsGpuBlockReason reason;
};

inline A_long BPS_ClampLong(A_long value, A_long lo, A_long hi)
{
	return value < lo ? lo : (value > hi ? hi : value);
}

inline PF_LRect BPS_FrameRect(const PF_InData *in_data)
{
	PF_LRect rect;
	rect.left = 0;
	rect.top = 0;
	rect.right = in_data ? in_data->width : 0;
	rect.bottom = in_data ? in_data->height : 0;
	return rect;
}

inline PF_LRect BPS_ClipRectToFrame(PF_LRect rect, const PF_InData *in_data)
{
	if (!in_data) {
		rect.left = rect.right = rect.top = rect.bottom = 0;
		return rect;
	}

	rect.left = BPS_ClampLong(rect.left, 0, in_data->width);
	rect.right = BPS_ClampLong(rect.right, 0, in_data->width);
	rect.top = BPS_ClampLong(rect.top, 0, in_data->height);
	rect.bottom = BPS_ClampLong(rect.bottom, 0, in_data->height);
	if (rect.right < rect.left) {
		rect.right = rect.left;
	}
	if (rect.bottom < rect.top) {
		rect.bottom = rect.top;
	}
	return rect;
}

inline bool BPS_IsFullFrameRequest(const PF_LRect &rect, const PF_InData *in_data)
{
	return in_data &&
		   rect.left <= 0 &&
		   rect.top <= 0 &&
		   rect.right >= in_data->width &&
		   rect.bottom >= in_data->height;
}

inline BpsGpuEligibility BPS_EvaluateGpuEligibility(
	const PF_InData *in_data,
	A_long direction,
	const PF_LRect &output_rect)
{
#if !defined(BPS_GPU_ENABLED)
	(void)in_data;
	(void)direction;
	(void)output_rect;
	return {false, BpsGpuBlockReason::NoBackendCompiled};
#else
	(void)output_rect;
	if (!in_data) {
		return {false, BpsGpuBlockReason::NoBackendCompiled};
	}

	if (in_data->appl_id == 'PrMr') {
		return {false, BpsGpuBlockReason::HostPremiere};
	}

	const A_long sort_axis_len =
		(direction == BPS_DIR_HORIZONTAL) ? in_data->width : in_data->height;
	if (sort_axis_len > BPS_GPU_MAX_LINE) {
		return {false, BpsGpuBlockReason::SortAxisTooLong};
	}

	return {true, BpsGpuBlockReason::None};
#endif
}

const char *BPS_ActiveGpuFrameworkName();
const char *BPS_ActiveGpuDeviceName();
bool BPS_IsGpuDeviceReady();
bool BPS_LastRenderUsedGpu();
void BPS_SetLastRenderUsedGpu(bool used_gpu);

#endif // BITONIC_PIXEL_SORTER_GPU_ELIGIBILITY_H
