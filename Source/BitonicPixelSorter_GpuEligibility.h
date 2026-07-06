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

// PF_InData width/height are full-resolution layer size (not downsampled).
// SmartFX checkout rects, GPU buffers and path samples after downsample live in
// current render space — use these helpers for that space.
inline A_long BPS_RenderWidth(const PF_InData *in_data)
{
	if (!in_data || in_data->width <= 0) {
		return 0;
	}
	const PF_RationalScale &s = in_data->downsample_x;
	if (s.num <= 0 || s.den <= 0) {
		return in_data->width;
	}
	return static_cast<A_long>(
		(static_cast<long long>(in_data->width) * s.num) / s.den);
}

inline A_long BPS_RenderHeight(const PF_InData *in_data)
{
	if (!in_data || in_data->height <= 0) {
		return 0;
	}
	const PF_RationalScale &s = in_data->downsample_y;
	if (s.num <= 0 || s.den <= 0) {
		return in_data->height;
	}
	return static_cast<A_long>(
		(static_cast<long long>(in_data->height) * s.num) / s.den);
}

inline PF_LRect BPS_FrameRect(const PF_InData *in_data)
{
	PF_LRect rect;
	rect.left = 0;
	rect.top = 0;
	rect.right = BPS_RenderWidth(in_data);
	rect.bottom = BPS_RenderHeight(in_data);
	return rect;
}

inline PF_LRect BPS_ClipRectToFrame(PF_LRect rect, const PF_InData *in_data)
{
	if (!in_data) {
		rect.left = rect.right = rect.top = rect.bottom = 0;
		return rect;
	}

	const A_long frame_w = BPS_RenderWidth(in_data);
	const A_long frame_h = BPS_RenderHeight(in_data);
	rect.left = BPS_ClampLong(rect.left, 0, frame_w);
	rect.right = BPS_ClampLong(rect.right, 0, frame_w);
	rect.top = BPS_ClampLong(rect.top, 0, frame_h);
	rect.bottom = BPS_ClampLong(rect.bottom, 0, frame_h);
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
	if (!in_data) {
		return false;
	}
	const A_long frame_w = BPS_RenderWidth(in_data);
	const A_long frame_h = BPS_RenderHeight(in_data);
	return rect.left <= 0 &&
		   rect.top <= 0 &&
		   rect.right >= frame_w &&
		   rect.bottom >= frame_h;
}

void BPS_RecordHostVersion(const char *host_version);
long BPS_HostMajorVersion();
long BPS_HostMinorVersion();
bool BPS_IsGpuFrameworkCompiled(PF_GPU_Framework framework);
bool BPS_ShouldAcceptGpuDeviceSetup(
	const PF_InData *in_data,
	PF_GPU_Framework framework);

inline BpsGpuEligibility BPS_EvaluateGpuEligibility(
	const PF_InData *in_data,
	A_long mode,
	A_long max_sort_axis_len,
	const PF_LRect &output_rect)
{
#if !defined(BPS_GPU_ENABLED)
	(void)in_data;
	(void)mode;
	(void)max_sort_axis_len;
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

	// Axis mode sorts in group-shared memory, so the sort-axis length is the
	// remaining structural limit. Non-axis modes use global-memory domain/map
	// paths whose line length may exceed the shared-memory budget.
	if (mode == BPS_MODE_AXIS) {
		if (max_sort_axis_len > BPS_GPU_MAX_LINE) {
			return {false, BpsGpuBlockReason::SortAxisTooLong};
		}
	}

	return {true, BpsGpuBlockReason::None};
#endif
}

inline bool BPS_UsesAxisLumaFastPath(
	const void *src_mem,
	const void *criterion_mem,
	const void *trigger_mem,
	const BitonicSorterParams &prm)
{
	return prm.mode == BPS_MODE_AXIS &&
		   criterion_mem == src_mem &&
		   trigger_mem == src_mem &&
		   prm.criterion == BPS_CRITERION_LUMINANCE &&
		   prm.trigger == BPS_CRITERION_LUMINANCE &&
		   prm.affect == BPS_AFFECT_INSIDE_THRESHOLDS &&
		   prm.cycleDegrees > -1.0e-6f &&
		   prm.cycleDegrees < 1.0e-6f;
}

inline bool BPS_AxisSourceFullSpan(
	int direction,
	int width,
	int height,
	int inputOriginX,
	int inputOriginY,
	int inputWidth,
	int inputHeight,
	int outputOriginX,
	int outputOriginY,
	int outputWidth,
	int outputHeight)
{
	return direction
		? (inputOriginX == 0 &&
		   inputWidth == width &&
		   inputOriginY <= outputOriginY &&
		   inputOriginY + inputHeight >= outputOriginY + outputHeight)
		: (inputOriginY == 0 &&
		   inputHeight == height &&
		   inputOriginX <= outputOriginX &&
		   inputOriginX + inputWidth >= outputOriginX + outputWidth);
}

inline bool BPS_AxisOutputFullSpan(
	int direction,
	int width,
	int height,
	int outputOriginX,
	int outputOriginY,
	int outputWidth,
	int outputHeight)
{
	return direction
		? (outputOriginX == 0 && outputWidth == width)
		: (outputOriginY == 0 && outputHeight == height);
}

const char *BPS_ActiveGpuFrameworkName();
const char *BPS_ActiveGpuDeviceName();
bool BPS_IsGpuDeviceReady();
bool BPS_LastRenderUsedGpu();
void BPS_SetLastRenderUsedGpu(bool used_gpu);

#endif // BITONIC_PIXEL_SORTER_GPU_ELIGIBILITY_H
