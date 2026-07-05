/*
	BitonicPixelSorter_PathGeometry.h

	Host-side mask-path sampling for Path mode: arc-length polylines with
	open-end straight extensions, closest-point queries, and domain bounds.
*/

#pragma once
#ifndef BITONIC_PIXEL_SORTER_PATH_GEOMETRY_H
#define BITONIC_PIXEL_SORTER_PATH_GEOMETRY_H

#include "BitonicPixelSorter.h"

#include <memory>
#include <vector>

// GPU/CPU shared sample layout (20 bytes, tightly packed floats).
struct BpsPathSample {
	float x;
	float y;
	float s;
	float tx;
	float ty;
};

// Immutable, geometry-only Path-mode pixel map. Depends solely on the mask
// shape, frame size, downsample and sort direction, so it is cached and shared
// across frames and effect instances (see BPS_AcquirePathMap).
struct BpsPathMap {
	std::vector<BpsMappedPixelRecord> records;
	std::vector<std::uint32_t> lineOffsets;
	std::vector<std::uint32_t> workOffsets;
	A_long maxLineLength = 0;
	A_long domainMaxLineLength = 0;
	A_long mappedRecordCount = 0;
	A_long mappedWorkItemCount = 0;
};

struct BitonicPreRenderData {
	BitonicSorterParams params;
	std::vector<BpsPathSample> pathSamples;
	std::vector<BpsMappedPixelRecord> mappedRecords;
	std::vector<std::uint32_t> mappedLineOffsets;
	std::vector<std::uint32_t> mappedWorkOffsets;
	// Path mode uses a cached, shared map instead of the vectors above.
	std::shared_ptr<const BpsPathMap> pathMap;
	// True when PreRender successfully reserved a non-None key-source layer.
	// GPU worlds often have a null CPU `data` pointer, so SmartRender must not
	// use `world->data` to decide whether an external source is active.
	bool has_trigger_source = false;
	bool has_criterion_source = false;
};

// Sample the selected mask path into an extended polyline and fill Path-mode
// domain fields on params. Returns PF_Err_NONE with an empty sample list when
// no path is selected or the outline is unavailable.
PF_Err BPS_BuildPathGeometry(
	PF_InData *in_data,
	PF_OutData *out_data,
	PF_PathID path_id,
	A_long path_direction,
	BitonicSorterParams *paramsP,
	std::vector<BpsPathSample> *samplesP);

// Closest point on the extended polyline. Writes arc length s and signed
// normal distance n (left of tangent is positive).
bool BPS_PathClosestPoint(
	const BpsPathSample *samples,
	A_long sample_count,
	float px,
	float py,
	float *s_out,
	float *n_out);

// Evaluate position and unit tangent at arc length s (clamped to the polyline).
bool BPS_PathEvalAtS(
	const BpsPathSample *samples,
	A_long sample_count,
	float s,
	float *x_out,
	float *y_out,
	float *tx_out,
	float *ty_out);

// Forward map (line, pos) -> pixel for Path mode.
bool BPS_PathCoordForPos(
	const BitonicSorterParams &prm,
	A_long line,
	A_long pos,
	A_long *x_out,
	A_long *y_out);

// Inverse map pixel -> (line, pos) for Path mode.
bool BPS_PathDomainPosForPixel(
	const BitonicSorterParams &prm,
	A_long x,
	A_long y,
	A_long *line_out,
	A_long *pos_out);

bool BPS_ClassifyMappedPixel(
	const BitonicSorterParams &prm,
	A_long frame_w,
	A_long frame_h,
	A_long x,
	A_long y,
	A_long *line_out,
	float *pos_key_out);

// Path mode: return the geometry-only pixel map for these params, building it
// (low-resolution Jump Flooding) only on a cache miss. `paramsP` must already
// carry the resolved path domain (BPS_BuildPathGeometry). The returned map is
// immutable and shared; keep the shared_ptr alive for the render's lifetime.
std::shared_ptr<const BpsPathMap> BPS_AcquirePathMap(
	A_long frame_w,
	A_long frame_h,
	const BitonicSorterParams &prm);

// Geometry-only cache key shared by the CPU path-map cache and GPU device-side
// path-map caches. The params must already carry resolved Path geometry.
std::uint64_t BPS_PathMapKey(
	A_long frame_w,
	A_long frame_h,
	const BitonicSorterParams &prm);

// Non-axis modes that sort via a cached pixel-owned map (Path + transform modes).
inline bool BPS_ModeUsesMappedSort(A_long mode)
{
	return mode >= BPS_MODE_FREE_ANGLE && mode <= BPS_MODE_PATH;
}

inline bool BPS_ModeUsesTransformMap(A_long mode)
{
	return mode >= BPS_MODE_FREE_ANGLE && mode <= BPS_MODE_SWIRL;
}

// Analytic transform modes: direct per-pixel classification (no JFA).
std::shared_ptr<const BpsPathMap> BPS_AcquireTransformMap(
	A_long frame_w,
	A_long frame_h,
	const BitonicSorterParams &prm);

std::uint64_t BPS_TransformMapKey(
	A_long frame_w,
	A_long frame_h,
	const BitonicSorterParams &prm);

#endif // BITONIC_PIXEL_SORTER_PATH_GEOMETRY_H
