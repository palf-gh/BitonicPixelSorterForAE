#pragma once

#if defined(BPS_HAS_HLSL)

#include "DirectXUtils.h"

// Loads compute shaders from build-time embedded CSO/RS blobs.
// Avoids a runtime DirectX_Assets folder beside the .aex module.
bool BPS_LoadEmbeddedDirectXSortShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader);

bool BPS_LoadEmbeddedDirectXAxisLumaSortShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader);

bool BPS_LoadEmbeddedDirectXAxisLumaFullSortShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader);

bool BPS_LoadEmbeddedDirectXDomainSortShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader);

bool BPS_LoadEmbeddedDirectXApplyDomainShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader);

bool BPS_LoadEmbeddedDirectXCopyShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader);

bool BPS_LoadEmbeddedDirectXMappedSortShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader);

bool BPS_LoadEmbeddedDirectXPathClassifyCountShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader);

bool BPS_LoadEmbeddedDirectXPathScatterRecordsShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader);

bool BPS_LoadEmbeddedDirectXPathSortRecordsShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader);

#endif
