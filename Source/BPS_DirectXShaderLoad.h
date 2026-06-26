#pragma once

#if defined(BPS_HAS_HLSL)

#include "DirectXUtils.h"

// Loads the bitonic sort compute shader from build-time embedded CSO/RS blobs.
// Avoids a runtime DirectX_Assets folder beside the .aex module.
bool BPS_LoadEmbeddedDirectXSortShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader);

#endif
