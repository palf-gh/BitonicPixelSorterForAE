#if defined(BPS_HAS_HLSL)

#include "BPS_DirectXShaderLoad.h"

#include "BitonicSortKernel.cso.h"
#include "BitonicSortKernel.rs.h"

#include <cstring>

bool BPS_LoadEmbeddedDirectXSortShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader)
{
	if (!context || !out_shader || !context->mDevice) {
		return false;
	}

	Microsoft::WRL::ComPtr<ID3DBlob> root_signature_blob;
	HRESULT res = D3DCreateBlob(
		static_cast<SIZE_T>(bps_directx_embedded::kBitonicSortKernel_rs_size),
		&root_signature_blob);
	if (FAILED(res)) {
		return false;
	}

	std::memcpy(
		root_signature_blob->GetBufferPointer(),
		bps_directx_embedded::kBitonicSortKernel_rs,
		bps_directx_embedded::kBitonicSortKernel_rs_size);

	res = context->mDevice->CreateRootSignature(
		0,
		root_signature_blob->GetBufferPointer(),
		root_signature_blob->GetBufferSize(),
		IID_PPV_ARGS(&out_shader->mRootSignature));
	if (FAILED(res)) {
		return false;
	}

	D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_state_desc = {};
	pipeline_state_desc.pRootSignature = out_shader->mRootSignature.Get();
	pipeline_state_desc.CS.pShaderBytecode = bps_directx_embedded::kBitonicSortKernel_cso;
	pipeline_state_desc.CS.BytecodeLength =
		static_cast<SIZE_T>(bps_directx_embedded::kBitonicSortKernel_cso_size);

	res = context->mDevice->CreateComputePipelineState(
		&pipeline_state_desc,
		IID_PPV_ARGS(&out_shader->mPipelineState));
	return SUCCEEDED(res);
}

#endif
