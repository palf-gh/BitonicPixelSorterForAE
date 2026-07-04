#if defined(BPS_HAS_HLSL)

#include "BPS_DirectXShaderLoad.h"

#include "BitonicSortKernel.cso.h"
#include "BitonicSortKernel.rs.h"
#include "BitonicSortDomain.cso.h"
#include "BitonicSortDomain.rs.h"
#include "BitonicApplyDomain.cso.h"
#include "BitonicApplyDomain.rs.h"
#include "BitonicCopyInput.cso.h"
#include "BitonicCopyInput.rs.h"
#include "BitonicSortMapped.cso.h"
#include "BitonicSortMapped.rs.h"

#include <cstring>

namespace {

bool BPS_LoadEmbeddedDirectXShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader,
	const void *rs_bytes,
	size_t rs_size,
	const void *cso_bytes,
	size_t cso_size)
{
	if (!context || !out_shader || !context->mDevice || !rs_bytes || !cso_bytes ||
		rs_size == 0 || cso_size == 0) {
		return false;
	}

	Microsoft::WRL::ComPtr<ID3DBlob> root_signature_blob;
	HRESULT res = D3DCreateBlob(static_cast<SIZE_T>(rs_size), &root_signature_blob);
	if (FAILED(res)) {
		return false;
	}

	std::memcpy(
		root_signature_blob->GetBufferPointer(),
		rs_bytes,
		rs_size);

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
	pipeline_state_desc.CS.pShaderBytecode = cso_bytes;
	pipeline_state_desc.CS.BytecodeLength = static_cast<SIZE_T>(cso_size);

	res = context->mDevice->CreateComputePipelineState(
		&pipeline_state_desc,
		IID_PPV_ARGS(&out_shader->mPipelineState));
	return SUCCEEDED(res);
}

} // namespace

bool BPS_LoadEmbeddedDirectXSortShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader)
{
	return BPS_LoadEmbeddedDirectXShader(
		context,
		out_shader,
		bps_directx_embedded::kBitonicSortKernel_rs,
		bps_directx_embedded::kBitonicSortKernel_rs_size,
		bps_directx_embedded::kBitonicSortKernel_cso,
		bps_directx_embedded::kBitonicSortKernel_cso_size);
}

bool BPS_LoadEmbeddedDirectXDomainSortShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader)
{
	return BPS_LoadEmbeddedDirectXShader(
		context,
		out_shader,
		bps_directx_embedded::kBitonicSortDomain_rs,
		bps_directx_embedded::kBitonicSortDomain_rs_size,
		bps_directx_embedded::kBitonicSortDomain_cso,
		bps_directx_embedded::kBitonicSortDomain_cso_size);
}

bool BPS_LoadEmbeddedDirectXApplyDomainShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader)
{
	return BPS_LoadEmbeddedDirectXShader(
		context,
		out_shader,
		bps_directx_embedded::kBitonicApplyDomain_rs,
		bps_directx_embedded::kBitonicApplyDomain_rs_size,
		bps_directx_embedded::kBitonicApplyDomain_cso,
		bps_directx_embedded::kBitonicApplyDomain_cso_size);
}

bool BPS_LoadEmbeddedDirectXCopyShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader)
{
	return BPS_LoadEmbeddedDirectXShader(
		context,
		out_shader,
		bps_directx_embedded::kBitonicCopyInput_rs,
		bps_directx_embedded::kBitonicCopyInput_rs_size,
		bps_directx_embedded::kBitonicCopyInput_cso,
		bps_directx_embedded::kBitonicCopyInput_cso_size);
}

bool BPS_LoadEmbeddedDirectXMappedSortShader(
	const DXContextPtr &context,
	ShaderObjectPtr &out_shader)
{
	return BPS_LoadEmbeddedDirectXShader(
		context,
		out_shader,
		bps_directx_embedded::kBitonicSortMapped_rs,
		bps_directx_embedded::kBitonicSortMapped_rs_size,
		bps_directx_embedded::kBitonicSortMapped_cso,
		bps_directx_embedded::kBitonicSortMapped_cso_size);
}

#endif
