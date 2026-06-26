/*
	BitonicPixelSorter_GpuHost.cpp

	Host version recording and per-framework GPU device setup eligibility.
	Uses the SDK opt-out model: GlobalSetup/PiPL keep static GPU flags; each
	PF_Cmd_GPU_DEVICE_SETUP call decides whether to confirm GPU for that device.
*/

#include "BitonicPixelSorter_GpuEligibility.h"

#include <atomic>
#include <cstdlib>

namespace {

std::atomic<long> &BPS_HostMajorStorage()
{
	static std::atomic<long> value(0);
	return value;
}

std::atomic<long> &BPS_HostMinorStorage()
{
	static std::atomic<long> value(0);
	return value;
}

long
BPS_ParseHostVersion(const char *version, long *minor_out)
{
	if (minor_out) {
		*minor_out = 0;
	}

	if (!version || !*version) {
		return 0;
	}

	char *end = nullptr;
	const long major = std::strtol(version, &end, 10);
	if (end == version) {
		return 0;
	}

	long minor = 0;
	if (*end == '.') {
		char *minor_end = nullptr;
		const long parsed_minor = std::strtol(end + 1, &minor_end, 10);
		if (minor_end != end + 1) {
			minor = parsed_minor;
		}
	}

	if (minor_out) {
		*minor_out = minor;
	}

	return major;
}

} // namespace

void BPS_RecordHostVersion(const char *host_version)
{
	long minor = 0;
	const long major = BPS_ParseHostVersion(host_version, &minor);
	BPS_HostMajorStorage().store(major, std::memory_order_release);
	BPS_HostMinorStorage().store(minor, std::memory_order_release);
}

long BPS_HostMajorVersion()
{
	return BPS_HostMajorStorage().load(std::memory_order_acquire);
}

long BPS_HostMinorVersion()
{
	return BPS_HostMinorStorage().load(std::memory_order_acquire);
}

bool BPS_IsGpuFrameworkCompiled(PF_GPU_Framework framework)
{
	switch (framework) {
#if defined(BPS_HAS_CUDA)
	case PF_GPU_Framework_CUDA:
		return true;
#endif
#if defined(BPS_HAS_OPENCL)
	case PF_GPU_Framework_OPENCL:
		return true;
#endif
#if defined(BPS_HAS_HLSL)
	case PF_GPU_Framework_DIRECTX:
		return true;
#endif
#if defined(BPS_HAS_METAL)
	case PF_GPU_Framework_METAL:
		return true;
#endif
	default:
		break;
	}

	return false;
}

bool BPS_ShouldAcceptGpuDeviceSetup(
	const PF_InData *in_data,
	PF_GPU_Framework framework)
{
	if (!in_data || framework == PF_GPU_Framework_NONE) {
		return false;
	}

	if (in_data->appl_id == kAppID_Premiere) {
		return false;
	}

	return BPS_IsGpuFrameworkCompiled(framework);
}
