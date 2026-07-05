/*
	BPS_PerfDiag.h

	Structured render diagnostics when BPS_RENDER_DIAG is enabled at compile time.
	SmartRender and PreRender log scoped timings via BPS_DiagLog.
*/

#pragma once

#if defined(BPS_RENDER_DIAG)

#include <chrono>
#include <cstdint>

void BPS_DiagLog(const char *format, ...);

class BpsPerfScope {
public:
	explicit BpsPerfScope(const char *label);
	~BpsPerfScope();

private:
	const char *label_;
	std::chrono::steady_clock::time_point start_;
};

#define BPS_PERF_SCOPE(label) BpsPerfScope _bps_perf_scope_##__LINE__(label)

inline void BPS_PerfLogMs(const char *label, double elapsed_ms)
{
	BPS_DiagLog("Perf %s elapsed_ms=%.3f", label, elapsed_ms);
}

#else

#define BPS_PERF_SCOPE(label) ((void)0)

inline void BPS_PerfLogMs(const char *, double) {}

#endif
