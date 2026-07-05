/*
	BPS_PerfDiag.cpp — scoped timing helpers for BPS_RENDER_DIAG builds.
*/

#include "BPS_PerfDiag.h"

#if defined(BPS_RENDER_DIAG)

#include <cstdarg>
#include <cstdio>

#if defined(_WIN32)
	#include <Windows.h>
#endif

void BPS_DiagLog(const char *format, ...)
{
	char message[1024];
	va_list args;
	va_start(args, format);
	std::vsnprintf(message, sizeof(message), format, args);
	va_end(args);

#if defined(_WIN32)
	OutputDebugStringA("[BitonicPixelSorter] ");
	OutputDebugStringA(message);
	OutputDebugStringA("\n");
#else
	FILE *file = std::fopen("/tmp/BitonicPixelSorter_render_diag.log", "a");
	if (file) {
		std::fprintf(file, "[BitonicPixelSorter] %s\n", message);
		std::fclose(file);
	}
#endif
}

BpsPerfScope::BpsPerfScope(const char *label)
	: label_(label), start_(std::chrono::steady_clock::now())
{
}

BpsPerfScope::~BpsPerfScope()
{
	const auto end = std::chrono::steady_clock::now();
	const double elapsed_ms =
		std::chrono::duration<double, std::milli>(end - start_).count();
	BPS_PerfLogMs(label_, elapsed_ms);
}

#endif
