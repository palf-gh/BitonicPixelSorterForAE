// BitonicPixelSorter_Target.h
//
// Plugin identity, version and global out-flags.
// This header contains ONLY preprocessor #defines so it can be included by both
// the C++ sources and the PiPL resource (.r), which is processed by a C
// preprocessor (MSVC /EP on Windows, clang -E on macOS) before PiPLtool / Rez.
//
// Keep it free of C/C++ declarations.
//
// clang-format off
#pragma once
#ifndef BITONIC_PIXEL_SORTER_TARGET_H
#define BITONIC_PIXEL_SORTER_TARGET_H

//-----------------------------------------------------------------------------
// Identity
//-----------------------------------------------------------------------------
#if defined(_DEBUG) || defined(DEBUG) || defined(BPS_DEBUG_BUILD)
	#define NAME			"Bitonic Pixel Sorter debug"
	#define MATCHNAME		"PALF BitonicPixelSorter debug"
#else
	#define NAME			"Bitonic Pixel Sorter"
	#define MATCHNAME		"PALF BitonicPixelSorter"
#endif

#define DESCRIPTION		"\nGPU-accelerated bitonic pixel sort.\rPorted from ruccho/BitonicPixelSorter (MIT)."
#define CATEGORY		"Stylize"
#define SUPPORT_URL		"https://x.com/PALF_MovieWorks"

//-----------------------------------------------------------------------------
// Version (stage: DEVELOP 0, PRERELEASE 1, ALPHA 2, BETA 3, RELEASE 4 in PF_VERSION;
// PiPL uses PF_Stage_DEVELOP=0 ... RELEASE=3 numeric below)
//-----------------------------------------------------------------------------
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	0		// PF_Stage_DEVELOP (numeric, so the PiPL preprocessor needs no AE enums)
#define	BUILD_VERSION	1

// Packed version, identical bit layout to PF_VERSION() in AE_EffectVers.h.
// Used by the PiPL resource (AE_Effect_Version) and GlobalSetup (my_version).
// PiPLtool's expression parser does not accept shifts or the 'L' suffix, so this
// is a precomputed literal. For (1,0,0,0,1): (1<<19)|1 = 524289.
#define	BPS_VERSION_PACKED	524289

//-----------------------------------------------------------------------------
// Global out-flags (see AE_Effect.h). Written as explicit bit shifts so the
// value is identical in the PiPL resource and in GlobalSetup() without needing
// the AE_Effect.h enums (which are C++-only) inside the .r preprocessing pass.
//
// GPU flags (PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 = 1<<25,
//            PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING = 1<<29) are added at runtime
// in GlobalSetup() for non-Premiere hosts once the GPU phase lands.
//-----------------------------------------------------------------------------
// Literals (not shift expressions): PiPLtool's parser rejects '<<' and 'L'.
//
// OUT_FLAGS = PF_OutFlag_DEEP_COLOR_AWARE (1<<25) = 33554432.
// NOTE: deliberately NOT PF_OutFlag_PIX_INDEPENDENT — pixel sorting moves pixels
// along a line, so an output pixel depends on its neighbours (not independent).
#define OUT_FLAGS		33554432

// OUT_FLAGS2 = PF_OutFlag2_SUPPORTS_SMART_RENDER (1<<10 = 1024)
//   | PF_OutFlag2_FLOAT_COLOR_AWARE (1<<12 = 4096)
//   | PF_OutFlag2_SUPPORTS_THREADED_RENDERING (1<<27 = 134217728) = 134222848.
#define OUT_FLAGS2		134222848

#endif // BITONIC_PIXEL_SORTER_TARGET_H
// clang-format on
