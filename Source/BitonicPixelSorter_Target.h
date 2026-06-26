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
// Host compatibility
//-----------------------------------------------------------------------------
// The SDK headers advertise the newest plug-in API supported by this SDK
// (currently 13.29 / AE 23.5). Older AE 2023 builds can reject a PiPL that
// declares that newer requirement before they ever call EffectMain. Keep the
// PiPL requirement at the AE 22.0 floor used by this plug-in's feature set;
// the runtime PluginDataEntryFunction2 still reports the SDK-mandated current
// PF_AE_PLUG_IN_VERSION/SUBVERS pair on hosts that support it.
#define BPS_PIPL_SPEC_VERSION			13
#define BPS_PIPL_SPEC_SUBVERS			27

// AE_Effect_Support_URL was added with API 13.28. The v2 entry point still
// supplies it to modern hosts, but the PiPL omits it for AE 2023.0-23.3 scans.
#define BPS_PIPL_HAS_SUPPORT_URL		0

// Legacy PluginDataEntryFunction fallback for older hosts that do not use the
// v2 entry point. Newer hosts receive the SDK's current version via v2.
#define BPS_LEGACY_PLUGIN_API_VERSION	13
#define BPS_LEGACY_PLUGIN_API_SUBVERS	27

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
// Global out-flags (see AE_Effect.h). Written as explicit integer literals so
// the PiPL resource and GlobalSetup() resolve to the same value. AE validates
// these two sources at load time and rejects any mismatch.
//-----------------------------------------------------------------------------
// Literals only: PiPLtool's parser rejects '<<' and 'L'.
//
// OUT_FLAGS = PF_OutFlag_DEEP_COLOR_AWARE (1<<25 = 33554432)
//   | PF_OutFlag_CUSTOM_UI (1<<15 = 32768)
//   | PF_OutFlag_SEND_UPDATE_PARAMS_UI (1<<26 = 67108864)
//   = 100696064.
// NOTE: deliberately NOT PF_OutFlag_PIX_INDEPENDENT — pixel sorting moves pixels
// along a line, so an output pixel depends on its neighbours (not independent).
#define OUT_FLAGS		100696064

// OUT_FLAGS2 = PF_OutFlag2_SUPPORTS_SMART_RENDER (1<<10 = 1024)
//   | PF_OutFlag2_FLOAT_COLOR_AWARE (1<<12 = 4096)
//   | PF_OutFlag2_SUPPORTS_THREADED_RENDERING (1<<27 = 134217728)
//   = 134222848.
//
// Threaded rendering advertises AE Multi-Frame Rendering support. PreRender and
// SmartRender can run on non-main threads concurrently with the UI, so render
// code must remain per-call/stateless or explicitly synchronised.
#define OUT_FLAGS2_BASE	134222848

#if defined(BPS_HAS_CUDA) || defined(BPS_HAS_OPENCL) || defined(BPS_HAS_HLSL) || defined(BPS_HAS_METAL)
	// PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 (1<<25) = 33554432.
	#define OUT_FLAGS2_GPU 33554432
#else
	#define OUT_FLAGS2_GPU 0
#endif

#if defined(BPS_HAS_HLSL)
	// PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING (1<<29) = 536870912.
	#define OUT_FLAGS2_DIRECTX 536870912
#else
	#define OUT_FLAGS2_DIRECTX 0
#endif

#define OUT_FLAGS2		(OUT_FLAGS2_BASE + OUT_FLAGS2_GPU + OUT_FLAGS2_DIRECTX)

#endif // BITONIC_PIXEL_SORTER_TARGET_H
// clang-format on
