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
// Version (PF_Stage: DEVELOP=0, ALPHA=1, BETA=2, RELEASE=3)
//-----------------------------------------------------------------------------
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	1
#define	BUG_VERSION		1
#define	STAGE_VERSION	3		// PF_Stage_RELEASE (numeric, so the PiPL preprocessor needs no AE enums)
#define	BUILD_VERSION	1

// Packed version, identical bit layout to PF_VERSION() in AE_Effect.h.
// Used by the PiPL resource (AE_Effect_Version) and GlobalSetup (my_version).
// PiPLtool's expression parser does not accept shifts or the 'L' suffix, so this
// is a precomputed literal.
// PF_VERSION(1,1,1,3,1):
//   vers<<19 | subvers<<15 | bugvers<<11 | stage<<9 | build
//   = (1<<19)|(1<<15)|(1<<11)|(3<<9)|1 = 560641.
#define	BPS_VERSION_PACKED	560641

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

// OUT_FLAGS2 base = PF_OutFlag2_SUPPORTS_SMART_RENDER (1<<10 = 1024)
//   | PF_OutFlag2_FLOAT_COLOR_AWARE (1<<12 = 4096)
//   | PF_OutFlag2_SUPPORTS_THREADED_RENDERING (1<<27 = 134217728)
//   = 134222848.
//
// Threaded rendering advertises AE Multi-Frame Rendering support. PreRender and
// SmartRender can run on non-main threads concurrently with the UI, so render
// code must remain per-call/stateless or explicitly synchronised.
//
// Precomputed literals only: Rez must not see addition expressions, or PiPL
// and GlobalSetup can disagree and AE reports a version mismatch (84601).
#if defined(BPS_HAS_HLSL)
	// base (134222848) | GPU (1<<25 = 33554432) | DirectX (1<<29 = 536870912)
	// = 704648192.
	#define OUT_FLAGS2		704648192
#elif defined(BPS_HAS_CUDA) || defined(BPS_HAS_OPENCL) || defined(BPS_HAS_METAL)
	// base | GPU (1<<25 = 33554432)
	#define OUT_FLAGS2		167777280
#else
	#define OUT_FLAGS2		134222848
#endif

#endif // BITONIC_PIXEL_SORTER_TARGET_H
// clang-format on
