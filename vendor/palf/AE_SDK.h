// AE_SDK.h
//
// Small umbrella header that aggregates the Adobe After Effects SDK headers
// needed by AELocalise.h. Vendored (copied) from Palf_Plugins/_PalfLib/AE_SDK.h
// so this plugin stays self-contained and only references the Adobe SDK.
//
// AE_EffectSuites.h is added here (relative to the upstream copy) so that
// PFAppSuite6 / PF_AppGetLanguage are available regardless of include order.
#pragma once
#ifndef AE_SDK_H
#define AE_SDK_H

#include "AEConfig.h"
#include "entry.h"
#include "AEFX_SuiteHelper.h"
#include "PrSDKAESupport.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectSuites.h"
#include "AE_Macros.h"
#include "AEGP_SuiteHandler.h"
#include "String_Utils.h"
#include "Param_Utils.h"
#include "Smart_Utils.h"
#include "AE_ChannelSuites.h"

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#endif
