// AELocalise.h
//
// Vendored (copied) from Palf_Plugins/_PalfLib/AELocalise.h for self-containment.
// Header-only CJK+EN (and more) localisation helper for After Effects plugins.
// Only dependency is the Adobe SDK (via AE_SDK.h) plus the OS text APIs.
//
// このヘッダーは、プロジェクト側で以下の順序でインクルードすることで、
// ローカライズ機能を提供します：
//
// 使用方法：
//   1. LocKey列挙型を定義したヘッダーをインクルード
//      #include "Localise/LocKeys.h"
//   2. 必要な言語の文字列リソースヘッダーをインクルード（すべて必要ではない）
//      #include "Localise/Strings_en_US.h"   // 英語（必須）
//      #include "Localise/Strings_ja_JP.h"   // 日本語（オプション）
//      #include "Localise/Strings_zh_CN.h"   // 中国語(簡体字)（オプション）
//      #include "Localise/Strings_ko_KR.h"   // 韓国語（オプション）
//   3. このヘッダーをインクルード
//      #include "vendor/palf/AELocalise.h"
//
// 注意：このヘッダーは、LocKey型と各言語の名前空間（EN_US, JA_JP, ZH_CN,
// KO_KR等）が定義されていることを前提とします。定義されていない言語は英語に
// フォールバックします。
#pragma once
#ifndef AELOCALISE_H
#define AELOCALISE_H

#include "AE_SDK.h"
#include <atomic>
#include <cstring>
#include <string>

// PF_AppSuiteの定義のために必要
#ifdef AE_OS_WIN
#include <Windows.h>
#include <vector>
#else
#include <Carbon/Carbon.h>
#include <CoreFoundation/CoreFoundation.h>
#endif

namespace AELocalise {
// PF_AppGetLanguage()を使用して現在のロケールIDを取得
// 返り値はロケールID（"ja_JP", "en_US", "zh_CN"など）
inline std::string GetCurrentLanguage(PF_InData *in_dataP) {
  if (!in_dataP || !in_dataP->pica_basicP) {
    return "en_US"; // デフォルトは英語（米国）
  }

  std::string result = "en_US"; // デフォルトは英語（米国）
  A_char langTag[PF_APP_LANG_TAG_SIZE] = {0};
  PF_Err err = PF_Err_NONE;

  PFAppSuite6 *app_suiteP = NULL;
  err = AEFX_AcquireSuite(in_dataP, NULL, kPFAppSuite, kPFAppSuiteVersion6,
                          NULL, (void **)&app_suiteP);

  if (!err && app_suiteP) {
    err = app_suiteP->PF_AppGetLanguage(langTag);

    if (err == PF_Err_NONE && strlen(langTag) > 0) {
      result = langTag;
    }

    (void)AEFX_ReleaseSuite(in_dataP, NULL, kPFAppSuite, kPFAppSuiteVersion6,
                            NULL);
  }

  return result;
}

#ifdef AE_OS_WIN
namespace Internal {
inline std::wstring UTF16PathToWideString(const A_UTF16Char *pathZ) {
  std::wstring result;
  if (!pathZ)
    return result;

  while (*pathZ) {
    result.push_back(static_cast<wchar_t>(*pathZ));
    ++pathZ;
  }
  return result;
}

inline std::wstring GetHostExecutablePath(PF_InData *in_dataP) {
  wchar_t modulePath[MAX_PATH] = {0};
  DWORD length = GetModuleFileNameW(NULL, modulePath, MAX_PATH);
  if (length > 0 && length < MAX_PATH)
    return modulePath;

  A_UTF16Char sdkPath[AEFX_MAX_PATH] = {0};

  if (in_dataP && in_dataP->utils && in_dataP->utils->get_platform_data) {
    PF_Err err = in_dataP->utils->get_platform_data(
        in_dataP->effect_ref, PF_PlatData_EXE_FILE_PATH_W, sdkPath);
    if (err == PF_Err_NONE && sdkPath[0]) {
      return UTF16PathToWideString(sdkPath);
    }
  }

  return L"";
}

inline bool GetWindowsFileMajorVersion(const std::wstring &path,
                                       A_long *majorP) {
  if (path.empty() || !majorP)
    return false;

  HMODULE versionDll = LoadLibraryW(L"version.dll");
  if (!versionDll)
    return false;

  using GetFileVersionInfoSizeWProc = DWORD(WINAPI *)(LPCWSTR, LPDWORD);
  using GetFileVersionInfoWProc = BOOL(WINAPI *)(LPCWSTR, DWORD, DWORD, LPVOID);
  using VerQueryValueWProc = BOOL(WINAPI *)(LPCVOID, LPCWSTR, LPVOID *, PUINT);

  auto getFileVersionInfoSizeW =
      reinterpret_cast<GetFileVersionInfoSizeWProc>(
          GetProcAddress(versionDll, "GetFileVersionInfoSizeW"));
  auto getFileVersionInfoW = reinterpret_cast<GetFileVersionInfoWProc>(
      GetProcAddress(versionDll, "GetFileVersionInfoW"));
  auto verQueryValueW = reinterpret_cast<VerQueryValueWProc>(
      GetProcAddress(versionDll, "VerQueryValueW"));

  bool success = false;
  if (getFileVersionInfoSizeW && getFileVersionInfoW && verQueryValueW) {
    DWORD handle = 0;
    DWORD infoSize = getFileVersionInfoSizeW(path.c_str(), &handle);
    if (infoSize > 0) {
      std::vector<BYTE> versionInfo(infoSize);
      if (getFileVersionInfoW(path.c_str(), 0, infoSize,
                              versionInfo.data())) {
        VS_FIXEDFILEINFO *fixedInfo = nullptr;
        UINT fixedInfoSize = 0;
        if (verQueryValueW(versionInfo.data(), L"\\",
                           reinterpret_cast<LPVOID *>(&fixedInfo),
                           &fixedInfoSize) &&
            fixedInfo && fixedInfoSize >= sizeof(VS_FIXEDFILEINFO) &&
            fixedInfo->dwSignature == 0xfeef04bd) {
          *majorP = HIWORD(fixedInfo->dwFileVersionMS);
          success = true;
        }
      }
    }
  }

  FreeLibrary(versionDll);
  return success;
}

inline bool GetHostExecutableMajorVersion(PF_InData *in_dataP,
                                          A_long *majorP) {
  return GetWindowsFileMajorVersion(GetHostExecutablePath(in_dataP), majorP);
}

inline bool ContainsAsciiCaseInsensitive(const std::wstring &value,
                                         const wchar_t *needle) {
  if (!needle || !*needle)
    return true;

  std::wstring loweredNeedle;
  for (const wchar_t *p = needle; *p; ++p) {
    wchar_t ch = *p;
    if (ch >= L'A' && ch <= L'Z')
      ch = static_cast<wchar_t>(ch - L'A' + L'a');
    loweredNeedle.push_back(ch);
  }

  for (size_t i = 0; i + loweredNeedle.size() <= value.size(); ++i) {
    bool matches = true;
    for (size_t j = 0; j < loweredNeedle.size(); ++j) {
      wchar_t ch = value[i + j];
      if (ch >= L'A' && ch <= L'Z')
        ch = static_cast<wchar_t>(ch - L'A' + L'a');
      if (ch != loweredNeedle[j]) {
        matches = false;
        break;
      }
    }
    if (matches)
      return true;
  }
  return false;
}

inline bool HostExecutablePathLooksLikeBeta(const std::wstring &path) {
  return ContainsAsciiCaseInsensitive(path, L"beta");
}

inline std::string WideStringToUTF8(const std::wstring &value) {
  if (value.empty())
    return "";

  int size =
      WideCharToMultiByte(CP_UTF8, 0, value.c_str(),
                          static_cast<int>(value.size()), nullptr, 0, nullptr,
                          nullptr);
  if (size <= 0)
    return "";

  std::string result(size, 0);
  if (WideCharToMultiByte(CP_UTF8, 0, value.c_str(),
                          static_cast<int>(value.size()), &result[0], size,
                          nullptr, nullptr) <= 0)
    return "";

  return result;
}

inline bool DebugLoggingEnabled() {
  char value[16] = {0};
  DWORD length = GetEnvironmentVariableA("PALF_AELOCALISE_DEBUG", value,
                                         static_cast<DWORD>(sizeof(value)));
  return length > 0 && value[0] != '0';
}

inline void AppendDebugLogLine(const std::string &line) {
  if (line.empty())
    return;

  std::wstring debugLine(line.begin(), line.end());
  debugLine += L"\r\n";
  OutputDebugStringW(debugLine.c_str());

  wchar_t tempPath[MAX_PATH] = {0};
  DWORD tempLength = GetTempPathW(MAX_PATH, tempPath);
  if (tempLength == 0 || tempLength >= MAX_PATH)
    return;

  std::wstring logPath(tempPath);
  logPath += L"Palf_AELocalise.log";

  HANDLE fileH =
      CreateFileW(logPath.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ, nullptr,
                  OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (fileH == INVALID_HANDLE_VALUE)
    return;

  std::string fileLine = line + "\r\n";
  DWORD written = 0;
  WriteFile(fileH, fileLine.data(), static_cast<DWORD>(fileLine.size()),
            &written, nullptr);
  CloseHandle(fileH);
}

inline void LogHostTextEncodingDecisionOnce(const std::wstring &executablePath,
                                            bool isAfterEffects,
                                            bool looksBeta,
                                            bool versionDetected,
                                            A_long detectedMajor,
                                            bool usesUTF8,
                                            const char *reason) {
  if (!DebugLoggingEnabled())
    return;

  static std::atomic<long> logged{0};
  long expected = 0;
  if (!logged.compare_exchange_strong(expected, 1, std::memory_order_acq_rel,
                                      std::memory_order_relaxed))
    return;

  std::string line = "[Palf AELocalise] is_after_effects=";
  line += isAfterEffects ? "1" : "0";
  line += " beta_path=";
  line += looksBeta ? "1" : "0";
  line += " version_detected=";
  line += versionDetected ? "1" : "0";
  line += " major=";
  line += versionDetected ? std::to_string(detectedMajor) : "unknown";
  line += " uses_utf8=";
  line += usesUTF8 ? "1" : "0";
  line += " reason=";
  line += reason ? reason : "unknown";
  line += " path=\"";
  line += WideStringToUTF8(executablePath);
  line += "\"";

  AppendDebugLogLine(line);
}

inline bool HostUsesUTF8ForAEText(PF_InData *in_dataP) {
  const bool isAfterEffects =
      in_dataP && in_dataP->appl_id == kAppID_AfterEffects;
  if (!isAfterEffects) {
    LogHostTextEncodingDecisionOnce(L"", false, false, false, 0, false,
                                    "not-after-effects");
    return false;
  }

  static std::atomic<long> cachedUsesUTF8{-1};
  long usesUTF8 = cachedUsesUTF8.load(std::memory_order_acquire);
  if (usesUTF8 < 0) {
    std::wstring executablePath = GetHostExecutablePath(in_dataP);
    bool looksBeta = HostExecutablePathLooksLikeBeta(executablePath);
    bool versionDetected = false;
    A_long detectedMajor = 0;
    const char *reason = "legacy-version";

    if (looksBeta) {
      usesUTF8 = 1;
      reason = "beta-path";
    } else {
      versionDetected =
          GetWindowsFileMajorVersion(executablePath, &detectedMajor);
      if (!versionDetected) {
        LogHostTextEncodingDecisionOnce(executablePath, true, false, false, 0,
                                        false, "version-detection-failed");
        return false;
      }
      usesUTF8 = detectedMajor >= 26 ? 1 : 0;
      if (usesUTF8)
        reason = "major-version";
    }

    long expected = -1;
    cachedUsesUTF8.compare_exchange_strong(
        expected, usesUTF8, std::memory_order_release,
        std::memory_order_relaxed);
    usesUTF8 = cachedUsesUTF8.load(std::memory_order_acquire);
    if (usesUTF8 < 0) {
      LogHostTextEncodingDecisionOnce(executablePath, true, looksBeta,
                                      versionDetected, detectedMajor, false,
                                      "cache-not-set");
      return false;
    }

    LogHostTextEncodingDecisionOnce(executablePath, true, looksBeta,
                                    versionDetected, detectedMajor,
                                    usesUTF8 != 0, reason);
  }

  return usesUTF8 != 0;
}
} // namespace Internal

// 言語IDからコードページを取得
inline UINT GetCodePageForLanguage(const std::string &localeId) {
  if (localeId == "ja_JP")
    return 932; // Shift-JIS
  else if (localeId == "zh_CN")
    return 936; // GBK
  else if (localeId == "ko_KR")
    return 949; // EUC-KR
  else if (localeId == "ru_RU")
    return 1251; // Cyrillic
  else
    return 1252; // Latin-1 (デフォルト)
}

// Windows: UTF-8から指定されたコードページへ変換
inline std::string ConvertUTF8ToEncoding(const std::string &utf8,
                                         UINT codePage) {
  if (utf8.empty())
    return "";

  int wideSize = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
  if (wideSize <= 0)
    return utf8;

  std::wstring wideStr(wideSize, 0);
  if (MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wideStr[0],
                          wideSize) <= 0)
    return utf8;

  int ansiSize = WideCharToMultiByte(codePage, 0, wideStr.c_str(), -1, nullptr,
                                     0, nullptr, nullptr);
  if (ansiSize <= 0)
    return utf8;

  std::string ansiStr(ansiSize, 0);
  if (WideCharToMultiByte(codePage, 0, wideStr.c_str(), -1, &ansiStr[0],
                          ansiSize, nullptr, nullptr) <= 0)
    return utf8;

  if (!ansiStr.empty() && ansiStr.back() == '\0')
    ansiStr.pop_back();

  return ansiStr;
}
#else
// 言語IDからエンコーディングを取得（macOS）
inline CFStringEncoding GetEncodingForLanguage(const std::string &localeId) {
  if (localeId == "ja_JP")
    return kCFStringEncodingShiftJIS;
  else if (localeId == "zh_CN")
    return kCFStringEncodingGB_18030_2000;
  else if (localeId == "ko_KR")
    return kCFStringEncodingEUC_KR;
  else if (localeId == "ru_RU")
    return kCFStringEncodingWindowsCyrillic;
  else
    return kCFStringEncodingMacRoman; // デフォルト
}

// macOS: UTF-8から指定されたエンコーディングへ変換
inline std::string ConvertUTF8ToEncoding(const std::string &utf8,
                                         CFStringEncoding encoding) {
  if (utf8.empty())
    return "";

  CFStringRef cfStr = CFStringCreateWithBytes(
      kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(utf8.c_str()),
      utf8.length(), kCFStringEncodingUTF8, false);

  if (!cfStr)
    return utf8;

  CFIndex maxLen =
      CFStringGetMaximumSizeForEncoding(CFStringGetLength(cfStr), encoding);

  if (maxLen <= 0) {
    CFRelease(cfStr);
    return utf8;
  }

  std::string result(maxLen, 0);
  CFIndex usedBufLen = 0;
  Boolean converted = CFStringGetBytes(
      cfStr, CFRangeMake(0, CFStringGetLength(cfStr)), encoding, 0, false,
      reinterpret_cast<UInt8 *>(&result[0]), maxLen, &usedBufLen);

  CFRelease(cfStr);

  if (!converted)
    return utf8;

  result.resize(usedBufLen);
  return result;
}
#endif

// 内部：変換後の文字列を一時バッファに保存（スレッドローカルストレージを使用）
namespace Internal {
thread_local static std::string s_tempBuffer;
}

// 内部：ロケールIDから文字列取得関数へのマッピング
// 定義されていない言語は英語にフォールバック
template <typename LocKeyType>
inline const char *GetStringByLanguage(LocKeyType key,
                                       const std::string &localeId) {
  if (localeId == "en_US") {
#ifdef AELOCALISE_HAS_EN_US
    return EN_US::GetString(key);
#else
    return "";
#endif
  } else if (localeId == "ja_JP") {
#ifdef AELOCALISE_HAS_JA_JP
    return JA_JP::GetString(key);
#else
    return "";
#endif
  } else if (localeId == "ko_KR") {
#ifdef AELOCALISE_HAS_KO_KR
    return KO_KR::GetString(key);
#else
    return "";
#endif
  } else if (localeId == "zh_CN") {
#ifdef AELOCALISE_HAS_ZH_CN
    return ZH_CN::GetString(key);
#else
    return "";
#endif
  }
#ifdef AELOCALISE_HAS_DE_DE
  else if (localeId == "de_DE") {
    return DE_DE::GetString(key);
  }
#endif
#ifdef AELOCALISE_HAS_ES_ES
  else if (localeId == "es_ES") {
    return ES_ES::GetString(key);
  }
#endif
#ifdef AELOCALISE_HAS_FR_FR
  else if (localeId == "fr_FR") {
    return FR_FR::GetString(key);
  }
#endif
#ifdef AELOCALISE_HAS_IT_IT
  else if (localeId == "it_IT") {
    return IT_IT::GetString(key);
  }
#endif
#ifdef AELOCALISE_HAS_PT_BR
  else if (localeId == "pt_BR") {
    return PT_BR::GetString(key);
  }
#endif
#ifdef AELOCALISE_HAS_RU_RU
  else if (localeId == "ru_RU") {
    return RU_RU::GetString(key);
  }
#endif

  // デフォルトは英語
  return EN_US::GetString(key);
}

// キーからローカライズ文字列を取得（UTF-8文字列）
template <typename LocKeyType> inline const char *GetString(LocKeyType key) {
  return EN_US::GetString(key);
}

// ローカライズ文字列を取得し、AEが受け付ける文字コードで返す。
// Windows版AE BetaまたはAE 26以降ではUTF-8を返し、それ以前は言語ごとのコードページへ変換する。
template <typename LocKeyType>
inline const char *GetStringForAE(LocKeyType key, PF_InData *in_dataP) {
  std::string localeId = GetCurrentLanguage(in_dataP);
  const char *utf8Str = GetStringByLanguage(key, localeId);

  if (!utf8Str || strlen(utf8Str) == 0) {
    utf8Str = EN_US::GetString(key);
    localeId = "en_US";
  }

  if (!utf8Str)
    return "";

#ifdef AE_OS_WIN
  if (Internal::HostUsesUTF8ForAEText(in_dataP)) {
    Internal::s_tempBuffer = utf8Str;
    return Internal::s_tempBuffer.c_str();
  }

  UINT codePage = GetCodePageForLanguage(localeId);
  Internal::s_tempBuffer = ConvertUTF8ToEncoding(utf8Str, codePage);
#else
  Internal::s_tempBuffer = utf8Str;
#endif
  return Internal::s_tempBuffer.c_str();
}
} // namespace AELocalise

#endif // AELOCALISE_H
