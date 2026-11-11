// AELocalise.h
//
// Header-only CJK+EN (and more) localisation helper for After Effects plugins.
// Kept in sync with Palf_Plugins/_PalfLib/AELocalise.h and
// AETextEncodingPolicy.h, and duplicated in-tree so this plugin stays
// self-contained and only references the Adobe SDK.
//
// AE 26 (driver major 126) always receives UTF-8. AE 27+ (driver major 127+)
// receives UTF-8 only after the effect declares PF_OutFlag2_STRINGS_ARE_UTF8 in
// both PiPL and GLOBAL_SETUP and calls AELocalise::SetUTF8FlagDeclared(true).
#pragma once
#ifndef AELOCALISE_H
#define AELOCALISE_H

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
//      その他の言語（DE_DE, ES_ES, FR_FR, IT_IT, PT_BR,
//      RU_RU）も同様にオプション
//   3. このヘッダーをインクルード
//      #include "Localise/AELocalise.h"
//
// 注意：このヘッダーは、LocKey型と各言語の名前空間（EN_US, JA_JP, ZH_CN,
// KO_KR等）が
// 定義されていることを前提とします。定義されていない言語は英語にフォールバックします。
//
// 対応ロケールID：
//   - de_DE: ドイツ語 (German)
//   - en_US: 英語 (English)
//   - es_ES: スペイン語 (Spanish)
//   - fr_FR: フランス語 (French)
//   - it_IT: イタリア語 (Italian)
//   - pt_BR: ポルトガル語(ブラジル) (Portuguese (Brazil))
//   - ru_RU: ロシア語 (Russian)
//   - ja_JP: 日本語 (Japanese)
//   - ko_KR: 韓国語 (Korean)
//   - zh_CN: 中国語(簡体字) (Chinese (Simplified))

#include "AE_SDK.h"
#include "AETextEncodingPolicy.h"
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
#include <cstdio>
#include <cstdlib>
#endif

// AEFX_SuiteHelperを使用してPF_AppSuiteを取得
// (AE_SDK.hに既に含まれている可能性があるが、明示的にインクルード)
// AEFX_AcquireSuite と AEFX_ReleaseSuite を使用するため

namespace AELocalise {

namespace Internal {
inline std::atomic<bool> utf8FlagDeclared{false};
// AE reports 125.x, 126.x and 127.x for product versions 25, 26 and 27.
// A failed acquisition is not cached so a later callback can retry.
inline bool GetAEDriverMajorVersion(PF_InData *in_dataP, A_long *majorP) {
  if (!in_dataP || !in_dataP->pica_basicP || !majorP ||
      in_dataP->appl_id != kAppID_AfterEffects)
    return false;
  const AEGP_UtilitySuite6 *suite = nullptr;
  auto *basic = in_dataP->pica_basicP;
  const A_Err err = basic->AcquireSuite(kAEGPUtilitySuite, kAEGPUtilitySuiteVersion6,
                                        reinterpret_cast<const void **>(&suite));
  if (err || !suite)
    return false;
  A_short major = 0, minor = 0;
  const A_Err versionErr = suite->AEGP_GetDriverImplementationVersion(&major, &minor);
  basic->ReleaseSuite(kAEGPUtilitySuite, kAEGPUtilitySuiteVersion6);
  if (versionErr || major <= 0)
    return false;
  *majorP = major;
  return true;
}
} // namespace Internal

// Call from GLOBAL_SETUP when this effect declares PF_OutFlag2_STRINGS_ARE_UTF8
// in both PiPL and its runtime out_flags2. Each plug-in DLL owns its state.
inline void SetUTF8FlagDeclared(bool declared) {
  Internal::utf8FlagDeclared.store(declared, std::memory_order_release);
}
// PF_AppGetLanguage()を使用して現在のロケールIDを取得
// 返り値はロケールID（"ja_JP", "en_US", "zh_CN"など）
inline std::string GetCurrentLanguage(PF_InData *in_dataP) {
  if (!in_dataP || !in_dataP->pica_basicP) {
    return "en_US"; // デフォルトは英語（米国）
  }

  // PF_AppGetLanguage()を使用して言語を取得
  // AEFX_AcquireSuiteを使用してPFAppSuite6を取得
  std::string result = "en_US"; // デフォルトは英語（米国）
  A_char langTag[PF_APP_LANG_TAG_SIZE] = {0};
  PF_Err err = PF_Err_NONE;

  // PFAppSuite6を取得（PF_AppGetLanguageはPFAppSuite6に含まれる）
  PFAppSuite6 *app_suiteP = NULL;
  err = AEFX_AcquireSuite(in_dataP, NULL, kPFAppSuite, kPFAppSuiteVersion6,
                          NULL, (void **)&app_suiteP);

  if (!err && app_suiteP) {
    // PF_AppGetLanguage()を呼び出す
    err = app_suiteP->PF_AppGetLanguage(langTag);

    if (err == PF_Err_NONE && strlen(langTag) > 0) {
      // langTagをそのままロケールIDとして使用（正規化しない）
      result = langTag;
    }

    // PFAppSuite6を解放
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
  if (!in_dataP || in_dataP->appl_id != kAppID_AfterEffects) {
    LogHostTextEncodingDecisionOnce(L"", false, false, false, 0, false,
                                   "not-after-effects");
    return false;
  }
  static std::atomic<long> cachedDriverMajor{-1};
  long major = cachedDriverMajor.load(std::memory_order_acquire);
  if (major < 0) {
    const auto executablePath = GetHostExecutablePath(in_dataP);
    const bool looksBeta = HostExecutablePathLooksLikeBeta(executablePath);
    A_long detectedMajor = 0;
    const char *reason = "driver-version";
    if (!GetAEDriverMajorVersion(in_dataP, &detectedMajor)) {
      A_long fallbackMajor = 0;
      if (!GetWindowsFileMajorVersion(executablePath, &fallbackMajor)) {
        LogHostTextEncodingDecisionOnce(executablePath, true, looksBeta, false, 0,
                                       false, "version-detection-failed");
        return false;
      }
      detectedMajor = fallbackMajor + 100;
      reason = "product-version-fallback";
    }
    long expected = -1;
    cachedDriverMajor.compare_exchange_strong(expected, detectedMajor,
                                             std::memory_order_release,
                                             std::memory_order_relaxed);
    major = cachedDriverMajor.load(std::memory_order_acquire);
    const bool utf8 = UseUTF8ForAEDriver(
        major, utf8FlagDeclared.load(std::memory_order_acquire));
    LogHostTextEncodingDecisionOnce(executablePath, true, looksBeta, true,
                                   major, utf8, reason);
  }
  // AE 26 predates the flag but requires UTF-8. On later hosts opt in only
  // after the effect has declared the flag in PiPL and GLOBAL_SETUP.
  return UseUTF8ForAEDriver(
      major, utf8FlagDeclared.load(std::memory_order_acquire));
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

  // UTF-8 → ワイド文字
  int wideSize = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
  if (wideSize <= 0)
    return utf8; // 変換失敗時は元の文字列を返す

  std::wstring wideStr(wideSize, 0);
  if (MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wideStr[0],
                          wideSize) <= 0)
    return utf8;

  // ワイド文字 → 指定されたコードページ
  int ansiSize = WideCharToMultiByte(codePage, 0, wideStr.c_str(), -1, nullptr,
                                     0, nullptr, nullptr);
  if (ansiSize <= 0)
    return utf8;

  std::string ansiStr(ansiSize, 0);
  if (WideCharToMultiByte(codePage, 0, wideStr.c_str(), -1, &ansiStr[0],
                          ansiSize, nullptr, nullptr) <= 0)
    return utf8;

  // null終端を削除
  if (!ansiStr.empty() && ansiStr.back() == '\0')
    ansiStr.pop_back();

  return ansiStr;
}
#else
namespace Internal {
inline std::string UTF16PathToUTF8(const A_UTF16Char *pathZ) {
  if (!pathZ || !pathZ[0])
    return "";

  size_t length = 0;
  while (pathZ[length])
    ++length;

  CFStringRef cfStr = CFStringCreateWithCharacters(
      kCFAllocatorDefault, reinterpret_cast<const UniChar *>(pathZ),
      static_cast<CFIndex>(length));
  if (!cfStr)
    return "";

  CFIndex maxLen = CFStringGetMaximumSizeForEncoding(
                       CFStringGetLength(cfStr), kCFStringEncodingUTF8) +
                   1;
  std::string result(static_cast<size_t>(maxLen), '\0');
  if (!CFStringGetCString(cfStr, &result[0], maxLen, kCFStringEncodingUTF8)) {
    CFRelease(cfStr);
    return "";
  }
  CFRelease(cfStr);
  result.resize(std::strlen(result.c_str()));
  return result;
}

inline std::string GetPathStringFromCFURL(CFURLRef url) {
  if (!url)
    return "";

  char path[4096] = {0};
  if (CFURLGetFileSystemRepresentation(url, true,
                                       reinterpret_cast<UInt8 *>(path),
                                       sizeof(path)))
    return path;
  return "";
}

inline std::string GetHostExecutablePath(PF_InData *in_dataP) {
  A_UTF16Char sdkPath[AEFX_MAX_PATH] = {0};

  if (in_dataP && in_dataP->utils && in_dataP->utils->get_platform_data) {
    PF_Err err = in_dataP->utils->get_platform_data(
        in_dataP->effect_ref, PF_PlatData_EXE_FILE_PATH_W, sdkPath);
    if (err == PF_Err_NONE && sdkPath[0]) {
      const std::string path = UTF16PathToUTF8(sdkPath);
      if (!path.empty() && path.find(".plugin") == std::string::npos)
        return path;
    }
  }

  // Loaded plug-ins run inside the host process; the main bundle is After
  // Effects, not the .plugin bundle (contrast Windows GetModuleFileNameW(NULL)).
  CFBundleRef mainBundle = CFBundleGetMainBundle();
  if (mainBundle) {
    CFURLRef exeURL = CFBundleCopyExecutableURL(mainBundle);
    if (exeURL) {
      const std::string path = GetPathStringFromCFURL(exeURL);
      CFRelease(exeURL);
      if (!path.empty())
        return path;
    }

    CFURLRef bundleURL = CFBundleCopyBundleURL(mainBundle);
    if (bundleURL) {
      const std::string path = GetPathStringFromCFURL(bundleURL);
      CFRelease(bundleURL);
      if (!path.empty())
        return path;
    }
  }

  return "";
}

inline bool ParseLeadingMajorVersion(const std::string &versionText,
                                     A_long *majorP) {
  if (!majorP || versionText.empty())
    return false;

  A_long major = 0;
  size_t i = 0;
  while (i < versionText.size() && versionText[i] >= '0' &&
         versionText[i] <= '9') {
    major = major * 10 + (versionText[i] - '0');
    ++i;
  }
  if (i == 0)
    return false;

  *majorP = major;
  return true;
}

inline bool CopyHostAppVersionString(char *buffer, size_t bufferSize) {
  if (!buffer || bufferSize == 0)
    return false;

  CFBundleRef mainBundle = CFBundleGetMainBundle();
  if (!mainBundle)
    return false;

  static const CFStringRef versionKeys[] = {
      CFSTR("CFBundleShortVersionString"),
      CFSTR("Adobe Product Version"),
  };

  for (const CFStringRef key : versionKeys) {
    CFTypeRef versionValue =
        CFBundleGetValueForInfoDictionaryKey(mainBundle, key);
    if (!versionValue || CFGetTypeID(versionValue) != CFStringGetTypeID())
      continue;

    buffer[0] = '\0';
    if (CFStringGetCString(static_cast<CFStringRef>(versionValue), buffer,
                           static_cast<CFIndex>(bufferSize),
                           kCFStringEncodingUTF8) &&
        buffer[0]) {
      return true;
    }
  }

  return false;
}

inline bool GetMacHostAppMajorVersion(A_long *majorP) {
  if (!majorP)
    return false;

  char versionBuffer[64] = {0};
  if (!CopyHostAppVersionString(versionBuffer, sizeof(versionBuffer)))
    return false;

  return ParseLeadingMajorVersion(versionBuffer, majorP);
}

inline bool ContainsAsciiCaseInsensitive(const std::string &value,
                                         const char *needle) {
  if (!needle || !*needle)
    return true;

  std::string loweredNeedle;
  for (const char *p = needle; *p; ++p) {
    char ch = *p;
    if (ch >= 'A' && ch <= 'Z')
      ch = static_cast<char>(ch - 'A' + 'a');
    loweredNeedle.push_back(ch);
  }

  for (size_t i = 0; i + loweredNeedle.size() <= value.size(); ++i) {
    bool matches = true;
    for (size_t j = 0; j < loweredNeedle.size(); ++j) {
      char ch = value[i + j];
      if (ch >= 'A' && ch <= 'Z')
        ch = static_cast<char>(ch - 'A' + 'a');
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

inline bool HostExecutablePathLooksLikeBeta(const std::string &path) {
  return ContainsAsciiCaseInsensitive(path, "beta");
}

inline bool DebugLoggingEnabled() {
  const char *value = std::getenv("PALF_AELOCALISE_DEBUG");
  return value && value[0] && value[0] != '0';
}

inline void AppendDebugLogLine(const std::string &line) {
  if (line.empty())
    return;

  FILE *file = std::fopen("/tmp/Palf_AELocalise.log", "ab");
  if (!file)
    return;

  const std::string fileLine = line + "\n";
  std::fwrite(fileLine.data(), 1, fileLine.size(), file);
  std::fclose(file);
}

inline void LogHostTextEncodingDecisionOnce(const std::string &executablePath,
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
  line += executablePath;
  line += "\"";

  AppendDebugLogLine(line);
}

inline bool HostUsesUTF8ForAEText(PF_InData *in_dataP) {
  if (!in_dataP || in_dataP->appl_id != kAppID_AfterEffects) {
    LogHostTextEncodingDecisionOnce("", false, false, false, 0, false,
                                   "not-after-effects");
    return false;
  }
  static std::atomic<long> cachedDriverMajor{-1};
  long major = cachedDriverMajor.load(std::memory_order_acquire);
  if (major < 0) {
    const auto executablePath = GetHostExecutablePath(in_dataP);
    const bool looksBeta = HostExecutablePathLooksLikeBeta(executablePath);
    A_long detectedMajor = 0;
    const char *reason = "driver-version";
    if (!GetAEDriverMajorVersion(in_dataP, &detectedMajor)) {
      A_long fallbackMajor = 0;
      if (!GetMacHostAppMajorVersion(&fallbackMajor)) {
        LogHostTextEncodingDecisionOnce(executablePath, true, looksBeta, false, 0,
                                       false, "version-detection-failed");
        return false;
      }
      detectedMajor = fallbackMajor + 100;
      reason = "product-version-fallback";
    }
    long expected = -1;
    cachedDriverMajor.compare_exchange_strong(expected, detectedMajor,
                                             std::memory_order_release,
                                             std::memory_order_relaxed);
    major = cachedDriverMajor.load(std::memory_order_acquire);
    const bool utf8 = UseUTF8ForAEDriver(
        major, utf8FlagDeclared.load(std::memory_order_acquire));
    LogHostTextEncodingDecisionOnce(executablePath, true, looksBeta, true,
                                   major, utf8, reason);
  }
  // AE 26 predates the flag but requires UTF-8. On later hosts opt in only
  // after the effect has declared the flag in PiPL and GLOBAL_SETUP.
  return UseUTF8ForAEDriver(
      major, utf8FlagDeclared.load(std::memory_order_acquire));
}
} // namespace Internal

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
// 各言語の名前空間（EN_US, JA_JP, ZH_CN, KO_KR, DE_DE, ES_ES, FR_FR, IT_IT,
// PT_BR, RU_RU等）が定義されていることを前提とする
// 定義されていない言語は英語にフォールバック
// 注意:
// 新しい言語の名前空間が定義されていない場合は、そのロケールIDのチェックをコメントアウトするか、
// または対応する名前空間を定義する必要があります
template <typename LocKeyType>
inline const char *GetStringByLanguage(LocKeyType key,
                                       const std::string &localeId) {
  // ロケールIDから直接名前空間を選択（完全一致）
  // 定義されていない言語は英語にフォールバック
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
  // デフォルトは英語を返す
  return EN_US::GetString(key);
}

// ローカライズ文字列を取得し、AEが受け付ける文字コードで返す。
// Driver major <=125: locale encoding; 126: UTF-8 (pre-flag host);
// >=127: UTF-8 only when the effect declares the flag in PiPL and GLOBAL_SETUP.
//
// Param / popup labels fed to PF_ADD_* must also respect
// PF_MAX_EFFECT_PARAM_NAME_LEN (31 bytes). Under AE 26+ UTF-8 that is about
// 10 Japanese characters; longer UTF-8 names truncate mid-code-unit and
// mojibake. Keep JA labels ≤31 UTF-8 bytes (Drawbot captions via
// GetStringUTF8 are exempt).
// Encode a UTF-8 C string for PF_ADD_* / param-name fields (same rules as
// GetStringForAE). Shared helpers such as PLGradientLocalise use this so they
// do not need a plugin LocKey enum.
inline const char *EncodeUTF8ForAE(const char *utf8Str, PF_InData *in_dataP,
                                   const std::string &localeIdHint = "") {
  if (!utf8Str)
    return "";
  std::string localeId =
      localeIdHint.empty() ? GetCurrentLanguage(in_dataP) : localeIdHint;

#ifdef AE_OS_WIN
  if (Internal::HostUsesUTF8ForAEText(in_dataP)) {
    Internal::s_tempBuffer = utf8Str;
    return Internal::s_tempBuffer.c_str();
  }
  UINT codePage = GetCodePageForLanguage(localeId);
  Internal::s_tempBuffer = ConvertUTF8ToEncoding(utf8Str, codePage);
#else
  if (Internal::HostUsesUTF8ForAEText(in_dataP)) {
    Internal::s_tempBuffer = utf8Str;
  } else {
    CFStringEncoding encoding = GetEncodingForLanguage(localeId);
    Internal::s_tempBuffer = ConvertUTF8ToEncoding(utf8Str, encoding);
  }
#endif
  return Internal::s_tempBuffer.c_str();
}

template <typename LocKeyType>
inline const char *GetStringForAE(LocKeyType key, PF_InData *in_dataP) {
  std::string localeId = GetCurrentLanguage(in_dataP);
  const char *utf8Str = GetStringByLanguage(key, localeId);

  if (!utf8Str || strlen(utf8Str) == 0) {
    // フォールバック: 英語を返す
    utf8Str = EN_US::GetString(key);
    localeId = "en_US"; // フォールバック時は英語のエンコーディングを使用
  }

  return EncodeUTF8ForAE(utf8Str, in_dataP, localeId);
}

// Returns the raw UTF-8 localised string for the host language, WITHOUT the
// platform legacy-encoding conversion that GetStringForAE applies. Use this for
// plugin-side rendering paths that expect UTF-8 and do their own UTF-16
// conversion — e.g. Drawbot custom UI. Feeding a legacy-encoded (Shift-JIS)
// string into a UTF-8 consumer corrupts it; that is why this separate accessor
// exists alongside GetStringForAE (which AE needs for param-name fields).
template <typename LocKeyType>
inline const char *GetStringUTF8(LocKeyType key, PF_InData *in_dataP) {
  std::string localeId = GetCurrentLanguage(in_dataP);
  const char *utf8Str = GetStringByLanguage(key, localeId);
  if (!utf8Str || strlen(utf8Str) == 0) {
    utf8Str = EN_US::GetString(key);
  }
  if (!utf8Str)
    return "";
  Internal::s_tempBuffer = utf8Str;
  return Internal::s_tempBuffer.c_str();
}
} // namespace AELocalise

#endif // AELOCALISE_H
