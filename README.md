# Bitonic Pixel Sorter for After Effects

[English](#english) · [日本語](#日本語) · [中文](#中文) · [한국어](#한국어)

---

## English

This repository is an Adobe After Effects plug-in fork of
[ruccho/BitonicPixelSorter](https://github.com/ruccho/BitonicPixelSorter), an
MIT-licensed Unity / URP GPU pixel sorter. The original project remains the
algorithmic and attribution source. This fork reorganises the codebase as a
self-contained After Effects SDK effect plug-in and removes the Unity project
surface from the active repository layout.

### Overview

Bitonic Pixel Sorter sorts contiguous spans of pixels whose brightness falls
inside a threshold range. Sorting can run horizontally by row or vertically by
column, and each eligible span can be ordered ascending or descending by luma.

The current AE port contains:

- CPU SmartFX rendering for 8-bit, 16-bit and 32-bit float AE worlds.
- CUDA, OpenCL and DirectX GPU build paths for AE BGRA128 GPU worlds.
- English, Japanese, Simplified Chinese and Korean parameter strings.
- A self-contained CMake build that references only the Adobe After Effects SDK
  Examples tree plus vendored, header-only local helpers.

Metal is planned but is not validated in this repository yet. macOS plug-in
build and host validation should be performed later on real Mac hardware.

### Parameters

| Parameter | Meaning |
| --- | --- |
| Direction | Horizontal row sort or vertical column sort. |
| Order | Ascending or descending brightness order. |
| Threshold Min | Lower luma bound for sortable pixels. |
| Threshold Max | Upper luma bound for sortable pixels. |

The luma key follows the upstream weights:

```text
0.298912 * R + 0.586611 * G + 0.114478 * B
```

### Repository Layout

```text
BitonicPixelSorterForAE/
  CMakeLists.txt              AE plug-in build definition
  Directory.Build.props       Windows SDK/MSBuild default output property
  Source/                     PF effect entry point, CPU path and GPU dispatch
  GPU/                        CUDA, OpenCL and HLSL bitonic kernels
  Localise/                   EN/JA/ZH/KO string tables
  PiPL/                       AE PiPL resource source
  Mac/                        macOS bundle metadata template
  cmake/                      build helper scripts
  vendor/palf/                copied header-only localisation shim
  docs/                       AE port plan and implementation notes
  dist/Win/Release/           tracked Windows release .aex package
  dist/Mac/Release/           tracked macOS release .plugin package
```

Debug and intermediate build outputs are ignored. Release plug-in artefacts
under `dist/Win/Release/` and `dist/Mac/Release/` are intentionally versionable
so tested deliverables can be attached to the repository history.

### Prerequisites

- Adobe After Effects SDK, with this repository placed inside the SDK
  `Examples` directory or configured with `-DAESDK_ROOT=<path-to-Examples>`.
- CMake 3.18 or newer.
- Windows: Visual Studio 2022 with MSVC v143.
- Optional Windows GPU backends:
  - CUDA Toolkit for CUDA.
  - OpenCL headers and library. The CUDA Toolkit commonly supplies both.
  - Windows SDK `dxc.exe`, D3D12 and `d3dcompiler` for DirectX.
  - Python 3 for embedding OpenCL and DirectX shader blobs during configure/build.
- macOS: Xcode or Command Line Tools, including `clang` and `Rez`.

### Build

Windows Release:

```powershell
cmake -S . -B build/Win -G "Visual Studio 17 2022" -A x64
cmake --build build/Win --config Release
```

Windows Debug:

```powershell
cmake --build build/Win --config Debug
```

macOS Release, to be validated on Mac hardware:

```sh
cmake -S . -B build/Mac -G Xcode
cmake --build build/Mac --config Release
```

Expected local outputs:

| Platform | Configuration | Output |
| --- | --- | --- |
| Windows | Release | `dist/Win/Release/BitonicPixelSorter.aex` |
| Windows | Debug | `dist/Win/Debug/BitonicPixelSorter_debug.aex` |
| macOS | Release | `dist/Mac/Release/BitonicPixelSorter.plugin` |
| macOS | Debug | `dist/Mac/Debug/BitonicPixelSorter.plugin` |

DirectX shader bytecode and root signatures are embedded into the `.aex` at build
time. No separate `DirectX_Assets` folder is required beside the plug-in module.

### Installing Into After Effects

Copy the generated `.aex` or `.plugin` package into an After Effects plug-ins
folder. On Windows this is commonly:

```text
C:\Program Files\Adobe\Adobe After Effects <version>\Support Files\Plug-ins\
```

CMake can also copy the built plug-in after each build:

```powershell
cmake -S . -B build/Win -DBPS_AE_PLUGIN_DIR="C:/Program Files/Adobe/Adobe After Effects 2025/Support Files/Plug-ins"
```

The effect is registered in the `Stylize` category as `Bitonic Pixel Sorter`.

### Status

Validated on the current Windows workstation:

- CMake configure with Visual Studio 17 2022 x64.
- Release and Debug builds with CUDA 13.1, OpenCL and DirectX enabled.
- Windows release `.aex` exports `EffectMain` and `PluginDataEntryFunction2`.
- Direct OpenCL runtime compile check succeeds; NVIDIA's compiler emits only a
  non-fatal noinline warning.

Still pending:

- After Effects in-host CPU/GPU visual parity validation.
- macOS `.plugin` build and host validation on real Mac hardware.
- Metal backend implementation.
- GPU hardening for partial Smart Render output rectangles and large-frame
  tiling edge cases.

### Licence And Attribution

Upstream project:

- `ruccho/BitonicPixelSorter`
- <https://github.com/ruccho/BitonicPixelSorter>
- MIT licence, preserved in `LICENSE`.

This fork keeps the upstream attribution explicit because the core effect,
brightness model and bitonic sorting strategy derive from that project. The AE
host integration, CMake build, PiPL resources, localisation files and CPU/GPU
backend adaptation are specific to this After Effects fork.

---

## 日本語

本リポジトリは、MIT ライセンスの Unity / URP GPU ピクセルソーター
[ruccho/BitonicPixelSorter](https://github.com/ruccho/BitonicPixelSorter) の
Adobe After Effects プラグイン版フォークです。アルゴリズムと帰属表示の原典は
上流プロジェクトのままです。本フォークはコードベースを自己完結型の After
Effects SDK エフェクトプラグインとして再構成し、アクティブなリポジトリ構成から
Unity プロジェクトの表層を取り除いています。

### 概要

Bitonic Pixel Sorter は、輝度がしきい値範囲内に収まる連続ピクセル区間を
ソートします。行方向の水平ソートと列方向の垂直ソートに対応し、対象区間は
輝度の昇順または降順で並べ替えられます。

現在の AE 移植版には次が含まれます。

- 8 bit / 16 bit / 32 bit float の AE ワールド向け CPU SmartFX レンダリング。
- AE BGRA128 GPU ワールド向けの CUDA / OpenCL / DirectX GPU ビルドパス。
- 英語・日本語・簡体中国語・韓国語のパラメータ文字列。
- Adobe After Effects SDK の Examples ツリーと、同梱のヘッダオンリー
  ローカルヘルパーのみを参照する自己完結型 CMake ビルド。

Metal は予定されていますが、本リポジトリではまだ検証されていません。macOS
プラグインのビルドとホスト検証は、実機 Mac 上で後日実施してください。

### パラメータ

| パラメータ | 説明 |
| --- | --- |
| 方向 | 行方向の水平ソート、または列方向の垂直ソート。 |
| 並び順 | 輝度の昇順または降順。 |
| しきい値 Min | ソート対象ピクセルの輝度下限。 |
| しきい値 Max | ソート対象ピクセルの輝度上限。 |

輝度キーは上流と同じ重み付けを用います。

```text
0.298912 * R + 0.586611 * G + 0.114478 * B
```

### リポジトリ構成

```text
BitonicPixelSorterForAE/
  CMakeLists.txt              AE プラグインのビルド定義
  Directory.Build.props       Windows SDK / MSBuild の既定出力プロパティ
  Source/                     PF エフェクトのエントリポイント、CPU パス、GPU ディスパッチ
  GPU/                        CUDA / OpenCL / HLSL のビトニックカーネル
  Localise/                   EN / JA / ZH / KO の文字列テーブル
  PiPL/                       AE PiPL リソースソース
  Mac/                        macOS バンドルメタデータテンプレート
  cmake/                      ビルド補助スクリプト
  vendor/palf/                コピー済みヘッダオンリーのローカライズシム
  docs/                       AE 移植計画と実装メモ
  dist/Win/Release/           追跡対象の Windows リリース .aex パッケージ
  dist/Mac/Release/           追跡対象の macOS リリース .plugin パッケージ
```

デバッグ用および中間ビルド出力は無視されます。`dist/Win/Release/` と
`dist/Mac/Release/` 配下のリリースプラグイン成果物は、検証済みの成果物を
リポジトリ履歴に残せるよう、意図的にバージョン管理対象としています。

### 前提条件

- Adobe After Effects SDK。本リポジトリを SDK の `Examples` ディレクトリ内に
  置くか、`-DAESDK_ROOT=<path-to-Examples>` でパスを指定します。
- CMake 3.18 以降。
- Windows: Visual Studio 2022（MSVC v143）。
- Windows の任意 GPU バックエンド:
  - CUDA: CUDA Toolkit。
  - OpenCL: ヘッダとライブラリ。CUDA Toolkit に同梱されていることが多いです。
  - DirectX: Windows SDK の `dxc.exe`、D3D12、`d3dcompiler`。
  - configure / build 時に OpenCL および DirectX シェーダ blob を埋め込む Python 3。
- macOS: Xcode または Command Line Tools（`clang` と `Rez` を含む）。

### ビルド

Windows Release:

```powershell
cmake -S . -B build/Win -G "Visual Studio 17 2022" -A x64
cmake --build build/Win --config Release
```

Windows Debug:

```powershell
cmake --build build/Win --config Debug
```

macOS Release（実機 Mac での検証予定）:

```sh
cmake -S . -B build/Mac -G Xcode
cmake --build build/Mac --config Release
```

想定されるローカル出力:

| プラットフォーム | 構成 | 出力 |
| --- | --- | --- |
| Windows | Release | `dist/Win/Release/BitonicPixelSorter.aex` |
| Windows | Debug | `dist/Win/Debug/BitonicPixelSorter_debug.aex` |
| macOS | Release | `dist/Mac/Release/BitonicPixelSorter.plugin` |
| macOS | Debug | `dist/Mac/Debug/BitonicPixelSorter.plugin` |

DirectX シェーダ bytecode と root signature はビルド時に `.aex` へ埋め込まれます。
プラグインモジュール横に `DirectX_Assets` フォルダを置く必要はありません。

### After Effects へのインストール

生成された `.aex` または `.plugin` を After Effects のプラグインフォルダへ
コピーします。Windows では通常次の場所です。

```text
C:\Program Files\Adobe\Adobe After Effects <version>\Support Files\Plug-ins\
```

CMake でビルド後に自動コピーすることもできます。

```powershell
cmake -S . -B build/Win -DBPS_AE_PLUGIN_DIR="C:/Program Files/Adobe/Adobe After Effects 2025/Support Files/Plug-ins"
```

エフェクトは `Stylize` カテゴリに `Bitonic Pixel Sorter` として登録されます。

### ステータス

現在の Windows ワークステーションで検証済み:

- Visual Studio 17 2022 x64 による CMake configure。
- CUDA 13.1、OpenCL、DirectX を有効にした Release / Debug ビルド。
- Windows リリース `.aex` が `EffectMain` と `PluginDataEntryFunction2` を
  エクスポートすること。
- OpenCL のランタイムコンパイル直接チェックの成功。NVIDIA コンパイラは
  非致命的な noinline 警告のみを出力。

未対応:

- After Effects ホスト内での CPU / GPU 画質一致の検証。
- 実機 Mac での macOS `.plugin` ビルドとホスト検証。
- Metal バックエンドの実装。
- 部分 Smart Render 出力矩形や大画面タイル境界での GPU 堅牢化。

### ライセンスと帰属

上流プロジェクト:

- `ruccho/BitonicPixelSorter`
- <https://github.com/ruccho/BitonicPixelSorter>
- MIT ライセンス（`LICENSE` に保持）

本フォークは、コアエフェクト・輝度モデル・ビトニックソート戦略が上流由来である
ことを明示的に保持しています。AE ホスト統合、CMake ビルド、PiPL リソース、
ローカライズファイル、CPU / GPU バックエンド適応は本 After Effects フォーク
固有の作業です。

---

## 中文

本仓库是 MIT 许可的 Unity / URP GPU 像素排序器
[ruccho/BitonicPixelSorter](https://github.com/ruccho/BitonicPixelSorter) 的
Adobe After Effects 插件分支。算法与归属仍以原项目为准。本分支将代码库重组为
自包含的 After Effects SDK 效果插件，并从当前仓库布局中移除了 Unity 项目表层。

### 概述

Bitonic Pixel Sorter 对亮度落在阈值范围内的连续像素区间进行排序。可按行水平
排序或按列垂直排序，每个符合条件的区间可按亮度升序或降序排列。

当前 AE 移植版包含：

- 面向 8 位、16 位和 32 位浮点 AE 世界的 CPU SmartFX 渲染。
- 面向 AE BGRA128 GPU 世界的 CUDA、OpenCL 和 DirectX GPU 构建路径。
- 英语、日语、简体中文和韩语参数字符串。
- 仅引用 Adobe After Effects SDK Examples 树及随附仅头文件本地辅助代码的
  自包含 CMake 构建。

Metal 已列入计划，但本仓库尚未验证。macOS 插件构建与宿主验证应在真实 Mac
硬件上后续进行。

### 参数

| 参数 | 含义 |
| --- | --- |
| 方向 | 按行水平排序或按列垂直排序。 |
| 顺序 | 按亮度升序或降序排列。 |
| 阈值最小值 | 可排序像素的亮度下限。 |
| 阈值最大值 | 可排序像素的亮度上限。 |

亮度键沿用上游权重：

```text
0.298912 * R + 0.586611 * G + 0.114478 * B
```

### 仓库结构

```text
BitonicPixelSorterForAE/
  CMakeLists.txt              AE 插件构建定义
  Directory.Build.props       Windows SDK / MSBuild 默认输出属性
  Source/                     PF 效果入口、CPU 路径与 GPU 调度
  GPU/                        CUDA、OpenCL 与 HLSL 双调内核
  Localise/                   英 / 日 / 中 / 韩字符串表
  PiPL/                       AE PiPL 资源源码
  Mac/                        macOS 包元数据模板
  cmake/                      构建辅助脚本
  vendor/palf/                复制的仅头文件本地化垫片
  docs/                       AE 移植计划与实现说明
  dist/Win/Release/           纳入版本管理的 Windows 发布 .aex 包
  dist/Mac/Release/           纳入版本管理的 macOS 发布 .plugin 包
```

调试与中间构建输出被忽略。`dist/Win/Release/` 与 `dist/Mac/Release/` 下的
发布插件产物有意纳入版本管理，以便将已测试的交付物保留在仓库历史中。

### 前置条件

- Adobe After Effects SDK。将本仓库置于 SDK 的 `Examples` 目录内，或通过
  `-DAESDK_ROOT=<path-to-Examples>` 配置路径。
- CMake 3.18 或更高版本。
- Windows：Visual Studio 2022（MSVC v143）。
- 可选 Windows GPU 后端：
  - CUDA：CUDA Toolkit。
  - OpenCL：头文件与库。CUDA Toolkit 通常同时提供二者。
  - DirectX：Windows SDK 的 `dxc.exe`、D3D12 与 `d3dcompiler`。
  - 在 configure / build 阶段嵌入 OpenCL 与 DirectX 着色器 blob 所需的 Python 3。
- macOS：Xcode 或 Command Line Tools（含 `clang` 与 `Rez`）。

### 构建

Windows Release：

```powershell
cmake -S . -B build/Win -G "Visual Studio 17 2022" -A x64
cmake --build build/Win --config Release
```

Windows Debug：

```powershell
cmake --build build/Win --config Debug
```

macOS Release（待在真实 Mac 硬件上验证）：

```sh
cmake -S . -B build/Mac -G Xcode
cmake --build build/Mac --config Release
```

预期本地输出：

| 平台 | 配置 | 输出 |
| --- | --- | --- |
| Windows | Release | `dist/Win/Release/BitonicPixelSorter.aex` |
| Windows | Debug | `dist/Win/Debug/BitonicPixelSorter_debug.aex` |
| macOS | Release | `dist/Mac/Release/BitonicPixelSorter.plugin` |
| macOS | Debug | `dist/Mac/Debug/BitonicPixelSorter.plugin` |

DirectX 着色器 bytecode 与 root signature 在构建时嵌入 `.aex`。无需在插件模块旁放置单独的 `DirectX_Assets` 文件夹。

### 安装到 After Effects

将生成的 `.aex` 或 `.plugin` 复制到 After Effects 插件文件夹。Windows 上
通常为：

```text
C:\Program Files\Adobe\Adobe After Effects <version>\Support Files\Plug-ins\
```

CMake 也可在每次构建后自动复制已构建的插件：

```powershell
cmake -S . -B build/Win -DBPS_AE_PLUGIN_DIR="C:/Program Files/Adobe/Adobe After Effects 2025/Support Files/Plug-ins"
```

效果在 `Stylize` 类别下注册为 `Bitonic Pixel Sorter`。

### 状态

已在当前 Windows 工作站上验证：

- 使用 Visual Studio 17 2022 x64 的 CMake configure。
- 启用 CUDA 13.1、OpenCL 与 DirectX 的 Release / Debug 构建。
- Windows 发布 `.aex` 导出 `EffectMain` 与 `PluginDataEntryFunction2`。
- 直接 OpenCL 运行时编译检查通过；NVIDIA 编译器仅发出非致命的 noinline 警告。

尚待完成：

- After Effects 宿主内 CPU / GPU 画面一致性验证。
- 在真实 Mac 硬件上的 macOS `.plugin` 构建与宿主验证。
- Metal 后端实现。
- 针对部分 Smart Render 输出矩形与大帧分块边界情况的 GPU 加固。

### 许可与归属

上游项目：

- `ruccho/BitonicPixelSorter`
- <https://github.com/ruccho/BitonicPixelSorter>
- MIT 许可，保留于 `LICENSE`。

本分支明确保留上游归属，因为核心效果、亮度模型与双调排序策略均源自该项目。
AE 宿主集成、CMake 构建、PiPL 资源、本地化文件以及 CPU / GPU 后端适配为本
After Effects 分支特有工作。

---

## 한국어

이 저장소는 MIT 라이선스의 Unity / URP GPU 픽셀 정렬기
[ruccho/BitonicPixelSorter](https://github.com/ruccho/BitonicPixelSorter)의
Adobe After Effects 플러그인 포크입니다. 알고리즘과 귀속의 원천은 상류
프로젝트에 그대로 둡니다. 이 포크는 코드베이스를 자체 완결형 After Effects SDK
이펙트 플러그인으로 재구성했으며, 활성 저장소 레이아웃에서 Unity 프로젝트
표면을 제거했습니다.

### 개요

Bitonic Pixel Sorter는 밝기가 임계값 범위 안에 들어가는 연속 픽셀 구간을
정렬합니다. 행 단위 수평 정렬과 열 단위 수직 정렬을 지원하며, 각 대상 구간은
휘도 기준 오름차순 또는 내림차순으로 정렬할 수 있습니다.

현재 AE 포트에는 다음이 포함됩니다.

- 8비트, 16비트, 32비트 float AE 월드용 CPU SmartFX 렌더링.
- AE BGRA128 GPU 월드용 CUDA, OpenCL, DirectX GPU 빌드 경로.
- 영어, 일본어, 간체 중국어, 한국어 매개변수 문자열.
- Adobe After Effects SDK Examples 트리와 동봉된 헤더 전용 로컬 헬퍼만
  참조하는 자체 완결형 CMake 빌드.

Metal은 계획되어 있으나 이 저장소에서는 아직 검증되지 않았습니다. macOS
플러그인 빌드와 호스트 검증은 실제 Mac 하드웨어에서 이후 수행해야 합니다.

### 매개변수

| 매개변수 | 의미 |
| --- | --- |
| 방향 | 행 단위 수평 정렬 또는 열 단위 수직 정렬. |
| 정렬 순서 | 밝기 오름차순 또는 내림차순. |
| 임계값 최소 | 정렬 대상 픽셀의 휘도 하한. |
| 임계값 최대 | 정렬 대상 픽셀의 휘도 상한. |

휘도 키는 상류와 동일한 가중치를 사용합니다.

```text
0.298912 * R + 0.586611 * G + 0.114478 * B
```

### 저장소 구조

```text
BitonicPixelSorterForAE/
  CMakeLists.txt              AE 플러그인 빌드 정의
  Directory.Build.props       Windows SDK / MSBuild 기본 출력 속성
  Source/                     PF 이펙트 진입점, CPU 경로, GPU 디스패치
  GPU/                        CUDA, OpenCL, HLSL 비토닉 커널
  Localise/                   EN / JA / ZH / KO 문자열 테이블
  PiPL/                       AE PiPL 리소스 소스
  Mac/                        macOS 번들 메타데이터 템플릿
  cmake/                      빌드 보조 스크립트
  vendor/palf/                복사된 헤더 전용 현지화 심
  docs/                       AE 포트 계획 및 구현 메모
  dist/Win/Release/           추적 대상 Windows 릴리스 .aex 패키지
  dist/Mac/Release/           추적 대상 macOS 릴리스 .plugin 패키지
```

디버그 및 중간 빌드 출력은 무시됩니다. `dist/Win/Release/`와
`dist/Mac/Release/` 아래의 릴리스 플러그인 산출물은 검증된 결과물을 저장소
기록에 남길 수 있도록 의도적으로 버전 관리 대상입니다.

### 사전 요구 사항

- Adobe After Effects SDK. 이 저장소를 SDK `Examples` 디렉터리 안에 두거나
  `-DAESDK_ROOT=<path-to-Examples>`로 경로를 지정합니다.
- CMake 3.18 이상.
- Windows: Visual Studio 2022(MSVC v143).
- 선택적 Windows GPU 백엔드:
  - CUDA: CUDA Toolkit.
  - OpenCL: 헤더와 라이브러리. CUDA Toolkit에 둘 다 포함되는 경우가 많습니다.
  - DirectX: Windows SDK `dxc.exe`, D3D12, `d3dcompiler`.
  - configure / build 시 OpenCL 및 DirectX 셰이더 blob을 임베드하는 Python 3.
- macOS: Xcode 또는 Command Line Tools(`clang`, `Rez` 포함).

### 빌드

Windows Release:

```powershell
cmake -S . -B build/Win -G "Visual Studio 17 2022" -A x64
cmake --build build/Win --config Release
```

Windows Debug:

```powershell
cmake --build build/Win --config Debug
```

macOS Release(실제 Mac 하드웨어에서 검증 예정):

```sh
cmake -S . -B build/Mac -G Xcode
cmake --build build/Mac --config Release
```

예상 로컬 출력:

| 플랫폼 | 구성 | 출력 |
| --- | --- | --- |
| Windows | Release | `dist/Win/Release/BitonicPixelSorter.aex` |
| Windows | Debug | `dist/Win/Debug/BitonicPixelSorter_debug.aex` |
| macOS | Release | `dist/Mac/Release/BitonicPixelSorter.plugin` |
| macOS | Debug | `dist/Mac/Debug/BitonicPixelSorter.plugin` |

DirectX 셰이더 bytecode와 root signature는 빌드 시 `.aex`에 임베드됩니다. 플러그인 모듈 옆에 `DirectX_Assets` 폴더를 둘 필요가 없습니다.

### After Effects에 설치

생성된 `.aex` 또는 `.plugin`을 After Effects 플러그인 폴더에 복사합니다.
Windows에서는 보통 다음 위치입니다.

```text
C:\Program Files\Adobe\Adobe After Effects <version>\Support Files\Plug-ins\
```

CMake로 빌드 후 자동 복사도 가능합니다.

```powershell
cmake -S . -B build/Win -DBPS_AE_PLUGIN_DIR="C:/Program Files/Adobe/Adobe After Effects 2025/Support Files/Plug-ins"
```

이펙트는 `Stylize` 카테고리에 `Bitonic Pixel Sorter`로 등록됩니다.

### 상태

현재 Windows 워크스테이션에서 검증됨:

- Visual Studio 17 2022 x64로 CMake configure.
- CUDA 13.1, OpenCL, DirectX를 사용한 Release / Debug 빌드.
- Windows 릴리스 `.aex`가 `EffectMain`과 `PluginDataEntryFunction2`를
  export합니다.
- OpenCL 런타임 컴파일 직접 검사 성공. NVIDIA 컴파일러는 치명적이지 않은
  noinline 경고만 출력.

아직 남은 항목:

- After Effects 호스트 내 CPU / GPU 화면 일치 검증.
- 실제 Mac 하드웨어에서의 macOS `.plugin` 빌드 및 호스트 검증.
- Metal 백엔드 구현.
- 부분 Smart Render 출력 사각형 및 대형 프레임 타일 경계에 대한 GPU 강화.

### 라이선스 및 귀속

상류 프로젝트:

- `ruccho/BitonicPixelSorter`
- <https://github.com/ruccho/BitonicPixelSorter>
- MIT 라이선스(`LICENSE`에 보존)

이 포크는 핵심 이펙트, 밝기 모델, 비토닉 정렬 전략이 상류에서 유래함을
명시적으로 유지합니다. AE 호스트 통합, CMake 빌드, PiPL 리소스, 현지화 파일,
CPU / GPU 백엔드 적응은 이 After Effects 포크 고유의 작업입니다.
