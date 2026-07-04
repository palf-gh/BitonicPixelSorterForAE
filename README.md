# Bitonic Pixel Sorter for After Effects

[English](#english) · [日本語](#日本語) · [中文](#中文) · [한국어](#한국어)

**Current plug-in version: 1.1 (release)**

---

## English

This repository is an Adobe After Effects plug-in fork of
[ruccho/BitonicPixelSorter](https://github.com/ruccho/BitonicPixelSorter), an
MIT-licensed Unity / URP GPU pixel sorter. The original project remains the
algorithmic and attribution source. This fork reorganises the codebase as a
self-contained After Effects SDK effect plug-in and removes the Unity project
surface from the active repository layout.

### Overview

Bitonic Pixel Sorter sorts contiguous spans of pixels whose selected trigger
key falls inside or outside a threshold range. Sorting can run on
horizontal/vertical axes, a free angle, circular rotation paths around a centre
point, radial paths from that centre, Archimedean swirl paths, or a user-chosen
layer mask path (with open-end straight extensions). Each eligible span can be ordered
ascending or descending by a separate sort criterion, and the sorted span can be
cycled by an angle value.

The current AE port contains:

- CPU SmartFX rendering for 8-bit, 16-bit and 32-bit float AE worlds.
- CUDA, OpenCL, DirectX and Metal GPU build paths for AE BGRA128 GPU worlds.
- English, Japanese, Simplified Chinese and Korean parameter strings.
- A self-contained CMake build that references only the Adobe After Effects SDK
  Examples tree plus vendored, header-only local helpers.
- Swirl and Path use the shared CPU/GPU pixel-owned transformed path map so
  each in-frame output pixel is owned once and rounded-domain holes or
  collisions cannot destruct the image.

Metal compiles and links into the universal `.plugin`, with the compute kernel
embedded as runtime-compiled Metal source. It has been validated on Apple
Silicon Macs in After Effects 2023 through 2026.

### Parameters

UI order matches the Effect Controls panel:

| Parameter | Meaning |
| --- | --- |
| GPU Acceleration | Read-only status of the Mercury GPU path. |
| Mode | Axis, Free Angle, Rotation, Radial, Swirl, or Path sorting. |
| Order | Ascending or descending key order. |
| Threshold Min | Lower trigger-key bound (percent). |
| Threshold Max | Upper trigger-key bound (percent). |
| Sort Trigger | Threshold key: Luminance, RGB Average/Product/Minimum/Maximum, Red, Green, Blue, Alpha, Hue, or Saturation. |
| Trigger Source Layer | Layer used to evaluate the trigger key (`None` = effect source). |
| Sort Criterion | Sort key, with the same choices as Sort Trigger. |
| Criterion Source Layer | Layer used to evaluate the sort key (`None` = effect source). |
| Affect | Sort pixels inside or outside the threshold range. |
| Cycle | Cyclic shift applied to each sorted run. |
| Direction | Horizontal row sort or vertical column sort in Axis mode. |
| Angle | Free Angle direction. |
| Centre | Rotation/Radial/Swirl centre point. |
| Swirl Amount | Twist amount for Swirl mode (360° = one turn to the farthest frame corner; sign selects direction). |
| Path | Layer mask path used in Path mode. |
| Path Direction | Sort along the path normal or tangent. |

Defaults: Mode = Axis, Order = Ascending, Threshold Min/Max = 40% / 60%,
Path Direction = Tangent, Trigger/Criterion source layers = `None`.

The default luminance key follows the upstream weights:

```text
0.298912 * R + 0.586611 * G + 0.114478 * B
```

Swirl is classified as `phase = wrap(theta - signedAmount * radius)`.
`Swirl Amount` is an angle control where 360° maps to one full turn at the
farthest frame corner; the sign selects direction. `0` therefore uses the same
line classification as Radial mode, and larger values form Archimedean spiral
paths while sorting outward by radius.

`Criterion Source Layer` and `Trigger Source Layer` default to `None`, which uses
the effect source. When set to another layer, sort keys and threshold runs are
evaluated from that layer at the same layer coordinates while pixels are still
moved from the effect source.

Path mode assigns every output pixel to the nearest segment of the sampled
polyline. Open path endpoints inside the frame are extended along their tangent
to the frame edge; closed paths are not extended. Self-intersections are handled
deterministically by nearest-segment ownership, with ties resolved by lower
arc length, then segment order, then pixel order.

### Repository Layout

```text
BitonicPixelSorterForAE/
  CMakeLists.txt              AE plug-in build definition
  Directory.Build.props       Windows SDK/MSBuild default output property
  Source/                     PF effect entry point, CPU path, GPU dispatch, Metal backend
  GPU/                        CUDA, OpenCL, HLSL and Metal bitonic kernels
  Localise/                   EN/JA/ZH/KO string tables and AELocalise helper
  PiPL/                       AE PiPL resource source
  Mac/                        macOS bundle metadata template
  cmake/                      build helper scripts
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
- Optional Windows GPU backends (enabled by default when the toolchain is found):
  - CUDA Toolkit for CUDA.
  - OpenCL headers and library. The CUDA Toolkit commonly supplies both.
  - Windows SDK `dxc.exe`, D3D12 and `d3dcompiler` for DirectX.
  - Python 3 for embedding OpenCL and DirectX shader blobs during configure/build.
- macOS: Xcode or Command Line Tools, including `clang` and `Rez`.
  - Python 3 for embedding the Metal kernel source during configure/build.
  - Builds target macOS 11.0 (Big Sur) or newer (`CMAKE_OSX_DEPLOYMENT_TARGET`).

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

macOS Release (Unix Makefiles or Xcode):

```sh
cmake -S . -B build/Mac
cmake --build build/Mac --config Release
```

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
When `dxc` is present, the default Windows ship build advertises CUDA, OpenCL and
DirectX together.

CMake options (all optional):

| Option | Default | Purpose |
| --- | --- | --- |
| `BPS_FORCE_CPU_ONLY` | OFF | Pure CPU build; no GPU backends or GPU PiPL flags. |
| `BPS_ENABLE_CUDA` | ON | CUDA backend when the toolkit is found. |
| `BPS_ENABLE_OPENCL` | ON | OpenCL backend when the SDK and Python 3 are found. |
| `BPS_ENABLE_DIRECTX` | ON | DirectX/HLSL backend when `dxc` and Python 3 are found. |
| `BPS_ENABLE_METAL` | ON | Metal backend on macOS when Python 3 is found. |
| `BPS_RENDER_DIAG` | OFF | Verbose render diagnostics. |
| `BPS_AE_PLUGIN_DIR` | — | Copy the built plug-in here after each build. |

### Installing Into After Effects

Copy the generated `.aex` or `.plugin` package into an After Effects plug-ins
folder.

Windows:

```text
C:\Program Files\Adobe\Adobe After Effects <version>\Support Files\Plug-ins\
```

macOS:

```text
/Applications/Adobe After Effects <version>/Plug-ins/
```

CMake can also copy the built plug-in after each build:

```powershell
cmake -S . -B build/Win -DBPS_AE_PLUGIN_DIR="C:/Program Files/Adobe/Adobe After Effects 2025/Support Files/Plug-ins"
```

```sh
cmake -S . -B build/Mac -DBPS_AE_PLUGIN_DIR="/Applications/Adobe After Effects 2025/Plug-ins"
```

The effect is registered in the `Stylize` category as `Bitonic Pixel Sorter`.

After rebuilding on macOS, **fully quit and relaunch After Effects** before
testing. AE caches plug-in metadata (including PiPL version and out-flags) in
`~/Library/Preferences/com.Adobe.After Effects.<version>.plist`. If the host
was still running, or the `.plugin` bundle directory timestamp did not change,
AE may load the cached metadata and report a version mismatch even though the
binary on disk is current.

### Verified Host Environments

Tested in the After Effects host:

| Platform | After Effects versions |
| --- | --- |
| macOS (Apple Silicon) | 2023, 2024, 2025, 2026 |
| Windows | 2023, 2024, 2025, 2026 |

GPU backend coverage:

| Platform | Verified | Experimental (built/advertised, not in-host verified) |
| --- | --- | --- |
| Windows | CUDA | OpenCL, DirectX |
| macOS (Apple Silicon) | Metal | — |

### Status

Validated:

- CMake configure and Release/Debug builds on Windows (Visual Studio 17 2022 x64)
  and macOS (Xcode or Unix Makefiles).
- In-host loading and effect operation on the AE versions listed above.
- Windows release `.aex` exports `EffectMain`, `PluginDataEntryFunction`, and
  `PluginDataEntryFunction2`.
- Direct OpenCL runtime compile check succeeds; NVIDIA's compiler emits only a
  non-fatal noinline warning.
- macOS universal `.plugin` ad-hoc signed with PiPL embedded in `__plgin` and
  `Contents/Resources`.

Still pending:

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

**現在のプラグインバージョン: 1.1（リリース）**

### 概要

Bitonic Pixel Sorter は、選択したトリガーキーがしきい値範囲の内側または外側にある
連続ピクセル区間をソートします。水平/垂直の軸方向、自由角度、中心点を基準にした
回転方向、中心点から外側へ伸びる放射方向、アルキメデス螺旋、または
ユーザー指定のマスクパス（オープンパスは両端を直線延長）に対応し、対象区間は
独立したソート基準の昇順または降順で並べ替えられます。ソート後の各区間には角度指定の
循環シフトも適用できます。

現在の AE 移植版には次が含まれます。

- 8 bit / 16 bit / 32 bit float の AE ワールド向け CPU SmartFX レンダリング。
- AE BGRA128 GPU ワールド向けの CUDA / OpenCL / DirectX / Metal GPU ビルドパス。
- 英語・日本語・簡体中国語・韓国語のパラメータ文字列。
- Adobe After Effects SDK の Examples ツリーと、同梱のヘッダオンリー
  ローカルヘルパーのみを参照する自己完結型 CMake ビルド。
- 螺旋とパスは CPU/GPU 共通の画素所有型変形パスマップを使い、フレーム内の
  各出力画素を一度だけ所有することで、丸め domain の穴や衝突による破壊的
  artefact を避けます。

Metal はユニバーサル `.plugin` としてコンパイル・リンクされ、
コンピュートカーネルはランタイムコンパイルされる Metal ソースとして埋め込まれます。
Apple Silicon Mac 上の After Effects 2023〜2026 で検証済みです。

### パラメータ

Effect Controls パネルと同じ名前・順番です。

| パラメータ | 説明 |
| --- | --- |
| GPUアクセラレーション | Mercury GPU パスの状態（読み取り専用）。 |
| モード | 軸方向、自由角度、回転、放射、螺旋、パスのソート方式。 |
| 並び順 | 昇順または降順。 |
| しきい値（下限） | トリガーキーの下限（パーセント）。 |
| しきい値（上限） | トリガーキーの上限（パーセント）。 |
| ソートトリガー | しきい値判定キー。輝度、RGB平均、RGB積、RGB最小、RGB最大、赤チャンネル、緑チャンネル、青チャンネル、アルファチャンネル、色相、彩度。 |
| トリガーソースレイヤー | トリガーキーを評価するレイヤー（`なし` = エフェクトソース）。 |
| ソート基準 | 並べ替えキー。選択肢はソートトリガーと同じ。 |
| 基準ソースレイヤー | ソートキーを評価するレイヤー（`なし` = エフェクトソース）。 |
| 影響 | しきい値内、またはしきい値外のどちらをソート対象にするか。 |
| 循環 | ソート後の各 run に適用する循環シフト量。 |
| 方向 | 軸方向モードでの水平ソート、または垂直ソート。 |
| 角度 | 自由角度モードのソート方向。 |
| 中心 | 回転/放射/螺旋モードの中心点。 |
| 螺旋量 | 螺旋モードの捩れ量（360° = フレーム最遠角までの 1 回転。符号で方向を選択）。 |
| パス | パスモードで使うレイヤーマスクパス。 |
| パス方向 | 法線方向またはタンジェント方向にソート。 |

既定値: モード = 軸方向、並び順 = 昇順、しきい値 = 40% / 60%、
パス方向 = タンジェント、トリガー/基準ソースレイヤー = `なし`。

既定の輝度キーは上流と同じ重み付けを用います。

```text
0.298912 * R + 0.586611 * G + 0.114478 * B
```

螺旋は `phase = wrap(theta - signedAmount * radius)` として分類されます。
螺旋量は角度コントロールで、360° がフレーム最遠角での 1 回転に対応し、符号で
方向を選びます。`0` では放射モードと同じ line 分類になり、値を大きくすると
半径方向に外へ進むアルキメデス螺旋としてソートされます。

基準ソースレイヤーとトリガーソースレイヤーの既定は `なし` で、その場合は
エフェクトソースを使います。別レイヤーを指定すると、同じレイヤー座標でその
レイヤーからキーを評価し、動かすピクセルはエフェクトソースのままです。

パスモードでは、全出力画素をサンプリング済み polyline の最近傍セグメントに
所属させます。端点がフレーム内にあるオープンパスは端点 tangent 方向へ
描画端まで延長し、クローズパスは延長しません。自己交差は最近傍セグメントの
Voronoi 的な所属として扱い、等距離の場合は短い arc length、セグメント順、
画素順で決定的に解決します。

### リポジトリ構成

```text
BitonicPixelSorterForAE/
  CMakeLists.txt              AE プラグインのビルド定義
  Directory.Build.props       Windows SDK / MSBuild の既定出力プロパティ
  Source/                     PF エフェクトのエントリポイント、CPU パス、GPU ディスパッチ、Metal バックエンド
  GPU/                        CUDA / OpenCL / HLSL / Metal のビトニックカーネル
  Localise/                   EN / JA / ZH / KO の文字列テーブルと AELocalise ヘルパ
  PiPL/                       AE PiPL リソースソース
  Mac/                        macOS バンドルメタデータテンプレート
  cmake/                      ビルド補助スクリプト
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
- Windows の GPU バックエンド（ツールチェーンが見つかれば既定で有効）:
  - CUDA: CUDA Toolkit。
  - OpenCL: ヘッダとライブラリ。CUDA Toolkit に同梱されていることが多いです。
  - DirectX: Windows SDK の `dxc.exe`、D3D12、`d3dcompiler`。
  - configure / build 時に OpenCL および DirectX シェーダ blob を埋め込む Python 3。
- macOS: Xcode または Command Line Tools（`clang` と `Rez` を含む）。
  - configure / build 時に Metal カーネルソースを埋め込む Python 3。
  - ビルド対象は macOS 11.0（Big Sur）以降（`CMAKE_OSX_DEPLOYMENT_TARGET`）。

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

macOS Release（Unix Makefiles または Xcode）:

```sh
cmake -S . -B build/Mac
cmake --build build/Mac --config Release
```

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
`dxc` がある環境では、既定の Windows リリースビルドは CUDA / OpenCL / DirectX を
まとめて広告します。

CMake オプション（いずれも任意）:

| オプション | 既定 | 用途 |
| --- | --- | --- |
| `BPS_FORCE_CPU_ONLY` | OFF | 純 CPU ビルド。GPU バックエンドと GPU PiPL フラグなし。 |
| `BPS_ENABLE_CUDA` | ON | ツールキットが見つかれば CUDA バックエンド。 |
| `BPS_ENABLE_OPENCL` | ON | SDK と Python 3 があれば OpenCL バックエンド。 |
| `BPS_ENABLE_DIRECTX` | ON | `dxc` と Python 3 があれば DirectX/HLSL バックエンド。 |
| `BPS_ENABLE_METAL` | ON | macOS で Python 3 があれば Metal バックエンド。 |
| `BPS_RENDER_DIAG` | OFF | 詳細なレンダ診断ログ。 |
| `BPS_AE_PLUGIN_DIR` | — | ビルド後にプラグインをここへコピー。 |

### After Effects へのインストール

生成された `.aex` または `.plugin` を After Effects のプラグインフォルダへ
コピーします。

Windows:

```text
C:\Program Files\Adobe\Adobe After Effects <version>\Support Files\Plug-ins\
```

macOS:

```text
/Applications/Adobe After Effects <version>/Plug-ins/
```

CMake でビルド後に自動コピーすることもできます。

```powershell
cmake -S . -B build/Win -DBPS_AE_PLUGIN_DIR="C:/Program Files/Adobe/Adobe After Effects 2025/Support Files/Plug-ins"
```

```sh
cmake -S . -B build/Mac -DBPS_AE_PLUGIN_DIR="/Applications/Adobe After Effects 2025/Plug-ins"
```

エフェクトは `Stylize` カテゴリに `Bitonic Pixel Sorter` として登録されます。

macOS で再ビルドしたあとは、**After Effects を完全終了してから起動し直して**
ください。AE はプラグインメタデータ（PiPL バージョンや out-flags を含む）を
`~/Library/Preferences/com.Adobe.After Effects.<version>.plist` の PluginCache に
保持します。AE 起動中のまま差し替えた場合や `.plugin` バンドルのディレクトリ
タイムスタンプが更新されない場合、ディスク上は最新でもキャッシュ由来の
バージョン不一致エラー（84601 / 88601）が出ることがあります。

### 検証環境

After Effects ホスト内で動作確認済み:

| プラットフォーム | After Effects バージョン |
| --- | --- |
| macOS（Apple Silicon） | 2023, 2024, 2025, 2026 |
| Windows | 2023, 2024, 2025, 2026 |

GPU バックエンドの検証状況:

| プラットフォーム | 検証済み | 試験的（ビルド・広告済み、ホスト内未検証） |
| --- | --- | --- |
| Windows | CUDA | OpenCL, DirectX |
| macOS（Apple Silicon） | Metal | — |

### ステータス

検証済み:

- Windows（Visual Studio 17 2022 x64）および macOS（Xcode または Unix Makefiles）での
  CMake configure と Release / Debug ビルド。
- 上記 AE バージョンでのホスト内ロードとエフェクト動作。
- Windows リリース `.aex` が `EffectMain`、`PluginDataEntryFunction`、
  `PluginDataEntryFunction2` をエクスポートすること。
- OpenCL のランタイムコンパイル直接チェックの成功。NVIDIA コンパイラは
  非致命的な noinline 警告のみを出力。
- macOS ユニバーサル `.plugin` の ad-hoc 署名と、`__plgin` /
  `Contents/Resources` への PiPL 埋め込み。

未対応:

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

**当前插件版本：1.1（发布版）**

### 概述

Bitonic Pixel Sorter 对所选触发键落在阈值范围内或范围外的连续像素区间进行排序。
它支持水平/垂直轴向、自由角度、围绕中心点的旋转路径、从中心点向外的放射路径、
阿基米德螺旋，以及用户指定的蒙版路径（开放路径两端直线延长）。每个符合条件的区间
可按独立的排序标准升序或降序排列，排序后的区间还可应用角度循环位移。

当前 AE 移植版包含：

- 面向 8 位、16 位和 32 位浮点 AE 世界的 CPU SmartFX 渲染。
- 面向 AE BGRA128 GPU 世界的 CUDA、OpenCL、DirectX 和 Metal GPU 构建路径。
- 英语、日语、简体中文和韩语参数字符串。
- 仅引用 Adobe After Effects SDK Examples 树及随附仅头文件本地辅助代码的
  自包含 CMake 构建。
- 螺旋与路径模式使用 CPU/GPU 共用的像素所属变换路径图，确保帧内每个输出像素
  仅被拥有一次，避免舍入域空洞或碰撞造成的破坏性伪影。

Metal 可编译并链接为通用 `.plugin`，计算内核以运行时编译的 Metal 源码形式
嵌入。已在 Apple Silicon Mac 的 After Effects 2023 至 2026 中验证。

### 参数

名称与顺序与 Effect Controls 面板一致。

| 参数 | 含义 |
| --- | --- |
| GPU 加速 | Mercury GPU 路径状态（只读）。 |
| 模式 | 轴向、自由角度、旋转、放射、螺旋或路径排序。 |
| 排序 | 升序或降序。 |
| 阈值下限 | 触发键下限（百分比）。 |
| 阈值上限 | 触发键上限（百分比）。 |
| 排序触发 | 阈值判定键：亮度、RGB平均值、RGB乘积、RGB最小值、RGB最大值、红色通道、绿色通道、蓝色通道、Alpha通道、色相、饱和度。 |
| 触发源图层 | 用于计算触发键的图层（`无` = 效果源）。 |
| 排序标准 | 排序键，选项与排序触发相同。 |
| 标准源图层 | 用于计算排序键的图层（`无` = 效果源）。 |
| 影响 | 对阈值内或阈值外的像素排序。 |
| 循环 | 对每个已排序区间应用的循环位移。 |
| 方向 | 轴向模式下的水平或垂直排序。 |
| 角度 | 自由角度模式的排序方向。 |
| 中心 | 旋转/放射/螺旋模式的中心点。 |
| 螺旋量 | 螺旋扭转量（360° = 到画面最远角的一整圈；符号选择方向）。 |
| 路径 | 路径模式使用的图层蒙版路径。 |
| 路径方向 | 沿法线或切线方向排序。 |

默认值：模式 = 轴向，排序 = 升序，阈值 = 40% / 60%，
路径方向 = 切线，触发/标准源图层 = `无`。

默认亮度键沿用上游权重：

```text
0.298912 * R + 0.586611 * G + 0.114478 * B
```

螺旋按 `phase = wrap(theta - signedAmount * radius)` 分类。螺旋量是角度控件，
360° 对应画面最远角处一整圈，符号选择方向。`0` 与放射模式使用相同的 line
分类；更大的值会形成向外的阿基米德螺旋路径。

标准源图层与触发源图层默认为 `无`，此时使用效果源。指定其他图层时，在相同
图层坐标上从该图层读取键，移动的像素仍来自效果源。

路径模式将每个输出像素分配到采样折线的最近线段。开放路径在帧内的端点沿切线
延伸至画面边缘；闭合路径不延伸。自交按最近线段所属关系确定性处理，距离相同时
按较短弧长、线段顺序、像素顺序解决。

### 仓库结构

```text
BitonicPixelSorterForAE/
  CMakeLists.txt              AE 插件构建定义
  Directory.Build.props       Windows SDK / MSBuild 默认输出属性
  Source/                     PF 效果入口、CPU 路径、GPU 调度、Metal 后端
  GPU/                        CUDA、OpenCL、HLSL 与 Metal 双调内核
  Localise/                   英 / 日 / 中 / 韩字符串表与 AELocalise 辅助头
  PiPL/                       AE PiPL 资源源码
  Mac/                        macOS 包元数据模板
  cmake/                      构建辅助脚本
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
- Windows GPU 后端（检测到工具链时默认启用）：
  - CUDA：CUDA Toolkit。
  - OpenCL：头文件与库。CUDA Toolkit 通常同时提供二者。
  - DirectX：Windows SDK 的 `dxc.exe`、D3D12 与 `d3dcompiler`。
  - 在 configure / build 阶段嵌入 OpenCL 与 DirectX 着色器 blob 所需的 Python 3。
- macOS：Xcode 或 Command Line Tools（含 `clang` 与 `Rez`）。
  - 在 configure / build 阶段嵌入 Metal 内核源码所需的 Python 3。
  - 构建目标为 macOS 11.0（Big Sur）或更高（`CMAKE_OSX_DEPLOYMENT_TARGET`）。

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

macOS Release（Unix Makefiles 或 Xcode）：

```sh
cmake -S . -B build/Mac
cmake --build build/Mac --config Release
```

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

DirectX 着色器 bytecode 与 root signature 在构建时嵌入 `.aex`。无需在插件模块旁放置单独的 `DirectX_Assets` 文件夹。当存在 `dxc` 时，默认 Windows 发布构建会同时声明 CUDA、OpenCL 与 DirectX。

CMake 选项（均为可选）：

| 选项 | 默认 | 用途 |
| --- | --- | --- |
| `BPS_FORCE_CPU_ONLY` | OFF | 纯 CPU 构建；无 GPU 后端与 GPU PiPL 标志。 |
| `BPS_ENABLE_CUDA` | ON | 检测到工具包时启用 CUDA 后端。 |
| `BPS_ENABLE_OPENCL` | ON | 检测到 SDK 与 Python 3 时启用 OpenCL 后端。 |
| `BPS_ENABLE_DIRECTX` | ON | 检测到 `dxc` 与 Python 3 时启用 DirectX/HLSL 后端。 |
| `BPS_ENABLE_METAL` | ON | macOS 上检测到 Python 3 时启用 Metal 后端。 |
| `BPS_RENDER_DIAG` | OFF | 详细渲染诊断日志。 |
| `BPS_AE_PLUGIN_DIR` | — | 每次构建后将插件复制到此目录。 |

### 安装到 After Effects

将生成的 `.aex` 或 `.plugin` 复制到 After Effects 插件文件夹。

Windows：

```text
C:\Program Files\Adobe\Adobe After Effects <version>\Support Files\Plug-ins\
```

macOS：

```text
/Applications/Adobe After Effects <version>/Plug-ins/
```

CMake 也可在每次构建后自动复制已构建的插件：

```powershell
cmake -S . -B build/Win -DBPS_AE_PLUGIN_DIR="C:/Program Files/Adobe/Adobe After Effects 2025/Support Files/Plug-ins"
```

```sh
cmake -S . -B build/Mac -DBPS_AE_PLUGIN_DIR="/Applications/Adobe After Effects 2025/Plug-ins"
```

效果在 `Stylize` 类别下注册为 `Bitonic Pixel Sorter`。

在 macOS 上重新构建后，**请完全退出并重新启动 After Effects** 再测试。AE 会在
`~/Library/Preferences/com.Adobe.After Effects.<version>.plist` 的 PluginCache 中
缓存插件元数据（含 PiPL 版本与 out-flags）。若 AE 仍在运行，或 `.plugin` 包目录
时间戳未更新，即使磁盘上的二进制已是最新，仍可能因缓存报告版本不匹配（84601 / 88601）。

### 验证环境

已在 After Effects 宿主中测试：

| 平台 | After Effects 版本 |
| --- | --- |
| macOS（Apple Silicon） | 2023, 2024, 2025, 2026 |
| Windows | 2023, 2024, 2025, 2026 |

GPU 后端验证情况：

| 平台 | 已验证 | 实验性（已构建/声明，宿主内未验证） |
| --- | --- | --- |
| Windows | CUDA | OpenCL, DirectX |
| macOS（Apple Silicon） | Metal | — |

### 状态

已验证：

- 在 Windows（Visual Studio 17 2022 x64）与 macOS（Xcode 或 Unix Makefiles）上的
  CMake configure 及 Release / Debug 构建。
- 上述 AE 版本中的宿主内加载与效果运行。
- Windows 发布 `.aex` 导出 `EffectMain`、`PluginDataEntryFunction` 与
  `PluginDataEntryFunction2`。
- 直接 OpenCL 运行时编译检查通过；NVIDIA 编译器仅发出非致命的 noinline 警告。
- macOS 通用 `.plugin` 的 ad-hoc 签名，以及 PiPL 嵌入 `__plgin` 与
  `Contents/Resources`。

尚待完成：

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

**현재 플러그인 버전: 1.1(릴리스)**

### 개요

Bitonic Pixel Sorter는 선택한 트리거 키가 임계값 범위 안 또는 밖에 있는 연속
픽셀 구간을 정렬합니다. 수평/수직 축 방향, 자유 각도, 중심점 기준 회전 경로,
중심점에서 바깥쪽으로 뻗는 방사 경로, 아르키메데스 나선, 사용자 지정 마스크
패스(열린 패스는 양 끝을 직선 연장)를 지원하며, 각 대상 구간은 독립적인 정렬
기준으로 오름차순 또는 내림차순 정렬할 수 있습니다. 정렬된 각 run에는 각도
순환 이동도 적용할 수 있습니다.

현재 AE 포트에는 다음이 포함됩니다.

- 8비트, 16비트, 32비트 float AE 월드용 CPU SmartFX 렌더링.
- AE BGRA128 GPU 월드용 CUDA, OpenCL, DirectX, Metal GPU 빌드 경로.
- 영어, 일본어, 간체 중국어, 한국어 매개변수 문자열.
- Adobe After Effects SDK Examples 트리와 동봉된 헤더 전용 로컬 헬퍼만
  참조하는 자체 완결형 CMake 빌드.
- 나선과 패스 모드는 CPU/GPU 공통의 픽셀 소유형 변형 경로 맵을 사용해
  프레임 내 각 출력 픽셀이 한 번만 소유되도록 하며, 반올림 domain의 구멍이나
  충돌로 인한 파괴적 artefact를 방지합니다.

Metal은 유니버설 `.plugin`으로 컴파일 및 링크되며, 컴퓨트 커널은 런타임
컴파일되는 Metal 소스로 임베드됩니다. Apple Silicon Mac의 After Effects
2023~2026에서 검증되었습니다.

### 매개변수

이름과 순서는 Effect Controls 패널과 같습니다.

| 매개변수 | 의미 |
| --- | --- |
| GPU 가속 | Mercury GPU 경로 상태(읽기 전용). |
| 모드 | 축 방향, 자유 각도, 회전, 방사, 나선, 패스 정렬. |
| 정렬 | 오름차순 또는 내림차순. |
| 임계값 하한 | 트리거 키 하한(퍼센트). |
| 임계값 상한 | 트리거 키 상한(퍼센트). |
| 정렬 트리거 | 임계값 판정 키: 휘도, RGB 평균, RGB 곱, RGB 최솟값, RGB 최댓값, 빨강 채널, 초록 채널, 파랑 채널, 알파 채널, 색상, 채도. |
| 트리거 소스 레이어 | 트리거 키를 평가할 레이어(`없음` = 효과 소스). |
| 정렬 기준 | 정렬 키. 선택지는 정렬 트리거와 동일. |
| 기준 소스 레이어 | 정렬 키를 평가할 레이어(`없음` = 효과 소스). |
| 영향 | 임계값 내부 또는 외부 픽셀을 정렬. |
| 순환 | 정렬된 각 run에 적용하는 순환 이동량. |
| 방향 | 축 방향 모드의 수평 또는 수직 정렬. |
| 각도 | 자유 각도 모드의 정렬 방향. |
| 중심 | 회전/방사/나선 모드의 중심점. |
| 나선량 | 나선 비틀림 양(360° = 프레임 최원각까지 한 바퀴, 부호로 방향 선택). |
| 패스 | 패스 모드에서 사용하는 레이어 마스크 패스. |
| 패스 방향 | 법선 또는 접선 방향으로 정렬. |

기본값: 모드 = 축 방향, 정렬 = 오름차순, 임계값 = 40% / 60%,
패스 방향 = 접선, 트리거/기준 소스 레이어 = `없음`.

기본 휘도 키는 상류와 동일한 가중치를 사용합니다.

```text
0.298912 * R + 0.586611 * G + 0.114478 * B
```

나선은 `phase = wrap(theta - signedAmount * radius)` 로 분류됩니다. 나선량은
각도 컨트롤이며, 360°가 프레임 최원각에서의 한 바퀴에 해당하고 부호로 방향을
고릅니다. `0`이면 방사 모드와 같은 line 분류를 쓰고, 값이 커지면 바깥으로
향하는 아르키메데스 나선으로 정렬됩니다.

기준 소스 레이어와 트리거 소스 레이어의 기본값은 `없음`이며, 이때는 효과
소스를 사용합니다. 다른 레이어를 지정하면 같은 레이어 좌표에서 해당 레이어의
키를 읽고, 이동하는 픽셀은 효과 소스 그대로입니다.

패스 모드는 각 출력 픽셀을 샘플링된 polyline의 가장 가까운 세그먼트에
할당합니다. 프레임 안의 열린 패스 끝점은 접선 방향으로 화면 가장자리까지
연장하고, 닫힌 패스는 연장하지 않습니다. 자기 교차는 가장 가까운 세그먼트
소유로 결정적으로 처리하며, 동일 거리일 때는 더 짧은 arc length, 세그먼트
순서, 픽셀 순서로 해결합니다.

### 저장소 구조

```text
BitonicPixelSorterForAE/
  CMakeLists.txt              AE 플러그인 빌드 정의
  Directory.Build.props       Windows SDK / MSBuild 기본 출력 속성
  Source/                     PF 이펙트 진입점, CPU 경로, GPU 디스패치, Metal 백엔드
  GPU/                        CUDA, OpenCL, HLSL, Metal 비토닉 커널
  Localise/                   EN / JA / ZH / KO 문자열 테이블 및 AELocalise 헬퍼
  PiPL/                       AE PiPL 리소스 소스
  Mac/                        macOS 번들 메타데이터 템플릿
  cmake/                      빌드 보조 스크립트
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
- Windows GPU 백엔드(툴체인이 발견되면 기본 활성):
  - CUDA: CUDA Toolkit.
  - OpenCL: 헤더와 라이브러리. CUDA Toolkit에 둘 다 포함되는 경우가 많습니다.
  - DirectX: Windows SDK `dxc.exe`, D3D12, `d3dcompiler`.
  - configure / build 시 OpenCL 및 DirectX 셰이더 blob을 임베드하는 Python 3.
- macOS: Xcode 또는 Command Line Tools(`clang`, `Rez` 포함).
  - configure / build 시 Metal 커널 소스를 임베드하는 Python 3.
  - 빌드 대상은 macOS 11.0(Big Sur) 이상(`CMAKE_OSX_DEPLOYMENT_TARGET`).

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

macOS Release(Unix Makefiles 또는 Xcode):

```sh
cmake -S . -B build/Mac
cmake --build build/Mac --config Release
```

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

DirectX 셰이더 bytecode와 root signature는 빌드 시 `.aex`에 임베드됩니다.
플러그인 모듈 옆에 `DirectX_Assets` 폴더를 둘 필요가 없습니다.
`dxc`가 있으면 기본 Windows 릴리스 빌드는 CUDA, OpenCL, DirectX를 함께
광고합니다.

CMake 옵션(모두 선택):

| 옵션 | 기본 | 용도 |
| --- | --- | --- |
| `BPS_FORCE_CPU_ONLY` | OFF | 순수 CPU 빌드. GPU 백엔드 및 GPU PiPL 플래그 없음. |
| `BPS_ENABLE_CUDA` | ON | 툴킷이 있으면 CUDA 백엔드. |
| `BPS_ENABLE_OPENCL` | ON | SDK와 Python 3가 있으면 OpenCL 백엔드. |
| `BPS_ENABLE_DIRECTX` | ON | `dxc`와 Python 3가 있으면 DirectX/HLSL 백엔드. |
| `BPS_ENABLE_METAL` | ON | macOS에서 Python 3가 있으면 Metal 백엔드. |
| `BPS_RENDER_DIAG` | OFF | 상세 렌더 진단 로그. |
| `BPS_AE_PLUGIN_DIR` | — | 빌드 후 플러그인을 여기로 복사. |

### After Effects에 설치

생성된 `.aex` 또는 `.plugin`을 After Effects 플러그인 폴더에 복사합니다.

Windows:

```text
C:\Program Files\Adobe\Adobe After Effects <version>\Support Files\Plug-ins\
```

macOS:

```text
/Applications/Adobe After Effects <version>/Plug-ins/
```

CMake로 빌드 후 자동 복사도 가능합니다.

```powershell
cmake -S . -B build/Win -DBPS_AE_PLUGIN_DIR="C:/Program Files/Adobe/Adobe After Effects 2025/Support Files/Plug-ins"
```

```sh
cmake -S . -B build/Mac -DBPS_AE_PLUGIN_DIR="/Applications/Adobe After Effects 2025/Plug-ins"
```

이펙트는 `Stylize` 카테고리에 `Bitonic Pixel Sorter`로 등록됩니다.

macOS에서 재빌드한 뒤에는 **After Effects를 완전히 종료한 다음 다시 실행**해
주세요. AE는 플러그인 메타데이터(PiPL 버전 및 out-flags 포함)를
`~/Library/Preferences/com.Adobe.After Effects.<version>.plist`의 PluginCache에
보관합니다. AE가 실행 중인 채로 교체했거나 `.plugin` 번들 디렉터리 타임스탬프가
갱신되지 않으면, 디스크의 바이너리가 최신이어도 캐시 때문에 버전 불일치
(84601 / 88601)가 발생할 수 있습니다.

### 검증 환경

After Effects 호스트에서 테스트 완료:

| 플랫폼 | After Effects 버전 |
| --- | --- |
| macOS(Apple Silicon) | 2023, 2024, 2025, 2026 |
| Windows | 2023, 2024, 2025, 2026 |

GPU 백엔드 검증 상황:

| 플랫폼 | 검증됨 | 실험적(빌드/광고됨, 호스트 내 미검증) |
| --- | --- | --- |
| Windows | CUDA | OpenCL, DirectX |
| macOS(Apple Silicon) | Metal | — |

### 상태

검증됨:

- Windows(Visual Studio 17 2022 x64) 및 macOS(Xcode 또는 Unix Makefiles)에서의
  CMake configure와 Release / Debug 빌드.
- 위 AE 버전에서의 호스트 내 로드 및 이펙트 동작.
- Windows 릴리스 `.aex`가 `EffectMain`, `PluginDataEntryFunction`,
  `PluginDataEntryFunction2`를 export함.
- OpenCL 런타임 컴파일 직접 검사 성공. NVIDIA 컴파일러는 치명적이지 않은
  noinline 경고만 출력.
- macOS 유니버설 `.plugin` ad-hoc 서명 및 PiPL `__plgin` /
  `Contents/Resources` 임베드.

아직 남은 항목:

- 부분 Smart Render 출력 사각형 및 대형 프레임 타일 경계에 대한 GPU 강화.

### 라이선스 및 귀속

상류 프로젝트:

- `ruccho/BitonicPixelSorter`
- <https://github.com/ruccho/BitonicPixelSorter>
- MIT 라이선스(`LICENSE`에 보존)

이 포크는 핵심 이펙트, 밝기 모델, 비토닉 정렬 전략이 상류에서 유래함을
명시적으로 유지합니다. AE 호스트 통합, CMake 빌드, PiPL 리소스, 현지화 파일,
CPU / GPU 백엔드 적응은 이 After Effects 포크 고유의 작업입니다.
