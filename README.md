# Bitonic Pixel Sorter for After Effects

| Language | Name |
| --- | --- |
| English | Bitonic Pixel Sorter for After Effects |
| 日本語 | After Effects 対応ビットニック・ピクセルソーター |
| 中文 | After Effects 版双调像素排序 |
| 한국어 | After Effects용 비토닉 픽셀 정렬 |

This repository is an Adobe After Effects plug-in fork of
[ruccho/BitonicPixelSorter](https://github.com/ruccho/BitonicPixelSorter), an
MIT-licensed Unity / URP GPU pixel sorter. The original project remains the
algorithmic and attribution source. This fork reorganises the codebase as a
self-contained After Effects SDK effect plug-in and removes the Unity project
surface from the active repository layout.

## Overview

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

## Parameters

| English | 日本語 | 中文 | 한국어 | Meaning |
| --- | --- | --- | --- | --- |
| Direction | 方向 | 方向 | 방향 | Horizontal row sort or vertical column sort. |
| Order | 並び順 | 顺序 | 정렬 순서 | Ascending or descending brightness order. |
| Threshold Min | しきい値 Min | 阈值最小值 | 임계값 최소 | Lower luma bound for sortable pixels. |
| Threshold Max | しきい値 Max | 阈值最大值 | 임계값 최대 | Upper luma bound for sortable pixels. |

The luma key follows the upstream weights:

```text
0.298912 * R + 0.586611 * G + 0.114478 * B
```

## Repository Layout

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

## Prerequisites

- Adobe After Effects SDK, with this repository placed inside the SDK
  `Examples` directory or configured with `-DAESDK_ROOT=<path-to-Examples>`.
- CMake 3.18 or newer.
- Windows: Visual Studio 2022 with MSVC v143.
- Optional Windows GPU backends:
  - CUDA Toolkit for CUDA.
  - OpenCL headers and library. The CUDA Toolkit commonly supplies both.
  - Windows SDK `dxc.exe`, D3D12 and `d3dcompiler` for DirectX.
  - Python 3 for embedding the OpenCL kernel source during configure/build.
- macOS: Xcode or Command Line Tools, including `clang` and `Rez`.

## Build

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

When the DirectX backend is enabled, the Windows release package also includes:

```text
dist/Win/Release/DirectX_Assets/BitonicSortKernel.cso
dist/Win/Release/DirectX_Assets/BitonicSortKernel.rs
```

## Installing Into After Effects

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

## Status

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
- Hardening for large-frame tiling and AE render banding edge cases.

## Licence And Attribution

Upstream project:

- `ruccho/BitonicPixelSorter`
- <https://github.com/ruccho/BitonicPixelSorter>
- MIT licence, preserved in `LICENSE`.

This fork keeps the upstream attribution explicit because the core effect,
brightness model and bitonic sorting strategy derive from that project. The AE
host integration, CMake build, PiPL resources, localisation files and CPU/GPU
backend adaptation are specific to this After Effects fork.
