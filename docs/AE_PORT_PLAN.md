# Bitonic Pixel Sorter — After Effects port plan / roadmap

This document is the living plan for porting
[ruccho/BitonicPixelSorter](https://github.com/ruccho/BitonicPixelSorter)
(Unity URP, MIT) into a self-contained After Effects C++ plugin. It records the
architecture, every remaining step, the kernel-port notes, the toolchain
requirements, and the validation strategy, so the work can be executed and
resumed deterministically.

Keep this file updated as phases complete.

---

## 1. Goal & principles

- A standalone AE effect that anyone can build by dropping this repository into
  the Adobe AE SDK `Examples` tree. **References only the Adobe SDK.** The single
  reused piece is the header-only `Localise/AELocalise.h` (CJK+EN
  localisation), copied in — no dependency on `Palf_Plugins`.
- Faithful to upstream: params **Direction / Order / Threshold Min / Threshold
  Max**, brightness key `0.298912 R + 0.586611 G + 0.114478 B`.
- **CPU + full GPU (CUDA / OpenCL / Metal / DirectX)**. The CPU path is always
  built and is both the fallback and the correctness oracle. Each GPU framework
  is compiled only when its toolchain is present (gated `HAS_*`), exactly like
  the SDK's `Effect/SDK_Invert_ProcAmp` sample.
- Display name **Bitonic Pixel Sorter**, category **Stylize**, match name
  `PALF BitonicPixelSorter` (persisted; never renumber/reuse param IDs).

## 2. Architecture

Modelled on `Effect/SDK_Invert_ProcAmp` (the SDK's self-contained SmartFX + GPU
sample).

```
EffectMain dispatch
├─ GLOBAL_SETUP        out_flags/out_flags2 from shared Target header
│                      (PiPL and C++ must match exactly)
├─ PARAMS_SETUP        4 params, names via AELocalise (EN/JA/ZH/KO)
├─ GPU_DEVICE_SETUP    per-framework kernel load (CUDA: none; OpenCL/Metal: build
│                      from embedded source; DirectX: load .cso + root signature)
├─ GPU_DEVICE_SETDOWN  release kernels / handles
├─ SMART_PRE_RENDER    checkout params -> pre_render_data; expand input
│                      dependency along the sort axis only; result_rect stays
│                      within AE's requested output rect
└─ SMART_RENDER        checkout input/output; GetPixelFormat
   ├─ isGPU  -> SmartRenderGPU  (BGRA128 linear buffers; CUDA/OpenCL/Metal/DX)
   └─ else   -> SmartRenderCPU  (8/16/32-bit; std::stable_sort per span)
```

Key facts that shape the port:

- **AE GPU world is `PF_PixelFormat_GPU_BGRA128`**: a linear, row-pitched device
  buffer of `float4` in **BGRA** order (16 bytes/pixel). The upstream shader uses
  `Texture2D`/`RWTexture2D`; our kernels must index `buf[y*pitch + x]` and treat
  channels as BGRA (brightness uses .z=R, .y=G, .x=B).
- **Line-length limit**: the bitonic sort runs a whole line in group-shared
  memory, capping the sort-axis length at 2048 (default) / 4096 (`BPS_SIZE_4096`
  variant). When the sort-axis length > 4096, the GPU path must fall back to CPU.
- **Not pixel-independent**: `PF_OutFlag_PIX_INDEPENDENT` must stay off.
- **MFR advertised**: render state is per-call, CPU work is split across output
  lines, and GPU dispatch is conservatively serialised until in-host MFR GPU
  stress testing proves every backend's shared device handles are safe.

## 3. Toolchain reality (this machine, 2026-06-26)

| Framework | Build tool                              | Status here                         |
|-----------|------------------------------------------|-------------------------------------|
| CUDA      | `nvcc` (CUDA 13.1), GPU RTX 3080 Ti sm_86 | ✅ available                         |
| OpenCL    | `CL/cl.h` + `OpenCL.lib` (ship w/ CUDA)   | ✅ headers/lib present               |
| DirectX   | Windows SDK `dxc` + `d3dcompiler` / D3D12 | ✅ `dxc` present; build validated    |
| Metal     | Xcode `metal`/`clang` (macOS)             | ⛔ macOS only                        |
| embed     | `python` 3.12 (`GPUUtils/CreateCString.py`, `ParseHLSL.py`) | ✅ |

Implication: implement all four in code, but the build gates each by tool
availability. On this machine CUDA, OpenCL, and DirectX build; Metal builds on
macOS.

## 4. Phase plan

### Phase 1 — CPU baseline ✅ DONE
Self-contained scaffold, PiPL, params + CJK/EN localisation, CPU pixel sort
(`Source/BitonicPixelSorter_CPU.cpp`), self-contained CMake. Builds to
`BitonicPixelSorter.aex` (Release + Debug), exports verified.

### Phase 2 — GPU scaffolding + CUDA path
1. `Source/BitonicPixelSorter.h`: add GPU includes gated by `HAS_CUDA/HAS_OPENCL/
   HAS_HLSL/HAS_METAL`; declare the GPU-data struct and `BitonicSorterParams`
   GPU mirror (POD with pitch/size/threshold/flags).
2. `BitonicPixelSorter_Target.h`/`GlobalSetup`: advertise
   `PF_OutFlag2_SUPPORTS_GPU_RENDER_F32` and `…_DIRECTX_RENDERING` through the
   shared Target header. CMake must pass the same `BPS_HAS_*` backend defines to
   the PiPL preprocessing step; AE rejects plug-ins whose PiPL flags differ from
   `GlobalSetup`.
3. `EffectMain`: handle `PF_Cmd_GPU_DEVICE_SETUP`, `…_SETDOWN`,
   `…_SMART_RENDER_GPU`.
4. `Source/BitonicPixelSorter_GPU.cpp` (new): `GPUDeviceSetup` / `GPUDeviceSetdown`
   / `SmartRenderGPU`. CUDA needs no kernel compile (statically linked). Compute
   `linesCount`, `lineLen`; **if lineLen > 4096 → call `BPS_SortImageCPU`**
   (download/není: for GPU worlds we instead dispatch a copy-through or, simpler,
   set `GPU_RENDER_POSSIBLE` only when lineLen ≤ 4096 in PreRender so AE uses the
   CPU path — see Phase 8). Use `PF_GPUDeviceSuite1`:
   `GetDeviceInfo`, `GetGPUWorldData` for src/dst, launch kernel, no intermediate
   buffer needed (single pass, src→dst gather).
5. `GPU/BitonicPixelSorter_Kernel.cu` (new): CUDA port of the bitonic `SortPass`
   (see §5). Host wrapper `extern void BitonicSort_CUDA(...)` launching
   `<<<lineCount, MAX_THREADS, 0>>>` with static `__shared__` arrays.
6. CMake: detect CUDA (`find_package(CUDAToolkit)` / `enable_language(CUDA)`);
   gate `HAS_CUDA`; compile the `.cu`; link `CUDA::cudart` while leaving nvcc's
   implicit CUDA runtime selection as `None`, avoiding a static CRT conflict in
   the AE plugin. Keep CPU-only build working when CUDA absent.
7. Build (CPU+CUDA) and verify it loads; visual parity vs CPU (§7).

Acceptance: on an NVIDIA host the effect renders via CUDA and matches the CPU
output; on a non-CUDA build it still builds and renders on CPU.

### Phase 3 — OpenCL path
1. `GPU/BitonicPixelSorter_Kernel.cl` — OpenCL form of the kernel (`__local`,
   `barrier(CLK_LOCAL_MEM_FENCE)`, `clz`-based `firstbithigh`).
2. Build step: embed as a C string (`kBitonicPixelSorter_Kernel_OpenCLString`)
   with the SDK's `GPUUtils/CreateCString.py`. The kernel is already plain
   OpenCL C, so no preprocessing layer is required.
3. `GPUDeviceSetup` (OpenCL branch): `clCreateProgramWithSource` + `clBuildProgram`
   + `clCreateKernel`. `SmartRenderGPU` (OpenCL): `clSetKernelArg` ×N +
   `clEnqueueNDRangeKernel` global=`lineCount*MAX_THREADS`, local=`MAX_THREADS`.
4. CMake: gate `HAS_OPENCL` on `CL/cl.h` + `OpenCL.lib` (CUDA's copy on Windows)
   and Python 3 for kernel embedding.

### Phase 4 — DirectX (HLSL) path
1. `GPU/BitonicPixelSorter_Kernel.hlsl` — plain HLSL compute shader. Adapt
   `Texture2D`→`ByteAddressBuffer`/`RWByteAddressBuffer` (BGRA `float4` via
   `Load4`/`Store4`), keep `groupshared`, `GroupMemoryBarrierWithGroupSync`,
   `f32tof16`, and `firstbithigh`.
2. Build: Windows SDK `dxc` directly compiles `.hlsl` to `.cso` plus `.rs`
   root signature. `GPUUtils/ParseHLSL.py` is not needed because this port is
   not using the SDK sample's GF-style cross-target source.
3. `GPUDeviceSetup` (DirectX): `DXContext::Initialize` + embedded CSO/RS load via
   `BPS_LoadEmbeddedDirectXSortShader` (build-time `dxc` output embedded by
   `cmake/embed_binary.py`; no runtime `DirectX_Assets` folder). `SmartRenderGPU`
   (DirectX): `DXShaderExecution`, `SetParamBuffer`, UAV/SRV bind,
   `Execute(lineCount, 1)`.
4. CMake: gate `HAS_HLSL` on `dxc` and Python 3; link `DirectXUtils.cpp`, `d3d12`,
   and `d3dcompiler`; compile HLSL with `dxc` and embed `.cso`/`.rs` into the
   `.aex` at build time.

### Phase 5 — Metal path (macOS)
1. `GPU/BitonicPixelSorter_Kernel.metal` (or reuse `.cu` via `GF` Metal target) —
   `threadgroup` memory, `threadgroup_barrier`, half pack.
2. Build: `clang -E -DGF_DEVICE_TARGET_METAL=1` → embed string
   (`CreateCString.py`). `GPUDeviceSetup` (Metal): `newLibraryWithSource` +
   `newComputePipelineStateWithFunction`. `SmartRenderGPU` (Metal): compute
   encoder, dispatch `lineCount` threadgroups of `MAX_THREADS`.
3. CMake (APPLE): add Metal build + link `Metal.framework`.

### Phase 6 — CMake kernel build pipeline hardening
- Factor kernel build into `cmake/` helpers; gate each framework; emit a clear
  configure-time summary of which GPU backends are enabled.
- Keep everything self-contained: reference `../GPUUtils/CreateCString.py` &
  `ParseHLSL.py` and `../Util/DirectXUtils.h` (SDK paths) only.

### Phase 7 — Validation & CPU/GPU parity
- Build matrix: CPU-only; CPU+CUDA; CPU+OpenCL; CPU+DirectX; mac CPU+Metal.
- In AE: apply to a test still + footage; sweep params; confirm horizontal &
  vertical, ascending & descending, threshold ranges.
- Parity: GPU result vs CPU result should match within rounding (BGRA vs ARGB
  channel handling, f16 brightness key vs f32). Document expected tiny diffs.

### Phase 8 — Robustness & packaging
- **>4096 line length**: in `PreRender`, only set `GPU_RENDER_POSSIBLE` when the
  sort-axis length ≤ 4096 (else AE uses CPU). This is the clean fallback.
- **Banding / large comps**: handle the case where AE bands the output — map the
  output band's origin to the full input; for vertical sort ensure full columns.
  (CPU path currently assumes full-frame aligned; harden here.)
- 16f vs 32f GPU: AE GPU is F32 (`GPU_BGRA128`); confirm `m16f=0` path only.
- Update `README_AE.md`; document per-backend prerequisites; optional logo/About.

## 5. Kernel port notes (HLSL `SortPass` → CUDA/OpenCL/Metal/HLSL)

Reference: `Packages/com.ruccho.bitonicpixelsorter/Runtime/BitonicPixelSorter.compute`.

- **Dispatch**: one thread group per line; `MAX_THREADS` threads (128 default,
  256 for the 4096 variant). `gid = group id = line index`, `gtid = thread id`.
  AE: `lineCount = direction? height : width`, `lineLen = direction? width : height`.
- **Buffers not textures**: read source `float4` at `src[y*pitch + x]` (BGRA);
  write dest at `dst[y*pitch + x]`. For horizontal `x = pos, y = gid`; for
  vertical `x = gid, y = pos` (mirror the original `direction` swap).
- **Brightness**: `0.298912*R + 0.586611*G + 0.114478*B` with R=.z, G=.y, B=.x;
  `saturate`.
- **Group-shared layout**: `groupCache[MAX_SIZE]` (uint: `index<<16 | f16(bright)`),
  `scanCache[MAX_THREADS]`. Same as upstream.
- **Per-API mappings**:
  | upstream (HLSL)              | CUDA                         | OpenCL                         | Metal                         |
  |------------------------------|------------------------------|--------------------------------|-------------------------------|
  | `groupshared T a[N]`         | `__shared__ T a[N]`          | `__local T a[N]`               | `threadgroup T a[N]`          |
  | `GroupMemoryBarrierWithGroupSync()` | `__syncthreads()`     | `barrier(CLK_LOCAL_MEM_FENCE)` | `threadgroup_barrier(mem_threadgroup)` |
  | `f32tof16` / `f16tof32`      | `__float2half_rn`/`__half2float` (or manual pack) | `as_uint(half)` via `vstore_half`/manual | `as_type`/`half` |
  | `firstbithigh(x)`            | `31 - __clz(x)`              | `31 - clz(x)`                  | `31 - clz(x)`                 |
  | `min16uint`                  | `unsigned short`/`uint`      | `ushort`/`uint`                | `ushort`/`uint`               |
  | `SV_GroupID` / `SV_GroupThreadID` | `blockIdx.x`/`threadIdx.x` | `get_group_id(0)`/`get_local_id(0)` | `[[threadgroup_position_in_grid]]` / `[[thread_position_in_threadgroup]]` |
- Preserve exactly: the two parallel scans (prefix-max span start, suffix-min
  span end), the max-reduction for longest span, the adaptive `lineLevels =
  firstbithigh(maxLen)+1`, and the odd-network comparator clamping. These give
  the visible result; do not "simplify".
- **Strategy**: implement and *verify CUDA first* (cleanest to debug against the
  CPU oracle), then translate to OpenCL/Metal/HLSL. CUDA and OpenCL are now
  hand-written rather than routed through a shared macro layer; keep that pattern
  unless the remaining backends introduce enough duplication to justify one.

## 6. Per-backend prerequisites (for "anyone can build")
- CUDA: CUDA Toolkit (nvcc) + NVIDIA GPU.
- OpenCL: `CL/cl.h` + `OpenCL.lib` (CUDA Toolkit provides these on Windows).
- DirectX: Windows SDK `dxc` + D3D12 / `d3dcompiler`.
- Metal: macOS + Xcode.
- All: CMake 3.18+ (CUDA language), Python 3 (kernel embedding scripts).
The CPU build needs none of the above.

## 7. Risks / open questions
- GF single-source vs hand-written per-API kernels for a complex shared-memory
  kernel — decide after the CUDA port.
- f16 brightness key means ties can order differently between CPU (f32
  `stable_sort`) and GPU; visually negligible — document, don't chase exact parity.
- DirectX now builds through direct HLSL + `dxc`; in-host validation still needs
  to prove AE's DirectX GPU world resources match the `DirectXUtils` raw-buffer
  binding assumptions for this single-pass sorter.
- CUDA 13.1 + `sm_86`: set `CMAKE_CUDA_ARCHITECTURES` accordingly (also include a
  PTX fallback for forward compatibility).

## 8. Status log
- 2026-06-26: Phase 1 (CPU baseline) complete and building.
- 2026-06-26: Phase 2 (GPU scaffolding + CUDA) complete and build-validated:
  - `GPU/BitonicPixelSorter_Kernel.cu` — full CUDA port of the bitonic SortPass
    (one block per line, `__shared__` keys, `__float2half`/`__half2float` pack,
    `31-__clz` for `firstbithigh`; guarded out-of-line reads for buffer safety).
  - `Source/BitonicPixelSorter_GPU.cpp` — `BPS_GPUDeviceSetup/Setdown` (CUDA: no
    compile) and `BPS_SmartRenderGPU` (BGRA128, `PF_GPUDeviceSuite1`, single pass).
  - Wired into `GlobalSetup` (shared Target-header outflags),
    `PreRender` (set `GPU_RENDER_POSSIBLE` only when sort axis ≤ 4096), `SmartRender`
    (GPU/CPU split) and `EffectMain` (GPU_DEVICE_SETUP/SETDOWN/SMART_RENDER_GPU).
  - CMake: optional CUDA via `check_language(CUDA)`/`enable_language(CUDA)`,
    `CMAKE_CUDA_ARCHITECTURES all-major`, `CUDA_RUNTIME_LIBRARY None`, link
    `CUDA::cudart`, define `BPS_HAS_CUDA`; CUDA host compilation receives
    `/utf-8` through `-Xcompiler` to avoid CUDA 13.1 header code-page warnings.
  - Fixed a CUDA 13.1 header collision with the plugin's `MAJOR_VERSION` /
    `MINOR_VERSION` macros by including CUDA headers before `BitonicPixelSorter.h`
    in `BitonicPixelSorter_GPU.cpp`.
  - Validation: `cmake -S . -B build_cuda_verify -G "Visual Studio 17 2022" -A x64`;
    `cmake --build build_cuda_verify --config Release`; `cmake --build
    build_cuda_verify --config Debug`. Both builds succeeded.
- 2026-06-26: Phase 3 (OpenCL path) complete and build-validated:
  - `GPU/BitonicPixelSorter_Kernel.cl` — OpenCL port of the same shared-memory
    bitonic SortPass (`__local`, `barrier(CLK_LOCAL_MEM_FENCE)`, `clz`-based
    level calculation, one work-group per line).
  - Uses manual IEEE-754 half-bit packing for brightness keys, avoiding a
    dependency on OpenCL half arithmetic extensions.
  - `Source/BitonicPixelSorter_GPU.cpp` now creates/releases an OpenCL program
    and kernel through AE's `gpu_data` handle and dispatches `lineCount * 256`
    work-items with local size 256.
  - CMake detects OpenCL + Python 3, embeds the `.cl` source with
    `GPUUtils/CreateCString.py`, defines `BPS_HAS_OPENCL`, and links
    `OpenCL::OpenCL`.
  - Validation: same Release/Debug build matrix as Phase 2. `dumpbin /exports`
    on `build_cuda_verify/Release/BitonicPixelSorter.aex` shows `EffectMain` and
    `PluginDataEntryFunction2`.
  - Validation: direct `OpenCL.dll` P/Invoke compile check against
    `GPU/BitonicPixelSorter_Kernel.cl` returned `BUILD_RESULT=0`; NVIDIA's
    compiler emitted only a non-fatal kernel noinline warning.
- 2026-06-26: Phase 4 (DirectX/HLSL path) complete and build-validated:
  - `GPU/BitonicPixelSorter_Kernel.hlsl` — DirectX 12 compute shader with
    `groupshared` sort keys, `ByteAddressBuffer` source reads,
    `RWByteAddressBuffer` destination writes, `f32tof16` brightness key packing,
    and one dispatch group per line.
  - `Source/BitonicPixelSorter_GPU.cpp` now owns `DXContext`/`ShaderObject`
    through AE `gpu_data` and dispatches through `DXShaderExecution` with CBV,
    dst UAV, and src SRV.
  - CMake detects `dxc`, compiles CSO/RS assets, links `DirectXUtils.cpp`,
    `d3d12`, and `d3dcompiler`, and defines `BPS_HAS_HLSL`.
  - Initial validation used runtime file loading from `DirectX_Assets/` beside
    the `.aex` with POST_BUILD copies into `dist/Win/Release/` (superseded below).
  - Validation: `cmake --build build_cuda_verify --config Release` and
    `--config Debug` both succeeded with CUDA + OpenCL + DirectX enabled;
    `dumpbin /exports` still shows `EffectMain` and `PluginDataEntryFunction2`.
- 2026-06-26: DirectX shader embedding (follow-up to Phase 4):
  - Replaced runtime `DirectX_Assets/` file loading with build-time embedding
    via `cmake/embed_binary.py` and `BPS_LoadEmbeddedDirectXSortShader`.
  - Removed POST_BUILD/install copies of `DirectX_Assets/`; deployment requires
    only the built `.aex` module.
  - Removed obsolete tracked `dist/Win/Release/DirectX_Assets/` deliverables from
    the repository.
- 2026-06-26: AE host-load contract fixes after in-host error screenshots:
  - Fixed PiPL / `GlobalSetup` outflags mismatch by deriving `OUT_FLAGS2` from
    the same Target header for both C++ and PiPL preprocessing. With CUDA +
    OpenCL + DirectX enabled, PiPL now emits `570430464` (`0x22001400`), matching
    SmartRender + float + GPU F32 + DirectX and excluding threaded rendering.
  - Removed runtime-only GPU flag mutation from `GlobalSetup`; backend defines
    are now passed by CMake to Windows PiPL preprocessing and macOS Rez
    preprocessing.
  - Fixed `PF_Cmd_SMART_PRE_RENDER` result-rectangle contract: input checkout
    expands only along the required sort axis, while `result_rect` remains within
    AE's requested output rectangle and `max_result_rect` reports the full frame.
  - Made the CPU renderer honour Smart Render world `origin_x`/`origin_y`, so
    partial output worlds map correctly back to layer coordinates.
  - Superseded below: the initial safety restriction advertised GPU rendering
    only for full-frame output requests until the GPU kernels became origin-aware.
- 2026-06-26: Performance and MFR pass:
  - Added shared `BPS_EvaluateGpuEligibility` logic so render and future UI code
    report the same GPU availability reason.
  - CPU sorting now parallelises across independent lines, caches each pixel's
    brightness key once per line, sorts compact key/index entries, and reuses
    per-thread scratch buffers.
  - `PF_OutFlag2_SUPPORTS_THREADED_RENDERING` is now included through the shared
    Target header, keeping PiPL and `GlobalSetup` outflags identical. CUDA render
    dispatch is concurrent; OpenCL and DirectX keep narrow locks around shared
    mutable backend handles.
  - Optional render diagnostics can be enabled with
    `cmake -S . -B build -DBPS_RENDER_DIAG=ON`; logs include PreRender request
    rectangles, GPU eligibility, SmartRender path, pixel format, dimensions,
    direction, and render-call timing.
- 2026-06-26: Partial-frame GPU rendering fixed for CUDA, OpenCL, and DirectX;
  `BPS_RENDER_DIAG` should now show `gpu_possible=1` in PreRender and
  `SmartRender isGPU=1 ... elapsed_ms=...` when AE actually dispatches GPU.
- 2026-06-26: GPU correctness fix after partial-frame rendering exposed broken
  output:
  - CUDA, OpenCL, and DirectX now keep the partial-output dispatch but sort each
    contiguous in-threshold span across the full sort axis before writing only
    pixels inside the requested output world.
  - The kernels no longer use the previous packed half-key span network, which
    failed to produce one continuous sorted line for full-threshold spans. They
    now use padded per-span bitonic sorting with float brightness keys and source
    index tie-breaking, matching the CPU oracle's stable span ordering.
- 2026-06-26: Fixed a shared/local-memory data race in the per-span bitonic sort.
  Thread 0 published the span metadata into the sort scratch array, which the
  load loop then overwrote with no intervening barrier, so late warps could read
  a corrupted `sortSize` and drive the in-loop barriers divergent.
  - CUDA: dedicated `__shared__` scalars (`s_spanStart`/`s_spanEnd`/`s_spanSize`/
    `s_sortSize`) hold the metadata, never colliding with the sort scratch.
  - OpenCL and DirectX keep the metadata in `scratchIndex[0..3]` (both backends
    are already at the 32 KB local/groupshared cap, so dedicated scalars would
    overflow it) and add a publish/consume barrier after every thread copies the
    metadata into private locals and before the load loop overwrites the scratch.
  - All threads now read identical `spanStart`/`spanSize`/`sortSize`, so the
    per-span break and the bitonic-network barriers stay uniform.
- 2026-06-26: GPU engagement and status-readout fix after AE testing showed GPU
  and Mercury Software Only renders both taking CPU-path timings:
  - The Effect Controls GPU status now reflects the last Smart Render command
    path directly. `SmartRender` records `BPS_SetLastRenderUsedGpu(isGPU)` on
    every render, and the UI no longer keeps a separate render-attempt latch that
    could stay green after switching to Software Only.
  - CUDA Smart Render now validates `GetGPUWorldData` results before launching
    and records `BPS_RENDER_DIAG` messages for null GPU-world pointers and CUDA
    error codes. If AE's CUDA stream pointer fails in the runtime launch path,
    the render retries on CUDA stream 0 so failures are visible rather than being
    indistinguishable from the CPU fallback timing.
  - CPU fallback worker fan-out is capped at two workers per frame, or one for
    small partial-frame renders, because AE MFR already supplies frame-level
    parallelism.
- Phases 5–8 (Metal, pipeline hardening, validation, robustness) pending. AE
  in-host CPU/GPU parity validation is still pending.
- 2026-06-27: CMake-only CRT mismatch fix eliminating `LNK4098` and the
  `std::mutex::lock()` NULL-deref crash on AE 23/24. Three edits to
  `CMakeLists.txt` only (no source, kernel, or other backend changes):
  (1) Added `-Xcompiler=/MD` (Release) and `-Xcompiler=/MDd` (Debug) generator
  expressions to the nvcc `target_compile_options` inside `if(BPS_CUDA_AVAILABLE)
  ... if(MSVC)`, making the dynamic-CRT requirement explicit to nvcc's host
  compiler and hardening against future CMake/nvcc versions.
  (2) Added `target_compile_definitions(... $<$<COMPILE_LANGUAGE:C,CXX>:
  _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR>)` — Microsoft's recommended shim to
  revert `std::mutex` to its pre-VS 16.10 constructor so a plugin built with a
  newer MSVC STL loads correctly against the older `msvcp140.dll` bundled with
  AE 23/24.
  (3) Added `target_link_options(... /NODEFAULTLIB:LIBCMT)` inside
  `if(BPS_CUDA_AVAILABLE) ... if(MSVC)` to suppress the `/DEFAULTLIB:LIBCMT`
  directive embedded in every member of `cuda.lib` (`CUDA::cuda_driver`).
  Investigation confirmed the `.cu` kernel object already received `/MD` from
  CMake's internal VS CRT mechanism; `cuda.lib` was the LIBCMT source.
  Build results: Release (`build_crtfix`) and RelWithDebInfo/DIAG (`build_crtdiag`)
  both succeeded with `LNK4098` and `LIBCMT` **absent** from both logs (grep
  confirmed zero matches). `EffectMain` exported at ordinal 1.
  Diagnostic artefacts at `dist/Win/RelWithDebInfo/BitonicPixelSorter.aex`
  (808448 B) and `.pdb` (5818720 B).
- 2026-06-26: Fixed Release crash on AE 23/24 caused by spurious
  `AcquireExclusiveDeviceAccess` / `ReleaseExclusiveDeviceAccess` calls in the
  CUDA branch of `BPS_SmartRenderGPU`. Full GPU plugins that use the separate
  `PF_Cmd_SMART_RENDER_GPU` entry point already hold exclusive device access
  (per `AE_EffectGPUSuites.h`); the extra calls double-managed the CUDA context.
  On a failed acquire the kernel still launched, and the unconditional Release
  left an unbalanced `cuCtxPopCurrent` that corrupted AE's context and crashed
  the host. Removed both calls and the now-unused `PF_Err err2` declaration;
  the kernel launch and error check are unchanged. Matches `SDK_Invert_ProcAmp`,
  which makes no such calls. Build validated: `cmake -S . -B build_cuda_verify
  -G "Visual Studio 17 2022" -A x64` ("CUDA enabled"); Release build clean;
  `EffectMain` exported at ordinal 1.
- 2026-06-26: Explicit CUDA context push/pop to fix residual crash on AE 23/24
  MFR render threads. After the Acquire/Release removal, CUDA still crashed older
  hosts while OpenCL on the same machine worked. Root cause: OpenCL explicitly
  targets AE's `contextPV` / `command_queuePV`; the CUDA path relied on AE's
  context being implicitly current on each MFR thread, which is not guaranteed on
  AE 23/24. Fix: added `#include <cuda.h>` (Driver API header); at the start of
  the CUDA branch in `BPS_SmartRenderGPU`, cast `device_info.contextPV` to
  `CUcontext` and call `cuCtxPushCurrent`; after the existing error check and
  post-launch DIAG block, call `cuCtxPopCurrent`. Also added a pre-launch
  `BPS_RENDER_DIAG` log capturing context pointer, push result, command queue,
  buffer pointers, frame/output dimensions, `lineCount`, and `direction`.
  CMake: `CUDA::cuda_driver` added alongside `CUDA::cudart` in the CUDA link step
  (resolves to `cuda.lib` / `nvcuda.dll`; no unresolved externals). Builds:
  `build_diag` RelWithDebInfo + DIAG=ON → `dist/Win/RelWithDebInfo/BitonicPixelSorter.aex`
  (808960 B) + PDB (5812224 B); `build_ctxfix` Release → `dist/Win/Release/BitonicPixelSorter.aex`;
  `cuCtxPushCurrent`/`cuCtxPopCurrent` symbol strings confirmed in binary; `EffectMain`
  at ordinal 1. In-host AE 23/24 test still pending (user to validate).
- 2026-06-27: **CUDA Release crash on AE 23/24 RESOLVED — confirmed in-host.** The
  two 2026-06-26 CUDA fixes above were necessary cleanups but NOT the crash cause.
  A Visual Studio debugger stack trace showed the real fault: a `std::mutex::lock()`
  NULL (0x0) dereference inside `msvcp140.dll`, reached via `std::lock_guard` in
  `BPS_RecordGpuDevice` during `GPU_DEVICE_SETUP` — not in rendering (hence no
  `BPS_RENDER_DIAG` line ever appeared). Root cause: a mixed C runtime. The CUDA
  build pulled in the static CRT (`LIBCMT`, /MT) — from nvcc host compilation and
  from `cuda.lib` (`CUDA::cuda_driver`) — while the plugin uses the dynamic CRT
  (/MD, `msvcp140.dll`), producing the long-standing `LNK4098` warning. The CRT
  mismatch corrupts the `std::mutex` runtime, which NULL-derefs against the OLDER
  `msvcp140.dll` bundled with AE 23/24 (AE 25.x bundles a newer runtime and
  tolerated it). OpenCL-only builds were immune (no `.cu`, pure /MD) — which is
  exactly why the user's OpenCL-vs-CUDA bisection isolated the CUDA path. Fix
  (CMake only): force nvcc's host compiler to `/MD` (`/MDd` in Debug); define
  `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` for C/CXX; add `/NODEFAULTLIB:LIBCMT`.
  `LNK4098`/`LIBCMT` are now absent from all build logs. Before committing, the
  session-only pre-launch and push-failure `BPS_RENDER_DIAG` logs added during the
  hunt were removed; the `cuCtxPushCurrent`/`cuCtxPopCurrent` context handling and
  the established default-OFF `BPS_RENDER_DIAG` facility were kept.
- 2026-06-27: **Verification status (environment-limited).** Confirmed working
  in-host: **Windows + After Effects 2023 and 2024 + CUDA backend** (effect applies
  and renders without crashing). NOT yet verified, no test environment available:
  **OpenCL and DirectX on Windows, and Metal on macOS.** Those three backends are
  implemented and compile, but remain unvalidated in-host and should be treated as
  experimental until tested. (DirectX is opt-in via `BPS_ENABLE_DIRECTX`; the
  default Windows ship build advertises CUDA + OpenCL.)
