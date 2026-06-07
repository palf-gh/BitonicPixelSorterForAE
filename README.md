# BitonicPixelSorter

True realtime pixel sort effect for Unity with GPU-accelerated bitonic sorting.

[Live Demo (WebGPU port)](https://ruccho.com/bps-webgpu/)

The screenshot below shows it running on an NVIDIA GeForce RTX 4070 and it keeps over 1000 FPS at FHD resolution.

<img width="1920" height="1032" alt="Image" src="https://github.com/user-attachments/assets/8330a931-a737-4239-a60a-4bf2bf637961" />

## Installation

Use UPM git dependencies.
1. Open Package Manager and click `+` > `Add package from git URL...`
2. Enter `https://github.com/ruccho/BitonicPixelSorter.git?path=/Packages/com.ruccho.bitonicpixelsorter#release`

3. (Optional) To use a RendererFeature for UniversalRP, also install `https://github.com/ruccho/BitonicPixelSorter.git?path=/Packages/com.ruccho.bitonicpixelsorter.urp#release`


## BitonicPixelSorter Component

![image](https://user-images.githubusercontent.com/16096562/125492519-6a363ad6-87b3-451b-a6a3-37b859821db5.png)

|Property|Type|Description|
|-|-|-|
|Use As Image Effect|`bool`|It works as an image effect when attached to the camera. This is only active when you are using builtin render pipeline.|
|Shader|`ComputeShader`|Set `BoitonicPixelSorter.compute`.|
|Direction|`bool`|Switches sorting direction between horizontal / vertical.|
|Ascending|`bool`|Switches ordering.|
|Threshold Min|`float`|Lower threshold of the brightness.|
|Threshold Max|`float`|Upper threshold of the brightness.|

### Use from code

```csharp
var sorter = GetComponent<BitonicPixelSorter>();

//BitonicPixelSorter.Execute(Texture src, RenderTexture dst)
sorter.Execute(sourceTexture, destinationTexture);
```

## Use RendererFeature for UniversalRP

In your renderer asset, add `Bitonic Pixel Sorting Feature` to the renderer feature list.

## Algorithm

Pixel sorting sorts contiguous spans of pixels whose brightness falls within a threshold range, along each row (or column) of the image. BitonicPixelSorter performs this entirely on the GPU in a **single compute dispatch** — one thread group per line — using a [bitonic sorting network](https://en.wikipedia.org/wiki/Bitonic_sorter), a data-independent comparison network that maps perfectly to GPU parallelism. Everything happens in group shared memory; the source texture is read once and the result is written once.

### 1. Load

Each thread loads a contiguous chunk of the line, computes the brightness of each pixel, and stores a packed sort key into group shared memory: the original pixel index in the high 16 bits and the f16 brightness in the low 16 bits, so a compare-exchange later is just a 32-bit swap. An in-threshold bitmask of the chunk is kept in registers.

### 2. Span detection — parallel scan

Before sorting, each pixel pair needs to know the boundaries of the contiguous in-threshold span it belongs to. Instead of scanning the line sequentially, this is computed as a pair of parallel scans in `O(log n)` steps:

- *Span starts*: each thread finds the last out-of-threshold index in its own chunk, then a prefix-**max** scan (Hillis–Steele) across the group propagates it, and a short per-chunk walk resolves the start index for every pair.
- *Span ends*: symmetrically, a suffix-**min** scan of the first out-of-threshold index resolves the end index for every pair.

Pixels outside the threshold range form spans of length 1, so they never move during sorting.

### 3. Adaptive phase count

A parallel max-reduction finds the **longest span in the line**, and the number of bitonic phases is chosen as just enough for that length — `floor(log2(maxLen)) + 1` — instead of `ceil(log2(lineWidth))` for the whole line. Since the comparison cost grows with the square of the phase count, lines whose spans are short (the common case with thresholding) finish much earlier, and lines with no sortable span at all skip the sorting network entirely. The extra merge phase included for exact power-of-two span lengths guarantees that every span in the image is finally merged in the same direction.

### 4. Per-span bitonic sort

The bitonic network runs over the whole line in group shared memory. Comparator indices are computed **relative to each pair's span start**, so every span in the line is sorted independently inside one shared network without any divergent control flow. Spans of arbitrary (non-power-of-two) length are handled with the odd-network technique (see References): comparators that would reach past the span end are clamped to compare an element with itself.

### 5. Output

Only the sorted *indices* are used to gather the full-color pixels from the source texture into the output — colors are never shuffled through shared memory.

Because the sort runs in group shared memory, the sortable line length is limited to **2048 pixels** by default. For sources between 2049 and **4096 pixels**, a shader variant (`BPS_SIZE_4096` keyword) that doubles the shared memory budget to 16 KiB is selected automatically at some occupancy cost.

## References

https://github.com/hiroakioishi/UnityGPUBitonicSort

https://www.inf.hs-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm
