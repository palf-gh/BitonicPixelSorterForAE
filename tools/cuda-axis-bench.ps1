param(
	[string]$ImagePath,
	[ValidateSet("Horizontal", "Vertical")]
	[string]$Direction = "Horizontal",
	[int]$Iterations = 60,
	[int]$Warmup = 6,
	[int]$LineStart = 0,
	[int]$LineCount = 0,
	[double]$ThresholdMin = 0.4,
	[double]$ThresholdMax = 0.6,
	[string]$Config = "RelWithDebInfo",
	[switch]$Descending
)

$ErrorActionPreference = "Stop"

if (-not $ImagePath) {
	throw "Pass -ImagePath with a JPEG/PNG file to benchmark."
}
if (-not (Test-Path -LiteralPath $ImagePath)) {
	throw "Image file not found: $ImagePath"
}

$magick = (Get-Command magick -ErrorAction SilentlyContinue).Source
if (-not $magick) {
	throw "ImageMagick 'magick' command was not found on PATH."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$benchCandidates = @(
	(Join-Path $repoRoot "build\Win-perf\tools\$Config\bps_cuda_axis_bench.exe"),
	(Join-Path $repoRoot "build\Win\tools\$Config\bps_cuda_axis_bench.exe"),
	(Join-Path $repoRoot "build\Win-perf\$Config\bps_cuda_axis_bench.exe"),
	(Join-Path $repoRoot "build\Win\$Config\bps_cuda_axis_bench.exe")
)

$benchExe = $benchCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $benchExe) {
	throw "bps_cuda_axis_bench.exe was not found. Build it first, for example: cmake --build build\Win-perf --config $Config --target bps_cuda_axis_bench"
}

$info = & $magick identify -format "%w %h" $ImagePath
if ($LASTEXITCODE -ne 0) {
	throw "ImageMagick identify failed."
}
$parts = $info -split "\s+"
$width = [int]$parts[0]
$height = [int]$parts[1]

$rawPath = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), ".rgba")
try {
	$rawSpec = "rgba:$rawPath"
	& $magick $ImagePath -alpha on -depth 8 $rawSpec
	if ($LASTEXITCODE -ne 0) {
		throw "ImageMagick raw export failed."
	}

	$args = @(
		"--raw", $rawPath,
		"--width", $width,
		"--height", $height,
		"--direction", $Direction.ToLowerInvariant(),
		"--iterations", $Iterations,
		"--warmup", $Warmup,
		"--line-start", $LineStart,
		"--line-count", $LineCount,
		"--threshold-min", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $ThresholdMin)),
		"--threshold-max", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}", $ThresholdMax))
	)
	if ($Descending) {
		$args += "--descending"
	}

	& $benchExe @args
	if ($LASTEXITCODE -ne 0) {
		throw "bps_cuda_axis_bench failed with exit code $LASTEXITCODE."
	}
}
finally {
	Remove-Item -LiteralPath $rawPath -Force -ErrorAction SilentlyContinue
}
