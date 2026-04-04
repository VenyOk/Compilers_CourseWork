param(
    [int]$Repeat = 5,
    [string[]]$Levels = @("O0", "O2", "O3"),
    [string]$Filter = "",
    [int64]$StackBytes = 67108864,
    [string]$Json = "outputs\bench_results_clang.json"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$BenchFiles = @(
    "bench_lcinv",
    "bench_csesin",
    "bench_matmul",
    "bench_stencil_2d",
    "bench_gs2d",
    "bench_gs3d",
    "bench_metelitsa_gs2d",
    "bench_metelitsa_dir2d",
    "bench_heavy_licm",
    "bench_heavy_sum",
    "bench_heavy_power",
    "bench_heavy_pi"
)

$BenchLabels = @{
    "bench_lcinv" = "LICM: SQRT invariant"
    "bench_csesin" = "CSE: SIN/COS repeated"
    "bench_matmul" = "Matmul 100x100"
    "bench_stencil_2d" = "2D stencil double-buffer"
    "bench_gs2d" = "2D Gauss-Seidel in-place"
    "bench_gs3d" = "3D Gauss-Seidel in-place"
    "bench_metelitsa_gs2d" = "Metelitsa: GS 2D Laplace"
    "bench_metelitsa_dir2d" = "Metelitsa: Dirichlet 2D"
    "bench_heavy_licm" = "[Heavy] LICM"
    "bench_heavy_sum" = "[Heavy] Sum"
    "bench_heavy_power" = "[Heavy] Power"
    "bench_heavy_pi" = "[Heavy] Pi"
}

$NativeParallelRuntime = Join-Path $RepoRoot "runtime\fortran_parallel_runtime.c"

function Resolve-ClangPath {
    $candidates = @(
        "C:\Program Files\LLVM\bin\clang.exe",
        (Join-Path $RepoRoot ".llvm\clang+llvm-22.1.2-x86_64-pc-windows-msvc\bin\clang.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    $cmd = Get-Command clang -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    throw "clang.exe not found"
}

function Normalize-Text([string]$Text) {
    if ($null -eq $Text) {
        return ""
    }
    return (($Text -replace "`r`n", "`n") -replace "`r", "`n").Trim()
}

function Try-ParseNumber([string]$Text, [ref]$Value) {
    return [double]::TryParse(
        $Text,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        $Value
    )
}

function Compare-Outputs([string]$Actual, [string]$Expected) {
    $actualNorm = Normalize-Text $Actual
    $expectedNorm = Normalize-Text $Expected
    if ($actualNorm -eq $expectedNorm) {
        return [pscustomobject]@{ Match = $true; Detail = "exact match" }
    }
    $actualLines = @()
    if ($actualNorm.Length -gt 0) {
        $actualLines = $actualNorm -split "`n"
    }
    $expectedLines = @()
    if ($expectedNorm.Length -gt 0) {
        $expectedLines = $expectedNorm -split "`n"
    }
    if ($actualLines.Count -ne $expectedLines.Count) {
        return [pscustomobject]@{ Match = $false; Detail = "line count mismatch" }
    }
    for ($index = 0; $index -lt $actualLines.Count; $index++) {
        $a = $actualLines[$index].Trim()
        $e = $expectedLines[$index].Trim()
        if ($a -eq $e) {
            continue
        }
        $av = 0.0
        $ev = 0.0
        if ((Try-ParseNumber $a ([ref]$av)) -and (Try-ParseNumber $e ([ref]$ev))) {
            if ([math]::Abs($av - $ev) -lt 1e-4) {
                continue
            }
        }
        return [pscustomobject]@{ Match = $false; Detail = "line $($index + 1) mismatch" }
    }
    return [pscustomobject]@{ Match = $true; Detail = "float tolerance" }
}

function Compile-LlToExe([string]$ClangPath, [string]$LlPath, [string]$ExePath, [int64]$StackSize) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $ClangPath
    $psi.WorkingDirectory = $RepoRoot
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    if (Test-Path $NativeParallelRuntime) {
        $psi.Arguments = ('"{0}" "{1}" "-Wl,/STACK:{2}" -o "{3}"' -f $LlPath, $NativeParallelRuntime, $StackSize, $ExePath)
    } else {
        $psi.Arguments = ('"{0}" "-Wl,/STACK:{1}" -o "{2}"' -f $LlPath, $StackSize, $ExePath)
    }
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    [void]$process.Start()
    $process.WaitForExit()
    $compileOutput = ($process.StandardOutput.ReadToEnd() + $process.StandardError.ReadToEnd()).Trim()
    $exitCode = $process.ExitCode
    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = $compileOutput.Trim()
    }
}

function Invoke-Exe([string]$ExePath, [int]$TimeoutMs = 120000) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $ExePath
    $psi.WorkingDirectory = (Split-Path -Parent $ExePath)
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    [void]$process.Start()
    if (-not $process.WaitForExit($TimeoutMs)) {
        try { $process.Kill() } catch {}
        $sw.Stop()
        return [pscustomobject]@{
            ExitCode = -999
            Output = ""
            Error = "TIMEOUT"
            Milliseconds = [double]$sw.Elapsed.TotalMilliseconds
        }
    }
    $sw.Stop()
    return [pscustomobject]@{
        ExitCode = $process.ExitCode
        Output = $process.StandardOutput.ReadToEnd()
        Error = $process.StandardError.ReadToEnd()
        Milliseconds = [double]$sw.Elapsed.TotalMilliseconds
    }
}

function Median([double[]]$Values) {
    if (-not $Values -or $Values.Count -eq 0) {
        return $null
    }
    $sorted = $Values | Sort-Object
    return [double]$sorted[[int]($sorted.Count / 2)]
}

function Speedup($Base, $Optimized) {
    if ($null -eq $Base -or $null -eq $Optimized -or $Optimized -le 0) {
        return $null
    }
    return [double]($Base / $Optimized)
}

function GeometricMean([double[]]$Values) {
    if (-not $Values -or $Values.Count -eq 0) {
        return $null
    }
    $sum = 0.0
    foreach ($value in $Values) {
        $sum += [math]::Log($value)
    }
    return [math]::Exp($sum / $Values.Count)
}

$clangPath = Resolve-ClangPath
$outDir = Join-Path $RepoRoot "outputs\clang_bench_exe"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$selected = $BenchFiles
if ($Filter) {
    $selected = $selected | Where-Object { $_ -like "*$Filter*" }
}

$results = @()

foreach ($bench in $selected) {
    $item = [ordered]@{
        file = "$bench.f"
        label = $BenchLabels[$bench]
        stack_bytes = $StackBytes
        levels = [ordered]@{}
    }
    $baselineOutput = $null
    foreach ($level in $Levels) {
        $llPath = Join-Path $RepoRoot ("outputs\" + $bench + "_" + $level + ".ll")
        if (-not (Test-Path $llPath)) {
            $item.levels[$level] = [ordered]@{
                compile_ok = $false
                run_ok = $false
                median_ms = $null
                output_ok = $false
                detail = "missing ll"
            }
            continue
        }
        $exePath = Join-Path $outDir ($bench + "_" + $level + ".exe")
        $compile = Compile-LlToExe $clangPath $llPath $exePath $StackBytes
        if ($compile.ExitCode -ne 0) {
            $item.levels[$level] = [ordered]@{
                compile_ok = $false
                run_ok = $false
                median_ms = $null
                output_ok = $false
                detail = $compile.Output
            }
            continue
        }
        $warmup = Invoke-Exe $exePath
        if ($warmup.ExitCode -ne 0) {
            $item.levels[$level] = [ordered]@{
                compile_ok = $true
                run_ok = $false
                median_ms = $null
                output_ok = $false
                detail = $warmup.Error.Trim()
            }
            continue
        }
        if ($null -eq $baselineOutput) {
            $baselineOutput = $warmup.Output
            $outputCheck = [pscustomobject]@{ Match = $true; Detail = "baseline" }
        } else {
            $outputCheck = Compare-Outputs $warmup.Output $baselineOutput
        }
        $samples = @()
        for ($i = 0; $i -lt $Repeat; $i++) {
            $run = Invoke-Exe $exePath
            if ($run.ExitCode -ne 0) {
                $item.levels[$level] = [ordered]@{
                    compile_ok = $true
                    run_ok = $false
                    median_ms = $null
                    output_ok = $false
                    detail = $run.Error.Trim()
                }
                $samples = $null
                break
            }
            $check = Compare-Outputs $run.Output $baselineOutput
            if (-not $check.Match) {
                $item.levels[$level] = [ordered]@{
                    compile_ok = $true
                    run_ok = $true
                    median_ms = $null
                    output_ok = $false
                    detail = $check.Detail
                }
                $samples = $null
                break
            }
            $samples += [double]$run.Milliseconds
        }
        if ($null -eq $samples) {
            continue
        }
        $item.levels[$level] = [ordered]@{
            compile_ok = $true
            run_ok = $true
            median_ms = (Median $samples)
            output_ok = $outputCheck.Match
            detail = $outputCheck.Detail
        }
    }
    $results += [pscustomobject]$item
}

$summary = [ordered]@{}
foreach ($level in $Levels) {
    if ($level -eq "O0") {
        continue
    }
    $values = @()
    foreach ($result in $results) {
        $o0 = $result.levels.O0
        $cur = $result.levels[$level]
        if ($null -ne $o0 -and $null -ne $cur -and $o0.run_ok -and $cur.run_ok) {
            $ratio = Speedup $o0.median_ms $cur.median_ms
            if ($null -ne $ratio) {
                $values += [double]$ratio
            }
        }
    }
    $summary[$level] = [ordered]@{
        geomean_speedup = (GeometricMean $values)
        count = $values.Count
    }
}

$jsonPath = Join-Path $RepoRoot $Json
$csvPath = [System.IO.Path]::ChangeExtension($jsonPath, ".csv")
$payload = [ordered]@{
    generated_at = (Get-Date).ToString("s")
    clang = $clangPath
    stack_bytes = $StackBytes
    repeat = $Repeat
    levels = $Levels
    results = $results
    summary = $summary
}
$payload | ConvertTo-Json -Depth 8 | Set-Content $jsonPath -Encoding utf8

$csvRows = @()
foreach ($result in $results) {
    $row = [ordered]@{
        file = $result.file
        label = $result.label
    }
    foreach ($level in $Levels) {
        $entry = $result.levels[$level]
        $row["${level}_ms"] = if ($null -ne $entry) { $entry.median_ms } else { $null }
        $row["${level}_ok"] = if ($null -ne $entry) { $entry.run_ok } else { $false }
    }
    if ($result.levels.O0.run_ok) {
        foreach ($level in $Levels) {
            if ($level -eq "O0") { continue }
            $entry = $result.levels[$level]
            $row["${level}_over_O0"] = if ($null -ne $entry -and $entry.run_ok) { Speedup $result.levels.O0.median_ms $entry.median_ms } else { $null }
        }
    }
    $csvRows += [pscustomobject]$row
}
$csvRows | Export-Csv -NoTypeInformation -Encoding utf8 $csvPath

Write-Output ""
Write-Output ("clang path: " + $clangPath)
Write-Output ("stack bytes: " + $StackBytes)
Write-Output ("repeat: " + $Repeat)
Write-Output ""
Write-Output ("{0,-28} {1,10} {2,10} {3,10} {4,10} {5,10}" -f "Benchmark", "O0", "O2", "O3", "O2/O0", "O3/O0")
Write-Output ("".PadRight(82, "-"))
foreach ($result in $results) {
    $o0 = $result.levels.O0
    $o2 = $result.levels.O2
    $o3 = $result.levels.O3
    $o2Speed = if ($o0.run_ok -and $o2.run_ok) { "{0:N2}x" -f (Speedup $o0.median_ms $o2.median_ms) } else { "-" }
    $o3Speed = if ($o0.run_ok -and $o3.run_ok) { "{0:N2}x" -f (Speedup $o0.median_ms $o3.median_ms) } else { "-" }
    $o0Text = if ($o0.run_ok) { "{0:N2}" -f $o0.median_ms } else { "ERR" }
    $o2Text = if ($o2.run_ok) { "{0:N2}" -f $o2.median_ms } else { "ERR" }
    $o3Text = if ($o3.run_ok) { "{0:N2}" -f $o3.median_ms } else { "ERR" }
    Write-Output ("{0,-28} {1,10} {2,10} {3,10} {4,10} {5,10}" -f $result.file, $o0Text, $o2Text, $o3Text, $o2Speed, $o3Speed)
}
Write-Output ("".PadRight(82, "-"))
foreach ($level in $Levels) {
    if ($level -eq "O0") { continue }
    $entry = $summary[$level]
    $geo = if ($null -ne $entry.geomean_speedup) { "{0:N2}x" -f $entry.geomean_speedup } else { "-" }
    Write-Output ("{0} geomean: {1} ({2} files)" -f $level, $geo, $entry.count)
}
Write-Output ("JSON: " + $jsonPath)
Write-Output ("CSV: " + $csvPath)
