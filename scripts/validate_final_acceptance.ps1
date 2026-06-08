param(
    [switch]$Long,
    [switch]$SkipPytest,
    [switch]$SkipScripts,
    [int]$RecencyBootstraps = 5,
    [int]$RecencyIterations = 20,
    [int]$BacktestBootstraps = 5
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Command
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    Write-Host ("    " + ($Command -join " "))

    & $Command[0] @($Command[1..($Command.Count - 1)])
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit code $LASTEXITCODE)"
    }
}

Invoke-Step "Python compile dashboard helpers" @(
    "python", "-m", "py_compile",
    "src/aac_adoption/dashboard/data.py",
    "streamlit_app.py"
)

if (-not $SkipScripts) {
    Invoke-Step "Calibration CLI help" @("python", "scripts/calibrate_classifiers.py", "--help")
    Invoke-Step "Backtesting CLI help" @("python", "scripts/evaluate_backtesting.py", "--help")
    Invoke-Step "Recency CLI help" @("python", "scripts/compare_recency.py", "--help")
}

if (-not $SkipPytest) {
    Invoke-Step "Dataset contract tests" @("python", "-m", "pytest", "tests/test_build_dataset.py", "-m", "not slow and not acceptance", "-q")
    Invoke-Step "Dashboard tests" @("python", "-m", "pytest", "tests/test_dashboard_data.py", "-m", "not slow and not acceptance", "-q")
    Invoke-Step "Temporal validation tests" @(
        "python", "-m", "pytest",
        "tests/test_backtesting.py",
        "tests/test_yearly_backtesting.py",
        "tests/test_recency_comparison.py",
        "-m", "not slow and not acceptance",
        "-q"
    )
    Invoke-Step "Method hardening tests" @(
        "python", "-m", "pytest",
        "tests/test_hyperparam_tuning.py",
        "tests/test_ensemble.py",
        "tests/test_diagnostics_outputs.py",
        "-m", "not slow and not acceptance",
        "-q"
    )
    Invoke-Step "Terminology/report tests" @(
        "python", "-m", "pytest",
        "tests/test_target_definitions.py",
        "tests/test_report_outputs.py",
        "-m", "not slow and not acceptance",
        "-q"
    )
}

if (-not $SkipScripts) {
    $TempRoot = Join-Path $env:TEMP "aac_smoke_$([guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $TempRoot | Out-Null
    Write-Host "Created temporary smoke root: $TempRoot" -ForegroundColor Gray

    Invoke-Step "Quick yearly backtesting" @(
        "python", "scripts/evaluate_backtesting.py",
        "--quick",
        "--n_bootstraps", "$BacktestBootstraps",
        "--output", "$TempRoot/backtesting.csv"
    )
    Invoke-Step "Quick recency comparison" @(
        "python", "scripts/compare_recency.py",
        "--quick",
        "--n-bootstraps", "$RecencyBootstraps",
        "--iterations", "$RecencyIterations",
        "--output", "$TempRoot/recency.csv",
        "--figure-output", "$TempRoot/recency.png"
    )
}

if ($Long) {
    $env:AAC_ACCEPTANCE = "1"
    Write-Host "Running in canonical acceptance mode (AAC_ACCEPTANCE=1)" -ForegroundColor Gray
    
    Invoke-Step "Full pytest suite" @("python", "-m", "pytest", "-q")
}

Write-Host ""
Write-Host "All requested validation steps passed." -ForegroundColor Green
