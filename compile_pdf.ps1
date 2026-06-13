$baseDir = "C:\Users\paula\Documents\mgr pjatk"
$tectonicUrl = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-pc-windows-msvc.zip"
$zipPath = Join-Path $baseDir "tectonic.zip"
$extractPath = Join-Path $baseDir "tectonic_bin"

If (-not (Test-Path $extractPath)) {
    Write-Output "Downloading tectonic..."
    Invoke-WebRequest -Uri $tectonicUrl -OutFile $zipPath
    Write-Output "Extracting tectonic..."
    Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
    Remove-Item $zipPath
}

$tectonicExe = Join-Path $extractPath "tectonic.exe"
$latexDir = Join-Path $baseDir "THESIS_LATEX_FINAL"

# Copy reports/figures into the LaTeX directory so tectonic can find them if referenced
$reportsDest = Join-Path $latexDir "reports\figures"
If (-not (Test-Path $reportsDest)) {
    New-Item -ItemType Directory -Path $reportsDest -Force | Out-Null
}
Copy-Item -Path (Join-Path $baseDir "reports\figures\*") -Destination $reportsDest -Recurse -Force

Write-Output "Compiling PDF with tectonic..."
Set-Location $latexDir
& $tectonicExe "glowny.tex"

Write-Output "Compilation finished."
