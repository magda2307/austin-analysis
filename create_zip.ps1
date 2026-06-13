$baseDir = "C:\Users\paula\Documents\mgr pjatk"
$tempDir = Join-Path $baseDir "latex-mgr-temp"
If (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
New-Item -ItemType Directory -Path $tempDir | Out-Null

Copy-Item -Path (Join-Path $baseDir "mgr_latex\szablon-pracy-dyplomowej-pjatk-master\*") -Destination $tempDir -Recurse

$reportsTemp = Join-Path $tempDir "reports"
New-Item -ItemType Directory -Path $reportsTemp | Out-Null
Copy-Item -Path (Join-Path $baseDir "reports\figures") -Destination $reportsTemp -Recurse

$zipPath = Join-Path $baseDir "latex-mgr-export.zip"
If (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path (Join-Path $tempDir "*") -DestinationPath $zipPath

Remove-Item -Recurse -Force $tempDir
Write-Output "Zip created successfully at $zipPath"
