param([Parameter(Mandatory)][string]$Tag, [string]$Settings = '', [string[]]$Voices = @(),
      [ValidateSet('character', 'call')][string]$Mode = 'character')
# Render one configuration and score it. Example:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\eval\run_config.ps1 -Tag H -Settings 'character.STABLE_TONE=True'
$root = Split-Path (Split-Path $PSScriptRoot)
Set-Location -LiteralPath $root
$env:PYTHONUTF8 = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:EVAL_SET = $Settings
$env:EVAL_MODE = $Mode
Remove-Item Env:EVAL_VARIANT -ErrorAction SilentlyContinue
& '.\.venv\Scripts\python.exe' 'tools\eval\eval_render.py' $Tag @Voices 2>&1 | ForEach-Object { "$_" } | Select-String 'SUMMARY|Error|Traceback|^set ' | ForEach-Object { $_.Line }
Remove-Item Env:EVAL_SET -ErrorAction SilentlyContinue
Remove-Item Env:EVAL_MODE -ErrorAction SilentlyContinue
& '.\.run\export-env\Scripts\python.exe' 'tools\eval\eval_asr.py' $Tag 2>&1 | ForEach-Object { "$_" } | Select-String 'CER|Error|Traceback' | ForEach-Object { $_.Line }
