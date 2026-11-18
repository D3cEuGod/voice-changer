param([string]$Sentences, [string]$OutDir)
# Synthesize each line with the Windows OneCore zh-CN voices (Huihui = female, Kangkang = male) into WAV files.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($operation, [Type]$resultType) {
    $task = $asTaskGeneric.MakeGenericMethod($resultType).Invoke($null, @($operation))
    $task.Wait(-1) | Out-Null
    $task.Result
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$lines = Get-Content -LiteralPath $Sentences -Encoding UTF8 | Where-Object { $_.Trim() }
$voices = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices
foreach ($pair in @(@('f', 'Huihui'), @('m', 'Kangkang'))) {
    $voice = $voices | Where-Object { $_.DisplayName -like "*$($pair[1])*" } | Select-Object -First 1
    if (-not $voice) { "voice not found: $($pair[1])"; continue }
    $synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
    $synth.Voice = $voice
    $index = 0
    foreach ($line in $lines) {
        $index++
        $stream = Await ($synth.SynthesizeTextToStreamAsync($line)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
        $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
        $null = Await ($reader.LoadAsync([uint32]$stream.Size)) ([uint32])
        $bytes = New-Object byte[] ([int]$stream.Size)
        $reader.ReadBytes($bytes)
        $path = Join-Path $OutDir ("{0}{1}.wav" -f $pair[0], $index)
        [System.IO.File]::WriteAllBytes($path, $bytes)
        "{0}  {1} bytes  {2}" -f (Split-Path $path -Leaf), $bytes.Length, $voice.DisplayName
    }
}
