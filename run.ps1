param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Text
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }
$message = $Text -join " "

if ([string]::IsNullOrWhiteSpace($message)) {
    Write-Error "请在命令后提供一段输入文本。"
    exit 2
}

& $python (Join-Path $projectRoot "scripts\vivy_cli.py") run $message
exit $LASTEXITCODE

