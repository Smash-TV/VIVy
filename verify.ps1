$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

& $python (Join-Path $projectRoot "scripts\verify_snapshot.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python -m pytest --version *> $null
if ($LASTEXITCODE -eq 0) {
    & $python -m pytest -q
    exit $LASTEXITCODE
}

Write-Host "公开副本哈希与核心导入验证已通过；当前 Python 未安装 pytest，跳过测试套件。"
exit 0

