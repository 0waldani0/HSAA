# Launcher for Google Colab Local Runtime
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "Google Colab - Local Runtime Server"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Запуск локального сервера для Google Colab (Local Runtime)" -ForegroundColor Cyan
Write-Host "  Рабочая директория: $(Resolve-Path "$PSScriptRoot\..")" -ForegroundColor Gray
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Инструкция для подключения:" -ForegroundColor Yellow
Write-Host "1. Дождитесь запуска сервера (появится URL вида http://127.0.0.1:8888/tree?token=...)" -ForegroundColor White
Write-Host "2. Скопируйте ссылку с токеном из консоли." -ForegroundColor White
Write-Host "3. В Google Colab нажмите: [Подключиться] -> [Подключиться к локальной среде]." -ForegroundColor White
Write-Host "4. Вставьте скопированный URL и нажмите [Подключиться]." -ForegroundColor White
Write-Host ""
Write-Host "Для остановки нажмите Ctrl+C." -ForegroundColor DarkGray
Write-Host "----------------------------------------------------------------" -ForegroundColor DarkGray

$rootDir = (Resolve-Path "$PSScriptRoot\..").Path

& python -m jupyter notebook `
  --ServerApp.allow_origin="https://colab.research.google.com" `
  --NotebookApp.allow_origin="https://colab.research.google.com" `
  --ServerApp.port_retries=0 `
  --NotebookApp.port_retries=0 `
  --ServerApp.root_dir="$rootDir" `
  --port=8888 `
  --no-browser
