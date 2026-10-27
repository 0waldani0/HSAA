@echo off
chcp 65001 >nul
title Google Colab - Local Runtime Server
echo ================================================================
echo   Запуск локального сервера для Google Colab (Local Runtime)
echo   GPU: NVIDIA RTX 3050 Laptop / CPU: Intel Core i7-11800H
echo ================================================================
echo.

set PYTHON_EXE=C:\Users\Администратор\AppData\Local\Programs\Python\Python312\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

echo Запуск Jupyter Server на порту 8888 с разрешением CORS для Google Colab...
echo.
echo Скопируйте URL с токеном из вывода ниже (например: http://localhost:8888/?token=...)
echo и вставьте его в Google Colab:
echo   [Подключиться] (вверху справа) -> [Подключиться к локальной среде]
echo.
echo Для остановки сервера нажмите Ctrl+C.
echo ----------------------------------------------------------------

"%PYTHON_EXE%" -m jupyter notebook ^
  --ServerApp.allow_origin="https://colab.research.google.com" ^
  --NotebookApp.allow_origin="https://colab.research.google.com" ^
  --ServerApp.port_retries=0 ^
  --NotebookApp.port_retries=0 ^
  --ServerApp.root_dir="%~dp0.." ^
  --port=8888 ^
  --no-browser

pause
