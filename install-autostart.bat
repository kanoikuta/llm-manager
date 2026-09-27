@echo off
chcp 65001 >nul
title 安装开机自启
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

rem 解释器：优先用 PYTHON 环境变量，其次 PATH 里的 python，最后让 uv 找
set PY=%PYTHON%
if not defined PY for /f "delims=" %%i in ('python -c "import sys;print(sys.executable)" 2^>nul') do set PY=%%i
if not defined PY for /f "delims=" %%i in ('uv python find 2^>nul') do set PY=%%i
if not defined PY (
  echo [x] 找不到 Python 解释器。
  echo     装一个 Python 3.10+ 并加进 PATH，或设 PYTHON 环境变量指向 python.exe。
  pause
  exit /b 1
)

"%PY%" "%~dp0autostart.py" install
pause
