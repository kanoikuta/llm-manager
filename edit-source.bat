@echo off
chcp 65001 >nul
title 查看源码
cd /d "%~dp0"

echo ============================================
echo   用记事本打开源码
echo ============================================
echo.
echo   model_manager.py   主程序（纯标准库，零依赖）
echo   ui.html            网页界面（HTML+CSS+JS）
echo.
echo   如果 .py 双击没反应，那是 Windows 没给 .py
echo   关联打开程序，不是文件坏了 —— 用这个 bat 就行。
echo ============================================
echo.

start "" notepad "%~dp0model_manager.py"
start "" notepad "%~dp0ui.html"

echo 已经交给记事本打开了。这个窗口可以关掉。
ping -n 4 127.0.0.1 >nul
