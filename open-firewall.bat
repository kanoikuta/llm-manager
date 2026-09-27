@echo off
chcp 65001 >nul
title 放行防火墙：管理器端口（需管理员）
echo ============================================
echo   给手机放行管理器端口
echo     TCP 80     控制台（/ 是网页，/v1 是 API）
echo     TCP 8091   OpenAI 兼容入口（旧客户端用）
echo.
echo   规则一律写成 profile=any（所有网络配置文件）。
echo   只写 Public 会踩这个坑：Windows 把家里 WiFi 判成
echo   「专用 Private」时，那些 Public 规则一条都不生效。
echo ============================================
echo.

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在请求管理员权限，请在 UAC 弹窗上点「是」...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    echo.
    echo 如果 UAC 窗口已经弹出，点「是」继续。
    pause
    exit /b
)

echo 已提权，开始写规则...
echo.

netsh advfirewall firewall delete rule name="llm-manager-80-lan" >nul 2>&1
netsh advfirewall firewall add rule name="llm-manager-80-lan" dir=in action=allow protocol=TCP localport=80 profile=any
netsh advfirewall firewall delete rule name="llm-manager-8091-lan" >nul 2>&1
netsh advfirewall firewall add rule name="llm-manager-8091-lan" dir=in action=allow protocol=TCP localport=8091 profile=any

echo.
echo 当前这台电脑在局域网里的地址（手机上要连的就是它）：
powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '169.254.*' -and $_.IPAddress -ne '127.0.0.1' } | Select-Object InterfaceAlias,IPAddress | Format-Table -AutoSize"

echo ============================================
echo   完成。
echo ============================================
pause
