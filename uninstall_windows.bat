@echo off
REM Double-clickable Windows uninstaller that runs uninstall_windows.ps1
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall_windows.ps1"
pause
