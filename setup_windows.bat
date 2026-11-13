@echo off
REM Double-clickable Windows launcher that runs setup_windows.ps1 bypassing PowerShell execution policy prompts
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
pause
