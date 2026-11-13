@echo off
REM Double-clickable Windows quick-start launcher
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_windows.ps1"
pause
