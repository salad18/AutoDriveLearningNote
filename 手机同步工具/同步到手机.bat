@echo off
chcp 65001 >nul
title 同步知识库到手机
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync-to-phone.ps1"
