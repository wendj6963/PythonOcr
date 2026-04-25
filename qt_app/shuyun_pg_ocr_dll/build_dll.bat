@echo off
setlocal

chcp 65001 >nul

REM 一键打包 shuyun_pg_ocr_dll.dll（需要安装 dotnet SDK）
cd /d "%~dp0"

set "LOG=%~dp0build_dll.log"

if not exist "shuyun_pg_ocr_dll.csproj" (
  echo 未找到 shuyun_pg_ocr_dll.csproj
  echo 未找到 shuyun_pg_ocr_dll.csproj > "%LOG%"
  pause
  exit /b 1
)

echo 开始构建... > "%LOG%"
dotnet build shuyun_pg_ocr_dll.csproj -c Release >> "%LOG%" 2>&1
if errorlevel 1 (
  echo 构建失败
  echo 构建失败，请查看日志：%LOG%
  pause
  exit /b 1
)

echo 构建完成: bin\Release\net8.0\shuyun_pg_ocr_dll.dll
 echo 构建完成: bin\Release\net8.0\shuyun_pg_ocr_dll.dll >> "%LOG%"
pause
endlocal
