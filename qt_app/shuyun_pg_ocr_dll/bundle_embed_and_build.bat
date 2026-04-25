@echo off
setlocal

chcp 65001 >nul

REM 一键打包：嵌入式 Python + 依赖 + DLL
cd /d "%~dp0"

set "ROOT=%~dp0"
set "PROJECT_ROOT=%ROOT%..\.."
set "EMBED_DIR=%PROJECT_ROOT%\csharp_ocr\pyembed"
set "APP_DIR=%EMBED_DIR%\app"
set "LOG=%ROOT%bundle_embed_and_build.log"
set "PY_VER=3.10.11"
set "PY_ZIP=python-%PY_VER%-embed-amd64.zip"
set "PY_URL=https://www.python.org/ftp/python/%PY_VER%/%PY_ZIP%"

if not exist "%ROOT%shuyun_pg_ocr_dll.csproj" (
  echo 未找到 shuyun_pg_ocr_dll.csproj
  echo 未找到 shuyun_pg_ocr_dll.csproj > "%LOG%"
  pause
  exit /b 1
)

echo 开始打包... > "%LOG%"

dotnet restore "%ROOT%shuyun_pg_ocr_dll.csproj" --source https://api.nuget.org/v3/index.json >> "%LOG%" 2>&1
if errorlevel 1 (
  echo 还原 NuGet 失败
  echo 还原 NuGet 失败 >> "%LOG%"
  pause
  exit /b 1
)

REM 准备嵌入式 Python
if exist "%EMBED_DIR%" (
  echo 清理旧的嵌入式 Python 目录... >> "%LOG%"
  rmdir /s /q "%EMBED_DIR%"
)

mkdir "%EMBED_DIR%" >> "%LOG%" 2>&1

powershell -NoProfile -Command "Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%EMBED_DIR%\%PY_ZIP%'" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo 下载嵌入式 Python 失败
  echo 下载嵌入式 Python 失败 >> "%LOG%"
  pause
  exit /b 1
)

powershell -NoProfile -Command "Expand-Archive -Path '%EMBED_DIR%\%PY_ZIP%' -DestinationPath '%EMBED_DIR%' -Force" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo 解压嵌入式 Python 失败
  echo 解压嵌入式 Python 失败 >> "%LOG%"
  pause
  exit /b 1
)

del /f /q "%EMBED_DIR%\%PY_ZIP%" >> "%LOG%" 2>&1

REM 启用 site-packages
powershell -NoProfile -Command "$pth=Join-Path '%EMBED_DIR%' 'python310._pth'; $lines=Get-Content $pth; $out=@(); foreach($l in $lines){ if($l -match '^#import site'){ $out += 'import site' } else { $out += $l } }; if(-not ($out -contains 'Lib\\site-packages')){ $out = @('Lib\\site-packages') + $out }; Set-Content -Path $pth -Value $out" >> "%LOG%" 2>&1

REM 安装 pip
powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%EMBED_DIR%\get-pip.py'" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo 下载 get-pip.py 失败
  echo 下载 get-pip.py 失败 >> "%LOG%"
  pause
  exit /b 1
)

"%EMBED_DIR%\python.exe" "%EMBED_DIR%\get-pip.py" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo 安装 pip 失败
  echo 安装 pip 失败 >> "%LOG%"
  pause
  exit /b 1
)

del /f /q "%EMBED_DIR%\get-pip.py" >> "%LOG%" 2>&1

REM 安装 PyTorch（CUDA 12.1）
"%EMBED_DIR%\python.exe" -m pip install --upgrade pip >> "%LOG%" 2>&1
"%EMBED_DIR%\python.exe" -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121 >> "%LOG%" 2>&1
if errorlevel 1 (
  echo 安装 PyTorch 失败
  echo 安装 PyTorch 失败 >> "%LOG%"
  pause
  exit /b 1
)

REM 安装项目依赖
"%EMBED_DIR%\python.exe" -m pip install -r "%ROOT%requirements_embed.txt" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo 安装依赖失败
  echo 安装依赖失败 >> "%LOG%"
  pause
  exit /b 1
)

REM 复制 Python 代码
mkdir "%APP_DIR%" >> "%LOG%" 2>&1
robocopy "%PROJECT_ROOT%\qt_app" "%APP_DIR%\qt_app" /E /NFL /NDL /NJH /NJS /XD ".vs" >> "%LOG%" 2>&1
robocopy "%PROJECT_ROOT%\src" "%APP_DIR%\src" /E /NFL /NDL /NJH /NJS /XD ".vs" >> "%LOG%" 2>&1
robocopy "%ROOT%" "%APP_DIR%\qt_app\shuyun_pg_ocr_dll" shuyun_pg_ocr.py det.pt rec.pt vocab_rec.txt /NFL /NDL /NJH /NJS >> "%LOG%" 2>&1

REM 构建 DLL

dotnet build "%ROOT%shuyun_pg_ocr_dll.csproj" -c Release >> "%LOG%" 2>&1
if errorlevel 1 (
  echo 构建 DLL 失败
  echo 构建 DLL 失败 >> "%LOG%"
  pause
  exit /b 1
)

echo 打包完成。嵌入式 Python: %EMBED_DIR% >> "%LOG%"
echo 打包完成。嵌入式 Python: %EMBED_DIR%

pause
endlocal
