@echo off
chcp 65001 >nul
setlocal

echo =========================================
echo 统计某个类别对应的图片名称
echo =========================================
echo.

cd /d "%~dp0.." || (echo 无法进入项目目录 & pause & exit /b 1)

REM 激活环境
set "CONDA_ACT=D:\Users\admin\anaconda3\Scripts\activate.bat"
set "CONDA_ENV=PythonOcr_Train_Gpu"
set "CONDA_ENV_DIR=D:\Users\admin\anaconda3\envs\PythonOcr_Train_Gpu"

if not exist "%CONDA_ACT%" (
  echo 错误：未找到 conda 激活脚本：%CONDA_ACT%
  pause
  exit /b 1
)

if not exist "%CONDA_ENV_DIR%" (
  echo 错误：未找到 conda 环境目录：%CONDA_ENV_DIR%
  echo 请先运行 tools\00_一键安装环境.bat 或手动创建环境。
  pause
  exit /b 1
)

call "%CONDA_ACT%" %CONDA_ENV%
if errorlevel 1 (
  echo 错误：激活 conda 环境失败：%CONDA_ENV%
  pause
  exit /b 1
)

set /p TARGET=请输入类别ID或字符(例如 13 或 G):
if "%TARGET%"=="" (
  echo 未输入类别。
  pause
  exit /b 1
)

set "OUT=datasets\rec\class_images.txt"
"%CONDA_ENV_DIR%\python.exe" tools\rec_class_images.py --target "%TARGET%" --vocab vocab_rec.txt --labels datasets\rec\labels_all --out "%OUT%"

if errorlevel 1 (
  echo.
  echo 失败：统计过程中出错。
  pause
  exit /b 1
)

echo.
echo 已输出：%OUT%
if exist "%OUT%" (
  echo 内容预览：
  type "%OUT%"
)

pause
exit /b 0

