@echo off
chcp 65001 >nul
setlocal

echo =========================================
echo PythonOcr_Train_Gpu - 一键安装环境（Windows）
echo =========================================
echo.
echo [1/3] 进入项目目录...
cd /d "%~dp0.." || (echo 无法进入项目目录 & pause & exit /b 1)

echo [2/3] 激活 Anaconda 环境（优先：PythonOcr_Train_Gpu）...
set "CONDA_ACT=D:\Users\admin\anaconda3\Scripts\activate.bat"
set "CONDA_ENV=PythonOcr_Train_Gpu"
set "CONDA_ENV_DIR=D:\Users\admin\anaconda3\envs\PythonOcr_Train_Gpu"

if not exist "%CONDA_ACT%" (
  echo 错误：未找到 conda 激活脚本：%CONDA_ACT%
  echo 请确认 Anaconda 安装目录是否为 D:\Users\admin\anaconda3
  pause
  exit /b 1
)

if not exist "%CONDA_ENV_DIR%" (
  echo 错误：未找到 conda 环境目录：%CONDA_ENV_DIR%
  echo 请先创建环境：conda create -n %CONDA_ENV% python=3.10
  pause
  exit /b 1
)

call "%CONDA_ACT%" %CONDA_ENV%
if errorlevel 1 (
  echo 错误：激活 conda 环境失败：%CONDA_ENV%
  pause
  exit /b 1
)

echo [3/3] 安装依赖（GPU）...
python -m pip install -U pip
python -m pip install -r requirements_gpu.txt
python -m pip install -e . --no-deps

echo.
echo 完成！
echo 下一步：双击 tools\01_标定助手_一键开始.bat
echo.
pause
