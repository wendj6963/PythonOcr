@echo off
chcp 65001 >nul
setlocal

echo =========================================
echo 输出 rec 类别越界清单（train/val）
echo =========================================
echo.

cd /d "%~dp0.." || (echo 无法进入项目目录 & pause & exit /b 1)

REM 激活环境（与 01/02/03/04 脚本保持一致）
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

set "VOCAB=vocab_rec.txt"
if not exist "%VOCAB%" set "VOCAB=vocab_0-9_A-Z.txt"
if not exist "%VOCAB%" (
  echo 错误：未找到词表文件（vocab_rec.txt 或 vocab_0-9_A-Z.txt）。
  pause
  exit /b 1
)

set "OUT=datasets\rec\out_of_range.txt"
"%CONDA_ENV_DIR%\python.exe" tools\rec_out_of_range.py --vocab "%VOCAB%" --labels-root datasets\rec\labels --out "%OUT%"

if errorlevel 1 (
  echo.
  echo 失败：生成越界清单时出错。
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

