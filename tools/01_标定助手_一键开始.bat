@echo off
chcp 65001 >nul
setlocal

echo =========================================
echo 标定助手（推荐：全量标注 -> 自动拆分 train/val）
echo =========================================
echo.
cd /d "%~dp0.." || (echo 无法进入项目目录 & pause & exit /b 1)

REM 激活环境（固定使用 Anaconda 环境：D:\Users\admin\anaconda3\envs\PythonOcr_Train_Gpu）
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

echo.
echo [1/4] 生成训练/验证集目录结构（仅复制图片，不会自动生成标注）...
pyocr prepare-det --src Images\trains --out datasets\det --val-ratio 0.2
pyocr prepare-rec --src Images\trains --out datasets\rec --val-ratio 0.2

REM 标定助手输入/输出目录
set "IMAGES_DIR=Images\trains"
set "DET_LABELS=datasets\det\labels_all"
set "REC_LABELS=datasets\rec\labels_all"

set "VOCAB=vocab_rec.txt"
if not exist "%VOCAB%" (
  echo.
  echo 未找到 %VOCAB%
  echo 将尝试兼容旧文件 vocab_0-9_A-Z.txt。
  set "VOCAB=vocab_0-9_A-Z.txt"
)

if not exist "%VOCAB%" (
  echo.
  echo 未找到 vocab 文件。
  echo 请在项目根目录准备 vocab 文件（UTF-8 每行一个字符）。
  pause
  exit /b 1
)

set "REC_CLASSES="
for /f %%i in ('"%CONDA_ENV_DIR%\python.exe" -c "from pathlib import Path; p=Path(r\"%VOCAB%\"); print(sum(1 for ln in p.read_text(encoding=\"utf-8\").splitlines() if ln.strip()))"') do set "REC_CLASSES=%%i"
if "%REC_CLASSES%"=="" set "REC_CLASSES=12"

echo.
echo([2/4] 启动标定助手（对全量 Images\trains 标注，输出到 labels_all 池）...
echo(说明：
echo( - 先画定位框（可旋转/缩放/移动）
echo( - TAB 切换选中框
echo( - E 给当前框输入“识别内容”(可多个字符，例如 AB12)
echo( - 若输入了新字符（词表中不存在），会自动追加为新类别
echo( - I 给当前框输入序号 index（用于排序）
echo( - n 下一张会自动保存 det\labels 和 rec\labels
echo.
echo(注：本模式会对 Images\trains 全量标注，不需要手动再标 val。
echo.

set "LOG_DIR=logs"
set "LOG_FILE=%LOG_DIR%\label_assistant_%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%.log"
set "LOG_FILE=%LOG_FILE: =0%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM 打印环境信息
where python >> "%LOG_FILE%" 2>&1
python -c "import sys; print(sys.executable); print(sys.version)" >> "%LOG_FILE%" 2>&1

REM 启动标定助手（输出写入日志）
pyocr label-assistant --images "%IMAGES_DIR%" --det-labels "%DET_LABELS%" --rec-labels "%REC_LABELS%" --vocab "%VOCAB%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo 标定助手启动失败，请查看日志：%LOG_FILE%
  type "%LOG_FILE%"
  pause
  exit /b 1
)

REM 正常退出也保留日志
pause

echo.
echo [3/4] 按 images/train|val 自动同步 det labels...
pyocr sync-split-labels --images-root datasets\det\images --labels-src datasets\det\labels_all --labels-root datasets\det\labels

echo.
echo [4/4] 按 images/train|val 自动同步 rec labels...
pyocr sync-split-labels --images-root datasets\rec\images --labels-src datasets\rec\labels_all --labels-root datasets\rec\labels

echo.
echo 已完成全量标注与拆分同步。建议运行：
echo   pyocr check-obb-labels --images datasets\det\images\train --labels datasets\det\labels\train --num-classes 1
echo   pyocr check-obb-labels --images datasets\det\images\val   --labels datasets\det\labels\val   --num-classes 1
echo   pyocr check-obb-labels --images datasets\rec\images\train --labels datasets\rec\labels\train --num-classes %REC_CLASSES%
echo   pyocr check-obb-labels --images datasets\rec\images\val   --labels datasets\rec\labels\val   --num-classes %REC_CLASSES%
echo 或者直接开始训练。
pause
