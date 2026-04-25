@echo off
chcp 65001 >nul
setlocal

echo =========================================
echo 自动拆分：同步 labels_all -> labels(train/val) 并检查
echo =========================================
echo.

cd /d "%~dp0.." || (echo 无法进入项目目录 & pause & exit /b 1)

REM 激活环境（与 01_标定助手_一键开始.bat 保持一致）
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

set "REC_CLASSES="
set "REC_TMP=%TEMP%\pyocr_rec_classes.txt"
"%CONDA_ENV_DIR%\python.exe" -c "from pathlib import Path; p=Path(r\"%VOCAB%\"); print(sum(1 for ln in p.read_text(encoding=\"utf-8\").splitlines() if ln.strip()))" > "%REC_TMP%" 2>nul
if exist "%REC_TMP%" (
  set /p REC_CLASSES=<"%REC_TMP%"
  del /f /q "%REC_TMP%" >nul 2>&1
)
if "%REC_CLASSES%"=="" set "REC_CLASSES=12"

echo [1/3] 同步 det labels...
pyocr sync-split-labels --images-root datasets\det\images --labels-src datasets\det\labels_all --labels-root datasets\det\labels
if errorlevel 1 goto :fail

echo.
echo [2/3] 同步 rec labels...
pyocr sync-split-labels --images-root datasets\rec\images --labels-src datasets\rec\labels_all --labels-root datasets\rec\labels
if errorlevel 1 goto :fail

echo.
echo [3/3] 检查标注完整性（det=1类，rec=%REC_CLASSES%类）...
pyocr check-obb-labels --images datasets\det\images\train --labels datasets\det\labels\train --num-classes 1
if errorlevel 1 goto :fail
pyocr check-obb-labels --images datasets\det\images\val   --labels datasets\det\labels\val   --num-classes 1
if errorlevel 1 goto :fail
pyocr check-obb-labels --images datasets\rec\images\train --labels datasets\rec\labels\train --num-classes %REC_CLASSES%
if errorlevel 1 (
  echo.
  echo 注意：rec/train 仍存在缺失 labels。通常是因为 labels_all 里还没标完这么多张图。
  echo       这不是格式错误，但训练前需要补齐。
  goto :rec_missing
)
pyocr check-obb-labels --images datasets\rec\images\val   --labels datasets\rec\labels\val   --num-classes %REC_CLASSES%
if errorlevel 1 (
  echo.
  echo 注意：rec/val 仍存在缺失 labels。通常是因为 labels_all 里还没标完这么多张图。
  echo       这不是格式错误，但训练前需要补齐。
  goto :rec_missing
)

echo.
echo 完成：同步与检查均通过。你可以开始训练。
echo.
echo 下一步建议（训练命令）：
echo   pyocr train-det --data datasets\det\det.yaml --name det_exp --device cpu --epochs 50 --imgsz 640 --batch 8
echo   pyocr train-rec --data datasets\rec\rec.yaml --name rec_exp --device cpu --epochs 50 --imgsz 640 --batch 8
echo.
echo 训练完成后可进行推理测试（示例）：
echo   pyocr infer --det-weights models\det_exp\weights\best.pt --rec-weights models\rec_exp\weights\best.pt --image Images\trains\1_7_131830248.bmp --vocab %VOCAB%
pause
exit /b 0

:rec_missing
echo.
echo ================ 建议下一步（补齐 rec 标注） ================
echo 1) 继续使用标定助手对 Images\trains 进行标注，输出到 datasets\rec\labels_all。
echo 2) 标完一批后，重新运行本脚本同步并复查。
echo.
echo 可选：生成缺失清单（PowerShell，复制到终端运行）：
echo   powershell -NoProfile -ExecutionPolicy Bypass -Command "^$imgs=Get-ChildItem datasets\rec\images\train -Filter *.bmp; ^$miss=@(); foreach(^$i in ^$imgs){ if(-not (Test-Path (Join-Path 'datasets\\rec\\labels\\train' (^$i.BaseName+'.txt')))){ ^$miss+=^$i.Name } }; ^$miss | Set-Content datasets\\rec\\missing_train.txt -Encoding utf8; 'missing_train='+^$miss.Count"
echo.
echo 生成后查看：datasets\rec\missing_train.txt
pause
exit /b 1

:fail
echo.
echo 失败：同步或检查未通过。请根据上方输出定位问题（缺失/空标注/类别越界/坐标越界）。
pause
exit /b 1

