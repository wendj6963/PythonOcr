@echo off
chcp 65001 >nul
setlocal

set "KMP_DUPLICATE_LIB_OK=TRUE"

echo =========================================
echo 一键训练 det+rec 并推理抽查（日志+结果落盘）
echo =========================================
echo.

cd /d "%~dp0.." || (echo 无法进入项目目录 & pause & exit /b 1)

REM 激活环境（与 01/02 脚本保持一致）
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

REM 允许运行参数覆盖：
REM 用法示例：03_一键训练_det_rec并推理抽查.bat 100 640 8 cpu det_exp rec_exp
set "EPOCHS=150"
set "IMGSZ=640"
set "BATCH=16"
set "DEVICE=0"
set "DET_NAME=det_exp"
set "REC_NAME=rec_exp"

if not "%~1"=="" set "EPOCHS=%~1"
if not "%~2"=="" set "IMGSZ=%~2"
if not "%~3"=="" set "BATCH=%~3"
if not "%~4"=="" set "DEVICE=%~4"
REM 兼容用户输入 gpu/GPU
if /I "%DEVICE%"=="gpu" set "DEVICE=0"

set "VOCAB=vocab_rec.txt"
if not exist "%VOCAB%" set "VOCAB=vocab_0-9_A-Z.txt"
if not exist "%VOCAB%" (
  echo 错误：未找到词表文件（vocab_rec.txt 或 vocab_0-9_A-Z.txt）。
  pause
  exit /b 1
)

echo 参数：epochs=%EPOCHS% imgsz=%IMGSZ% batch=%BATCH% device=%DEVICE% det_name=%DET_NAME% rec_name=%REC_NAME%

REM device auto-detect already handled by torch.cuda

REM batch 自适应：GPU 默认更大一点；CPU 保持小批次
if "%~3"=="" (
  if "%DEVICE%"=="cpu" (
    set "BATCH=8"
  ) else (
    set "BATCH=16"
  )
)

REM 输出目录（时间戳）
for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set "D=%%a-%%b-%%c"
for /f "tokens=1-3 delims=:., " %%a in ("%time%") do set "T=%%a%%b%%c"
set "RUN_DIR=models\runs\%D%_%T%"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

echo.
echo [1/4] 训练 det...
pyocr train-det --data datasets\det\det.yaml --name "%DET_NAME%" --device "%DEVICE%" --epochs %EPOCHS% --imgsz %IMGSZ% --batch %BATCH% --project models > "%RUN_DIR%\train_det.log" 2>&1
if errorlevel 1 (
  echo det 训练失败，详见：%RUN_DIR%\train_det.log
  pause
  exit /b 1
)

echo.
echo [2/4] 训练 rec...
pyocr train-rec --data datasets\rec\rec.yaml --name "%REC_NAME%" --device "%DEVICE%" --epochs %EPOCHS% --imgsz %IMGSZ% --batch %BATCH% --project models > "%RUN_DIR%\train_rec.log" 2>&1
if errorlevel 1 (
  echo rec 训练失败，详见：%RUN_DIR%\train_rec.log
  pause
  exit /b 1
)

REM 训练产物默认落在 runs\obb 下，这里先按实际路径查找
set "DET_W=runs\obb\models\%DET_NAME%\weights\best.pt"
set "REC_W=runs\obb\models\%REC_NAME%\weights\best.pt"

REM 兼容旧路径（如果用户自行移动过文件）
if not exist "%DET_W%" set "DET_W=models\%DET_NAME%\weights\best.pt"
if not exist "%REC_W%" set "REC_W=models\%REC_NAME%\weights\best.pt"

if not exist "%DET_W%" (
  echo 未找到 det 权重：%DET_W%
  pause
  exit /b 1
)
if not exist "%REC_W%" (
  echo 未找到 rec 权重：%REC_W%
  pause
  exit /b 1
)

REM 将权重同步一份到 models\<name>\weights，方便其他脚本/Qt 使用
mkdir "models\%DET_NAME%\weights" 2>nul
mkdir "models\%REC_NAME%\weights" 2>nul
copy /y "%DET_W%" "models\%DET_NAME%\weights\" >nul
copy /y "%REC_W%" "models\%REC_NAME%\weights\" >nul

set "DET_W=models\%DET_NAME%\weights\best.pt"
set "REC_W=models\%REC_NAME%\weights\best.pt"

echo.
echo [3/4] 推理抽查（随机挑 5 张 train 图）...
pyocr infer-sample --det-weights "%DET_W%" --rec-weights "%REC_W%" --images datasets\det\images\train --num 5 --vocab "%VOCAB%" --outdir "%RUN_DIR%\infer_train" --device "%DEVICE%" > "%RUN_DIR%\infer_train.log" 2>&1
if errorlevel 1 (
  echo 推理抽查(train)失败，详见：%RUN_DIR%\infer_train.log
  pause
  exit /b 1
)

echo.
echo [3b/4] 推理抽查（随机挑 5 张 val 图）...
pyocr infer-sample --det-weights "%DET_W%" --rec-weights "%REC_W%" --images datasets\det\images\val --num 5 --vocab "%VOCAB%" --outdir "%RUN_DIR%\infer_val" --device "%DEVICE%" > "%RUN_DIR%\infer_val.log" 2>&1
if errorlevel 1 (
  echo 推理抽查(val)失败，详见：%RUN_DIR%\infer_val.log
  pause
  exit /b 1
)

echo.
echo [4/4] 完成。
echo 训练日志：
echo   %RUN_DIR%\train_det.log
echo   %RUN_DIR%\train_rec.log
echo 推理结果目录：
echo   %RUN_DIR%\infer_train
echo   %RUN_DIR%\infer_val
echo.
echo 你可以对比不同 runs 目录下的推理截图，快速判断补标/改参是否有效。
pause
exit /b 0

