# OCR 解码测试工具

本工具用于对单张图片进行 OCR 定位 + 识别推理，并在界面上可视化结果。

## 运行方式

在项目根目录执行：

```powershell
python -m qt_app.ocr_rec_app.main
```

## ONNX.enc 解码测试窗体

- 在主界面点击“onnx.enc解码”打开新窗体。
- 新窗体固定使用 `models/det_best.onnx.enc` 与 `models/rec_best.onnx.enc`。
- 也可直接运行：

```powershell
python -m qt_app.ocr_rec_app.onnx_views.main
```

## 模型路径配置

默认配置文件：`qt_app/ocr_rec_app/config/app_config.json`

默认路径示例：
- 定位模型：`qt_app/models/det/default_det.onnx`
- 识别模型：`qt_app/models/rec/default_rec.onnx`

如果你使用 03 脚本训练得到的 `best.pt`，请在界面中选择对应路径，或修改配置文件并保存。

## 参数说明

- rec_backend：识别后端（`yolo` 或 `ctc`）
- det_conf：定位置信度阈值
- det_iou：定位 NMS 阈值
- rec_conf：识别置信度阈值
- rec_iou：识别 NMS 阈值
- det_imgsz / rec_imgsz：输入尺寸
- use_gpu：使用 GPU（device=0）
- num_threads：CPU 线程数（0 表示默认）
- vocab：字符集文件（UTF-8，每行一个字符）
- sort_by：输出排序方式（x 或 y）
- mock_mode：启用 mock 模式，仅用于验证 UI 绘制

### 后端切换说明

- `yolo`：沿用当前“检测+类别”识别模型（兼容现有参数）
- `ctc`：使用 CRNN+CTC 识别模型（`rec_model` 建议选择 `train-rec-ctc` 产出的 `best.pt`）
- 两个后端共用同一套定位模型与 ROI 可视化流程
- 界面会根据 `rec_backend` 动态显示“当前后端生效参数说明”，帮助避免无效参数配置
- 在“文件与模型选择”区域可单独设置 `CTC模型(可选)`：
  - 当 `rec_backend=ctc` 时，优先使用该模型
  - 若未设置，会回退到自动探测路径
- 顶部“文件与模型选择”新增 `CTC模型(可选)` 输入框：
  - 当 `rec_backend=ctc` 时，优先使用该路径
  - A/B 对比时，CTC 分支也优先使用该路径

### A/B 一键对比

- 点击 `A/B对比（YOLO vs CTC）` 按钮，可在同一张图上连续执行两次推理
- 日志会输出并排对比信息：
  - 总耗时 / det耗时 / rec耗时 / 框数量 / 平均分数
  - 每个 ROI 的 `YOLO识别结果 vs CTC识别结果`
- 目前 A/B 对比仅支持“单张识别”模式
- A/B 对比中：YOLO 分支使用“识别模型”；CTC 分支优先使用 `CTC模型(可选)`。
  若未设置，则自动尝试 `models/rec_ctc_exp/weights/best.pt`。
- A/B 对比中：YOLO 分支使用当前“识别模型”路径；CTC 分支优先使用 `CTC模型(可选)` 路径。
  若未设置，则回退到自动探测（`models/rec_ctc_exp/weights/best.pt` 等）

## 模型转换与加密
- 在界面“模型转换”区域填写加密 KEY，点击“导出&加密”。
- 导出目录：`qt_app/ocr_rec_app/models`
- 输出文件：`*.onnx` 与 `*.onnx.enc`

### 命令行
```powershell
python -m qt_app.ocr_rec_app.model_export --model C:\path\to\best.pt --imgsz 640 --out qt_app\ocr_rec_app\models --key "PG&shuyun@568.com"
```

### 依赖
- `ultralytics`（导出 ONNX）
- `cryptography`（AES-256-GCM 加密）

## 常见问题

1. 提示 "模型不存在"
   - 请确认模型路径是否正确（支持 .pt/.onnx）。

2. 提示 "未安装 ultralytics"
   - 请先安装依赖或使用 Anaconda 环境运行。

3. 无法推理但 UI 可打开
   - 可先开启 mock_mode 检查界面绘制链路。
