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
