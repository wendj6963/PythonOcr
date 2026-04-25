# shuyun_pg_ocr

OCR 推理封装模块（与 `qt_app/ocr_rec_app` 测试工具默认参数一致）。

## 功能
- 封装定位/识别/公共参数（带默认值）。
- 根据图片路径执行 OCR，并保存回显图与 ROI 图片。
- 提供可直接运行的示例。

## 运行示例
在项目根目录执行：

```powershell
python -m qt_app.shuyun_pg_ocr_dll.shuyun_pg_ocr
```

说明：
- 默认模型路径为 `det.pt` / `rec.pt`，请放在项目根目录或传入绝对路径。
- 默认词表为 `vocab_rec.txt`。
- 示例会自动从 `Images/trains` 中选择一张图片。

## 关键接口
- `build_det_params()`：定位参数。
- `build_rec_params()`：识别参数。
- `build_common_params()`：公共参数。
- `run_ocr_by_path()`：根据图片路径识别并落盘。

## 输出
- `decode_img.bmp`：带定位框与识别结果的回显图。
- `roi_1.png`、`roi_2.png`...：每个 ROI 的裁剪结果。

## 嵌入式 Python 打包（免安装环境）
在 `qt_app/shuyun_pg_ocr_dll` 目录运行：

```bat
bundle_embed_and_build.bat
```

脚本会：
- 下载并解压 Python 3.10 嵌入式包
- 安装 pip 与依赖
- 复制 `qt_app` 与 `src` 到 `csharp_ocr/pyembed/app`
- 构建 `shuyun_pg_ocr_dll.dll`

生成目录：
- `csharp_ocr/pyembed`（嵌入式 Python + 依赖）

## C# 调用（嵌入式目录）
```csharp
var client = new ShuyunPgOcrDllClient();
client.InitializeEmbeddedDefault();
```
