from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QColor, QFont, QImage, QPixmap, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsPolygonItem,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
    QColorDialog,
    QGridLayout,
    QMessageBox,
)

from qt_app.ocr_rec_app.ocr_runner import OcrParams, OcrResult, run_ocr
from qt_app.ocr_rec_app.viewer import ImageView
from qt_app.ocr_rec_app.model_export import export_and_encrypt, validate_encrypted_model

CONFIG_PATH = Path("qt_app/ocr_rec_app/config/app_config.json")
DEFAULT_CONFIG = {
    "paths": {
        "image": "C:\\Users\\admin\\Desktop\\2_5_095949528.bmp",
        "image_dir": "",
        "det_model": "C:\\Users\\admin\\Desktop\\PythonOcr\\runs\\obb\\models\\det_exp\\weights\\det.pt",
        "rec_model": "C:\\Users\\admin\\Desktop\\PythonOcr\\runs\\obb\\models\\rec_exp\\weights\\rec.pt",
        "vocab": "vocab_rec.txt",
    },
    "params": {
        "det_conf": 0.25,
        "det_iou": 0.7,
        "det_max_det": 3,
        "rec_conf": 0.25,
        "rec_iou": 0.7,
        "det_imgsz": 640,
        "rec_imgsz": 320,
        "use_gpu": False,
        "num_threads": 0,
        "sort_by": "x",
        "mock_mode": False,
        "obb_color": "#00C800",
        "obb_line_width": 2,
        "rec_top_n": 14,
        "rec_min_score": 0.5,
        "rec_flip_enable": False,
        "rec_flip_min_score": 0.5,
        "roi_pad_ratio": 0.06,
        "roi_pad_px": 4,
        "rec_min_box": 4,
        "rec_row_thresh": 0.6,
        "rec_auto_imgsz": True,
        "rec_imgsz_min": 60,
        "rec_imgsz_max": 640,
        "roi_expected_lengths": "8,14,2",
        "export_key": "PG&shuyun@568.com",
    },
    "ui": {
        "mode": "单张识别",
    },
}


class OcrRecMainWindow(QMainWindow):
    # 初始化主窗口与状态
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OCR 解码测试工具")
        self.resize(1280, 800)

        self._scene = QGraphicsScene(self)
        self._view = ImageView(self)
        self._view.setScene(self._scene)

        self._image_path = ""
        self._pixmap_item = None
        self._batch_images: list[Path] = []
        self._batch_index = 0
        self._roi_list: list[np.ndarray] = []
        self._roi_index = 0

        self._build_ui()
        self._default_config = DEFAULT_CONFIG
        self._load_config()
        self._apply_mode_ui(self.mode_combo.currentText())

    # 构建主界面布局与控件
    def _build_ui(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)

        # Top: file/model selection
        top_box = QGroupBox("文件与模型选择")
        top_layout = QFormLayout(top_box)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["单张识别", "多张识别"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.mode_combo.currentIndexChanged.connect(lambda _: self._apply_mode_ui(self.mode_combo.currentText()))
        self.mode_combo.activated.connect(lambda _: self._apply_mode_ui(self.mode_combo.currentText()))
        top_layout.addRow("功能", self.mode_combo)

        self.image_path_edit = QLineEdit()
        self.image_browse_btn = QPushButton("选择图片")
        self.image_browse_btn.clicked.connect(self._choose_image)
        img_row = QHBoxLayout()
        img_row.addWidget(self.image_path_edit)
        img_row.addWidget(self.image_browse_btn)
        top_layout.addRow("图片", img_row)

        self.det_path_edit = QLineEdit()
        self.det_browse_btn = QPushButton("定位模型")
        self.det_reset_btn = QPushButton("恢复默认")
        self.det_browse_btn.clicked.connect(lambda: self._choose_model(self.det_path_edit))
        self.det_reset_btn.clicked.connect(lambda: self._reset_path("det_model"))
        det_row = QHBoxLayout()
        det_row.addWidget(self.det_path_edit)
        det_row.addWidget(self.det_browse_btn)
        det_row.addWidget(self.det_reset_btn)
        top_layout.addRow("定位模型", det_row)

        self.rec_path_edit = QLineEdit()
        self.rec_browse_btn = QPushButton("识别模型")
        self.rec_reset_btn = QPushButton("恢复默认")
        self.rec_browse_btn.clicked.connect(lambda: self._choose_model(self.rec_path_edit))
        self.rec_reset_btn.clicked.connect(lambda: self._reset_path("rec_model"))
        rec_row = QHBoxLayout()
        rec_row.addWidget(self.rec_path_edit)
        rec_row.addWidget(self.rec_browse_btn)
        rec_row.addWidget(self.rec_reset_btn)
        top_layout.addRow("识别模型", rec_row)

        export_box = QGroupBox("模型转换")
        export_layout = QFormLayout(export_box)
        self.export_key_edit = QLineEdit()
        self.export_key_edit.setText("PG&shuyun@568.com")
        self.export_btn = QPushButton("导出&加密")
        self.export_btn.clicked.connect(self._export_and_encrypt)
        self.export_validate_btn = QPushButton("加密模型验证")
        self.export_validate_btn.clicked.connect(self._validate_encrypted_models)

        export_layout.addRow("加密KEY", self.export_key_edit)
        export_btn_row = QHBoxLayout()
        export_btn_row.addWidget(self.export_btn)
        export_btn_row.addWidget(self.export_validate_btn)
        export_layout.addRow(export_btn_row)


        # Center: image view
        image_box = QGroupBox("图片预览")
        image_layout = QVBoxLayout(image_box)
        image_layout.addWidget(self._view)

        # Params
        param_box = QGroupBox("推理参数")
        param_layout = QVBoxLayout(param_box)

        det_box = QGroupBox("定位参数")
        det_layout = QFormLayout(det_box)

        self.det_conf = QDoubleSpinBox()
        self.det_conf.setRange(0.0, 1.0)
        self.det_conf.setSingleStep(0.01)
        det_layout.addRow("定位阈值", self.det_conf)

        self.det_iou = QDoubleSpinBox()
        self.det_iou.setRange(0.0, 1.0)
        self.det_iou.setSingleStep(0.01)
        det_layout.addRow("NMS 阈值", self.det_iou)

        self.det_imgsz = QSpinBox()
        self.det_imgsz.setRange(256, 2048)
        self.det_imgsz.setSingleStep(32)
        det_layout.addRow("定位尺寸", self.det_imgsz)

        self.det_max_det = QSpinBox()
        self.det_max_det.setRange(1, 9999)
        self.det_max_det.setSingleStep(1)
        det_layout.addRow("定位数量", self.det_max_det)

        self.obb_color_edit = QLineEdit("#00C800")
        self.obb_color_btn = QPushButton("选择颜色")
        self.obb_color_btn.clicked.connect(self._choose_obb_color)
        color_row = QHBoxLayout()
        color_row.addWidget(self.obb_color_edit)
        color_row.addWidget(self.obb_color_btn)
        det_layout.addRow("框颜色", color_row)

        self.obb_line_width = QSpinBox()
        self.obb_line_width.setRange(1, 10)
        self.obb_line_width.setSingleStep(1)
        det_layout.addRow("线宽", self.obb_line_width)

        rec_box = QGroupBox("识别参数")
        rec_layout = QGridLayout(rec_box)
        rec_layout.setColumnStretch(1, 1)
        rec_layout.setColumnStretch(3, 1)

        self.rec_conf = QDoubleSpinBox()
        self.rec_conf.setRange(0.0, 1.0)
        self.rec_conf.setSingleStep(0.01)

        self.rec_iou = QDoubleSpinBox()
        self.rec_iou.setRange(0.0, 1.0)
        self.rec_iou.setSingleStep(0.01)

        self.rec_imgsz = QSpinBox()
        self.rec_imgsz.setRange(64, 1024)
        self.rec_imgsz.setSingleStep(32)

        self.rec_auto_imgsz = QCheckBox("自动识别尺寸")
        self.rec_auto_imgsz.toggled.connect(self._update_rec_imgsz_ui)

        self.rec_imgsz_min = QSpinBox()
        self.rec_imgsz_min.setRange(64, 1024)
        self.rec_imgsz_min.setSingleStep(32)

        self.rec_imgsz_max = QSpinBox()
        self.rec_imgsz_max.setRange(64, 1024)
        self.rec_imgsz_max.setSingleStep(32)

        self.roi_pad_ratio = QDoubleSpinBox()
        self.roi_pad_ratio.setRange(0.0, 0.2)
        self.roi_pad_ratio.setSingleStep(0.01)

        self.roi_pad_px = QSpinBox()
        self.roi_pad_px.setRange(0, 50)
        self.roi_pad_px.setSingleStep(1)

        self.rec_min_box = QSpinBox()
        self.rec_min_box.setRange(1, 50)
        self.rec_min_box.setSingleStep(1)

        self.rec_row_thresh = QDoubleSpinBox()
        self.rec_row_thresh.setRange(0.1, 2.0)
        self.rec_row_thresh.setSingleStep(0.1)

        self.roi_expected_len = QLineEdit()
        self.roi_expected_len.setPlaceholderText("例如: 8,14,2")

        self.rec_top_n = QSpinBox()
        self.rec_top_n.setRange(1, 9999)
        self.rec_top_n.setSingleStep(1)

        self.rec_min_score = QDoubleSpinBox()
        self.rec_min_score.setRange(0.0, 1.0)
        self.rec_min_score.setSingleStep(0.01)

        self.rec_flip_enable = QCheckBox("启用 180° 旋转识别")

        self.rec_flip_min_score = QDoubleSpinBox()
        self.rec_flip_min_score.setRange(0.0, 1.0)
        self.rec_flip_min_score.setSingleStep(0.01)

        self.use_gpu = QCheckBox("使用 GPU")

        self.num_threads = QSpinBox()
        self.num_threads.setRange(0, 64)

        self.vocab_path_edit = QLineEdit()
        self.vocab_browse_btn = QPushButton("选择字典")
        self.vocab_browse_btn.clicked.connect(lambda: self._choose_vocab())
        vocab_row = QHBoxLayout()
        vocab_row.addWidget(self.vocab_path_edit)
        vocab_row.addWidget(self.vocab_browse_btn)
        vocab_wrap = QWidget()
        vocab_wrap.setLayout(vocab_row)

        self.sort_by = QComboBox()
        self.sort_by.addItems(["x", "y", "line"])

        self.mock_mode = QCheckBox("启用 mock 模式")

        # 识别参数按优先级分成两列布局
        left_items = [
            ("识别阈值", self.rec_conf),
            ("识别 NMS", self.rec_iou),
            ("识别尺寸", self.rec_imgsz),
            ("尺寸自适应", self.rec_auto_imgsz),
            ("识别最小尺寸", self.rec_imgsz_min),
            ("识别最大尺寸", self.rec_imgsz_max),
            ("TopN", self.rec_top_n),
            ("最小分数", self.rec_min_score),
        ]
        right_items = [
            ("ROI 边缘比例", self.roi_pad_ratio),
            ("ROI 边缘像素", self.roi_pad_px),
            ("最小框尺寸", self.rec_min_box),
            ("行分组阈值", self.rec_row_thresh),
            ("ROI长度(逗号)", self.roi_expected_len),
            ("旋转增强", self.rec_flip_enable),
            ("旋转最小分数", self.rec_flip_min_score),
            ("设备", self.use_gpu),
            ("线程数", self.num_threads),
            ("字符集", vocab_wrap),
            ("排序方式", self.sort_by),
            ("调试", self.mock_mode),
        ]
        for row, (label, widget) in enumerate(left_items):
            rec_layout.addWidget(QLabel(label), row, 0)
            rec_layout.addWidget(widget, row, 1)
        for row, (label, widget) in enumerate(right_items):
            rec_layout.addWidget(QLabel(label), row, 2)
            rec_layout.addWidget(widget, row, 3)

        self.reset_params_btn = QPushButton("恢复默认参数")
        self.reset_params_btn.clicked.connect(self._reset_params)

        self._update_rec_imgsz_ui(self.rec_auto_imgsz.isChecked())

        preview_box = QGroupBox("ROI 预览")
        preview_layout = QVBoxLayout(preview_box)
        self.roi_preview = QLabel("无")
        self.roi_preview.setAlignment(Qt.AlignCenter)
        self.roi_preview.setMinimumHeight(160)

        roi_btn_row = QHBoxLayout()
        self.roi_prev_btn = QPushButton("上一张 ROI")
        self.roi_prev_btn.clicked.connect(self._roi_prev)
        self.roi_next_btn = QPushButton("下一张 ROI")
        self.roi_next_btn.clicked.connect(self._roi_next)

        roi_btn_row.addWidget(self.roi_prev_btn)
        roi_btn_row.addWidget(self.roi_next_btn)

        preview_layout.addWidget(self.roi_preview)
        preview_layout.addLayout(roi_btn_row)
        param_layout.addWidget(det_box)
        param_layout.addWidget(rec_box)
        param_layout.addWidget(self.reset_params_btn)
        param_layout.addWidget(preview_box)

        # Bottom: actions + log
        action_box = QGroupBox("执行区")
        action_layout = QVBoxLayout(action_box)

        stats_box = QGroupBox("统计")
        stats_layout = QHBoxLayout(stats_box)
        self.stats_label = QLabel("图片数量: 0  当前: 0/0")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("识别")
        self.run_btn.clicked.connect(self._run_ocr)
        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self._run_prev)
        self.prev_btn.setVisible(False)
        self.save_cfg_btn = QPushButton("保存配置")
        self.save_cfg_btn.clicked.connect(self._save_config)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.prev_btn)
        btn_row.addWidget(self.save_cfg_btn)
        btn_row.addStretch()

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        action_layout.addWidget(stats_box)
        action_layout.addLayout(btn_row)
        action_layout.addWidget(self.log)

        # Layout composition
        mid_layout = QHBoxLayout()
        left_col = QVBoxLayout()
        left_col.addWidget(top_box)
        left_col.addWidget(export_box)
        left_col.addWidget(image_box, 1)
        mid_layout.addLayout(left_col, 2)
        mid_layout.addWidget(param_box, 1)

        root_layout.addLayout(mid_layout, 5)
        root_layout.addWidget(action_box, 2)

        self.setCentralWidget(root)

    # 追加一行日志到输出框
    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {msg}")

    # 从配置文件加载默认路径与参数
    def _load_config(self) -> None:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        else:
            data = DEFAULT_CONFIG
        self._default_config = data
        paths = data.get("paths", {})
        params = data.get("params", {})
        ui = data.get("ui", {})

        self.image_path_edit.setText(paths.get("image", ""))
        self.det_path_edit.setText(paths.get("det_model", ""))
        self.rec_path_edit.setText(paths.get("rec_model", ""))
        self.vocab_path_edit.setText(paths.get("vocab", ""))

        self.mode_combo.setCurrentText(ui.get("mode", "单张识别"))

        self.det_conf.setValue(float(params.get("det_conf", 0.25)))
        self.det_iou.setValue(float(params.get("det_iou", 0.7)))
        self.det_max_det.setValue(int(params.get("det_max_det", 3)))
        self.rec_conf.setValue(float(params.get("rec_conf", 0.25)))
        self.rec_iou.setValue(float(params.get("rec_iou", 0.7)))
        self.det_imgsz.setValue(int(params.get("det_imgsz", 640)))
        self.rec_imgsz.setValue(int(params.get("rec_imgsz", 320)))
        self.use_gpu.setChecked(bool(params.get("use_gpu", False)))
        self.num_threads.setValue(int(params.get("num_threads", 0)))
        self.sort_by.setCurrentText(params.get("sort_by", "x"))
        self.mock_mode.setChecked(bool(params.get("mock_mode", False)))
        self.obb_color_edit.setText(params.get("obb_color", "#00C800"))
        self.obb_line_width.setValue(int(params.get("obb_line_width", 2)))
        self.rec_top_n.setValue(int(params.get("rec_top_n", 14)))
        self.rec_min_score.setValue(float(params.get("rec_min_score", 0.5)))
        self.rec_flip_enable.setChecked(bool(params.get("rec_flip_enable", False)))
        self.rec_flip_min_score.setValue(float(params.get("rec_flip_min_score", 0.5)))
        self.roi_pad_ratio.setValue(float(params.get("roi_pad_ratio", 0.03)))
        self.roi_pad_px.setValue(int(params.get("roi_pad_px", 2)))
        self.rec_min_box.setValue(int(params.get("rec_min_box", 4)))
        self.rec_row_thresh.setValue(float(params.get("rec_row_thresh", 0.6)))
        roi_len_cfg = params.get("roi_expected_lengths", "")
        self.roi_expected_len.setText(self._format_roi_expected_lengths(roi_len_cfg))
        self.export_key_edit.setText(str(params.get("export_key", "PG&shuyun@568.com")))
        self.rec_auto_imgsz.setChecked(bool(params.get("rec_auto_imgsz", False)))
        self.rec_imgsz_min.setValue(int(params.get("rec_imgsz_min", 160)))
        self.rec_imgsz_max.setValue(int(params.get("rec_imgsz_max", 640)))
        self._update_rec_imgsz_ui(self.rec_auto_imgsz.isChecked())

        image_path = self.image_path_edit.text().strip()
        if image_path and self._is_single_mode():
            self._load_image(image_path)
        self._update_stats()

    # 保存当前路径与参数到配置文件
    def _save_config(self) -> None:
        data = {
            "paths": {
                "image": self.image_path_edit.text().strip(),
                "image_dir": self.image_path_edit.text().strip() if self._is_multi_mode() else "",
                "det_model": self.det_path_edit.text().strip(),
                "rec_model": self.rec_path_edit.text().strip(),
                "vocab": self.vocab_path_edit.text().strip(),
            },
            "params": {
                **asdict(self._collect_params()),
                "obb_color": self.obb_color_edit.text().strip() or "#00C800",
                "obb_line_width": self.obb_line_width.value(),
                "export_key": self.export_key_edit.text().strip(),
            },
            "ui": {
                "mode": self.mode_combo.currentText(),
            },
        }
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log("配置已保存")

    # 恢复模型/路径到默认配置
    def _reset_path(self, key: str) -> None:
        paths = self._default_config.get("paths", {})
        if key == "det_model":
            self.det_path_edit.setText(paths.get("det_model", ""))
        elif key == "rec_model":
            self.rec_path_edit.setText(paths.get("rec_model", ""))
        elif key == "image_dir":
            self.image_path_edit.setText(paths.get("image_dir", ""))

    # 恢复参数到默认配置
    def _reset_params(self) -> None:
        params = self._default_config.get("params", {})
        self.det_conf.setValue(float(params.get("det_conf", 0.25)))
        self.det_iou.setValue(float(params.get("det_iou", 0.7)))
        self.det_max_det.setValue(int(params.get("det_max_det", 3)))
        self.rec_conf.setValue(float(params.get("rec_conf", 0.25)))
        self.rec_iou.setValue(float(params.get("rec_iou", 0.7)))
        self.det_imgsz.setValue(int(params.get("det_imgsz", 640)))
        self.rec_imgsz.setValue(int(params.get("rec_imgsz", 320)))
        self.use_gpu.setChecked(bool(params.get("use_gpu", False)))
        self.num_threads.setValue(int(params.get("num_threads", 0)))
        self.sort_by.setCurrentText(params.get("sort_by", "x"))
        self.mock_mode.setChecked(bool(params.get("mock_mode", False)))
        self.obb_color_edit.setText(params.get("obb_color", "#00C800"))
        self.obb_line_width.setValue(int(params.get("obb_line_width", 2)))
        self.rec_top_n.setValue(int(params.get("rec_top_n", 14)))
        self.rec_min_score.setValue(float(params.get("rec_min_score", 0.5)))
        self.rec_flip_enable.setChecked(bool(params.get("rec_flip_enable", False)))
        self.rec_flip_min_score.setValue(float(params.get("rec_flip_min_score", 0.5)))
        self.roi_pad_ratio.setValue(float(params.get("roi_pad_ratio", 0.03)))
        self.roi_pad_px.setValue(int(params.get("roi_pad_px", 2)))
        self.rec_min_box.setValue(int(params.get("rec_min_box", 4)))
        self.rec_row_thresh.setValue(float(params.get("rec_row_thresh", 0.6)))
        self.roi_expected_len.setText(self._format_roi_expected_lengths(params.get("roi_expected_lengths", "")))
        self.export_key_edit.setText(str(params.get("export_key", "PG&shuyun@568.com")))
        self.rec_auto_imgsz.setChecked(bool(params.get("rec_auto_imgsz", False)))
        self.rec_imgsz_min.setValue(int(params.get("rec_imgsz_min", 160)))
        self.rec_imgsz_max.setValue(int(params.get("rec_imgsz_max", 640)))
        self._update_rec_imgsz_ui(self.rec_auto_imgsz.isChecked())

    # 选择单张图片或多张文件夹
    def _choose_image(self) -> None:
        if self._is_multi_mode():
            path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "")
            if not path:
                return
            self.image_path_edit.setText(path)
            self._batch_images = self._scan_images(Path(path))
            self._batch_index = 0
            if self._batch_images:
                self._load_image(str(self._batch_images[0]))
            else:
                self._log("文件夹内未找到图片")
            self._update_stats()
            return

        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.bmp)")
        if not path:
            return
        self.image_path_edit.setText(path)
        self._load_image(path)
        self._update_stats()

    # 加载图片并显示在画布
    def _load_image(self, path: str) -> None:
        self._scene.clear()
        self._view.reset_view()
        image = QImage(path)
        if image.isNull():
            self._log(f"无法读取图片: {path}")
            return
        pixmap = QPixmap.fromImage(image)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._image_path = path
        self._log(f"已加载图片: {path}")

    # 选择模型文件
    def _choose_model(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "Model (*.pt *.onnx)")
        if not path:
            return
        target.setText(path)

    # 选择字符集文件
    def _choose_vocab(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择字符集", "", "Text (*.txt)")
        if not path:
            return
        self.vocab_path_edit.setText(path)

    # 选择 OBB 框颜色
    def _choose_obb_color(self) -> None:
        color = QColorDialog.getColor(QColor(self.obb_color_edit.text()), self, "选择框颜色")
        if not color.isValid():
            return
        self.obb_color_edit.setText(color.name())

    # 从界面控件收集推理参数
    def _collect_params(self) -> OcrParams:
        vocab_text = self.vocab_path_edit.text().strip()
        vocab_path = Path(vocab_text) if vocab_text else None
        return OcrParams(
            det_conf=self.det_conf.value(),
            det_iou=self.det_iou.value(),
            det_max_det=self.det_max_det.value(),
            rec_conf=self.rec_conf.value(),
            rec_iou=self.rec_iou.value(),
            det_imgsz=self.det_imgsz.value(),
            rec_imgsz=self.rec_imgsz.value(),
            use_gpu=self.use_gpu.isChecked(),
            num_threads=self.num_threads.value(),
            vocab_path=vocab_path,
            sort_by=self.sort_by.currentText(),
            mock_mode=self.mock_mode.isChecked(),
            rec_top_n=self.rec_top_n.value(),
            rec_min_score=self.rec_min_score.value(),
            rec_flip_enable=self.rec_flip_enable.isChecked(),
            rec_flip_min_score=self.rec_flip_min_score.value(),
            roi_pad_ratio=self.roi_pad_ratio.value(),
            roi_pad_px=self.roi_pad_px.value(),
            rec_min_box=self.rec_min_box.value(),
            rec_row_thresh=self.rec_row_thresh.value(),
            rec_auto_imgsz=self.rec_auto_imgsz.isChecked(),
            rec_imgsz_min=self.rec_imgsz_min.value(),
            rec_imgsz_max=self.rec_imgsz_max.value(),
            roi_expected_lengths=self._parse_roi_expected_lengths(),
        )

    # 切换模式时重置状态并更新 UI
    def _on_mode_changed(self, text: str) -> None:
        self._batch_images = []
        self._batch_index = 0
        self._apply_mode_ui(text)
        self._update_stats()
        self._log(f"已切换模式: {text}")

    # 根据模式更新按钮与文案
    def _apply_mode_ui(self, text: str) -> None:
        if text == "多张识别":
            self.image_browse_btn.setText("选择文件夹")
            self.run_btn.setText("下一张")
            self.prev_btn.setVisible(True)
        else:
            self.image_browse_btn.setText("选择图片")
            self.run_btn.setText("识别")
            self.prev_btn.setVisible(False)

    # 刷新统计区域
    def _update_stats(self) -> None:
        total = len(self._batch_images)
        if self._is_multi_mode() and total > 0:
            current = max(1, min(self._batch_index, total))
            self.stats_label.setText(f"图片数量: {total}  当前: {current}/{total}")
        else:
            self.stats_label.setText("图片数量: 0  当前: 0/0")

    # 扫描文件夹内的图片列表
    def _scan_images(self, folder: Path) -> list[Path]:
        if not folder.exists():
            return []
        exts = {".bmp", ".png", ".jpg", ".jpeg"}
        imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
        return sorted(imgs)

    # 取下一张图片（用于多张模式）
    def _next_batch_image(self) -> Path | None:
        if self._batch_index >= len(self._batch_images):
            return None
        img = self._batch_images[self._batch_index]
        self._batch_index += 1
        return img

    # 取上一张图片（用于多张模式）
    def _prev_batch_image(self) -> Path | None:
        if self._batch_index <= 1:
            if self._batch_images:
                self._batch_index = 1
                return self._batch_images[0]
            return None
        self._batch_index -= 1
        return self._batch_images[self._batch_index - 1]

    # 执行推理入口（单张/多张分流）
    def _run_ocr(self) -> None:
        img_path = self.image_path_edit.text().strip()
        det_model = self.det_path_edit.text().strip()
        rec_model = self.rec_path_edit.text().strip()

        params = self._collect_params()

        if self._is_multi_mode():
            if not img_path:
                self._log("请先选择图片文件夹")
                return
            folder = Path(img_path)
            if not folder.exists():
                self._log(f"文件夹不存在: {img_path}")
                return
            if not self._batch_images:
                self._batch_images = self._scan_images(folder)
                self._batch_index = 0
            if not self._batch_images:
                self._log("文件夹内未找到图片")
                return
            next_img = self._next_batch_image()
            if next_img is None:
                self._log("多张识别完成")
                return
            self._load_image(str(next_img))
            img_path = str(next_img)
            self._update_stats()
        else:
            if not img_path:
                self._log("请先选择图片")
                return
            if not Path(img_path).exists():
                self._log(f"图片不存在: {img_path}")
                return

        self._run_ocr_on_path(img_path, det_model, rec_model, params)

    # 在指定路径上执行推理并回显结果
    def _run_ocr_on_path(self, img_path: str, det_model: str, rec_model: str, params: OcrParams) -> None:
        if not params.mock_mode:
            if not det_model:
                self._log("请先选择定位模型")
                return
            if not rec_model:
                self._log("请先选择识别模型")
                return
            if not Path(det_model).exists():
                self._log(f"定位模型不存在: {det_model}")
                return
            if not Path(rec_model).exists():
                self._log(f"识别模型不存在: {rec_model}")
                return

        roi_len_info = ",".join(str(v) for v in (params.roi_expected_lengths or ())) or "-"
        self._log(
            "推理参数: det_conf={:.2f}, det_iou={:.2f}, det_max_det={}, rec_conf={:.2f}, rec_iou={:.2f}, det_imgsz={}, rec_imgsz={}, auto_imgsz={}, imgsz_min={}, imgsz_max={}, topN={}, min_score={:.2f}, min_box={}, row_thresh={:.2f}, roi_len={}, pad_ratio={:.2f}, pad_px={}, flip={}, flip_min_score={:.2f}, device={}".format(
                params.det_conf,
                params.det_iou,
                params.det_max_det,
                params.rec_conf,
                params.rec_iou,
                params.det_imgsz,
                params.rec_imgsz,
                "ON" if params.rec_auto_imgsz else "OFF",
                params.rec_imgsz_min,
                params.rec_imgsz_max,
                params.rec_top_n,
                params.rec_min_score,
                params.rec_min_box,
                params.rec_row_thresh,
                roi_len_info,
                params.roi_pad_ratio,
                params.roi_pad_px,
                "ON" if params.rec_flip_enable else "OFF",
                params.rec_flip_min_score,
                "GPU" if params.use_gpu else "CPU",
            )
        )
        self._log("开始推理...")
        try:
            result = run_ocr(Path(img_path), Path(det_model), Path(rec_model), params)
        except Exception as exc:
            self._log(f"推理失败: {exc}")
            return

        self._draw_result(result)
        self._log(
            f"det={result.det_time_ms:.1f}ms, rec={result.rec_time_ms:.1f}ms, total={result.total_time_ms:.1f}ms"
        )
        self._log(f"框数量={len(result.boxes)}, 平均置信度={result.mean_score:.3f}")

    # 多张模式上一张并执行识别
    def _run_prev(self) -> None:
        if not self._is_multi_mode():
            return
        folder_path = self.image_path_edit.text().strip()
        if not folder_path:
            self._log("请先选择图片文件夹")
            return
        if not self._batch_images:
            self._batch_images = self._scan_images(Path(folder_path))
            self._batch_index = 0
        if not self._batch_images:
            self._log("文件夹内未找到图片")
            return
        prev_img = self._prev_batch_image()
        if prev_img is None:
            self._log("已是第一张")
            return
        self._load_image(str(prev_img))
        self._update_stats()
        self._run_ocr_on_path(str(prev_img), self.det_path_edit.text().strip(), self.rec_path_edit.text().strip(), self._collect_params())

    # 将识别结果绘制到画布
    def _draw_result(self, result: OcrResult) -> None:
        if self._pixmap_item is None:
            return
        # keep image, remove previous overlays
        for item in list(self._scene.items()):
            if item is self._pixmap_item:
                continue
            self._scene.removeItem(item)

        box_pen = QPen(QColor(0, 200, 0))
        text_color = QColor(255, 255, 0)
        bg_color = QColor(0, 0, 0, 160)
        font = QFont("Arial", 12)

        obb_color = QColor(self.obb_color_edit.text())
        box_pen.setColor(obb_color)
        box_pen.setWidth(self.obb_line_width.value())

        for box in result.boxes:
            if box.quad and len(box.quad) == 8:
                pts = [QPointF(box.quad[i], box.quad[i + 1]) for i in range(0, 8, 2)]
                poly_item = QGraphicsPolygonItem(QPolygonF(pts))
                poly_item.setPen(box_pen)
                poly_item.setZValue(2)
                self._scene.addItem(poly_item)
                label_x = min([p.x() for p in pts])
                label_y = min([p.y() for p in pts])
            else:
                rect = QRectF(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1)
                rect_item = QGraphicsRectItem(rect)
                rect_item.setPen(box_pen)
                rect_item.setZValue(2)
                self._scene.addItem(rect_item)
                label_x = rect.x()
                label_y = rect.y()

            label_prefix = f"ROI{box.roi_index}: " if box.roi_index is not None else ""
            label = f"{label_prefix}{box.text} ({box.score:.2f})" if box.text else f"{label_prefix}({box.score:.2f})"
            text_item = QGraphicsSimpleTextItem(label)
            text_item.setFont(font)
            text_item.setBrush(text_color)
            text_item.setZValue(3)
            text_rect = text_item.boundingRect()
            text_item.setPos(label_x, max(0.0, label_y - text_rect.height() - 4))

            bg_rect = QGraphicsRectItem(text_item.boundingRect().adjusted(-2, -1, 2, 1))
            bg_rect.setPos(text_item.pos())
            bg_rect.setBrush(bg_color)
            bg_rect.setPen(Qt.NoPen)
            bg_rect.setZValue(1)

            self._scene.addItem(bg_rect)
            self._scene.addItem(text_item)

        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._roi_list = result.roi_previews
        self._roi_index = 0
        self._update_roi_preview()


    # 更新 ROI 预览
    def _update_roi_preview(self) -> None:
        if not self._roi_list:
            self.roi_preview.setText("无")
            self.roi_preview.setPixmap(QPixmap())
            return
        if self._roi_index < 0 or self._roi_index >= len(self._roi_list):
            self._roi_index = 0
        roi = self._roi_list[self._roi_index]
        h, w = roi.shape[:2]
        if h == 0 or w == 0:
            self.roi_preview.setText("无")
            self.roi_preview.setPixmap(QPixmap())
            return
        rgb = roi[:, :, ::-1]
        rgb = np.ascontiguousarray(rgb)
        image = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        self.roi_preview.setPixmap(
            QPixmap.fromImage(image).scaled(self.roi_preview.size(), Qt.AspectRatioMode.KeepAspectRatio)
        )
        self._log(f"ROI 预览: {self._roi_index + 1}/{len(self._roi_list)}")

    # 上一张 ROI
    def _roi_prev(self) -> None:
        if not self._roi_list:
            return
        self._roi_index = max(0, self._roi_index - 1)
        self._update_roi_preview()

    # 下一张 ROI
    def _roi_next(self) -> None:
        if not self._roi_list:
            return
        self._roi_index = min(len(self._roi_list) - 1, self._roi_index + 1)
        self._update_roi_preview()

    # 判断是否为多张模式
    def _is_multi_mode(self) -> bool:
        return self.mode_combo.currentText() == "多张识别"

    # 判断是否为单张模式
    def _is_single_mode(self) -> bool:
        return self.mode_combo.currentText() == "单张识别"

    # 切换识别尺寸控件状态
    def _update_rec_imgsz_ui(self, enabled: bool) -> None:
        self.rec_imgsz.setEnabled(not enabled)
        self.rec_imgsz_min.setVisible(enabled)
        self.rec_imgsz_max.setVisible(enabled)
        # 同步标签可见性（同一行的 QLabel）
        for label in self.findChildren(QLabel):
            text = label.text()
            if text == "识别最小尺寸":
                label.setVisible(enabled)
            elif text == "识别最大尺寸":
                label.setVisible(enabled)

    def _parse_roi_expected_lengths(self) -> tuple[int, ...] | None:
        raw = self.roi_expected_len.text().strip()
        if not raw:
            return None
        raw = raw.replace("，", ",")
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        values: list[int] = []
        for p in parts:
            if p.isdigit():
                values.append(int(p))
        return tuple(values) if values else None

    def _format_roi_expected_lengths(self, value: object) -> str:
        if isinstance(value, (list, tuple)):
            return ",".join(str(int(v)) for v in value if str(v).strip())
        if isinstance(value, str):
            return value
        return ""

    def _export_and_encrypt(self) -> None:
        det_path = self.det_path_edit.text().strip()
        rec_path = self.rec_path_edit.text().strip()
        if not det_path or not rec_path:
            self._log("请先选择定位模型与识别模型")
            return
        out_dir = Path("models")
        key = self.export_key_edit.text().strip()
        if not key:
            self._log("加密KEY不能为空")
            return

        self._log("导出参数: det_nms=ON, rec_nms=OFF")

        try:
            det_res = export_and_encrypt(
                Path(det_path),
                int(self.det_imgsz.value()),
                out_dir,
                key,
                onnx_name="det.onnx",
                enc_name="det.onnx.enc",
                nms=True,
            )
            rec_res = export_and_encrypt(
                Path(rec_path),
                int(self.rec_imgsz.value()),
                out_dir,
                key,
                onnx_name="rec.onnx",
                enc_name="rec.onnx.enc",
                nms=True,
            )
        except Exception as exc:
            self._log(f"导出失败: {exc}")
            return

        min_delta = 69
        max_delta = 84
        det_delta = det_res.bytes_out - det_res.bytes_in
        rec_delta = rec_res.bytes_out - rec_res.bytes_in
        self._log(f"det.onnx.enc 比 det.onnx 大 {det_delta} bytes")
        self._log(f"rec.onnx.enc 比 rec.onnx 大 {rec_delta} bytes")

        def _delta_ok(delta: int) -> bool:
            return min_delta <= delta <= max_delta

        if not _delta_ok(det_delta) or not _delta_ok(rec_delta):
            self._log(f"加密文件大小异常：差值不在 {min_delta}~{max_delta} bytes 范围内")
            QMessageBox.warning(self, "加密校验失败", "加密文件大小异常，请检查导出/加密流程。")
            return

        self._log(f"定位模型导出: {det_res.onnx_path}")
        self._log(f"定位模型加密: {det_res.enc_path}")
        self._log(f"识别模型导出: {rec_res.onnx_path}")
        self._log(f"识别模型加密: {rec_res.enc_path}")
        QMessageBox.information(self, "导出完成", "模型导出并加密成功，已保存到 models 目录。")

    # 验证加密模型是否可正确解密并与 ONNX 一致
    def _validate_encrypted_models(self) -> None:
        out_dir = Path("models")
        key = self.export_key_edit.text().strip()
        if not key:
            self._log("加密KEY不能为空")
            return

        det_onnx = out_dir / "det_best.onnx"
        rec_onnx = out_dir / "rec_best.onnx"
        det_enc = out_dir / "det_best.onnx.enc"
        rec_enc = out_dir / "rec_best.onnx.enc"

        det_res = validate_encrypted_model(det_enc, key, det_onnx)
        rec_res = validate_encrypted_model(rec_enc, key, rec_onnx)

        self._log(f"验证定位模型: {det_res.message}")
        if det_res.ok:
            self._log(f"定位模型解密大小: {det_res.bytes_dec} bytes")
        self._log(f"验证识别模型: {rec_res.message}")
        if rec_res.ok:
            self._log(f"识别模型解密大小: {rec_res.bytes_dec} bytes")

        if det_res.ok and rec_res.ok:
            QMessageBox.information(self, "验证通过", "加密模型验证通过，可用于解码。")
        else:
            QMessageBox.warning(self, "验证失败", "加密模型验证失败，请查看日志。")
