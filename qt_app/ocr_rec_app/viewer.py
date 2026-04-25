from __future__ import annotations

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QWheelEvent
from PySide6.QtWidgets import QGraphicsView


class ImageView(QGraphicsView):
    # 初始化图像视图与缩放参数
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom = 0

    # 鼠标滚轮缩放
    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.angleDelta().y() == 0:
            return
        zoom_in = event.angleDelta().y() > 0
        factor = 1.25 if zoom_in else 0.8
        self._zoom += 1 if zoom_in else -1
        if self._zoom < -8:
            self._zoom = -8
            return
        if self._zoom > 24:
            self._zoom = 24
            return
        self.scale(factor, factor)

    # 重置缩放与变换
    def reset_view(self) -> None:
        self._zoom = 0
        self.resetTransform()

    # 居中显示指定点
    def center_on_point(self, pt: QPointF) -> None:
        self.centerOn(pt)
