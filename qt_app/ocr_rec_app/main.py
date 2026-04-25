from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

# Allow running as a standalone script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qt_app.ocr_rec_app.app import OcrRecMainWindow  # noqa: E402


def main() -> None:
    app = QApplication(sys.argv)
    window = OcrRecMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

