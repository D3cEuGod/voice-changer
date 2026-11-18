"""生成原创的程序图标 assets/app-icon.png 和 assets/app-icon.ico（粉色圆角方块上一支白色麦克风）。

公开版没有角色头像，窗口、任务栏和快捷方式都用这个图标。改了 gui/widgets.py 里的 app_pixmap 后重新运行：
  .venv-gui\\Scripts\\python.exe tools\\make_app_icon.py
"""
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QByteArray, QIODevice  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402

_APP = None
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def png_bytes(side: int) -> bytes:
    from gui.widgets import app_pixmap
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.WriteOnly)
    app_pixmap(side).save(buffer, "PNG")
    return bytes(data)


def ico(images: dict[int, bytes]) -> bytes:
    """Windows Vista 起支持 PNG 压缩的 ICO 条目；每个尺寸一条。"""
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for side, blob in images.items():
        entries += struct.pack("<BBBBHHII", side % 256, side % 256, 0, 0, 1, 32, len(blob), offset + len(blobs))
        blobs += blob
    return header + entries + blobs


def main() -> None:
    global _APP  # 画图需要一个活着的 QGuiApplication
    _APP = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    images = {side: png_bytes(side) for side in SIZES}
    (ROOT / "assets" / "app-icon.png").write_bytes(images[256])
    (ROOT / "assets" / "app-icon.ico").write_bytes(ico(images))
    print("wrote assets/app-icon.png and assets/app-icon.ico")


if __name__ == "__main__":
    main()
