"""角色变声器的 PySide6 桌面界面。启动：python -m gui，或双击根目录的 start-gui.cmd。"""
from pathlib import Path

__version__ = (Path(__file__).resolve().parents[1] / "VERSION").read_text(encoding="utf-8").strip()
