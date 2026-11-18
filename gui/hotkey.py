"""全局快捷键：窗口不在前台（比如正在玩游戏）时也能一键开关变声。

用 Windows 的 RegisterHotKey 注册到界面线程；WM_HOTKEY 通过 Qt 的原生事件过滤器收到。
其他系统上 available 为 False，界面会隐藏这项设置。
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Signal
from PySide6.QtWidgets import QApplication

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x4000
WM_HOTKEY = 0x0312
VK_F8, VK_F9, VK_F10 = 0x77, 0x78, 0x79

# 显示名 → (修饰键, 虚拟键码)。空字符串表示不使用快捷键。
CHOICES: dict[str, tuple[int, int] | None] = {
    "Ctrl+Alt+V": (MOD_CONTROL | MOD_ALT, ord("V")),
    "Ctrl+Shift+V": (MOD_CONTROL | MOD_SHIFT, ord("V")),
    "Ctrl+Alt+M": (MOD_CONTROL | MOD_ALT, ord("M")),
    "F8": (0, VK_F8),
    "F9": (0, VK_F9),
    "F10": (0, VK_F10),
    "": None,
}
HOTKEY_ID = 0x5643  # 'VC'

available = sys.platform == "win32"


class _Filter(QAbstractNativeEventFilter):
    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def nativeEventFilter(self, event_type, message):
        if event_type in (b"windows_generic_MSG", "windows_generic_MSG"):
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                self._callback()
                return True, 0
        return False, 0


class GlobalHotkey(QObject):
    """同一时间只注册一个组合键。register() 返回 None 表示成功，否则返回给用户看的原因。"""

    activated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current = ""
        self._filter = None
        if available:
            self._filter = _Filter(self.activated.emit)
            QApplication.instance().installNativeEventFilter(self._filter)

    def register(self, name: str) -> str | None:
        self.unregister()
        combo = CHOICES.get(name)
        if not name or combo is None:
            return None if not name else f"不支持的快捷键：{name}"
        if not available:
            return "当前系统不支持全局快捷键"
        modifiers, key = combo
        if not ctypes.windll.user32.RegisterHotKey(None, HOTKEY_ID, modifiers | MOD_NOREPEAT, key):
            return f"{name} 已被其他程序占用，请在设置里换一个"
        self.current = name
        return None

    def unregister(self) -> None:
        if self.current and available:
            ctypes.windll.user32.UnregisterHotKey(None, HOTKEY_ID)
        self.current = ""

    def close(self) -> None:
        self.unregister()
        if self._filter is not None:
            app = QApplication.instance()
            if app is not None:
                app.removeNativeEventFilter(self._filter)
            self._filter = None
