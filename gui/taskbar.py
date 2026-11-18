"""Windows 任务栏身份。

进程和窗口使用与快捷方式（make-shortcut.ps1）相同的 AppUserModelID，固定后的图标和运行中的窗口归为一组。
窗口另外带上“重新启动”信息：在任务栏上右键运行中的图标选“固定到任务栏”时，固定的是 start-gui.cmd，
而不是一个不带参数的 pythonw.exe。
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

from .backend import ROOT

APP_USER_MODEL_ID = "LocalVoiceChanger.Gui"
DISPLAY_NAME = "角色变声器"
ICON = next((path for path in (ROOT / "assets" / "taffy-avatar.ico", ROOT / "assets" / "app-icon.ico")
             if path.exists()), ROOT / "assets" / "app-icon.ico")  # 公开版没有角色头像，用原创图标
START_GUI = ROOT / "start-gui.cmd"

_FMTID_APP_USER_MODEL = "{9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}"
_IID_PROPERTY_STORE = "{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}"
_PID_RELAUNCH_COMMAND, _PID_RELAUNCH_ICON, _PID_RELAUNCH_NAME, _PID_ID = 2, 3, 4, 5
_VT_LPWSTR = 31


class _GUID(ctypes.Structure):
    _fields_ = [("data1", wintypes.DWORD), ("data2", wintypes.WORD), ("data3", wintypes.WORD),
                ("data4", ctypes.c_ubyte * 8)]


class _PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", wintypes.DWORD)]


class _PROPVARIANT(ctypes.Structure):
    _fields_ = [("vt", ctypes.c_ushort), ("reserved", ctypes.c_ushort * 3),
                ("pointer", ctypes.c_void_p), ("padding", ctypes.c_void_p)]


def _guid(text: str) -> _GUID:
    guid = _GUID()
    ctypes.oledll.ole32.CLSIDFromString(ctypes.c_wchar_p(text), ctypes.byref(guid))
    return guid


def relaunch_command() -> str:
    cmd = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "cmd.exe")
    return f'"{cmd}" /d /c ""{START_GUI}""'


def set_process_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(APP_USER_MODEL_ID))
    except Exception:
        pass


def set_window_relaunch(hwnd: int) -> dict[str, str] | None:
    """给窗口写入 AppUserModelID 和重新启动信息，返回读回的值；不是 Windows 或失败时返回 None。"""
    if sys.platform != "win32":
        return None
    try:
        shell32 = ctypes.oledll.shell32
        shell32.SHGetPropertyStoreForWindow.argtypes = [wintypes.HWND, ctypes.POINTER(_GUID),
                                                        ctypes.POINTER(ctypes.c_void_p)]
        store = ctypes.c_void_p()
        shell32.SHGetPropertyStoreForWindow(wintypes.HWND(hwnd), ctypes.byref(_guid(_IID_PROPERTY_STORE)),
                                            ctypes.byref(store))
        vtable = ctypes.cast(store, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        key_pointer, value_pointer = ctypes.POINTER(_PROPERTYKEY), ctypes.POINTER(_PROPVARIANT)
        release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
        get_value = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, key_pointer, value_pointer)(vtable[5])
        set_value = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, key_pointer, value_pointer)(vtable[6])
        commit = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p)(vtable[7])
        fmtid = _guid(_FMTID_APP_USER_MODEL)
        values = {  # ID 必须先于重新启动信息写入
            _PID_ID: APP_USER_MODEL_ID,
            _PID_RELAUNCH_COMMAND: relaunch_command(),
            _PID_RELAUNCH_NAME: DISPLAY_NAME,
            _PID_RELAUNCH_ICON: f"{ICON},0",
        }
        try:
            for pid, text in values.items():
                buffer = ctypes.create_unicode_buffer(text)
                value = _PROPVARIANT(vt=_VT_LPWSTR, pointer=ctypes.cast(buffer, ctypes.c_void_p))
                set_value(store, ctypes.byref(_PROPERTYKEY(fmtid, pid)), ctypes.byref(value))
            commit(store)
            read_back = {}
            names = {_PID_ID: "id", _PID_RELAUNCH_COMMAND: "command", _PID_RELAUNCH_NAME: "name",
                     _PID_RELAUNCH_ICON: "icon"}
            for pid, name in names.items():
                value = _PROPVARIANT()
                get_value(store, ctypes.byref(_PROPERTYKEY(fmtid, pid)), ctypes.byref(value))
                read_back[name] = ctypes.wstring_at(value.pointer) if value.vt == _VT_LPWSTR and value.pointer else ""
                ctypes.windll.ole32.PropVariantClear(ctypes.byref(value))
            return read_back
        finally:
            release(store)
    except Exception:
        return None
