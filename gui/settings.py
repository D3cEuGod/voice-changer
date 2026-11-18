"""用户设置：记住上次的角色、设备、强度和开关，下次打开直接用。

保存在 %APPDATA%\\CharacterVoiceChanger\\settings.json（其他系统在 ~/.config 下）。
path 为 None 时只存在内存里，自检和测试用这种方式，不会改动用户的设置。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_DIR_NAME = "CharacterVoiceChanger"
RESTART_CHOICES = (0, 3000, 5000)
PACKET_CHOICES = (100, 120)


def default_path() -> Path:
    base = os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / ".config"
    return root / APP_DIR_NAME / "settings.json"


@dataclass
class Settings:
    voice_id: str | None = None
    strength: int = 100
    mode: str = "character"
    pitch: str = "auto2"
    mic: str | None = None          # 设备按名称记，Windows 的设备编号每次启动都可能变
    output: str | None = None
    send_to_call: bool = True       # 送到虚拟麦克风，给 Discord 等语音软件
    monitor: bool = False           # 耳返：自己也从耳机听到变声结果
    hotkey: str = "Ctrl+Alt+V"
    close_to_tray: bool = True
    onboarded: bool = False
    live_restart_packets: int = 3000
    live_packet_ms: int = 120
    path: Path | None = field(default=None, repr=False, compare=False)

    @classmethod
    def load(cls, path: Path | None) -> "Settings":
        """读取设置；文件不存在、损坏或某项取值不合法时，那一项用默认值。"""
        settings = cls(path=path)
        if path is None:
            return settings
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return settings
        if not isinstance(data, dict):
            return settings
        for item in fields(cls):
            if item.name == "path" or item.name not in data:
                continue
            value, default = data[item.name], getattr(settings, item.name)
            if default is None:
                ok = value is None or isinstance(value, str)
            elif isinstance(default, bool):
                ok = isinstance(value, bool)
            elif isinstance(default, int):
                ok = isinstance(value, int) and not isinstance(value, bool)
            else:
                ok = isinstance(value, type(default))
            if ok:
                setattr(settings, item.name, value)
        settings.strength = min(100, max(0, settings.strength))
        if settings.live_restart_packets not in RESTART_CHOICES:
            settings.live_restart_packets = 3000
        if settings.live_packet_ms not in PACKET_CHOICES:
            settings.live_packet_ms = 120
        if settings.mode not in ("character", "call"):
            settings.mode = "character"
        return settings

    def save(self) -> bool:
        """原子写入：先写临时文件再替换，写到一半断电也不会留下坏文件。"""
        if self.path is None:
            return True
        data = {k: v for k, v in asdict(self).items() if k != "path"}
        target = Path(self.path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".tmp")
            temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temp, target)
            return True
        except OSError:
            return False
