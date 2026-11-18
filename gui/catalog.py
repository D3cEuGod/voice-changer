"""角色分类和每位角色的界面美术，来自 assets/characters.json。

新增角色不用改代码：头像放进 assets/，在 characters.json 的 voices 里加一条，再把 id 放进某个分类。
后端返回、但 characters.json 没有登记的角色，会归入自动生成的“其他”分类，用通用图案显示。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from .backend import ROOT

CATALOG_FILE = ROOT / "assets" / "characters.json"
OTHER = "other"


@dataclass(frozen=True)
class VoiceStyle:
    color: str      # 角色色：选中描边、开关填充、标注文字
    accent: str     # 主视觉描边
    art: str        # 没有头像文件时画的图案：mic | guitar | note
    avatar: str     # assets/ 下的头像文件名，可以为空
    chibi: str = "" # assets/ 下的 Q 版形象，可以为空


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    voices: tuple[str, ...]


FALLBACK_STYLES = (
    VoiceStyle("#5DBB8A", "#A6E3B8", "note", ""),
    VoiceStyle("#9B7BE0", "#C9B6F2", "note", ""),
    VoiceStyle("#E09A5D", "#F2C6A6", "note", ""),
)


@lru_cache(maxsize=1)
def _load() -> tuple[tuple[Category, ...], dict[str, VoiceStyle]]:
    try:
        data = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return (), {}
    categories = tuple(Category(str(c["id"]), str(c["name"]), tuple(str(v) for v in c.get("voices", ())))
                       for c in data.get("categories", ()))
    styles = {str(key): VoiceStyle(value.get("color", "#E9739B"), value.get("accent", "#F8C537"),
                                   value.get("art", "note"), value.get("avatar", ""), value.get("chibi", ""))
              for key, value in data.get("voices", {}).items()}
    return categories, styles


# 界面主色和 Q 版形象。私人版用塔菲的配色和形象；公开版没有角色素材时用中性的粉色和麦克风图案。
DEFAULT_BRAND = VoiceStyle("#E9739B", "#F8C537", "mic", "")
BRAND_VOICE = "taffy"


def brand_style() -> VoiceStyle:
    return _load()[1].get(BRAND_VOICE, DEFAULT_BRAND)


def voice_style(voice_id: str) -> VoiceStyle:
    styles = _load()[1]
    if voice_id in styles:
        return styles[voice_id]
    return FALLBACK_STYLES[sum(voice_id.encode("utf-8")) % len(FALLBACK_STYLES)]


def categories_for(voice_ids: list[str]) -> list[Category]:
    """按 characters.json 的顺序列出有可用角色的分类；没登记的角色放进最后的“其他”。"""
    available = list(dict.fromkeys(voice_ids))
    result, placed = [], set()
    for category in _load()[0]:
        members = tuple(v for v in category.voices if v in available and v not in placed)
        if members:
            result.append(Category(category.id, category.name, members))
            placed.update(members)
    rest = tuple(v for v in available if v not in placed)
    if rest:
        result.append(Category(OTHER, "我的声线" if not result else "其他", rest))
    return result
