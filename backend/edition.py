"""公开版开关。公开版不带任何角色：只有用户自己导入到 models/local-rvc/ 的声线。

根目录有 PUBLIC_EDITION 文件（公开版导出脚本会放进去），或设置 VOICE_EDITION=public 时启用。
这个模块不导入任何大库，setup_models.py 和界面都可以直接读它。
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (ROOT / "PUBLIC_EDITION").exists() or os.environ.get("VOICE_EDITION", "").lower() == "public"
