"""配色与样式表：白色底，粉色点缀。粉色和向日葵黄取自 assets/taffy-avatar.jpg，文字用头像里的海军蓝。"""
from .catalog import VoiceStyle, brand_style, voice_style  # noqa: F401  角色美术由 assets/characters.json 提供

BACKGROUND = "#FFF7F9"   # 窗口底色：带一点粉的白
CARD = "#FFFFFF"         # 卡片
FIELD = "#FBF2F5"        # 输入框、电平条底
RAISED = "#FDEAF0"       # 次要按钮、选中的标签页
LINE = "#F3DCE4"         # 描边、分隔线
TEXT = "#2B2F4E"         # 正文：海军蓝
MUTED = "#8C8399"        # 次要文字
DIM = "#C9C0CB"          # 不可用
SAKURA = "#F7B9CB"
SAKURA_DEEP = "#E9739B"  # 主色：按钮、选中状态
SAKURA_HOVER = "#F08DAE"
SUNFLOWER = "#F8C537"
SKY = "#7CC4F2"
DANGER = "#E8466A"
OK = "#2E9E6A"
WARN = "#C98A00"
WHITE = "#FFFFFF"
INK = "#3A2A00"          # 黄色底上的文字

FONT_FAMILY = '"Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif'

STYLESHEET = f"""
QWidget {{
    color: {TEXT};
    font-family: {FONT_FAMILY};
    font-size: 10pt;
}}
QWidget#root {{ background: {BACKGROUND}; }}
QFrame#card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 18px; }}
QFrame#divider {{ background: {LINE}; max-height: 1px; min-height: 1px; border: none; }}
QLabel {{ background: transparent; border: none; }}
QLabel#appTitle {{ font-size: 13.5pt; font-weight: 800; color: {TEXT}; }}
QLabel#version {{ color: {SAKURA_DEEP}; font-size: 9pt; font-weight: 700; }}
QLabel#lamp {{
    font-family: "Segoe UI", sans-serif; font-size: 8.5pt; font-weight: 800;
    padding: 4px 11px; border-radius: 10px; color: {WHITE}; background: {DANGER};
}}
QLabel#lamp[state="rec"] {{ color: {INK}; background: {SUNFLOWER}; }}
QLabel#pill {{
    padding: 4px 11px; border-radius: 11px; font-size: 9pt; font-weight: 600;
    background: {CARD}; border: 1px solid {LINE}; color: {MUTED};
}}
QLabel#pill[state="ok"] {{ color: {OK}; }}
QLabel#pill[state="busy"] {{ color: {WARN}; }}
QLabel#pill[state="bad"] {{ color: {DANGER}; }}
QLabel#sectionLabel {{ color: {MUTED}; font-size: 9pt; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; font-size: 9pt; }}
QLabel#muted[state="ok"] {{ color: {OK}; }}
QLabel#muted[state="warn"] {{ color: {WARN}; }}
QLabel#muted[state="error"] {{ color: {DANGER}; }}
QLabel#detail {{
    color: {MUTED}; background: {FIELD}; border-radius: 10px; padding: 7px 11px; font-size: 8.5pt;
}}
QLabel#micHint {{
    color: {TEXT}; background: #FFF4D6; border: 1px solid #F6DE9A; border-radius: 10px;
    padding: 7px 11px; font-size: 9pt;
}}
QLabel#heading {{ font-size: 13pt; font-weight: 800; }}
QLabel#body {{ color: {TEXT}; font-size: 10pt; }}
QDialog {{ background: {BACKGROUND}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 5px; border: 1px solid {LINE}; background: {FIELD};
}}
QCheckBox::indicator:checked {{ background: {SAKURA_DEEP}; border-color: {SAKURA_DEEP}; }}
QLabel#value {{ color: {TEXT}; font-size: 10pt; font-weight: 700; }}
QLabel#clock {{ font-size: 14pt; font-weight: 700; font-family: "Consolas", "Segoe UI", monospace; }}
QLabel#heroName {{ font-size: 17pt; font-weight: 800; }}
QLabel#heroCaption {{ font-family: "Segoe UI", sans-serif; font-size: 8.5pt; font-weight: 700; }}
QLabel#liveTitle {{ font-size: 15pt; font-weight: 800; }}
QLabel#notice {{
    background: #FFE9EE; color: {TEXT}; border: 1px solid #F7C4D2; border-radius: 12px;
    padding: 9px 14px; font-size: 9.5pt;
}}
QLabel#resultTag {{
    color: {MUTED}; background: {FIELD}; border-radius: 10px; padding: 8px 12px; font-size: 9.5pt;
}}
QLabel#resultTag[state="fresh"] {{ color: {OK}; }}
QLabel#resultTag[state="stale"] {{ color: {WARN}; }}
QLabel#statusBar {{ color: {MUTED}; font-size: 9pt; padding: 0 6px; }}
QLabel#statusBar[state="error"] {{ color: {DANGER}; }}
QLabel#statusBar[state="ok"] {{ color: {OK}; }}

QPushButton {{
    background: {RAISED}; border: 1px solid transparent; border-radius: 10px;
    padding: 7px 14px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {SAKURA}; }}
QPushButton:pressed {{ background: #FAD9E4; }}
QPushButton:focus {{ border-color: {SAKURA_DEEP}; }}
QPushButton:disabled {{ color: {DIM}; background: {FIELD}; }}
QPushButton#primary {{
    background: {SAKURA_DEEP}; color: {WHITE}; font-weight: 800; font-size: 11pt; padding: 10px 18px;
}}
QPushButton#primary:hover {{ background: {SAKURA_HOVER}; border-color: transparent; }}
QPushButton#primary:pressed {{ background: #D9618A; }}
QPushButton#primary:disabled {{ background: #F6D3DF; color: {WHITE}; }}
QPushButton#ghost {{ background: transparent; color: {MUTED}; padding: 5px 10px; font-size: 9pt; }}
QPushButton#ghost:hover {{ color: {SAKURA_DEEP}; border-color: transparent; }}
QPushButton#ghost:disabled {{ color: {DIM}; }}
QPushButton#tab {{
    background: transparent; color: {MUTED}; font-size: 11pt; font-weight: 700;
    padding: 8px 18px; border-radius: 12px;
}}
QPushButton#tab:hover {{ color: {TEXT}; border-color: transparent; }}
QPushButton#tab:checked {{ background: {RAISED}; color: {SAKURA_DEEP}; }}
QPushButton#segLeft, QPushButton#segRight, QPushButton#segMiddle {{
    padding: 6px 16px; font-weight: 700; background: {FIELD}; color: {MUTED}; border: 1px solid {LINE};
}}
QPushButton#segLeft {{ border-top-right-radius: 0; border-bottom-right-radius: 0; }}
QPushButton#segMiddle {{ border-radius: 0; border-left: none; }}
QPushButton#segRight {{ border-top-left-radius: 0; border-bottom-left-radius: 0; border-left: none; }}
QPushButton#segLeft:checked, QPushButton#segRight:checked, QPushButton#segMiddle:checked {{
    background: {SAKURA_DEEP}; color: {WHITE}; border-color: {SAKURA_DEEP};
}}
QPushButton#segLeft:disabled, QPushButton#segRight:disabled, QPushButton#segMiddle:disabled {{
    color: {DIM}; background: {FIELD};
}}
QPushButton#segSolo {{
    padding: 6px 16px; font-weight: 700; background: {SAKURA_DEEP}; color: {WHITE}; border-radius: 10px;
}}

QComboBox {{
    background: {FIELD}; border: 1px solid {LINE}; border-radius: 9px;
    padding: 6px 30px 6px 10px; min-height: 20px; color: {TEXT};
}}
QComboBox:hover, QComboBox:focus {{ border-color: {SAKURA}; }}
QComboBox:disabled {{ color: {DIM}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox::down-arrow {{ image: none; }}
QComboBox QAbstractItemView {{
    background: {CARD}; color: {TEXT}; border: 1px solid {LINE};
    selection-background-color: {SAKURA_DEEP}; selection-color: {WHITE}; outline: none; padding: 4px;
}}

QSlider::groove:horizontal {{ height: 6px; background: {FIELD}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {SAKURA}, stop:1 {SAKURA_DEEP}); border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0; border-radius: 8px; background: {WHITE};
    border: 2px solid {SAKURA_DEEP};
}}
QSlider::handle:horizontal:hover {{ background: {RAISED}; }}
QSlider::handle:horizontal:disabled {{ border-color: {DIM}; }}
QSlider::sub-page:horizontal:disabled {{ background: {LINE}; }}

QProgressBar {{ background: {FIELD}; border: none; border-radius: 2px; max-height: 4px; min-height: 4px; }}
QProgressBar::chunk {{ background: {SAKURA_DEEP}; border-radius: 2px; }}

QToolTip {{ background: {CARD}; color: {TEXT}; border: 1px solid {LINE}; padding: 5px 8px; }}
"""
