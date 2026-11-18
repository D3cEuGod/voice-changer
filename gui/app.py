"""主窗口：左边选角色和强度，右边是实时变声与文件变声两页。只在 QApplication 创建之后导入。

1.0 起界面只放常用的东西：角色、强度、一个大开关、“发到语音软件”和“耳返”两个开关。
技术数字收进“详细信息”，快捷键、托盘和高级选项放进设置面板；上次的选择会被记住。
"""
from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMenu,
                               QProgressBar, QPushButton, QSizePolicy, QSlider, QStackedWidget, QSystemTrayIcon,
                               QVBoxLayout, QWidget)

from . import __version__, audio, catalog, hotkey, theme
from .backend import (CONTENT_TYPES, DEFAULT_URL, DEFAULT_PITCH_PRESET, ROOT, START_SCRIPT, BackendClient, BackendError,
                      BackendLauncher, stop_process_tree, PITCH_PRESETS)
from . import backend as backend_module
from .live import LiveMonitor
from .settings import Settings
from .widgets import (Card, ConsoleCombo, HeroArt, LevelMeter, LiveButton, RecButton, Segmented, ToggleSwitch,
                      VoiceCarousel, app_pixmap, divider, label, make_icon, set_state)

ASSETS = ROOT / "assets"
MAX_RESULTS = 8
MODE_NAMES = {"character": "音色优先", "call": "通话清晰"}
# PortAudio 报这些错时，多半是窗口打开后 Windows 的设备编号变了（显示器音频唤醒、装驱动、改默认设备）。
STALE_DEVICE_ERRORS = ("-9996", "-9985", "-9986", "Invalid device", "Device unavailable",
                       "Error querying device")
LIVE_DEVICE_SETTLE_MS = 250


@dataclass
class Config:
    backend_url: str = DEFAULT_URL
    autostart: bool = True
    script: Path = START_SCRIPT
    live_restart_packets: int = 3000
    live_packet_ms: int = 120
    pitch_preset: str = DEFAULT_PITCH_PRESET
    prestarted: object = None  # 界面初始化前已经拉起的后端进程（gui.backend.start_early）
    # 用户设置。默认只在内存里（path=None），自检不会读写用户的设置文件。
    settings: Settings = field(default_factory=Settings)
    # 托盘图标、全局快捷键和首次引导。正式启动时打开；自检和测试保持关闭，行为与以前一致。
    desktop: bool = False


@dataclass
class InputClip:
    serial: int
    label: str
    data: bytes           # 原样提交给 POST /api/convert 的文件字节
    content_type: str
    samples: np.ndarray   # 本地试听用的单声道样本
    rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.rate


@dataclass
class ResultClip:
    key: tuple            # (输入序号, voice_id, 强度百分比, 处理方式, 音高预设)
    wav: bytes            # 后端返回的 WAV，另存时原样写出
    samples: np.ndarray
    rate: int
    voice_name: str
    input_label: str
    elapsed: float
    created: float = field(default_factory=time.time)


class Bridge(QObject):
    """工作线程通过这些信号把结果送回界面线程。"""

    backend_progress = Signal(str)
    backend_stage = Signal(object, object, object, str)  # 启动器、阶段、模型加载进度、说明
    backend_ready = Signal(object, object, bool)
    backend_failed = Signal(object, str)
    voices_loaded = Signal(object)
    voices_failed = Signal(str)
    health_result = Signal(bool, str)
    convert_done = Signal(object, object, float)
    convert_failed = Signal(object, str)
    live_connected = Signal(object, object)
    live_stats = Signal(object, object)
    live_stopped = Signal(object, str, str)


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


class MainWindow(QWidget):
    startup_stage = Signal(object, object, str)  # 给加载页：阶段、进度、说明
    startup_finished = Signal(bool, str)         # 给加载页：后端和角色列表就绪，或失败原因

    def __init__(self, config: Config | None = None):
        super().__init__()
        self.config = config or Config()
        self.client = BackendClient(self.config.backend_url)
        self.bridge = Bridge()

        self.backend_ok = False
        self.reference_engine = False
        self.reference_voices = set()
        self.voice_models = {}
        self.live_packet_options = {120, 200}
        self.pitch_supported = False
        self.max_seconds = 60.0
        self.max_bytes = 24 * 1024 * 1024
        self.voices: list[dict] = []
        self.voice_id: str | None = None
        self.voice_names: dict[str, str] = {}          # 后端返回的角色 id → 中文名
        self.categories: list[catalog.Category] = []
        self.category_id: str | None = None           # 角色转盘当前显示的分类
        self._last_voice: dict[str, str] = {}         # 每个分类上次选中的角色
        self.settings = self.config.settings
        self.strength_pct = self.settings.strength
        self.mic_advisor = audio.MicAdvisor()
        self.discord_pair = None
        self.tray: QSystemTrayIcon | None = None
        self.hotkey: hotkey.GlobalHotkey | None = None
        self.hotkey_error = ""
        self._quitting = False
        self._tray_hint_shown = False
        self.input: InputClip | None = None
        self.results: dict[tuple, ResultClip] = {}
        self.converting_key: tuple | None = None
        self.ab = "input"
        self.recorder: audio.Recorder | None = None
        self.player = audio.ClipPlayer()
        self.playing_tag: tuple | None = None
        self.live: LiveMonitor | None = None
        self.owned_backend = None

        self._serial = 0
        self._launcher: BackendLauncher | None = None
        self._health_inflight = False
        self._health_failures = 0
        self._convert_started = 0.0
        self._rec_started = 0.0
        self._live_restart = False
        self._live_resume_monitor = None
        self._live_stats: dict = {}
        self._live_auto_refreshes = 0
        self._closing = False
        self._startup_reported = False
        self._last_dir = str(ROOT)
        self._avatars: dict[str, QPixmap | None] = {}
        self._avatar = self._pixmap(catalog.BRAND_VOICE)

        self.setObjectName("root")
        self.setWindowTitle("角色变声器")
        self.setWindowIcon(QIcon(self._avatar if self._avatar is not None else app_pixmap()))
        self._build()
        self._wire()
        self._populate_devices(announce=False)
        self._show_voice(None)
        self._refresh()

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(40)
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start()
        self._health_timer = QTimer(self)
        self._health_timer.setInterval(3000)
        self._health_timer.timeout.connect(self._poll_health)
        self._health_timer.start()
        self._strength_timer = QTimer(self)
        self._strength_timer.setSingleShot(True)
        self._strength_timer.setInterval(450)
        self._strength_timer.timeout.connect(self._restart_live)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self.save_settings)
        if self.config.desktop:
            self._setup_desktop()

        QTimer.singleShot(0, self.connect_backend)

    # ---- 界面搭建 -------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 10)
        root.setSpacing(12)

        top = QHBoxLayout()
        top.setSpacing(10)
        chibi = QPixmap(str(ASSETS / catalog.brand_style().chibi)) if catalog.brand_style().chibi else QPixmap()
        ratio = self.devicePixelRatioF() or 1.0
        self.mascot = QLabel()
        if not chibi.isNull():
            scaled = chibi.scaled(int(40 * ratio), int(40 * ratio), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.mascot.setToolTip("Q 版永雏塔菲")
        else:
            scaled = app_pixmap(int(30 * ratio))  # 公开版没有 Q 版形象，用程序图标
        scaled.setDevicePixelRatio(ratio)
        self.mascot.setPixmap(scaled)
        top.addWidget(self.mascot, 0, Qt.AlignVCenter)
        top.addWidget(label("角色变声器", "appTitle"))
        top.addWidget(label(f"v{__version__}", "version"), 0, Qt.AlignBottom)
        top.addStretch(1)
        self.lamp = label("STANDBY", "lamp")
        self.lamp.hide()
        top.addWidget(self.lamp, 0, Qt.AlignVCenter)
        self.backend_pill = label("● 引擎检查中", "pill")
        self.backend_pill.setToolTip("本机变声引擎的状态")
        top.addWidget(self.backend_pill, 0, Qt.AlignVCenter)
        self.reconnect_button = QPushButton("重新连接")
        self.reconnect_button.setObjectName("ghost")
        self.reconnect_button.setToolTip("检查本机变声引擎；没有运行时启动它")
        self.reconnect_button.hide()  # 只在引擎没连上时出现
        top.addWidget(self.reconnect_button, 0, Qt.AlignVCenter)
        self.settings_button = QPushButton("设置")
        self.settings_button.setObjectName("ghost")
        self.settings_button.setIcon(make_icon("gear", theme.MUTED, 14))
        self.settings_button.setIconSize(QSize(14, 14))
        self.settings_button.setToolTip("快捷键、托盘、高级选项和关于")
        for ghost in (self.reconnect_button, self.settings_button):
            ghost.setFocusPolicy(Qt.TabFocus)  # 鼠标点过后不留焦点框
        top.addWidget(self.settings_button, 0, Qt.AlignVCenter)
        root.addLayout(top)

        self.notice = QLabel()
        self.notice.setObjectName("notice")
        self.notice.setWordWrap(True)
        self.notice.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.notice.hide()
        root.addWidget(self.notice)

        columns = QHBoxLayout()
        columns.setSpacing(14)
        columns.addWidget(self._build_voice(), 0)
        columns.addWidget(self._build_work(), 1)
        root.addLayout(columns, 1)

        self.status_label = QLabel("")
        self.status_label.setObjectName("statusBar")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.status_label)

    def _build_voice(self) -> QWidget:
        """左栏：当前角色、分类和角色转盘、音色强度与处理方式。"""
        card = Card()
        card.setFixedWidth(336)
        card.body.setSpacing(10)
        self.hero = HeroArt(250)
        self.hero.setMinimumSize(150, 150)
        card.body.addWidget(self.hero, 1)
        names = QVBoxLayout()
        names.setSpacing(0)
        self.hero_name = label("…", "heroName")
        self.hero_name.setAlignment(Qt.AlignCenter)
        names.addWidget(self.hero_name)
        self.hero_caption = label("正在读取角色列表", "heroCaption")
        self.hero_caption.setAlignment(Qt.AlignCenter)
        names.addWidget(self.hero_caption)
        card.body.addLayout(names)
        self.category_bar = Segmented()
        self.category_bar.setAccessibleName("角色分类")
        card.body.addWidget(self.category_bar, 0, Qt.AlignHCenter)
        self.carousel = VoiceCarousel()
        self.carousel.setToolTip("点两侧的三角或旁边的角色切换；也可以用键盘左右键")
        card.body.addWidget(self.carousel)
        self.voices_placeholder = label("连接变声引擎后，这里显示可用的声线。")
        self.voices_placeholder.setWordWrap(True)
        self.voices_placeholder.setAlignment(Qt.AlignCenter)
        card.body.addWidget(self.voices_placeholder)
        self.import_voice_button = QPushButton("＋ 导入声线")
        self.import_voice_button.setObjectName("ghost")
        self.import_voice_button.setFocusPolicy(Qt.TabFocus)
        self.import_voice_button.setToolTip("把 RVC v2 声线包（.pth 模型 + .index 索引）装进变声器")
        card.body.addWidget(self.import_voice_button, 0, Qt.AlignHCenter)
        card.body.addWidget(divider())
        strength = QHBoxLayout()
        strength.addWidget(label("音色强度", "sectionLabel"))
        strength.addStretch(1)
        self.strength_value = label("100%", "value")
        strength.addWidget(self.strength_value)
        card.body.addLayout(strength)
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setSingleStep(5)
        self.strength_slider.setPageStep(10)
        self.strength_slider.setValue(self.strength_pct)
        self.strength_slider.setMinimumHeight(22)
        self.strength_slider.setAccessibleName("音色强度")
        self.strength_slider.setToolTip("0% 是原声，100% 最接近角色")
        card.body.addWidget(self.strength_slider)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        mode_row.addWidget(label("风格", "sectionLabel"))
        mode_row.addStretch(1)
        self.voice_mode = Segmented()
        for mode, name in MODE_NAMES.items():
            self.voice_mode.addItem(name, mode)
        self.voice_mode.setCurrentIndex(max(0, self.voice_mode.findData(self.settings.mode)))
        self.voice_mode.setAccessibleName("处理方式")
        self.voice_mode.setToolTip("音色优先：更像角色。通话清晰：咬字更清楚，适合语速快或环境吵的时候。")
        mode_row.addWidget(self.voice_mode)
        card.body.addLayout(mode_row)
        pitch_row = QHBoxLayout()
        pitch_row.setSpacing(8)
        pitch_row.addWidget(label("音高", "sectionLabel"))
        self.pitch_combo = ConsoleCombo()
        for preset, (name, _, _) in PITCH_PRESETS.items():
            self.pitch_combo.addItem(name, preset)
        self.pitch_combo.setCurrentIndex(max(0, self.pitch_combo.findData(self.config.pitch_preset)))
        self.pitch_combo.setMinimumContentsLength(6)
        self.pitch_combo.setAccessibleName("音高")
        self.pitch_combo.setToolTip("推荐“自动 +2”：先按角色音域自动调整，再升 2 个半音。声音太尖选“模型默认”，不够高选“自动 +4”。")
        pitch_row.addWidget(self.pitch_combo, 1)
        self._any_reference = any((ASSETS / "references").glob("*.wav"))
        self.reference_button = QPushButton("角色原声")
        self.reference_button.setIconSize(QSize(11, 11))
        self.reference_button.setToolTip("播放当前角色的原始配音 / 直播录音，用来对照变声效果")
        pitch_row.addWidget(self.reference_button)
        card.body.addLayout(pitch_row)
        return card

    def _build_work(self) -> QWidget:
        """右栏：实时变声和文件变声两页，底部是设备。"""
        card = Card()
        tabs = QHBoxLayout()
        tabs.setSpacing(4)
        self.tab_group = QButtonGroup(self)
        self.tab_group.setExclusive(True)
        self.tab_buttons = {}
        for index, (name, text) in enumerate((("live", "实时变声"), ("file", "文件变声"))):
            button = QPushButton(text)
            button.setObjectName("tab")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            self.tab_group.addButton(button, index)
            self.tab_buttons[name] = button
            tabs.addWidget(button)
        tabs.addStretch(1)
        card.body.addLayout(tabs)
        self.pages = QStackedWidget()
        self.live_page = self._build_live()
        self.file_page = self._build_file()
        self.pages.addWidget(self.live_page)
        self.pages.addWidget(self.file_page)
        card.body.addWidget(self.pages, 1)
        self.tab_buttons["live"].setChecked(True)
        card.body.addWidget(divider())
        card.body.addLayout(self._build_devices())
        return card

    def _build_live(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addStretch(2)
        self.live_switch = LiveButton()
        layout.addWidget(self.live_switch, 0, Qt.AlignHCenter)
        self.live_title = label("点一下开始变声", "liveTitle")
        self.live_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.live_title)
        switches = QHBoxLayout()
        switches.setSpacing(28)
        switches.addStretch(1)
        self.send_toggle = ToggleSwitch("发到语音软件")
        self.send_toggle.setToolTip("把变声后的声音送进虚拟麦克风，Discord、QQ、YY、游戏语音都能用")
        self.send_toggle.setChecked(self.settings.send_to_call)
        switches.addWidget(self.send_toggle)
        self.monitor_toggle = ToggleSwitch("耳返（自己听）")
        self.monitor_toggle.setToolTip("同时从下面选的耳机里听到变声后的声音。请戴耳机，外放会被麦克风收回去。")
        self.monitor_toggle.setChecked(self.settings.monitor)
        switches.addWidget(self.monitor_toggle)
        switches.addStretch(1)
        layout.addLayout(switches)
        self.route_hint = label("")
        self.route_hint.setWordWrap(True)
        self.route_hint.setAlignment(Qt.AlignCenter)
        self.route_hint.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.route_hint)
        layout.addStretch(1)
        meters = QGridLayout()
        meters.setHorizontalSpacing(12)
        meters.setVerticalSpacing(8)
        self.live_in_meter = LevelMeter()
        self.live_out_meter = LevelMeter()
        meters.addWidget(label("麦克风", "sectionLabel"), 0, 0)
        meters.addWidget(self.live_in_meter, 0, 1)
        meters.addWidget(label("变声后", "sectionLabel"), 1, 0)
        meters.addWidget(self.live_out_meter, 1, 1)
        meters.setColumnStretch(1, 1)
        layout.addLayout(meters)
        self.mic_hint = label("")
        self.mic_hint.setObjectName("micHint")
        self.mic_hint.setWordWrap(True)
        self.mic_hint.hide()
        layout.addWidget(self.mic_hint)
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self.live_status = label("")
        self.live_status.setWordWrap(True)
        self.live_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_row.addWidget(self.live_status, 1)
        self.detail_button = QPushButton("详细信息")
        self.detail_button.setObjectName("ghost")
        self.detail_button.setCheckable(True)
        self.detail_button.setToolTip("显示设备、延迟和每包处理时间等技术数字")
        self.detail_button.hide()
        status_row.addWidget(self.detail_button, 0, Qt.AlignTop)
        layout.addLayout(status_row)
        self.live_detail = label("")
        self.live_detail.setObjectName("detail")
        self.live_detail.setWordWrap(True)
        self.live_detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.live_detail.hide()
        layout.addWidget(self.live_detail)
        layout.addStretch(1)
        return page

    def _build_file(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addStretch(1)
        row = QHBoxLayout()
        row.setSpacing(14)
        self.rec_button = RecButton()
        self.rec_button.setToolTip("开始录音，再点一次结束")
        row.addWidget(self.rec_button, 0, Qt.AlignVCenter)
        side = QVBoxLayout()
        side.setSpacing(6)
        self.rec_clock = label("0:00.0", "clock")
        side.addWidget(self.rec_clock)
        self.rec_meter = LevelMeter()
        side.addWidget(self.rec_meter)
        self.rec_hint = label("点红色按钮开始录音")
        self.rec_hint.setWordWrap(True)
        side.addWidget(self.rec_hint)
        row.addLayout(side, 1)
        self.import_button = QPushButton("导入音频…")
        self.import_button.setToolTip("支持 WAV、FLAC、MP3、OGG")
        row.addWidget(self.import_button, 0, Qt.AlignVCenter)
        layout.addLayout(row)
        self.input_label = label("当前输入：还没有")
        self.input_label.setWordWrap(True)
        layout.addWidget(self.input_label)
        self.convert_button = QPushButton("开始变声")
        self.convert_button.setObjectName("primary")
        self.convert_button.setMinimumHeight(44)
        layout.addWidget(self.convert_button)
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setTextVisible(False)
        keep = self.busy_bar.sizePolicy()
        keep.setRetainSizeWhenHidden(True)
        self.busy_bar.setSizePolicy(keep)
        self.busy_bar.hide()
        layout.addWidget(self.busy_bar)
        layout.addSpacing(6)
        layout.addWidget(divider())
        layout.addSpacing(6)

        play = QHBoxLayout()
        play.setSpacing(12)
        seg = QHBoxLayout()
        seg.setSpacing(0)
        self.seg_input = QPushButton("原声")
        self.seg_input.setObjectName("segLeft")
        self.seg_input.setCheckable(True)
        self.seg_input.setChecked(True)
        self.seg_result = QPushButton("变声结果")
        self.seg_result.setObjectName("segRight")
        self.seg_result.setCheckable(True)
        self.seg_group = QButtonGroup(self)
        self.seg_group.setExclusive(True)
        self.seg_group.addButton(self.seg_input)
        self.seg_group.addButton(self.seg_result)
        seg.addWidget(self.seg_input)
        seg.addWidget(self.seg_result)
        play.addLayout(seg)
        self.play_button = QPushButton("播放")
        self.play_button.setIconSize(QSize(13, 13))
        self.play_button.setMinimumWidth(84)
        play.addWidget(self.play_button)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.setMinimumHeight(22)
        self.seek_slider.setAccessibleName("播放进度")
        play.addWidget(self.seek_slider, 1)
        self.time_label = label("0:00.0 / 0:00.0")
        self.time_label.setMinimumWidth(100)
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        play.addWidget(self.time_label)
        layout.addLayout(play)
        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        self.result_tag = QLabel()
        self.result_tag.setObjectName("resultTag")
        self.result_tag.setWordWrap(True)
        self.result_tag.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bottom.addWidget(self.result_tag, 1)
        self.save_button = QPushButton("另存为 WAV…")
        self.save_button.setMinimumHeight(36)
        bottom.addWidget(self.save_button, 0, Qt.AlignVCenter)
        layout.addLayout(bottom)
        layout.addStretch(1)
        return page

    def _build_devices(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        self.mic_combo = ConsoleCombo()
        self.mic_combo.setAccessibleName("麦克风")
        self.out_combo = ConsoleCombo()
        self.out_combo.setAccessibleName("输出设备")
        self.rescan_button = QPushButton("刷新")
        self.rescan_button.setObjectName("ghost")
        self.rescan_button.setIconSize(QSize(13, 13))
        self.rescan_button.setIcon(make_icon("refresh", theme.MUTED, 13))
        self.rescan_button.setToolTip("插拔耳机或麦克风后点这里重新扫描")
        # 一行一个设备，下拉框够宽，系统报告的完整设备名不被截断
        grid.setVerticalSpacing(8)
        output_label = label("耳机", "sectionLabel")
        output_label.setToolTip("耳返、试听和角色原声从这里播放。发到语音软件的声音不经过它。")
        grid.addWidget(label("麦克风", "sectionLabel"), 0, 0)
        grid.addWidget(self.mic_combo, 0, 1)
        grid.addWidget(output_label, 1, 0)
        grid.addWidget(self.out_combo, 1, 1)
        grid.addWidget(self.rescan_button, 1, 2)
        grid.setColumnStretch(1, 1)
        return grid

    def show_tab(self, name: str) -> None:
        """切到“实时变声”（live）或“文件变声”（file）页。"""
        self.tab_buttons[name].setChecked(True)
        self.pages.setCurrentWidget(self.live_page if name == "live" else self.file_page)

    def reveal(self, widget: QWidget) -> None:
        """让某个控件所在的页显示出来。"""
        for name, page in (("live", self.live_page), ("file", self.file_page)):
            if page.isAncestorOf(widget):
                self.show_tab(name)

    def _wire(self) -> None:
        b = self.bridge
        b.backend_progress.connect(self._on_backend_progress)
        b.backend_stage.connect(self._on_backend_stage)
        b.backend_ready.connect(self._on_backend_ready)
        b.backend_failed.connect(self._on_backend_failed)
        b.voices_loaded.connect(self._on_voices_loaded)
        b.voices_failed.connect(self._on_voices_failed)
        b.health_result.connect(self._on_health)
        b.convert_done.connect(self._on_convert_done)
        b.convert_failed.connect(self._on_convert_failed)
        b.live_connected.connect(self._on_live_connected)
        b.live_stats.connect(self._on_live_stats)
        b.live_stopped.connect(self._on_live_stopped)

        self.reconnect_button.clicked.connect(self.connect_backend)
        self.tab_buttons["live"].clicked.connect(lambda: self.show_tab("live"))
        self.tab_buttons["file"].clicked.connect(lambda: self.show_tab("file"))
        self.reference_button.clicked.connect(lambda: self.toggle_preview("reference"))
        self.carousel.chosen.connect(self.select_voice)
        self.category_bar.currentIndexChanged.connect(self._on_category_changed)
        self.mic_combo.currentIndexChanged.connect(self._on_device_changed)
        self.out_combo.currentIndexChanged.connect(self._on_device_changed)
        self.send_toggle.toggled.connect(self._route_changed)
        self.monitor_toggle.toggled.connect(self._route_changed)
        self.settings_button.clicked.connect(self.open_settings)
        self.import_voice_button.clicked.connect(self.open_voice_import)
        self.detail_button.toggled.connect(self.live_detail.setVisible)
        self.rescan_button.clicked.connect(self.rescan_devices)
        self.rec_button.toggled.connect(self._on_rec_toggled)
        self.import_button.clicked.connect(self.import_dialog)
        self.strength_slider.valueChanged.connect(self._on_strength_changed)
        self.voice_mode.currentIndexChanged.connect(self._on_voice_mode_changed)
        self.pitch_combo.currentIndexChanged.connect(self._on_voice_mode_changed)
        self.convert_button.clicked.connect(self.convert)
        self.seg_input.clicked.connect(lambda: self.set_ab("input"))
        self.seg_result.clicked.connect(lambda: self.set_ab("result"))
        self.play_button.clicked.connect(self.toggle_play)
        self.seek_slider.sliderReleased.connect(self._on_seek)
        self.save_button.clicked.connect(self.save_dialog)
        self.live_switch.toggled.connect(self._on_live_toggled)
        for signal in (self.strength_slider.valueChanged, self.voice_mode.currentIndexChanged,
                       self.pitch_combo.currentIndexChanged, self.mic_combo.currentIndexChanged,
                       self.out_combo.currentIndexChanged, self.send_toggle.toggled, self.monitor_toggle.toggled):
            signal.connect(self._schedule_save)

    # ---- 设置、托盘与快捷键 ----------------------------------------------------
    def _schedule_save(self, *_) -> None:
        self._save_timer.start()

    def save_settings(self) -> None:
        """把当前选择写回设置。设备按名称记；没找到设备时保留原来记住的名称。"""
        s = self.settings
        if self.voice_id is not None:
            s.voice_id = self.voice_id
        s.strength = self.strength_pct
        s.mode = self.current_mode()
        s.pitch = self.pitch_combo.currentData()
        for combo, key in ((self.mic_combo, "mic"), (self.out_combo, "output")):
            if combo.currentData() is not None:
                setattr(s, key, combo.currentData().name)
        s.send_to_call = self.send_toggle.isChecked()
        s.monitor = self.monitor_toggle.isChecked()
        s.save()

    def _setup_desktop(self) -> None:
        """托盘图标和全局快捷键。只在正式启动时打开。"""
        icon = self.windowIcon()
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray = QSystemTrayIcon(icon, self)
            self.tray.setToolTip("角色变声器")
            menu = QMenu()
            self.tray_show = QAction("打开主窗口", menu)
            self.tray_show.triggered.connect(self.bring_to_front)
            self.tray_toggle = QAction("开始变声", menu)
            self.tray_toggle.triggered.connect(self.toggle_live)
            self.tray_quit = QAction("退出", menu)
            self.tray_quit.triggered.connect(self.quit_app)
            for action in (self.tray_show, self.tray_toggle):
                menu.addAction(action)
            menu.addSeparator()
            menu.addAction(self.tray_quit)
            self.tray_menu = menu  # 托盘不持有菜单的所有权，留一个引用
            self.tray.setContextMenu(menu)
            self.tray.activated.connect(lambda reason: reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick)
                                        and self.bring_to_front())
            self.tray.show()
            QApplication.instance().setQuitOnLastWindowClosed(False)
            # 注销或关机时 Qt 直接退出事件循环，也要停掉由本程序启动的后端
            QApplication.instance().aboutToQuit.connect(self.quit_app)
        self.hotkey = hotkey.GlobalHotkey(self)
        self.hotkey.activated.connect(self.toggle_live)
        self.apply_hotkey(self.settings.hotkey)

    def apply_hotkey(self, name: str) -> str:
        """注册全局快捷键，返回出错原因（成功时为空）。"""
        if self.hotkey is None:
            return ""
        self.hotkey_error = self.hotkey.register(name) or ""
        if self.hotkey_error and name:
            self.set_status(self.hotkey_error, "error")
        return self.hotkey_error

    def hotkey_text(self) -> str:
        return self.hotkey.current if self.hotkey is not None else ""

    def bring_to_front(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def toggle_live(self) -> None:
        """快捷键和托盘菜单用：和点大圆键一样。开关不可用时说明原因。"""
        if self.live_switch.isEnabled():
            self.live_switch.click()
        else:
            self.set_status(f"现在不能开始变声：{self.live_switch.toolTip()}", "error")
            if self.tray is not None and not self.isVisible():
                self.tray.showMessage("角色变声器", f"现在不能开始变声：{self.live_switch.toolTip()}",
                                      QSystemTrayIcon.Warning, 3000)

    def open_settings(self) -> None:
        from .dialogs import SettingsDialog
        SettingsDialog(self).exec()

    def open_voice_import(self) -> None:
        from .dialogs import VoiceImportDialog
        VoiceImportDialog(self).exec()

    def reload_backend(self, select: str | None = None) -> bool:
        """导入声线后重启由本程序启动的引擎，让它读到新声线。不是本程序启动的引擎时返回 False。"""
        if select:
            self.settings.voice_id = select
            self.voice_id = None
            backend_module.WARM_FIRST = select  # 重启后先加载新声线
        if self.owned_backend is None:
            return False
        if self.live is not None:
            self.live_switch.setChecked(False)
            self.live.request_stop(abort=True)
            self.live.join(3.0)
        stop_process_tree(self.owned_backend)
        self.owned_backend = None
        self.voices, self.voice_names = [], {}
        self._set_backend(False, "正在重新启动变声引擎以载入新声线…")
        self.connect_backend()
        return True

    def maybe_onboard(self) -> None:
        """第一次打开时显示新手引导。只在正式启动时调用。"""
        if self.config.desktop and not self.settings.onboarded:
            QTimer.singleShot(300, self.run_onboarding)

    def run_onboarding(self) -> None:
        from .dialogs import OnboardingDialog
        if self.live is not None or self.recorder is not None:
            return
        OnboardingDialog(self).exec()
        self.settings.onboarded = True
        self.save_settings()

    def quit_app(self) -> None:
        """真正退出（托盘菜单“退出”、加载页“退出”）。"""
        self._quitting = True
        if not self._closing:
            self.close()

    # ---- 通用 -----------------------------------------------------------
    def set_status(self, text: str, state: str = "") -> None:
        self.status_label.setText(text)
        set_state(self.status_label, state)

    def _voice_name(self, voice_id: str | None) -> str:
        return next((v["name"] for v in self.voices if v["id"] == voice_id), voice_id or "")

    def _pixmap(self, voice_id: str) -> QPixmap | None:
        """角色头像；assets/ 里没有对应文件时返回 None，界面改画图案。"""
        if voice_id not in self._avatars:
            name = theme.voice_style(voice_id).avatar
            pixmap = QPixmap(str(ASSETS / name)) if name else QPixmap()
            self._avatars[voice_id] = None if pixmap.isNull() else pixmap
        return self._avatars[voice_id]

    def current_key(self) -> tuple | None:
        if self.input is None or self.voice_id is None:
            return None
        return (self.input.serial, self.voice_id, self.strength_pct, self.current_mode(), self.current_pitch())

    def current_mode(self) -> str:
        return self.voice_mode.currentData()

    def current_pitch(self) -> str:
        if not self.pitch_supported or self.voice_id in self.reference_voices:
            return "model"
        return self.pitch_combo.currentData()

    def current_result(self) -> ResultClip | None:
        key = self.current_key()
        return self.results.get(key) if key else None

    @staticmethod
    def _device(combo) -> audio.Device | None:
        return combo.currentData()

    def _refresh(self) -> None:
        """按当前状态统一设置各控件是否可用，以及结果与当前设置的对应关系。"""
        result = self.current_result()
        recording = self.recorder is not None
        live_on = self.live is not None or self._live_resume_monitor is not None
        converting = self.converting_key is not None
        has_mic = self._device(self.mic_combo) is not None
        has_out = self._device(self.out_combo) is not None

        self.rec_button.setEnabled(recording or (has_mic and not live_on))
        self.rec_button.setToolTip("开始录音，再点一次结束" if has_mic and not live_on else
                                   "监听占用着麦克风，先关闭监听" if live_on else "没有找到麦克风")
        self.import_button.setEnabled(not recording)
        self.mic_combo.setEnabled(has_mic and not recording)
        self.out_combo.setEnabled(has_out)
        self.rescan_button.setEnabled(not recording and not live_on)
        if not recording:
            self.rec_hint.setText("点红色按钮开始录音" if has_mic else "没有找到麦克风，可以导入音频文件")

        if self.input is None:
            self.input_label.setText("当前输入：还没有")
        else:
            clip = self.input
            self.input_label.setText(f"当前输入：{clip.label} · {clip.seconds:.2f} 秒 · {clip.rate} Hz")

        can_convert = (self.backend_ok and self.input is not None and self.voice_id is not None
                       and not converting and not live_on and not recording)
        self.convert_button.setEnabled(can_convert)
        self.convert_button.setText("变声中…" if converting else "开始变声")
        self.convert_button.setToolTip(
            "" if can_convert else
            "监听占用着同一个引擎，关闭监听后才能转换文件" if live_on else
            "正在转换" if converting else
            "后端未连接" if not self.backend_ok else
            "先结束录音" if recording else
            "先录音或导入一段音频" if self.input is None else
            "还没有声线，先导入一个声线包" if not self.voice_names else "先选择角色")
        self.busy_bar.setVisible(converting)

        if self.ab == "result" and result is None:
            if self.playing_tag is not None and self.playing_tag[0] == "result":
                self.stop_playback()
            self.ab = "input"
        self.seg_input.setChecked(self.ab == "input")
        self.seg_result.setChecked(self.ab == "result")
        self.seg_input.setEnabled(self.input is not None)
        self.seg_result.setEnabled(result is not None)
        self.seg_result.setToolTip("" if result is not None else "当前输入、角色和强度还没有对应的结果")
        playing_ab = self.playing_tag is not None and self.playing_tag[0] in ("input", "result")
        self.play_button.setEnabled(playing_ab or (has_out and self.input is not None))
        self.play_button.setText("停止" if playing_ab else "播放")
        self.play_button.setIcon(make_icon("stop" if playing_ab else "play", theme.TEXT, 14))
        self.seek_slider.setEnabled(self.input is not None)
        self.save_button.setEnabled(result is not None)
        self.save_button.setToolTip("" if result is not None else "只有与当前输入、角色、强度对应的结果才能保存")
        self._update_result_tag(result)

        sending, monitoring = self.send_toggle.isChecked(), self.monitor_toggle.isChecked()
        route_ok = (sending and bool(self.discord_pair)) or (monitoring and has_out)
        self.live_switch.setEnabled(live_on or (self.backend_ok and has_mic and route_ok and not converting
                                                and not recording and self.voice_id is not None))
        self.live_switch.setToolTip(
            "" if self.live_switch.isEnabled() else
            "正在转换文件，完成后才能变声" if converting else
            "先结束录音" if recording else
            "变声引擎未连接" if not self.backend_ok else
            ("还没有声线，先导入一个声线包" if not self.voice_names else "先选择角色") if self.voice_id is None else
            "没有找到麦克风" if not has_mic else
            "“发到语音软件”和“耳返”至少打开一个" if not sending and not monitoring else
            "没有找到虚拟麦克风，先安装 VB-CABLE，或者打开耳返试听" if sending and not monitoring else
            "没有找到耳机或音箱")
        if not live_on:
            self.live_title.setText("点一下开始变声" if self.live_switch.isEnabled() else
                                    self.live_switch.toolTip())
        self.hero.set_on_air(live_on)
        if self.tray is not None:
            self.tray_toggle.setText("停止变声" if live_on else "开始变声")
            self.tray.setToolTip(f"角色变声器 · {'正在变声' if live_on else '未在变声'}"
                                 + (f"\n快捷键 {self.hotkey_text()} 开关变声" if self.hotkey_text() else ""))
        self.lamp.setText("ON AIR" if live_on else "REC" if recording else "STANDBY")
        set_state(self.lamp, "live" if live_on else "rec" if recording else "")
        self.lamp.setVisible(live_on or recording)

    def _update_result_tag(self, result: ResultClip | None) -> None:
        key = self.current_key()
        if result is not None:
            text = (f"当前结果：{result.voice_name} · 强度 {result.key[2]}% · {MODE_NAMES[result.key[3]]} · {PITCH_PRESETS[result.key[4]][0]} · 来源「{result.input_label}」"
                    f" · {len(result.samples) / result.rate:.2f} 秒，可试听和保存")
            state = "fresh"
        elif self.input is None:
            text, state = "先录音或导入一段音频，再点“开始变声”。", ""
        elif self.converting_key is not None and self.converting_key == key:
            text, state = f"正在生成：{self._voice_name(self.voice_id)} · 强度 {self.strength_pct}%…", ""
        else:
            others = [r for r in self.results.values() if r.key[0] == self.input.serial]
            if others:
                listed = "、".join(f"{r.voice_name} {r.key[2]}% {MODE_NAMES[r.key[3]]} {PITCH_PRESETS[r.key[4]][0]}"
                                  for r in sorted(others, key=lambda r: r.created))
                text = (f"当前设置（{self._voice_name(self.voice_id)} · 强度 {self.strength_pct}%）还没有结果。"
                        f"已有结果属于：{listed}。改回对应设置可继续试听和保存，或重新变声。")
                state = "stale"
            else:
                text, state = "这段输入还没有结果，点“开始变声”。", ""
        self.result_tag.setText(text)
        set_state(self.result_tag, state)

    # ---- 后端 -----------------------------------------------------------
    def connect_backend(self) -> None:
        if self._launcher is not None or self._closing:
            return
        self.notice.hide()
        self.backend_pill.setText("● 引擎检查中")
        set_state(self.backend_pill, "busy")
        self.reconnect_button.setEnabled(False)
        launcher = BackendLauncher(
            self.client,
            on_progress=lambda text: self._emit(self.bridge.backend_progress, text),
            on_ready=lambda health, started: self._emit(self.bridge.backend_ready, launcher, health, started),
            on_failed=lambda reason: self._emit(self.bridge.backend_failed, launcher, reason),
            on_stage=lambda stage, fraction, text: self._emit(self.bridge.backend_stage, launcher, stage,
                                                               fraction, text),
            autostart=self.config.autostart, script=self.config.script, process=self.config.prestarted)
        self.config.prestarted = None  # 只用一次；重试时由启动器自己判断要不要再拉起
        self._launcher = launcher
        launcher.start()

    def _on_backend_progress(self, text: str) -> None:
        if self._launcher is not None:
            self.set_status(text)
            if "启动" in text:
                self.backend_pill.setText("● 引擎启动中")

    def _on_backend_stage(self, launcher, stage, fraction, text: str) -> None:
        if launcher is self._launcher:
            self.startup_stage.emit(stage, fraction, text)

    def _report_startup(self, ok: bool, message: str = "") -> None:
        """第一次就绪时告诉加载页；失败可以重试，所以失败不算报告过。"""
        if self._startup_reported:
            return
        self._startup_reported = ok
        self.startup_finished.emit(ok, message)

    def _on_backend_ready(self, launcher, health: dict, started: bool) -> None:
        if launcher is not self._launcher:
            return
        self._launcher = None
        self.reconnect_button.setEnabled(True)
        if started:
            self.owned_backend = launcher.process
        self.max_seconds = float(health.get("max_seconds", self.max_seconds))
        self.max_bytes = int(health.get("max_bytes", self.max_bytes))
        self.reference_engine = bool(health.get("reference_engine"))
        self.reference_voices = set(health.get("reference_voices", ("tomori", "taffy") if self.reference_engine else ()))
        self.voice_models = health.get("voice_models", {})
        self.live_packet_options = set(health.get("live_packet_ms", (120, 200)))
        self.pitch_supported = bool(health.get("pitch_control"))
        if not self.pitch_supported:
            self.pitch_combo.setCurrentIndex(self.pitch_combo.findData("model"))
        self._set_backend(True)
        if health.get("reference_error"):
            self.notice.setText(f"音色优先暂时使用兼容引擎：{health['reference_error']}")
            self.notice.show()
        how = "由本程序启动（关闭窗口时一并停止）" if started else "复用已在运行的后端"
        self.set_status("变声引擎已就绪。", "ok")
        self.status_label.setToolTip(f"{self.client.base_url} · {health.get('engine', '')} · {how}")
        self._load_voices()

    def _on_backend_failed(self, launcher, reason: str) -> None:
        if launcher is not self._launcher:
            return
        self._launcher = None
        self.reconnect_button.setEnabled(True)
        stop_process_tree(launcher.process)
        self._set_backend(False, reason)
        self.set_status("后端没有就绪，转换和监听暂时不可用。", "error")
        self._report_startup(False, reason)

    def _set_backend(self, ok: bool, reason: str = "") -> None:
        self.backend_ok = ok
        self._health_failures = 0
        self.backend_pill.setText("● 引擎就绪" if ok else "● 引擎未连接")
        set_state(self.backend_pill, "ok" if ok else "bad")
        self.reconnect_button.setVisible(not ok)
        if ok:
            self.notice.hide()
        else:
            self.notice.setText(f"变声引擎未连接：{reason}\n点右上角“重新连接”会再检查一次，需要时自动启动引擎。")
            self.notice.show()
        self._refresh()

    def _poll_health(self) -> None:
        if self._health_inflight or self._launcher is not None or self._closing:
            return
        self._health_inflight = True

        def work():
            try:
                info = self.client.health()
                outcome = (info.get("status") == "ok", "后端正在加载模型。")
            except BackendError as exc:
                outcome = (False, str(exc))
            self._emit(self.bridge.health_result, *outcome)

        threading.Thread(target=work, daemon=True, name="health").start()

    def _on_health(self, ok: bool, message: str) -> None:
        self._health_inflight = False
        if self._launcher is not None:
            return
        if ok:
            self._health_failures = 0
            if not self.backend_ok:
                self._set_backend(True)
                self.set_status("后端恢复连接。", "ok")
                if not self.voices:
                    self._load_voices()
        else:
            self._health_failures += 1
            if self.backend_ok and self._health_failures >= 2:
                self._set_backend(False, message)
                self.set_status("与后端的连接断开了。", "error")

    def _load_voices(self) -> None:
        def work():
            try:
                voices = self.client.voices()
            except BackendError as exc:
                self._emit(self.bridge.voices_failed, str(exc))
                return
            self._emit(self.bridge.voices_loaded, voices)

        threading.Thread(target=work, daemon=True, name="voices").start()

    def _on_voices_failed(self, message: str) -> None:
        self.notice.setText(f"读取角色列表失败：{message}")
        self.notice.show()
        self.set_status("读取角色列表失败。", "error")
        self._report_startup(False, f"读取角色列表失败：{message}")

    def _on_voices_loaded(self, voices: list) -> None:
        self.voices = voices
        self.voice_names = {v["id"]: v["name"] for v in voices}
        ids = list(self.voice_names)
        self.categories = catalog.categories_for(ids)
        self.category_id = None
        self.category_bar.blockSignals(True)
        self.category_bar.clear()
        for category in self.categories:
            self.category_bar.addItem(category.name, category.id)
        self.category_bar.blockSignals(False)
        self.voices_placeholder.setVisible(not ids)
        if not ids:
            # 公开版刚装好时还没有任何声线：照常进入主界面，引导用户导入声线包
            self.carousel.set_items([], None)
            self.voice_id = None
            self._show_voice(None)
            self.hero_name.setText("还没有声线")
            self.hero_caption.hide()
            self.voices_placeholder.setText("点下面的“导入声线”，把声线包（.pth 模型 + .index 索引）装进来就能开始用。")
            self.set_status("还没有声线，先导入一个声线包。")
            self._refresh()
            self._report_startup(True)
            return
        remembered = self.settings.voice_id
        target = (self.voice_id if self.voice_id in ids else remembered if remembered in ids
                  else "taffy" if "taffy" in ids else ids[0])
        self.voice_id = None
        self.select_voice(target)
        self._report_startup(True)

    # ---- 角色 -----------------------------------------------------------

    def _category_of(self, voice_id: str) -> catalog.Category | None:
        return next((c for c in self.categories if voice_id in c.voices), None)

    def _carousel_items(self, category: catalog.Category) -> list[dict]:
        return [{"id": v, "name": self.voice_names.get(v, v), "style": catalog.voice_style(v),
                 "pixmap": self._pixmap(v)} for v in category.voices]

    def _on_category_changed(self, index: int) -> None:
        """切换分类：转盘换成这个分类的角色，并选回这个分类上次用的角色。"""
        if not 0 <= index < len(self.categories):
            return
        category = self.categories[index]
        if self.voice_id in category.voices:
            return
        self.select_voice(self._last_voice.get(category.id, category.voices[0]))

    def _show_voice(self, voice_id: str | None) -> None:
        self.voice_mode.setEnabled(voice_id is not None)
        self.pitch_combo.setEnabled(voice_id is not None and self.pitch_supported and voice_id not in self.reference_voices)
        has_reference = voice_id is not None and (ASSETS / "references" / f"{voice_id}.wav").is_file()
        self.reference_button.setEnabled(has_reference)
        self.reference_button.setVisible(has_reference or self._any_reference)  # 导入的声线没有原声片段
        style = theme.voice_style(voice_id) if voice_id else theme.brand_style()
        self.hero.set_voice(style, self._pixmap(voice_id or catalog.BRAND_VOICE))
        self.live_switch.set_color(style.color)
        self.send_toggle.set_color(style.color)
        self.monitor_toggle.set_color(style.color)
        self.reference_button.setIcon(make_icon("play", style.color, 11))
        if voice_id is None:
            self.hero_name.setText("…")
            self.hero_caption.setText("正在读取角色列表")
        else:
            self.hero_name.setText(self._voice_name(voice_id))
            self.hero_caption.setText("")
            self.hero_name.setToolTip(f"模型：{self.voice_models.get(voice_id, voice_id.upper())}")
        self.hero_caption.setStyleSheet(f"color: {style.color};")
        self.hero_caption.setVisible(voice_id is None)

    def select_voice(self, voice_id: str) -> None:
        if voice_id == self.voice_id or voice_id not in self.voice_names:
            return
        if self.playing_tag == ("preview", "reference"):
            self.stop_playback()
        self.voice_id = voice_id
        category = self._category_of(voice_id)
        if category is not None:
            self._last_voice[category.id] = voice_id
            if category.id != self.category_id:
                self.category_id = category.id
                self.carousel.set_items(self._carousel_items(category), voice_id)
                self.category_bar.blockSignals(True)
                self.category_bar.setCurrentIndex(self.categories.index(category))
                self.category_bar.blockSignals(False)
            else:
                self.carousel.set_current(voice_id)
        self._show_voice(voice_id)
        self._refresh()
        if self.live is not None:
            self._restart_live()

    def _on_strength_changed(self, value: int) -> None:
        self.strength_pct = int(value)
        self.strength_value.setText(f"{value}%")
        self._refresh()
        if self.live is not None:
            self._strength_timer.start()

    def _on_voice_mode_changed(self) -> None:
        self.stop_playback()
        self._refresh()
        if self.live is not None:
            self._restart_live()

    # ---- 设备 -----------------------------------------------------------
    def _populate_devices(self, announce: bool = True) -> None:
        counts = []
        for combo, kind, empty in ((self.mic_combo, "input", "未找到麦克风"),
                                   (self.out_combo, "output", "未找到输出设备")):
            previous = (combo.currentData().name if combo.currentData() is not None else
                        self.settings.mic if combo is self.mic_combo else self.settings.output)
            try:
                devices = audio.list_devices(kind)
            except audio.AudioDeviceError as exc:
                devices = []
                self.set_status(str(exc), "error")
            combo.blockSignals(True)
            combo.clear()
            for device in devices:
                combo.addItem(device.name + ("（默认）" if device.is_default else ""), device)
                combo.setItemData(combo.count() - 1,
                                  f"{device.name}\n{device.hostapi} · {device.rate} Hz · {device.channels} 声道",
                                  Qt.ToolTipRole)
            if not devices:
                combo.addItem(empty, None)
            index = next((i for i in range(combo.count())
                          if combo.itemData(i) is not None and combo.itemData(i).name == previous), 0)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
            combo.setToolTip(combo.itemData(combo.currentIndex(), Qt.ToolTipRole) or empty)
            counts.append(len(devices))
        def available(combo):
            return [combo.itemData(i) for i in range(combo.count()) if combo.itemData(i)]
        self.discord_pair = audio.discord_pair(available(self.mic_combo), available(self.out_combo))
        self._route_changed()
        if announce:
            self.set_status(f"设备列表已刷新：{counts[0]} 个麦克风，{counts[1]} 个输出设备。")

    def _route_changed(self, *_):
        sending, monitoring = self.send_toggle.isChecked(), self.monitor_toggle.isChecked()
        pair = self.discord_pair
        warn = (sending and not pair) or not (sending or monitoring)
        lines = []
        if sending:
            lines.append(f"在 Discord 等语音软件里，把输入设备（麦克风）选为 {pair[1].name}" if pair else
                         "没有找到虚拟麦克风。安装 VB-CABLE 后点“刷新”，或者先打开耳返试听。")
        if monitoring:
            lines.append("耳返从下面选的耳机播放，请戴耳机，外放会被麦克风收回去。")
        if not lines:
            lines.append("打开“发到语音软件”或“耳返”，才能开始变声。")
        self.route_hint.setText("\n".join(lines))
        set_state(self.route_hint, "warn" if warn else "")
        self._refresh()
        if self.live is not None:
            self._restart_live()

    def _retry_after_rescan(self, attempt):
        """运行 attempt()；PortAudio 说设备无效时，重新扫描一次设备再试。

        PortAudio 只在启动时给设备编号。窗口开着的时候 Windows 的音频设备有变化，旧编号就会失效，
        报“Invalid device [PaErrorCode -9996]”。重新扫描后按名称找回原来选的设备；
        设备真的不在了，就明确告诉用户，而不是悄悄换成别的设备。
        """
        try:
            return attempt()
        except audio.AudioDeviceError as exc:
            if (not any(code in str(exc) for code in STALE_DEVICE_ERRORS)
                    or self.recorder is not None or self.live is not None):
                raise
            wanted = {combo: combo.currentData().name for combo in (self.mic_combo, self.out_combo)
                      if combo.currentData() is not None}
            self.player.stop()  # 重新扫描前必须关掉所有音频流
            audio.rescan()
            self._populate_devices(announce=False)
            self._refresh()
            missing = [name for combo, name in wanted.items()
                       if combo.currentData() is None or combo.currentData().name != name]
            if missing:
                raise audio.AudioDeviceError(
                    f"“{'”、“'.join(missing)}”已经不在系统的设备列表里（可能被拔出或停用）。"
                    "设备列表已刷新，请重新选择。") from exc
            result = attempt()
            self.set_status("音频设备有变化，已自动刷新设备列表。", "ok")
            return result

    def rescan_devices(self) -> None:
        if self.recorder is not None or self.live is not None:
            return
        self.stop_playback()
        try:
            audio.rescan()
        except audio.AudioDeviceError as exc:
            self.set_status(str(exc), "error")
            return
        self._populate_devices()
        self._refresh()

    def _on_device_changed(self) -> None:
        for combo in (self.mic_combo, self.out_combo):
            combo.setToolTip(combo.itemData(combo.currentIndex(), Qt.ToolTipRole) or "")
        self._refresh()
        if self.live is not None:
            self._restart_live()

    # ---- 输入：录音与导入 -------------------------------------------------
    def _on_rec_toggled(self, checked: bool) -> None:
        if checked and self.recorder is None:
            self._start_recording()
        elif not checked and self.recorder is not None:
            self._stop_recording()

    def _start_recording(self) -> None:
        device = self._device(self.mic_combo)
        try:
            if device is None:
                raise audio.AudioDeviceError("没有找到麦克风，无法录音。")
            self.stop_playback()

            def open_recorder():
                opened = audio.Recorder(self._device(self.mic_combo), self.max_seconds)
                opened.start()
                return opened

            recorder = self._retry_after_rescan(open_recorder)
            device = recorder.device
        except audio.AudioDeviceError as exc:
            self.rec_button.setChecked(False)
            self.set_status(str(exc), "error")
            self._refresh()
            return
        self.recorder = recorder
        self._rec_started = time.monotonic()
        self.rec_hint.setText(f"正在录：{device.name}")
        self.set_status(f"正在用“{device.name}”录音，最长 {self.max_seconds:g} 秒。")
        self._refresh()

    def _stop_recording(self, note: str = "", lost: bool = False) -> None:
        recorder, self.recorder = self.recorder, None
        if recorder is None:
            return
        samples = recorder.stop()
        self.rec_button.setChecked(False)
        self.rec_meter.reset()
        seconds = len(samples) / recorder.rate
        self.rec_clock.setText(_clock(seconds if seconds >= audio.MIN_SECONDS else 0.0))
        if seconds < audio.MIN_SECONDS:
            self.set_status(f"{note}录音只有 {seconds:.2f} 秒，至少需要 {audio.MIN_SECONDS} 秒，没有替换当前输入。", "error")
            self._refresh()
            return
        self._set_input(f"录音 {time.strftime('%H:%M:%S')}", audio.encode_wav(samples, recorder.rate),
                        "audio/wav", samples, recorder.rate)
        if float(np.max(np.abs(samples))) < 0.003:
            self.set_status(f"{note}录好了 {seconds:.1f} 秒，但几乎没有声音，请检查“{recorder.device.name}”是否有信号。",
                            "error")
        else:
            self.set_status(f"{note}录好了 {seconds:.1f} 秒。", "error" if lost else "ok")

    def import_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "导入音频", self._last_dir, "音频文件 (*.wav *.flac *.mp3 *.ogg);;所有文件 (*)")
        if path:
            self.load_file(Path(path))

    def load_file(self, path: Path) -> bool:
        path = Path(path)
        try:
            size = path.stat().st_size
            if size > self.max_bytes:
                raise audio.AudioFileError(
                    f"文件有 {size / 1048576:.1f} MiB，超过后端上限 {self.max_bytes / 1048576:g} MiB。")
            data = path.read_bytes()
            samples, rate = audio.probe_file(data, self.max_seconds)
        except OSError as exc:
            self.set_status(f"无法读取 {path.name}：{exc}", "error")
            return False
        except audio.AudioFileError as exc:
            self.set_status(f"{path.name}：{exc}", "error")
            return False
        self._last_dir = str(path.parent)
        self._set_input(path.name, data, CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"),
                        samples, rate)
        self.set_status(f"已导入 {path.name}（{len(samples) / rate:.2f} 秒）。", "ok")
        return True

    def _set_input(self, label: str, data: bytes, content_type: str, samples: np.ndarray, rate: int) -> None:
        """换输入：旧输入的结果全部作废，避免把旧录音的结果当成新结果。"""
        if self.playing_tag is not None and self.playing_tag[0] in ("input", "result"):
            self.stop_playback()
        self._serial += 1
        self.input = InputClip(self._serial, label, data, content_type, samples, rate)
        self.results = {}
        self.ab = "input"
        self.seek_slider.setValue(0)
        self.time_label.setText(f"0:00.0 / {_clock(self.input.seconds)}")
        self._refresh()

    # ---- 转换 -----------------------------------------------------------
    def convert(self) -> None:
        key = self.current_key()
        if key is None or self.converting_key is not None or self.live is not None or not self.backend_ok:
            return
        clip, voice_id, strength = self.input, self.voice_id, self.strength_pct / 100
        self.converting_key = key
        self._convert_started = time.monotonic()
        self.set_status(f"正在把「{clip.label}」变成 {self._voice_name(voice_id)}（强度 {self.strength_pct}%）…")
        self._refresh()

        def work():
            started = time.perf_counter()
            try:
                wav = self.client.convert(clip.data, clip.content_type, voice_id, strength, mode=key[3], pitch_preset=key[4])
            except BackendError as exc:
                self._emit(self.bridge.convert_failed, key, str(exc))
                return
            self._emit(self.bridge.convert_done, key, wav, time.perf_counter() - started)

        threading.Thread(target=work, daemon=True, name="convert").start()

    def _emit(self, signal, *args) -> None:
        """工作线程用它回到界面线程。窗口关闭后不再发信号：那时 Qt 对象可能正在销毁。"""
        if self._closing:
            return
        try:
            signal.emit(*args)
        except RuntimeError:
            pass  # 窗口已销毁

    def _on_convert_done(self, key: tuple, wav: bytes, elapsed: float) -> None:
        self.converting_key = None
        if self.input is None or key[0] != self.input.serial:
            self.set_status("转换完成时输入已经更换，旧输入的结果已丢弃。")
            self._refresh()
            return
        try:
            samples, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
        except Exception as exc:
            self.set_status(f"后端返回的结果无法解码：{exc}", "error")
            self._refresh()
            return
        result = ResultClip(key, wav, samples.mean(axis=1), rate, self._voice_name(key[1]),
                            self.input.label, elapsed)
        self.results[key] = result
        while len(self.results) > MAX_RESULTS:
            oldest = min(self.results.values(), key=lambda r: r.created)
            del self.results[oldest.key]
        described = f"{result.voice_name} · 强度 {key[2]}% · {MODE_NAMES[key[3]]} · {PITCH_PRESETS[key[4]][0]}"
        if key == self.current_key():
            was_playing = self.playing_tag is not None and self.playing_tag[0] == "input"
            self.ab = "result"
            if was_playing:
                self.stop_playback()
            self.set_status(f"变声完成：{described}，用时 {elapsed:.1f} 秒。可以试听并另存为 WAV。", "ok")
        else:
            self.set_status(f"变声完成：{described}（用时 {elapsed:.1f} 秒）。当前设置已经改过，改回这组设置即可试听和保存。")
        self._refresh()

    def _on_convert_failed(self, key: tuple, message: str) -> None:
        self.converting_key = None
        self.set_status(f"转换失败：{message}", "error")
        self._refresh()

    # ---- 试听 -----------------------------------------------------------
    def _play(self, tag: tuple, samples: np.ndarray, rate: int, start: float = 0.0) -> bool:
        device = self._device(self.out_combo)
        try:
            if device is None:
                raise audio.AudioDeviceError("没有找到输出设备，无法播放。")
            self._retry_after_rescan(lambda: self.player.play(samples, rate, self._device(self.out_combo), start))
        except audio.AudioDeviceError as exc:
            self.stop_playback()
            self.set_status(str(exc), "error")
            return False
        self.playing_tag = tag
        self._sync_preview_buttons()
        self._refresh()
        return True

    def stop_playback(self) -> None:
        was_ab = self.playing_tag is not None and self.playing_tag[0] in ("input", "result")
        self.player.stop()
        self.playing_tag = None
        self._sync_preview_buttons()
        if was_ab:
            self.seek_slider.setValue(0)
            self._show_time(0.0)
            self._refresh()

    def _sync_preview_buttons(self) -> None:
        playing = self.playing_tag == ("preview", "reference")
        self.reference_button.setText("停止" if playing else "角色原声")
        color = (theme.voice_style(self.voice_id) if self.voice_id else theme.brand_style()).color
        self.reference_button.setIcon(make_icon("stop" if playing else "play", color, 11))

    def toggle_preview(self, name: str = "reference") -> None:
        """播放当前角色已核实出处的原始配音或直播录音；再点一次停止。"""
        if self.playing_tag == ("preview", name):
            self.stop_playback()
            return
        if self.voice_id is None:
            return
        path = ASSETS / "references" / f"{self.voice_id}.wav"
        try:
            samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as exc:
            self.set_status(f"无法读取角色原声 {path.name}：{exc}", "error")
            return
        self.stop_playback()
        if self._play(("preview", name), samples.mean(axis=1), rate):
            self.set_status(f"正在播放{self._voice_name(self.voice_id)}的角色原声（{path.name}）"
                            f"→ {self._device(self.out_combo).name}")

    def _ab_clip(self, which: str):
        if which == "result":
            result = self.current_result()
            return (result.samples, result.rate) if result is not None else None
        return (self.input.samples, self.input.rate) if self.input is not None else None

    def _ab_duration(self) -> float:
        clip = self._ab_clip(self.ab)
        return len(clip[0]) / clip[1] if clip is not None else 0.0

    def _show_time(self, position: float) -> None:
        self.time_label.setText(f"{_clock(position)} / {_clock(self._ab_duration())}")

    def set_ab(self, which: str) -> None:
        """切换原声 / 结果。播放中切换时从同一时间点接着播。"""
        if which == "result" and self.current_result() is None:
            which = "input"
        playing = self.playing_tag is not None and self.playing_tag[0] in ("input", "result")
        position = self.player.position if playing else self.seek_slider.value() / 1000 * self._ab_duration()
        self.ab = which
        if playing:
            self.player.stop()
            self.playing_tag = None
            clip = self._ab_clip(which)
            duration = len(clip[0]) / clip[1]
            self._play((which, self.current_key()), clip[0], clip[1], position if position < duration - 0.05 else 0.0)
        else:
            self._refresh()
        self._show_time(position if playing else self.seek_slider.value() / 1000 * self._ab_duration())

    def toggle_play(self) -> None:
        if self.playing_tag is not None and self.playing_tag[0] in ("input", "result"):
            self.stop_playback()
            return
        clip = self._ab_clip(self.ab)
        if clip is None:
            return
        self.stop_playback()
        duration = len(clip[0]) / clip[1]
        fraction = self.seek_slider.value() / 1000
        start = fraction * duration if fraction < 0.98 else 0.0
        if self._play((self.ab, self.current_key()), clip[0], clip[1], start):
            what = "变声结果" if self.ab == "result" else "原声"
            self.set_status(f"正在播放{what} → {self._device(self.out_combo).name}")

    def _on_seek(self) -> None:
        duration = self._ab_duration()
        position = self.seek_slider.value() / 1000 * duration
        if self.playing_tag is not None and self.playing_tag[0] in ("input", "result"):
            self.player.seek(self.seek_slider.value() / 1000 * self.player.duration)
        self._show_time(position)

    # ---- 保存 -----------------------------------------------------------
    def save_dialog(self) -> None:
        result = self.current_result()
        if result is None:
            return
        stem = Path(result.input_label).stem.replace(":", "-").replace(" ", "_") or "voice"
        mode_suffix = f"_{result.key[3]}" if result.key[1] == "taffy" else ""
        default = str(Path(self._last_dir) / f"{stem}_{result.key[1]}_{result.key[2]}{mode_suffix}_{result.key[4]}.wav")
        path, _ = QFileDialog.getSaveFileName(self, "另存为 WAV", default, "WAV 音频 (*.wav)")
        if path:
            self.save_result(Path(path))

    def save_result(self, path: Path) -> bool:
        """只保存与当前输入、角色、强度对应的结果，写出后端返回的原始 WAV 字节。"""
        result = self.current_result()
        if result is None:
            self.set_status("当前输入、角色和强度没有对应的结果，没有保存。", "error")
            return False
        path = Path(path)
        if path.suffix.lower() != ".wav":
            path = path.with_name(path.name + ".wav")
        try:
            path.write_bytes(result.wav)
        except OSError as exc:
            self.set_status(f"保存失败：{exc}", "error")
            return False
        self._last_dir = str(path.parent)
        self.set_status(f"已保存 {path}（{result.voice_name} · 强度 {result.key[2]}% · {MODE_NAMES[result.key[3]]} · {PITCH_PRESETS[result.key[4]][0]} · 来源「{result.input_label}」）", "ok")
        return True

    # ---- 麦克风监听 -------------------------------------------------------
    def _on_live_toggled(self, checked: bool) -> None:
        if self._live_resume_monitor is not None:
            if not checked:
                self._live_resume_monitor = None
                self.live_status.setText("变声已关闭，已取消自动重开。")
                self.set_status("变声已关闭。")
                self._refresh()
            return
        if checked and self.live is None:
            self._start_live()
        elif checked:
            self._live_restart = True  # 上一个会话还在收尾，结束后接着开
        elif self.live is not None:
            self._live_restart = False
            self.live.request_stop()
            self.live_title.setText("正在停止…")

    def _open_live_devices(self):
        """在界面线程打开并启动采集和播放流，之后交给工作线程。

        发到语音软件时主输出是虚拟麦克风，耳返另开一路到耳机；只开耳返时主输出就是耳机。
        主输出决定缓冲和统计，和以前的“Discord 通话 / 本机监听”完全一样；耳返那一路出错不影响通话。
        """
        mic_device, headphones = self._device(self.mic_combo), self._device(self.out_combo)
        sending, monitoring = self.send_toggle.isChecked(), self.monitor_toggle.isChecked()
        receiver = None
        if not sending and not monitoring:
            raise audio.AudioDeviceError("“发到语音软件”和“耳返”至少要打开一个。")
        if sending and not self.discord_pair:
            if not monitoring:
                raise audio.AudioDeviceError("没有找到虚拟麦克风（VB-CABLE 或 Steam）。安装后点“刷新”，或者先只用耳返试听。")
            sending = False  # 没有虚拟麦克风时先只开耳返，提示已经写在开关下面
        if sending:
            out_device, receiver = self.discord_pair
            if mic_device == receiver:
                raise audio.AudioDeviceError("麦克风请选择你说话用的设备，不要选虚拟麦克风，以免形成回路。")
        else:
            out_device = headphones
        if mic_device is None:
            raise audio.AudioDeviceError("没有找到麦克风，无法变声。")
        if out_device is None:
            raise audio.AudioDeviceError("没有找到耳机或音箱，无法播放耳返。")
        ear = headphones if (sending and monitoring and headphones is not None
                             and headphones.name != out_device.name) else None
        mic = audio.MicStream(mic_device)
        opened = [mic]
        try:
            # Virtual output keeps more headroom for inference and device jitter.
            slack = audio.live_slack(self.voice_id in self.reference_voices, self.voice_id, self.current_mode(),
                                     receiver is not None, out_device.name)
            speaker = audio.SpeakerStream(out_device, slack_seconds=slack, receiver=receiver)
            opened.append(speaker)
            if ear is not None:
                try:
                    speaker = audio.TeeSpeaker(speaker, audio.SpeakerStream(
                        ear, slack_seconds=audio.live_slack(False, self.voice_id, self.current_mode(), False)))
                except audio.AudioDeviceError as exc:
                    self.set_status(f"耳返没有打开（{exc}），只发到语音软件。", "error")
            mic.start()
            speaker.start()
        except audio.AudioDeviceError:
            for stream in opened:
                stream.close()
            raise
        return mic, speaker

    def _start_live(self) -> None:
        if not self.backend_ok or self.voice_id is None or self.converting_key is not None:
            self.live_switch.setChecked(False)
            return
        try:
            mic, speaker = self._retry_after_rescan(self._open_live_devices)
        except audio.AudioDeviceError as exc:
            self.live_switch.setChecked(False)
            self.live_status.setText(f"变声没有打开：{exc}")
            set_state(self.live_status, "error")
            self.set_status(str(exc), "error")
            self._refresh()
            return
        packet_ms = audio.live_packet_ms(self.voice_id in self.reference_voices, speaker.name)
        if packet_ms == 120:
            packet_ms = self.config.live_packet_ms
        if packet_ms not in self.live_packet_options:
            packet_ms = 120 if 120 in self.live_packet_options else 200
        pitch_preset = self.current_pitch()
        url = self.client.live_url(self.voice_id, self.strength_pct / 100, self.current_mode(),
                                   packet_ms=packet_ms, pitch_preset=pitch_preset)
        monitor = LiveMonitor(
            url, mic, speaker,
            on_connected=lambda hello: self._emit(self.bridge.live_connected, monitor, hello),
            on_stats=lambda stats: self._emit(self.bridge.live_stats, monitor, stats),
            on_stopped=lambda kind, message: self._emit(self.bridge.live_stopped, monitor, kind, message),
            restart_after_packets=self.config.live_restart_packets)
        monitor.label = f"{self._voice_name(self.voice_id)} · 强度 {self.strength_pct}% · {MODE_NAMES[self.current_mode()]}"
        monitor.pitch_preset = pitch_preset
        self.live = monitor
        self._live_stats = {}
        self.mic_advisor.reset()
        self.live_title.setText("正在连接…")
        self.live_status.setText("正在连接变声引擎…")
        set_state(self.live_status, "")
        self.live_detail.setText(f"{mic.name} → {speaker.name}")
        self.detail_button.show()
        monitor.start()
        self._refresh()

    def _restart_live(self) -> None:
        """切换角色、强度或设备时重新连接。"""
        if self.live is None or not self.live_switch.isChecked():
            return
        self._live_restart = True
        self.live.request_stop(abort=True)
        self.live_title.setText("正在淡出并重启实时变声…")

    def _on_live_connected(self, monitor, hello: dict) -> None:
        if monitor is not self.live:
            return
        self.live_title.setText(f"正在变声 · {self._voice_name(self.voice_id)}")
        speaker = monitor.speaker
        where = ("语音软件和耳返" if isinstance(speaker, audio.TeeSpeaker) else
                 "语音软件" if self.discord_pair and speaker.name == self.discord_pair[0].name else "耳返")
        self.set_status(f"实时变声已开始，输出到{where}。", "ok")
        self.status_label.setToolTip(f"{hello['sample_rate']} Hz，每包 {hello['frame_samples']} 个样本")

    def _on_live_stats(self, monitor, stats: dict) -> None:
        if monitor is not self.live:
            return
        self._live_stats = stats
        limit = monitor.restart_after_packets
        if limit and stats["frames"] >= limit and not self._live_restart and self.live_switch.isChecked():
            self.live_title.setText("等待说话停顿后自动刷新…")
        speaker = monitor.speaker
        # 估算本机链路，包含采集、前瞻、重采样和断流后增加的缓冲；不含手机网络与 Discord。
        frame_ms = monitor.hello["frame_samples"] / monitor.hello["sample_rate"] * 1000
        overlap_ms = monitor.hello.get("lookahead_samples", 1024) / monitor.hello["sample_rate"] * 1000
        delay = (frame_ms + stats["avg_ms"]
                 + getattr(speaker, "target_buffer_seconds", getattr(speaker, "slack_seconds", 0)) * 1000 + overlap_ms
                 + (getattr(speaker, "latency", 0) + getattr(monitor.mic, "latency", 0)) * 1000
                 + stats.get("resampler_delay_ms", 0)) / 1000
        if stats["avg_ms"] > frame_ms * .9:
            health, state = "电脑处理不过来，声音可能断续。请关掉占用显卡的程序再试。", "warn"
        elif stats["underruns"] > 3 and stats["underruns"] * 50 > stats["frames"]:
            health, state = f"声音有断续（{stats['underruns']} 次），请关掉占用显卡的程序再试。", "warn"
        else:
            health, state = "运行正常", "ok"
        self.live_status.setText(f"延迟约 {delay:.1f} 秒 · {health}")
        set_state(self.live_status, state)
        ear = getattr(speaker, "monitor", None)
        self.live_detail.setText(
            f"{monitor.mic.name} → {speaker.name}" + (f" + 耳返 {ear.name}" if ear is not None else "") + "\n"
            f"{PITCH_PRESETS[monitor.pitch_preset][0]} · {MODE_NAMES[self.current_mode()]} · 每包处理 {stats['avg_ms']:.0f} 毫秒"
            f" · 本轮 {min(stats['frames'],limit) if limit else stats['frames']}"
            f"{f'/{limit}' if limit else ''} 包 · 自动刷新 {self._live_auto_refreshes} 次"
            f" · 丢弃输入 {stats['mic_dropped']} 块 · 播放断流 {stats['underruns']} 次")

    def _on_live_stopped(self, monitor, kind: str, message: str) -> None:
        if monitor is not self.live:
            return
        self.live = None
        restart, self._live_restart = self._live_restart, False
        self.live_in_meter.reset()
        self.live_out_meter.reset()
        self.mic_hint.hide()
        if (restart and kind == "user" or kind == "refresh") and self.live_switch.isChecked() and not self._closing:
            if kind == "refresh":
                self._live_auto_refreshes += 1
            # The stopped signal arrives before Thread.run() has returned.
            # Give native WASAPI teardown time to finish before reopening it;
            # the running switch remains a cancellable request to resume.
            self._live_resume_monitor = monitor
            self.live_title.setText("正在释放设备，稍后重新变声…")
            QTimer.singleShot(LIVE_DEVICE_SETTLE_MS, lambda: self._resume_live(monitor))
            return
        self.live_switch.setChecked(False)
        self.detail_button.setChecked(False)
        self.detail_button.hide()
        if kind == "error":
            self.live_status.setText(f"变声已停止：{message}")
            set_state(self.live_status, "error")
            self.set_status(message, "error")
            if self.tray is not None and not self.isVisible():
                self.tray.showMessage("角色变声器", f"变声已停止：{message}", QSystemTrayIcon.Warning, 4000)
        else:
            self.live_status.setText("变声已关闭，麦克风已释放。")
            set_state(self.live_status, "")
            self.set_status("变声已关闭。")
        self._refresh()

    def _resume_live(self, monitor) -> None:
        if self._live_resume_monitor is not monitor:
            return
        if self._closing or not self.live_switch.isChecked():
            self._live_resume_monitor = None
            return
        if monitor.is_alive():
            QTimer.singleShot(25, lambda: self._resume_live(monitor))
            return
        monitor.join()
        self._live_resume_monitor = None
        self._start_live()

    # ---- 定时刷新 ---------------------------------------------------------
    def _tick(self) -> None:
        if self.playing_tag is not None:
            if self.player.playing:
                if self.playing_tag[0] in ("input", "result") and not self.seek_slider.isSliderDown():
                    duration = self.player.duration or 1.0
                    self.seek_slider.setValue(int(self.player.position / duration * 1000))
                    self._show_time(self.player.position)
            else:
                self.stop_playback()
        if self.recorder is not None:
            self.rec_meter.set_peak(self.recorder.level)
            self.rec_clock.setText(_clock(self.recorder.seconds))
            if self.recorder.full:
                self._stop_recording(f"已到 {self.max_seconds:g} 秒上限，自动停止。")
            elif not self.recorder.active:
                self._stop_recording("麦克风已断开，录音提前结束。", lost=True)
        if self.live is not None:
            self.live_in_meter.set_peak(self.live.mic.level)
            self.live_out_meter.set_peak(getattr(self.live.speaker, "level", 0.0))
            if self.live.hello:
                state, text = self.mic_advisor.feed(self.live.mic.level)
                if state:
                    self.mic_hint.setText(text)
                    self.mic_hint.setVisible(bool(text))
        if self.converting_key is not None:
            self.convert_button.setText(f"变声中… {time.monotonic() - self._convert_started:.0f} 秒")

    # ---- 退出 -----------------------------------------------------------
    def closeEvent(self, event) -> None:
        """关闭窗口：停止播放和录音，结束监听并释放设备，停掉由本程序启动的后端。

        有托盘图标且设置里选了“关闭时最小化到托盘”时，只隐藏窗口，变声继续。
        """
        if (self.tray is not None and self.settings.close_to_tray and not self._quitting
                and QApplication.instance() is not None and not QApplication.instance().closingDown()):
            event.ignore()
            self.hide()
            if not self._tray_hint_shown:
                self._tray_hint_shown = True
                self.tray.showMessage("角色变声器还在运行",
                                      "在托盘图标上右键可以开关变声或退出。可以在设置里改成关闭即退出。",
                                      QSystemTrayIcon.Information, 4000)
            return
        if self._closing:
            event.accept()
            return
        self._save_timer.stop()
        self.save_settings()
        self._closing = True
        self._live_resume_monitor = None
        self._tick_timer.stop()
        self._health_timer.stop()
        self._strength_timer.stop()
        self.player.stop()
        if self.recorder is not None:
            self.recorder.stop()
            self.recorder = None
        if self.live is not None:
            self.live.request_stop(abort=True)
            self.live.join(3.0)
        if self._launcher is not None:
            self._launcher.cancel()
            stop_process_tree(self._launcher.process)
        stop_process_tree(self.owned_backend)
        if self.hotkey is not None:
            self.hotkey.close()
        if self.tray is not None:
            self.tray.hide()
            QApplication.instance().quit()
        event.accept()
