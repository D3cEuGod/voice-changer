"""启动加载页：和主界面同样的白底粉色卡片，Q 版塔菲拉着进度条往前跑。模型加载完成后再进入主界面。"""
from __future__ import annotations

import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from . import __version__, catalog, theme
from .backend import ROOT
from .widgets import Card, label

# 进度条各段：启动脚本的检查只占一小段，模型加载占大头（后端报 0–1，映射到 35%–100%）。
SCRIPT_STAGES = {
    "env": (0.03, "检查运行环境"),
    "files": (0.08, "检查模型文件"),
    "server": (0.15, "启动后端服务"),
}
MODELS_START = 0.35
SERVER_CREEP_SECONDS = 3.0  # 后端导入程序库这段没有进度，按时间慢慢挪，免得看起来卡住
MODEL_CREEP_SECONDS = 2.5   # 单个模型加载要一两秒，期间也往前挪一点（最多走完剩余的三成）


class TaffyRunner(QWidget):
    """进度条：粉色填充往右走，Q 版塔菲在最前面拉着它跑，后面扬起一点小尘土。"""

    TRACK_HEIGHT = 16
    CHIBI = 96

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._phase = 0.0
        self._running = True
        chibi_file = catalog.brand_style().chibi
        image = QPixmap(str(ROOT / "assets" / chibi_file)) if chibi_file else QPixmap()
        # 原图朝左，翻过来让她朝前进的方向跑。
        self._chibi = image.transformed(QTransform().scale(-1, 1), Qt.SmoothTransformation) if not image.isNull() else None
        self.setMinimumHeight(self.CHIBI + 26)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(1.0, value))
        self.update()

    def set_running(self, running: bool) -> None:
        self._running = running

    def _tick(self) -> None:
        if self._running:
            self._phase = (self._phase + 0.09) % 1.0
        self.update()

    def _track(self) -> QRectF:
        left = 10
        right = self.width() - self.CHIBI - 12  # 右边留出塔菲的位置，到 100% 时她也不出框
        top = self.height() - self.TRACK_HEIGHT - 6
        return QRectF(left, top, max(40.0, right - left), self.TRACK_HEIGHT)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        track = self._track()
        radius = track.height() / 2
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.setBrush(QColor(theme.FIELD))
        p.drawRoundedRect(track, radius, radius)
        fill_width = max(track.height(), track.width() * self._value)
        fill = QRectF(track.left(), track.top(), fill_width, track.height())
        gradient = QLinearGradient(fill.topLeft(), fill.topRight())
        gradient.setColorAt(0, QColor(theme.SAKURA))
        gradient.setColorAt(1, QColor(theme.SAKURA_DEEP))
        p.setPen(Qt.NoPen)
        p.setBrush(gradient)
        p.drawRoundedRect(fill, radius, radius)
        # 填充上一排斜纹，跑的时候往前滚动
        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(fill, radius, radius)
        p.setClipPath(clip)
        p.setPen(QPen(QColor(255, 255, 255, 70), 5))
        step = 14
        shift = self._phase * step if self._running else 0
        x = fill.left() - track.height() + shift
        while x < fill.right() + track.height():
            p.drawLine(QPointF(x, fill.bottom() + 2), QPointF(x + track.height(), fill.top() - 2))
            x += step
        p.restore()
        head = QPointF(fill.right() - radius, track.center().y())
        if self._chibi is None:
            return
        # 塔菲：跑步的上下颠动和左右摆，脚落在进度条上方，手里的绳子拴着填充的前端
        swing = math.sin(self._phase * 2 * math.pi)
        hop = abs(swing) * 6 if self._running else 0
        side = float(self.CHIBI)
        body = QRectF(fill.right() + 4, track.top() - side + 10 - hop, side, side)
        shadow = QColor(theme.SAKURA_DEEP)
        shadow.setAlpha(45)
        p.setPen(Qt.NoPen)
        p.setBrush(shadow)
        p.drawEllipse(QPointF(body.center().x(), track.top() + 2), side * 0.28 - hop * 0.8, 4)
        hand = QPointF(body.left() + side * 0.30, body.top() + side * 0.62)
        p.setPen(QPen(QColor(theme.SAKURA_DEEP), 2.2, Qt.SolidLine, Qt.RoundCap))
        rope = QPainterPath(head)
        rope.quadTo(QPointF((head.x() + hand.x()) / 2, max(head.y(), hand.y()) + 4), hand)
        p.setBrush(Qt.NoBrush)
        p.drawPath(rope)
        if self._running:  # 脚后跟扬起的小尘土
            for i in range(3):
                age = (self._phase + i / 3) % 1.0
                dust = QColor(theme.SAKURA)
                dust.setAlpha(int(150 * (1 - age)))
                p.setPen(Qt.NoPen)
                p.setBrush(dust)
                p.drawEllipse(QPointF(body.left() + 6 - age * 22, track.top() - 2 - age * 8), 2 + age * 4, 2 + age * 4)
        p.save()
        p.translate(body.center().x(), body.bottom())
        p.rotate(swing * 6 if self._running else 0)
        p.translate(-body.center().x(), -body.bottom())
        p.drawPixmap(body, self._chibi, QRectF(self._chibi.rect()))
        p.restore()


class StartupWindow(QWidget):
    """启动加载页。主窗口在后台连接后端；就绪后调用 finish()，失败调用 fail()。"""

    done = Signal()             # 进度走满，可以显示主界面了
    retry_requested = Signal()
    open_anyway = Signal()
    quit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("root")
        self.setWindowTitle("角色变声器")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.resize(600, 340)
        self._target = 0.0
        self._shown = 0.0
        self._server_since: float | None = None
        self._model_since: float | None = None
        self._last_fraction = -1.0  # 健康检查的回包可能比启动日志晚到，不让旧阶段盖掉新阶段
        self._finishing = False
        self._failed = False
        self._closing_by_code = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)
        card = Card()
        outer.addWidget(card)
        card.body.setContentsMargins(28, 24, 28, 24)
        card.body.setSpacing(10)
        top = QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(label("角色变声器", "appTitle"))
        top.addWidget(label(f"v{__version__} 内测", "version"), 0, Qt.AlignBottom)
        top.addStretch(1)
        self.quit_button = QPushButton("退出")
        self.quit_button.setObjectName("ghost")
        self.quit_button.clicked.connect(self.quit_requested)
        top.addWidget(self.quit_button)
        card.body.addLayout(top)
        self.subtitle = label("正在准备声线，马上就好", "sectionLabel")
        card.body.addWidget(self.subtitle)
        card.body.addStretch(1)
        self.runner = TaffyRunner()
        card.body.addWidget(self.runner)
        info = QHBoxLayout()
        self.stage = label("正在检查本地后端…")
        self.stage.setWordWrap(True)
        info.addWidget(self.stage, 1)
        self.percent = label("0%", "value")
        info.addWidget(self.percent, 0, Qt.AlignRight | Qt.AlignTop)
        card.body.addLayout(info)
        self.error = label("", "notice")
        self.error.setWordWrap(True)
        self.error.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.error.hide()
        card.body.addWidget(self.error)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.open_button = QPushButton("仍然打开主界面")
        self.open_button.setObjectName("ghost")
        self.open_button.clicked.connect(self.open_anyway)
        self.retry_button = QPushButton("重试")
        self.retry_button.setObjectName("primary")
        self.retry_button.clicked.connect(self._retry)
        for button in (self.open_button, self.retry_button):
            button.hide()
            buttons.addWidget(button)
        card.body.addLayout(buttons)
        card.body.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._advance)
        self._timer.start()
        self._drag_from = None

    # ---- 进度 --------------------------------------------------------------------------
    def set_stage(self, stage: str, fraction, text: str) -> None:
        """stage：脚本阶段名（env / files / server）或 "models"；fraction：模型加载进度 0–1。"""
        if self._failed or self._finishing:
            return
        if stage == "models" and fraction is not None:
            if float(fraction) < self._last_fraction:
                return
            self._last_fraction = float(fraction)
            self._server_since = None
            value = MODELS_START + (1 - MODELS_START) * float(fraction)
            if value > self._target:
                self._target = value
                self._model_since = time.monotonic()
            self.stage.setText(f"{text}…" if fraction < 1 else "加载完成")
        elif stage in SCRIPT_STAGES:
            value, name = SCRIPT_STAGES[stage]
            self._target = max(self._target, value)
            self.stage.setText(f"{name}…")
            if stage == "server":
                self._server_since = time.monotonic()
        elif text:
            self.stage.setText(text)

    def finish(self) -> None:
        """后端和角色列表都就绪：进度走满后发 done。"""
        if self._failed:
            return
        self._finishing = True
        self._target = 1.0
        self.stage.setText("加载完成，正在进入主界面…")

    def fail(self, message: str) -> None:
        self._failed = True
        self.runner.set_running(False)
        self.subtitle.setText("后端没有准备好")
        self.stage.setText("")
        self.error.setText(message)
        self.error.show()
        self.open_button.show()
        self.retry_button.show()

    def _retry(self) -> None:
        self._failed = False
        self._finishing = False
        self._target = self._shown = 0.0
        self._server_since = self._model_since = None
        self._last_fraction = -1.0
        self.runner.set_running(True)
        self.subtitle.setText("正在准备声线，马上就好")
        self.stage.setText("正在重新连接后端…")
        self.error.hide()
        self.open_button.hide()
        self.retry_button.hide()
        self.retry_requested.emit()

    def _advance(self) -> None:
        target = self._target
        if self._server_since is not None:  # 导入程序库这段按时间往 MODELS_START 慢慢挪，但不碰到它
            elapsed = time.monotonic() - self._server_since
            target = max(target, SCRIPT_STAGES["server"][0]
                         + (MODELS_START - 0.02 - SCRIPT_STAGES["server"][0]) * (1 - math.exp(-elapsed / SERVER_CREEP_SECONDS)))
        if self._model_since is not None and not self._finishing:
            elapsed = time.monotonic() - self._model_since
            target = max(target, self._target + (1 - self._target) * 0.3 * (1 - math.exp(-elapsed / MODEL_CREEP_SECONDS)))
        if self._shown < target:
            # 就绪后快速走满；平时追着目标走，不跳格
            step = 0.05 if self._finishing else max(0.004, (target - self._shown) * 0.12)
            self._shown = min(target, self._shown + step)
        self.runner.set_value(self._shown)
        self.percent.setText(f"{round(self._shown * 100)}%")
        if self._finishing and self._shown >= 0.999:
            self._finishing = False
            self._timer.stop()
            QTimer.singleShot(150, self.done.emit)

    # ---- 窗口 --------------------------------------------------------------------------
    def close_for_main(self) -> None:
        self._closing_by_code = True
        self.close()

    def mousePressEvent(self, event):  # 无边框窗口：按住空白处拖动
        if event.button() == Qt.LeftButton:
            self._drag_from = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_from is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_from)

    def mouseReleaseEvent(self, event):
        self._drag_from = None

    def closeEvent(self, event):
        if not self._closing_by_code:  # 用户从任务栏关掉加载页：当作退出
            self.quit_requested.emit()
        event.accept()
