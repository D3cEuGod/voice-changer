"""自绘控件：角色主视觉、角色转盘、变声开关、录音键、电平条。尺寸都用逻辑像素，随系统缩放。"""
from __future__ import annotations

import math

from PySide6.QtCore import QEasingCurve, QPointF, QRectF, QSize, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import (QColor, QIcon, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                           QRadialGradient)
from PySide6.QtWidgets import (QAbstractButton, QButtonGroup, QComboBox, QFrame, QGraphicsBlurEffect,
                               QGraphicsPixmapItem, QGraphicsScene, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from . import theme


# ---- 图案 -------------------------------------------------------------------
def _pen(color, width) -> QPen:
    pen = QPen(QColor(color), width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def paint_mic(p: QPainter, rect: QRectF, color: QColor) -> None:
    """麦克风：100×100 的坐标系缩放到 rect。"""
    p.save()
    p.translate(rect.x(), rect.y())
    p.scale(rect.width() / 100, rect.height() / 100)
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    p.drawRoundedRect(QRectF(35, 8, 30, 50), 15, 15)
    p.setBrush(Qt.NoBrush)
    p.setPen(_pen(color, 6))
    holder = QPainterPath()
    holder.moveTo(24, 42)
    holder.arcTo(QRectF(24, 18, 52, 52), 180, 180)
    p.drawPath(holder)
    p.drawLine(QPointF(50, 70), QPointF(50, 86))
    p.drawLine(QPointF(34, 89), QPointF(66, 89))
    p.restore()


def paint_guitar(p: QPainter, rect: QRectF, color: QColor) -> None:
    """吉他：琴身、琴颈、琴头，整体倾斜。"""
    p.save()
    p.translate(rect.center())
    p.scale(rect.width() / 100, rect.height() / 100)
    p.rotate(38)
    p.translate(-50, -52)
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    body = QPainterPath()
    body.addEllipse(QPointF(50, 76), 21, 19)
    upper = QPainterPath()
    upper.addEllipse(QPointF(50, 53), 14.5, 13)
    p.drawPath(body.united(upper))
    p.drawRoundedRect(QRectF(46.5, 6, 7, 40), 2, 2)
    p.drawRoundedRect(QRectF(43, -4, 14, 14), 4, 4)
    shade = QColor(0, 0, 0, 70)
    p.setBrush(shade)
    p.drawEllipse(QPointF(50, 66), 6.5, 6.5)
    p.drawRoundedRect(QRectF(42, 82, 16, 4), 2, 2)
    p.restore()


def paint_note(p: QPainter, rect: QRectF, color: QColor) -> None:
    p.save()
    p.translate(rect.x(), rect.y())
    p.scale(rect.width() / 100, rect.height() / 100)
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    p.drawEllipse(QPointF(34, 74), 13, 10)
    p.drawEllipse(QPointF(72, 64), 13, 10)
    p.setPen(_pen(color, 6))
    p.drawLine(QPointF(44, 74), QPointF(44, 24))
    p.drawLine(QPointF(82, 64), QPointF(82, 14))
    p.drawLine(QPointF(44, 24), QPointF(82, 14))
    p.restore()


ART_PAINTERS = {"mic": paint_mic, "guitar": paint_guitar, "note": paint_note}


def paint_voice_art(p: QPainter, clip: QPainterPath, rect: QRectF, style: theme.VoiceStyle,
                    pixmap: QPixmap | None) -> None:
    """在 clip 里画角色美术：有头像就完整显示整张图，没有就画角色色背景和图案。"""
    p.save()
    p.setClipPath(clip)
    if pixmap is not None and not pixmap.isNull():
        p.fillRect(rect, QColor(theme.RAISED))
        scale = max(rect.width() / pixmap.width(), rect.height() / pixmap.height())
        if abs(pixmap.width() - pixmap.height()) > 2:  # 非正方形的图等比缩小，完整保留面部
            scale = min(rect.width() / pixmap.width(), rect.height() / pixmap.height())
        w, h = pixmap.width() * scale, pixmap.height() * scale
        target = QRectF(rect.center().x() - w / 2, rect.center().y() - h / 2, w, h)
        p.drawPixmap(target, pixmap, QRectF(pixmap.rect()))
    else:
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QColor(style.color).darker(130))
        gradient.setColorAt(1, QColor(theme.RAISED))
        p.fillRect(rect, gradient)
        glow = QRadialGradient(rect.center(), rect.width() * 0.55)
        glow.setColorAt(0, QColor(255, 255, 255, 70))
        glow.setColorAt(1, QColor(255, 255, 255, 0))
        p.fillRect(rect, glow)
        side = rect.width() * 0.56
        art = QRectF(rect.center().x() - side / 2, rect.center().y() - side / 2, side, side)
        ART_PAINTERS.get(style.art, paint_note)(p, art, QColor(theme.WHITE))
    p.restore()


def app_pixmap(side: int = 256) -> QPixmap:
    """没有角色头像时的程序图标：粉色圆角方块上一支白色麦克风。"""
    pixmap = QPixmap(side, side)
    pixmap.fill(Qt.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.Antialiasing)
    rect = QRectF(side * .04, side * .04, side * .92, side * .92)
    gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
    gradient.setColorAt(0, QColor(theme.SAKURA))
    gradient.setColorAt(1, QColor(theme.SAKURA_DEEP))
    p.setPen(Qt.NoPen)
    p.setBrush(gradient)
    p.drawRoundedRect(rect, side * .22, side * .22)
    mic = side * .56
    paint_mic(p, QRectF((side - mic) / 2, (side - mic) / 2, mic, mic), QColor(theme.WHITE))
    p.end()
    return pixmap


def make_icon(kind: str, color: str, size: int = 18) -> QIcon:
    """播放、停止、刷新小图标，按 3 倍像素画，缩放后保持清晰。"""
    ratio = 3
    pixmap = QPixmap(size * ratio, size * ratio)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    s = float(size)
    if kind == "play":
        path = QPainterPath()
        path.moveTo(s * 0.28, s * 0.16)
        path.lineTo(s * 0.84, s * 0.5)
        path.lineTo(s * 0.28, s * 0.84)
        path.closeSubpath()
        p.drawPath(path)
    elif kind == "stop":
        p.drawRoundedRect(QRectF(s * 0.22, s * 0.22, s * 0.56, s * 0.56), s * 0.1, s * 0.1)
    elif kind == "refresh":
        p.setBrush(Qt.NoBrush)
        p.setPen(_pen(color, s * 0.11))
        p.drawArc(QRectF(s * 0.2, s * 0.2, s * 0.6, s * 0.6), 40 * 16, 280 * 16)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(color))
        tip = QPainterPath()
        tip.moveTo(s * 0.86, s * 0.14)
        tip.lineTo(s * 0.86, s * 0.46)
        tip.lineTo(s * 0.56, s * 0.38)
        tip.closeSubpath()
        p.drawPath(tip)
    elif kind == "gear":
        p.translate(s / 2, s / 2)
        for _ in range(8):
            p.drawRoundedRect(QRectF(-s * 0.07, -s * 0.46, s * 0.14, s * 0.2), s * 0.03, s * 0.03)
            p.rotate(45)
        ring = QPainterPath()
        ring.addEllipse(QPointF(0, 0), s * 0.32, s * 0.32)
        hole = QPainterPath()
        hole.addEllipse(QPointF(0, 0), s * 0.13, s * 0.13)
        p.drawPath(ring.subtracted(hole))
    p.end()
    return QIcon(pixmap)


# ---- 控件 -------------------------------------------------------------------
class Card(QFrame):
    """一块圆角底板，内容放进 body。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(20, 18, 20, 18)
        self.body.setSpacing(12)


def divider() -> QFrame:
    line = QFrame()
    line.setObjectName("divider")
    return line


def label(text: str, name: str = "muted") -> QLabel:
    widget = QLabel(text)
    widget.setObjectName(name)
    return widget


class ConsoleCombo(QComboBox):
    """下拉框，自己画箭头，长设备名在列表里完整显示。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(12)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.view().setTextElideMode(Qt.ElideNone)

    def showPopup(self):
        width = max((self.fontMetrics().horizontalAdvance(self.itemText(i)) for i in range(self.count())),
                    default=0) + 40
        self.view().setMinimumWidth(max(self.width(), width))
        super().showPopup()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(_pen(theme.MUTED if self.isEnabled() else theme.DIM, 1.8))
        cx, cy = self.width() - 16, self.height() / 2
        p.drawLine(QPointF(cx - 4, cy - 2), QPointF(cx, cy + 2.5))
        p.drawLine(QPointF(cx, cy + 2.5), QPointF(cx + 4, cy - 2))


class Segmented(QWidget):
    """两三个互斥选项排成一条。接口与下拉框一致：addItem、currentData、currentText、currentIndexChanged。"""

    currentIndexChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._data: list = []
        self._group.idToggled.connect(lambda index, checked: checked and self.currentIndexChanged.emit(index))

    def addItem(self, text: str, data=None) -> None:
        button = QPushButton(text)
        button.setCheckable(True)
        button.setCursor(Qt.PointingHandCursor)
        self._group.addButton(button, len(self._data))
        self._data.append(data)
        self._row.addWidget(button)
        buttons = self._group.buttons()
        for index, item in enumerate(buttons):
            item.setObjectName("segSolo" if len(buttons) == 1 else "segLeft" if index == 0 else
                               "segRight" if index == len(buttons) - 1 else "segMiddle")
            item.style().unpolish(item)
            item.style().polish(item)
        if len(buttons) == 1:
            button.setChecked(True)

    def clear(self) -> None:
        for button in self._group.buttons():
            self._group.removeButton(button)
            button.hide()
            button.deleteLater()
        self._data = []

    def count(self) -> int:
        return len(self._data)

    def currentIndex(self) -> int:
        return self._group.checkedId()

    def setCurrentIndex(self, index: int) -> None:
        self._group.button(index).setChecked(True)

    def currentData(self):
        return self._data[self.currentIndex()] if self._data else None

    def findData(self, data) -> int:
        return self._data.index(data) if data in self._data else -1

    def currentText(self) -> str:
        button = self._group.checkedButton()
        return button.text() if button is not None else ""

    def button(self, index: int) -> QPushButton:
        return self._group.button(index)


class HeroArt(QWidget):
    """当前角色的主视觉，带角色色描边；变声进行时描边变成直播红。"""

    def __init__(self, side: int = 280, parent=None):
        super().__init__(parent)
        self._style = theme.brand_style()
        self._pixmap: QPixmap | None = None
        self._on_air = False
        self._side = side
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(self._side, self._side)

    def set_voice(self, style: theme.VoiceStyle, pixmap: QPixmap | None) -> None:
        self._style, self._pixmap = style, pixmap
        self.update()

    def set_on_air(self, on_air: bool) -> None:
        if on_air != self._on_air:
            self._on_air = on_air
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        side = min(self.width(), self.height()) - 8
        rect = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        radius = side * 0.11
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.DANGER if self._on_air else self._style.accent))
        p.drawRoundedRect(rect.adjusted(-3, -3, 3, 3), radius + 3, radius + 3)
        clip = QPainterPath()
        clip.addRoundedRect(rect, radius, radius)
        paint_voice_art(p, clip, rect, self._style, self._pixmap)


def blurred(pixmap: QPixmap, radius: float) -> QPixmap:
    """用 Qt 的模糊效果画一张模糊版本，给两侧没选中的角色用。"""
    if pixmap is None or pixmap.isNull():
        return pixmap
    scene = QGraphicsScene()
    item = QGraphicsPixmapItem(pixmap)
    effect = QGraphicsBlurEffect()
    effect.setBlurRadius(radius)
    effect.setBlurHints(QGraphicsBlurEffect.QualityHint)
    item.setGraphicsEffect(effect)
    scene.addItem(item)
    image = QImage(pixmap.size(), QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
    scene.render(painter, QRectF(image.rect()), QRectF(pixmap.rect()))
    painter.end()
    return QPixmap.fromImage(image)


class ArrowButton(QAbstractButton):
    """左右切换角色用的三角形按钮。"""

    def __init__(self, direction: int, parent=None):
        super().__init__(parent)
        self.direction = direction  # -1 向左，+1 向右
        self.setFixedSize(26, 44)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover)
        self.setAccessibleName("上一位角色" if direction < 0 else "下一位角色")
        self.setToolTip(self.accessibleName())

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = QColor(theme.SAKURA_DEEP if self.isEnabled() else theme.LINE)
        if self.isEnabled() and not (self.underMouse() or self.hasFocus()):
            color.setAlpha(170)
        w, h = self.width(), self.height()
        tip = 6 if self.direction < 0 else w - 6
        back = w - 8 if self.direction < 0 else 8
        path = QPainterPath()
        path.moveTo(tip, h / 2)
        path.lineTo(back, h / 2 - 11)
        path.lineTo(back, h / 2 + 11)
        path.closeSubpath()
        p.setPen(_pen(color, 3))
        p.setBrush(color)
        p.drawPath(path)


class VoiceStage(QWidget):
    """角色转盘：选中的角色居中放大，前后两位缩小、虚化放在两侧，切换时平滑滑动。"""

    chosen = Signal(str)
    CENTER = 78
    SIDE = 52

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[dict] = []
        self._index = 0
        self._offset = 0.0
        self.setMinimumSize(220, self.CENTER + 34)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(220)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.valueChanged.connect(self._on_offset)

    def sizeHint(self) -> QSize:
        return QSize(260, self.CENTER + 34)

    def set_items(self, items: list[dict], current: str | None) -> None:
        """items: [{id, name, style, pixmap}]。头像的清晰版和模糊版在这里预先做好。"""
        self._animation.stop()
        self._offset = 0.0
        self._items = []
        for item in items:
            sharp = item["pixmap"]
            if sharp is not None and not sharp.isNull():
                side = min(sharp.width(), sharp.height())
                sharp = sharp.copy((sharp.width() - side) // 2, (sharp.height() - side) // 2, side, side).scaled(
                    self.CENTER * 3, self.CENTER * 3, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._items.append({**item, "sharp": sharp, "soft": blurred(sharp, 7)})
        ids = [i["id"] for i in self._items]
        self._index = ids.index(current) if current in ids else 0
        self.update()

    def ids(self) -> list[str]:
        return [item["id"] for item in self._items]

    def current_id(self) -> str | None:
        return self._items[self._index]["id"] if self._items else None

    def set_current(self, voice_id: str, animate: bool = True) -> None:
        ids = self.ids()
        if voice_id not in ids:
            return
        index = ids.index(voice_id)
        if index == self._index:
            return
        start = (index - self._index) + self._offset
        self._index = index
        if animate:
            self._animation.stop()
            self._animation.setStartValue(float(start))
            self._animation.setEndValue(0.0)
            self._animation.start()
        else:
            self._offset = 0.0
            self.update()

    def _on_offset(self, value) -> None:
        self._offset = float(value)
        self.update()

    def _slot(self, position: float) -> tuple[QRectF, float]:
        """转盘上某个位置（0 在中间，±1 在两侧）对应的头像方框和清晰程度。"""
        far = min(abs(position), 1.0)
        size = self.CENTER + (self.SIDE - self.CENTER) * far
        spacing = min(self.width() * 0.33, 96)
        x = self.width() / 2 + position * spacing
        top = 4 + (self.CENTER - size) / 2
        return QRectF(x - size / 2, top, size, size), 1.0 - far

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        order = sorted(range(len(self._items)), key=lambda i: -abs(i - self._index + self._offset))
        for i in order:  # 远的先画，中间的盖在上面
            position = i - self._index + self._offset
            if abs(position) > 1.6:
                continue
            item = self._items[i]
            rect, sharpness = self._slot(position)
            fade = max(0.0, 1.0 - max(0.0, abs(position) - 1.0) / 0.6)
            p.save()
            p.setOpacity((0.62 + 0.38 * sharpness) * fade)
            clip = QPainterPath()
            clip.addEllipse(rect)
            p.setClipPath(clip)
            if item["sharp"] is not None and not item["sharp"].isNull():
                p.drawPixmap(rect, item["soft"], QRectF(item["soft"].rect()))
                p.setOpacity((0.62 + 0.38 * sharpness) * fade * sharpness)
                p.drawPixmap(rect, item["sharp"], QRectF(item["sharp"].rect()))
            else:
                paint_voice_art(p, clip, rect, item["style"], None)
            p.restore()
            p.save()
            p.setOpacity(fade)
            if sharpness > 0.5:
                p.setPen(_pen(item["style"].color, 3))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(rect.adjusted(-3, -3, 3, 3))
            else:
                p.setPen(_pen(theme.LINE, 1.5))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(rect)
            font = p.font()
            font.setPointSizeF(10 if sharpness > 0.5 else 8.5)
            font.setBold(sharpness > 0.5)
            p.setFont(font)
            p.setPen(QColor(theme.TEXT if sharpness > 0.5 else theme.MUTED))
            text_rect = QRectF(rect.center().x() - 60, 4 + self.CENTER + 6, 120, 22)
            p.drawText(text_rect, Qt.AlignHCenter | Qt.AlignTop, item["name"])
            p.restore()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        for i in range(len(self._items)):
            position = i - self._index + self._offset
            if abs(position) <= 1.05:
                rect, _ = self._slot(position)
                if rect.adjusted(-6, -6, 6, 28).contains(event.position()):
                    self.chosen.emit(self._items[i]["id"])
                    return


class VoiceCarousel(QWidget):
    """一个分类里的角色转盘：◀ 两侧虚化的角色 · 居中的当前角色 ▶。"""

    chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self.left = ArrowButton(-1)
        self.right = ArrowButton(1)
        self.stage = VoiceStage()
        row.addWidget(self.left, 0, Qt.AlignTop)
        row.addWidget(self.stage, 1)
        row.addWidget(self.right, 0, Qt.AlignTop)
        self.left.setContentsMargins(0, 0, 0, 0)
        for arrow in (self.left, self.right):  # 三角形与中间头像垂直居中
            arrow.setFixedHeight(VoiceStage.CENTER + 8)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("角色切换")
        self.left.clicked.connect(lambda: self.step(-1))
        self.right.clicked.connect(lambda: self.step(1))
        self.stage.chosen.connect(self.chosen)
        self._sync_arrows()

    def set_items(self, items: list[dict], current: str | None) -> None:
        self.stage.set_items(items, current)
        self._sync_arrows()

    def set_current(self, voice_id: str, animate: bool = True) -> None:
        self.stage.set_current(voice_id, animate)
        self._sync_arrows()

    def ids(self) -> list[str]:
        return self.stage.ids()

    def current_id(self) -> str | None:
        return self.stage.current_id()

    def step(self, direction: int) -> None:
        ids, current = self.ids(), self.current_id()
        if current is None:
            return
        index = ids.index(current) + direction
        if 0 <= index < len(ids):
            self.chosen.emit(ids[index])

    def _sync_arrows(self) -> None:
        ids, current = self.ids(), self.current_id()
        index = ids.index(current) if current in ids else 0
        self.left.setEnabled(index > 0)
        self.right.setEnabled(index < len(ids) - 1)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.step(-1)
        elif event.key() == Qt.Key_Right:
            self.step(1)
        else:
            super().keyPressEvent(event)


class _PulseButton(QAbstractButton):
    """按下后带脉动光环的圆形开关。"""

    def __init__(self, side: int, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(side, side)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover)
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)
        self.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool) -> None:
        if checked:
            self._timer.start()
        else:
            self._timer.stop()
            self._phase = 0.0
        self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.05) % 1.0
        self.update()


class LiveButton(_PulseButton):
    """实时变声的总开关：一个大圆键，打开后填充角色色。"""

    def __init__(self, parent=None):
        super().__init__(116, parent)
        self._color = QColor(theme.SAKURA)
        self.setAccessibleName("实时变声")

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        center = QPointF(self.width() / 2, self.height() / 2)
        radius = self.width() / 2 - 14
        enabled, on = self.isEnabled(), self.isChecked()
        if on:
            glow = QColor(self._color)
            glow.setAlpha(int(130 * (1 - self._phase)))
            p.setPen(_pen(glow, 3))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(center, radius + 2 + 11 * self._phase, radius + 2 + 11 * self._phase)
        if on:
            fill, ink, edge = self._color, QColor(theme.WHITE), self._color
        elif enabled:
            fill, ink = QColor(theme.WHITE), self._color
            edge = self._color if self.underMouse() or self.hasFocus() else QColor(theme.SAKURA)
        else:
            fill, ink, edge = QColor(theme.FIELD), QColor(theme.DIM), QColor(theme.LINE)
        p.setPen(_pen(edge, 2.5))
        p.setBrush(fill)
        p.drawEllipse(center, radius, radius)
        side = radius * 0.95
        paint_mic(p, QRectF(center.x() - side / 2, center.y() - side / 2, side, side), ink)


class RecButton(_PulseButton):
    """录音键。未按下是红色圆点，录音中是方块加脉动光环。"""

    def __init__(self, parent=None):
        super().__init__(64, parent)
        self.setAccessibleName("录音")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        center = QPointF(self.width() / 2, self.height() / 2)
        radius = self.width() / 2 - 8
        enabled = self.isEnabled()
        red = QColor(theme.DANGER if enabled else theme.DIM)
        if self.isChecked():
            glow = QColor(theme.DANGER)
            glow.setAlpha(int(120 * (1 - self._phase)))
            p.setPen(_pen(glow, 3))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(center, radius + 1 + 6 * self._phase, radius + 1 + 6 * self._phase)
        edge = theme.DANGER if enabled and (self.underMouse() or self.hasFocus() or self.isChecked()) else theme.LINE
        p.setPen(_pen(edge if enabled else theme.DIM, 2))
        p.setBrush(QColor(theme.WHITE))
        p.drawEllipse(center, radius, radius)
        p.setPen(Qt.NoPen)
        p.setBrush(red)
        if self.isChecked():
            half = radius * 0.36
            p.drawRoundedRect(QRectF(center.x() - half, center.y() - half, half * 2, half * 2), 4, 4)
        else:
            p.drawEllipse(center, radius * 0.56, radius * 0.56)


class LevelMeter(QWidget):
    """细长的电平条，输入峰值 0–1，按 dB 显示并缓慢回落。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self.setMinimumSize(80, 6)
        self.setMaximumHeight(6)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_peak(self, peak: float) -> None:
        level = 0.0 if peak <= 1e-4 else max(0.0, min(1.0, 1 + 20 * math.log10(peak) / 48))
        value = max(level, self._value - 0.06)
        if abs(value - self._value) > 1e-3:
            self._value = value
            self.update()

    def reset(self) -> None:
        self._value = 0.0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect())
        radius = rect.height() / 2
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.FIELD))
        p.drawRoundedRect(rect, radius, radius)
        if self._value > 0.01:
            gradient = QLinearGradient(rect.topLeft(), rect.topRight())
            gradient.setColorAt(0, QColor(theme.SKY))
            gradient.setColorAt(0.7, QColor(theme.SAKURA))
            gradient.setColorAt(0.92, QColor(theme.SUNFLOWER))
            gradient.setColorAt(1, QColor(theme.DANGER))
            p.setBrush(gradient)
            p.setClipRect(QRectF(0, 0, rect.width() * self._value, rect.height()))
            p.drawRoundedRect(rect, radius, radius)


class ToggleSwitch(QAbstractButton):
    """左边一个圆角小开关，右边一行字。和复选框一样用 isChecked / toggled。"""

    TRACK_W, TRACK_H = 34, 20

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(text)
        self._color = QColor(theme.SAKURA_DEEP)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self.TRACK_W + 10 + self.fontMetrics().horizontalAdvance(self.text()) + 4,
                     max(self.TRACK_H + 6, self.fontMetrics().height() + 6))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        enabled, on = self.isEnabled(), self.isChecked()
        track = QRectF(1, (self.height() - self.TRACK_H) / 2, self.TRACK_W, self.TRACK_H)
        if on:
            fill = QColor(self._color) if enabled else QColor(theme.DIM)
            edge = fill
        else:
            fill = QColor(theme.FIELD)
            edge = QColor(self._color if enabled and (self.underMouse() or self.hasFocus()) else theme.LINE)
        p.setPen(_pen(edge, 1.2))
        p.setBrush(fill)
        p.drawRoundedRect(track, track.height() / 2, track.height() / 2)
        knob = track.height() - 6
        x = track.right() - 3 - knob if on else track.left() + 3
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.WHITE if on else (theme.MUTED if enabled else theme.DIM)))
        p.drawEllipse(QRectF(x, track.top() + 3, knob, knob))
        p.setPen(QColor(theme.TEXT if enabled else theme.DIM))
        text_rect = QRectF(track.right() + 10, 0, self.width() - track.right() - 10, self.height())
        p.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self.text())


def set_state(widget: QWidget, state: str) -> None:
    """改动态属性 state 并让样式表重新生效。"""
    if widget.property("state") != state:
        widget.setProperty("state", state)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()
