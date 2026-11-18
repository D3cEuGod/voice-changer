"""入口：python -m gui [--backend-url URL] [--no-autostart] [--backend-script PATH]"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from .backend import (DEFAULT_URL, DEFAULT_PITCH_PRESET, PITCH_PRESETS, ROOT, START_SCRIPT, BackendClient,
                      start_early)

from . import backend as backend_module
from .settings import Settings, default_path
from .taskbar import set_process_app_id, set_window_relaunch

LOG_FILE = ROOT / ".run" / "gui.log"
_APPLICATION = None


def cleanup_qt_styles():
    """Release Qt's stylesheet proxies before widget wrappers are torn down."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is not None:
        app.setStyleSheet("")


def create_app(argv=None):
    """先创建 QApplication，再导入会加载 PortAudio 的模块（顺序反了会让 Qt 的原生文件对话框失效）。"""
    global _APPLICATION
    set_process_app_id()  # 要在创建任何窗口之前
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QGuiApplication
    from PySide6.QtWidgets import QApplication

    from . import theme

    # 125%、150% 等非整数缩放按系统比例原样渲染
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(list(argv or sys.argv[:1]))
    # Keep Qt alive while main() releases its window wrappers and Python runs
    # PortAudio's exit handlers. Destroying QApplication first can crash QtWidgets.
    _APPLICATION = app
    if not app.property("voiceQuitCleanupConnected"):
        app.aboutToQuit.connect(cleanup_qt_styles)
        app.setProperty("voiceQuitCleanupConnected", True)
    app.setApplicationName("角色变声器")
    app.setStyle("Fusion")
    font = QFont("Microsoft YaHei UI", 10)
    font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(font)
    app.setStyleSheet(theme.STYLESHEET)
    return app


def _log_exception(kind, value, trace) -> None:
    text = "".join(traceback.format_exception(kind, value, trace))
    try:
        LOG_FILE.parent.mkdir(exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as log:
            log.write(text + "\n")
    except OSError:
        pass
    if sys.stderr is not None:
        sys.stderr.write(text)
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "角色变声器出错了", f"{value}\n\n详细信息已写入 {LOG_FILE}")
    except Exception:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gui", description="角色变声器桌面界面")
    parser.add_argument("--backend-url", default=os.environ.get("VOICE_GUI_BACKEND_URL", DEFAULT_URL),
                        help=f"后端地址，默认 {DEFAULT_URL}")
    parser.add_argument("--no-autostart", action="store_true", help="后端未运行时不自动启动")
    parser.add_argument("--backend-script", default=str(START_SCRIPT), help="启动后端用的脚本")
    parser.add_argument("--live-restart-packets", type=int, choices=(0, 3000, 5000), default=None,
                        help="自动重开实时变声的包数，默认用设置里的值（3000）；0 关闭自动刷新")
    parser.add_argument("--live-packet-ms", type=int, choices=(100,120), default=None,
                        help="VB-CABLE 包长；默认用设置里的值（稳定档 120），100 为实验低延迟档")
    parser.add_argument("--pitch-preset", choices=tuple(PITCH_PRESETS), default=None,
                        help="音高：默认用上次的选择（首次为 auto2 自动音域再升 2），auto4 升 4，model 恢复模型默认")
    parser.add_argument("--settings", default=None,
                        help="设置文件路径，默认 %%APPDATA%%\\CharacterVoiceChanger\\settings.json")
    parser.add_argument("--fresh", action="store_true", help="这次不读也不保存设置（排查问题用）")
    args = parser.parse_args(argv)
    settings = Settings.load(None if args.fresh else Path(args.settings) if args.settings else default_path())
    if settings.pitch not in PITCH_PRESETS:
        settings.pitch = DEFAULT_PITCH_PRESET
    pitch = args.pitch_preset or settings.pitch
    if settings.voice_id:  # 上次用的声线先加载，进主界面更快
        backend_module.WARM_FIRST = settings.voice_id
    restart = settings.live_restart_packets if args.live_restart_packets is None else args.live_restart_packets
    packet_ms = settings.live_packet_ms if args.live_packet_ms is None else args.live_packet_ms

    sys.excepthook = _log_exception
    # 先把后端拉起来，再加载界面程序库：两边同时进行，能省下界面初始化的那一两秒。
    prestarted = None if args.no_autostart else start_early(BackendClient(args.backend_url),
                                                            Path(args.backend_script))
    app = create_app()
    from .app import Config, MainWindow
    from .splash import StartupWindow

    splash = StartupWindow()
    set_window_relaunch(int(splash.winId()))
    splash.show()
    app.processEvents()  # 先把加载页画出来，再初始化主窗口
    window = MainWindow(Config(args.backend_url, not args.no_autostart, Path(args.backend_script),
                               restart, packet_ms, pitch, prestarted, settings=settings, desktop=True))
    available = app.primaryScreen().availableGeometry()
    window.resize(min(1060, available.width() - 40), min(700, available.height() - 60))
    set_window_relaunch(int(window.winId()))  # 在窗口显示之前写入，任务栏按这些信息建立图标

    def enter_main() -> None:
        if window.isVisible():
            return
        window.show()
        window.raise_()
        window.activateWindow()
        splash.close_for_main()
        window.maybe_onboard()

    def leave() -> None:
        splash.close_for_main()
        window.quit_app()  # 停掉由本程序启动的后端
        app.quit()

    window.startup_stage.connect(splash.set_stage)
    window.startup_finished.connect(lambda ok, message: splash.finish() if ok else splash.fail(message))
    splash.done.connect(enter_main)
    splash.open_anyway.connect(enter_main)
    splash.retry_requested.connect(window.connect_backend)
    splash.quit_requested.connect(leave)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
