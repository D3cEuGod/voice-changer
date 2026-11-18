"""自检：打开真实窗口，用真实后端把主要流程跑一遍，逐项打印 PASS / FAIL / SKIP。

运行（后端最好已在运行）：
    .venv-gui\\Scripts\\python.exe -m gui.selfcheck

会用到的设备和文件：
- 文件转换用 assets/previews/original.wav，结果和截图写到 .run/selfcheck/。
- 试听从系统默认输出设备播放（会出声）；其余重复播放尽量送到虚拟输出设备。
- 录音和监听需要一个有信号的麦克风。脚本会寻找“虚拟音频线”（一对名字相同的虚拟输出和虚拟麦克风，
  例如 Steam Streaming Microphone），把 original.wav 播进去当作麦克风信号；找不到就跳过这些项目。
- “设备被拔出”“没有设备”“引擎正忙”“后端未运行”“启动脚本失败”都是模拟出来的，输出里会注明。
"""
from __future__ import annotations

import argparse
import ctypes
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .__main__ import create_app, cleanup_qt_styles
from .backend import DEFAULT_URL, ROOT, BackendClient, BackendError

ORIGINAL = ROOT / "assets" / "previews" / "original.wav"
ALT_URL = "http://127.0.0.1:8766"
DEAD_URL = "http://127.0.0.1:8799"


class Checker:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def _add(self, status: str, name: str, detail: str) -> None:
        self.rows.append((status, name, detail))
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    def check(self, name: str, ok, detail: str = "") -> bool:
        self._add("PASS" if ok else "FAIL", name, detail)
        return bool(ok)

    def skip(self, name: str, why: str) -> None:
        self._add("SKIP", name, why)

    def summary(self) -> int:
        counts = {s: sum(1 for row in self.rows if row[0] == s) for s in ("PASS", "FAIL", "SKIP")}
        print(f"\n合计：{counts['PASS']} 项通过，{counts['FAIL']} 项失败，{counts['SKIP']} 项跳过", flush=True)
        for status, name, detail in self.rows:
            if status != "PASS":
                print(f"  {status}: {name} — {detail}")
        return 1 if counts["FAIL"] else 0


class SilentMic:
    """模拟麦克风：按实时速度送出静音块。用来占住引擎或模拟设备掉线。"""

    def __init__(self, rate: int = 48000):
        self.rate, self.name, self.level, self.dropped = rate, "模拟麦克风（静音）", 0.0, 0
        self.blocks: queue.Queue = queue.Queue(maxsize=75)
        self.active = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        block = np.zeros(int(self.rate * 0.02), dtype=np.float32)
        while self.active:
            time.sleep(0.02)
            try:
                self.blocks.put_nowait(block)
            except queue.Full:
                self.dropped += 1

    def close(self):
        self.active = False


class NullSpeaker:
    rate, name, active, underruns, buffered_seconds = 22050, "模拟输出（丢弃）", True, 0, 0.0

    def write(self, samples):
        pass

    def close(self):
        self.active = False


def _check_native_dialog(window, c) -> None:
    """打开真实的 Windows“导入音频”对话框，由后台线程发 WM_CLOSE 关掉，确认没有改动输入。"""
    user32 = ctypes.windll.user32
    seen = {"dialog": False}

    def close_native_dialog():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            hwnd = user32.FindWindowW("#32770", "导入音频")
            if hwnd:
                seen["dialog"] = True
                time.sleep(0.6)
                user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
                return
            time.sleep(0.2)

    threading.Thread(target=close_native_dialog, daemon=True).start()
    before = window.input
    window.import_dialog()
    c.check("原生“导入音频”对话框能打开并取消", seen["dialog"] and window.input is before,
            "对话框由脚本发送关闭消息，未选择文件")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gui.selfcheck")
    parser.add_argument("--out", default=str(ROOT / ".run" / "selfcheck"), help="保存结果和截图的目录")
    parser.add_argument("--skip-autostart", action="store_true", help="不在 8766 端口另起一个后端来测试自动启动")
    parser.add_argument("--backend-url", default=DEFAULT_URL, help=f"被测后端的地址，默认 {DEFAULT_URL}")
    parser.add_argument("--allow-vb-cable", action="store_true",
                        help="允许把 VB-CABLE 当成测试用的虚拟音频线。它通常是 Discord 的麦克风，测试音频会进入通话")
    parser.add_argument("--output", default="",
                        help="试听和监听用的输出设备（名称片段）。默认用系统默认输出；系统默认输出是 Discord 用的虚拟线时，"
                             "请指定别的设备，以免测试音频进入通话")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    app = create_app()
    import soundfile as sf
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QFileDialog, QPushButton

    from . import audio
    from .app import Config, MainWindow
    from .live import LiveMonitor

    c = Checker()
    client = BackendClient(args.backend_url)

    def pump(milliseconds: float) -> None:
        """处理界面事件并等待。不用 QTest.qWait：它等待时一直占着 GIL，音频回调会被饿死，
        而程序正常运行时（app.exec）不会这样。"""
        end = time.monotonic() + milliseconds / 1000
        while True:
            app.processEvents()
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.005, remaining))

    def wait(condition, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if condition():
                return True
            pump(40)
        return bool(condition())

    def shot(widget, name: str) -> None:
        pump(150)
        widget.grab().save(str(out_dir / f"{name}.png"))

    def choose(combo, fragment: str):
        for i in range(combo.count()):
            device = combo.itemData(i)
            if device is not None and fragment in device.name:
                combo.setCurrentIndex(i)
                return device
        return None

    def click(widget) -> None:
        top = widget.window()
        if hasattr(top, "reveal"):  # 控件可能在没显示的那一页上
            top.reveal(widget)
            pump(30)
        QTest.mouseClick(widget, Qt.LeftButton)
        pump(60)

    def pick(win, voice_id: str) -> None:
        """像用户一样换角色：先点分类，再点转盘两侧的三角形，直到这位角色居中。"""
        category = next(cat for cat in win.categories if voice_id in cat.voices)
        index = win.categories.index(category)
        if win.category_bar.currentIndex() != index:
            click(win.category_bar.button(index))
        for _ in range(len(category.voices)):
            current = win.carousel.current_id()
            if current == voice_id:
                break
            order = win.carousel.ids()
            click(win.carousel.right if order.index(voice_id) > order.index(current) else win.carousel.left)
        pump(260)  # 等转盘动画走完

    # ------------------------------------------------------------------ 启动
    window = MainWindow(Config(args.backend_url))
    # 只开耳返：等同以前的“本机监听”，测试音频不会进入语音软件
    window.send_toggle.setChecked(False)
    window.monitor_toggle.setChecked(True)
    window.resize(1060, 700)
    window.show()
    wait(lambda: bool(window.voice_names) or window.notice.isVisible(), 30)
    how = "由自检启动" if window.owned_backend is not None else "复用已运行的后端"
    if not c.check("启动：后端就绪", window.backend_ok, f"{how}；{window.status_label.text()}"):
        print(window.notice.text())
        window.close()
        return c.summary()
    api_voices = client.voices()
    model_labels = client.health().get("voice_models", {})
    ids = [v["id"] for v in api_voices]
    c.check("角色列表来自 GET /api/voices", window.voices == api_voices and list(window.voice_names) == ids,
            "、".join(f"{v['id']}={v['name']}" for v in api_voices))
    c.check("默认选中永雏塔菲，主视觉是头像",
            window.voice_id == "taffy" and window.hero_name.text() == "永雏塔菲" and window._avatar is not None,
            f"头像 {window._avatar.width()}×{window._avatar.height()}" if window._avatar is not None else "头像没有加载")
    shot(window, "01_startup_taffy")

    inputs, outputs = audio.list_devices("input"), audio.list_devices("output")
    print("麦克风：", [d.name for d in inputs])
    print("输出设备：", [d.name for d in outputs])
    default_out = next((d for d in outputs if d.is_default), outputs[0] if outputs else None)
    if args.output:
        default_out = next((d for d in outputs if args.output in d.name), None)
        if default_out is None:
            print(f"找不到名称包含“{args.output}”的输出设备")
            window.close()
            return 2
    print("试听和监听输出：", default_out.name if default_out else "没有")
    def loops_back(mic, out) -> bool:
        """往 out 播一小段 440 Hz，看 mic 能不能录到，确认这一对设备确实是一根虚拟音频线。"""
        tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(22050) / 22050)).astype(np.float32)
        try:
            recorder = audio.Recorder(mic, 5)
            recorder.start()
            tone_player = audio.ClipPlayer()
            tone_player.play(tone, 22050, out)
            pump(700)
            tone_player.stop()
            heard = recorder.stop()
        except audio.AudioDeviceError:
            return False
        return len(heard) > 0 and float(np.max(np.abs(heard))) > 0.05

    cable_in = cable_out = None
    for mic in inputs:
        # VB-CABLE 是 Discord 用的麦克风：往里播测试音频，通话里的人会听到。除非明确允许，不拿它当测试线。
        if "cable" in mic.name.casefold() and (not args.allow_vb_cable
                or default_out and "cable" in default_out.name.casefold()):
            continue
        inner = mic.name[mic.name.find("(") + 1:mic.name.rfind(")")] if "(" in mic.name else mic.name
        partner = next((o for o in outputs if inner and inner in o.name and not o.is_default), None)
        # A conversion test must use a different virtual pair as its source;
        # using the target cable as both microphone and output would feed back.
        if partner is not None and partner != default_out and loops_back(mic, partner):
            cable_in, cable_out = mic, partner
            break
    print("虚拟音频线：", f"{cable_out.name} → {cable_in.name}" if cable_in else "没有找到")
    # 第二个试听输出。指定了 --output 时只在名称匹配的设备里挑，绝不落到扬声器或耳机上。
    allowed = [o for o in outputs if args.output in o.name] if args.output else outputs
    quiet_out = next((o for o in allowed if o is not default_out and o is not cable_out
                      and "cable" not in o.name.casefold()), None) or default_out
    combo_names = [window.mic_combo.itemText(i) for i in range(window.mic_combo.count())]
    c.check("设备下拉框显示系统报告的完整名称",
            all(any(d.name in text for text in combo_names) for d in inputs) and bool(inputs) and bool(outputs),
            f"{len(inputs)} 个麦克风，{len(outputs)} 个输出设备")

    # ------------------------------------------------------------------ 角色分类与切换
    groups = {cat.id: cat.voices for cat in window.categories}
    c.check("角色按分类显示：MyGO 一类，V圈一类", groups.get("mygo") == ("tomori", "anon")
            and groups.get("vtuber") == ("taffy",) and window.category_bar.count() == len(groups),
            "；".join(f"{cat.name}: {'、'.join(cat.voices)}" for cat in window.categories))
    if 'tannhauser' in window.voice_names:
        c.check('赛马娘分类包含诗歌剧',groups.get('umamusume') == ('tannhauser',)
                and window.voice_names['tannhauser'] == '诗歌剧')
    c.check("默认分类是永雏塔菲所在的 V圈，转盘只显示这一类", window.category_id == "vtuber"
            and window.carousel.ids() == ["taffy"] and not window.carousel.left.isEnabled()
            and not window.carousel.right.isEnabled())
    click(window.category_bar.button(window.categories.index(next(cat for cat in window.categories if cat.id == "mygo"))))
    pump(300)
    c.check("切到 MyGO 分类：转盘换成灯和爱音，选中灯居中", window.voice_id == "tomori"
            and window.carousel.ids() == ["tomori", "anon"] and window.carousel.current_id() == "tomori"
            and not window.carousel.left.isEnabled() and window.carousel.right.isEnabled(), window.hero_name.text())
    shot(window, "02_voice_tomori")
    click(window.carousel.right)
    pump(300)
    c.check("点右侧三角切到爱音，灯移到左侧", window.voice_id == "anon" and window.carousel.current_id() == "anon"
            and window.carousel.left.isEnabled() and not window.carousel.right.isEnabled(), window.hero_name.text())
    shot(window, "02_voice_anon")
    window.carousel.setFocus()
    QTest.keyClick(window.carousel, Qt.Key_Left)
    pump(300)
    c.check("键盘左键切回灯", window.voice_id == "tomori")
    stage = window.carousel.stage
    side_rect, _ = stage._slot(1.0)
    QTest.mouseClick(stage, Qt.LeftButton, pos=side_rect.center().toPoint())
    pump(300)
    c.check("点转盘侧边虚化的爱音也能切换", window.voice_id == "anon")
    click(window.category_bar.button(window.categories.index(next(cat for cat in window.categories if cat.id == "vtuber"))))
    pump(300)
    click(window.category_bar.button(window.categories.index(next(cat for cat in window.categories if cat.id == "mygo"))))
    pump(300)
    c.check("切走再切回 MyGO 时选回上次用的爱音", window.voice_id == "anon" and window.carousel.current_id() == "anon")
    for voice in api_voices:
        pick(window, voice["id"])
        c.check(f"切换角色：{voice['name']}", window.voice_id == voice["id"] and window.hero_name.text() == voice["name"]
                and window.carousel.current_id() == voice["id"], window.hero_caption.text())
        c.check(f"角色原声与处理方式：{voice['name']}", window.reference_button.isEnabled()
                and window.voice_mode.isVisible() and window.current_mode() == "character"
                and window.hero_name.toolTip() == f"模型：{model_labels.get(voice['id'], voice['id'].upper())}",
                window.hero_name.toolTip())
        c.check(f"默认自动音域再 +2：{voice['name']}", window.current_pitch() == "auto2"
                and window.pitch_combo.count() == 3 and window.pitch_combo.findData("fixed4") == -1)
    pick(window, "taffy")
    c.check("示例按钮已经去掉", not hasattr(window, "original_button")
            and all(not b.text().startswith("示例") for b in window.findChildren(QPushButton)))

    # ------------------------------------------------------------------ 角色原声
    if default_out is None:
        c.skip("角色原声", "没有输出设备")
    else:
        choose(window.out_combo, default_out.name)
        click(window.reference_button)
        started = wait(lambda: window.playing_tag == ("preview", "reference") and window.player.playing, 3)
        pump(900)
        position = window.player.position
        click(window.reference_button)
        c.check("角色原声：点一下立即播放，再点停止", started and position > 0.5 and window.playing_tag is None,
                f"taffy 的原声在“{default_out.name}”上播了 {position:.2f} 秒后停止（脚本听不到声音，只确认音频按时送出）")
        choose(window.out_combo, quiet_out.name)
        click(window.reference_button)
        ok = wait(lambda: window.playing_tag == ("preview", "reference") and window.player.playing, 3)
        pick(window, "anon")
        c.check("播放角色原声，换角色后停止以免错配", ok and window.playing_tag is None)
        click(window.reference_button)
        ok = wait(lambda: window.playing_tag == ("preview", "reference") and window.player.playing, 3)
        pump(500)
        c.check("角色原声：千早爱音", ok and window.player.position > 0.2, f"输出到“{quiet_out.name}”")
        t0 = time.monotonic()
        finished = wait(lambda: window.playing_tag is None, 20)
        c.check("角色原声播完后自动复位", finished, f"播完用了 {time.monotonic() - t0 + 0.5:.1f} 秒")
        pick(window, "taffy")
        choose(window.out_combo, default_out.name)

    # ------------------------------------------------------------------ 原生文件对话框
    # 离屏模式（QT_QPA_PLATFORM=offscreen）没有原生对话框，exec 会一直等下去，所以跳过这一项。
    if QApplication.platformName() == "offscreen":
        c.skip("原生“导入音频”对话框能打开并取消", "离屏模式下没有原生对话框")
    else:
        _check_native_dialog(window, c)

    # ------------------------------------------------------------------ 导入 → 变声 → 试听 → 保存
    real_open, real_save = QFileDialog.getOpenFileName, QFileDialog.getSaveFileName
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(ORIGINAL), ""))
    click(window.import_button)
    info = sf.info(str(ORIGINAL))
    c.check("导入 WAV", window.input is not None and window.input.label == "original.wav"
            and abs(window.input.seconds - info.duration) < 0.01,
            f"{window.input_label.text()}（文件对话框的返回值由脚本代填）" if window.input else "没有输入")
    c.check("导入后还没有结果：结果试听和保存不可用",
            not window.save_button.isEnabled() and not window.seg_result.isEnabled())

    click(window.convert_button)
    busy = wait(lambda: window.converting_key is not None, 2)
    live_blocked = not window.live_switch.isEnabled()
    done = wait(lambda: window.converting_key is None, 180)
    result = window.current_result()
    c.check("提交 POST /api/convert 并拿到结果", busy and done and result is not None,
            window.status_label.text())
    c.check("转换期间监听开关不可用", live_blocked)
    if result is not None:
        c.check("结果与生成参数关联", result.key == (window.input.serial, "taffy", 100, "character", window.current_pitch()) and window.ab == "result"
                and window.result_tag.property("state") == "fresh", window.result_tag.text())
        original_pitch = window.current_pitch()
        window.pitch_combo.setCurrentIndex(window.pitch_combo.findData("model"))
        c.check("切换音高后旧结果不能试听保存", original_pitch == "auto2"
                and window.current_result() is None and not window.save_button.isEnabled())
        window.pitch_combo.setCurrentIndex(window.pitch_combo.findData(original_pitch))
        c.check("切回音高后恢复对应结果", window.current_result() is result and window.save_button.isEnabled())
        window.voice_mode.setCurrentIndex(window.voice_mode.findData("call"))
        c.check("切换处理方式后旧结果不能试听保存", window.current_result() is None
                and not window.save_button.isEnabled())
        window.voice_mode.setCurrentIndex(window.voice_mode.findData("character"))
        c.check("切回原处理方式恢复对应结果", window.current_result() is result
                and window.save_button.isEnabled())
        click(window.seg_result)
        shot(window, "03_converted")
        if default_out is not None:
            click(window.play_button)
            playing = wait(lambda: window.playing_tag is not None and window.playing_tag[0] == "result"
                           and window.player.playing, 3)
            pump(1000)
            at_switch = window.player.position
            click(window.seg_input)
            on_input = (window.playing_tag is not None and window.playing_tag[0] == "input"
                        and window.player.playing and abs(window.player.position - at_switch) < 0.3)
            pump(700)
            click(window.seg_result)
            back = (window.playing_tag is not None and window.playing_tag[0] == "result"
                    and window.player.position > at_switch)
            finished = wait(lambda: window.playing_tag is None, 12)
            c.check("试听结果并在播放中切换 原声 / 结果", playing and on_input and back and finished,
                    f"在“{default_out.name}”上播放；{at_switch:.2f} 秒处切到原声，位置保持，再切回结果播完"
                    "（脚本听不到声音，只确认音频按时送出）")
        saved = out_dir / "selfcheck_original_taffy_100.wav"
        saved.unlink(missing_ok=True)
        QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(saved), ""))
        click(window.save_button)
        ok = saved.is_file() and saved.read_bytes() == result.wav
        detail = ""
        if saved.is_file():
            meta = sf.info(str(saved))
            ok = ok and meta.samplerate == 22050 and meta.channels == 1 and meta.subtype == "PCM_16"
            detail = (f"{saved}：{meta.samplerate} Hz，{meta.channels} 声道，{meta.subtype}，"
                      f"{meta.duration:.2f} 秒，{saved.stat().st_size} 字节")
        c.check("另存为 WAV", ok, detail)

        # 参数改变后，旧结果不能被当成新结果
        window.strength_slider.setValue(60)
        stale_path = out_dir / "should_not_exist.wav"
        stale_path.unlink(missing_ok=True)
        refused = not window.save_result(stale_path) and not stale_path.exists()
        c.check("改强度后旧结果不能试听或保存", window.current_result() is None and not window.save_button.isEnabled()
                and not window.seg_result.isEnabled() and window.result_tag.property("state") == "stale" and refused,
                window.result_tag.text())
        shot(window, "04_stale_after_strength_change")
        window.strength_slider.setValue(100)
        c.check("改回原设置后结果恢复可用", window.current_result() is result and window.save_button.isEnabled())
        pick(window, "anon")
        c.check("换角色后旧结果不能保存", window.current_result() is None and not window.save_button.isEnabled())
        click(window.convert_button)
        wait(lambda: window.converting_key is not None, 2)
        pick(window, "tomori")  # 转换途中改设置
        wait(lambda: window.converting_key is None, 180)
        anon_key = (window.input.serial, "anon", 100, "character", window.current_pitch())
        c.check("转换途中换角色：结果记在提交时的参数下", anon_key in window.results and window.current_result() is None
                and not window.save_button.isEnabled(), window.status_label.text())
        pick(window, "anon")
        c.check("回到千早爱音后可保存她的结果", window.current_result() is window.results.get(anon_key)
                and window.save_button.isEnabled())
        pick(window, "taffy")
        c.check("回到永雏塔菲后仍是她的结果", window.current_result() is result)
        QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(ORIGINAL.with_name("tomori.wav")), ""))
        click(window.import_button)
        c.check("换输入后旧输入的结果全部作废", not window.results and not window.save_button.isEnabled()
                and window.input.label == "tomori.wav", window.result_tag.text())
    QFileDialog.getOpenFileName, QFileDialog.getSaveFileName = real_open, real_save

    # ------------------------------------------------------------------ 无效文件与后端错误
    bad = out_dir / "not_audio.wav"
    bad.write_bytes(b"this is not audio" * 100)
    short = out_dir / "too_short.wav"
    sf.write(str(short), np.zeros(2205, dtype=np.float32), 22050, subtype="PCM_16")
    huge = out_dir / "too_big.wav"
    with huge.open("wb") as handle:
        handle.truncate(window.max_bytes + 1)
    kept = window.input
    messages = []
    for path in (bad, short, huge):
        accepted = window.load_file(path)
        messages.append(f"{path.name} → {window.status_label.text()}")
        c.check(f"拒绝无效文件 {path.name}", not accepted and window.input is kept, window.status_label.text())
    huge.unlink(missing_ok=True)
    for name, call, status in (
            ("后端 422（无法解码）", lambda: client.convert(bad.read_bytes(), "audio/wav", "taffy", 1.0), 422),
            ("后端 422（角色不存在，detail 为数组）", lambda: client.convert(ORIGINAL.read_bytes(), "audio/wav", "nobody", 1.0), 422)):
        try:
            call()
            c.check(name, False, "后端没有报错")
        except BackendError as exc:
            c.check(name, exc.status == status, str(exc))

    # ------------------------------------------------------------------ 录音（虚拟音频线）
    original_samples, original_rate = sf.read(str(ORIGINAL), dtype="float32")
    feeder = audio.ClipPlayer()

    def feed_cable(repeats: int, gap: float = 0) -> None:
        clip = np.r_[original_samples,np.zeros(round(gap*original_rate),np.float32)] if gap else original_samples
        feeder.play(np.tile(clip, repeats), original_rate, cable_out)

    if cable_in is None:
        c.skip("录音", "没有找到有信号的麦克风（虚拟音频线），真实录音未验证")
    else:
        choose(window.mic_combo, cable_in.name)
        feed_cable(2)
        pump(300)
        click(window.rec_button)
        recording = window.recorder is not None and window.lamp.text() == "REC"
        peak = 0.0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.2:
            pump(40)
            peak = max(peak, window.recorder.level if window.recorder else 0.0)
        shot(window, "05_recording")
        click(window.rec_button)
        clip = window.input
        ok = (recording and window.recorder is None and clip is not None and clip.label.startswith("录音")
              and 1.9 < clip.seconds < 2.8 and float(np.max(np.abs(clip.samples))) > 0.02)
        c.check("录音", ok, f"{window.input_label.text()}；峰值 {peak:.2f}；麦克风“{cable_in.name}”的信号是"
                          f"脚本播进“{cable_out.name}”的 original.wav")
        recorded = clip
        click(window.rec_button)
        pump(120)
        click(window.rec_button)
        c.check("过短的录音被丢弃，不替换当前输入", window.input is recorded, window.status_label.text())
        feeder.stop()
        if ok:
            click(window.convert_button)
            wait(lambda: window.converting_key is not None, 2)
            wait(lambda: window.converting_key is None, 180)
            c.check("录音提交转换", window.current_result() is not None, window.status_label.text())

    # ------------------------------------------------------------------ 麦克风监听
    monitors = []
    if cable_in is None or default_out is None:
        c.skip("麦克风监听（真实设备）", "没有找到有信号的麦克风或输出设备")
    else:
        choose(window.mic_combo, cable_in.name)
        choose(window.out_combo, default_out.name)
        window.strength_slider.setValue(100)
        pick(window, "taffy")
        pump(600)
        feed_cable(6)
        click(window.live_switch)
        first = window.live
        monitors.append(first)
        heard: list[np.ndarray] = []
        if first is not None:  # 把送去播放的监听回包另存一份，方便事后回听
            real_write = first.speaker.write

            def tee(samples):
                heard.append(np.array(samples, dtype=np.float32))
                real_write(samples)

            first.speaker.write = tee
        connected = wait(lambda: first is not None and bool(first.hello), 8)
        levels = {"in": 0.0, "out": 0.0}

        def enough(monitor, frames):
            def condition():
                levels["in"] = max(levels["in"], monitor.mic.level)
                levels["out"] = max(levels["out"], getattr(monitor.speaker, "level", 0.0))
                return monitor.frames_sent >= frames
            return condition

        flowing = connected and wait(enough(first, 18), 15)
        c.check("监听：连接 /api/live 并持续收发", flowing and window.lamp.text() == "ON AIR",
                f"{first.hello}；{window.live_status.text()}".replace("\n", "；") if first else "没有启动")
        c.check("实时连接带上所选音高且后端确认", first is not None
                and first.pitch_preset == window.current_pitch()
                and first.hello.get("pitch_shift") is None and first.hello.get("pitch_offset") == 2,
                first.url if first else "没有启动")
        c.check("监听：输入和输出都有信号", levels["in"] > 0.02 and levels["out"] > 0.02,
                f"输入峰值 {levels['in']:.2f}，输出峰值 {levels['out']:.2f}；麦克风“{cable_in.name}”的信号是脚本"
                f"播进“{cable_out.name}”的 original.wav，输出到“{default_out.name}”（脚本听不到声音，只确认音频按时送出）")
        c.check("监听时文件转换和录音被禁用", not window.convert_button.isEnabled() and not window.rec_button.isEnabled(),
                window.convert_button.toolTip())
        shot(window, "06_live_on_air")

        if 'tannhauser' in window.voice_names:
            pick(window,'tannhauser')
            connected = wait(lambda: window.live is not None and window.live is not first
                             and bool(window.live.hello),10)
            horse = window.live
            if horse is not None:
                monitors.append(horse)
            c.check('实时切至诗歌剧：先淡出旧声并完整重开',connected
                    and horse.pitch_preset == 'auto2' and 'voice_id=tannhauser' in horse.url
                    and not first.is_alive() and not first.speaker.active
                    and first.speaker._fade_done.is_set() and len(first.speaker.fifo) == 0)

        pick(window, "anon")
        wait(lambda: window.live is not None and window.live is not first and bool(window.live.hello), 8)
        second = window.live
        monitors.append(second)
        ok = second is not first and second is not None and "voice_id=anon" in second.url and wait(enough(second, 4), 10)
        c.check("监听中切换角色会重新连接", ok and not first.is_alive() and not first.mic.active,
                f"{second.url if second else ''}；旧会话已结束并释放设备")
        if heard:
            monitored = np.concatenate(heard)
            kept = out_dir / "live_monitor_taffy.wav"
            sf.write(str(kept), monitored, first.speaker.rate, subtype="PCM_16")
            rms = float(np.sqrt(np.mean(monitored ** 2)))
            expected_seconds = (18*first.hello['frame_samples']-first.hello.get('lookahead_samples',1024))/22050
            c.check("监听回包是有声音的连续音频", len(monitored) / first.speaker.rate > .9*expected_seconds and rms > 0.01,
                    f"第一段监听送去播放的 {len(monitored) / first.speaker.rate:.1f} 秒音频已另存为 {kept}，"
                    f"RMS {rms:.3f}，可以回听")
        window.strength_slider.setValue(60)
        wait(lambda: window.live is not None and window.live is not second and bool(window.live.hello), 8)
        third = window.live
        monitors.append(third)
        ok = third is not second and third is not None and "strength=0.60" in third.url and wait(enough(third, 4), 10)
        c.check("监听中改强度会重新连接", ok, third.url if third else "")

        click(window.live_switch)
        stopped = wait(lambda: window.live is None, 8)
        released = all(not m.is_alive() and not m.mic.active and not m.speaker.active for m in monitors if m)
        lookahead = third.hello.get("lookahead_samples", 1024)
        c.check("关闭监听：按协议收尾并释放麦克风和输出流", stopped and released and third.packet_samples[-1] == lookahead,
                f"最后一个会话送 {third.frames_sent} 帧，回包样本数 {third.packet_samples[:3]}…{third.packet_samples[-2:]}"
                f"，丢弃输入 {third.mic.dropped} 块，播放断流 {third.speaker.underruns} 次")
        size = third.hello["frame_samples"]
        c.check("实时音频样本数与协商帧长一致", third.packet_samples[0] == size - lookahead
                and all(n == size for n in third.packet_samples[1:-1]) and third.samples_received == third.frames_sent * size)
        c.check("关闭监听后恢复文件转换", window.convert_button.isEnabled() and window.rec_button.isEnabled()
                and window.lamp.text() == "STANDBY", window.live_status.text())
        feeder.stop()

        # 1.0：同时发到语音软件和耳返。VB-CABLE 通常是 Discord 的麦克风，除非明确允许，只用 Steam 那一对测。
        pair = window.discord_pair
        if not pair:
            c.skip("同时发到语音软件和耳返", "没有找到虚拟麦克风")
        elif "cable" in pair[0].name.casefold() and not args.allow_vb_cable:
            c.skip("同时发到语音软件和耳返", f"虚拟麦克风是 {pair[1].name}，可能正接着 Discord；加 --allow-vb-cable 才测")
        elif pair[1] == cable_in or pair[0] == default_out:
            c.skip("同时发到语音软件和耳返", "虚拟麦克风和测试音频线是同一对设备，会形成回路")
        else:
            window.send_toggle.setChecked(True)
            window.monitor_toggle.setChecked(True)
            pump(100)
            sent = audio.Recorder(pair[1], 30)
            sent.start()
            feed_cable(3)
            click(window.live_switch)
            both = window.live
            flowing = wait(lambda: both is not None and both.frames_sent >= 12, 15)
            peak_ear = 0.0
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.5:
                pump(40)
                peak_ear = max(peak_ear, getattr(both.speaker, "monitor_level", 0.0) if both else 0.0)
            shot(window, "06b_send_and_monitor")
            click(window.live_switch)
            wait(lambda: window.live is None, 8)
            heard_call = sent.stop()
            feeder.stop()
            call_rms = float(np.sqrt(np.mean(heard_call ** 2))) if len(heard_call) else 0.0
            c.check("同时发到语音软件和耳返：两路都有声音", flowing and isinstance(both.speaker, audio.TeeSpeaker)
                    and call_rms > .003 and peak_ear > .02 and not both.speaker.monitor.active,
                    f"从 {pair[1].name} 录回 RMS {call_rms:.4f}；耳返 {default_out.name} 峰值 {peak_ear:.2f}")
            window.send_toggle.setChecked(False)
            window.monitor_toggle.setChecked(True)
        window.send_toggle.setChecked(False)
        window.monitor_toggle.setChecked(False)
        pump(60)
        c.check("两个输出都关掉时，开关不可用并说明原因", not window.live_switch.isEnabled()
                and "至少打开一个" in window.live_switch.toolTip(), window.live_switch.toolTip())
        window.monitor_toggle.setChecked(True)
        pump(60)

        # Shorten only this test's threshold; production keeps 3000 packets.
        previous_limit = window.config.live_restart_packets
        window.config.live_restart_packets = 12
        feed_cable(4, gap=.9)
        settings = (window.voice_id, window.strength_pct, window.current_mode(), window.current_pitch(),
                    window._device(window.mic_combo).name, window._device(window.out_combo).name)
        before_refreshes = window._live_auto_refreshes
        pair = audio.discord_pair(inputs, outputs)
        refresh_capture = (audio.Recorder(pair[1], 90) if pair and pair[0] == default_out
                           and args.allow_vb_cable else None)
        if refresh_capture:
            refresh_capture.start()
        try:
            click(window.live_switch)
            old_live = window.live
            # Only the first session expires. Leave enough speech in the new
            # session to verify actual output rather than a successful hello.
            window.config.live_restart_packets = 0
            renewed = wait(lambda: window.live is not None and window.live is not old_live
                           and bool(window.live.hello), 15)
            new_live = window.live
            same = (window.voice_id, window.strength_pct, window.current_mode(), window.current_pitch(),
                    window._device(window.mic_combo).name, window._device(window.out_combo).name)
            c.check("实时变声按包数自动重开且保留设置",
                    renewed and settings == same and window.live_switch.isChecked()
                    and window._live_auto_refreshes == before_refreshes+1
                    and wait(lambda: not old_live.is_alive(), 2)
                    and not old_live.mic.active and not old_live.speaker.active,
                    f"测试阈值 12 包；自动刷新 {window._live_auto_refreshes-before_refreshes} 次；"
                    "窗口和后端保持运行，旧采集和播放流已释放")
            if new_live is not None:
                levels = {"in": 0.0, "out": 0.0}
                capture_start = refresh_capture.seconds if refresh_capture else 0
                flowing = wait(enough(new_live, 20), 12)
                c.check("自动刷新后继续输出有声回包",
                        flowing and levels["in"] > .02 and levels["out"] > .02,
                        f"输入 {levels['in']:.3f}；输出 {levels['out']:.3f}；{new_live.frames_sent} 包")
                if refresh_capture:
                    captured = refresh_capture.stop()
                    rate = refresh_capture.rate
                    post_refresh = captured[round((capture_start+.7)*rate):]
                    rms = float(np.sqrt(np.mean(post_refresh**2))) if len(post_refresh) else 0
                    sf.write(str(out_dir/'after-refresh-virtual-mic.wav'), post_refresh, rate)
                    c.check("自动刷新后的真实虚拟麦克风有声音",
                            len(post_refresh) > rate and rms > .003,
                            f"连续跨重开录回；新连接后丢开前 0.7 秒旧尾音；RMS {rms:.4f}；"
                            f"{len(post_refresh)/rate:.2f} 秒合成测试语音；"
                            f"新连接时 {capture_start:.2f} 秒，总采集 {len(captured)/rate:.2f} 秒")
                    refresh_capture = None
            click(window.live_switch)
            stopped = wait(lambda: window.live is None, 8)
            pump(500)
            c.check("自动刷新后手动停止不会自行重开",
                    stopped and window.live is None and not window.live_switch.isChecked()
                    and window._live_auto_refreshes == before_refreshes+1)
        finally:
            if refresh_capture:
                refresh_capture.stop()
            if window.live_switch.isChecked():
                click(window.live_switch)
            wait(lambda: window.live is None, 8)
            window.config.live_restart_packets = previous_limit
            feeder.stop()

        # 模拟：监听途中麦克风被拔出（直接关掉采集流）
        feed_cable(3)
        click(window.live_switch)
        lost = window.live
        wait(lambda: lost is not None and lost.frames_sent >= 2, 10)
        lost.mic.close()
        ended = wait(lambda: window.live is None, 8)
        c.check("模拟麦克风掉线：监听停止并释放输出流", ended and not lost.speaker.active
                and not window.live_switch.isChecked() and "已断开" in window.live_status.text(),
                window.live_status.text() + "（模拟：脚本关闭了采集流）")
        feeder.stop()

        # 模拟：另一个会话占着引擎（1013 / 429）
        holder = LiveMonitor(client.live_url("taffy", 1.0), SilentMic(), NullSpeaker(),
                             lambda hello: None, lambda stats: None, lambda kind, message: None)
        holder.start()
        wait(lambda: bool(holder.hello), 5)
        click(window.live_switch)
        busy_live = wait(lambda: window.live is None and "1013" in window.live_status.text(), 10)
        c.check("引擎被占用时监听报 1013 并复位", busy_live and not window.live_switch.isChecked(),
                window.live_status.text() + "（模拟：另一个监听会话占着引擎）")
        click(window.convert_button)
        wait(lambda: window.converting_key is None, 15)
        c.check("引擎被占用时文件转换报 429", "429" in window.status_label.text(), window.status_label.text())
        holder.request_stop()
        holder.join(8)

    # ------------------------------------------------------------------ 设备不可用（模拟）
    ghost = audio.Device(index=9999, name="已拔出的设备（模拟）", rate=48000, channels=2, is_default=False, hostapi="模拟")
    def choose_ghost(combo):
        """选中模拟的失效设备。界面遇到失效设备会自动重新扫描并删掉它，所以每次测试前重新加上。"""
        if combo.itemData(combo.count() - 1) is not ghost:
            combo.addItem(ghost.name, ghost)
        combo.setCurrentIndex(combo.count() - 1)

    choose_ghost(window.mic_combo)
    click(window.rec_button)
    c.check("模拟麦克风打不开：录音报错并复位", window.recorder is None and not window.rec_button.isChecked()
            and any(w in window.status_label.text() for w in ("无法打开", "不在系统")), window.status_label.text())
    choose_ghost(window.mic_combo)
    click(window.live_switch)
    pump(200)
    c.check("模拟麦克风打不开：监听报错并复位", window.live is None and not window.live_switch.isChecked()
            and any(w in window.live_status.text() for w in ("无法打开", "不在系统")), window.live_status.text())
    if inputs:
        choose(window.mic_combo, (cable_in or inputs[0]).name)
    choose_ghost(window.out_combo)
    click(window.reference_button)
    c.check("模拟输出设备打不开：试听报错", window.playing_tag is None and any(w in window.status_label.text() for w in ("无法打开", "不在系统")),
            window.status_label.text())
    if inputs:
        choose_ghost(window.out_combo)  # 不重新加上的话，开关会用真实设备开始监听
        click(window.live_switch)
        pump(300)
        c.check("模拟输出设备打不开：监听报错并释放麦克风", window.live is None and not window.live_switch.isChecked(),
                window.live_status.text())
    real_list = audio.list_devices
    audio.list_devices = lambda kind: []
    window._populate_devices()
    window._refresh()
    c.check("模拟没有任何设备：录音、试听、监听不可用并有提示",
            not window.rec_button.isEnabled() and not window.live_switch.isEnabled()
            and not window.play_button.isEnabled() and window.mic_combo.currentText() == "未找到麦克风"
            and window.out_combo.currentText() == "未找到输出设备" and window.import_button.isEnabled(),
            f"{window.rec_hint.text()}；监听开关提示：{window.live_switch.toolTip()}")
    shot(window, "07_no_devices")
    audio.list_devices = real_list
    window.rescan_devices()
    c.check("刷新设备后恢复", window.mic_combo.currentData() is not None and window.out_combo.currentData() is not None,
            window.status_label.text())

    # ------------------------------------------------------------------ 正常退出（带着监听和播放关闭窗口）
    closing = None
    if inputs and default_out is not None:
        choose(window.mic_combo, (cable_in or inputs[0]).name)
        choose(window.out_combo, quiet_out.name)
        click(window.live_switch)
        closing = window.live
        wait(lambda: closing is not None and bool(closing.hello), 8)
        click(window.reference_button)
    t0 = time.monotonic()
    window.close()
    took = time.monotonic() - t0
    pump(300)
    leftovers = [t.name for t in threading.enumerate() if t.name in ("live-monitor", "convert")]
    ok = not window.player.playing and window.recorder is None and not leftovers
    if closing is not None:
        ok = ok and not closing.is_alive() and not closing.mic.active and not closing.speaker.active
    c.check("关闭窗口：停止播放，结束监听，释放设备", ok,
            f"关闭用时 {took:.2f} 秒；{'关闭时监听和示例播放都在进行' if closing is not None else '关闭时没有监听'}")
    try:
        client.health()
        alive = True
    except BackendError:
        alive = False
    if window.owned_backend is None:
        c.check("关闭窗口后，复用的后端保持运行", alive)
    else:
        c.check("关闭窗口后，由本程序启动的后端已停止", not alive)

    # ------------------------------------------------------------------ 记住设置（临时设置文件，不碰用户的设置）
    from .settings import Settings
    settings_file = Path(tempfile.mkdtemp(prefix="voice-gui-settings-")) / "settings.json"
    first = MainWindow(Config(args.backend_url, autostart=False, settings=Settings.load(settings_file)))
    first.show()
    wait(lambda: bool(first.voice_names), 15)
    if "anon" in first.voice_names:
        pick(first, "anon")
    first.strength_slider.setValue(70)
    first.voice_mode.setCurrentIndex(first.voice_mode.findData("call"))
    first.monitor_toggle.setChecked(True)
    first.send_toggle.setChecked(False)
    first.close()
    saved = Settings.load(settings_file)
    second = MainWindow(Config(args.backend_url, autostart=False, settings=saved, pitch_preset=saved.pitch))
    second.show()
    wait(lambda: bool(second.voice_names), 15)
    c.check("关闭后重开：记住角色、强度、风格和输出开关",
            second.voice_id == ("anon" if "anon" in second.voice_names else second.voice_id)
            and second.strength_pct == 70 and second.current_mode() == "call"
            and second.monitor_toggle.isChecked() and not second.send_toggle.isChecked(),
            f"{settings_file}：{saved}")
    second.close()

    # ------------------------------------------------------------------ 未连接后端
    offline = MainWindow(Config(DEAD_URL, autostart=False))
    offline.resize(1180, 730)
    offline.show()
    wait(lambda: offline.notice.isVisible(), 10)
    offline.load_file(ORIGINAL)
    c.check("未连接后端：显示原因，转换和监听不可用，仍可导入和试听原声",
            not offline.backend_ok and any(reason in offline.notice.text() for reason in ("连接被拒绝", "连接超时"))
            and not offline.convert_button.isEnabled()
            and not offline.live_switch.isEnabled() and not offline.voice_names and offline.input is not None
            and offline.convert_button.toolTip() == "后端未连接",
            offline.notice.text().replace("\n", " "))
    shot(offline, "08_backend_offline")
    offline.close()

    scripts = Path(tempfile.mkdtemp(prefix="voice-gui-selfcheck-"))
    failing = scripts / "failing-backend.cmd"
    failing.write_text("@echo off\r\necho Simulated failure: model file is missing\r\nexit /b 3\r\n", encoding="ascii")
    broken = MainWindow(Config(DEAD_URL, autostart=True, script=failing))
    broken.show()
    wait(lambda: broken.notice.isVisible(), 20)
    c.check("启动脚本失败：显示退出码和脚本输出", "退出码 3" in broken.notice.text()
            and "Simulated failure" in broken.notice.text(), broken.notice.text().replace("\n", " ") + "（模拟脚本）")
    shot(broken, "09_backend_start_failed")
    broken.close()
    missing = MainWindow(Config(DEAD_URL, autostart=True, script=scripts / "missing.cmd"))
    missing.show()
    wait(lambda: missing.notice.isVisible(), 10)
    c.check("启动脚本不存在：显示路径", "找不到启动脚本" in missing.notice.text(), missing.notice.text().replace("\n", " "))
    missing.close()

    alt_url = ALT_URL if client.port != 8766 else "http://127.0.0.1:8767"
    alt = BackendClient(alt_url)
    try:
        alt.health(timeout=1)
        port_free = False
    except BackendError:
        port_free = True
    if args.skip_autostart or not port_free or not (ROOT / ".venv" / "Scripts" / "python.exe").is_file():
        c.skip("自动启动后端", "已用 --skip-autostart 跳过" if args.skip_autostart else "8766 端口已被占用或后端环境不存在")
    else:
        # 与 start-backend.cmd 的最后一步相同，只把端口换成 8766，避免和正在运行的后端抢 8765。
        starter = scripts / "start-backend-8766.cmd"
        starter.write_text(
            '@echo off\r\ncd /d "%VOICE_SELFCHECK_ROOT%"\r\nset PYTHONUTF8=1\r\n'
            '".venv\\Scripts\\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port %VOICE_SELFCHECK_PORT% '
            "--ws-max-size 16384 --ws-max-queue 2\r\n", encoding="ascii")
        os.environ["VOICE_SELFCHECK_ROOT"] = str(ROOT)
        os.environ["VOICE_SELFCHECK_PORT"] = str(alt.port)
        started = MainWindow(Config(alt_url, autostart=True, script=starter))
        started.show()
        t0 = time.monotonic()
        ready = wait(lambda: started.backend_ok and bool(started.voice_names), 180)
        c.check("后端未运行时自动启动并等到 /api/health 就绪", ready and started.owned_backend is not None,
                f"{time.monotonic() - t0:.1f} 秒；{started.status_label.text()}（脚本与 start-backend.cmd 的启动命令相同，端口改为 8766）")
        if ready:
            started.load_file(ORIGINAL)
            click(started.convert_button)
            wait(lambda: started.converting_key is not None, 2)
            wait(lambda: started.converting_key is None, 180)
            c.check("用自动启动的后端完成一次转换", started.current_result() is not None, started.status_label.text())
        process = started.owned_backend
        started.close()
        pump(1500)
        try:
            alt.health(timeout=1)
            gone = False
        except BackendError:
            gone = True
        c.check("关闭窗口时停止由本程序启动的后端", gone and (process is None or process.poll() is not None),
                "8766 端口不再响应")

    # 等后台线程（健康检查、后端输出读取）收尾，再让解释器退出
    for thread in threading.enumerate():
        if thread is not threading.current_thread():
            thread.join(6)
    print(f"\n结果和截图在 {out_dir}", flush=True)
    cleanup_qt_styles()
    return c.summary()


if __name__ == "__main__":
    sys.exit(main())
