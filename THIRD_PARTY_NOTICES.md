# 第三方组件与许可

角色变声器使用了下面这些开源软件、模型和外部组件。本项目自己的源代码使用 MIT 许可（见 `LICENSE`），下面这些组件不受它约束，各自按原许可使用。每一项的许可以上游仓库或发布页为准；这里列出的是 1.0.0 发布时核对的版本和许可类型，方便分发前检查。

## 程序库

| 组件 | 版本 | 许可 | 用在哪里 |
| --- | --- | --- | --- |
| [Python](https://www.python.org/)（Windows 嵌入式发行版） | 3.13 | PSF-2.0 | 便携版自带的运行环境 |
| [Qt 6 / PySide6](https://doc.qt.io/qtforpython-6/) | 6.12.0 | LGPL-3.0 | 桌面界面 |
| [NumPy](https://numpy.org/) | 2.5.4 | BSD-3-Clause | 音频数组 |
| [SciPy](https://scipy.org/) | 1.18.1 | BSD-3-Clause | 后端信号处理 |
| [python-sounddevice](https://github.com/spatialaudio/python-sounddevice) / [PortAudio](https://www.portaudio.com/) | 0.5.6 | MIT | 录音和播放 |
| [python-soundfile](https://github.com/bastibe/python-soundfile) / [libsndfile](https://libsndfile.github.io/libsndfile/) | 0.14.0 | BSD-3-Clause / LGPL-2.1 | 读写音频文件 |
| [python-soxr](https://github.com/dofuuz/python-soxr) / libsoxr | 1.1.0 | LGPL-2.1+ | 重采样 |
| [websockets](https://github.com/python-websockets/websockets) | 17.4 | BSD-3-Clause | 实时变声通道 |
| [FastAPI](https://github.com/fastapi/fastapi) | 0.162.4 | MIT | 本机引擎接口 |
| [Uvicorn](https://github.com/encode/uvicorn) | 0.54.0 | BSD-3-Clause | 本机引擎服务 |
| [ONNX Runtime (DirectML)](https://github.com/microsoft/onnxruntime) | 1.24.4 | MIT | 模型推理 |
| [ONNX](https://github.com/onnx/onnx) | 1.23.2 | Apache-2.0 | 模型格式 |
| [FAISS](https://github.com/facebookresearch/faiss) | 1.16.0 | MIT | 检索索引 |
| [Praat / parselmouth](https://github.com/YannickJadoul/Parselmouth) | 0.4.7 | **GPL-3.0** | 音高提取（部分声线） |

源代码版的程序库由安装脚本从 PyPI 下载，不包含在本仓库里。便携版（Releases 里的 zip）把这些程序库原样放在 `runtime\Lib\site-packages\`，各自的许可文件在对应的 `*.dist-info` 文件夹里；Qt / PySide6 以动态库形式使用，可以自行替换。

**parselmouth（Praat）是 GPL-3.0**，只在旧版兼容声线里用来提取音高，导入的 RVC 声线用 RMVPE，用不到它。所以**便携版不包含 parselmouth**。如果你自己把它连同程序打包再分发，整个安装包的分发要遵守 GPL-3.0。

## 模型与算法

| 组件 | 许可 | 说明 |
| --- | --- | --- |
| [RVC (Retrieval-based-Voice-Conversion-WebUI)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | MIT | 推理流程参考，见 `third_party/rvc/` |
| [RMVPE](https://github.com/yxlllc/RMVPE) | 以上游为准 | 音高提取模型 |
| [ContentVec](https://github.com/auspicious3000/contentvec) | MIT | 语音内容特征 |
| [GTCRN](https://github.com/Xiaobin-Rong/gtcrn)（sherpa-onnx 导出） | MIT | 降噪 |
| [Silero VAD](https://github.com/snakers4/silero-vad) | MIT | 语音检测 |
| [OpenVoice V2](https://github.com/myshell-ai/OpenVoice) | MIT | 旧版兼容声线的音色校正（公开版不使用），见 `models/LICENSE.txt` |

## 声线

本仓库不包含任何声线模型、角色头像或配音片段。你导入的声线（存放在 `models/local-rvc/`，不会进入 Git）的权利属于它的作者或权利人。请只导入你有权使用的声线。

## 外部组件

[VB-CABLE](https://vb-audio.com/Cable/) 是 VB-Audio 的 donationware 虚拟声卡驱动，不是本项目的开源代码。本仓库不包含它，请从官网下载安装；把它打进自己的安装包再分发，需要按 [VB-Audio 授权说明](https://vb-audio.com/Services/licensing.htm) 取得许可。
