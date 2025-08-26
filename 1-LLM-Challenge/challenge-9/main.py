r"""
Challenge 9: 语音实时交互（Realtime）— 使用 gpt-4o-realtime-preview

说明：
- 本脚本通过 WebSocket 对接 OpenAI Realtime API，采集麦克风音频并播放模型的语音回复。
- 默认使用“按键说话”(PTT)模式：按回车键提交当前已录制的语音片段，模型返回语音并播放。
- 可选“服务端 VAD”模式（--mode vad），由服务端进行语音端点检测（实验性，可能依赖网关支持）。

用法（PowerShell）：
  python .\main.py --list-devices               # 列出音频设备
  python .\main.py --input-device 1 --output-device 3  # 选择设备索引
  python .\main.py --mode ptt                    # 默认按键说话模式
  python .\main.py --mode vad                    # 服务端 VAD 实验模式

环境变量：
- OPENAI_API_KEY（必需）
- OPENAI_BASE_URL（可选，自定义网关；若支持 Realtime，将自动转换为 wss 端点）

依赖：
- websockets, sounddevice, numpy, openai
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Optional

import numpy as np
import sounddevice as sd

try:
    import websockets
except Exception:
    print("缺少依赖 websockets，请先安装：pip install websockets", file=sys.stderr)
    raise


DEFAULT_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
SAMPLE_RATE = 24000  # 与 Realtime 语音默认采样率对齐
CHANNELS = 1
DTYPE = "int16"  # PCM16


def ensure_api_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    print("❌ 未检测到 OPENAI_API_KEY 环境变量。", file=sys.stderr)
    print("💡 在 PowerShell 设置: $env:OPENAI_API_KEY='your-key'", file=sys.stderr)
    sys.exit(2)


def list_devices_and_exit() -> None:
    print("可用音频设备：")
    print(sd.query_devices())
    sys.exit(0)


def make_ws_uri(model: str) -> str:
    base = os.getenv("OPENAI_BASE_URL")
    if base:
        # 将 http(s) 转为 ws(s) 并拼接 realtime 路径
        if base.startswith("https://"):
            ws_base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            ws_base = "ws://" + base[len("http://"):]
        else:
            ws_base = base  # 已是 ws(s) 形式或网关自定义
        if not ws_base.endswith("/"):
            ws_base += "/"
        return f"{ws_base}v1/realtime?model={model}"
    # OpenAI 公网默认地址
    return f"wss://api.openai.com/v1/realtime?model={model}"


@dataclass
class AudioConfig:
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    dtype: str = DTYPE


class MicStreamer:
    """麦克风采集线程，将 PCM16 原始字节放入队列。"""

    def __init__(self, cfg: AudioConfig, queue: Queue[bytes]):
        self.cfg = cfg
        self.queue = queue
        self.stream: Optional[sd.InputStream] = None
        self._stop = threading.Event()

    def _callback(self, indata, frames, time_info, status):  # noqa: ANN001
        if status:
            # 记录潜在的溢出/欠载
            print(f"[mic] status: {status}")
        data_bytes = indata.astype(self.cfg.dtype).tobytes()
        self.queue.put(data_bytes)

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype=self.cfg.dtype,
            callback=self._callback,
            device=self.cfg.input_device,
            blocksize=0,
        )
        self.stream.start()

    def stop(self):
        self._stop.set()
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass


class Speaker:
    """扬声器播放器，向输出设备写入 PCM16 数据。"""

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.stream: Optional[sd.OutputStream] = None

    def start(self):
        self.stream = sd.OutputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype=self.cfg.dtype,
            device=self.cfg.output_device,
            blocksize=0,
        )
        self.stream.start()

    def write(self, pcm_bytes: bytes):
        if not self.stream:
            return
        # 将 bytes -> int16 -> float32 并写入（sounddevice 允许直接 int16）
        try:
            arr = np.frombuffer(pcm_bytes, dtype=np.int16)
            self.stream.write(arr)
        except Exception as e:
            print(f"[spk] 写入失败: {e}")

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass


async def ws_sender(ws, mic_q: Queue[bytes], mode: str):
    """将麦克风数据持续发送到 Realtime：input_audio_buffer.append。

    在 ptt 模式下，回车后提交缓冲并请求生成一次响应。
    在 vad 模式下，仅持续 append，由服务端自动进行端点检测与响应。
    """
    loop = asyncio.get_event_loop()

    async def send_json(payload: dict):
        await ws.send(json.dumps(payload))

    print("🎤 开始采集麦克风。PTT 模式下按回车提交一次语音并请求回复。Ctrl+C 结束。")

    # 后台线程监听键盘（仅 ptt 模式）
    commit_requested = False

    def key_listener():
        nonlocal commit_requested
        try:
            while True:
                _ = sys.stdin.readline()
                commit_requested = True
        except Exception:
            pass

    if mode == "ptt":
        t = threading.Thread(target=key_listener, daemon=True)
        t.start()

    last_commit = time.time()
    while True:
        try:
            chunk = await loop.run_in_executor(None, lambda: mic_q.get(timeout=0.2))
        except Exception:
            chunk = None

        if chunk:
            await send_json({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("utf-8"),
            })

        if mode == "ptt" and commit_requested:
            # 提交当前缓冲并创建一次响应（语音输出）
            await send_json({"type": "input_audio_buffer.commit"})
            await send_json({
                "type": "response.create",
                "response": {
                    "modalities": ["audio"],
                    "instructions": "请简要回答并用中文语音回复。",
                    "audio": {"voice": "alloy", "format": "pcm16"},
                },
            })
            commit_requested = False
            last_commit = time.time()

        # 在 vad 模式下，可选择定时 commit 作为兼容（部分服务端可能仍需 commit）
        if mode == "vad" and time.time() - last_commit > 4.0:
            await send_json({"type": "input_audio_buffer.commit"})
            last_commit = time.time()


async def ws_receiver(ws, speaker: Speaker):
    """接收服务端事件，提取音频 delta 并播放。"""
    async for message in ws:
        try:
            data = json.loads(message)
        except Exception:
            continue

        evt_type = data.get("type")
        # 常见事件：response.output_audio.delta / response.audio.delta
        if isinstance(evt_type, str) and "audio" in evt_type and "delta" in evt_type:
            # 取音频字段（不同实现可能使用 audio / delta / chunk 等命名）
            audio_b64 = (
                data.get("audio")
                or data.get("delta")
                or (data.get("output_audio") or {}).get("delta")
            )
            if isinstance(audio_b64, str):
                try:
                    pcm = base64.b64decode(audio_b64)
                    speaker.write(pcm)
                except Exception as e:
                    print(f"[recv] 音频解码失败: {e}")
        elif evt_type == "error":
            print(f"[error] {data}")
        elif evt_type and evt_type.endswith("completed"):
            # 一次回复完成
            pass


async def run(model: str, input_device: Optional[int], output_device: Optional[int], mode: str):
    ensure_api_key()

    cfg = AudioConfig(input_device=input_device, output_device=output_device)
    mic_q: Queue[bytes] = Queue(maxsize=64)

    mic = MicStreamer(cfg, mic_q)
    spk = Speaker(cfg)

    # 启动本地音频 I/O
    spk.start()
    mic.start()

    uri = make_ws_uri(model)
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(uri, extra_headers=headers, max_size=16 * 1024 * 1024) as ws:
        # 配置会话：指定输入/输出音频格式与 VAD
        session_update = {
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": "alloy",
            },
        }
        if mode == "vad":
            session_update["session"]["turn_detection"] = {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 600,
            }

        await ws.send(json.dumps(session_update))

        sender_task = asyncio.create_task(ws_sender(ws, mic_q, mode))
        receiver_task = asyncio.create_task(ws_receiver(ws, spk))

        # 等待任一任务结束（或 Ctrl+C）
        done, pending = await asyncio.wait(
            {sender_task, receiver_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

    # 关闭音频
    mic.stop()
    spk.stop()


def parse_args(argv: list[str]):
    import argparse

    p = argparse.ArgumentParser(description="gpt-4o-realtime-preview 语音交互")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Realtime 模型名")
    p.add_argument("--list-devices", action="store_true", help="列出可用音频设备并退出")
    p.add_argument("--input-device", type=int, default=None, help="输入设备索引")
    p.add_argument("--output-device", type=int, default=None, help="输出设备索引")
    p.add_argument("--mode", choices=["ptt", "vad"], default="ptt", help="交互模式：ptt=按键说话, vad=服务端VAD(实验)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.list_devices:
        list_devices_and_exit()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 优雅退出
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:
            # Windows 下可能不支持
            pass

    try:
        loop.run_until_complete(run(
            model=args.model,
            input_device=args.input_device,
            output_device=args.output_device,
            mode=args.mode,
        ))
        return 0
    except Exception as e:
        print(f"❌ 运行失败: {e}", file=sys.stderr)
        return 1
    finally:
        try:
            loop.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
