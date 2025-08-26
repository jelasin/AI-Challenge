"""
Challenge 7: 文本转语音（TTS）— 使用 OpenAI tts-1 生成音频

用法（PowerShell）：
  python main.py --text "你好" --voice alloy --format mp3

参数：
- --text 文本内容（或使用 --file 从文本文件读取）
- --file 从文件读取文本（优先级高于 --text）
- --voice 语音风格，默认 alloy（示例：alloy, verse, coral 等，具体取决于网关支持）
- --format 输出格式（mp3/wav/aac/flac/opus/pcm，默认 mp3）
- --outdir 输出目录（默认 voice）
- --name 输出文件基础名（默认 tts_时间戳）

环境变量：
- OPENAI_API_KEY（必需）
- OPENAI_BASE_URL（可选，自定义网关地址）
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, cast
from pydantic import BaseModel, Field
from langchain_core.tools import tool

try:
	from openai import OpenAI
except Exception:
	print("未安装 openai 包，请先安装: pip install openai", file=sys.stderr)
	raise


def ensure_api_key() -> None:
	if not os.getenv("OPENAI_API_KEY"):
		print("❌ 未检测到 OPENAI_API_KEY 环境变量。", file=sys.stderr)
		print("💡 在 PowerShell 设置: $env:OPENAI_API_KEY='your-key'", file=sys.stderr)
		sys.exit(2)


def parse_args(argv: list[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="使用 OpenAI tts-1 将文本合成为语音")
	parser.add_argument("--text", help="要合成的文本内容")
	parser.add_argument("--file", help="包含文本的文件路径（优先级高于 --text）")
	parser.add_argument("--voice", default="alloy", help="语音风格，默认 alloy")
	parser.add_argument(
		"--format",
		default="mp3",
		choices=["mp3", "wav", "flac", "opus", "aac", "pcm"],
		help="输出格式",
	)
	parser.add_argument("--outdir", default="voice", help="输出目录，默认 voice")
	parser.add_argument("--name", default=None, help="输出文件基础名，默认 tts_时间戳")
	return parser.parse_args(argv)


def read_text(args: argparse.Namespace) -> str:
	if args.file:
		p = Path(args.file)
		if not p.exists() or not p.is_file():
			raise FileNotFoundError(f"未找到文本文件: {p}")
		for enc in ("utf-8", "gbk", "latin-1"):
			try:
				return p.read_text(encoding=enc)
			except Exception:
				continue
		raise RuntimeError(f"无法读取文件（编码不兼容）: {p}")
	if args.text:
		return str(args.text)
	return "欢迎使用文本转语音示例。This is a demo of OpenAI tts-1 text to speech."


ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


def tts_synthesize(text: str, voice: str, fmt: str, outdir: Path, basename: str) -> Path:
	client_kwargs = {}
	client_kwargs["base_url"] = os.getenv("OPENAI_API_BASE")
	client = OpenAI(**client_kwargs)

	outdir.mkdir(parents=True, exist_ok=True)
	outfile = outdir / f"{basename}.{fmt}"

	# 使用 streaming 写文件，更稳妥
	try:
		rf: ResponseFormat = cast(ResponseFormat, fmt)
		with client.audio.speech.with_streaming_response.create(
			model="tts-1",
			voice=voice,
			input=text,
			response_format=rf,
		) as response:
			response.stream_to_file(str(outfile))
	except AttributeError:
		# 回退：非 streaming 模式（部分 SDK 或网关）
		rf: ResponseFormat = cast(ResponseFormat, fmt)
		resp = client.audio.speech.create(
			model="tts-1",
			voice=voice,
			input=text,
			response_format=rf,
		)
		# 新版 SDK 通常支持 .read_bytes() 或 .content；
		data = getattr(resp, "content", None) or getattr(resp, "audio", None)
		if data is None and hasattr(resp, "to_dict"):
			d = resp.to_dict()  # type: ignore[attr-defined]
			data = d.get("data") if isinstance(d, dict) else None
		if isinstance(data, (bytes, bytearray)):
			outfile.write_bytes(data)
		else:
			# 如果返回 base64/URL 等，给出提示
			raise RuntimeError("TTS 返回格式无法直接保存，请升级 openai 包或更换网关")

	return outfile


# LangChain v0.3: 定义一个 TTS 工具，便于在链/代理中复用
class TTSToolInput(BaseModel):
	text: str = Field(..., description="要合成的文本内容")
	voice: str = Field("alloy", description="语音风格")
	response_format: ResponseFormat = Field("mp3", description="音频格式")
	outdir: str = Field("voice", description="输出目录")
	basename: str | None = Field(None, description="输出文件基础名，不含扩展名")


@tool("openai_tts", args_schema=TTSToolInput)
def openai_tts(
	text: str,
	voice: str = "alloy",
	response_format: ResponseFormat = "mp3",
	outdir: str = "voice",
	basename: str | None = None,
) -> str:
	"""使用 OpenAI tts-1 将文本合成为语音，并返回生成文件路径。"""
	base = basename or datetime.now().strftime("tts_%Y%m%d_%H%M%S")
	path = tts_synthesize(text=text, voice=voice, fmt=response_format, outdir=Path(outdir), basename=base)
	return str(path)


def main(argv: list[str]) -> int:
	ensure_api_key()
	args = parse_args(argv)

	try:
		text = read_text(args)
	except Exception as e:
		print(f"❌ 读取文本失败: {e}", file=sys.stderr)
		return 1

	basename = args.name or datetime.now().strftime("tts_%Y%m%d_%H%M%S")
	outdir = Path(args.outdir)

	try:
		# 通过 LangChain 工具接口执行（v0.3 推荐风格）
		outfile_str = openai_tts.invoke(
			{
				"text": text,
				"voice": args.voice,
				"response_format": cast(ResponseFormat, args.format),
				"outdir": str(outdir),
				"basename": basename,
			}
		)
		outfile = Path(outfile_str)
	except Exception as e:
		print(f"❌ 合成失败: {e}", file=sys.stderr)
		return 1

	print(f"✅ 已保存: {outfile}")
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))

