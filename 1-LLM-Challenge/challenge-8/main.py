"""
Challenge 8: 语音转文本（STT）— 使用 OpenAI whisper-1 转写音频

用法（PowerShell）：
	python main.py -f audio.mp3 --language zh

参数：
- -f / --file 音频文件路径（支持 mp3/mp4/mpeg/mpga/m4a/wav/webm）
- --language 可选语言提示（如 zh/en）

环境变量：
- OPENAI_API_KEY（必需）
- OPENAI_BASE_URL（可选，自定义网关地址）
- OPENAI_TRANSCRIBE_MODEL（可选，默认 whisper-1）
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

try:
	from openai import OpenAI
except Exception as e:  # pragma: no cover - helpful message if dependency missing
	print("[错误] 缺少依赖 'openai'。请将其加入 requirements 并安装。", file=sys.stderr)
	raise


SUPPORTED_AUDIO_EXTS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}

def _extract_text(resp) -> str:
	"""尽力从 SDK 响应对象中提取转写文本。"""
	# OpenAI SDK v1.x 通常返回包含 `.text` 字段的对象
	text = getattr(resp, "text", None)
	if isinstance(text, str) and text.strip():
		return text
	# 某些实现可能返回 dict
	if isinstance(resp, dict) and isinstance(resp.get("text"), str):
		return resp["text"]
	return str(resp)


def ensure_api_key() -> None:
	if os.getenv("OPENAI_API_KEY"):
		return
	print("❌ 未检测到 OPENAI_API_KEY 环境变量。", file=sys.stderr)
	print("💡 在 PowerShell 下可执行: $env:OPENAI_API_KEY='your-openai-key'", file=sys.stderr)
	sys.exit(2)


def transcribe(file_path: str, model: Optional[str] = None, language: Optional[str] = None) -> str:
	"""
	使用 OpenAI whisper-1 将音频文件转写为文本。

	参数：
		file_path: 音频文件路径。
		model: 模型名称；不指定时使用环境变量 OPENAI_TRANSCRIBE_MODEL 或 'whisper-1'。
		language: 可选语言提示（BCP-47），例如 'zh'、'en'。

	返回：
		转写文本。
	"""
	model_name = model or os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")
	# 允许通过 OPENAI_BASE_URL 指定自定义网关
	client_kwargs = {}
	base_url = os.getenv("OPENAI_API_BASE")
	if base_url:
		client_kwargs["base_url"] = base_url
	client = OpenAI(**client_kwargs)
	with open(file_path, "rb") as f:
		if language:
			resp = client.audio.transcriptions.create(  # type: ignore[attr-defined]
				model=model_name,
				file=f,
				language=language,
			)
		else:
			resp = client.audio.transcriptions.create(  # type: ignore[attr-defined]
				model=model_name,
				file=f,
			)
	return _extract_text(resp)


# LangChain v0.3: 定义一个可复用的转写工具
class TranscribeToolInput(BaseModel):
	file: str = Field(..., description="音频文件路径，支持 mp3/mp4/mpeg/mpga/m4a/wav/webm")
	language: Optional[str] = Field(None, description="可选语言提示，如 'zh' 或 'en'")
	model: Optional[str] = Field(None, description="可选模型名，默认 whisper-1")


@tool("openai_transcribe", args_schema=TranscribeToolInput)
def openai_transcribe(file: str, language: Optional[str] = None, model: Optional[str] = None) -> str:
	"""使用 OpenAI whisper-1 将音频转写为文本并返回结果字符串。"""
	p = Path(file)
	if not p.exists() or not p.is_file():
		raise FileNotFoundError(f"未找到音频文件: {p}")
	# 轻量格式校验（仅告警不拦截）在 CLI 中执行；工具中直接执行
	return transcribe(str(p), model=model, language=language)


def main(argv: Optional[list[str]] = None) -> int:
	ensure_api_key()
	parser = argparse.ArgumentParser(description="使用 OpenAI whisper-1 将音频转写为文本（LangChain 工具风格）")
	parser.add_argument("-f", "--file", required=True, help="音频文件路径 (mp3/mp4/mpeg/mpga/m4a/wav/webm)")
	parser.add_argument("--language", default=None, help="可选语言提示，如 'zh' 或 'en'")
	args = parser.parse_args(argv)

	audio_path = Path(args.file)
	if not audio_path.exists() or not audio_path.is_file():
		print(f"[错误] 未找到文件: {audio_path}", file=sys.stderr)
		return 2

	ext = audio_path.suffix.lower()
	if ext not in SUPPORTED_AUDIO_EXTS:
		print(f"[警告] 未识别的文件扩展名 '{ext}'，将继续尝试转写…", file=sys.stderr)

	try:
		text = openai_transcribe.invoke({
			"file": str(audio_path),
			"language": args.language,
		})
	except Exception as e:
		print(f"[错误] 转写失败: {e}", file=sys.stderr)
		return 1

	print(text)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

