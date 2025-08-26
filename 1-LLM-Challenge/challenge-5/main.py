"""
LangChain v0.3 多模态示例：图片识别

功能
- 支持本地图片或网络图片 URL 作为输入
- 使用 LangChain v0.3 的 init_chat_model（如 gpt-4o）进行图像理解

环境变量
- OPENAI_API_KEY: OpenAI 密钥
- OPENAI_BASE_URL: 可选，自定义网关/代理地址
- OPENAI_VISION_MODEL: 可选，默认 gpt-4o

使用
	python main.py --image <本地路径或URL> [--question "描述需求"]
示例
  python main.py --image https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg
  python main.py --image ./demo.jpg --question "图中有几个人？"
"""

from __future__ import annotations

import argparse
import json
import base64
import mimetypes
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


def is_url(s: str) -> bool:
	try:
		u = urlparse(s)
		return u.scheme in {"http", "https", "data"} and bool(u.netloc or u.scheme == "data")
	except Exception:
		return False


def to_data_url(image_path: str) -> str:
	"""
	将本地图片转为 data URL（data:<mime>;base64,<...>）。
	"""
	p = Path(image_path)
	if not p.exists() or not p.is_file():
		raise FileNotFoundError(f"未找到图片文件: {p}")
	mime, _ = mimetypes.guess_type(str(p))
	mime = mime or "image/png"
	with p.open("rb") as f:
		b64 = base64.b64encode(f.read()).decode("utf-8")
	return f"data:{mime};base64,{b64}"


def url_to_data_url(url: str) -> str:
	"""
	下载网络图片并转为 data URL。
	"""
	req = Request(url, headers={
		"User-Agent": (
			"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
			"(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
		),
		"Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
	})
	with urlopen(req, timeout=20) as resp:
		content_type = None
		if hasattr(resp, "headers"):
			get_ct = getattr(resp.headers, "get_content_type", None)
			if callable(get_ct):
				content_type = get_ct()
			else:
				content_type = resp.headers.get("Content-Type")
		if isinstance(content_type, str):
			content_type = content_type.split(";")[0].strip()
		else:
			content_type = mimetypes.guess_type(url)[0] or "image/png"
		data = resp.read()
		b64 = base64.b64encode(data).decode("utf-8")
		return f"data:{content_type};base64,{b64}"


def build_image_content(image_input: str, question: str):
	"""
	构造多模态 HumanMessage 的 content 列表结构。
	"""
	if is_url(image_input):
		# 若已是 data: 协议，直接使用；否则本地下载并转 data URL
		u = urlparse(image_input)
		if u.scheme == "data":
			image_url = image_input
		else:
			try:
				image_url = url_to_data_url(image_input)
			except Exception:
				# 兜底：直接使用 URL（可能仍会被拒绝）
				image_url = image_input
	else:
		image_url = to_data_url(image_input)

	return [
		{"type": "text", "text": question},
		{"type": "image_url", "image_url": {"url": image_url}},
	]


def run(image: str, question: str, model_name: str | None = None) -> str:
	"""
	运行图像识别任务
	"""
	model_name =  "gpt-4o"

	# 初始化模型（LangChain v0.3 接口）
	llm = init_chat_model(model_name, temperature=0)

	# 构造系统与用户消息（用户消息包含图片）
	system = SystemMessage(content="你是一个专业的视觉 AI 助手，请使用简体中文作答。")
	human = HumanMessage(content=build_image_content(image, question))

	resp = llm.invoke([system, human])

	# 统一字符串输出
	content = getattr(resp, "content", resp)
	if isinstance(content, str):
		return content
	try:
		# 兼容列表/分段内容，尽量提取可读文本
		if isinstance(content, list):
			parts: list[str] = []
			for part in content:
				if isinstance(part, dict):
					if "text" in part:
						parts.append(str(part["text"]))
					elif part.get("type") == "output_text" and "text" in part:
						parts.append(str(part["text"]))
			if parts:
				return "\n".join(parts)
		return json.dumps(content, ensure_ascii=False)
	except Exception:
		return str(content)


def main():
	parser = argparse.ArgumentParser(description="LangChain v0.3 多模态图片识别示例")
	parser.add_argument("--image", required=False, help="本地图片路径或网络图片 URL")
	parser.add_argument(
		"--question",
		default="这张图片里有什么？请简要描述，并指出主要元素。",
		help="向模型提出的问题/需求",
	)
	args = parser.parse_args()

	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		print("未检测到 OPENAI_API_KEY 环境变量。请先配置后再运行。", file=sys.stderr)
		print("提示: 可以在根目录执行 load_env.ps1 或自行设置环境变量。", file=sys.stderr)
		sys.exit(2)

	image = args.image
	if not image:
		# 提供一个演示用的公开图片 URL（若运行环境无法访问外网，请改用本地图片路径）
		image = "https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg"

	try:
		answer = run(image=image, question=args.question)
		print("=== 模型响应 ===")
		print(answer)
	except FileNotFoundError as e:
		print(f"错误: {e}", file=sys.stderr)
		sys.exit(1)
	except Exception as e:
		# 打印更简洁的错误信息，便于排错
		print(f"调用模型失败: {e}", file=sys.stderr)
		sys.exit(1)


if __name__ == "__main__":
	main()

