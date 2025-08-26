r"""
Challenge 6: 使用 LangChain（v0.3）调用 DALL·E 3 生图并保存到本地

用法（PowerShell）：
  python .\main.py --prompt "一只在太空中弹吉他的猫女孩" --n 1 --size 1024x1024

环境变量：
- OPENAI_API_KEY（必需）
- OPENAI_BASE_URL（可选，LangChain 将透传给 OpenAI SDK）
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List
from urllib.request import urlopen, Request

from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

def ensure_api_key() -> None:
	if not os.getenv("OPENAI_API_KEY"):
		print("未检测到 OPENAI_API_KEY 环境变量。", file=sys.stderr)
		print("提示: 可以在仓库根目录执行 load_env.ps1，或在 PowerShell 设置: $env:OPENAI_API_KEY='...'", file=sys.stderr)
		sys.exit(2)


def save_image_b64(b64: str, outdir: Path, basename: str, idx: int) -> Path:
	outdir.mkdir(parents=True, exist_ok=True)
	fname = f"{basename}_{idx:02d}.png"
	fpath = outdir / fname
	img_bytes = base64.b64decode(b64)
	fpath.write_bytes(img_bytes)
	return fpath


def save_image_url(url: str, outdir: Path, basename: str, idx: int) -> Path:
	outdir.mkdir(parents=True, exist_ok=True)
	fname = f"{basename}_{idx:02d}.png"
	fpath = outdir / fname
	req = Request(url, headers={
		"User-Agent": (
			"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
			"(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
		)
	})
	with urlopen(req, timeout=30) as resp:
		fpath.write_bytes(resp.read())
	return fpath


def extract_b64_or_url(data: object) -> str:
	"""从 DallEAPIWrapper 返回结果中提取 base64 或 URL。
	兼容：字符串(URL或b64)、JSON 字符串、dict/list等。
	返回：若为 b64 则以 "data:image/" 或纯 b64 起头，也可能直接是 http(s) URL。
	"""
	# 直接字符串
	if isinstance(data, str):
		return data
	# JSON 字符串
	try:
		if isinstance(data, (bytes, bytearray)):
			data = data.decode("utf-8", errors="ignore")
		if isinstance(data, str):
			parsed = json.loads(data)
			data = parsed
	except Exception:
		pass
	# dict/list 结构
	try:
		if isinstance(data, dict):
			# 常见格式：{"data":[{"b64_json": "..."}]} 或 {"url": "..."}
			if "data" in data and isinstance(data["data"], list) and data["data"]:
				first = data["data"][0]
				if isinstance(first, dict):
					if "b64_json" in first and isinstance(first["b64_json"], str):
						return first["b64_json"]
					if "url" in first and isinstance(first["url"], str):
						return first["url"]
			if "b64_json" in data and isinstance(data["b64_json"], str):
				return data["b64_json"]
			if "url" in data and isinstance(data["url"], str):
				return data["url"]
		if isinstance(data, list) and data:
			# 取第一个项做兜底
			return extract_b64_or_url(data[0])
	except Exception:
		pass
	# 兜底为字符串化
	return str(data)


def generate_images(prompt: str, n: int, size: str, outdir: Path, basename: str) -> List[Path]:
	# 初始化 LangChain 的 Dall-E 封装
	wrapper = DallEAPIWrapper()
	# 兼容性设置：如果封装支持这些属性，则设置之
	for k, v in {
		"model": "dall-e-3",
		"size": size,
		"n": 1,  # 大多数网关/官方 DALL·E 3 一次一张；需要多张则循环
		"response_format": "b64_json",
		"quality": "high",
	}.items():
		if hasattr(wrapper, k):
			try:
				setattr(wrapper, k, v)
			except Exception:
				pass

	saved: List[Path] = []
	for i in range(n):
		# 常见调用：run(prompt) -> 返回 URL 或 JSON
		result = wrapper.run(prompt)
		value = extract_b64_or_url(result)
		if isinstance(value, str) and value.startswith("http"):
			saved.append(save_image_url(value, outdir, basename, i + 1))
		else:
			# 既支持 data:image/...;base64,xxx 也支持纯 b64 字符串
			if value.startswith("data:image"):
				try:
					comma = value.index(",")
					value = value[comma + 1 :]
				except Exception:
					pass
			saved.append(save_image_b64(value, outdir, basename, i + 1))
	return saved


def parse_args(argv: list[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="使用 LangChain 调用 DALL·E 3 生图并保存到 image 目录")
	parser.add_argument("--prompt", required=False, help="图片提示词")
	parser.add_argument("--n", type=int, default=1, help="生成张数（通常建议 1，>1 将循环生成）")
	parser.add_argument("--size", default="1024x1024", help="图片尺寸：256x256/512x512/1024x1024/1024x1536/1536x1024 等")
	parser.add_argument("--outdir", default="image", help="输出目录，默认 image")
	parser.add_argument("--name", default=None, help="输出文件基础名，默认使用时间戳")
	return parser.parse_args(argv)


def main(argv: list[str]) -> int:
	ensure_api_key()
	args = parse_args(argv)

	prompt = args.prompt or "A cute cat astronaut playing guitar in space, high detail, digital art"
	n = max(1, int(args.n))
	size = str(args.size)
	outdir = Path(args.outdir)
	basename = args.name or datetime.now().strftime("dalle3_%Y%m%d_%H%M%S")

	try:
		paths = generate_images(prompt=prompt, n=n, size=size, outdir=outdir, basename=basename)
	except Exception as e:
		print(f"生成失败: {e}", file=sys.stderr)
		return 1

	print("生成完成，文件列表：")
	for p in paths:
		print(f" - {p}")
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))

