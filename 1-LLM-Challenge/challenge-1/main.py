#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Challenge 1: LangChain v0.3 风格的基础翻译器（LCEL + Streaming + 可选结构化输出）

要点（v0.3 推荐范式）：
- 使用 ChatPromptTemplate 组织消息（system/human）
- 采用 LCEL：prompt | model | parser
- 使用 chain.stream(input) 做流式输出
- 通过 with_structured_output(PydanticModel) 获得结构化结果（可选）
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, cast

from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class TranslationResult(BaseModel):
    """翻译结果结构（用于可选的结构化输出模式）。"""

    translated_text: str = Field(description="翻译后的文本，仅包含译文本身")
    source_language: str = Field(description="检测到的源语言或用户声明的源语言")
    target_language: str = Field(description="目标语言")
    confidence: Optional[str] = Field(
        description="译文质量置信度（模型主观评估）", default="high"
    )


def build_stream_chain(model):
    """构建流式输出的 LCEL 翻译链：prompt | model | StrOutputParser。"""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名专业的翻译助手。只输出最终译文，不要解释，不要附加说明。"
                "保持语义和语气自然准确，专有名词采用常见/标准译法。",
            ),
            (
                "human",
                "请将以下文本从{src_language}翻译成{dst_language}：\n\n{text_message}",
            ),
        ]
    )
    return prompt | model | StrOutputParser()


def build_structured_chain(model):
    """构建结构化输出链：prompt | model.with_structured_output(TranslationResult)。"""

    structured_model = model.with_structured_output(TranslationResult)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名专业的翻译助手。返回的内容必须严格匹配给定的模式。"
                "要求：\n"
                "- translated_text 仅包含译文，不得包含任何解释；\n"
                "- confidence 必须是 'low'、'medium' 或 'high' 之一；无法判断时用 'high'；不得返回 null/None。",
            ),
            (
                "human",
                "从{src_language}翻译到{dst_language}：\n\n{text_message}\n\n"
                "请按模式返回字段（translated_text/source_language/target_language/confidence），"
                "其中 confidence ∈ {{low, medium, high}}。",
            ),
        ]
    )
    return prompt | structured_model


def ensure_api_key() -> bool:
    """检查 OPENAI_API_KEY 环境变量。"""

    if os.getenv("OPENAI_API_KEY"):
        return True
    print("❌ 未检测到 OPENAI_API_KEY 环境变量。")
    print("💡 在 PowerShell 下可执行：")
    print("   $env:OPENAI_API_KEY = 'your-openai-key'")
    print("或在 Linux/macOS：")
    print("   export OPENAI_API_KEY='your-openai-key'")
    return False


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangChain v0.3 风格翻译器（支持流式与结构化两种模式）"
    )
    parser.add_argument("--src", dest="src_language", default="自动检测", help="源语言")
    parser.add_argument("--dst", dest="dst_language", required=False, help="目标语言")
    parser.add_argument("--text", dest="text_message", required=False, help="待翻译文本")
    parser.add_argument(
        "--mode",
        choices=["stream", "structured"],
        default="stream",
        help="输出模式：stream=流式字符输出，structured=返回结构化对象",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI 模型名称（默认：gpt-4o-mini）",
    )
    return parser.parse_args(argv)


def interactive_fill(args: argparse.Namespace) -> argparse.Namespace:
    """交互式补全缺失参数。"""

    print("\n🌍 欢迎使用 LangChain v0.3 智能翻译器")
    print("-" * 48)
    if not args.src_language:
        args.src_language = input("📝 请输入源语言 (如: 中文, English，默认自动检测): ").strip() or "自动检测"
    if not args.dst_language:
        args.dst_language = input("🎯 请输入目标语言 (如: 英文, 中文): ").strip()
    if not args.text_message:
        args.text_message = input("📄 请输入要翻译的文本: ").strip()
    return args


def run_stream_mode(model_name: str, src_language: str, dst_language: str, text_message: str) -> None:
    try:
        # 使用 LangChain 提供的通用初始化方法，自动推断提供商
        llm = init_chat_model(model_name, temperature=0)
    except Exception as e:
        print(f"❌ 初始化模型失败：{e}")
        return

    chain = build_stream_chain(llm)
    inputs = {
        "src_language": src_language or "自动检测",
        "dst_language": dst_language,
        "text_message": text_message,
    }

    print("\n📡 正在流式生成译文：\n")
    collected = []
    try:
        for chunk in chain.stream(inputs):  # 增量 token 输出
            print(chunk, end="", flush=True)
            collected.append(chunk)
        translated = ("".join(collected)).strip()

        print("\n" + "=" * 60)
        print("🎉 翻译完成（流式）")
        print("=" * 60)
        print(f"🔤 源语言：{src_language or '自动检测'}")
        print(f"🎯 目标语言：{dst_language}")
        print(f"📄 原文：{text_message}")
        print(f"✨ 译文：{translated}")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 翻译失败：{e}")


def run_structured_mode(model_name: str, src_language: str, dst_language: str, text_message: str) -> None:
    try:
        # 使用 LangChain 提供的通用初始化方法，自动推断提供商
        llm = init_chat_model(model_name, temperature=0)
    except Exception as e:
        print(f"❌ 初始化模型失败：{e}")
        return

    chain = build_structured_chain(llm)
    inputs = {
        "src_language": src_language or "自动检测",
        "dst_language": dst_language,
        "text_message": text_message,
    }

    print("\n🧩 正在生成结构化结果...\n")
    try:
        result = cast(TranslationResult, chain.invoke(inputs))
        print("=" * 60)
        print("🎉 翻译完成（结构化）")
        print("=" * 60)
        print(f"🔤 源语言：{result.source_language}")
        print(f"🎯 目标语言：{result.target_language}")
        print(f"✨ 译文：{result.translated_text}")
        print(f"📊 置信度：{result.confidence}")
        print("=" * 60)
    except Exception as e:
        print(f"❌ 结构化模式失败：{e}")


def main(argv: list[str]) -> int:
    if not ensure_api_key():
        return 1

    args = parse_args(argv)
    args = interactive_fill(args)

    # 校验必要字段
    if not args.dst_language:
        print("❌ 目标语言不能为空。")
        return 2
    if not args.text_message:
        print("❌ 待翻译文本不能为空。")
        return 2

    if args.mode == "structured":
        run_structured_mode(args.model, args.src_language, args.dst_language, args.text_message)
    else:
        run_stream_mode(args.model, args.src_language, args.dst_language, args.text_message)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
