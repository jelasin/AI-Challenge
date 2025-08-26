#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Challenge 2: LangChain 工具调用系统（v0.3 + 交互式）
=================================================

要点（v0.3 推荐特性）：
- @tool + args_schema（Pydantic 2）定义工具
- init_chat_model() 一行初始化并支持工具调用
- 模型 bind_tools(tools) 后自动选择并调用工具
- 使用 SystemMessage 约束“必须用工具，不要心算”
- 增加交互式/单次运行两种模式（命令行参数）
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage,
)


def ensure_api_key() -> bool:
    if os.getenv("OPENAI_API_KEY"):
        return True
    print("❌ 未检测到 OPENAI_API_KEY 环境变量。")
    print("💡 在 PowerShell 下：$env:OPENAI_API_KEY = 'your-openai-key'")
    return False


def build_tools():
    """定义并返回可用工具列表。"""

    class AdditionInput(BaseModel):
        a: int = Field(..., description="第一个加数")
        b: int = Field(..., description="第二个加数")

    @tool("Addition", args_schema=AdditionInput)
    def Addition(a: int, b: int) -> int:
        """执行两个整数的加法并返回结果。

        参数:
        - a: 第一个加数
        - b: 第二个加数
        返回: a 与 b 的和
        """
        result = a + b
        print(f"   🔢 执行加法: {a} + {b} = {result}")
        return result

    class MultiplyInput(BaseModel):
        a: int = Field(..., description="被乘数")
        b: int = Field(..., description="乘数")

    @tool("Multiply", args_schema=MultiplyInput)
    def Multiply(a: int, b: int) -> int:
        """执行两个整数的乘法并返回结果。

        参数:
        - a: 被乘数
        - b: 乘数
        返回: a 与 b 的积
        """
        result = a * b
        print(f"   🔢 执行乘法: {a} × {b} = {result}")
        return result

    return [Addition, Multiply]


def run_once(question: str, model_name: str = "gpt-4o-mini") -> None:
    """执行一次“LLM + 工具调用”的完整流程。"""

    print("🚀 初始化支持工具调用的模型...")
    try:
        llm = init_chat_model(model_name, temperature=0)
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return
    print("✅ 模型初始化成功")

    print("\n🛠 创建并绑定数学工具...")
    tools = build_tools()
    llm_with_tools = llm.bind_tools(tools)
    print(f"✅ 已绑定 {len(tools)} 个工具")

    # System 约束：数学题必须通过工具计算，避免心算/幻觉
    sys_prompt = (
        "你是一个严谨的数学助理。对于任何涉及加法或乘法的计算，"
        "必须调用提供的工具完成计算，不要心算，不要只给出大致答案。"
    )

    print(f"\n❓ 用户问题: {question}")
    messages: List[BaseMessage] = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=question),
    ]

    # 第一次调用：模型规划要调用的工具
    print("\n🧠 LLM 分析问题并规划工具调用...")
    try:
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        tool_calls = getattr(ai_msg, "tool_calls", None)
        if not tool_calls:
            print("⚠️ 模型未调用任何工具，直接回答：")
            print(ai_msg.content)
            return
        print(f"🎯 模型计划调用 {len(tool_calls)} 个工具：")
        for i, tc in enumerate(tool_calls, 1):
            print(f"   {i}. 工具: {tc['name']} | 参数: {tc['args']}")
    except Exception as e:
        print(f"❌ 模型调用失败: {e}")
        return

    # 执行工具
    print("\n⚙️ 执行工具调用...")
    tool_map = {t.name: t for t in tools}
    try:
        for tc in tool_calls:
            name = tc["name"]
            args = tc["args"]
            call_id = tc["id"]
            tool_obj = tool_map[name]
            print(f"   🔧 调用工具: {name}")
            output = tool_obj.invoke(args)
            messages.append(ToolMessage(content=str(output), tool_call_id=call_id))
    except Exception as e:
        print(f"❌ 工具执行失败: {e}")
        return

    # 第二次调用：整合工具结果
    print("\n🤖 LLM 整合工具结果，生成最终答案...")
    try:
        final_msg = llm_with_tools.invoke(messages)
    except Exception as e:
        print(f"❌ 生成最终答案失败: {e}")
        return

    class CalculationResult(BaseModel):
        question: str = Field(description="用户提出的问题")
        answer: str = Field(description="AI 的最终答案")
        tools_used: List[str] = Field(description="使用的工具列表")

    result = CalculationResult(
        question=question,
        answer=str(final_msg.content),
        tools_used=[tc["name"] for tc in tool_calls],
    )

    print("\n" + "=" * 60)
    print("🎉 计算完成！结果如下：")
    print("=" * 60)
    print(f"❓ 问题: {result.question}")
    print(f"🛠  使用工具: {', '.join(result.tools_used)}")
    print(f"✅ 答案: {result.answer}")
    print("=" * 60)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangChain v0.3 工具调用演示（交互/单次）")
    parser.add_argument("--model", default="gpt-4o-mini", help="模型名称（默认：gpt-4o-mini）")
    parser.add_argument("--question", help="单次模式下的问题，不提供则进入交互模式")
    parser.add_argument("--once", action="store_true", help="单次模式，仅执行一次并退出")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    print("🎯 LangChain Challenge 2: 工具调用系统 (v0.3)")
    print("学习目标：@tool、Function Calling、工具集成与交互")
    print("-" * 60)

    if not ensure_api_key():
        return 1

    args = parse_args(argv)

    # 单次模式优先
    if args.once or args.question:
        q = args.question or input("❓ 请输入你的计算问题: ").strip()
        if not q:
            print("❌ 问题不能为空")
            return 2
        run_once(q, model_name=args.model)
        return 0

    # 交互式循环
    print("进入交互式模式（输入 exit/quit 退出）\n")
    while True:
        q = input("❓ 问题> ").strip()
        if q.lower() in {"exit", "quit", ":q", "q"}:
            print("👋 已退出。")
            break
        if not q:
            continue
        run_once(q, model_name=args.model)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
