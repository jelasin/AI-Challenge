# -*- coding: utf-8 -*-
"""
Challenge 3: 高级 Prompt & Few-shot v0.3

变更要点：
- 使用 LangChain 0.3 的 LCEL 可组合链写法与 structured_output
- 通过 `-f` 选项只指定代码文件，由 LLM 自动识别语言
- 取消内置 test_code，不再内置示例运行
"""

from __future__ import annotations

import argparse
import os
from operator import itemgetter
from typing import List

from pydantic import BaseModel, Field

from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS


# ---------------------------
# 数据模型（structured output）
# ---------------------------
class LanguageGuess(BaseModel):
    language: str = Field(description="识别到的主要编程语言")
    confidence: int = Field(ge=1, le=100, description="置信度 1-100")


class CodeReviewResult(BaseModel):
    overall_rating: int = Field(description="代码整体评分（1-10）", ge=1, le=10)
    issues: List[str] = Field(description="发现的问题列表")
    suggestions: List[str] = Field(description="改进建议列表")
    strengths: List[str] = Field(description="代码优点列表")
    summary: str = Field(description="评审总结")


# ---------------------------
# 规则/工具函数
# ---------------------------
def rules_for_language(lang: str) -> str:
    l = (lang or "").strip().lower()
    common = (
        "通用: 可读性、健壮性、边界条件、错误处理、日志、注释、测试、性能与安全最佳实践。"
    )
    mapping = {
        "python": "Python: PEP8、类型提示、异常处理、迭代器/生成器、列表推导、上下文管理、GIL/并发。",
        "java": "Java: OOP 设计、异常规范、线程安全、集合与流 API、内存与GC、注解与文档。",
        "javascript": "JavaScript: 异步/Promise、错误处理、ES 模块、原型链与作用域、XSS/CSRF。",
        "typescript": "TypeScript: 类型完整性、泛型、严格模式、接口/类型、枚举、Union/Never。",
        "c#": "C#: 异步/await、LINQ、内存/Span、异常与日志、Nullable、依赖注入。",
        "c++": "C/C++: RAII、内存管理、异常安全、拷贝/移动语义、并发、UB 风险。",
        "c/c++": "C/C++: RAII、内存管理、异常安全、拷贝/移动语义、并发、UB 风险。",
        "go": "Go: 错误处理、并发 goroutine/context、接口与切片、逃逸分析、包结构。",
        "rust": "Rust: 所有权与借用、生命周期、Result/Option、并发/Send/Sync、unsafe 审慎使用。",
        "php": "PHP: 类型声明、输入校验、错误级别、依赖管理、模板注入、防注入。",
        "ruby": "Ruby: 可读 DSL、块/Proc、异常处理、元编程约束、Rails 约定。",
        "shell": "Shell: set -euo pipefail、安全引用、可移植性、外部命令错误处理。",
        "sql": "SQL: 索引与执行计划、事务与隔离级别、注入防护、分页与聚合性能。",
        "javascript": "JavaScript: 异步/Promise、错误处理、模块化、XSS/CSRF。",
    }
    return mapping.get(l, common + " 若语言未知则从语法与上下文推断。")


def calc_mode_by_length(code: str) -> str:
    """根据代码长度选择输出模式（简洁/详细）。"""
    lines = len(code.splitlines())
    return "简洁" if lines > 200 else "详细"


def load_code(path: str) -> str:
    """尽力读取文件（尝试 utf-8 / gbk / latin-1）。"""
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    raise ValueError(f"无法读取文件: {path}")


# typed helpers for LCEL lambdas
def _escape_braces(text: str) -> str:
    """
    将花括号转义为用于 f-string 模板的安全形式。
    """
    return text.replace("{", "{{").replace("}", "}}")


def _get_language_from_guess(guess: object) -> str:
    if hasattr(guess, "language"):
        try:
            return str(getattr(guess, "language"))
        except Exception:
            pass
    if isinstance(guess, dict):  # type: ignore[reportGeneralTypeIssues]
        return str(guess.get("language", ""))
    return ""


def _step_rules(x: dict) -> dict:
    code = str(x.get("code", ""))
    language = str(x.get("language", ""))
    return {**x, "rules": rules_for_language(language), "mode": calc_mode_by_length(code)}


def _promptvalue_to_str(x: object) -> str:
    """将 PromptValue/消息安全转换为字符串，用于插入到模板变量中。"""
    try:
        if hasattr(x, "to_string") and callable(getattr(x, "to_string")):
            return str(getattr(x, "to_string")())
        if hasattr(x, "text"):
            return str(getattr(x, "text"))
        return str(x)
    except Exception:
        return str(x)


# ---------------------------
# 构建 Few-shot 示例选择器（语义相似度）
# ---------------------------
def build_example_selector():
    examples = [
        {
            "language": "python",
            "code": "def average(xs):\n    return sum(xs)/len(xs)",
            "review": "处理空列表，添加类型提示与异常处理。",
        },
        {
            "language": "java",
            "code": "class C { int add(int a,int b){ return a+b; } }",
            "review": "添加文档注释、参数校验，考虑溢出与单元测试。",
        },
        {
            "language": "javascript",
            "code": "function greet(n){ console.log('hi '+n) }",
            "review": "校验参数类型，处理 null/undefined，避免 XSS。",
        },
        {
            "language": "go",
            "code": "func Sum(a,b int) int { return a+b }",
            "review": "错误处理、命名规范、测试用例与基准测试。",
        },
        {
            "language": "rust",
            "code": "fn add(a:i32,b:i32)->i32{a+b}",
            "review": "使用 Result 处理错误，添加文档与单元测试。",
        },
    ]
    # 为模板渲染安全地转义示例中的花括号
    for ex in examples:
        ex["code_escaped"] = _escape_braces(ex["code"])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(
        texts=[f"{ex['language']}: {ex['code']}" for ex in examples],
        embedding=embeddings,
        metadatas=examples,
    )

    selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore, k=2)
    return selector


# ---------------------------
# 构建链：语言识别 -> Prompt(含 Few-shot) -> 结构化输出
# ---------------------------
def build_chain():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("未检测到 OPENAI_API_KEY 环境变量")

    # 语言识别
    lang_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是编程语言识别器。判断用户给定代码的主要编程语言。"),
        (
            "human",
            "只返回 JSON，字段 language(语言名) 与 confidence(1-100)。\n代码：\n```\n{code}\n```",
        ),
    ])
    lang_llm = init_chat_model("gpt-4o-mini", temperature=0)
    lang_detector = lang_prompt | lang_llm.with_structured_output(LanguageGuess)

    # Few-shot（动态示例选择）
    # 注意：使用已转义的 code_escaped，避免花括号触发模板占位符错误
    example_prompt = PromptTemplate.from_template(
        "编程语言: {language}\n代码: {code_escaped}\n评审: {review}"
    )
    selector = build_example_selector()
    few_shot = FewShotPromptTemplate(
        example_selector=selector,
        example_prompt=example_prompt,
        input_variables=["language", "code"],
        prefix="以下为历史评审示例：",
        suffix="——示例结束——",
    )

    # 主评审 Prompt（支持规则与模式）
    main_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是资深代码评审专家。语言: {language}。审查重点: {rules}。输出模式: {mode}。\n"
                "请输出严格的 JSON，字段包括 overall_rating(1-10)、issues、suggestions、strengths、summary。",
            ),
            ("system", "以下为若干参考评审示例：\n{few_shot}"),
            # 使用 code_for_prompt（已转义花括号）
            ("human", "请评审以下代码：\n```{language}\n{code_for_prompt}\n```"),
        ]
    )

    review_llm = init_chat_model("gpt-4o", temperature=0.1)

    # 组合链（LCEL）
    # 1) 提取 code 并识别语言
    step_detect = RunnableMap(
        {
            "code": itemgetter("code"),
            "language": RunnableLambda(lambda x: x["code"])  # str code
            | lang_detector  # -> LanguageGuess
            | RunnableLambda(_get_language_from_guess),
        }
    )

    # 2) 基于上一步的输出，计算规则与模式
    step_rules = RunnableLambda(_step_rules)

    # 3) 生成 few-shot 文本并拼装最终 prompt 变量
    step_fewshot = RunnableMap(
        {
            "code": itemgetter("code"),
            "language": itemgetter("language"),
            "rules": itemgetter("rules"),
            "mode": itemgetter("mode"),
            # FewShotPromptTemplate -> PromptValue，需要转换为纯字符串
            "few_shot": few_shot | RunnableLambda(_promptvalue_to_str),
            # 为最终主提示提供已转义的代码文本，防止花括号干扰
            "code_for_prompt": itemgetter("code") | RunnableLambda(_escape_braces),
        }
    )

    chain = step_detect | step_rules | step_fewshot | main_prompt | review_llm.with_structured_output(
        CodeReviewResult
    )

    return chain


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="智能代码评审助手（LangChain 0.3）")
    parser.add_argument(
        "-f", "--file", required=True, help="需要评审的代码文件路径"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    code = load_code(args.file)

    try:
        chain = build_chain()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("请确认已安装依赖并设置 OPENAI_API_KEY。")
        return

    print("🔎 正在分析并识别语言…")
    # 先单独跑一次语言识别，给用户一个可见回显
    lang_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是编程语言识别器。判断用户给定代码的主要编程语言。"),
        (
            "human",
            "只返回 JSON，字段 language 与 confidence。\n代码：\n```\n{code}\n```",
        ),
    ])
    lang_llm = init_chat_model("gpt-4o-mini", temperature=0)
    lang_detector = lang_prompt | lang_llm.with_structured_output(LanguageGuess)
    try:
        guess_raw = lang_detector.invoke({"code": code})
        # 兼容 dict 或模型
        if isinstance(guess_raw, dict):
            lang = guess_raw.get("language", "未知")
            conf = guess_raw.get("confidence", "?")
        else:
            lang = getattr(guess_raw, "language", "未知")
            conf = getattr(guess_raw, "confidence", "?")
        print(f"语言识别: {lang}（置信度 {conf}）")
    except Exception:
        print("语言识别阶段发生问题，将在评审链内重试。")

    print("🧪 正在进行代码评审…")
    try:
        result_any = chain.invoke({"code": code})
        # 统一为模型实例
        if isinstance(result_any, dict):
            # Pydantic v2: model_validate；若为 v1，可回退到构造函数
            try:
                result: CodeReviewResult = CodeReviewResult.model_validate(result_any)  # type: ignore[attr-defined]
            except Exception:
                result = CodeReviewResult(**result_any)
        else:
            result = result_any  # type: ignore[assignment]
    except Exception as e:
        print(f"❌ 评审失败: {e}")
        return

    print("\n📊 代码评审结果")
    print(f"整体评分: {result.overall_rating}/10")
    print(f"发现的问题: {', '.join(result.issues) if result.issues else '无'}")
    print(f"改进建议: {', '.join(result.suggestions) if result.suggestions else '无'}")
    print(f"代码优点: {', '.join(result.strengths) if result.strengths else '无'}")
    print(f"总结: {result.summary}")


if __name__ == "__main__":
    main()
