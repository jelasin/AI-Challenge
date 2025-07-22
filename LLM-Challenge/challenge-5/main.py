# -*- coding: utf-8 -*-
"""
Challenge 5: LCEL (LangChain Expression Language) 和链组合
难度：中级到高级

学习目标：
1. 掌握LCEL语法和核心概念
2. 学习Runnable接口的使用
3. 实现复杂链的组合和分支
4. 掌握并行处理和路由
5. 学习流式处理和异步操作
6. 实现错误处理和回退机制

任务描述：
创建一个智能内容处理管道，能够：
1. 并行处理多种内容分析任务
2. 根据内容类型动态路由
3. 实现流式输出
4. 添加错误处理和回退
5. 支持配置化和可扩展性
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableParallel, 
    RunnableLambda,
    RunnableBranch,
    Runnable
)
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union, Literal
import asyncio
import json
import os
import time

class ContentAnalysis(BaseModel):
    """内容分析结果模型"""
    content_type: Literal["article", "code", "data", "unknown"] = Field(description="内容类型")
    language: Optional[str] = Field(description="语言（如果是代码或文本）")
    sentiment: Optional[Literal["positive", "negative", "neutral"]] = Field(description="情感倾向")
    complexity: Literal["low", "medium", "high"] = Field(description="复杂度")
    key_topics: List[str] = Field(description="关键主题")
    summary: str = Field(description="内容摘要")
    confidence: float = Field(description="分析置信度", ge=0.0, le=1.0)

class ProcessingResult(BaseModel):
    """处理结果模型"""
    analysis: ContentAnalysis
    suggestions: List[str]
    metadata: Dict[str, Any]
    processing_time: float

def create_content_classifier():
    """创建内容分类器"""
    prompt = ChatPromptTemplate.from_template("""
    分析以下内容并确定其类型：

    内容：{content}

    请判断这是：
    - article: 文章或文档
    - code: 代码片段
    - data: 数据或表格
    - unknown: 无法确定

    只返回类型名称，不要解释。
    """)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    return prompt | llm | StrOutputParser()

def create_sentiment_analyzer():
    """创建情感分析器"""
    prompt = ChatPromptTemplate.from_template("""
    分析以下内容的情感倾向：

    内容：{content}

    请判断情感倾向：positive, negative, 或 neutral
    只返回一个词，不要解释。
    """)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    return prompt | llm | StrOutputParser()

def create_summarizer():
    """创建摘要生成器"""
    prompt = ChatPromptTemplate.from_template("""
    请为以下内容生成一个简洁的摘要（不超过100字）：

    内容：{content}

    摘要：
    """)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    return prompt | llm | StrOutputParser()

def create_topic_extractor():
    """创建主题提取器"""
    prompt = ChatPromptTemplate.from_template("""
    从以下内容中提取3-5个关键主题或关键词：

    内容：{content}

    请以JSON格式返回主题列表：
    {{"topics": ["主题1", "主题2", "主题3"]}}
    """)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    return prompt | llm | JsonOutputParser()

def create_complexity_analyzer():
    """创建复杂度分析器"""
    prompt = ChatPromptTemplate.from_template("""
    评估以下内容的复杂度：

    内容：{content}

    复杂度标准：
    - low: 简单易懂，基础概念
    - medium: 需要一定专业知识
    - high: 复杂概念，需要深度专业知识

    只返回复杂度等级：low, medium, 或 high
    """)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    return prompt | llm | StrOutputParser()

def create_language_detector():
    """创建语言检测器（针对代码）"""
    def detect_language(content: str) -> str:
        """简单的语言检测逻辑"""
        content_lower = content.lower()
        
        # 检测常见的代码特征
        if any(keyword in content_lower for keyword in ['def ', 'import ', 'from ', 'print(']):
            return "python"
        elif any(keyword in content_lower for keyword in ['function', 'var ', 'let ', 'const ']):
            return "javascript"  
        elif any(keyword in content_lower for keyword in ['public class', 'private ', 'static void']):
            return "java"
        elif any(keyword in content_lower for keyword in ['#include', 'int main', 'printf']):
            return "c"
        elif any(keyword in content_lower for keyword in ['SELECT', 'FROM', 'WHERE', 'INSERT']):
            return "sql"
        else:
            return "unknown"
    
    return RunnableLambda(lambda x: detect_language(x.get("content", "") if isinstance(x, dict) else str(x)))

def demo_basic_lcel():
    """演示基本的LCEL语法"""
    print("🔗 LCEL基础语法演示")
    print("=" * 50)
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 请设置 OPENAI_API_KEY 环境变量")
        return
    
    # 1. 简单的链组合
    print("1. 简单链组合 (prompt | llm | parser):")
    
    prompt = ChatPromptTemplate.from_template("将以下文本翻译成英文：{text}")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    parser = StrOutputParser()
    
    # 使用管道操作符组合
    simple_chain = prompt | llm | parser
    
    result = simple_chain.invoke({"text": "你好，世界！"})
    print(f"   输入: 你好，世界！")
    print(f"   输出: {result}")
    
    # 2. 使用RunnablePassthrough
    print("\n2. 使用RunnablePassthrough保持输入:")
    
    passthrough_chain = RunnableParallel({
        "original": RunnablePassthrough(),
        "translation": prompt | llm | parser
    })
    
    result = passthrough_chain.invoke({"text": "早上好"})
    print(f"   结果: {result}")
    
    # 3. 使用RunnableParallel并行处理
    print("\n3. 并行处理演示:")
    
    parallel_chain = RunnableParallel({
        "english": ChatPromptTemplate.from_template("Translate to English: {text}") | llm | parser,
        "summary": ChatPromptTemplate.from_template("Summarize in Chinese: {text}") | llm | parser,
        "keywords": ChatPromptTemplate.from_template("Extract 3 keywords from: {text}") | llm | parser
    })
    
    result = parallel_chain.invoke({"text": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"})
    print(f"   英文翻译: {result['english'][:50]}...")
    print(f"   中文摘要: {result['summary'][:50]}...")
    print(f"   关键词: {result['keywords'][:50]}...")

def create_content_router():
    """创建内容路由器"""
    
    def route_content(content_info: Dict[str, Any]) -> Runnable:
        """根据内容类型路由到不同的处理链"""
        content_type = content_info.get("content_type", "unknown")
        
        if content_type == "code":
            return create_code_processor()
        elif content_type == "article":
            return create_article_processor()
        elif content_type == "data":
            return create_data_processor()
        else:
            return create_default_processor()
    
    return RunnableLambda(route_content)

def create_code_processor():
    """创建代码处理器"""
    prompt = ChatPromptTemplate.from_template("""
    分析以下代码：

    代码：{content}
    语言：{language}

    请提供：
    1. 代码功能说明
    2. 代码质量评估
    3. 优化建议

    以JSON格式返回：
    {{
        "functionality": "功能说明",
        "quality_score": "评分(1-10)",
        "suggestions": ["建议1", "建议2"]
    }}
    """)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    return prompt | llm | JsonOutputParser()

def create_article_processor():
    """创建文章处理器"""
    prompt = ChatPromptTemplate.from_template("""
    分析以下文章：

    文章：{content}

    请提供：
    1. 文章主要观点
    2. 写作质量评估
    3. 改进建议

    以JSON格式返回：
    {{
        "main_points": ["观点1", "观点2"],
        "writing_quality": "质量评估",
        "suggestions": ["建议1", "建议2"]
    }}
    """)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    return prompt | llm | JsonOutputParser()

def create_data_processor():
    """创建数据处理器"""
    prompt = ChatPromptTemplate.from_template("""
    分析以下数据：

    数据：{content}

    请提供：
    1. 数据结构分析
    2. 数据质量评估
    3. 分析建议

    以JSON格式返回：
    {{
        "structure": "结构描述",
        "quality": "质量评估", 
        "suggestions": ["建议1", "建议2"]
    }}
    """)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    return prompt | llm | JsonOutputParser()

def create_default_processor():
    """创建默认处理器"""
    prompt = ChatPromptTemplate.from_template("""
    分析以下内容：

    内容：{content}

    请提供通用的内容分析和建议。

    以JSON格式返回：
    {{
        "analysis": "分析结果",
        "suggestions": ["建议1", "建议2"]
    }}
    """)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    return prompt | llm | JsonOutputParser()

def create_advanced_content_pipeline():
    """创建高级内容处理管道"""
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("请设置 OPENAI_API_KEY 环境变量")
    
    print("🏗️ 构建高级内容处理管道...")
    
    # 第一阶段：并行分析
    analysis_stage = RunnableParallel({
        "content_type": create_content_classifier(),
        "sentiment": create_sentiment_analyzer(), 
        "summary": create_summarizer(),
        "topics": create_topic_extractor(),
        "complexity": create_complexity_analyzer(),
        "language": create_language_detector(),
        "original": RunnablePassthrough()
    })
    
    # 第二阶段：路由处理
    def create_routing_logic():
        """创建路由逻辑"""
        def route_and_process(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
            content_type = analysis_result.get("content_type", "unknown").strip()
            content = analysis_result["original"]["content"]
            
            # 根据内容类型选择处理器
            if "code" in content_type.lower():
                processor = create_code_processor()
            elif "article" in content_type.lower():
                processor = create_article_processor()
            elif "data" in content_type.lower():
                processor = create_data_processor()
            else:
                processor = create_default_processor()
            
            # 处理内容
            try:
                processing_result = processor.invoke({
                    "content": content,
                    "language": analysis_result.get("language", "unknown")
                })
            except Exception as e:
                processing_result = {
                    "error": f"处理失败: {str(e)}",
                    "suggestions": ["请检查内容格式", "尝试重新提交"]
                }
            
            # 合并结果
            return {
                "analysis": ContentAnalysis(
                    content_type=content_type if content_type in ["article", "code", "data"] else "unknown",
                    language=analysis_result.get("language"),
                    sentiment=analysis_result.get("sentiment"),
                    complexity=analysis_result.get("complexity", "medium"),
                    key_topics=analysis_result.get("topics", {}).get("topics", []),
                    summary=analysis_result.get("summary", ""),
                    confidence=0.8  # 简化的置信度
                ),
                "processing_result": processing_result,
                "metadata": {
                    "processing_time": time.time(),
                    "content_length": len(content)
                }
            }
        
        return RunnableLambda(route_and_process)
    
    # 组合完整管道
    complete_pipeline = analysis_stage | create_routing_logic()
    
    return complete_pipeline

def demo_streaming():
    """演示流式处理"""
    print("\n🌊 流式处理演示")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 请设置 OPENAI_API_KEY 环境变量")
        return
    
    # 创建支持流式输出的链
    prompt = ChatPromptTemplate.from_template("""
    请详细解释以下概念，包括定义、特点、应用场景等：

    概念：{concept}
    """)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, streaming=True)
    chain = prompt | llm | StrOutputParser()
    
    print("流式输出示例（概念解释）:")
    print("-" * 30)
    
    # 流式处理
    for chunk in chain.stream({"concept": "机器学习"}):
        print(chunk, end="", flush=True)
    
    print("\n")

async def demo_async_processing():
    """演示异步处理"""
    print("\n⚡ 异步处理演示")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 请设置 OPENAI_API_KEY 环境变量")
        return
    
    # 创建异步链
    prompt = ChatPromptTemplate.from_template("用一句话概括：{topic}")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    
    # 准备多个任务
    topics = ["人工智能", "区块链", "量子计算", "生物技术", "新能源"]
    
    print("并发处理多个主题...")
    start_time = time.time()
    
    # 并发执行
    tasks = [chain.ainvoke({"topic": topic}) for topic in topics]
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    print(f"处理完成，用时：{end_time - start_time:.2f}秒")
    for topic, result in zip(topics, results):
        print(f"  {topic}: {result}")

def demo_error_handling():
    """演示错误处理和回退机制"""
    print("\n🛡️ 错误处理演示")
    print("=" * 50)
    
    # 创建可能失败的链
    def failing_function(x):
        if "error" in x["text"].lower():
            raise ValueError("故意触发的错误")
        return x["text"].upper()
    
    # 创建回退链
    def fallback_function(x):
        return f"回退处理: {x['text']}"
    
    # 使用try-except包装的链
    def safe_processing(x):
        try:
            return failing_function(x)
        except Exception as e:
            print(f"  ⚠️ 捕获错误: {e}")
            return fallback_function(x)
    
    safe_chain = RunnableLambda(safe_processing)
    
    # 测试正常情况
    print("1. 正常处理:")
    result = safe_chain.invoke({"text": "hello world"})
    print(f"   结果: {result}")
    
    # 测试错误情况
    print("\n2. 错误处理:")
    result = safe_chain.invoke({"text": "this will cause an error"})
    print(f"   结果: {result}")

def main():
    """主函数"""
    try:
        print("🔗 LangChain Challenge 5: LCEL和链组合")
        print("=" * 60)
        
        # 基础LCEL演示
        demo_basic_lcel()
        
        # 流式处理演示
        demo_streaming()
        
        # 错误处理演示
        demo_error_handling()
        
        print("\n" + "=" * 60)
        print("🏗️ 构建和测试高级内容处理管道...")
        
        # 创建高级管道
        pipeline = create_advanced_content_pipeline()
        
        # 测试不同类型的内容
        test_contents = [
            {
                "content": """
                def fibonacci(n):
                    if n <= 1:
                        return n
                    return fibonacci(n-1) + fibonacci(n-2)
                """,
                "description": "Python代码"
            },
            {
                "content": """
                人工智能的发展正在改变我们的世界。从自动驾驶汽车到智能语音助手，
                AI技术已经深入到我们生活的方方面面。然而，这种快速发展也带来了
                新的挑战和机遇。我们需要谨慎地平衡技术进步与伦理考量，确保AI
                的发展能够造福全人类。
                """,
                "description": "中文文章"
            },
            {
                "content": """
                Name,Age,Department,Salary
                Alice,25,Engineering,75000
                Bob,30,Marketing,65000
                Carol,28,Sales,70000
                David,35,Engineering,85000
                """,
                "description": "CSV数据"
            }
        ]
        
        for i, test_case in enumerate(test_contents, 1):
            print(f"\n📝 测试案例 {i}: {test_case['description']}")
            print("-" * 40)
            
            start_time = time.time()
            result = pipeline.invoke({"content": test_case["content"]})
            end_time = time.time()
            
            analysis = result["analysis"]
            processing_result = result["processing_result"]
            
            print(f"🔍 分析结果:")
            print(f"   类型: {analysis.content_type}")
            print(f"   语言: {analysis.language}")
            print(f"   情感: {analysis.sentiment}")
            print(f"   复杂度: {analysis.complexity}")
            print(f"   关键主题: {', '.join(analysis.key_topics[:3])}")
            print(f"   摘要: {analysis.summary[:100]}...")
            print(f"   置信度: {analysis.confidence:.2f}")
            
            print(f"\n⚙️ 处理结果:")
            if isinstance(processing_result, dict):
                for key, value in processing_result.items():
                    if key != "suggestions":
                        print(f"   {key}: {str(value)[:80]}...")
            
            print(f"\n⏱️ 处理时间: {end_time - start_time:.2f}秒")
        
        print("\n" + "=" * 60)
        print("⚡ 异步处理演示...")
        
        # 异步处理演示
        asyncio.run(demo_async_processing())
        
        print("\n" + "=" * 60)
        print("🎯 练习任务:")
        print("1. 实现条件分支路由 (RunnableBranch)")
        print("2. 添加链的配置化功能 (RunnableConfig)")
        print("3. 实现自定义Runnable类")
        print("4. 添加链的监控和日志功能")
        print("5. 实现链的序列化和反序列化")
        print("6. 添加更复杂的错误处理和重试机制")
        print("7. 实现动态链组合和插件系统")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("\n请确保:")
        print("1. 已设置 OPENAI_API_KEY 环境变量")
        print("2. 已安装所需的依赖包: pip install langchain langchain-openai")

if __name__ == "__main__":
    main()
