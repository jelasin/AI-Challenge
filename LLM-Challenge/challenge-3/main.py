# -*- coding: utf-8 -*-
"""
Challenge 3: 高级Prompt Template和Few-shot Learning
难度：中级

学习目标：
1. 掌握复杂的PromptTemplate使用
2. 实现Few-shot Learning
3. 使用Example Selector
4. 实现动态Prompt组合
5. 学习部分格式化（Partial Formatting）

任务描述：
创建一个智能代码评审助手，能够：
1. 根据编程语言动态选择评审规则
2. 使用Few-shot learning提供评审示例
3. 根据代码长度和复杂度选择合适的评审模板
4. 支持多种输出格式（简洁/详细）
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate, 
    FewShotPromptTemplate, 
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from typing import List, Optional
import os

class CodeReviewResult(BaseModel):
    """代码评审结果模型"""
    overall_rating: int = Field(description="代码整体评分（1-10）", ge=1, le=10)
    issues: List[str] = Field(description="发现的问题列表")
    suggestions: List[str] = Field(description="改进建议列表")
    strengths: List[str] = Field(description="代码优点列表")
    summary: str = Field(description="评审总结")

def create_code_review_assistant():
    """创建代码评审助手"""
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("请设置 OPENAI_API_KEY 环境变量")
    
    # 初始化模型
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,  # 保持一致性，但允许少量创造性
        streaming=False
    ).with_structured_output(CodeReviewResult)
    
    # Few-shot学习示例
    examples = [
        {
            "language": "python",
            "code": "def calculate_average(numbers): return sum(numbers) / len(numbers)",
            "review": "评分6分，问题是没有处理空列表情况，建议添加边界检查"
        },
        {
            "language": "java",
            "code": "public class Calculator public static int add int a int b return a plus b",
            "review": "评分7分，代码简洁，建议添加文档注释和溢出处理"
        }
    ]
    
    # 定义示例模板
    example_prompt = PromptTemplate(
        input_variables=["language", "code", "review"],
        template="编程语言: {language}\n代码: {code}\n评审: {review}"
    )
    
    # 创建基于长度的示例选择器
    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=1500,  # 控制prompt长度
    )
    
    # TODO: 实现基于语义相似度的示例选择器
    # 提示：使用SemanticSimilarityExampleSelector和OpenAIEmbeddings
    
    # 创建Few-shot prompt模板
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="你是一个专业的代码评审专家。以下是一些代码评审示例:\n",
        suffix="\n现在请评审以下代码:\n编程语言: {language}\n代码: {code}\n\n请提供结构化的评审结果:",
        input_variables=["language", "code"]
    )
    
    # TODO: 实现动态模板选择
    # 根据代码长度和复杂度选择不同的评审模板
    
    return llm, few_shot_prompt

def create_semantic_example_selector():
    """
    创建基于语义相似度的示例选择器
    
    任务：
    1. 使用OpenAIEmbeddings创建嵌入
    2. 使用FAISS作为向量存储
    3. 创建SemanticSimilarityExampleSelector
    4. 设置k=2（选择最相似的2个示例）
    """
    # 示例数据
    examples = [
        {
            "language": "python",
            "code": "def calculate_average(numbers): return sum(numbers) / len(numbers)",
            "review": "评分6分，问题是没有处理空列表情况，建议添加边界检查"
        },
        {
            "language": "java",
            "code": "public class Calculator public static int add int a int b return a plus b",
            "review": "评分7分，代码简洁，建议添加文档注释和溢出处理"
        },
        {
            "language": "python",
            "code": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "review": "评分5分，递归实现简洁但效率低，建议使用动态规划优化"
        },
        {
            "language": "javascript",
            "code": "function greet(name) console.log hello + name",
            "review": "评分4分，缺少参数验证和错误处理，建议添加输入检查"
        }
    ]
    
    # 创建嵌入模型
    embeddings = OpenAIEmbeddings()
    
    # 创建向量存储
    vectorstore = FAISS.from_texts(
        texts=[f"{ex['language']}: {ex['code']}" for ex in examples],
        embedding=embeddings,
        metadatas=examples
    )
    
    # 创建语义相似度示例选择器
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2  # 选择最相似的2个示例
    )
    
    return example_selector, examples

def create_complex_prompt_template():
    """
    创建复杂的Prompt模板组合
    
    任务：
    1. 使用ChatPromptTemplate创建对话式prompt
    2. 添加系统消息、人类消息和示例消息
    3. 支持部分格式化（partial formatting）
    4. 实现条件性prompt组件
    """
    from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
    
    # 系统消息模板
    system_template = SystemMessagePromptTemplate.from_template(
        "你是一个经验丰富的{role}，专门从事{specialty}。"
        "你的评审风格是{style}，评审时请关注{focus_areas}。"
    )
    
    # 人类消息模板
    human_template = HumanMessagePromptTemplate.from_template(
        "请评审以下{language}代码：\n```{language}\n{code}\n```\n"
        "评审要求：{requirements}\n"
        "输出格式：{output_format}"
    )
    
    # 创建聊天模板
    chat_template = ChatPromptTemplate.from_messages([
        system_template,
        human_template
    ])
    
    # 部分格式化：预设一些常用值
    partial_template = chat_template.partial(
        role="高级软件工程师",
        specialty="代码质量和性能优化",
        style="严谨但建设性",
        focus_areas="代码可读性、性能、安全性和最佳实践",
        requirements="提供详细的问题分析和改进建议",
        output_format="结构化JSON格式"
    )
    
    return partial_template

def demo_partial_formatting():
    """
    演示部分格式化功能
    
    部分格式化允许你预先填入一些变量，在运行时再填入其他变量
    """
    print("=== 部分格式化演示 ===")
    
    # 创建一个需要多个变量的模板
    template = PromptTemplate.from_template(
        "作为{role}，请在{context}的背景下，对以下{language}代码进行评审：\n{code}"
    )
    
    # 部分格式化：预先设置role和context
    partial_template = template.partial(
        role="高级软件工程师",
        context="生产环境部署前的最终检查"
    )
    
    print(f"原始模板变量: {template.input_variables}")
    print(f"部分格式化后的变量: {partial_template.input_variables}")
    
    # 现在只需要提供language和code
    final_prompt = partial_template.format(
        language="Python",
        code="def hello(): print('world')"
    )
    print(f"\n最终Prompt:\n{final_prompt}")

def demo_semantic_selector():
    """演示语义相似度示例选择器"""
    print("\n=== 语义相似度示例选择器演示 ===")
    
    try:
        # 创建语义选择器
        example_selector, examples = create_semantic_example_selector()
        
        # 测试查询
        test_query = "python function with loop"
        selected_examples = example_selector.select_examples({"query": test_query})
        
        print(f"查询: {test_query}")
        print(f"选中的示例数量: {len(selected_examples)}")
        for i, example in enumerate(selected_examples):
            print(f"示例 {i+1}: {example['language']} - {example['code'][:50]}...")
            
    except Exception as e:
        print(f"语义选择器演示失败: {e}")
        print("这可能需要有效的OpenAI API密钥")

def demo_complex_template():
    """演示复杂模板"""
    print("\n=== 复杂聊天模板演示 ===")
    
    try:
        # 创建复杂模板
        template = create_complex_prompt_template()
        
        # 格式化模板
        formatted = template.format(
            language="python",
            code="def hello(): print('world')"
        )
        
        print("生成的聊天消息:")
        print(f"类型: {type(formatted)}")
        print(f"内容: {formatted}")
            
    except Exception as e:
        print(f"复杂模板演示失败: {e}")

def demo_prompt_composition():
    """
    演示Prompt组合功能
    """
    print("\n=== Prompt组合演示 ===")
    
    # 创建可重用的Prompt组件
    system_template = PromptTemplate.from_template(
        "你是一个{expertise}专家，专门从事{focus_area}。"
    )
    
    context_template = PromptTemplate.from_template(
        "当前任务上下文：{context}\n评审标准：{standards}"
    )
    
    task_template = PromptTemplate.from_template(
        "请评审以下{language}代码：\n{code}"
    )
    
    # 手动组合模板字符串，避免嵌套花括号
    combined_template_str = (
        "你是一个{expertise}专家，专门从事{focus_area}。\n\n"
        "当前任务上下文：{context}\n评审标准：{standards}\n\n"
        "请评审以下{language}代码：\n{code}"
    )
    
    combined_template = PromptTemplate.from_template(combined_template_str)
    
    print(f"组合后的模板:\n{combined_template.template}")
    print(f"需要的变量: {combined_template.input_variables}")
    """
    演示Prompt组合功能
    """
    print("\n=== Prompt组合演示 ===")
    
    # 创建可重用的Prompt组件
    system_template = PromptTemplate.from_template(
        "你是一个{expertise}专家，专门从事{focus_area}。"
    )
    
    context_template = PromptTemplate.from_template(
        "当前任务上下文：{context}\n评审标准：{standards}"
    )
    
    task_template = PromptTemplate.from_template(
        "请评审以下{language}代码：\n{code}"
    )
    
    # 手动组合模板字符串，避免嵌套花括号
    combined_template_str = (
        "你是一个{expertise}专家，专门从事{focus_area}。\n\n"
        "当前任务上下文：{context}\n评审标准：{standards}\n\n"
        "请评审以下{language}代码：\n{code}"
    )
    
    combined_template = PromptTemplate.from_template(combined_template_str)
    
    print(f"组合后的模板:\n{combined_template.template}")
    print(f"需要的变量: {combined_template.input_variables}")

def main():
    """主函数"""
    try:
        print("🔍 LangChain Challenge 3: 高级Prompt Template和Few-shot Learning")
        print("=" * 60)
        
        # 演示部分格式化
        demo_partial_formatting()
        
        # 演示Prompt组合
        demo_prompt_composition()
        
        # 演示语义选择器
        demo_semantic_selector()
        
        # 演示复杂模板
        demo_complex_template()
        
        print("\n" + "=" * 60)
        print("开始代码评审演示...")
        
        # 创建评审助手
        llm, prompt = create_code_review_assistant()
        
        # 测试代码
        test_code = """
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
    return result
        """
        
        # 生成评审
        try:
            formatted_prompt = prompt.format(
                language="python",
                code=test_code
            )
            
            print(f"\n生成的Prompt:\n{formatted_prompt}")
            print(f"\nPrompt长度: {len(formatted_prompt)} 字符")
            
        except Exception as format_error:
            print(f"格式化Prompt时发生错误: {format_error}")
            # 简化错误处理，避免引用未定义的变量
            print("可能是示例数据格式问题，请检查代码中的示例定义")
            return
        
        # 获取评审结果
        print("\n正在分析代码...")
        review_result = llm.invoke(formatted_prompt)
        
        print(f"\n📊 代码评审结果:")
        if isinstance(review_result, CodeReviewResult):
            print(f"整体评分: {review_result.overall_rating}/10")
            print(f"发现的问题: {', '.join(review_result.issues)}")
            print(f"改进建议: {', '.join(review_result.suggestions)}")
            print(f"代码优点: {', '.join(review_result.strengths)}")
            print(f"总结: {review_result.summary}")
        else:
            # 如果返回的是字典或其他格式
            print(f"评审结果: {review_result}")
        
        print("\n" + "=" * 60)
        print("🎯 练习任务:")
        print("1. 实现create_semantic_example_selector()函数")
        print("2. 实现create_complex_prompt_template()函数")
        print("3. 添加代码复杂度检测，根据复杂度选择不同的评审模板")
        print("4. 实现支持多种输出格式的动态模板")
        print("5. 添加更多编程语言的评审示例")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("\n请确保:")
        print("1. 已设置 OPENAI_API_KEY 环境变量")
        print("2. 已安装所需的依赖包: pip install langchain langchain-openai faiss-cpu")

if __name__ == "__main__":
    main()
