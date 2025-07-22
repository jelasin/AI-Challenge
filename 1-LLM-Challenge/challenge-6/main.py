# -*- coding: utf-8 -*-
"""
Challenge 6: 智能Agent和工具集成
难度：高级

学习目标：
1. 理解Agent的概念和架构
2. 创建自定义工具和工具集
3. 实现工具调用和结果处理
4. 构建多步推理Agent
5. 实现工具错误处理和重试
6. 学习Agent的内存和状态管理

任务描述：
创建一个多功能智能助手Agent，能够：
1. 调用多种工具完成复杂任务
2. 进行多步推理和规划
3. 处理工具调用失败
4. 维护对话历史和状态
5. 支持并行工具调用
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Type, Union
import requests
import json
import math
import random
import time
import os
from datetime import datetime, timedelta

# =========================== 工具定义 ===========================

class CalculatorInput(BaseModel):
    """计算器输入模型"""
    expression: str = Field(description="数学表达式，支持基本运算符 (+, -, *, /, **, sqrt, sin, cos等)")

@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """
    安全的数学表达式计算器
    支持基本算术运算、三角函数、平方根等
    """
    try:
        # 安全的数学表达式求值
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow,
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
            'tan': math.tan, 'log': math.log, 'log10': math.log10,
            'exp': math.exp, 'pi': math.pi, 'e': math.e
        }
        
        # 替换一些常见的数学符号
        expression = expression.replace('^', '**')
        
        # 计算结果
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {expression} = {result}"
        
    except Exception as e:
        return f"计算错误: {str(e)}"

class WeatherInput(BaseModel):
    """天气查询输入模型"""
    city: str = Field(description="城市名称")
    days: int = Field(default=1, description="查询天数（1-7天）", ge=1, le=7)

@tool("weather", args_schema=WeatherInput)
def get_weather(city: str, days: int = 1) -> str:
    """
    获取指定城市的天气信息
    注意：这是一个模拟的天气API，实际使用时应该调用真实的天气服务
    """
    try:
        # 模拟天气数据
        weather_conditions = ["晴", "多云", "小雨", "中雨", "阴", "雾"]
        temperatures = list(range(15, 30))
        
        weather_info = []
        base_date = datetime.now()
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            condition = random.choice(weather_conditions)
            temp = random.choice(temperatures)
            humidity = random.randint(40, 80)
            
            day_weather = {
                "日期": date.strftime("%Y-%m-%d"),
                "城市": city,
                "天气": condition,
                "温度": f"{temp}°C",
                "湿度": f"{humidity}%"
            }
            weather_info.append(day_weather)
        
        if days == 1:
            info = weather_info[0]
            return f"{info['城市']}今天天气：{info['天气']}，温度{info['温度']}，湿度{info['湿度']}"
        else:
            result = f"{city}未来{days}天天气预报：\\n"
            for info in weather_info:
                result += f"{info['日期']}：{info['天气']}，{info['温度']}，湿度{info['湿度']}\\n"
            return result
            
    except Exception as e:
        return f"天气查询失败: {str(e)}"

class TextAnalysisInput(BaseModel):
    """文本分析输入模型"""
    text: str = Field(description="要分析的文本")
    analysis_type: str = Field(description="分析类型：sentiment(情感), keywords(关键词), summary(摘要)", default="summary")

@tool("text_analyzer", args_schema=TextAnalysisInput)
def analyze_text(text: str, analysis_type: str = "summary") -> str:
    """
    文本分析工具：提供情感分析、关键词提取、文本摘要等功能
    """
    try:
        if analysis_type == "sentiment":
            # 简单的情感分析（实际应该使用NLP库）
            positive_words = ["好", "棒", "优秀", "满意", "喜欢", "开心", "fantastic", "great", "good"]
            negative_words = ["差", "糟糕", "失望", "讨厌", "生气", "bad", "terrible", "awful"]
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = "积极"
            elif negative_count > positive_count:
                sentiment = "消极"
            else:
                sentiment = "中性"
            
            return f"文本情感分析结果：{sentiment} (积极词汇: {positive_count}, 消极词汇: {negative_count})"
        
        elif analysis_type == "keywords":
            # 简单的关键词提取（实际应该使用TF-IDF等算法）
            words = text.replace("，", " ").replace("。", " ").replace("！", " ").replace("？", " ").split()
            word_freq = {}
            for word in words:
                if len(word) > 1:  # 忽略单字符
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # 取频率最高的几个词作为关键词
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            keyword_list = [word for word, freq in keywords if freq > 1]
            
            return f"文本关键词：{', '.join(keyword_list)}"
        
        elif analysis_type == "summary":
            # 简单的文本摘要（取前两句话）
            sentences = text.replace("！", "。").replace("？", "。").split("。")
            summary_sentences = [s.strip() for s in sentences[:2] if s.strip()]
            summary = "。".join(summary_sentences)
            if summary and not summary.endswith("。"):
                summary += "。"
            
            return f"文本摘要：{summary}"
        
        else:
            return f"不支持的分析类型：{analysis_type}"
            
    except Exception as e:
        return f"文本分析失败: {str(e)}"

class FileOperationInput(BaseModel):
    """文件操作输入模型"""
    operation: str = Field(description="操作类型：create(创建), read(读取), write(写入), list(列表)")
    filename: str = Field(description="文件名")
    content: Optional[str] = Field(description="文件内容（写入时需要）", default=None)

@tool("file_manager", args_schema=FileOperationInput)
def file_manager(operation: str, filename: str, content: Optional[str] = None) -> str:
    """
    文件管理工具：支持文件的创建、读取、写入和列表操作
    注意：仅限操作临时文件，实际使用时需要加强安全控制
    """
    try:
        # 安全检查：只允许操作temp目录下的文件
        import tempfile
        temp_dir = tempfile.gettempdir()
        safe_filename = os.path.join(temp_dir, f"agent_temp_{filename}")
        
        if operation == "create":
            if content is None:
                content = ""
            with open(safe_filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"文件已创建：{filename}"
        
        elif operation == "read":
            if os.path.exists(safe_filename):
                with open(safe_filename, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                return f"文件内容：\\n{file_content}"
            else:
                return f"文件不存在：{filename}"
        
        elif operation == "write":
            if content is None:
                return "写入操作需要提供内容"
            with open(safe_filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"内容已写入文件：{filename}"
        
        elif operation == "list":
            temp_files = [f for f in os.listdir(temp_dir) if f.startswith("agent_temp_")]
            if temp_files:
                return f"临时文件列表：{', '.join([f.replace('agent_temp_', '') for f in temp_files])}"
            else:
                return "暂无临时文件"
        
        else:
            return f"不支持的操作：{operation}"
            
    except Exception as e:
        return f"文件操作失败: {str(e)}"

class WebSearchInput(BaseModel):
    """网络搜索输入模型"""
    query: str = Field(description="搜索查询")
    num_results: int = Field(default=3, description="返回结果数量", ge=1, le=10)

@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str, num_results: int = 3) -> str:
    """
    网络搜索工具（模拟）
    注意：这是一个模拟的搜索功能，实际使用时应该集成真实的搜索API
    """
    try:
        # 模拟搜索结果
        mock_results = [
            {
                "title": f"关于'{query}'的综合介绍",
                "url": f"https://example.com/article1?q={query}",
                "snippet": f"这是关于{query}的详细介绍和分析，包含了最新的信息和观点..."
            },
            {
                "title": f"{query}的实践应用指南",
                "url": f"https://example.com/guide?q={query}",
                "snippet": f"本文详细讲解了{query}的实际应用方法和最佳实践..."
            },
            {
                "title": f"{query}相关新闻和动态",
                "url": f"https://news.example.com/news?q={query}",
                "snippet": f"最新关于{query}的新闻报道和行业动态..."
            }
        ]
        
        results = []
        for i, result in enumerate(mock_results[:num_results], 1):
            results.append(f"{i}. {result['title']}\\n   网址: {result['url']}\\n   摘要: {result['snippet']}")
        
        return f"搜索结果（{query}）：\\n" + "\\n\\n".join(results)
        
    except Exception as e:
        return f"搜索失败: {str(e)}"

# =========================== Agent构建 ===========================

def create_intelligent_agent():
    """创建智能助手Agent"""
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("请设置 OPENAI_API_KEY 环境变量")
    
    print("🤖 创建智能助手Agent...")
    
    # 创建LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        streaming=False
    )
    
    # 工具列表
    tools = [
        calculator,
        get_weather,
        analyze_text,
        file_manager,
        web_search
    ]
    
    # 创建Agent Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能助手Agent，可以使用多种工具来帮助用户完成任务。

你有以下工具可以使用：
- calculator: 数学计算器，支持各种数学运算
- weather: 天气查询工具
- text_analyzer: 文本分析工具（情感分析、关键词提取、摘要）
- file_manager: 文件管理工具
- web_search: 网络搜索工具

使用工具的指导原则：
1. 仔细理解用户的需求
2. 选择合适的工具来完成任务
3. 如果需要多个步骤，合理规划执行顺序
4. 对工具返回的结果进行总结和解释
5. 如果工具调用失败，尝试其他方法或向用户说明

请始终以友好、专业的方式与用户交互。"""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # 创建工具调用Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 创建Agent执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,  # 最大迭代次数
        handle_parsing_errors=True  # 处理解析错误
    )
    
    return agent_executor

def demo_basic_tool_calling():
    """演示基本工具调用"""
    print("🔧 基本工具调用演示")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 请设置 OPENAI_API_KEY 环境变量")
        return
    
    # 创建简单的工具调用示例
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # 绑定工具到模型
    llm_with_tools = llm.bind_tools([calculator, get_weather])
    
    # 测试工具调用
    print("1. 测试数学计算:")
    message = HumanMessage(content="计算 25 * 4 + sqrt(144)")
    
    ai_msg = llm_with_tools.invoke([message])
    print(f"   AI响应: {ai_msg.content}")
    
    # 检查AI消息是否包含工具调用信息
    tool_calls = getattr(ai_msg, "additional_kwargs", {}).get("tool_calls", None)
    if tool_calls:
        print(f"   工具调用: {len(tool_calls)} 个")
        for i, tool_call in enumerate(tool_calls, 1):
            print(f"     {i}. 工具: {tool_call['name']}")
            print(f"        参数: {tool_call['args']}")
            
            # 执行工具调用
            if tool_call['name'] == 'calculator':
                result = calculator.invoke(tool_call['args'])
                print(f"        结果: {result}")

def demo_multi_step_reasoning():
    """演示多步推理"""
    print("\n🧠 多步推理演示")
    print("=" * 50)
    
    agent = create_intelligent_agent()
    
    # 复杂任务示例
    complex_tasks = [
        "帮我计算一下，如果我每天走10000步，一年能走多少公里？（假设平均步长0.7米）",
        "分析这段文本的情感，然后将结果保存到文件中：'今天天气真不错，心情也很好，工作很顺利！'",
        "搜索关于人工智能的信息，然后总结关键点并计算如果AI发展速度每年增长20%，5年后会是现在的多少倍？"
    ]
    
    for i, task in enumerate(complex_tasks, 1):
        print(f"\n📋 任务 {i}: {task}")
        print("-" * 40)
        
        try:
            result = agent.invoke({"input": task})
            print(f"🤖 完成结果: {result['output']}")
        except Exception as e:
            print(f"❌ 任务失败: {e}")

def demo_conversation_with_memory():
    """演示带记忆的对话"""
    print("\n💭 对话记忆演示")
    print("=" * 50)
    
    agent = create_intelligent_agent()
    
    # 模拟多轮对话
    conversation_history = []
    
    conversations = [
        "你好，我是张三，请帮我计算一下 15 * 23",
        "请帮我查询北京的天气",
        "把刚才的计算结果和天气信息保存到一个文件中，文件名叫做'今日信息.txt'",
        "请读取刚才保存的文件内容"
    ]
    
    for i, user_input in enumerate(conversations, 1):
        print(f"\n👤 用户 {i}: {user_input}")
        
        try:
            # 将对话历史传递给Agent
            result = agent.invoke({
                "input": user_input,
                "chat_history": conversation_history
            })
            
            ai_response = result['output']
            print(f"🤖 助手: {ai_response}")
            
            # 更新对话历史
            conversation_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=ai_response)
            ])
            
        except Exception as e:
            print(f"❌ 对话失败: {e}")

def demo_parallel_tool_execution():
    """演示并行工具执行"""
    print("\n⚡ 并行工具执行演示")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 请设置 OPENAI_API_KEY 环境变量")
        return
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools([calculator, get_weather, analyze_text])
    
    # 需要并行处理的任务
    parallel_task = """
    请同时帮我做这几件事：
    1. 计算 100 * 365 
    2. 查询上海的天气
    3. 分析这段文本的情感："今天是个美好的日子"
    """
    
    print(f"📝 并行任务: {parallel_task}")
    print("-" * 40)
    
    messages = [HumanMessage(content=parallel_task)]
    ai_msg = llm_with_tools.invoke(messages)
    
    tool_calls = getattr(ai_msg, "additional_kwargs", {}).get("tool_calls", None)
    if tool_calls:
        print(f"🔧 同时调用 {len(tool_calls)} 个工具:")
        
        # 模拟并行执行工具调用
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            print(f"   - {tool_name}: {tool_args}")
            
            # 根据工具名执行相应工具
            if tool_name == 'calculator':
                result = calculator.invoke(tool_args)
            elif tool_name == 'weather':
                result = get_weather.invoke(tool_args)
            elif tool_name == 'text_analyzer':
                result = analyze_text.invoke(tool_args)
            else:
                result = f"未知工具: {tool_name}"
            
            results.append(f"{tool_name}: {result}")
        
        print("\n📊 执行结果:")
        for result in results:
            print(f"   {result}")

def demo_error_handling():
    """演示错误处理"""
    print("\n🛡️ 错误处理演示")
    print("=" * 50)
    
    # 测试各种错误情况
    error_cases = [
        ("calculator", {"expression": "1/0"}),  # 除零错误
        ("weather", {"city": "", "days": 1}),  # 空参数
        ("file_manager", {"operation": "read", "filename": "不存在的文件.txt"}),  # 文件不存在
        ("text_analyzer", {"text": "", "analysis_type": "unknown"})  # 不支持的分析类型
    ]
    
    for tool_name, args in error_cases:
        print(f"\n🧪 测试错误情况: {tool_name} with {args}")
        
        try:
            if tool_name == "calculator":
                result = calculator.invoke(args)
            elif tool_name == "weather":
                result = get_weather.invoke(args)
            elif tool_name == "file_manager":
                result = file_manager.invoke(args)
            elif tool_name == "text_analyzer":
                result = analyze_text.invoke(args)
            
            print(f"   结果: {result}")
            
        except Exception as e:
            print(f"   异常: {e}")

def main():
    """主函数"""
    try:
        print("🤖 LangChain Challenge 6: 智能Agent和工具集成")
        print("=" * 60)
        
        # 基本工具调用演示
        demo_basic_tool_calling()
        
        # 并行工具执行演示
        demo_parallel_tool_execution()
        
        # 错误处理演示
        demo_error_handling()
        
        print("\n" + "=" * 60)
        print("🧠 多步推理任务演示...")
        
        # 多步推理演示
        demo_multi_step_reasoning()
        
        print("\n" + "=" * 60)
        print("💭 对话记忆演示...")
        
        # 对话记忆演示
        demo_conversation_with_memory()
        
        print("\n" + "=" * 60)
        print("🎯 练习任务:")
        print("1. 创建更多自定义工具（数据库查询、API调用等）")
        print("2. 实现工具调用的重试和降级机制")
        print("3. 添加工具调用的权限控制和安全检查")
        print("4. 实现Agent的状态持久化")
        print("5. 创建专门的Agent工作流（如数据分析Agent）")
        print("6. 实现多Agent协作系统")
        print("7. 添加Agent性能监控和优化")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("\n请确保:")
        print("1. 已设置 OPENAI_API_KEY 环境变量")
        print("2. 已安装所需的依赖包: pip install langchain langchain-openai")

if __name__ == "__main__":
    main()
