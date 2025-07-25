"""
Challenge 2: 条件路由和工具调用

学习目标:
- 掌握条件边(Conditional Edges)的使用
- 学习动态路由决策
- 实现工具集成和调用
- 处理错误和重试机制

核心概念:
1. add_conditional_edges() - 条件边添加
2. 路由函数设计 - 动态决策逻辑
3. 工具绑定和调用 - 函数工具集成
4. 状态更新策略 - 结果处理和状态管理
"""

import os
import json
import requests
from datetime import datetime
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# 检查环境变量
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  请设置 OPENAI_API_KEY 环境变量")
    exit(1)

# 1. 定义状态结构
class AgentState(TypedDict):
    """智能助手的状态定义"""
    messages: Annotated[list, add_messages]
    user_intent: str  # 用户意图: search, calculate, translate, weather, chat
    tool_calls_count: int  # 工具调用次数
    last_tool_result: str  # 最后的工具调用结果

# 2. 定义工具函数
@tool
def calculator(expression: str) -> str:
    """执行数学计算
    
    Args:
        expression: 数学表达式，如 "2+3*4"
    
    Returns:
        计算结果
    """
    try:
        # 安全的数学计算
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool  
def translator(text: str, target_language: str = "英文") -> str:
    """翻译文本
    
    Args:
        text: 要翻译的文本
        target_language: 目标语言，默认为英文
        
    Returns:
        翻译结果
    """
    # 模拟翻译功能
    translations = {
        "英文": {
            "你好": "Hello",
            "再见": "Goodbye", 
            "谢谢": "Thank you",
            "早上好": "Good morning"
        },
        "中文": {
            "hello": "你好",
            "goodbye": "再见",
            "thank you": "谢谢",
            "good morning": "早上好"
        }
    }
    
    text_lower = text.lower()
    if target_language in translations and text_lower in translations[target_language]:
        result = translations[target_language][text_lower]
        return f"翻译结果: {text} → {result} ({target_language})"
    else:
        return f"模拟翻译: '{text}' 翻译为{target_language}"

@tool
def web_search(query: str) -> str:
    """模拟网络搜索
    
    Args:
        query: 搜索查询
        
    Returns:
        搜索结果摘要
    """
    # 模拟搜索结果
    mock_results = {
        "python": "Python是一种高级编程语言，广泛用于Web开发、数据科学、人工智能等领域。",
        "ai": "人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "langgraph": "LangGraph是LangChain的一个扩展，用于构建有状态的、多步骤的AI应用程序。"
    }
    
    query_lower = query.lower()
    for key, value in mock_results.items():
        if key in query_lower:
            return f"搜索结果: {value}"
    
    return f"模拟搜索: 关于'{query}'的相关信息..."

@tool
def get_current_time() -> str:
    """获取当前时间
    
    Returns:
        当前日期和时间
    """
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"

# 3. 初始化LLM和工具
tools = [calculator, translator, web_search, get_current_time]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
llm_with_tools = llm.bind_tools(tools)

# 4. 意图识别函数
def analyze_intent(user_input: str) -> str:
    """分析用户意图"""
    user_input_lower = user_input.lower()
    
    # 计算意图
    if any(word in user_input_lower for word in ["计算", "算", "+", "-", "*", "/", "等于"]):
        return "calculate"
    
    # 翻译意图  
    if any(word in user_input_lower for word in ["翻译", "translate", "英文", "中文"]):
        return "translate"
    
    # 搜索意图
    if any(word in user_input_lower for word in ["搜索", "查找", "搜", "什么是", "介绍"]):
        return "search"
    
    # 时间意图
    if any(word in user_input_lower for word in ["时间", "现在", "几点", "日期"]):
        return "time"
    
    # 默认聊天意图
    return "chat"

# 5. 节点函数定义
def intent_analysis_node(state: AgentState) -> dict:
    """意图分析节点"""
    print("🧠 [意图分析] 分析用户意图...")
    
    last_message = state["messages"][-1]
    user_input = last_message.content
    
    intent = analyze_intent(user_input)
    print(f"   识别意图: {intent}")
    
    return {
        "user_intent": intent,
        "tool_calls_count": state.get("tool_calls_count", 0)
    }

def tool_calling_node(state: AgentState) -> dict:
    """工具调用节点"""
    print("🔧 [工具调用] 使用工具处理请求...")
    
    # 调用LLM来决定使用哪个工具
    response = llm_with_tools.invoke(state["messages"])
    
    tool_calls_count = state.get("tool_calls_count", 0) + 1
    
    # 如果LLM决定调用工具
    if response.tool_calls:
        tool_result = ""
        messages_to_add = [response]
        
        for tool_call in response.tool_calls:
            print(f"   调用工具: {tool_call['name']}")
            print(f"   参数: {tool_call['args']}")
            
            # 执行工具调用
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # 找到对应的工具并执行
            for tool in tools:
                if tool.name == tool_name:
                    result = tool.invoke(tool_args)
                    tool_result = result
                    
                    # 添加工具消息
                    messages_to_add.append(
                        ToolMessage(
                            content=result,
                            tool_call_id=tool_call["id"]
                        )
                    )
                    break
        
        return {
            "messages": messages_to_add,
            "tool_calls_count": tool_calls_count,
            "last_tool_result": tool_result
        }
    else:
        return {
            "messages": [response],
            "tool_calls_count": tool_calls_count,
            "last_tool_result": "未使用工具"
        }

def chat_node(state: AgentState) -> dict:
    """普通聊天节点"""
    print("💬 [聊天] 生成对话回复...")
    
    # 构建上下文
    context = f"""你是一个友好的AI助手。
当前对话轮数: {state.get('tool_calls_count', 0)}
最近工具结果: {state.get('last_tool_result', '无')}

请生成自然、有帮助的回复。"""
    
    messages = [
        {"role": "system", "content": context}
    ] + [
        {"role": "human" if isinstance(msg, HumanMessage) else "assistant", 
         "content": msg.content}
        for msg in state["messages"]
        if not isinstance(msg, ToolMessage)  # 过滤工具消息
    ]
    
    response = llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=response.content)],
        "tool_calls_count": state.get("tool_calls_count", 0) + 1
    }

def final_response_node(state: AgentState) -> dict:
    """最终回复整合节点"""
    print("✨ [最终回复] 整合工具结果...")
    
    # 如果有工具结果，生成包含工具结果的回复
    if state.get("last_tool_result") and state.get("last_tool_result") != "未使用工具":
        prompt = f"""根据工具执行结果，生成一个自然的回复给用户。

工具结果: {state['last_tool_result']}
对话历史: {[msg.content for msg in state['messages'] if isinstance(msg, (HumanMessage, AIMessage))]}

请生成一个友好、有帮助的回复，自然地整合工具结果。"""
        
        response = llm.invoke([{"role": "user", "content": prompt}])
        
        return {
            "messages": [AIMessage(content=response.content)]
        }
    else:
        # 没有工具结果，直接返回最后的AI消息
        return {}

# 6. 路由函数
def route_by_intent(state: AgentState) -> Literal["tool_calling", "chat"]:
    """根据意图路由到不同节点"""
    intent = state.get("user_intent", "chat")
    
    if intent in ["calculate", "translate", "search", "time"]:
        return "tool_calling"
    else:
        return "chat"

def should_use_final_response(state: AgentState) -> Literal["final_response", "end"]:
    """判断是否需要最终回复节点"""
    if state.get("last_tool_result") and state.get("last_tool_result") != "未使用工具":
        return "final_response"
    else:
        return "end"

# 7. 构建状态图
def create_smart_assistant():
    """创建智能助手"""
    
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("intent_analysis", intent_analysis_node)
    workflow.add_node("tool_calling", tool_calling_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("final_response", final_response_node)
    
    # 设置入口点
    workflow.add_edge(START, "intent_analysis")
    
    # 根据意图路由
    workflow.add_conditional_edges(
        "intent_analysis",
        route_by_intent,
        {
            "tool_calling": "tool_calling",
            "chat": "chat"
        }
    )
    
    # 工具调用后的路由
    workflow.add_conditional_edges(
        "tool_calling",
        should_use_final_response,
        {
            "final_response": "final_response", 
            "end": END
        }
    )
    
    # 聊天和最终回复都结束
    workflow.add_edge("chat", END)
    workflow.add_edge("final_response", END)
    
    return workflow.compile()

# 8. 交互式演示
def run_smart_assistant():
    """运行智能助手演示"""
    print("=" * 60)
    print("🤖 Challenge 2: 条件路由和工具调用智能助手")
    print("=" * 60)
    print("功能演示:")
    print("🧮 计算: '计算 2+3*4'")
    print("🌐 翻译: '翻译 hello 为中文'")
    print("🔍 搜索: '搜索 Python 编程'")
    print("🕐 时间: '现在几点了'")
    print("💬 聊天: '你好，今天天气怎么样'")
    print("\n输入 'quit' 退出")
    print("-" * 60)
    
    assistant = create_smart_assistant()
    
    while True:
        user_input = input("\n👤 你: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 再见!")
            break
            
        if not user_input:
            continue
            
        try:
            # 构建初始状态
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "user_intent": "",
                "tool_calls_count": 0,
                "last_tool_result": ""
            }
            
            print(f"\n🔄 处理请求: {user_input}")
            print("-" * 40)
            
            # 执行工作流
            result = assistant.invoke(initial_state)
            
            # 获取最终回复
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                final_response = ai_messages[-1].content
                print(f"\n🤖 助手: {final_response}")
            
            # 显示执行统计
            print(f"\n📊 执行统计:")
            print(f"   识别意图: {result.get('user_intent', '未知')}")
            print(f"   工具调用次数: {result.get('tool_calls_count', 0)}")
            print(f"   最后工具结果: {result.get('last_tool_result', '无')[:50]}...")
            
        except Exception as e:
            print(f"❌ 处理错误: {e}")

def demo_routing_logic():
    """演示路由逻辑"""
    print("\n📈 路由逻辑演示:")
    print("-" * 30)
    
    test_inputs = [
        "计算 5+3",
        "翻译 hello",
        "搜索 AI",
        "现在几点",
        "你好吗"
    ]
    
    for input_text in test_inputs:
        intent = analyze_intent(input_text)
        route = "tool_calling" if intent in ["calculate", "translate", "search", "time"] else "chat"
        print(f"输入: '{input_text}' → 意图: {intent} → 路由: {route}")

if __name__ == "__main__":
    print("🚀 启动 Challenge 2: 条件路由和工具调用")
    
    # 演示路由逻辑
    demo_routing_logic()
    
    # 运行智能助手
    run_smart_assistant()
