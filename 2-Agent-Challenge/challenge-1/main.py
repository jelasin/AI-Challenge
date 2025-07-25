"""
Challenge 1: 基础状态图Agent

学习目标:
- 理解LangGraph的基础概念
- 掌握StateGraph的创建和配置
- 学习基本节点定义和状态管理
- 构建简单的对话Agent

核心概念:
1. StateGraph - 状态图的核心组件
2. State - 图中传递的状态数据
3. Node - 执行特定任务的函数
4. Edge - 连接节点的路径
"""

import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 检查环境变量
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  请设置 OPENAI_API_KEY 环境变量")
    print("   export OPENAI_API_KEY='your-api-key'")
    exit(1)

# 1. 定义状态结构
class AgentState(TypedDict):
    """Agent的状态定义
    
    messages: 对话历史消息列表
    user_name: 用户名称
    conversation_count: 对话轮数
    """
    messages: Annotated[list, add_messages]
    user_name: str
    conversation_count: int

# 2. 初始化LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7
)

# 3. 定义节点函数
def greeting_node(state: AgentState) -> dict:
    """问候节点 - 处理初始问候"""
    print("🤖 [问候节点] 处理用户问候...")
    
    # 获取最新的用户消息
    last_message = state["messages"][-1]
    user_input = last_message.content
    
    # 检查是否包含姓名
    if "我是" in user_input or "我叫" in user_input:
        # 提取姓名(简单实现)
        parts = user_input.replace("我是", "").replace("我叫", "").strip()
        user_name = parts.split()[0] if parts else "朋友"
    else:
        user_name = state.get("user_name", "朋友")
    
    # 生成问候回复
    greeting = f"你好，{user_name}！我是你的AI助手，很高兴为你服务。有什么可以帮助你的吗？"
    
    return {
        "messages": [AIMessage(content=greeting)],
        "user_name": user_name,
        "conversation_count": state.get("conversation_count", 0) + 1
    }

def chat_node(state: AgentState) -> dict:
    """对话节点 - 处理普通对话"""
    print("🤖 [对话节点] 生成回复...")
    
    # 构建系统提示
    system_prompt = f"""你是一个友好的AI助手。
用户名: {state.get('user_name', '朋友')}
对话轮数: {state.get('conversation_count', 0)}

请根据对话历史，生成自然、有帮助的回复。保持友好和专业的语调。"""
    
    # 准备消息
    messages = [
        {"role": "system", "content": system_prompt}
    ] + [
        {"role": "human" if isinstance(msg, HumanMessage) else "assistant", 
         "content": msg.content}
        for msg in state["messages"]
    ]
    
    # 调用LLM生成回复
    response = llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=response.content)],
        "user_name": state.get("user_name", "朋友"),
        "conversation_count": state.get("conversation_count", 0) + 1
    }

def farewell_node(state: AgentState) -> dict:
    """告别节点 - 处理结束对话"""
    print("🤖 [告别节点] 处理告别...")
    
    user_name = state.get("user_name", "朋友")
    count = state.get("conversation_count", 0)
    
    farewell = f"再见，{user_name}！我们一共聊了{count}轮。希望我的回答对你有帮助。期待下次再聊！"
    
    return {
        "messages": [AIMessage(content=farewell)],
        "user_name": user_name,
        "conversation_count": count + 1
    }

# 4. 路由函数
def route_conversation(state: AgentState) -> str:
    """决定对话应该路由到哪个节点"""
    last_message = state["messages"][-1]
    user_input = last_message.content.lower()
    
    # 检查是否是问候
    if any(greeting in user_input for greeting in ["你好", "hello", "hi", "我是", "我叫"]):
        return "greeting"
    
    # 检查是否是告别
    if any(farewell in user_input for farewell in ["再见", "拜拜", "goodbye", "bye", "结束"]):
        return "farewell"
    
    # 默认进入对话节点
    return "chat"

# 5. 构建状态图
def create_agent():
    """创建并返回配置好的Agent"""
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("farewell", farewell_node)
    
    # 添加条件边 - 从START根据输入路由到不同节点
    workflow.add_conditional_edges(
        START,
        route_conversation,
        {
            "greeting": "greeting",
            "chat": "chat", 
            "farewell": "farewell"
        }
    )
    
    # 添加边 - 问候和对话后可以继续对话或结束
    workflow.add_conditional_edges(
        "greeting",
        lambda state: "continue",  # 问候后继续对话
        {"continue": "chat"}
    )
    
    workflow.add_conditional_edges(
        "chat", 
        lambda state: "end",  # 对话后等待用户输入
        {"end": END}
    )
    
    # 告别后结束
    workflow.add_edge("farewell", END)
    
    # 编译图
    app = workflow.compile()
    
    return app

# 6. 交互式对话函数
def run_interactive_chat():
    """运行交互式对话"""
    print("=" * 50)
    print("🤖 Challenge 1: 基础状态图Agent")
    print("=" * 50)
    print("这是一个基于LangGraph StateGraph的简单对话Agent")
    print("功能:")
    print("- 智能问候和姓名识别")
    print("- 基于上下文的对话")
    print("- 优雅的告别处理")
    print("- 对话轮数统计")
    print("\n输入 'quit' 退出程序")
    print("-" * 50)
    
    # 创建Agent
    agent = create_agent()
    
    # 初始状态
    current_state = {
        "messages": [],
        "user_name": "",
        "conversation_count": 0
    }
    
    while True:
        # 获取用户输入
        user_input = input("\n👤 你: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 程序结束!")
            break
        
        if not user_input:
            continue
        
        try:
            # 添加用户消息到状态
            current_state["messages"].append(HumanMessage(content=user_input))
            
            # 调用Agent处理
            print("\n🔄 Agent处理中...")
            result = agent.invoke(AgentState(**current_state))
            
            # 获取AI回复
            last_ai_message = result["messages"][-1]
            print(f"\n🤖 助手: {last_ai_message.content}")
            
            # 更新状态
            current_state["messages"] = result["messages"]
            current_state["user_name"] = result.get("user_name", "")
            current_state["conversation_count"] = result.get("conversation_count", 0)
            
            # 显示状态信息
            print(f"\n📊 状态信息:")
            print(f"   用户名: {result.get('user_name', '未知')}")
            print(f"   对话轮数: {result.get('conversation_count', 0)}")
            
        except Exception as e:
            print(f"❌ 处理出错: {e}")
            print("请检查API密钥和网络连接")

# 7. 演示函数
def demo_graph_structure():
    """演示图结构和可视化"""
    print("\n📈 Graph结构演示:")
    print("-" * 30)
    
    agent = create_agent()
    
    # 显示图的基本信息
    print("节点列表:")
    print("- START (起始节点)")
    print("- greeting (问候节点)")  
    print("- chat (对话节点)")
    print("- farewell (告别节点)")
    print("- END (结束节点)")
    
    print("\n边连接:")
    print("- START → [greeting|chat|farewell] (条件路由)")
    print("- greeting → chat")
    print("- chat → END")
    print("- farewell → END")
    
    print("\n状态流转:")
    print("用户输入 → 路由判断 → 执行节点 → 更新状态 → 返回结果")

if __name__ == "__main__":
    print("🚀 启动 Challenge 1: 基础状态图Agent")
    
    # 演示图结构
    demo_graph_structure()
    
    # 运行交互式对话
    run_interactive_chat()
