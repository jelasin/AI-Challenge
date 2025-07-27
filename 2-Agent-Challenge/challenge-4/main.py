"""
Challenge 4: 检查点和状态持久化

学习目标:
- 掌握Checkpointer机制
- 实现状态持久化
- 学习故障恢复
- 管理长期记忆

核心概念:
1. MemorySaver/SqliteSaver - 检查点存储
2. 检查点配置 - 自动保存和恢复
3. 状态恢复 - 中断后继续执行
4. 持久化策略 - 不同存储方案
"""

import os
import sqlite3
import json
import time
from datetime import datetime
from typing import TypedDict, Annotated, Optional, cast, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
# 注意：如果SqliteSaver不可用，我们将使用替代方案
try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    SqliteSaver = None  # type: ignore
    print("⚠️  SqliteSaver 不可用，将使用 MemorySaver 替代")
    
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 检查环境变量
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  请设置 OPENAI_API_KEY 环境变量")
    exit(1)

# 1. 定义状态结构
class ConversationState(TypedDict):
    """持久化对话状态"""
    messages: Annotated[list, add_messages]
    user_profile: dict  # 用户档案
    conversation_history: list  # 对话历史摘要
    preferences: dict  # 用户偏好
    session_count: int  # 会话计数
    last_checkpoint: str  # 最后检查点时间

class TaskState(TypedDict):
    """长期任务状态"""
    task_id: str
    task_type: str  # research, analysis, writing
    current_step: int
    total_steps: int
    step_results: list
    status: str  # running, paused, completed, failed
    created_at: str
    updated_at: str

# 2. 初始化LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 3. 检查点存储器
def create_memory_saver():
    """创建内存检查点存储器"""
    return MemorySaver()

def create_sqlite_saver(db_path: str = "checkpoints.db"):
    """创建SQLite检查点存储器"""
    if SQLITE_AVAILABLE and SqliteSaver is not None:
        return SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
    else:
        print("⚠️  使用 MemorySaver 替代 SqliteSaver")
        return MemorySaver()

# 4. 用户档案管理节点
def profile_analysis_node(state: ConversationState) -> dict:
    """分析和更新用户档案"""
    print("👤 [档案分析] 更新用户档案...")
    
    last_message = state["messages"][-1] if state["messages"] else None
    if not last_message:
        return {}
    
    user_input = last_message.content
    current_profile = state.get("user_profile", {})
    
    # 使用LLM分析用户特征
    analysis_prompt = f"""基于用户输入，分析用户特征并更新档案:

当前档案: {json.dumps(current_profile, ensure_ascii=False, indent=2)}

用户输入: "{user_input}"

请更新以下字段(JSON格式):
- name: 姓名
- interests: 兴趣爱好列表
- communication_style: 沟通风格
- expertise_level: 专业水平
- preferred_topics: 偏好话题
- last_interaction: 最后互动时间

只返回JSON格式的更新档案。"""
    
    try:
        response = llm.invoke([{"role": "user", "content": analysis_prompt}])
        
        # 尝试解析LLM返回的JSON
        try:
            # 确保response.content是字符串类型
            content = response.content if isinstance(response.content, str) else str(response.content)
            updated_profile = json.loads(content)
        except json.JSONDecodeError:
            # 如果解析失败，保留原档案并添加基本信息
            updated_profile = current_profile.copy()
            updated_profile["last_interaction"] = datetime.now().isoformat()
            updated_profile["interaction_count"] = updated_profile.get("interaction_count", 0) + 1
        
        return {
            "user_profile": updated_profile,
            "last_checkpoint": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"   档案分析出错: {e}")
        return {}

def conversation_node(state: ConversationState) -> dict:
    """主对话节点 - 基于档案生成个性化回复"""
    print("💬 [对话处理] 生成个性化回复...")
    
    user_profile = state.get("user_profile", {})
    preferences = state.get("preferences", {})
    conversation_history = state.get("conversation_history", [])
    
    # 构建个性化上下文
    context = f"""你是一个智能助手，需要基于用户档案提供个性化回复。

用户档案:
{json.dumps(user_profile, ensure_ascii=False, indent=2)}

用户偏好:
{json.dumps(preferences, ensure_ascii=False, indent=2)}

对话历史摘要:
{'; '.join(conversation_history[-3:]) if conversation_history else '无'}

请根据用户的特点和偏好，生成自然、个性化的回复。"""
    
    # 准备对话消息
    messages = [{"role": "system", "content": context}]
    
    # 添加最近的对话历史
    for msg in state["messages"][-5:]:  # 最近5条消息
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            messages.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            messages.append({"role": "assistant", "content": content})
    
    try:
        response = llm.invoke(messages)
        
        # 更新对话历史摘要
        new_summary = f"用户询问: {state['messages'][-1].content[:50]}..."
        updated_history = conversation_history + [new_summary]
        
        # 保持历史记录在合理长度
        if len(updated_history) > 10:
            updated_history = updated_history[-10:]
        
        return {
            "messages": [AIMessage(content=response.content)],
            "conversation_history": updated_history,
            "session_count": state.get("session_count", 0) + 1,
            "last_checkpoint": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"   对话处理出错: {e}")
        error_message = "抱歉，我现在无法正常回复。请稍后再试。"
        return {
            "messages": [AIMessage(content=error_message)],
            "last_checkpoint": datetime.now().isoformat()
        }

def memory_consolidation_node(state: ConversationState) -> dict:
    """记忆巩固节点 - 整理和压缩长期记忆"""
    print("🧠 [记忆巩固] 整理长期记忆...")
    
    # 每10次交互进行一次记忆巩固
    session_count = state.get("session_count", 0)
    if session_count % 10 != 0:
        return {}
    
    conversation_history = state.get("conversation_history", [])
    user_profile = state.get("user_profile", {})
    
    if len(conversation_history) < 5:
        return {}
    
    consolidation_prompt = f"""整理以下对话历史，生成简洁的记忆摘要:

用户档案: {json.dumps(user_profile, ensure_ascii=False)}

对话历史:
{chr(10).join(conversation_history)}

请生成:
1. 关键对话主题摘要
2. 用户偏好更新建议
3. 重要信息提取

返回JSON格式:
{{
    "key_topics": ["主题1", "主题2"],
    "preference_updates": {{"偏好1": "值1"}},
    "important_facts": ["事实1", "事实2"]
}}"""
    
    try:
        response = llm.invoke([{"role": "user", "content": consolidation_prompt}])
        # 确保content是字符串类型
        content = response.content if isinstance(response.content, str) else str(response.content)
        consolidation_result = json.loads(content)
        
        # 更新偏好
        current_preferences = state.get("preferences", {})
        new_preferences = {
            **current_preferences,
            **consolidation_result.get("preference_updates", {})
        }
        
        # 压缩对话历史
        compressed_history = [
            f"主题摘要: {', '.join(consolidation_result.get('key_topics', []))}",
            f"重要事实: {', '.join(consolidation_result.get('important_facts', []))}"
        ]
        
        print("   记忆巩固完成")
        return {
            "conversation_history": compressed_history,
            "preferences": new_preferences,
            "last_checkpoint": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"   记忆巩固出错: {e}")
        return {}

# 5. 长期任务节点
def task_execution_node(state: TaskState) -> dict:
    """任务执行节点 - 可中断和恢复的长期任务"""
    print(f"🔄 [任务执行] 执行步骤 {state['current_step']}/{state['total_steps']}")
    
    current_step = state["current_step"]
    task_type = state["task_type"]
    
    # 模拟不同类型任务的步骤
    step_prompts = {
        "research": [
            "收集相关资料和数据源",
            "分析现有研究和文献",
            "总结关键发现和观点",
            "形成研究结论和建议"
        ],
        "analysis": [
            "数据收集和清理",
            "探索性数据分析",
            "深度统计分析",
            "结果解释和可视化"
        ],
        "writing": [
            "确定文章结构和大纲",
            "撰写第一稿内容",
            "修订和完善内容",
            "最终校对和格式化"
        ]
    }
    
    if task_type not in step_prompts:
        return {"status": "failed", "updated_at": datetime.now().isoformat()}
    
    steps = step_prompts[task_type]
    if current_step > len(steps):
        return {"status": "completed", "updated_at": datetime.now().isoformat()}
    
    current_task = steps[current_step - 1]
    print(f"   执行: {current_task}")
    
    # 模拟任务执行时间
    time.sleep(1)
    
    # 生成步骤结果
    step_result = f"步骤{current_step}完成: {current_task} - {datetime.now().strftime('%H:%M:%S')}"
    
    step_results = state.get("step_results", [])
    step_results.append(step_result)
    
    next_step = current_step + 1
    status = "completed" if next_step > len(steps) else "running"
    
    return {
        "current_step": next_step,
        "step_results": step_results,
        "status": status,
        "updated_at": datetime.now().isoformat()
    }

# 6. 构建持久化对话系统
def create_persistent_chat_system(checkpointer):
    """创建带检查点的对话系统"""
    
    workflow = StateGraph(ConversationState)
    
    # 添加节点
    workflow.add_node("profile_analysis", profile_analysis_node)
    workflow.add_node("conversation", conversation_node)
    workflow.add_node("memory_consolidation", memory_consolidation_node)
    
    # 设置流程
    workflow.add_edge(START, "profile_analysis")
    workflow.add_edge("profile_analysis", "conversation")
    workflow.add_edge("conversation", "memory_consolidation")
    workflow.add_edge("memory_consolidation", END)
    
    # 编译时添加检查点
    return workflow.compile(checkpointer=checkpointer)

def create_task_system(checkpointer):
    """创建长期任务系统"""
    
    workflow = StateGraph(TaskState)
    workflow.add_node("execute_step", task_execution_node)
    
    # 条件循环：继续或完成
    def should_continue(state: TaskState) -> str:
        status = state.get("status", "running")
        return "continue" if status == "running" else "end"
    
    workflow.add_edge(START, "execute_step")
    workflow.add_conditional_edges(
        "execute_step",
        should_continue,
        {
            "continue": "execute_step",
            "end": END
        }
    )
    
    return workflow.compile(checkpointer=checkpointer)

# 7. 演示函数
def demo_persistent_chat():
    """演示持久化对话"""
    print("=" * 60)
    print("💬 持久化对话系统演示")
    print("=" * 60)
    print("特性:")
    print("- 用户档案持久化")
    print("- 对话历史记忆")
    print("- 偏好学习和更新")
    print("- 会话中断和恢复")
    print("\n输入 'save' 查看检查点，'load' 恢复，'quit' 退出")
    print("-" * 60)
    
    # 创建检查点存储
    memory_saver = create_memory_saver()
    
    # 创建对话系统
    chat_system = create_persistent_chat_system(memory_saver)
    
    # 线程ID用于标识会话
    thread_id = "demo_session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {"configurable": {"thread_id": thread_id}}
    
    # 初始化状态
    initial_state = {
        "messages": [],
        "user_profile": {},
        "conversation_history": [],
        "preferences": {},
        "session_count": 0,
        "last_checkpoint": datetime.now().isoformat()
    }
    
    print(f"🆔 会话ID: {thread_id}")
    
    while True:
        user_input = input("\n👤 你: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 会话结束!")
            break
        elif user_input.lower() == 'save':
            # 显示当前检查点信息
            try:
                config_typed = cast(Any, config)
                checkpoint = memory_saver.get(config_typed)
                if checkpoint:
                    print("💾 当前检查点:")
                    state = checkpoint.get("channel_values", {})
                    print(f"   会话计数: {state.get('session_count', 0)}")
                    print(f"   用户档案: {json.dumps(state.get('user_profile', {}), ensure_ascii=False, indent=2)}")
                    print(f"   最后更新: {state.get('last_checkpoint', 'N/A')}")
                else:
                    print("📝 暂无检查点")
            except Exception as e:
                print(f"❌ 检查点查看失败: {e}")
            continue
        elif user_input.lower() == 'load':
            print("🔄 从检查点恢复会话...")
            continue
        elif not user_input:
            continue
        
        try:
            # 添加用户消息
            current_state = cast(ConversationState, {
                **initial_state,
                "messages": [HumanMessage(content=user_input)]
            })
            
            print("🔄 处理中...")
            
            # 执行对话流程
            config_typed = cast(Any, config)
            result = chat_system.invoke(current_state, config_typed)
            
            # 获取AI回复
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                print(f"\n🤖 助手: {ai_messages[-1].content}")
            
            # 显示状态更新
            print(f"\n📊 状态更新:")
            print(f"   会话计数: {result.get('session_count', 0)}")
            if result.get('user_profile'):
                print(f"   档案更新: ✅")
            if result.get('conversation_history'):
                print(f"   历史记录: {len(result.get('conversation_history', []))} 条")
            
            # 更新初始状态用于下次对话
            initial_state = result
            
        except Exception as e:
            print(f"❌ 处理错误: {e}")

def demo_task_recovery():
    """演示任务恢复功能"""
    print("\n" + "="*60)
    print("🔄 长期任务恢复演示")
    print("="*60)
    
    # 创建SQLite检查点存储
    sqlite_saver = create_sqlite_saver("demo_tasks.db")
    task_system = create_task_system(sqlite_saver)
    
    # 任务配置
    task_id = "demo_task_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {"configurable": {"thread_id": task_id}}
    
    print("选择任务类型:")
    print("1. research - 研究任务")
    print("2. analysis - 分析任务") 
    print("3. writing - 写作任务")
    
    choice = input("请选择 (1-3): ").strip()
    task_types = {"1": "research", "2": "analysis", "3": "writing"}
    task_type = task_types.get(choice, "research")
    
    # 初始任务状态
    initial_task: TaskState = {
        "task_id": task_id,
        "task_type": task_type,
        "current_step": 1,
        "total_steps": 4,
        "step_results": [],
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    print(f"\n🚀 启动 {task_type} 任务: {task_id}")
    print("按 Ctrl+C 可以中断任务，稍后恢复")
    print("-" * 40)
    
    try:
        # 执行任务
        config_typed = cast(Any, config)
        result = task_system.invoke(initial_task, config_typed)
        
        print(f"\n✅ 任务完成!")
        print(f"状态: {result.get('status')}")
        print(f"完成步骤: {result.get('current_step', 1) - 1}/{result.get('total_steps', 4)}")
        
        print("\n📋 执行记录:")
        for step_result in result.get("step_results", []):
            print(f"   {step_result}")
            
    except KeyboardInterrupt:
        print(f"\n⏸️  任务已中断，保存在检查点: {task_id}")
        print("可以稍后使用相同的thread_id恢复任务")
        
        # 显示如何恢复
        print("\n恢复命令示例:")
        print(f"task_system.invoke(None, {{'configurable': {{'thread_id': '{task_id}'}}}})")

if __name__ == "__main__":
    print("🚀 启动 Challenge 4: 检查点和状态持久化")
    
    print("\n选择演示:")
    print("1. 持久化对话系统")
    print("2. 长期任务恢复")
    
    choice = input("请选择 (1-2): ").strip()
    
    if choice == "2":
        demo_task_recovery()
    else:
        demo_persistent_chat()
