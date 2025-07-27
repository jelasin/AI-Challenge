"""
Challenge 7: 流式处理和实时Agent

学习目标:
- 掌握LangGraph的流式处理能力
- 学习实时状态更新和响应
- 实现异步Agent操作
- 构建事件驱动的Agent系统

核心概念:
1. stream() - 流式执行状态图
2. 实时状态更新 - 状态变化的实时传递
3. 异步处理 - 非阻塞的Agent操作
4. 事件流监控 - 实时监控Agent状态变化
"""

import os
import asyncio
import json
import time
from datetime import datetime
from typing import TypedDict, Annotated, AsyncIterator, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool

# 检查环境变量
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  请设置 OPENAI_API_KEY 环境变量")
    exit(1)

# 1. 定义状态结构
class StreamingAgentState(TypedDict):
    """流式Agent状态"""
    messages: Annotated[list, add_messages]
    current_task: str  # 当前执行的任务
    progress: float  # 任务进度 (0-1)
    status: str  # running, completed, error
    events: List[Dict[str, Any]]  # 事件日志
    stream_data: List[str]  # 流式数据缓冲区
    real_time_metrics: Dict[str, Any]  # 实时监控指标

# 2. 工具定义
@tool
def process_data_stream(data: str, batch_size: int = 10) -> str:
    """
    模拟流式数据处理
    
    Args:
        data: 要处理的数据
        batch_size: 批处理大小
    
    Returns:
        处理结果
    """
    # 模拟数据处理延迟
    time.sleep(0.1)
    
    processed_items = len(data.split()) // batch_size + 1
    return f"已处理 {processed_items} 个数据批次，总计 {len(data)} 个字符"

@tool
def real_time_monitor(metric_name: str, value: float) -> str:
    """
    实时监控指标记录
    
    Args:
        metric_name: 指标名称
        value: 指标值
    
    Returns:
        监控结果
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    return f"[{timestamp}] {metric_name}: {value:.2f}"

# 3. 初始化LLM和工具
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    streaming=True  # 启用流式输出
)

tools = [process_data_stream, real_time_monitor]
llm_with_tools = llm.bind_tools(tools)

# 4. 节点函数定义
def initialize_stream(state: StreamingAgentState) -> StreamingAgentState:
    """初始化流式处理"""
    print("🚀 初始化流式Agent系统...")
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 添加初始化事件
    event = {
        "timestamp": current_time,
        "type": "initialization",
        "message": "流式Agent系统启动"
    }
    
    return {
        **state,
        "current_task": "初始化中",
        "progress": 0.0,
        "status": "running",
        "events": [event],
        "stream_data": [],
        "real_time_metrics": {
            "start_time": current_time,
            "processed_items": 0,
            "error_count": 0
        }
    }

def stream_processor(state: StreamingAgentState) -> StreamingAgentState:
    """流式数据处理节点"""
    print("📊 开始流式数据处理...")
    
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    if not last_message or not isinstance(last_message, HumanMessage):
        return {
            **state,
            "status": "error",
            "current_task": "错误: 缺少用户输入"
        }
    
    user_input = last_message.content
    
    # 模拟流式处理过程
    stream_chunks = []
    total_chunks = 5
    
    for i in range(total_chunks):
        # 模拟处理延迟
        time.sleep(0.2)
        
        chunk = f"处理块 {i+1}/{total_chunks}: {user_input[:20]}..."
        stream_chunks.append(chunk)
        
        # 更新进度
        progress = (i + 1) / total_chunks
        
        # 实时事件记录
        event = {
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "type": "processing",
            "chunk_id": i + 1,
            "progress": progress,
            "data": chunk
        }
        
        print(f"  ⚡ [{event['timestamp']}] 进度: {progress*100:.1f}% - {chunk}")
    
    return {
        **state,
        "current_task": "流式处理",
        "progress": 1.0,
        "stream_data": stream_chunks,
        "real_time_metrics": {
            **state["real_time_metrics"],
            "processed_items": state["real_time_metrics"]["processed_items"] + len(stream_chunks)
        }
    }

def tool_caller_node(state: StreamingAgentState) -> StreamingAgentState:
    """工具调用节点"""
    print("🔧 执行工具调用...")
    
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    if not last_message:
        return {
            **state,
            "current_task": "工具执行",
            "events": state["events"] + [{
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "type": "tool_execution",
                "tools_called": 0,
                "results": []
            }]
        }
    
    # 基于消息内容模拟工具调用
    user_content = str(last_message.content) if hasattr(last_message, 'content') else ""
    
    tool_results = []
    
    # 模拟数据处理工具调用
    if "数据" in user_content or "处理" in user_content:
        result = process_data_stream.invoke({"data": user_content, "batch_size": 5})
        tool_results.append({
            "tool": "process_data_stream",
            "args": {"data": user_content, "batch_size": 5},
            "result": result
        })
        print(f"  🛠️  调用工具: process_data_stream")
        print(f"  📊 结果: {result}")
    
    # 模拟监控工具调用
    if "监控" in user_content or "性能" in user_content or "指标" in user_content:
        # 提取数值进行监控
        import re
        numbers = re.findall(r'\d+\.?\d*', user_content)
        if numbers:
            value = float(numbers[0])
            result = real_time_monitor.invoke({"metric_name": "系统指标", "value": value})
            tool_results.append({
                "tool": "real_time_monitor", 
                "args": {"metric_name": "系统指标", "value": value},
                "result": result
            })
            print(f"  🛠️  调用工具: real_time_monitor")
            print(f"  📊 结果: {result}")
    
    # 如果没有识别到特定工具需求，执行通用处理
    if not tool_results:
        result = f"已分析输入内容，检测到 {len(user_content)} 个字符的数据"
        tool_results.append({
            "tool": "通用分析",
            "args": {"content": user_content[:50] + "..."},
            "result": result
        })
        print(f"  🛠️  执行通用分析")
        print(f"  📊 结果: {result}")
    
    # 创建AI响应消息
    tool_summary = f"执行了 {len(tool_results)} 个工具调用:\n"
    for tr in tool_results:
        tool_summary += f"- {tr['tool']}: {tr['result']}\n"
    
    ai_response = AIMessage(content=tool_summary)
    
    # 记录工具调用事件
    event = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "type": "tool_execution",
        "tools_called": len(tool_results),
        "results": tool_results
    }
    
    return {
        **state,
        "messages": state["messages"] + [ai_response],
        "current_task": "工具执行",
        "events": state["events"] + [event]
    }

def real_time_monitor_node(state: StreamingAgentState) -> StreamingAgentState:
    """实时监控节点"""
    print("📊 更新实时监控指标...")
    
    # 计算实时指标
    current_time = datetime.now()
    start_time_str = state["real_time_metrics"]["start_time"]
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    
    elapsed_time = (current_time - start_time).total_seconds()
    
    metrics = {
        **state["real_time_metrics"],
        "elapsed_time": elapsed_time,
        "processing_rate": state["real_time_metrics"]["processed_items"] / max(elapsed_time, 1),
        "current_status": state["status"],
        "memory_usage": len(str(state)) / 1024,  # 模拟内存使用KB
        "last_update": current_time.strftime("%H:%M:%S")
    }
    
    # 打印实时指标
    print(f"  📈 处理速率: {metrics['processing_rate']:.2f} items/sec")
    print(f"  ⏱️  运行时间: {metrics['elapsed_time']:.1f}s")
    print(f"  💾 内存使用: {metrics['memory_usage']:.1f}KB")
    
    return {
        **state,
        "real_time_metrics": metrics,
        "current_task": "监控更新"
    }

def response_generator(state: StreamingAgentState) -> StreamingAgentState:
    """响应生成节点"""
    print("💬 生成流式响应...")
    
    # 收集处理结果
    events = state["events"]
    metrics = state["real_time_metrics"]
    stream_data = state["stream_data"]
    
    # 生成总结响应
    summary = f"""
🎯 **流式处理完成报告**

📊 **处理统计**:
- 总处理时间: {metrics.get('elapsed_time', 0):.1f}秒
- 处理项目数: {metrics.get('processed_items', 0)}
- 处理速率: {metrics.get('processing_rate', 0):.2f} items/sec
- 流式数据块: {len(stream_data)}

📈 **实时指标**:
- 内存使用: {metrics.get('memory_usage', 0):.1f}KB
- 当前状态: {state['status']}
- 最后更新: {metrics.get('last_update', 'N/A')}

📝 **事件日志**: {len(events)} 个事件已记录

✅ 所有流式处理任务已完成，系统运行正常。
"""
    
    ai_response = AIMessage(content=summary)
    
    return {
        **state,
        "messages": state["messages"] + [ai_response],
        "current_task": "已完成",
        "status": "completed"
    }

# 5. 路由函数
def should_continue(state: StreamingAgentState) -> str:
    """决定工作流是否继续"""
    if state["status"] == "error":
        return END
    elif state["progress"] < 1.0:
        return "stream_processor"
    else:
        return "tool_caller"

def after_tools(state: StreamingAgentState) -> str:
    """工具调用后的路由"""
    return "monitor"

# 6. 构建状态图
def build_streaming_graph():
    """构建流式处理状态图"""
    
    workflow = StateGraph(StreamingAgentState)
    
    # 添加节点
    workflow.add_node("initialize", initialize_stream)
    workflow.add_node("stream_processor", stream_processor)
    workflow.add_node("tool_caller", tool_caller_node)
    workflow.add_node("monitor", real_time_monitor_node)
    workflow.add_node("response", response_generator)
    
    # 添加边
    workflow.add_edge(START, "initialize")
    workflow.add_conditional_edges(
        "initialize",
        should_continue
    )
    workflow.add_conditional_edges(
        "stream_processor", 
        should_continue
    )
    workflow.add_conditional_edges(
        "tool_caller",
        after_tools
    )
    workflow.add_edge("monitor", "response")
    workflow.add_edge("response", END)
    
    return workflow.compile()

# 7. 流式处理演示
async def run_streaming_demo():
    """运行流式处理演示"""
    print("🌊 启动LangGraph流式处理演示")
    print("=" * 50)
    
    app = build_streaming_graph()
    
    # 测试用例
    test_cases = [
        "请处理这批销售数据: 北京100万, 上海150万, 广州80万, 深圳120万",
        "分析用户行为数据: 点击率3.2%, 转化率1.8%, 留存率65%",
        "监控系统性能: CPU使用率75%, 内存使用率60%, 网络延迟20ms"
    ]
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"\n🎯 测试案例 {i}: {user_input}")
        print("-" * 40)
        
        # 初始状态
        initial_state: StreamingAgentState = {
            "messages": [HumanMessage(content=user_input)],
            "current_task": "",
            "progress": 0.0,
            "status": "initializing",
            "events": [],
            "stream_data": [],
            "real_time_metrics": {}
        }
        
        # 使用stream方法获取实时更新
        print("🔄 开始流式执行...")
        
        try:
            # 流式执行状态图
            async for chunk in app.astream(initial_state):
                for node_name, node_output in chunk.items():
                    print(f"📍 节点 '{node_name}' 执行完成")
                    print(f"   当前任务: {node_output.get('current_task', 'N/A')}")
                    print(f"   进度: {node_output.get('progress', 0)*100:.1f}%")
                    print(f"   状态: {node_output.get('status', 'unknown')}")
                    
                    # 显示最新事件
                    events = node_output.get('events', [])
                    if events:
                        latest_event = events[-1]
                        print(f"   📝 最新事件: {latest_event.get('type', 'unknown')} - {latest_event.get('message', latest_event.get('data', 'N/A'))}")
                    
                    print()
                    
                    # 模拟实时处理延迟
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            print(f"❌ 流式处理出错: {e}")
        
        print(f"✅ 测试案例 {i} 完成\n")
        await asyncio.sleep(1)  # 案例间间隔

def run_interactive_demo():
    """运行交互式演示"""
    print("🌊 LangGraph流式处理交互演示")
    print("=" * 40)
    print("输入 'quit' 退出程序")
    print("输入 'stream' 查看流式效果")
    print()
    
    app = build_streaming_graph()
    
    while True:
        user_input = input("💬 请输入您的需求: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'stream':
            print("🚀 运行流式处理演示...")
            asyncio.run(run_streaming_demo())
            continue
        elif not user_input:
            continue
        
        print(f"\n🎯 处理请求: {user_input}")
        print("-" * 30)
        
        # 初始状态
        initial_state: StreamingAgentState = {
            "messages": [HumanMessage(content=user_input)],
            "current_task": "",
            "progress": 0.0,
            "status": "initializing", 
            "events": [],
            "stream_data": [],
            "real_time_metrics": {}
        }
        
        try:
            # 执行状态图
            result = app.invoke(initial_state)
            
            # 显示最终结果
            final_message = result["messages"][-1]
            if hasattr(final_message, 'content'):
                print("📋 处理结果:")
                print(final_message.content)
            
        except Exception as e:
            print(f"❌ 处理出错: {e}")
        
        print("\n" + "="*50 + "\n")

# 8. 主程序
if __name__ == "__main__":
    print("🌊 LangGraph Challenge 7: 流式处理和实时Agent")
    print("=" * 50)
    
    # 选择运行模式
    mode = input("选择运行模式:\n1. 交互模式\n2. 流式演示\n请输入选择 (1/2): ").strip()
    
    if mode == "2":
        print("\n🚀 启动流式演示模式...")
        asyncio.run(run_streaming_demo())
    else:
        print("\n🚀 启动交互模式...")
        run_interactive_demo()
    
    print("\n👋 感谢使用 LangGraph 流式处理系统!")
