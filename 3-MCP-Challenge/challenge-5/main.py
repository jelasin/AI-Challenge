# -*- coding: utf-8 -*-
"""
Challenge 5: LangGraph与MCP集成

学习目标:
1. 掌握在LangGraph状态图中集成MCP工具
2. 学习MCP工具作为图节点的实现方式
3. 实现动态工具选择和智能路由
4. 理解状态管理和工具执行的协调

核心概念:
- StateGraph + MCP Integration: 状态图与MCP工具集成
- Dynamic Tool Routing: 动态工具路由
- State-based Tool Selection: 基于状态的工具选择
- Async Tool Execution: 异步工具执行
- Error Recovery: 错误恢复机制

实战场景:
构建一个智能Agent系统，使用LangGraph编排MCP工具执行复杂的
多步骤任务，实现自动化的工作流处理和智能决策。
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, TypedDict, Annotated, Union
from datetime import datetime
from enum import Enum

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import BaseTool
    
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    
    from pydantic import BaseModel, Field
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装必要的包:")
    print("pip install langchain-mcp-adapters langchain-openai langgraph")
    sys.exit(1)

class TaskType(Enum):
    """任务类型枚举"""
    CALCULATION = "calculation"
    FILE_OPERATION = "file_operation"
    DATA_ANALYSIS = "data_analysis"
    CONTENT_GENERATION = "content_generation"
    MIXED = "mixed"

class WorkflowState(TypedDict):
    """工作流状态定义"""
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], add_messages]
    current_task: Optional[str]
    task_type: Optional[str]
    intermediate_results: Dict[str, Any]
    tools_used: List[str]
    error_count: int
    max_retries: int
    workflow_status: str
    user_input: Optional[str]

class MCPAgentWorkflow:
    """MCP Agent工作流系统"""
    
    def __init__(self):
        """初始化工作流系统"""
        # MCP客户端配置
        self.server_configs = {
            "math": {
                "command": "python",
                "args": [str(project_root / "mcp_servers" / "math_server.py")],
                "transport": "stdio"
            },
            "file": {
                "command": "python",
                "args": [str(project_root / "mcp_servers" / "file_server.py")],
                "transport": "stdio"
            }
        }
        
        # 组件初始化
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.llm: Optional[ChatOpenAI] = None
        self.available_tools: List[BaseTool] = []
        self.workflow_graph: Optional[Any] = None  # 使用Any来避免复杂的泛型类型问题
        
        # 工作流统计
        self.execution_stats = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "tools_invoked": 0,
            "avg_execution_time": 0
        }
    
    async def initialize(self) -> bool:
        """初始化Agent工作流系统"""
        print("🔧 初始化MCP Agent工作流系统...")
        
        try:
            # 初始化MCP客户端
            self.mcp_client = MultiServerMCPClient(self.server_configs)  # type: ignore
            
            # 加载MCP工具
            print("📡 加载MCP工具...")
            self.available_tools = await self.mcp_client.get_tools()
            print(f"✅ 加载了 {len(self.available_tools)} 个MCP工具")
            
            # 初始化LLM
            if os.getenv("OPENAI_API_KEY"):
                self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
                print("✅ LLM初始化完成")
            else:
                print("⚠️  未设置OPENAI_API_KEY，将使用模拟响应")
            
            # 构建工作流图
            await self.build_workflow_graph()
            
            print("✅ Agent工作流系统初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False
    
    async def build_workflow_graph(self):
        """构建LangGraph工作流"""
        print("🏗️  构建Agent工作流图...")
        
        # 创建状态图
        workflow = StateGraph(WorkflowState)
        
        # 添加节点
        workflow.add_node("analyze_task", self.analyze_task_node)
        workflow.add_node("select_tools", self.select_tools_node)
        workflow.add_node("execute_tools", self.execute_tools_node)
        workflow.add_node("process_results", self.process_results_node)
        workflow.add_node("handle_error", self.handle_error_node)
        workflow.add_node("finalize", self.finalize_node)
        
        # 定义边和路由
        workflow.set_entry_point("analyze_task")
        
        workflow.add_edge("analyze_task", "select_tools")
        workflow.add_edge("select_tools", "execute_tools")
        
        # 条件路由
        workflow.add_conditional_edges(
            "execute_tools",
            self.route_after_execution,
            {
                "success": "process_results",
                "error": "handle_error",
                "retry": "select_tools"
            }
        )
        
        workflow.add_conditional_edges(
            "handle_error",
            self.route_after_error,
            {
                "retry": "select_tools",
                "fail": "finalize"
            }
        )
        
        workflow.add_edge("process_results", "finalize")
        workflow.add_edge("finalize", END)
        
        # 编译工作流
        self.workflow_graph = workflow.compile()
        print("✅ 工作流图构建完成")
    
    async def analyze_task_node(self, state: WorkflowState) -> WorkflowState:
        """分析任务节点"""
        print("🔍 分析任务...")
        
        # 获取用户输入
        if state["messages"]:
            user_message = state["messages"][-1]
            task_content = user_message.content if hasattr(user_message, 'content') else str(user_message)
        else:
            task_content = state.get("user_input", "")
        
        # 确保task_content是字符串
        if not isinstance(task_content, str):
            task_content = str(task_content) if task_content else ""
        
        # 简单的任务类型识别
        task_type = self.classify_task(task_content)
        
        # 更新状态
        state["current_task"] = task_content
        state["task_type"] = task_type.value
        state["workflow_status"] = "analyzing"
        
        # 添加分析消息
        analysis_message = AIMessage(
            content=f"任务分析完成。任务类型: {task_type.value}，内容: {task_content[:100]}..."
        )
        state["messages"].append(analysis_message)
        
        print(f"📊 任务类型: {task_type.value}")
        return state
    
    def classify_task(self, task_content: str) -> TaskType:
        """分类任务类型"""
        content_lower = task_content.lower()
        
        # 简单的关键词匹配分类
        if any(word in content_lower for word in ["计算", "数学", "加", "减", "乘", "除"]):
            return TaskType.CALCULATION
        elif any(word in content_lower for word in ["文件", "保存", "读取", "目录"]):
            return TaskType.FILE_OPERATION
        elif any(word in content_lower for word in ["分析", "统计", "数据"]):
            return TaskType.DATA_ANALYSIS
        elif any(word in content_lower for word in ["生成", "创建", "写"]):
            return TaskType.CONTENT_GENERATION
        else:
            return TaskType.MIXED
    
    async def select_tools_node(self, state: WorkflowState) -> WorkflowState:
        """选择工具节点"""
        print("🔧 选择合适的工具...")
        
        task_type = state.get("task_type", "mixed") or "mixed"
        current_task = state.get("current_task", "") or ""
        
        # 根据任务类型选择工具
        selected_tools = self.select_tools_for_task(task_type, current_task)
        
        state["workflow_status"] = "tool_selection"
        
        # 记录选择的工具
        tool_names = [tool.name for tool in selected_tools]
        selection_message = AIMessage(
            content=f"已选择工具: {', '.join(tool_names)}"
        )
        state["messages"].append(selection_message)
        
        # 将工具信息存储在中间结果中
        state["intermediate_results"]["selected_tools"] = selected_tools
        
        print(f"🎯 选择的工具: {tool_names}")
        return state
    
    def select_tools_for_task(self, task_type: str, task_content: str) -> List[BaseTool]:
        """根据任务类型和内容选择工具"""
        selected_tools = []
        
        # 数学相关工具
        math_tools = ["add", "subtract", "multiply", "divide", "power", "sqrt", "factorial"]
        
        # 文件相关工具
        file_tools = ["read_file", "write_file", "list_directory", "create_directory"]
        
        if task_type == TaskType.CALCULATION.value:
            # 选择数学工具
            for tool in self.available_tools:
                if tool.name in math_tools:
                    selected_tools.append(tool)
        
        elif task_type == TaskType.FILE_OPERATION.value:
            # 选择文件工具
            for tool in self.available_tools:
                if tool.name in file_tools:
                    selected_tools.append(tool)
        
        else:
            # 混合任务，选择所有相关工具
            for tool in self.available_tools:
                if tool.name in math_tools or tool.name in file_tools:
                    selected_tools.append(tool)
        
        return selected_tools[:5]  # 限制工具数量
    
    async def execute_tools_node(self, state: WorkflowState) -> WorkflowState:
        """执行工具节点"""
        print("⚡ 执行工具...")
        
        selected_tools = state["intermediate_results"].get("selected_tools", [])
        current_task = state.get("current_task", "") or ""
        
        if not selected_tools:
            state["workflow_status"] = "error"
            state["intermediate_results"]["error"] = "没有可用的工具"
            return state
        
        # 模拟智能工具调用
        execution_results = []
        
        try:
            if state.get("task_type") == TaskType.CALCULATION.value:
                # 执行数学计算
                results = await self.execute_math_task(selected_tools, current_task)
                execution_results.extend(results)
            
            elif state.get("task_type") == TaskType.FILE_OPERATION.value:
                # 执行文件操作
                results = await self.execute_file_task(selected_tools, current_task)
                execution_results.extend(results)
            
            else:
                # 执行混合任务
                results = await self.execute_mixed_task(selected_tools, current_task)
                execution_results.extend(results)
            
            state["intermediate_results"]["execution_results"] = execution_results
            state["tools_used"].extend([tool.name for tool in selected_tools])
            state["workflow_status"] = "success"
            
            # 更新统计
            self.execution_stats["tools_invoked"] += len(selected_tools)
            
        except Exception as e:
            print(f"❌ 工具执行失败: {e}")
            state["workflow_status"] = "error"
            state["intermediate_results"]["error"] = str(e)
            state["error_count"] += 1
        
        return state
    
    async def execute_math_task(self, tools: List[BaseTool], task: str) -> List[Dict[str, Any]]:
        """执行数学任务"""
        results = []
        
        # 简单的数学表达式解析和执行
        if "+" in task:
            # 查找加法工具
            add_tool = next((t for t in tools if t.name == "add"), None)
            if add_tool:
                try:
                    # 简单解析
                    parts = task.split("+")
                    if len(parts) >= 2:
                        a = float(parts[0].strip().split()[-1])
                        b = float(parts[1].strip().split()[0])
                        result = await add_tool.ainvoke({"a": a, "b": b})
                        results.append({"tool": "add", "args": {"a": a, "b": b}, "result": result})
                except Exception as e:
                    results.append({"tool": "add", "error": str(e)})
        
        return results
    
    async def execute_file_task(self, tools: List[BaseTool], task: str) -> List[Dict[str, Any]]:
        """执行文件任务"""
        results = []
        
        # 简单的文件操作
        if "保存" in task or "写入" in task:
            write_tool = next((t for t in tools if t.name == "write_file"), None)
            if write_tool:
                try:
                    result = await write_tool.ainvoke({
                        "file_path": "task_result.txt",
                        "content": f"任务执行结果：{task}"
                    })
                    results.append({"tool": "write_file", "result": result})
                except Exception as e:
                    results.append({"tool": "write_file", "error": str(e)})
        
        return results
    
    async def execute_mixed_task(self, tools: List[BaseTool], task: str) -> List[Dict[str, Any]]:
        """执行混合任务"""
        results = []
        
        # 尝试执行多种操作
        math_results = await self.execute_math_task(tools, task)
        file_results = await self.execute_file_task(tools, task)
        
        results.extend(math_results)
        results.extend(file_results)
        
        return results
    
    def route_after_execution(self, state: WorkflowState) -> str:
        """执行后路由决策"""
        status = state.get("workflow_status", "error")
        
        if status == "success":
            return "success"
        elif state.get("error_count", 0) < state.get("max_retries", 3):
            return "retry"
        else:
            return "error"
    
    async def process_results_node(self, state: WorkflowState) -> WorkflowState:
        """处理结果节点"""
        print("📊 处理执行结果...")
        
        execution_results = state["intermediate_results"].get("execution_results", [])
        
        # 汇总结果
        summary = self.summarize_results(execution_results)
        
        state["intermediate_results"]["final_summary"] = summary
        state["workflow_status"] = "processed"
        
        # 添加结果消息
        result_message = AIMessage(content=f"任务执行完成。{summary}")
        state["messages"].append(result_message)
        
        print(f"✅ 结果汇总: {summary}")
        return state
    
    def summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """汇总执行结果"""
        if not results:
            return "没有执行结果"
        
        success_count = len([r for r in results if "error" not in r])
        error_count = len([r for r in results if "error" in r])
        
        summary = f"成功执行 {success_count} 个操作"
        if error_count > 0:
            summary += f"，{error_count} 个操作失败"
        
        return summary
    
    async def handle_error_node(self, state: WorkflowState) -> WorkflowState:
        """错误处理节点"""
        print("🚨 处理执行错误...")
        
        error_info = state["intermediate_results"].get("error", "未知错误")
        error_count = state.get("error_count", 0)
        max_retries = state.get("max_retries", 3)
        
        state["workflow_status"] = "error_handling"
        
        # 错误恢复策略
        if error_count < max_retries:
            print(f"🔄 准备重试 ({error_count + 1}/{max_retries})")
            error_message = AIMessage(content=f"遇到错误: {error_info}。正在重试...")
        else:
            print("❌ 超过最大重试次数，任务失败")
            error_message = AIMessage(content=f"任务失败: {error_info}")
        
        state["messages"].append(error_message)
        return state
    
    def route_after_error(self, state: WorkflowState) -> str:
        """错误后路由决策"""
        error_count = state.get("error_count", 0)
        max_retries = state.get("max_retries", 3)
        
        if error_count < max_retries:
            return "retry"
        else:
            return "fail"
    
    async def finalize_node(self, state: WorkflowState) -> WorkflowState:
        """终结节点"""
        print("🏁 完成工作流...")
        
        # 更新统计信息
        self.execution_stats["total_workflows"] += 1
        
        if state.get("workflow_status") in ["processed", "success"]:
            self.execution_stats["successful_workflows"] += 1
        else:
            self.execution_stats["failed_workflows"] += 1
        
        state["workflow_status"] = "completed"
        
        # 添加完成消息
        final_message = AIMessage(content="工作流执行完成")
        state["messages"].append(final_message)
        
        return state
    
    async def run_workflow(self, user_input: str) -> Dict[str, Any]:
        """运行工作流"""
        if not self.workflow_graph:
            raise ValueError("工作流图未初始化")
        
        # 初始化状态
        initial_state: WorkflowState = {
            "messages": [HumanMessage(content=user_input)],
            "current_task": None,
            "task_type": None,
            "intermediate_results": {},
            "tools_used": [],
            "error_count": 0,
            "max_retries": 3,
            "workflow_status": "started",
            "user_input": user_input
        }
        
        print(f"🚀 启动工作流: {user_input}")
        
        # 执行工作流
        try:
            final_state = await self.workflow_graph.ainvoke(initial_state)  # type: ignore
            return final_state
        except Exception as e:
            print(f"❌ 工作流执行失败: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def demonstrate_workflow_execution(self):
        """演示工作流执行"""
        print("\n" + "="*60)
        print("⚡ Agent工作流执行演示")
        print("="*60)
        
        # 测试用例
        test_cases = [
            "计算 15 + 25 的结果",
            "将计算结果保存到文件",
            "创建一个新目录并写入文件",
            "计算 2 的 10 次方并保存结果"
        ]
        
        for i, task in enumerate(test_cases, 1):
            print(f"\n🎯 测试任务 {i}: {task}")
            print("-" * 40)
            
            try:
                result = await self.run_workflow(task)
                
                if "error" in result:
                    print(f"❌ 任务失败: {result['error']}")
                else:
                    print(f"✅ 任务完成")
                    print(f"📊 状态: {result.get('workflow_status', '未知')}")
                    print(f"🔧 使用工具: {result.get('tools_used', [])}")
                    
                    # 显示消息历史
                    messages = result.get("messages", [])
                    if messages:
                        print(f"💬 对话历史 ({len(messages)} 条消息):")
                        for msg in messages[-3:]:  # 显示最后3条消息
                            msg_type = type(msg).__name__
                            content = getattr(msg, 'content', str(msg))
                            print(f"   {msg_type}: {content[:100]}...")
                
            except Exception as e:
                print(f"❌ 执行异常: {e}")
            
            await asyncio.sleep(2)
    
    def show_execution_stats(self):
        """显示执行统计"""
        print("\n📊 工作流执行统计:")
        print("-" * 30)
        
        stats = self.execution_stats
        total = stats["total_workflows"]
        
        if total > 0:
            success_rate = (stats["successful_workflows"] / total) * 100
            print(f"总工作流数: {total}")
            print(f"成功: {stats['successful_workflows']} ({success_rate:.1f}%)")
            print(f"失败: {stats['failed_workflows']}")
            print(f"工具调用次数: {stats['tools_invoked']}")
            print(f"平均每工作流调用工具: {stats['tools_invoked'] / total:.1f} 个")
        else:
            print("暂无执行统计数据")

async def demo_langgraph_integration():
    """Challenge 5 主演示函数"""
    print("🚀 Challenge 5: LangGraph与MCP集成")
    print("="*60)
    
    # 创建Agent工作流系统
    workflow_system = MCPAgentWorkflow()
    
    # 初始化
    if not await workflow_system.initialize():
        print("❌ 无法初始化Agent工作流系统，演示结束")
        return
    
    try:
        # 1. 工作流执行演示
        await workflow_system.demonstrate_workflow_execution()
        
        # 2. 显示执行统计
        workflow_system.show_execution_stats()
        
        print("\n🎉 Challenge 5 演示完成！")
        print("\n📚 学习要点总结:")
        print("  ✅ 掌握了LangGraph与MCP工具的集成方式")
        print("  ✅ 学会了构建智能Agent工作流")
        print("  ✅ 实现了动态工具选择和路由机制")
        print("  ✅ 体验了状态管理和错误恢复功能")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")

def main():
    """直接运行此Challenge的主函数"""
    asyncio.run(demo_langgraph_integration())

if __name__ == "__main__":
    main()
