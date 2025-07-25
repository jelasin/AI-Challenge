"""
Challenge 6: 高级记忆和多Agent系统

学习目标:
- 实现多Agent协作
- 构建高级记忆系统
- 管理复杂状态
- 系统级优化

核心概念:
1. 多Agent通信 - Agent间消息传递
2. 共享状态管理 - 全局状态协调
3. 长短期记忆 - 分层记忆系统
4. 系统级优化 - 性能和可扩展性
"""

import os
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 检查环境变量
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  请设置 OPENAI_API_KEY 环境变量")
    exit(1)

# 1. 定义状态结构
class TeamState(TypedDict):
    """团队协作状态"""
    messages: Annotated[list, add_messages]
    project_brief: dict  # 项目简介
    task_assignments: dict  # 任务分配
    agent_outputs: dict  # 各Agent的输出
    collaboration_history: list  # 协作历史
    shared_memory: dict  # 共享记忆
    current_phase: str  # 当前阶段
    completion_status: dict  # 完成状态

class AgentMemory(TypedDict):
    """Agent记忆结构"""
    agent_id: str
    short_term_memory: list  # 短期记忆 (最近对话)
    long_term_memory: dict  # 长期记忆 (知识库)
    working_memory: dict  # 工作记忆 (当前任务)
    episodic_memory: list  # 情节记忆 (重要事件)
    skill_memory: dict  # 技能记忆 (专业知识)

# 2. 初始化多个专业LLM
researcher_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
analyst_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)  
writer_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
reviewer_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

# 3. 记忆管理系统
class MemoryManager:
    """高级记忆管理器"""
    
    def __init__(self):
        self.agent_memories: Dict[str, AgentMemory] = {}
        self.global_memory = {
            "facts": {},
            "relationships": {},
            "patterns": {},
            "insights": []
        }
    
    def get_agent_memory(self, agent_id: str) -> AgentMemory:
        """获取Agent记忆"""
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = {
                "agent_id": agent_id,
                "short_term_memory": [],
                "long_term_memory": {},
                "working_memory": {},
                "episodic_memory": [],
                "skill_memory": {}
            }
        return self.agent_memories[agent_id]
    
    def update_short_term_memory(self, agent_id: str, content: str):
        """更新短期记忆"""
        memory = self.get_agent_memory(agent_id)
        memory["short_term_memory"].append({
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # 保持短期记忆在合理长度
        if len(memory["short_term_memory"]) > 10:
            memory["short_term_memory"] = memory["short_term_memory"][-10:]
    
    def add_to_long_term_memory(self, agent_id: str, key: str, value: Any):
        """添加到长期记忆"""
        memory = self.get_agent_memory(agent_id)
        memory["long_term_memory"][key] = {
            "value": value,
            "created_at": datetime.now().isoformat(),
            "access_count": 0
        }
    
    def retrieve_relevant_memory(self, agent_id: str, query: str) -> dict:
        """检索相关记忆"""
        memory = self.get_agent_memory(agent_id)
        
        # 简单的关键词匹配 (实际应用可以使用向量搜索)
        relevant = {
            "short_term": [],
            "long_term": {},
            "episodic": []
        }
        
        query_lower = query.lower()
        
        # 搜索短期记忆
        for item in memory["short_term_memory"]:
            if any(word in item["content"].lower() for word in query_lower.split()):
                relevant["short_term"].append(item)
        
        # 搜索长期记忆
        for key, value in memory["long_term_memory"].items():
            if query_lower in key.lower() or query_lower in str(value["value"]).lower():
                relevant["long_term"][key] = value
                value["access_count"] += 1
        
        return relevant

# 初始化记忆管理器
memory_manager = MemoryManager()

# 4. 专业Agent节点
def researcher_agent_node(state: TeamState) -> dict:
    """研究员Agent - 负责信息收集和研究"""
    print("🔬 [研究员] 进行信息收集和研究...")
    
    project_brief = state.get("project_brief", {})
    topic = project_brief.get("topic", "未知主题")
    
    # 检索相关记忆
    relevant_memory = memory_manager.retrieve_relevant_memory("researcher", topic)
    
    research_prompt = f"""作为专业研究员，请对以下主题进行深入研究:

主题: {topic}
项目简介: {json.dumps(project_brief, ensure_ascii=False, indent=2)}

相关记忆:
{json.dumps(relevant_memory, ensure_ascii=False, indent=2)}

请提供:
1. 背景调研和现状分析
2. 关键数据和统计信息
3. 重要趋势和发展方向
4. 相关案例和最佳实践
5. 潜在挑战和机遇

输出格式: 详细的研究报告"""
    
    try:
        response = researcher_llm.invoke([{"role": "user", "content": research_prompt}])
        research_output = response.content
        
        # 更新记忆
        memory_manager.update_short_term_memory("researcher", f"研究主题: {topic}")
        memory_manager.add_to_long_term_memory("researcher", f"research_{topic}", research_output)
        
        print(f"   研究完成，输出长度: {len(research_output)} 字符")
        
        return {
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "researcher": {
                    "output": research_output,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }
            },
            "collaboration_history": state.get("collaboration_history", []) + [
                f"研究员完成了关于 '{topic}' 的研究"
            ]
        }
        
    except Exception as e:
        print(f"   研究失败: {e}")
        return {
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "researcher": {
                    "output": f"研究失败: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed"
                }
            }
        }

def analyst_agent_node(state: TeamState) -> dict:
    """分析师Agent - 负责数据分析和洞察"""
    print("📊 [分析师] 进行数据分析...")
    
    researcher_output = state.get("agent_outputs", {}).get("researcher", {}).get("output", "")
    project_brief = state.get("project_brief", {})
    
    # 检索分析相关记忆
    relevant_memory = memory_manager.retrieve_relevant_memory("analyst", "分析")
    
    analysis_prompt = f"""作为专业数据分析师，请分析研究员提供的信息:

研究报告:
{researcher_output}

项目要求:
{json.dumps(project_brief, ensure_ascii=False, indent=2)}

历史分析经验:
{json.dumps(relevant_memory, ensure_ascii=False, indent=2)}

请提供:
1. 关键指标和KPI分析
2. 趋势分析和预测
3. SWOT分析
4. 风险评估和机会识别
5. 可行性分析
6. 数据驱动的建议

输出格式: 结构化的分析报告"""
    
    try:
        response = analyst_llm.invoke([{"role": "user", "content": analysis_prompt}])
        analysis_output = response.content
        
        # 更新记忆
        memory_manager.update_short_term_memory("analyst", "完成数据分析")
        memory_manager.add_to_long_term_memory("analyst", "latest_analysis", analysis_output)
        
        print(f"   分析完成，输出长度: {len(analysis_output)} 字符")
        
        return {
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "analyst": {
                    "output": analysis_output,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }
            },
            "collaboration_history": state.get("collaboration_history", []) + [
                "分析师完成了数据分析和洞察提取"
            ]
        }
        
    except Exception as e:
        print(f"   分析失败: {e}")
        return {
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "analyst": {
                    "output": f"分析失败: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed"
                }
            }
        }

def writer_agent_node(state: TeamState) -> dict:
    """撰写员Agent - 负责内容创作"""
    print("✍️ [撰写员] 进行内容创作...")
    
    agent_outputs = state.get("agent_outputs", {})
    research_output = agent_outputs.get("researcher", {}).get("output", "")
    analysis_output = agent_outputs.get("analyst", {}).get("output", "")
    project_brief = state.get("project_brief", {})
    
    # 检索写作相关记忆
    relevant_memory = memory_manager.retrieve_relevant_memory("writer", "写作")
    
    writing_prompt = f"""作为专业撰写员，基于研究和分析结果创作内容:

研究报告:
{research_output[:1000]}...

分析报告:
{analysis_output[:1000]}...

项目要求:
{json.dumps(project_brief, ensure_ascii=False, indent=2)}

写作经验:
{json.dumps(relevant_memory, ensure_ascii=False, indent=2)}

请创作:
1. 引人入胜的标题和摘要
2. 逻辑清晰的内容结构
3. 数据支撑的论证
4. 具体可行的建议
5. 专业且易懂的表达

输出格式: 完整的专业文档"""
    
    try:
        response = writer_llm.invoke([{"role": "user", "content": writing_prompt}])
        writing_output = response.content
        
        # 更新记忆
        memory_manager.update_short_term_memory("writer", "完成内容创作")
        memory_manager.add_to_long_term_memory("writer", "latest_writing", writing_output)
        
        print(f"   创作完成，输出长度: {len(writing_output)} 字符")
        
        return {
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "writer": {
                    "output": writing_output,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }
            },
            "collaboration_history": state.get("collaboration_history", []) + [
                "撰写员完成了内容创作"
            ]
        }
        
    except Exception as e:
        print(f"   创作失败: {e}")
        return {
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "writer": {
                    "output": f"创作失败: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed"
                }
            }
        }

def reviewer_agent_node(state: TeamState) -> dict:
    """审查员Agent - 负责质量控制"""
    print("🔍 [审查员] 进行质量审查...")
    
    agent_outputs = state.get("agent_outputs", {})
    writing_output = agent_outputs.get("writer", {}).get("output", "")
    project_brief = state.get("project_brief", {})
    
    # 检索审查相关记忆
    relevant_memory = memory_manager.retrieve_relevant_memory("reviewer", "审查")
    
    review_prompt = f"""作为专业审查员，请全面审查以下内容:

待审查文档:
{writing_output}

项目要求:
{json.dumps(project_brief, ensure_ascii=False, indent=2)}

审查标准:
{json.dumps(relevant_memory, ensure_ascii=False, indent=2)}

请检查:
1. 内容准确性和完整性
2. 逻辑结构和连贯性
3. 语言表达和专业性
4. 数据支撑和论证强度
5. 格式规范和一致性
6. 目标对象适配性

请提供:
1. 质量评分 (1-10分)
2. 优点和亮点
3. 问题和改进建议
4. 最终审查结论

输出格式: 详细的审查报告"""
    
    try:
        response = reviewer_llm.invoke([{"role": "user", "content": review_prompt}])
        review_output = response.content
        
        # 更新记忆
        memory_manager.update_short_term_memory("reviewer", "完成质量审查")
        memory_manager.add_to_long_term_memory("reviewer", "latest_review", review_output)
        
        print(f"   审查完成，输出长度: {len(review_output)} 字符")
        
        return {
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "reviewer": {
                    "output": review_output,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }
            },
            "collaboration_history": state.get("collaboration_history", []) + [
                "审查员完成了质量审查"
            ],
            "completion_status": {
                "all_agents_completed": True,
                "final_review_done": True,
                "project_status": "completed"
            }
        }
        
    except Exception as e:
        print(f"   审查失败: {e}")
        return {
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "reviewer": {
                    "output": f"审查失败: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed"
                }
            }
        }

def coordination_node(state: TeamState) -> dict:
    """协调节点 - 管理Agent间协作"""
    print("🤝 [协调] 管理团队协作...")
    
    agent_outputs = state.get("agent_outputs", {})
    collaboration_history = state.get("collaboration_history", [])
    
    # 检查各Agent完成状态
    completed_agents = []
    for agent_name, output in agent_outputs.items():
        if output.get("status") == "completed":
            completed_agents.append(agent_name)
    
    # 更新共享记忆
    shared_memory = state.get("shared_memory", {})
    shared_memory["team_progress"] = {
        "completed_agents": completed_agents,
        "total_agents": 4,
        "completion_rate": len(completed_agents) / 4 * 100,
        "last_update": datetime.now().isoformat()
    }
    
    # 生成协作总结
    if len(completed_agents) == 4:
        coordination_summary = "🎉 所有Agent都已完成任务！团队协作成功结束。"
        current_phase = "completed"
    else:
        remaining = 4 - len(completed_agents)
        coordination_summary = f"📊 已完成 {len(completed_agents)}/4 个任务，还有 {remaining} 个任务进行中..."
        current_phase = "in_progress"
    
    print(f"   {coordination_summary}")
    
    return {
        "shared_memory": shared_memory,
        "current_phase": current_phase,
        "collaboration_history": collaboration_history + [coordination_summary]
    }

# 5. 构建多Agent系统
def create_multi_agent_system():
    """创建多Agent协作系统"""
    
    workflow = StateGraph(TeamState)
    
    # 添加Agent节点
    workflow.add_node("researcher", researcher_agent_node)
    workflow.add_node("analyst", analyst_agent_node) 
    workflow.add_node("writer", writer_agent_node)
    workflow.add_node("reviewer", reviewer_agent_node)
    workflow.add_node("coordination", coordination_node)
    
    # 设置协作流程
    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "coordination")
    workflow.add_edge("coordination", "analyst")
    workflow.add_edge("analyst", "coordination")
    workflow.add_edge("coordination", "writer") 
    workflow.add_edge("writer", "coordination")
    workflow.add_edge("coordination", "reviewer")
    workflow.add_edge("reviewer", "coordination")
    workflow.add_edge("coordination", END)
    
    return workflow.compile(checkpointer=MemorySaver())

# 6. 演示函数
def demo_multi_agent_collaboration():
    """演示多Agent协作"""
    print("=" * 80)
    print("🤖 Challenge 6: 高级记忆和多Agent系统")
    print("=" * 80)
    print("团队成员:")
    print("🔬 研究员 - 信息收集和研究")
    print("📊 分析师 - 数据分析和洞察") 
    print("✍️  撰写员 - 内容创作")
    print("🔍 审查员 - 质量控制")
    print("🤝 协调员 - 团队协作管理")
    print("-" * 80)
    
    # 获取项目信息
    print("请输入项目信息:")
    topic = input("研究主题: ").strip() or "人工智能在教育领域的应用"
    objective = input("项目目标: ").strip() or "分析AI教育应用的现状、趋势和建议"
    deadline = input("项目期限 (天数): ").strip() or "7"
    
    try:
        deadline_days = int(deadline)
    except ValueError:
        deadline_days = 7
    
    # 构建项目简介
    project_brief = {
        "topic": topic,
        "objective": objective,
        "deadline": deadline_days,
        "requirements": [
            "深入的背景研究",
            "数据驱动的分析", 
            "专业的内容输出",
            "严格的质量控制"
        ],
        "deliverables": [
            "研究报告",
            "分析报告", 
            "专业文档",
            "质量评估"
        ]
    }
    
    # 创建多Agent系统
    team_system = create_multi_agent_system()
    
    # 初始状态
    initial_state = {
        "messages": [HumanMessage(content=f"启动项目: {topic}")],
        "project_brief": project_brief,
        "task_assignments": {
            "researcher": "信息收集和背景研究",
            "analyst": "数据分析和洞察提取",
            "writer": "专业内容创作",
            "reviewer": "质量审查和优化"
        },
        "agent_outputs": {},
        "collaboration_history": [],
        "shared_memory": {},
        "current_phase": "starting",
        "completion_status": {}
    }
    
    # 配置
    thread_id = f"team_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n🚀 启动团队项目: {thread_id}")
    print(f"📋 项目主题: {topic}")
    print(f"🎯 项目目标: {objective}")
    print("=" * 60)
    
    try:
        start_time = time.time()
        
        # 执行多Agent协作
        print("\n🔄 开始团队协作...")
        result = team_system.invoke(initial_state, config)
        
        execution_time = time.time() - start_time
        
        # 显示协作结果
        print("\n" + "="*60)
        print("📊 团队协作结果")
        print("="*60)
        
        # 显示各Agent输出摘要
        agent_outputs = result.get("agent_outputs", {})
        for agent_name, output in agent_outputs.items():
            status = output.get("status", "unknown")
            content_length = len(output.get("output", ""))
            timestamp = output.get("timestamp", "N/A")
            
            status_icon = "✅" if status == "completed" else "❌"
            print(f"{status_icon} {agent_name}: {status} ({content_length} 字符) - {timestamp}")
        
        # 显示协作历史
        print(f"\n📝 协作历史:")
        for event in result.get("collaboration_history", []):
            print(f"   • {event}")
        
        # 显示共享记忆状态
        shared_memory = result.get("shared_memory", {})
        team_progress = shared_memory.get("team_progress", {})
        if team_progress:
            print(f"\n📈 团队进度:")
            print(f"   完成率: {team_progress.get('completion_rate', 0):.1f}%")
            print(f"   完成Agent: {', '.join(team_progress.get('completed_agents', []))}")
        
        # 显示最终输出
        print(f"\n📋 最终输出预览:")
        for agent_name in ["researcher", "analyst", "writer", "reviewer"]:
            if agent_name in agent_outputs:
                output = agent_outputs[agent_name].get("output", "")
                preview = output[:200] + "..." if len(output) > 200 else output
                print(f"\n🔹 {agent_name}输出:")
                print(f"   {preview}")
        
        # 性能统计
        print(f"\n⏱️  执行统计:")
        print(f"   总执行时间: {execution_time:.2f}秒")
        print(f"   项目状态: {result.get('current_phase', 'unknown')}")
        
        # 记忆系统统计
        print(f"\n🧠 记忆系统统计:")
        for agent_id, memory in memory_manager.agent_memories.items():
            short_term_count = len(memory["short_term_memory"])
            long_term_count = len(memory["long_term_memory"])
            print(f"   {agent_id}: 短期记忆 {short_term_count} 条，长期记忆 {long_term_count} 条")
        
    except KeyboardInterrupt:
        print(f"\n⏸️  团队项目已暂停: {thread_id}")
        print("可以稍后恢复继续执行")
    except Exception as e:
        print(f"❌ 团队协作错误: {e}")
        import traceback
        traceback.print_exc()

def demo_memory_system():
    """演示记忆系统"""
    print("\n🧠 记忆系统演示:")
    print("-" * 40)
    
    # 模拟记忆操作
    test_agent = "demo_agent"
    
    # 添加短期记忆
    print("添加短期记忆...")
    memory_manager.update_short_term_memory(test_agent, "讨论了AI在教育领域的应用")
    memory_manager.update_short_term_memory(test_agent, "分析了市场趋势和用户需求")
    
    # 添加长期记忆
    print("添加长期记忆...")
    memory_manager.add_to_long_term_memory(test_agent, "ai_education_facts", {
        "market_size": "500亿美元",
        "growth_rate": "15%",
        "key_players": ["Google", "Microsoft", "IBM"]
    })
    
    # 检索记忆
    print("检索相关记忆...")
    relevant = memory_manager.retrieve_relevant_memory(test_agent, "AI教育")
    
    print("📊 检索结果:")
    print(f"   短期记忆: {len(relevant['short_term'])} 条")
    print(f"   长期记忆: {len(relevant['long_term'])} 条")
    
    for item in relevant["short_term"]:
        print(f"   • {item['content']}")

if __name__ == "__main__":
    print("🚀 启动 Challenge 6: 高级记忆和多Agent系统")
    
    print("\n选择演示:")
    print("1. 多Agent团队协作")
    print("2. 记忆系统演示")
    
    choice = input("请选择 (1-2): ").strip()
    
    if choice == "2":
        demo_memory_system()
    else:
        demo_multi_agent_collaboration()
