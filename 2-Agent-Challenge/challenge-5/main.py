"""
Challenge 5: 人机交互和审批流程

学习目标:
- 掌握Human-in-the-loop模式
- 实现中断和恢复机制
- 构建审批工作流
- 学习动态干预

核心概念:
1. interrupt_before/interrupt_after - 人工干预点
2. 人工干预点 - 暂停等待人工输入
3. 状态修改和继续 - 人工修改后继续执行
4. 审批流程设计 - 多级审批工作流
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, Literal, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 检查环境变量
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  请设置 OPENAI_API_KEY 环境变量")
    exit(1)

# 1. 定义状态结构
class ApprovalState(TypedDict):
    """审批流程状态"""
    messages: Annotated[list, add_messages]
    request_type: str  # 请求类型: expense, purchase, vacation, project
    request_details: dict  # 请求详情
    current_approver: str  # 当前审批人
    approval_chain: list  # 审批链
    approvals: dict  # 审批记录
    status: str  # pending, approved, rejected, needs_revision
    comments: list  # 审批意见
    created_at: str
    updated_at: str

class DocumentState(TypedDict):
    """文档审核状态"""
    messages: Annotated[list, add_messages]
    document_type: str  # 文档类型
    content: str  # 文档内容
    review_checklist: dict  # 审核清单
    reviewer_feedback: list  # 审核反馈
    revision_history: list  # 修订历史
    final_approval: bool  # 最终批准
    quality_score: int  # 质量分数

# 2. 初始化LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# 3. 审批规则配置
APPROVAL_RULES = {
    "expense": {
        "thresholds": [500, 2000, 10000],  # 审批金额阈值
        "approvers": ["supervisor", "manager", "director", "ceo"]
    },
    "purchase": {
        "thresholds": [1000, 5000, 25000],
        "approvers": ["team_lead", "dept_manager", "finance_director", "ceo"]
    },
    "vacation": {
        "thresholds": [5, 10, 20],  # 天数阈值
        "approvers": ["supervisor", "hr_manager"]
    },
    "project": {
        "thresholds": [10000, 50000, 200000],  # 预算阈值
        "approvers": ["project_manager", "dept_head", "vp", "board"]
    }
}

# 4. 审批流程节点
def request_analysis_node(state: ApprovalState) -> dict:
    """请求分析节点 - 分析请求并确定审批链"""
    print("🔍 [请求分析] 分析请求详情...")
    
    request_details = state.get("request_details", {})
    request_type = state.get("request_type", "")
    
    if not request_type or request_type not in APPROVAL_RULES:
        return {
            "status": "rejected",
            "comments": ["无效的请求类型"],
            "updated_at": datetime.now().isoformat()
        }
    
    rules = APPROVAL_RULES[request_type]
    amount = request_details.get("amount", 0)
    
    # 确定所需审批级别
    approval_level = 0
    for threshold in rules["thresholds"]:
        if amount > threshold:
            approval_level += 1
        else:
            break
    
    # 构建审批链
    required_approvers = rules["approvers"][:approval_level + 1]
    
    analysis_prompt = f"""分析以下{request_type}请求的合理性:

请求详情: {json.dumps(request_details, ensure_ascii=False, indent=2)}

请评估:
1. 请求的合理性和必要性
2. 金额是否合理
3. 是否符合公司政策
4. 潜在风险和建议

返回JSON格式的分析结果:
{{
    "risk_level": "low/medium/high",
    "recommendation": "approve/reject/needs_revision",
    "key_concerns": ["关注点1", "关注点2"],
    "suggestions": ["建议1", "建议2"]
}}"""
    
    try:
        response = llm.invoke([{"role": "user", "content": analysis_prompt}])
        analysis = json.loads(response.content)
        
        print(f"   风险等级: {analysis.get('risk_level', 'unknown')}")
        print(f"   建议: {analysis.get('recommendation', 'unknown')}")
        
        return {
            "approval_chain": required_approvers,
            "current_approver": required_approvers[0] if required_approvers else None,
            "comments": [f"AI分析: {analysis.get('recommendation', '需要人工审核')}"],
            "status": "pending",
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"   分析失败: {e}")
        return {
            "approval_chain": required_approvers,
            "current_approver": required_approvers[0] if required_approvers else None,
            "comments": ["AI分析失败，需要人工审核"],
            "status": "pending",
            "updated_at": datetime.now().isoformat()
        }

def approval_review_node(state: ApprovalState) -> dict:
    """审批审核节点 - 等待人工审批"""
    print("⏳ [等待审批] 等待人工审批...")
    
    current_approver = state.get("current_approver")
    request_details = state.get("request_details", {})
    comments = state.get("comments", [])
    
    print(f"   当前审批人: {current_approver}")
    print(f"   请求详情: {json.dumps(request_details, ensure_ascii=False, indent=2)}")
    print(f"   历史意见: {'; '.join(comments[-3:])}")
    
    # 这里会被interrupt_before中断，等待人工输入
    return {
        "status": "awaiting_approval",
        "updated_at": datetime.now().isoformat()
    }

def approval_decision_node(state: ApprovalState) -> dict:
    """审批决策节点 - 处理审批结果"""
    print("✅ [审批决策] 处理审批结果...")
    
    # 检查最新的消息中是否包含审批决定
    last_message = state["messages"][-1] if state["messages"] else None
    
    if not last_message:
        return {"status": "error", "updated_at": datetime.now().isoformat()}
    
    decision_text = last_message.content.lower()
    current_approver = state.get("current_approver")
    approval_chain = state.get("approval_chain", [])
    approvals = state.get("approvals", {})
    comments = state.get("comments", [])
    
    # 解析审批决定
    if "批准" in decision_text or "approve" in decision_text:
        decision = "approved"
    elif "拒绝" in decision_text or "reject" in decision_text:
        decision = "rejected"
    elif "修改" in decision_text or "revision" in decision_text:
        decision = "needs_revision"
    else:
        decision = "pending"
    
    # 记录审批结果
    approvals[current_approver] = {
        "decision": decision,
        "comment": last_message.content,
        "timestamp": datetime.now().isoformat()
    }
    
    comments.append(f"{current_approver}: {decision} - {last_message.content}")
    
    # 处理不同的决定
    if decision == "rejected":
        return {
            "approvals": approvals,
            "comments": comments,
            "status": "rejected",
            "updated_at": datetime.now().isoformat()
        }
    elif decision == "needs_revision":
        return {
            "approvals": approvals,
            "comments": comments,
            "status": "needs_revision",
            "updated_at": datetime.now().isoformat()
        }
    elif decision == "approved":
        # 查找下一个审批人
        try:
            current_index = approval_chain.index(current_approver)
            if current_index + 1 < len(approval_chain):
                next_approver = approval_chain[current_index + 1]
                print(f"   转到下一级审批: {next_approver}")
                return {
                    "current_approver": next_approver,
                    "approvals": approvals,
                    "comments": comments,
                    "status": "pending",
                    "updated_at": datetime.now().isoformat()
                }
            else:
                # 所有审批完成
                print("   所有审批完成!")
                return {
                    "approvals": approvals,
                    "comments": comments,
                    "status": "approved", 
                    "updated_at": datetime.now().isoformat()
                }
        except ValueError:
            return {
                "approvals": approvals,
                "comments": comments,
                "status": "error",
                "updated_at": datetime.now().isoformat()
            }
    
    return {
        "approvals": approvals,
        "comments": comments,
        "status": "pending",
        "updated_at": datetime.now().isoformat()
    }

def notification_node(state: ApprovalState) -> dict:
    """通知节点 - 发送状态通知"""
    print("📧 [通知] 发送状态通知...")
    
    status = state.get("status")
    request_type = state.get("request_type")
    request_details = state.get("request_details", {})
    
    notifications = {
        "approved": f"✅ {request_type}请求已批准: {request_details.get('title', 'N/A')}",
        "rejected": f"❌ {request_type}请求已拒绝: {request_details.get('title', 'N/A')}",
        "needs_revision": f"📝 {request_type}请求需要修改: {request_details.get('title', 'N/A')}"
    }
    
    notification = notifications.get(status, f"📋 {request_type}请求状态更新")
    
    print(f"   通知内容: {notification}")
    
    return {
        "messages": [AIMessage(content=notification)],
        "updated_at": datetime.now().isoformat()
    }

# 5. 路由函数
def route_approval_flow(state: ApprovalState) -> Literal["approval_review", "notification"]:
    """路由审批流程"""
    status = state.get("status", "pending")
    
    if status == "pending":
        return "approval_review"
    else:
        return "notification"

def route_after_decision(state: ApprovalState) -> Literal["approval_review", "notification"]:
    """审批决定后的路由"""
    status = state.get("status", "pending")
    
    if status == "pending":
        return "approval_review"  # 需要下一级审批
    else:
        return "notification"  # 最终状态，发送通知

# 6. 构建审批工作流
def create_approval_workflow():
    """创建审批工作流"""
    
    workflow = StateGraph(ApprovalState)
    
    # 添加节点
    workflow.add_node("request_analysis", request_analysis_node)
    workflow.add_node("approval_review", approval_review_node)
    workflow.add_node("approval_decision", approval_decision_node)
    workflow.add_node("notification", notification_node)
    
    # 设置流程
    workflow.add_edge(START, "request_analysis")
    
    workflow.add_conditional_edges(
        "request_analysis",
        route_approval_flow,
        {
            "approval_review": "approval_review",
            "notification": "notification"
        }
    )
    
    # 在审批节点前中断，等待人工输入
    workflow.add_edge("approval_review", "approval_decision")
    
    workflow.add_conditional_edges(
        "approval_decision",
        route_after_decision,
        {
            "approval_review": "approval_review",
            "notification": "notification"
        }
    )
    
    workflow.add_edge("notification", END)
    
    # 编译时设置中断点
    return workflow.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["approval_review"]  # 在审批节点前中断
    )

# 7. 文档审核工作流
def document_review_node(state: DocumentState) -> dict:
    """文档审核节点"""
    print("📄 [文档审核] 自动审核文档...")
    
    content = state.get("content", "")
    document_type = state.get("document_type", "")
    
    review_prompt = f"""请审核以下{document_type}文档:

{content}

请检查:
1. 内容完整性和准确性
2. 格式和结构
3. 语言表达和语法
4. 符合规范要求

返回JSON格式的审核结果:
{{
    "quality_score": 0-100,
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"],
    "requires_human_review": true/false
}}"""
    
    try:
        response = llm.invoke([{"role": "user", "content": review_prompt}])
        review_result = json.loads(response.content)
        
        quality_score = review_result.get("quality_score", 0)
        issues = review_result.get("issues", [])
        suggestions = review_result.get("suggestions", [])
        
        checklist = {
            "grammar_check": quality_score > 80,
            "format_check": len(issues) < 3,
            "completeness_check": quality_score > 70,
            "compliance_check": review_result.get("requires_human_review", False)
        }
        
        feedback = [
            f"质量分数: {quality_score}/100",
            f"发现问题: {len(issues)}个",
            f"改进建议: {len(suggestions)}条"
        ]
        
        return {
            "review_checklist": checklist,
            "reviewer_feedback": feedback,
            "quality_score": quality_score,
            "final_approval": quality_score > 85 and len(issues) < 2
        }
        
    except Exception as e:
        print(f"   审核失败: {e}")
        return {
            "reviewer_feedback": ["自动审核失败，需要人工审核"],
            "quality_score": 0,
            "final_approval": False
        }

def create_document_workflow():
    """创建文档审核工作流"""
    
    workflow = StateGraph(DocumentState)
    workflow.add_node("document_review", document_review_node)
    
    workflow.add_edge(START, "document_review")
    workflow.add_edge("document_review", END)
    
    return workflow.compile(
        checkpointer=MemorySaver(),
        interrupt_after=["document_review"]  # 在审核后中断，等待人工确认
    )

# 8. 演示函数
def demo_approval_workflow():
    """演示审批工作流"""
    print("=" * 70)
    print("🏢 审批工作流演示")
    print("=" * 70)
    print("支持的请求类型:")
    print("1. expense - 费用申请")
    print("2. purchase - 采购申请")
    print("3. vacation - 休假申请")
    print("4. project - 项目申请")
    print("-" * 70)
    
    # 选择请求类型
    request_type = input("选择请求类型 (expense/purchase/vacation/project): ").strip()
    if request_type not in APPROVAL_RULES:
        request_type = "expense"
    
    # 输入请求详情
    print(f"\n输入{request_type}请求详情:")
    
    if request_type == "expense":
        title = input("费用项目: ") or "办公用品采购"
        amount = float(input("金额: ") or "1500")
        details = {
            "title": title,
            "amount": amount,
            "category": "office_supplies",
            "description": f"{title} - {amount}元"
        }
    elif request_type == "vacation":
        title = input("休假类型: ") or "年假"
        days = int(input("天数: ") or "7")
        details = {
            "title": title,
            "amount": days,  # 使用天数作为amount用于阈值判断
            "start_date": "2024-03-01",
            "end_date": "2024-03-08",
            "reason": "个人休假"
        }
    else:
        title = input("项目/采购名称: ") or "新设备采购"
        amount = float(input("预算/金额: ") or "8000")
        details = {
            "title": title,
            "amount": amount,
            "description": f"{title} - 预算{amount}元"
        }
    
    # 创建工作流
    approval_flow = create_approval_workflow()
    
    # 初始状态
    initial_state = {
        "messages": [HumanMessage(content=f"提交{request_type}请求")],
        "request_type": request_type,
        "request_details": details,
        "current_approver": "",
        "approval_chain": [],
        "approvals": {},
        "status": "pending",
        "comments": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    # 配置
    thread_id = f"approval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n🚀 启动审批流程: {thread_id}")
    print("-" * 50)
    
    try:
        # 第一次执行 - 分析请求
        result = approval_flow.invoke(initial_state, config)
        
        print("\n📊 审批分析结果:")
        print(f"审批链: {' → '.join(result.get('approval_chain', []))}")
        print(f"当前审批人: {result.get('current_approver', 'N/A')}")
        print(f"状态: {result.get('status', 'N/A')}")
        
        # 模拟审批过程
        while result.get("status") == "pending":
            current_approver = result.get("current_approver")
            
            print(f"\n⏳ 等待 {current_approver} 审批...")
            print("请输入审批意见 (包含'批准'、'拒绝'或'修改'):")
            
            approval_input = input(f"{current_approver}: ").strip()
            if not approval_input:
                approval_input = "批准，同意此请求"
            
            # 添加审批意见并继续执行
            approval_message = HumanMessage(content=approval_input)
            
            # 更新状态并继续
            result = approval_flow.invoke(
                {"messages": [approval_message]},
                config
            )
            
            print(f"✅ {current_approver} 已审批")
            print(f"当前状态: {result.get('status', 'N/A')}")
            
            # 显示审批记录
            approvals = result.get("approvals", {})
            if approvals:
                print("\n📋 审批记录:")
                for approver, record in approvals.items():
                    decision = record.get("decision", "unknown")
                    comment = record.get("comment", "无意见")
                    print(f"   {approver}: {decision} - {comment[:50]}...")
        
        # 最终结果
        print(f"\n🎯 最终结果: {result.get('status', 'unknown')}")
        
        # 显示完整审批记录
        print("\n📑 完整审批记录:")
        for comment in result.get("comments", []):
            print(f"   {comment}")
            
    except KeyboardInterrupt:
        print(f"\n⏸️  审批流程已暂停: {thread_id}")
        print("可以稍后恢复继续处理")
    except Exception as e:
        print(f"❌ 审批流程错误: {e}")

def demo_document_review():
    """演示文档审核"""
    print("\n" + "="*60)
    print("📄 文档审核工作流演示")
    print("="*60)
    
    document_type = input("文档类型 (report/proposal/manual): ").strip() or "report"
    
    print("请输入文档内容 (多行输入，空行结束):")
    content_lines = []
    while True:
        line = input()
        if not line:
            break
        content_lines.append(line)
    
    content = "\n".join(content_lines) or "这是一个示例报告文档，包含项目进展和分析结果。"
    
    # 创建文档审核工作流
    doc_workflow = create_document_workflow()
    
    initial_state = {
        "messages": [HumanMessage(content="开始文档审核")],
        "document_type": document_type,
        "content": content,
        "review_checklist": {},
        "reviewer_feedback": [],
        "revision_history": [],
        "final_approval": False,
        "quality_score": 0
    }
    
    config = {"configurable": {"thread_id": f"doc_review_{int(time.time())}"}}
    
    print("\n🔍 开始自动审核...")
    result = doc_workflow.invoke(initial_state, config)
    
    print("\n📊 审核结果:")
    print(f"质量分数: {result.get('quality_score', 0)}/100")
    print(f"最终批准: {'是' if result.get('final_approval', False) else '否'}")
    
    print("\n📝 审核反馈:")
    for feedback in result.get("reviewer_feedback", []):
        print(f"   {feedback}")
    
    print("\n✅ 审核清单:")
    checklist = result.get("review_checklist", {})
    for check, passed in checklist.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")

if __name__ == "__main__":
    print("🚀 启动 Challenge 5: 人机交互和审批流程")
    
    print("\n选择演示:")
    print("1. 审批工作流")
    print("2. 文档审核工作流")
    
    choice = input("请选择 (1-2): ").strip()
    
    if choice == "2":
        demo_document_review()
    else:
        demo_approval_workflow()
