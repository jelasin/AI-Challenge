"""
Challenge 3: 并行处理和子图

学习目标:
- 掌握并行节点执行
- 学习子图(Subgraph)设计
- 理解复杂工作流编排
- 实现结果聚合策略

核心概念:
1. 并行节点处理 - 同时执行多个任务
2. 子图嵌套 - 模块化工作流设计
3. 状态合并 - 多个结果的整合
4. 性能优化 - 并行执行的优势
"""

import os
import asyncio
import json
import time
from datetime import datetime
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 检查环境变量
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  请设置 OPENAI_API_KEY 环境变量")
    exit(1)

# 1. 定义状态结构
class AnalysisState(TypedDict):
    """数据分析状态"""
    messages: Annotated[list, add_messages]
    raw_data: str  # 原始数据
    data_summary: str  # 数据摘要
    chart_description: str  # 图表描述
    insights: str  # 数据洞察
    report: str  # 最终报告
    processing_time: dict  # 处理时间记录

class TaskState(TypedDict):
    """子任务状态"""
    task_id: str
    input_data: str
    result: str
    processing_time: float
    status: str

# 2. 初始化LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# 3. 模拟数据源
SAMPLE_DATA = {
    "sales": """
    产品销售数据 (Q1 2024):
    - iPhone: 1200万台, 收入: 180亿美元
    - MacBook: 400万台, 收入: 80亿美元  
    - iPad: 800万台, 收入: 60亿美元
    - Watch: 600万台, 收入: 20亿美元
    增长率: iPhone(+15%), MacBook(+8%), iPad(-3%), Watch(+25%)
    """,
    
    "user_behavior": """
    用户行为数据:
    - 日活跃用户: 280万
    - 平均使用时长: 45分钟
    - 用户留存率: 85%
    - 最受欢迎功能: 搜索(60%), 推荐(25%), 社交(15%)
    地域分布: 北京(30%), 上海(25%), 深圳(20%), 其他(25%)
    """,
    
    "market": """
    市场数据:
    - 行业总规模: 1000亿美元
    - 市场份额: 公司A(25%), 公司B(20%), 公司C(18%), 其他(37%)
    - 增长趋势: +12% YoY
    - 主要驱动因素: AI技术采用, 移动端普及, 云服务需求
    """
}

# 4. 并行分析节点
def data_summary_node(state: AnalysisState) -> dict:
    """数据摘要节点 - 并行任务1"""
    print("📊 [数据摘要] 分析数据概况...")
    start_time = time.time()
    
    prompt = f"""请对以下数据进行摘要分析:

{state['raw_data']}

要求:
1. 识别关键数据点
2. 计算主要指标
3. 生成简洁摘要

返回格式: 数据摘要文本"""
    
    response = llm.invoke([{"role": "user", "content": prompt}])
    processing_time = time.time() - start_time
    
    print(f"   处理完成，耗时: {processing_time:.2f}秒")
    
    return {
        "data_summary": response.content,
        "processing_time": {
            **state.get("processing_time", {}),
            "data_summary": processing_time
        }
    }

def chart_description_node(state: AnalysisState) -> dict:
    """图表描述节点 - 并行任务2"""
    print("📈 [图表描述] 生成可视化建议...")
    start_time = time.time()
    
    prompt = f"""基于以下数据，设计合适的图表和可视化方案:

{state['raw_data']}

要求:
1. 推荐最佳图表类型
2. 确定关键可视化维度
3. 建议交互功能
4. 设计颜色和布局

返回格式: 图表设计描述"""
    
    response = llm.invoke([{"role": "user", "content": prompt}])
    processing_time = time.time() - start_time
    
    print(f"   处理完成，耗时: {processing_time:.2f}秒")
    
    return {
        "chart_description": response.content,
        "processing_time": {
            **state.get("processing_time", {}),
            "chart_description": processing_time
        }
    }

def insights_generation_node(state: AnalysisState) -> dict:
    """洞察生成节点 - 并行任务3"""
    print("💡 [洞察生成] 挖掘数据洞察...")
    start_time = time.time()
    
    prompt = f"""深度分析以下数据，挖掘有价值的业务洞察:

{state['raw_data']}

要求:
1. 识别趋势和模式
2. 发现异常和机会
3. 提供可行性建议
4. 预测未来发展

返回格式: 深度洞察分析"""
    
    response = llm.invoke([{"role": "user", "content": prompt}])
    processing_time = time.time() - start_time
    
    print(f"   处理完成，耗时: {processing_time:.2f}秒")
    
    return {
        "insights": response.content,
        "processing_time": {
            **state.get("processing_time", {}),
            "insights": processing_time
        }
    }

# 5. 子图 - 数据预处理工作流
def create_preprocessing_subgraph():
    """创建数据预处理子图"""
    
    def data_validation_node(state: TaskState) -> dict:
        """数据验证节点"""
        print(f"   🔍 [验证] 任务 {state['task_id']}")
        time.sleep(0.5)  # 模拟处理时间
        
        return {
            "result": f"数据验证完成: {state['input_data'][:30]}...",
            "status": "validated"
        }
    
    def data_cleaning_node(state: TaskState) -> dict:
        """数据清洗节点"""
        print(f"   🧹 [清洗] 任务 {state['task_id']}")
        time.sleep(0.3)  # 模拟处理时间
        
        return {
            "result": f"数据清洗完成: 移除异常值和重复项",
            "status": "cleaned"
        }
    
    def data_transformation_node(state: TaskState) -> dict:
        """数据转换节点"""
        print(f"   🔄 [转换] 任务 {state['task_id']}")
        time.sleep(0.4)  # 模拟处理时间
        
        return {
            "result": f"数据转换完成: 标准化格式",
            "status": "transformed"
        }
    
    # 构建子图
    subgraph = StateGraph(TaskState)
    subgraph.add_node("validate", data_validation_node)
    subgraph.add_node("clean", data_cleaning_node)
    subgraph.add_node("transform", data_transformation_node)
    
    subgraph.add_edge(START, "validate")
    subgraph.add_edge("validate", "clean")
    subgraph.add_edge("clean", "transform")
    subgraph.add_edge("transform", END)
    
    return subgraph.compile()

# 6. 主工作流节点
def data_preprocessing_node(state: AnalysisState) -> dict:
    """数据预处理节点 - 使用子图"""
    print("🔧 [数据预处理] 执行预处理子图...")
    start_time = time.time()
    
    # 创建预处理子图
    preprocessor = create_preprocessing_subgraph()
    
    # 为每种数据类型创建子任务
    data_types = ["sales", "user_behavior", "market"]
    results = []
    
    for i, data_type in enumerate(data_types):
        task_state: TaskState = {
            "task_id": f"preprocess_{data_type}",
            "input_data": SAMPLE_DATA.get(data_type, ""),
            "result": "",
            "processing_time": 0.0,
            "status": "pending"
        }
        
        print(f"   处理 {data_type} 数据...")
        result = preprocessor.invoke(task_state)
        results.append(result["result"])
    
    processing_time = time.time() - start_time
    preprocessed_data = "\n".join(results)
    
    print(f"   预处理完成，耗时: {processing_time:.2f}秒")
    
    return {
        "raw_data": state["raw_data"] + f"\n\n预处理结果:\n{preprocessed_data}",
        "processing_time": {
            **state.get("processing_time", {}),
            "preprocessing": processing_time
        }
    }

def report_generation_node(state: AnalysisState) -> dict:
    """报告生成节点 - 汇总所有结果"""
    print("📋 [报告生成] 汇总分析结果...")
    start_time = time.time()
    
    prompt = f"""基于以下分析结果，生成综合数据分析报告:

数据摘要:
{state.get('data_summary', '未完成')}

图表描述:
{state.get('chart_description', '未完成')}

数据洞察:
{state.get('insights', '未完成')}

要求:
1. 整合所有分析结果
2. 形成逻辑清晰的报告结构
3. 提供执行建议
4. 包含关键指标和可视化建议

返回格式: 完整的数据分析报告"""
    
    response = llm.invoke([{"role": "user", "content": prompt}])
    processing_time = time.time() - start_time
    
    print(f"   报告生成完成，耗时: {processing_time:.2f}秒")
    
    return {
        "report": response.content,
        "processing_time": {
            **state.get("processing_time", {}),
            "report_generation": processing_time
        }
    }

# 7. 并行执行函数
def parallel_analysis_node(state: AnalysisState) -> dict:
    """并行分析节点 - 同时执行多个分析任务"""
    print("⚡ [并行分析] 启动多任务并行处理...")
    
    # 模拟并行执行(在实际应用中可以使用asyncio或线程池)
    start_time = time.time()
    
    # 顺序执行模拟并行(为了演示清晰)
    results = {}
    
    # 任务1: 数据摘要
    summary_result = data_summary_node(state)
    results.update(summary_result)
    
    # 任务2: 图表描述  
    chart_result = chart_description_node(state)
    results.update(chart_result)
    
    # 任务3: 洞察生成
    insights_result = insights_generation_node(state)
    results.update(insights_result)
    
    total_time = time.time() - start_time
    print(f"⚡ 并行任务完成，总耗时: {total_time:.2f}秒")
    
    # 合并处理时间
    processing_times = results.get("processing_time", {})
    processing_times["parallel_total"] = total_time
    results["processing_time"] = processing_times
    
    return results

# 8. 构建主工作流
def create_analysis_workflow():
    """创建数据分析工作流"""
    
    workflow = StateGraph(AnalysisState)
    
    # 添加节点
    workflow.add_node("preprocessing", data_preprocessing_node)
    workflow.add_node("parallel_analysis", parallel_analysis_node)  
    workflow.add_node("report_generation", report_generation_node)
    
    # 设置流程
    workflow.add_edge(START, "preprocessing")
    workflow.add_edge("preprocessing", "parallel_analysis")
    workflow.add_edge("parallel_analysis", "report_generation")
    workflow.add_edge("report_generation", END)
    
    return workflow.compile()

# 9. 演示函数
def run_data_analysis_demo():
    """运行数据分析演示"""
    print("=" * 70)
    print("📊 Challenge 3: 并行处理和子图 - 数据分析系统")
    print("=" * 70)
    print("功能特性:")
    print("🔧 数据预处理子图 - 验证、清洗、转换")
    print("⚡ 并行分析处理 - 摘要、图表、洞察同时进行")
    print("📋 结果汇总整合 - 生成完整分析报告")
    print("⏱️  性能监控 - 记录各阶段处理时间")
    print("-" * 70)
    
    # 选择数据源
    print("\n选择要分析的数据:")
    for i, (key, value) in enumerate(SAMPLE_DATA.items(), 1):
        print(f"{i}. {key}: {value[:50]}...")
    
    try:
        choice = input("\n选择数据源 (1-3, 或直接回车使用销售数据): ").strip()
        
        if choice == "2":
            selected_data = SAMPLE_DATA["user_behavior"]
        elif choice == "3":
            selected_data = SAMPLE_DATA["market"]
        else:
            selected_data = SAMPLE_DATA["sales"]
        
        print(f"\n✅ 已选择数据源，开始分析...")
        print("=" * 50)
        
        # 创建工作流
        analyzer = create_analysis_workflow()
        
        # 初始状态
        initial_state: AnalysisState = {
            "messages": [HumanMessage(content="开始数据分析")],
            "raw_data": selected_data,
            "data_summary": "",
            "chart_description": "",
            "insights": "",
            "report": "",
            "processing_time": {}
        }
        
        # 执行分析
        print("🚀 启动分析工作流...")
        total_start = time.time()
        
        result = analyzer.invoke(initial_state)
        
        total_time = time.time() - total_start
        
        # 显示结果
        print("\n" + "="*70)
        print("📋 分析报告")
        print("="*70)
        print(result.get("report", "报告生成失败"))
        
        # 显示性能统计
        print("\n" + "="*70)
        print("⏱️  性能统计")
        print("="*70)
        processing_times = result.get("processing_time", {})
        
        for stage, duration in processing_times.items():
            print(f"{stage:20}: {duration:.2f}秒")
        
        print(f"{'总处理时间':20}: {total_time:.2f}秒")
        
        # 计算性能提升
        sequential_time = sum(processing_times.get(key, 0) 
                            for key in ["data_summary", "chart_description", "insights"])
        parallel_time = processing_times.get("parallel_total", sequential_time)
        
        if sequential_time > 0:
            speedup = sequential_time / parallel_time
            print(f"\n🚀 并行处理性能提升: {speedup:.2f}x")
        
    except KeyboardInterrupt:
        print("\n👋 分析中断")
    except Exception as e:
        print(f"❌ 分析错误: {e}")

def demo_subgraph():
    """演示子图功能"""
    print("\n🔧 子图演示:")
    print("-" * 30)
    
    preprocessor = create_preprocessing_subgraph()
    
    test_task: TaskState = {
        "task_id": "demo_task",
        "input_data": "示例数据: 销售记录, 用户行为, 市场数据",
        "result": "",
        "processing_time": 0.0,
        "status": "pending"
    }
    
    print("执行预处理子图...")
    result = preprocessor.invoke(test_task)
    
    print(f"结果: {result['result']}")
    print(f"状态: {result['status']}")

if __name__ == "__main__":
    print("🚀 启动 Challenge 3: 并行处理和子图")
    
    # 演示子图
    demo_subgraph()
    
    # 运行数据分析演示
    run_data_analysis_demo()
