#!/usr/bin/env python3
"""
3-MCP-Challenge - Challenge 8: 综合应用：智能工作流引擎
难度：⭐⭐⭐⭐⭐⭐⭐⭐

学习目标：
1. 复杂工作流设计与执行
2. 多模态数据处理
3. 智能决策引擎
4. 工作流可视化
5. 错误恢复与重试机制
6. 性能优化与缓存
7. 实时监控与告警

参考链接：
- MCP Workflow Patterns: https://modelcontextprotocol.io/docs/workflow-patterns
- MCP Multi-Modal: https://modelcontextprotocol.io/docs/multi-modal
- MCP Performance: https://modelcontextprotocol.io/docs/performance
"""

import os
import json
import time
import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import sqlite3
import uuid
import base64
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

from fastmcp import FastMCP
from mcp import types


# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """工作流状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class NodeType(Enum):
    """节点类型枚举"""
    INPUT = "input"
    PROCESSING = "processing"
    DECISION = "decision"
    OUTPUT = "output"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"


@dataclass
class WorkflowNode:
    """工作流节点"""
    id: str
    name: str
    node_type: NodeType
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    conditions: Dict[str, str] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 300.0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class WorkflowEdge:
    """工作流连接边"""
    from_node: str
    to_node: str
    condition: Optional[str] = None
    weight: float = 1.0


@dataclass
class WorkflowExecution:
    """工作流执行记录"""
    id: str
    workflow_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class WorkflowDefinition:
    """工作流定义"""
    id: str
    name: str
    description: str
    version: str
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]
    global_config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


class WorkflowState(TypedDict):
    """LangGraph工作流状态"""
    execution_id: str
    current_node: str
    data: Dict[str, Any]
    context: Dict[str, Any]
    errors: List[str]
    iteration_count: int
    start_time: float


class MultiModalProcessor:
    """多模态数据处理器"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
    
    async def process_text(self, content: str, task: str = "analyze") -> Dict[str, Any]:
        """处理文本数据"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert text processor. Analyze the given text according to the task."),
                ("human", f"Task: {task}\nContent: {content}")
            ])
            
            result = await self.llm.ainvoke(prompt.format_messages())
            return {
                "type": "text",
                "task": task,
                "input_length": len(content),
                "result": result.content,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return {"error": str(e), "type": "text", "task": task}
    
    async def process_json(self, data: Dict[str, Any], task: str = "validate") -> Dict[str, Any]:
        """处理JSON数据"""
        try:
            if task == "validate":
                # 验证JSON结构
                result = {
                    "valid": True,
                    "keys_count": len(data.keys()) if isinstance(data, dict) else 0,
                    "data_type": type(data).__name__,
                    "structure": self._analyze_json_structure(data)
                }
            elif task == "transform":
                # 数据转换
                result = self._transform_json_data(data)
            elif task == "extract":
                # 提取关键信息
                result = self._extract_key_info(data)
            else:
                result = {"message": f"Unknown task: {task}"}
            
            return {
                "type": "json",
                "task": task,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"JSON processing failed: {e}")
            return {"error": str(e), "type": "json", "task": task}
    
    async def process_file(self, file_path: str, task: str = "analyze") -> Dict[str, Any]:
        """处理文件数据"""
        try:
            if not os.path.exists(file_path):
                return {"error": "File not found", "path": file_path}
            
            file_info = {
                "path": file_path,
                "size": os.path.getsize(file_path),
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                "extension": Path(file_path).suffix.lower()
            }
            
            if file_info["extension"] in [".txt", ".md", ".py", ".js", ".html", ".css"]:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    text_result = await self.process_text(content, task)
                    file_info.update(text_result)
            
            return {
                "type": "file",
                "task": task,
                "result": file_info,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            return {"error": str(e), "type": "file", "path": file_path}
    
    def _analyze_json_structure(self, data: Any, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
        """分析JSON结构"""
        if current_depth >= max_depth:
            return {"truncated": True, "type": type(data).__name__}
        
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys())[:10],  # 限制显示前10个键
                "key_count": len(data),
                "sample_values": {
                    k: self._analyze_json_structure(v, max_depth, current_depth + 1)
                    for k, v in list(data.items())[:3]  # 只分析前3个值
                }
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "sample_items": [
                    self._analyze_json_structure(item, max_depth, current_depth + 1)
                    for item in data[:3]  # 只分析前3个元素
                ]
            }
        else:
            return {
                "type": type(data).__name__,
                "value": str(data)[:100] if isinstance(data, str) else data
            }
    
    def _transform_json_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """转换JSON数据"""
        # 这里实现数据转换逻辑
        transformed = {}
        for key, value in data.items():
            # 示例转换：驼峰命名转下划线
            new_key = self._camel_to_snake(key)
            transformed[new_key] = value
        return transformed
    
    def _camel_to_snake(self, name: str) -> str:
        """驼峰命名转下划线命名"""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _extract_key_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """提取关键信息"""
        key_info = {
            "summary": f"Data contains {len(data)} top-level fields",
            "important_fields": [],
            "data_types": defaultdict(int)
        }
        
        for key, value in data.items():
            key_info["data_types"][type(value).__name__] += 1
            if any(keyword in key.lower() for keyword in ["id", "name", "title", "email", "status"]):
                key_info["important_fields"].append({
                    "field": key,
                    "type": type(value).__name__,
                    "value": str(value)[:50]
                })
        
        return key_info


class DecisionEngine:
    """智能决策引擎"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        self.rules: List[Dict[str, Any]] = []
    
    def add_rule(self, name: str, condition: str, action: str, priority: int = 1):
        """添加决策规则"""
        self.rules.append({
            "name": name,
            "condition": condition,
            "action": action,
            "priority": priority,
            "created_at": datetime.now()
        })
        # 按优先级排序
        self.rules.sort(key=lambda x: x["priority"], reverse=True)
    
    async def make_decision(self, context: Dict[str, Any], 
                           question: str = None) -> Dict[str, Any]:
        """做出智能决策"""
        try:
            # 首先尝试基于规则的决策
            rule_decision = self._apply_rules(context)
            if rule_decision:
                return {
                    "type": "rule_based",
                    "decision": rule_decision,
                    "confidence": 0.9,
                    "reasoning": f"Applied rule: {rule_decision['rule_name']}"
                }
            
            # 如果没有匹配的规则，使用LLM决策
            if question:
                llm_decision = await self._llm_decision(context, question)
                return {
                    "type": "llm_based",
                    "decision": llm_decision,
                    "confidence": 0.7,
                    "reasoning": "Generated by language model"
                }
            
            # 默认决策
            return {
                "type": "default",
                "decision": {"action": "continue", "next_node": None},
                "confidence": 0.5,
                "reasoning": "No matching rules or specific question"
            }
            
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            return {
                "type": "error",
                "decision": {"action": "abort", "error": str(e)},
                "confidence": 0.0,
                "reasoning": f"Error occurred: {e}"
            }
    
    def _apply_rules(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """应用规则决策"""
        for rule in self.rules:
            try:
                # 简单的条件评估（实际应用中应该使用更安全的表达式评估器）
                condition = rule["condition"]
                # 替换上下文变量
                for key, value in context.items():
                    condition = condition.replace(f"{{{key}}}", str(value))
                
                # 评估条件（注意：这里使用eval是不安全的，仅用于演示）
                if self._safe_eval(condition, context):
                    return {
                        "rule_name": rule["name"],
                        "action": rule["action"],
                        "matched_condition": rule["condition"]
                    }
            except Exception as e:
                logger.warning(f"Rule evaluation failed for {rule['name']}: {e}")
                continue
        
        return None
    
    def _safe_eval(self, expression: str, context: Dict[str, Any]) -> bool:
        """安全的表达式评估"""
        # 这是一个简化的实现，实际应用中应该使用更安全的表达式评估库
        try:
            # 只允许简单的比较操作
            allowed_chars = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_<>=!&| ()")
            if not all(c in allowed_chars for c in expression):
                return False
            
            # 创建安全的评估环境
            safe_dict = {
                "__builtins__": {},
                "True": True,
                "False": False
            }
            safe_dict.update(context)
            
            return eval(expression, safe_dict)
        except:
            return False
    
    async def _llm_decision(self, context: Dict[str, Any], question: str) -> Dict[str, Any]:
        """基于LLM的决策"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent decision engine. Based on the provided context and question, 
            make a decision and provide your reasoning. Respond in JSON format with the following structure:
            {
                "action": "continue|pause|retry|abort|branch",
                "next_node": "node_id or null",
                "parameters": {},
                "reasoning": "explanation of the decision"
            }"""),
            ("human", f"Context: {json.dumps(context, indent=2)}\n\nQuestion: {question}")
        ])
        
        result = await self.llm.ainvoke(prompt.format_messages())
        try:
            return json.loads(result.content)
        except:
            return {
                "action": "continue",
                "reasoning": result.content,
                "parameters": {}
            }


class WorkflowEngine:
    """工作流引擎"""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.processor = MultiModalProcessor()
        self.decision_engine = DecisionEngine()
        self.db_path = "workflow_engine.db"
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                definition TEXT NOT NULL,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS executions (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                status TEXT NOT NULL,
                input_data TEXT,
                output_data TEXT,
                execution_log TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                error_message TEXT,
                FOREIGN KEY (workflow_id) REFERENCES workflows (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def create_workflow(self, definition: WorkflowDefinition) -> str:
        """创建工作流"""
        self.workflows[definition.id] = definition
        
        # 保存到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO workflows 
            (id, name, definition, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            definition.id,
            definition.name,
            json.dumps(asdict(definition)),
            definition.created_at,
            definition.updated_at
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Workflow created: {definition.name} ({definition.id})")
        return definition.id
    
    async def execute_workflow(self, workflow_id: str, 
                              input_data: Dict[str, Any] = None) -> str:
        """执行工作流"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        execution = WorkflowExecution(
            id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            start_time=datetime.now(),
            input_data=input_data or {}
        )
        
        self.executions[execution.id] = execution
        
        # 异步执行工作流
        asyncio.create_task(self._run_workflow_execution(execution, workflow))
        
        return execution.id
    
    async def _run_workflow_execution(self, execution: WorkflowExecution, 
                                    workflow: WorkflowDefinition):
        """运行工作流执行"""
        try:
            execution.status = WorkflowStatus.RUNNING
            self._log_execution(execution, "Workflow execution started")
            
            # 构建LangGraph
            graph = self._build_langgraph(workflow)
            
            # 初始状态
            initial_state: WorkflowState = {
                "execution_id": execution.id,
                "current_node": self._find_start_node(workflow),
                "data": execution.input_data.copy(),
                "context": {"workflow_id": workflow.id, "execution_id": execution.id},
                "errors": [],
                "iteration_count": 0,
                "start_time": time.time()
            }
            
            # 执行工作流
            result = await graph.ainvoke(initial_state)
            
            # 更新执行结果
            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.now()
            execution.output_data = result.get("data", {})
            
            self._log_execution(execution, "Workflow execution completed successfully")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()
            execution.error_message = str(e)
            
            logger.error(f"Workflow execution failed: {e}")
            self._log_execution(execution, f"Workflow execution failed: {e}")
        
        # 保存执行记录到数据库
        self._save_execution_to_db(execution)
    
    def _build_langgraph(self, workflow: WorkflowDefinition) -> StateGraph:
        """构建LangGraph"""
        graph = StateGraph(WorkflowState)
        
        # 添加节点
        for node in workflow.nodes:
            if node.node_type == NodeType.INPUT:
                graph.add_node(node.id, self._create_input_node(node))
            elif node.node_type == NodeType.PROCESSING:
                graph.add_node(node.id, self._create_processing_node(node))
            elif node.node_type == NodeType.DECISION:
                graph.add_node(node.id, self._create_decision_node(node))
            elif node.node_type == NodeType.OUTPUT:
                graph.add_node(node.id, self._create_output_node(node))
            elif node.node_type == NodeType.CONDITION:
                graph.add_node(node.id, self._create_condition_node(node))
        
        # 添加连接
        for edge in workflow.edges:
            if edge.condition:
                graph.add_conditional_edges(
                    edge.from_node,
                    self._create_condition_function(edge.condition),
                    {True: edge.to_node, False: END}
                )
            else:
                graph.add_edge(edge.from_node, edge.to_node)
        
        # 设置入口点
        start_node = self._find_start_node(workflow)
        if start_node:
            graph.set_entry_point(start_node)
        
        return graph.compile()
    
    def _create_input_node(self, node: WorkflowNode) -> Callable:
        """创建输入节点"""
        async def input_node(state: WorkflowState) -> WorkflowState:
            try:
                # 处理输入数据
                input_config = node.config
                input_type = input_config.get("type", "text")
                
                if input_type == "file" and "path" in state["data"]:
                    result = await self.processor.process_file(
                        state["data"]["path"], 
                        input_config.get("task", "analyze")
                    )
                    state["data"]["processed_input"] = result
                elif input_type == "json":
                    result = await self.processor.process_json(
                        state["data"], 
                        input_config.get("task", "validate")
                    )
                    state["data"]["processed_input"] = result
                else:
                    # 文本输入
                    content = state["data"].get("content", "")
                    result = await self.processor.process_text(
                        content, 
                        input_config.get("task", "analyze")
                    )
                    state["data"]["processed_input"] = result
                
                state["current_node"] = node.id
                return state
                
            except Exception as e:
                state["errors"].append(f"Input node {node.id} failed: {e}")
                return state
        
        return input_node
    
    def _create_processing_node(self, node: WorkflowNode) -> Callable:
        """创建处理节点"""
        async def processing_node(state: WorkflowState) -> WorkflowState:
            try:
                processing_type = node.config.get("type", "transform")
                input_data = state["data"]
                
                if processing_type == "transform":
                    # 数据转换
                    result = self._transform_data(input_data, node.config)
                elif processing_type == "validate":
                    # 数据验证
                    result = self._validate_data(input_data, node.config)
                elif processing_type == "enrich":
                    # 数据丰富
                    result = await self._enrich_data(input_data, node.config)
                else:
                    result = input_data
                
                state["data"][f"{node.id}_output"] = result
                state["current_node"] = node.id
                return state
                
            except Exception as e:
                state["errors"].append(f"Processing node {node.id} failed: {e}")
                return state
        
        return processing_node
    
    def _create_decision_node(self, node: WorkflowNode) -> Callable:
        """创建决策节点"""
        async def decision_node(state: WorkflowState) -> WorkflowState:
            try:
                question = node.config.get("question", "What should be the next action?")
                context = {
                    "current_data": state["data"],
                    "execution_context": state["context"],
                    "node_config": node.config
                }
                
                decision = await self.decision_engine.make_decision(context, question)
                
                state["data"][f"{node.id}_decision"] = decision
                state["current_node"] = node.id
                
                # 根据决策结果设置下一个节点
                if decision.get("decision", {}).get("next_node"):
                    state["context"]["next_node"] = decision["decision"]["next_node"]
                
                return state
                
            except Exception as e:
                state["errors"].append(f"Decision node {node.id} failed: {e}")
                return state
        
        return decision_node
    
    def _create_output_node(self, node: WorkflowNode) -> Callable:
        """创建输出节点"""
        async def output_node(state: WorkflowState) -> WorkflowState:
            try:
                output_config = node.config
                output_format = output_config.get("format", "json")
                
                if output_format == "json":
                    state["data"]["final_output"] = state["data"]
                elif output_format == "summary":
                    # 生成摘要
                    summary = self._generate_summary(state["data"])
                    state["data"]["final_output"] = {"summary": summary}
                
                state["current_node"] = node.id
                return state
                
            except Exception as e:
                state["errors"].append(f"Output node {node.id} failed: {e}")
                return state
        
        return output_node
    
    def _create_condition_node(self, node: WorkflowNode) -> Callable:
        """创建条件节点"""
        async def condition_node(state: WorkflowState) -> WorkflowState:
            try:
                conditions = node.config.get("conditions", {})
                
                for condition_name, condition_expr in conditions.items():
                    result = self.decision_engine._safe_eval(condition_expr, state["data"])
                    state["data"][f"{node.id}_{condition_name}"] = result
                
                state["current_node"] = node.id
                return state
                
            except Exception as e:
                state["errors"].append(f"Condition node {node.id} failed: {e}")
                return state
        
        return condition_node
    
    def _create_condition_function(self, condition: str) -> Callable:
        """创建条件判断函数"""
        def condition_func(state: WorkflowState) -> bool:
            try:
                return self.decision_engine._safe_eval(condition, state["data"])
            except:
                return False
        
        return condition_func
    
    def _find_start_node(self, workflow: WorkflowDefinition) -> Optional[str]:
        """查找开始节点"""
        # 查找没有输入的节点作为开始节点
        all_to_nodes = {edge.to_node for edge in workflow.edges}
        for node in workflow.nodes:
            if node.id not in all_to_nodes:
                return node.id
        
        # 如果没有找到，返回第一个输入节点
        for node in workflow.nodes:
            if node.node_type == NodeType.INPUT:
                return node.id
        
        # 最后返回第一个节点
        return workflow.nodes[0].id if workflow.nodes else None
    
    def _transform_data(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """数据转换"""
        # 这里实现数据转换逻辑
        transformed = data.copy()
        
        # 示例转换规则
        rules = config.get("rules", [])
        for rule in rules:
            if rule["type"] == "rename":
                old_key = rule["from"]
                new_key = rule["to"]
                if old_key in transformed:
                    transformed[new_key] = transformed.pop(old_key)
            elif rule["type"] == "calculate":
                # 简单计算
                expression = rule["expression"]
                try:
                    result = eval(expression, {"__builtins__": {}}, transformed)
                    transformed[rule["target"]] = result
                except:
                    pass
        
        return transformed
    
    def _validate_data(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """数据验证"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        required_fields = config.get("required_fields", [])
        for field in required_fields:
            if field not in data:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required field: {field}")
        
        type_checks = config.get("type_checks", {})
        for field, expected_type in type_checks.items():
            if field in data:
                actual_type = type(data[field]).__name__
                if actual_type != expected_type:
                    validation_result["warnings"].append(
                        f"Field {field} expected {expected_type}, got {actual_type}"
                    )
        
        return validation_result
    
    async def _enrich_data(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """数据丰富"""
        enriched = data.copy()
        
        # 添加时间戳
        if config.get("add_timestamp", False):
            enriched["timestamp"] = datetime.now().isoformat()
        
        # 添加ID
        if config.get("add_id", False):
            enriched["id"] = str(uuid.uuid4())
        
        # 计算统计信息
        if config.get("add_stats", False):
            stats = {
                "field_count": len(enriched),
                "text_fields": len([v for v in enriched.values() if isinstance(v, str)]),
                "numeric_fields": len([v for v in enriched.values() if isinstance(v, (int, float))])
            }
            enriched["stats"] = stats
        
        return enriched
    
    def _generate_summary(self, data: Dict[str, Any]) -> str:
        """生成数据摘要"""
        summary_parts = []
        
        summary_parts.append(f"Data contains {len(data)} fields")
        
        for key, value in data.items():
            if isinstance(value, dict):
                summary_parts.append(f"{key}: object with {len(value)} properties")
            elif isinstance(value, list):
                summary_parts.append(f"{key}: array with {len(value)} items")
            elif isinstance(value, str):
                summary_parts.append(f"{key}: text ({len(value)} characters)")
            else:
                summary_parts.append(f"{key}: {type(value).__name__} value")
        
        return "; ".join(summary_parts)
    
    def _log_execution(self, execution: WorkflowExecution, message: str):
        """记录执行日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "status": execution.status.value
        }
        execution.execution_log.append(log_entry)
        logger.info(f"[{execution.id}] {message}")
    
    def _save_execution_to_db(self, execution: WorkflowExecution):
        """保存执行记录到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO executions 
            (id, workflow_id, status, input_data, output_data, execution_log, 
             start_time, end_time, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            execution.id,
            execution.workflow_id,
            execution.status.value,
            json.dumps(execution.input_data),
            json.dumps(execution.output_data),
            json.dumps(execution.execution_log),
            execution.start_time,
            execution.end_time,
            execution.error_message
        ))
        
        conn.commit()
        conn.close()
    
    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """获取执行状态"""
        return self.executions.get(execution_id)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """取消执行"""
        execution = self.executions.get(execution_id)
        if execution and execution.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = datetime.now()
            self._log_execution(execution, "Execution cancelled by user")
            return True
        return False
    
    async def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """获取工作流指标"""
        executions = [e for e in self.executions.values() if e.workflow_id == workflow_id]
        
        if not executions:
            return {"error": "No executions found for this workflow"}
        
        total_executions = len(executions)
        completed = len([e for e in executions if e.status == WorkflowStatus.COMPLETED])
        failed = len([e for e in executions if e.status == WorkflowStatus.FAILED])
        
        avg_duration = 0
        completed_executions = [e for e in executions 
                              if e.status == WorkflowStatus.COMPLETED and e.end_time]
        if completed_executions:
            total_duration = sum(
                (e.end_time - e.start_time).total_seconds() 
                for e in completed_executions
            )
            avg_duration = total_duration / len(completed_executions)
        
        return {
            "workflow_id": workflow_id,
            "total_executions": total_executions,
            "completed": completed,
            "failed": failed,
            "success_rate": (completed / total_executions) * 100 if total_executions > 0 else 0,
            "avg_duration_seconds": avg_duration,
            "last_execution": max(executions, key=lambda x: x.start_time).start_time.isoformat()
        }


class SmartWorkflowDemo:
    """智能工作流引擎演示"""
    
    def __init__(self):
        self.engine = WorkflowEngine()
        self._setup_decision_rules()
    
    def _setup_decision_rules(self):
        """设置决策规则"""
        # 添加一些示例决策规则
        self.engine.decision_engine.add_rule(
            name="high_priority_rule",
            condition="priority > 8",
            action="expedite",
            priority=10
        )
        
        self.engine.decision_engine.add_rule(
            name="error_handling_rule",
            condition="len(errors) > 0",
            action="retry_or_abort",
            priority=9
        )
        
        self.engine.decision_engine.add_rule(
            name="data_size_rule",
            condition="data_size > 1000000",
            action="batch_process",
            priority=5
        )
    
    async def run_demo(self):
        """运行演示"""
        print("🤖 智能工作流引擎演示")
        print("=" * 60)
        
        try:
            await self._demonstrate_workflow_creation()
            await self._demonstrate_multimodal_processing()
            await self._demonstrate_decision_engine()
            await self._demonstrate_workflow_execution()
            await self._demonstrate_monitoring_and_metrics()
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
    
    async def _demonstrate_workflow_creation(self):
        """演示工作流创建"""
        print("\n📋 工作流创建演示")
        print("-" * 40)
        
        # 创建数据处理工作流
        nodes = [
            WorkflowNode(
                id="input_1",
                name="数据输入",
                node_type=NodeType.INPUT,
                config={"type": "json", "task": "validate"}
            ),
            WorkflowNode(
                id="process_1",
                name="数据处理",
                node_type=NodeType.PROCESSING,
                config={
                    "type": "transform",
                    "rules": [
                        {"type": "rename", "from": "old_name", "to": "new_name"}
                    ]
                }
            ),
            WorkflowNode(
                id="decision_1",
                name="智能决策",
                node_type=NodeType.DECISION,
                config={"question": "Should we proceed with further processing?"}
            ),
            WorkflowNode(
                id="output_1",
                name="结果输出",
                node_type=NodeType.OUTPUT,
                config={"format": "summary"}
            )
        ]
        
        edges = [
            WorkflowEdge("input_1", "process_1"),
            WorkflowEdge("process_1", "decision_1"),
            WorkflowEdge("decision_1", "output_1")
        ]
        
        workflow = WorkflowDefinition(
            id="demo_workflow_1",
            name="数据处理工作流",
            description="演示数据输入、处理、决策和输出的完整流程",
            version="1.0.0",
            nodes=nodes,
            edges=edges
        )
        
        workflow_id = await self.engine.create_workflow(workflow)
        print(f"✅ 创建工作流成功: {workflow.name} (ID: {workflow_id})")
        print(f"   节点数: {len(nodes)}")
        print(f"   连接数: {len(edges)}")
        
        return workflow_id
    
    async def _demonstrate_multimodal_processing(self):
        """演示多模态处理"""
        print("\n🎭 多模态数据处理演示")
        print("-" * 40)
        
        # 文本处理
        text_result = await self.engine.processor.process_text(
            "This is a sample text for analysis. It contains important information about our product.",
            "analyze"
        )
        print(f"📝 文本处理结果: {text_result['result'][:100]}...")
        
        # JSON处理
        json_data = {
            "productName": "Smart Widget",
            "price": 99.99,
            "category": "Electronics",
            "features": ["WiFi", "Bluetooth", "Voice Control"],
            "availability": True
        }
        json_result = await self.engine.processor.process_json(json_data, "transform")
        print(f"📊 JSON处理结果: {json_result['result']}")
        
        # 文件处理（如果存在README.md）
        readme_path = "README.md"
        if os.path.exists(readme_path):
            file_result = await self.engine.processor.process_file(readme_path, "analyze")
            print(f"📁 文件处理结果: {file_result['result']['size']} bytes")
    
    async def _demonstrate_decision_engine(self):
        """演示决策引擎"""
        print("\n🧠 决策引擎演示")
        print("-" * 40)
        
        # 测试不同的决策场景
        test_contexts = [
            {
                "name": "高优先级任务",
                "context": {"priority": 9, "data_size": 500, "errors": []},
                "question": "How should we handle this high priority task?"
            },
            {
                "name": "错误处理场景",
                "context": {"priority": 5, "data_size": 1000, "errors": ["connection timeout"]},
                "question": "What should we do about the errors?"
            },
            {
                "name": "大数据处理",
                "context": {"priority": 3, "data_size": 2000000, "errors": []},
                "question": "How to handle this large dataset?"
            }
        ]
        
        for test in test_contexts:
            decision = await self.engine.decision_engine.make_decision(
                test["context"], test["question"]
            )
            print(f"🎯 {test['name']}:")
            print(f"   决策类型: {decision['type']}")
            print(f"   置信度: {decision['confidence']}")
            print(f"   推理: {decision['reasoning']}")
            print()
    
    async def _demonstrate_workflow_execution(self):
        """演示工作流执行"""
        print("\n🚀 工作流执行演示")
        print("-" * 40)
        
        # 创建简单的工作流
        workflow_id = await self._create_demo_workflow()
        
        # 执行工作流
        input_data = {
            "content": "This is input data for processing",
            "old_name": "legacy_field",
            "priority": 7,
            "data_size": 1500
        }
        
        execution_id = await self.engine.execute_workflow(workflow_id, input_data)
        print(f"🎬 开始执行工作流: {execution_id}")
        
        # 等待执行完成
        max_wait = 30  # 最大等待30秒
        wait_count = 0
        while wait_count < max_wait:
            execution = await self.engine.get_execution_status(execution_id)
            if execution and execution.status in [
                WorkflowStatus.COMPLETED, 
                WorkflowStatus.FAILED, 
                WorkflowStatus.CANCELLED
            ]:
                break
            await asyncio.sleep(1)
            wait_count += 1
        
        # 显示执行结果
        execution = await self.engine.get_execution_status(execution_id)
        if execution:
            print(f"📊 执行状态: {execution.status.value}")
            if execution.status == WorkflowStatus.COMPLETED:
                print(f"✅ 执行成功!")
                print(f"   输出数据键数: {len(execution.output_data)}")
                print(f"   执行日志条数: {len(execution.execution_log)}")
            elif execution.status == WorkflowStatus.FAILED:
                print(f"❌ 执行失败: {execution.error_message}")
        
        return execution_id
    
    async def _create_demo_workflow(self) -> str:
        """创建演示工作流"""
        nodes = [
            WorkflowNode(
                id="input_demo",
                name="输入节点",
                node_type=NodeType.INPUT,
                config={"type": "text", "task": "analyze"}
            ),
            WorkflowNode(
                id="process_demo",
                name="处理节点",
                node_type=NodeType.PROCESSING,
                config={
                    "type": "enrich",
                    "add_timestamp": True,
                    "add_stats": True
                }
            ),
            WorkflowNode(
                id="output_demo",
                name="输出节点",
                node_type=NodeType.OUTPUT,
                config={"format": "summary"}
            )
        ]
        
        edges = [
            WorkflowEdge("input_demo", "process_demo"),
            WorkflowEdge("process_demo", "output_demo")
        ]
        
        workflow = WorkflowDefinition(
            id="simple_demo_workflow",
            name="简单演示工作流",
            description="用于演示的简单工作流",
            version="1.0.0",
            nodes=nodes,
            edges=edges
        )
        
        return await self.engine.create_workflow(workflow)
    
    async def _demonstrate_monitoring_and_metrics(self):
        """演示监控和指标"""
        print("\n📈 监控和指标演示")
        print("-" * 40)
        
        # 获取工作流指标
        for workflow_id in self.engine.workflows.keys():
            metrics = await self.engine.get_workflow_metrics(workflow_id)
            
            if "error" not in metrics:
                workflow_name = self.engine.workflows[workflow_id].name
                print(f"📊 工作流: {workflow_name}")
                print(f"   总执行次数: {metrics['total_executions']}")
                print(f"   成功次数: {metrics['completed']}")
                print(f"   失败次数: {metrics['failed']}")
                print(f"   成功率: {metrics['success_rate']:.1f}%")
                if metrics['avg_duration_seconds'] > 0:
                    print(f"   平均执行时间: {metrics['avg_duration_seconds']:.2f}秒")
                print()
        
        # 显示系统概览
        total_workflows = len(self.engine.workflows)
        total_executions = len(self.engine.executions)
        
        print(f"🎯 系统概览:")
        print(f"   工作流总数: {total_workflows}")
        print(f"   执行总数: {total_executions}")
        
        if self.engine.executions:
            running_count = len([e for e in self.engine.executions.values() 
                               if e.status == WorkflowStatus.RUNNING])
            completed_count = len([e for e in self.engine.executions.values() 
                                 if e.status == WorkflowStatus.COMPLETED])
            print(f"   运行中: {running_count}")
            print(f"   已完成: {completed_count}")


async def demo_workflow_engine():
    """智能工作流引擎演示函数（供start.py调用）"""
    demo = SmartWorkflowDemo()
    await demo.run_demo()


async def main():
    """主函数"""
    await demo_workflow_engine()


if __name__ == "__main__":
    asyncio.run(main())
