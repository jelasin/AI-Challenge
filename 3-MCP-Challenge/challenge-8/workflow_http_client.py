#!/usr/bin/env python3
"""
工作流引擎 HTTP 客户端
Challenge 8: 智能工作流编排与执行系统

本客户端通过 HTTP API 与工作流引擎服务器通信，提供：
1. 工作流的创建、执行和管理
2. 复杂工作流模板的使用
3. 实时状态监控和报告生成
4. 多种工作流执行模式
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
import aiohttp
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowHttpClient:
    """工作流引擎 HTTP 客户端"""
    
    def __init__(self, server_url: str = "http://localhost:8009"):
        """初始化客户端"""
        self.server_url = server_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def initialize(self):
        """初始化客户端连接"""
        try:
            # 创建会话
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # 测试服务器连接
            async with self.session.get(f"{self.server_url}/") as response:
                if response.status == 200:
                    server_info = await response.json()
                    return True
                else:
                    logger.error(f"服务器响应错误: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"连接工作流引擎服务器失败: {e}")
            return False
    
    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """通用请求方法"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.server_url}{path}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.content_type == 'application/json':
                    return await response.json()
                else:
                    text = await response.text()
                    return {"success": False, "error": f"Unexpected response: {text}"}
        except Exception as e:
            logger.error(f"请求失败 {method} {url}: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_workflow(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """创建新工作流"""
        return await self._request("POST", "/workflows", json={"workflow_definition": workflow_definition})
    
    async def create_workflow_from_template(self, template_id: str, customization: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """基于模板创建工作流"""
        return await self._request("POST", f"/templates/{template_id}/create", json=customization)
    
    async def execute_workflow(self, workflow_id: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行工作流"""
        return await self._request("POST", f"/workflows/{workflow_id}/execute", json=variables)
    
    async def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """获取工作流执行状态"""
        return await self._request("GET", f"/executions/{execution_id}")
    
    async def list_workflows(self) -> Dict[str, Any]:
        """列出所有工作流"""
        return await self._request("GET", "/workflows")
    
    async def list_executions(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """列出执行记录"""
        params = {}
        if workflow_id:
            params["workflow_id"] = workflow_id
        return await self._request("GET", "/executions", params=params)
    
    async def pause_workflow(self, execution_id: str) -> Dict[str, Any]:
        """暂停工作流"""
        return await self._request("POST", f"/executions/{execution_id}/pause")
    
    async def resume_workflow(self, execution_id: str) -> Dict[str, Any]:
        """恢复工作流"""
        return await self._request("POST", f"/executions/{execution_id}/resume")
    
    async def cancel_workflow(self, execution_id: str) -> Dict[str, Any]:
        """取消工作流"""
        return await self._request("POST", f"/executions/{execution_id}/cancel")
    
    async def generate_execution_report(self, execution_id: str) -> Dict[str, Any]:
        """生成执行报告"""
        return await self._request("GET", f"/executions/{execution_id}/report")
    
    async def get_workflow_templates(self) -> Dict[str, Any]:
        """获取工作流模板"""
        return await self._request("GET", "/templates")
    
    async def validate_workflow_definition(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """验证工作流定义"""
        return await self._request("POST", "/workflows/validate", json={"workflow_definition": workflow_definition})
    
    async def monitor_workflow_execution(self, execution_id: str, check_interval: int = 5):
        """监控工作流执行，实时输出状态"""
        print(f"\n🔄 开始监控工作流执行: {execution_id}")
        print("=" * 60)
        
        previous_status = None
        completed_tasks = set()
        
        while True:
            try:
                status_result = await self.get_workflow_status(execution_id)
                
                if not status_result.get("success"):
                    print(f"❌ 获取状态失败: {status_result.get('error', '未知错误')}")
                    break
                
                execution = status_result["execution"]
                current_status = execution["status"]
                
                # 状态变化时输出
                if current_status != previous_status:
                    print(f"📊 执行状态: {current_status}")
                    previous_status = current_status
                
                # 输出新完成的任务
                for task_id, task_result in execution["task_results"].items():
                    if task_id not in completed_tasks and task_result["status"] in ["completed", "failed", "skipped"]:
                        status_emoji = {
                            "completed": "✅",
                            "failed": "❌", 
                            "skipped": "⏭️"
                        }[task_result["status"]]
                        
                        print(f"  {status_emoji} 任务 {task_id}: {task_result['status']}")
                        
                        if task_result["status"] == "failed" and task_result.get("error"):
                            print(f"    错误: {task_result['error']}")
                        
                        if task_result.get("execution_time"):
                            print(f"    执行时间: {task_result['execution_time']:.2f}秒")
                        
                        completed_tasks.add(task_id)
                
                # 检查是否完成
                if current_status in ["completed", "failed", "cancelled"]:
                    print(f"\n🏁 工作流执行结束: {current_status}")
                    
                    if execution.get("total_execution_time"):
                        print(f"⏱️ 总执行时间: {execution['total_execution_time']:.2f}秒")
                    
                    if execution.get("error_message"):
                        print(f"❌ 错误信息: {execution['error_message']}")
                    
                    # 生成并显示执行报告
                    print("\n📋 正在生成执行报告...")
                    report_result = await self.generate_execution_report(execution_id)
                    
                    if report_result.get("success"):
                        report = report_result["report"]
                        self._display_execution_summary(report)
                    
                    break
                
                await asyncio.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("\n⏹️ 监控已停止")
                break
            except Exception as e:
                print(f"❌ 监控过程中发生错误: {e}")
                break
    
    def _display_execution_summary(self, report: Dict[str, Any]):
        """显示执行摘要"""
        print("\n" + "=" * 60)
        print("📊 执行报告摘要")
        print("=" * 60)
        
        print(f"🆔 执行ID: {report['execution_id']}")
        print(f"📝 工作流: {report['workflow_name']}")
        print(f"⏱️ 开始时间: {report['start_time']}")
        print(f"⏱️ 结束时间: {report['end_time']}")
        print(f"⏱️ 总耗时: {report['total_execution_time']:.2f}秒")
        
        stats = report['task_statistics']
        print(f"\n📈 任务统计:")
        print(f"  总任务数: {stats['total']}")
        print(f"  成功任务: {stats['completed']}")
        print(f"  失败任务: {stats['failed']}")
        print(f"  成功率: {stats['success_rate']:.1%}")
        
        if report.get('error_message'):
            print(f"\n❌ 错误信息: {report['error_message']}")
        
        print("\n" + "=" * 60)
    
    async def cleanup(self):
        """清理资源"""
        if self.session:
            await self.session.close()

# 示例工作流定义
SAMPLE_WORKFLOWS = {
    "simple_file_processing": {
        "name": "简单文件处理工作流",
        "description": "演示基础文件操作：读取、处理、保存",
        "variables": {
            "source_file": {"type": "string", "default": "sample_data.txt"},
            "output_format": {"type": "string", "default": "processed"}
        },
        "steps": [
            {
                "id": "create_source",
                "name": "创建源文件",
                "type": "function",
                "action": "write_file",
                "parameters": {
                    "path": "{{source_file}}",
                    "content": "This is sample data for processing: 123, 456, 789"
                }
            },
            {
                "id": "read_source",
                "name": "读取源文件",
                "type": "function",
                "action": "read_file",
                "depends_on": ["create_source"],
                "parameters": {
                    "path": "{{source_file}}"
                }
            },
            {
                "id": "create_output_dir",
                "name": "创建输出目录",
                "type": "function",
                "action": "create_directory",
                "depends_on": ["read_source"],
                "parameters": {
                    "path": "output"
                }
            },
            {
                "id": "save_result",
                "name": "保存处理结果",
                "type": "function",
                "action": "write_file",
                "depends_on": ["create_output_dir"],
                "parameters": {
                    "path": "output/{{output_format}}_data.txt",
                    "content": "文件处理完成 - 源文件: {{source_file}}, 格式: {{output_format}}"
                }
            }
        ]
    },
    
    "math_calculation_workflow": {
        "name": "数学计算工作流",
        "description": "演示数学运算和结果处理",
        "variables": {
            "number_a": {"type": "float", "default": 10.5},
            "number_b": {"type": "float", "default": 3.2},
            "operation": {"type": "string", "default": "multiply"}
        },
        "steps": [
            {
                "id": "perform_calculation",
                "name": "执行数学计算",
                "type": "function",
                "action": "{{operation}}",
                "parameters": {
                    "a": "{{number_a}}",
                    "b": "{{number_b}}"
                }
            },
            {
                "id": "format_result",
                "name": "格式化计算结果",
                "type": "function",
                "action": "format_prompt",
                "depends_on": ["perform_calculation"],
                "parameters": {
                    "template": "计算结果: {a} {op} {b} = {result}",
                    "variables": {
                        "a": "{{number_a}}",
                        "op": "{{operation}}",
                        "b": "{{number_b}}",
                        "result": "计算完成"
                    }
                }
            },
            {
                "id": "save_calculation",
                "name": "保存计算结果到文件",
                "type": "function",
                "action": "write_file",
                "depends_on": ["format_result"],
                "parameters": {
                    "path": "calculation_result.txt",
                    "content": "数学运算: {{number_a}} {{operation}} {{number_b}} = 结果已计算"
                }
            }
        ]
    },
    
    "database_workflow": {
        "name": "数据库操作工作流",
        "description": "演示SQLite数据库的创建、插入和查询操作",
        "variables": {
            "db_name": {"type": "string", "default": "workflow_test.db"},
            "table_name": {"type": "string", "default": "workflow_results"}
        },
        "steps": [
            {
                "id": "create_table",
                "name": "创建数据表",
                "type": "function",
                "action": "create_table",
                "parameters": {
                    "table_name": "{{table_name}}"
                }
            },
            {
                "id": "insert_data",
                "name": "插入测试数据",
                "type": "function",
                "action": "insert_data",
                "depends_on": ["create_table"],
                "parameters": {
                    "table_name": "{{table_name}}",
                    "data": {"name": "workflow_test", "value": "42.5"}
                }
            },
            {
                "id": "query_data",
                "name": "查询数据",
                "type": "function",
                "action": "query_data",
                "depends_on": ["insert_data"],
                "parameters": {
                    "table_name": "{{table_name}}"
                }
            },
            {
                "id": "save_query_result",
                "name": "保存查询结果",
                "type": "function",
                "action": "write_file",
                "depends_on": ["query_data"],
                "parameters": {
                    "path": "db_query_result.txt",
                    "content": "数据库查询完成 - 表: {{table_name}}, 数据库: {{db_name}}"
                }
            }
        ]
    },
    
    "comprehensive_mcp_workflow": {
        "name": "综合MCP工具演示工作流",
        "description": "展示所有可用MCP工具（文件、数学、数据库、提示）的综合使用",
        "variables": {
            "input_value": {"type": "float", "default": 50.0},
            "multiplier": {"type": "float", "default": 2.5}
        },
        "steps": [
            {
                "id": "create_input_file",
                "name": "创建输入数据文件",
                "type": "function",
                "action": "write_file",
                "parameters": {
                    "path": "input_data.txt",
                    "content": "输入值: {{input_value}}, 乘数: {{multiplier}}"
                }
            },
            {
                "id": "perform_math_operation",
                "name": "执行数学运算",
                "type": "function",
                "action": "multiply",
                "depends_on": ["create_input_file"],
                "parameters": {
                    "a": "{{input_value}}",
                    "b": "{{multiplier}}"
                }
            },
            {
                "id": "setup_database",
                "name": "设置数据库表",
                "type": "function",
                "action": "create_table",
                "depends_on": ["perform_math_operation"],
                "parameters": {
                    "table_name": "mcp_demo_results"
                }
            },
            {
                "id": "save_to_database",
                "name": "保存结果到数据库",
                "type": "function",
                "action": "insert_data",
                "depends_on": ["setup_database"],
                "parameters": {
                    "table_name": "mcp_demo_results",
                    "data": {"name": "comprehensive_test", "value": "calculation_complete"}
                }
            },
            {
                "id": "format_summary_prompt",
                "name": "格式化摘要报告",
                "type": "function",
                "action": "format_prompt",
                "depends_on": ["save_to_database"],
                "parameters": {
                    "template": "综合工作流完成报告\\n输入值: {input}\\n乘数: {mult}\\n状态: 成功完成所有MCP工具调用",
                    "variables": {
                        "input": "{{input_value}}",
                        "mult": "{{multiplier}}"
                    }
                }
            },
            {
                "id": "save_final_report",
                "name": "保存最终报告文件",
                "type": "function",
                "action": "write_file",
                "depends_on": ["format_summary_prompt"],
                "parameters": {
                    "path": "comprehensive_workflow_report.txt",
                    "content": "综合MCP工作流执行完成\\n- 文件操作: ✓\\n- 数学运算: ✓\\n- 数据库操作: ✓\\n- 提示处理: ✓\\n所有工具测试成功！"
                }
            },
            {
                "id": "verify_database_data",
                "name": "验证数据库数据",
                "type": "function",
                "action": "query_data",
                "depends_on": ["save_final_report"],
                "parameters": {
                    "table_name": "mcp_demo_results"
                }
            }
        ]
    }
}

def print_welcome():
    """打印欢迎信息"""
    print("=" * 80)
    print("🚀 Challenge 8: 智能工作流编排与执行系统")
    print("=" * 80)
    print()
    print("欢迎来到最后的挑战！这是一个综合性的工作流引擎系统，")
    print("它集成了之前所有挑战的概念和技术，提供了强大的工作流编排能力。")
    print()
    print("🎯 挑战目标:")
    print("  • 掌握复杂工作流的设计和实现")
    print("  • 理解任务依赖管理和并行执行")
    print("  • 学习条件分支和循环控制")
    print("  • 掌握错误处理和重试机制")
    print("  • 实现工作流状态监控和报告")
    print()
    print("💡 核心特性:")
    print("  • 基于YAML/JSON的工作流定义")
    print("  • 支持复杂的任务依赖图")
    print("  • 并行任务执行和条件分支")
    print("  • 智能错误处理和故障恢复")
    print("  • 实时状态监控和性能分析")
    print("  • 工作流模板和继承机制")
    print()
    print("=" * 80)

if __name__ == "__main__":
    print("🔧 这是工作流HTTP客户端库文件")
    print("请运行 main.py 开始Challenge 8")
