#!/usr/bin/env python3
"""
Challenge 8: 智能工作流编排与执行系统
终极挑战 - 工作流引擎的完整实现与应用

本挑战是MCP系列的最终挑战，集成了所有之前学到的概念和技术，
提供了一个功能完整的智能工作流编排与执行系统。

挑战特色：
1. 复杂的工作流定义和管理（YAML/JSON格式）
2. 多步骤任务执行与协调
3. 条件分支和循环控制
4. 智能错误处理和重试机制
5. 实时工作流状态监控
6. 数据传递和变量管理
7. 并发任务执行
8. 工作流模板和继承
9. 事件驱动触发器
10. 性能指标和报告生成
"""

import asyncio
import json
import subprocess
import time
import signal
import os
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from workflow_http_client import WorkflowHttpClient, SAMPLE_WORKFLOWS, print_welcome

async def check_workflow_server():
    """检查工作流 HTTP 服务器是否运行"""
    print("\n🔍 检查工作流引擎服务器状态...")
    
    client = WorkflowHttpClient()
    try:
        success = await client.initialize()
        if success:
            print("✅ 工作流引擎服务器已在运行")
            await client.cleanup()
            return True
        await client.cleanup()
    except Exception as e:
        print(f"❌ 无法连接到工作流引擎服务器: {e}")
    
    print("❌ 工作流引擎服务器未运行")
    print("   请先独立启动服务器:")
    print("   python 3-MCP-Challenge/mcp_servers/workflow_engine_http.py")
    return False

async def check_server_connection():
    """检查服务器连接"""
    print("\n🔗 检查服务器连接状态...")
    
    client = WorkflowHttpClient()
    try:
        success = await client.initialize()
        if success:
            print("✅ 成功连接到工作流引擎服务器")
            return True
        else:
            print("❌ 无法连接到工作流引擎服务器")
            return False
    except Exception as e:
        print(f"❌ 连接检查失败: {e}")
        return False
    finally:
        await client.cleanup()

async def demonstrate_workflow_creation():
    """演示工作流创建功能"""
    print("\n🔧 演示 1: 工作流创建和验证")
    print("-" * 50)
    
    client = WorkflowHttpClient()
    await client.initialize()
    
    try:
        # 首先验证工作流定义
        workflow_def = SAMPLE_WORKFLOWS["simple_file_processing"]
        print(f"📋 验证工作流定义: {workflow_def['name']}")
        
        validation_result = await client.validate_workflow_definition(workflow_def)
        if validation_result["success"] and validation_result["is_valid"]:
            print("✅ 工作流定义有效")
            
            # 创建工作流
            create_result = await client.create_workflow(workflow_def)
            if create_result["success"]:
                workflow_id = create_result["workflow_id"]
                print(f"✅ 工作流创建成功，ID: {workflow_id}")
                return workflow_id
            else:
                print(f"❌ 工作流创建失败: {create_result.get('message', '未知错误')}")
        else:
            print("❌ 工作流定义无效")
            
    except Exception as e:
        print(f"❌ 演示过程发生错误: {e}")
    finally:
        await client.cleanup()
    
    return None

async def demonstrate_template_usage():
    """演示模板使用功能"""
    print("\n🎨 演示 2: 工作流模板使用")
    print("-" * 50)
    
    client = WorkflowHttpClient()
    await client.initialize()
    
    try:
        # 获取可用模板
        templates_result = await client.get_workflow_templates()
        if templates_result["success"]:
            templates = templates_result["templates"]
            print(f"📂 可用模板数量: {len(templates)}")
            
            for template_id, template_info in templates.items():
                print(f"  📋 {template_id}: {template_info['name']}")
            
            # 使用计算工作流模板创建工作流
            print(f"\n🔄 使用模板创建工作流: calculation_workflow")
            customization = {
                "name": "自定义数学计算工作流",
                "variables": {
                    "number_a": {"default": 25.5},
                    "number_b": {"default": 4.2},
                    "operation": {"default": "add"}
                }
            }
            
            result = await client.create_workflow_from_template(
                "calculation_workflow", 
                customization
            )
            
            if result["success"]:
                workflow_id = result["workflow_id"]
                print(f"✅ 基于模板创建工作流成功，ID: {workflow_id}")
                return workflow_id
            else:
                print(f"❌ 基于模板创建失败: {result.get('message', '未知错误')}")
        else:
            print("❌ 获取模板失败")
            
    except Exception as e:
        print(f"❌ 演示过程发生错误: {e}")
    finally:
        await client.cleanup()
    
    return None

async def demonstrate_workflow_execution():
    """演示工作流执行功能"""
    print("\n⚡ 演示 3: 工作流执行与监控")
    print("-" * 50)
    
    client = WorkflowHttpClient()
    await client.initialize()
    
    try:
        # 创建一个用于演示的工作流 - 使用新的综合MCP工具演示
        demo_workflow = SAMPLE_WORKFLOWS["comprehensive_mcp_workflow"]
        create_result = await client.create_workflow(demo_workflow)
        
        if create_result["success"]:
            workflow_id = create_result["workflow_id"]
            print(f"✅ 创建演示工作流成功，ID: {workflow_id}")
            
            # 执行工作流
            execution_variables = {
                "input_value": 75.0,
                "multiplier": 3.2
            }
            
            print(f"🚀 启动工作流执行...")
            execute_result = await client.execute_workflow(workflow_id, execution_variables)
            
            if execute_result["success"]:
                execution_id = execute_result["execution_id"]
                print(f"✅ 工作流开始执行，执行ID: {execution_id}")
                
                # 监控执行过程
                await client.monitor_workflow_execution(execution_id, check_interval=3)
                
                return execution_id
            else:
                print(f"❌ 工作流执行失败: {execute_result.get('message', '未知错误')}")
        else:
            print(f"❌ 创建工作流失败: {create_result.get('message', '未知错误')}")
            
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了执行监控")
    except Exception as e:
        print(f"❌ 演示过程发生错误: {e}")
    finally:
        await client.cleanup()
    
    return None

async def demonstrate_error_handling():
    """演示错误处理和重试机制"""
    print("\n🛡️ 演示 4: 错误处理和重试机制")
    print("-" * 50)
    
    client = WorkflowHttpClient()
    await client.initialize()
    
    try:
        # 创建包含错误处理的工作流，使用模板
        execute_result = await client.create_workflow_from_template(
            "error_handling_workflow", 
            {
                "name": "错误处理演示",
                "variables": {
                    "input_value": {"default": "test_input_data"}
                }
            }
        )
        
        if execute_result["success"]:
            workflow_id = execute_result["workflow_id"]
            print(f"✅ 创建错误处理演示工作流，ID: {workflow_id}")
            
            # 执行包含错误处理的工作流
            execute_result = await client.execute_workflow(workflow_id, {"input_value": "valid_test_data"})
            
            if execute_result["success"]:
                execution_id = execute_result["execution_id"]
                print(f"🚀 开始执行错误处理演示...")
                
                # 监控执行过程，观察重试和错误处理
                await client.monitor_workflow_execution(execution_id, check_interval=2)
                
            else:
                print(f"❌ 工作流执行失败: {execute_result.get('message', '未知错误')}")
        else:
            print(f"❌ 创建工作流失败: {execute_result.get('message', '未知错误')}")
            
    except Exception as e:
        print(f"❌ 演示过程发生错误: {e}")
    finally:
        await client.cleanup()

async def demonstrate_workflow_management():
    """演示工作流管理功能"""
    print("\n📊 演示 5: 工作流管理和状态查询")
    print("-" * 50)
    
    client = WorkflowHttpClient()
    await client.initialize()
    
    try:
        # 列出所有工作流
        workflows_result = await client.list_workflows()
        if workflows_result["success"]:
            workflows = workflows_result["workflows"]
            print(f"📋 系统中共有 {len(workflows)} 个工作流:")
            
            for workflow in workflows[:5]:  # 显示前5个
                print(f"  🔹 {workflow.get('name', 'Unknown')}: {workflow.get('id', 'N/A')}")
            
            if len(workflows) > 5:
                print(f"  ... 还有 {len(workflows) - 5} 个工作流")
        
        # 列出执行历史
        print(f"\n🕒 查询执行历史:")
        executions_result = await client.list_executions()
        if executions_result["success"]:
            executions = executions_result["executions"]
            print(f"📈 共有 {len(executions)} 次执行记录:")
            
            for execution in executions[-3:]:  # 显示最近3次
                print(f"  📊 执行ID: {execution['execution_id'][:8]}...")
                print(f"      状态: {execution['status']}")
                print(f"      开始时间: {execution['start_time']}")
                if execution.get('total_execution_time'):
                    print(f"      耗时: {execution['total_execution_time']:.2f}秒")
                print()
        
        # 如果有执行记录，生成最新的执行报告
        if executions and len(executions) > 0:
            latest_execution = executions[-1]
            execution_id = latest_execution["execution_id"]
            
            print(f"📋 生成最新执行报告: {execution_id[:8]}...")
            report_result = await client.generate_execution_report(execution_id)
            
            if report_result["success"]:
                report = report_result["report"]
                print(f"✅ 报告生成成功")
                print(f"    工作流: {report['workflow_name']}")
                print(f"    总任务: {report['task_statistics']['total']}")
                print(f"    成功率: {report['task_statistics']['success_rate']:.1%}")
            else:
                print(f"❌ 报告生成失败")
        
    except Exception as e:
        print(f"❌ 演示过程发生错误: {e}")
    finally:
        await client.cleanup()

async def demonstrate_advanced_features():
    """演示高级特性"""
    print("\n🎯 演示 6: 高级特性展示")
    print("-" * 50)
    
    client = WorkflowHttpClient()
    await client.initialize()
    
    try:
        # 创建一个复杂的工作流，展示多种真实MCP工具的综合应用
        advanced_workflow = {
            "name": "高级特性演示工作流",
            "description": "展示文件操作、数学运算、数据库操作、提示处理的综合应用",
            "variables": {
                "base_value": {"type": "float", "default": 100.0},
                "multiplier": {"type": "float", "default": 1.5},
                "data_source": {"type": "string", "default": "advanced_demo.txt"}
            },
            "steps": [
                {
                    "id": "create_source_file",
                    "name": "创建源数据文件",
                    "type": "function",
                    "action": "write_file",
                    "parameters": {
                        "path": "{{data_source}}",
                        "content": "Advanced workflow demo data: {{base_value}}"
                    }
                },
                {
                    "id": "perform_calculation",
                    "name": "执行数学运算",
                    "type": "function",
                    "action": "multiply",
                    "depends_on": ["create_source_file"],
                    "parameters": {
                        "a": "{{base_value}}",
                        "b": "{{multiplier}}"
                    }
                },
                {
                    "id": "create_db_table",
                    "name": "创建数据库表",
                    "type": "function", 
                    "action": "create_table",
                    "depends_on": ["perform_calculation"],
                    "parameters": {
                        "table_name": "advanced_results"
                    }
                },
                {
                    "id": "insert_calculation_result",
                    "name": "插入计算结果",
                    "type": "function",
                    "action": "insert_data", 
                    "depends_on": ["create_db_table"],
                    "parameters": {
                        "table_name": "advanced_results",
                        "data": {"name": "multiply_operation", "value": "calculation_result"}
                    }
                },
                {
                    "id": "format_summary",
                    "name": "格式化摘要报告",
                    "type": "function",
                    "action": "format_prompt",
                    "depends_on": ["insert_calculation_result"],
                    "parameters": {
                        "template": "高级工作流执行完成\\n基础值: {base}\\n乘数: {mult}\\n计算结果: {base} × {mult} = {result}\\n数据已保存到数据库",
                        "variables": {
                            "base": "{{base_value}}",
                            "mult": "{{multiplier}}", 
                            "result": "{{base_value}} × {{multiplier}}"
                        }
                    }
                },
                {
                    "id": "save_final_report",
                    "name": "保存最终报告",
                    "type": "function",
                    "action": "write_file",
                    "depends_on": ["format_summary"],
                    "parameters": {
                        "path": "advanced_workflow_report.txt",
                        "content": "高级工作流执行报告\\n执行时间: $(timestamp)\\n基础值: {{base_value}}\\n乘数: {{multiplier}}\\n所有操作已成功完成"
                    }
                },
                {
                    "id": "verify_results",
                    "name": "验证执行结果",
                    "type": "function",
                    "action": "query_data",
                    "depends_on": ["save_final_report"],
                    "parameters": {
                        "table_name": "advanced_results"
                    }
                }
            ]
        }
        
        print(f"🔧 创建高级特性演示工作流...")
        create_result = await client.create_workflow(advanced_workflow)
        
        if create_result["success"]:
            workflow_id = create_result["workflow_id"]
            print(f"✅ 高级工作流创建成功，ID: {workflow_id}")
            
            # 执行工作流
            execution_variables = {
                "base_value": 250.0,
                "multiplier": 2.5,
                "data_source": "advanced_demo_custom.txt"
            }
            
            print(f"🚀 启动高级工作流执行...")
            execute_result = await client.execute_workflow(workflow_id, execution_variables)
            
            if execute_result["success"]:
                execution_id = execute_result["execution_id"]
                print(f"⚡ 高级工作流开始执行，ID: {execution_id}")
                
                # 监控执行，观察所有步骤
                await client.monitor_workflow_execution(execution_id, check_interval=2)
                
            else:
                print(f"❌ 工作流执行失败: {execute_result.get('message', '未知错误')}")
        else:
            print(f"❌ 工作流创建失败: {create_result.get('message', '未知错误')}")
            
    except Exception as e:
        print(f"❌ 演示过程发生错误: {e}")
    finally:
        await client.cleanup()

async def interactive_workflow_builder():
    """真正的交互式工作流构建器"""
    print("\n🎮 交互式工作流构建器")
    print("=" * 60)
    print("欢迎使用真正的交互式工作流构建器！")
    print("你可以逐步构建自己的自定义工作流。")
    print("=" * 60)
    
    client = WorkflowHttpClient()
    await client.initialize()
    
    # 可用的MCP工具和它们的参数
    available_tools = {
        "文件操作": {
            "read_file": {"参数": ["path"], "描述": "读取文件内容"},
            "write_file": {"参数": ["path", "content"], "描述": "写入文件"},
            "list_files": {"参数": ["directory_path"], "描述": "列出目录文件"},
            "delete_file": {"参数": ["path"], "描述": "删除文件"}
        },
        "数学运算": {
            "add": {"参数": ["a", "b"], "描述": "加法运算"},
            "subtract": {"参数": ["a", "b"], "描述": "减法运算"},
            "multiply": {"参数": ["a", "b"], "描述": "乘法运算"},
            "divide": {"参数": ["a", "b"], "描述": "除法运算"},
            "power": {"参数": ["base", "exponent"], "描述": "幂运算"}
        },
        "数据库操作": {
            "create_table": {"参数": ["table_name"], "描述": "创建数据表"},
            "insert_data": {"参数": ["table_name", "data"], "描述": "插入数据"},
            "query_data": {"参数": ["table_name", "query"], "描述": "查询数据"},
            "update_data": {"参数": ["table_name", "data", "condition"], "描述": "更新数据"}
        },
        "提示处理": {
            "format_prompt": {"参数": ["template", "variables"], "描述": "格式化提示模板"},
            "process_text": {"参数": ["text", "operation"], "描述": "处理文本内容"}
        }
    }
    
    try:
        # 步骤1：工作流基本信息
        print("\n📝 步骤 1: 工作流基本信息")
        print("-" * 40)
        
        workflow_name = input("请输入工作流名称: ").strip()
        if not workflow_name:
            workflow_name = "用户自定义工作流"
            
        workflow_description = input("请输入工作流描述: ").strip()
        if not workflow_description:
            workflow_description = "通过交互式构建器创建的工作流"
        
        # 步骤2：变量定义
        print("\n🔧 步骤 2: 定义工作流变量")
        print("-" * 40)
        print("是否需要定义工作流变量？(y/n)")
        
        variables = {}
        if input().strip().lower() in ['y', 'yes', '是']:
            print("请逐个定义变量（输入空行结束）:")
            while True:
                var_name = input("变量名: ").strip()
                if not var_name:
                    break
                    
                var_type = input("变量类型 (string/int/float/bool): ").strip()
                if var_type not in ['string', 'int', 'float', 'bool']:
                    var_type = 'string'
                    
                var_default = input("默认值: ").strip()
                
                # 类型转换
                if var_type == 'int' and var_default.isdigit():
                    var_default = int(var_default)
                elif var_type == 'float':
                    try:
                        var_default = float(var_default)
                    except:
                        var_default = 0.0
                elif var_type == 'bool':
                    var_default = var_default.lower() in ['true', 'yes', '1', '是']
                
                variables[var_name] = {"type": var_type, "default": var_default}
                print(f"✅ 已添加变量: {var_name} ({var_type}) = {var_default}")
        
        # 步骤3：构建工作流步骤
        print("\n⚙️ 步骤 3: 构建工作流步骤")
        print("-" * 40)
        print("现在开始添加工作流步骤...")
        
        steps = []
        step_counter = 1
        
        while True:
            print(f"\n--- 步骤 {step_counter} ---")
            
            # 显示可用工具
            print("\n可用的工具类别:")
            for i, category in enumerate(available_tools.keys(), 1):
                print(f"  {i}. {category}")
            
            category_choice = input("选择工具类别 (输入数字，或输入 'done' 完成): ").strip()
            
            if category_choice.lower() == 'done':
                break
                
            try:
                category_idx = int(category_choice) - 1
                category_name = list(available_tools.keys())[category_idx]
                tools_in_category = available_tools[category_name]
            except (ValueError, IndexError):
                print("❌ 无效的选择，请重试")
                continue
            
            # 显示该类别下的工具
            print(f"\n{category_name} 中可用的工具:")
            tool_list = list(tools_in_category.keys())
            for i, tool_name in enumerate(tool_list, 1):
                tool_info = tools_in_category[tool_name]
                print(f"  {i}. {tool_name} - {tool_info['描述']}")
                print(f"     参数: {', '.join(tool_info['参数'])}")
            
            tool_choice = input("选择工具 (输入数字): ").strip()
            try:
                tool_idx = int(tool_choice) - 1
                selected_tool = tool_list[tool_idx]
                tool_info = tools_in_category[selected_tool]
            except (ValueError, IndexError):
                print("❌ 无效的选择，请重试")
                continue
            
            # 步骤详细信息
            step_name = input(f"步骤名称 (默认: {tool_info['描述']}): ").strip()
            if not step_name:
                step_name = tool_info['描述']
            
            step_id = f"step_{step_counter}"
            
            # 参数设置
            print(f"\n设置 {selected_tool} 的参数:")
            parameters = {}
            for param in tool_info['参数']:
                param_value = input(f"  {param}: ").strip()
                
                # 支持变量引用
                if param_value.startswith("{{") and param_value.endswith("}}"):
                    parameters[param] = param_value
                elif param in ['a', 'b', 'base', 'exponent'] and param_value.replace('.', '').replace('-', '').isdigit():
                    # 数值参数
                    if '.' in param_value:
                        parameters[param] = float(param_value)
                    else:
                        parameters[param] = int(param_value)
                elif param == 'data' and selected_tool == 'insert_data':
                    # JSON数据
                    try:
                        parameters[param] = json.loads(param_value)
                    except:
                        parameters[param] = {"value": param_value}
                else:
                    parameters[param] = param_value
            
            # 依赖关系
            dependencies = []
            if steps:  # 如果已有步骤
                print(f"\n设置依赖关系 (已有步骤: {[s['id'] for s in steps]})")
                deps_input = input("依赖的步骤ID (多个用逗号分隔，留空表示无依赖): ").strip()
                if deps_input:
                    dependencies = [dep.strip() for dep in deps_input.split(',') if dep.strip()]
            
            # 创建步骤
            step = {
                "id": step_id,
                "name": step_name,
                "type": "function",
                "action": selected_tool,
                "parameters": parameters
            }
            
            if dependencies:
                step["depends_on"] = dependencies
            
            steps.append(step)
            print(f"✅ 已添加步骤: {step_name}")
            step_counter += 1
            
            # 询问是否继续
            continue_adding = input("\n是否继续添加步骤？(y/n): ").strip().lower()
            if continue_adding not in ['y', 'yes', '是']:
                break
        
        if not steps:
            print("❌ 没有添加任何步骤，无法创建工作流")
            return
        
        # 步骤4：构建完整工作流定义
        print("\n🔨 步骤 4: 生成工作流定义")
        print("-" * 40)
        
        user_workflow = {
            "name": workflow_name,
            "description": workflow_description,
            "steps": steps
        }
        
        if variables:
            user_workflow["variables"] = variables
        
        # 显示工作流摘要
        print("\n📋 工作流摘要:")
        print(f"  名称: {workflow_name}")
        print(f"  描述: {workflow_description}")
        print(f"  变量: {len(variables)} 个")
        print(f"  步骤: {len(steps)} 个")
        for step in steps:
            deps_str = f" (依赖: {', '.join(step.get('depends_on', []))})" if step.get('depends_on') else ""
            print(f"    - {step['name']} [{step['action']}]{deps_str}")
        
        # 步骤5：确认并创建
        print(f"\n✅ 步骤 5: 创建和执行工作流")
        print("-" * 40)
        
        confirm = input("确认创建此工作流？(y/n): ").strip().lower()
        if confirm not in ['y', 'yes', '是']:
            print("❌ 用户取消了工作流创建")
            return
        
        print(f"🔧 正在创建工作流: {workflow_name}")
        create_result = await client.create_workflow(user_workflow)
        
        if create_result["success"]:
            workflow_id = create_result["workflow_id"]
            print(f"✅ 工作流创建成功！ID: {workflow_id}")
            
            # 询问是否立即执行
            execute_now = input("是否立即执行此工作流？(y/n): ").strip().lower()
            if execute_now in ['y', 'yes', '是']:
                
                # 收集执行时变量值
                execution_variables = {}
                if variables:
                    print("\n请提供执行时的变量值:")
                    for var_name, var_info in variables.items():
                        current_value = input(f"  {var_name} (默认: {var_info['default']}): ").strip()
                        if current_value:
                            # 类型转换
                            if var_info['type'] == 'int':
                                try:
                                    execution_variables[var_name] = int(current_value)
                                except:
                                    execution_variables[var_name] = var_info['default']
                            elif var_info['type'] == 'float':
                                try:
                                    execution_variables[var_name] = float(current_value)
                                except:
                                    execution_variables[var_name] = var_info['default']
                            elif var_info['type'] == 'bool':
                                execution_variables[var_name] = current_value.lower() in ['true', 'yes', '1', '是']
                            else:
                                execution_variables[var_name] = current_value
                        else:
                            execution_variables[var_name] = var_info['default']
                
                print(f"🚀 开始执行工作流...")
                execute_result = await client.execute_workflow(workflow_id, execution_variables)
                
                if execute_result["success"]:
                    execution_id = execute_result["execution_id"]
                    print(f"⚡ 工作流开始执行，执行ID: {execution_id}")
                    await client.monitor_workflow_execution(execution_id, check_interval=2)
                else:
                    print(f"❌ 执行失败: {execute_result.get('message', '未知错误')}")
            else:
                print(f"💾 工作流已保存，ID: {workflow_id}")
                print("你可以稍后通过工作流管理功能执行它")
        else:
            print(f"❌ 工作流创建失败: {create_result.get('message', '未知错误')}")
            
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了交互式构建过程")
    except Exception as e:
        print(f"❌ 交互式构建过程发生错误: {e}")
    finally:
        await client.cleanup()

def print_challenge_summary():
    """打印挑战总结"""
    print("\n" + "=" * 80)
    print("🎉 Challenge 8 完成总结")
    print("=" * 80)
    print()
    print("恭喜！你已经完成了MCP系列的最终挑战！")
    print()
    print("🏆 你已经掌握的技能:")
    print("  ✅ 复杂工作流的设计和实现")
    print("  ✅ 多步骤任务协调和依赖管理")
    print("  ✅ 并行任务执行和性能优化")
    print("  ✅ 条件分支和循环控制逻辑")
    print("  ✅ 智能错误处理和重试机制")
    print("  ✅ 实时工作流状态监控")
    print("  ✅ 工作流模板和继承系统")
    print("  ✅ 变量管理和数据传递")
    print("  ✅ 执行报告和性能分析")
    print("  ✅ 事件驱动和触发器机制")
    print()
    print("🚀 应用场景:")
    print("  • ETL数据处理管道")
    print("  • 机器学习训练流程")
    print("  • 微服务协调和编排")
    print("  • DevOps CI/CD流程")
    print("  • 业务流程自动化")
    print("  • 数据科学实验管理")
    print("  • 批处理任务调度")
    print()
    print("🎓 学习成果:")
    print("  你现在已经完全掌握了Model Context Protocol的所有核心概念，")
    print("  能够构建复杂的分布式系统和智能工作流引擎。")
    print()
    print("🔮 下一步:")
    print("  • 探索更多MCP服务器的集成")
    print("  • 构建你自己的业务特定工作流")
    print("  • 贡献到MCP开源社区")
    print("  • 将MCP集成到实际项目中")
    print()
    print("=" * 80)
    print("感谢完成整个MCP学习之旅！🎊")
    print("=" * 80)

async def main():
    """主函数 - 运行完整的Challenge 8演示"""
    
    print_welcome()
    
    # 检查工作流 HTTP 服务器
    if not await check_workflow_server():
        print("❌ 服务器检查失败，退出演示")
        return
    
    # 检查服务器连接
    if not await check_server_connection():
        print("❌ 服务器连接失败，退出演示")
        return
    
    print("\n🎯 Challenge 8 将通过以下步骤展示工作流引擎的能力:")
    print("  1. 工作流创建和验证")
    print("  2. 工作流模板使用") 
    print("  3. 工作流执行与监控")
    print("  4. 错误处理和重试机制")
    print("  5. 工作流管理和状态查询")
    print("  6. 高级特性展示")
    print("  7. 交互式工作流构建器")
    
    try:
        # 运行所有演示
        await demonstrate_workflow_creation()
        await asyncio.sleep(2)
        
        await demonstrate_template_usage()
        await asyncio.sleep(2)
        
        await demonstrate_workflow_execution()
        await asyncio.sleep(2)
        
        await demonstrate_error_handling()
        await asyncio.sleep(2)
        
        await demonstrate_workflow_management()
        await asyncio.sleep(2)
        
        await demonstrate_advanced_features()
        await asyncio.sleep(2)
        
        await interactive_workflow_builder()
        
        # 打印总结
        print_challenge_summary()
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了挑战")
    except Exception as e:
        print(f"\n❌ 挑战过程中发生错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())
