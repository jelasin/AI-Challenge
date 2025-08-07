# -*- coding: utf-8 -*-
"""
Challenge 2: 多服务器工具协调

学习目标:
1. 掌握多MCP服务器的同时连接和管理
2. 学习工具命名空间和冲突处理策略
3. 实现动态服务器连接和断线重连
4. 理解工具发现、枚举和分类管理

核心概念:
- 多服务器配置和连接池管理
- 工具命名冲突检测和解决
- 服务器健康检查和故障转移
- 工具路由和智能调度

实战场景:
构建一个多功能的AI助手系统，同时连接数学计算、文件操作、
天气查询等多个MCP服务器，实现智能工具选择和协调执行。
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field
    import time
    from datetime import datetime
    from typing import cast, Any
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装必要的包:")
    print("pip install langchain-mcp-adapters langchain-openai")
    sys.exit(1)

class ServerStatus(BaseModel):
    """服务器状态模型"""
    name: str
    connected: bool
    last_check: datetime
    tool_count: int
    error_count: int = 0
    last_error: Optional[str] = None

class MultiServerMCPManager:
    """多服务器MCP管理器"""
    
    def __init__(self):
        """初始化多服务器管理器"""
        # 多服务器配置 - 启用真实的两个MCP服务器
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
        
        # 服务器描述（单独存储）
        self.server_descriptions = {
            "math": "数学计算服务器 - 提供基础数学运算",
            "file": "文件系统服务器 - 提供文件操作功能"
        }
        
        # 管理器状态
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.server_status: Dict[str, ServerStatus] = {}
        self.tools_by_server: Dict[str, List[BaseTool]] = {}
        self.all_tools: List[BaseTool] = []
        self.tool_conflicts: Dict[str, List[str]] = defaultdict(list)
        
        # LLM用于智能工具路由
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1
        ) if os.getenv("OPENAI_API_KEY") else None
    
    async def initialize_servers(self) -> bool:
        """初始化所有服务器连接"""
        print("🔧 初始化多服务器MCP管理器...")
        
        try:
            # 创建MCP客户端 - 使用类型转换避免类型错误
            configs = cast(Any, self.server_configs)
            self.mcp_client = MultiServerMCPClient(configs)
            
            # 逐个检查服务器连接（添加超时处理）
            for server_name, config in self.server_configs.items():
                description = self.server_descriptions.get(server_name, "")
                print(f"📡 连接服务器: {server_name} - {description}")
                try:
                    # 添加超时处理
                    await asyncio.wait_for(
                        self.check_server_status(server_name), 
                        timeout=10.0  # 增加超时时间以处理多服务器连接
                    )
                except asyncio.TimeoutError:
                    print(f"  ⏰ {server_name}: 连接超时")
                    self.server_status[server_name] = ServerStatus(
                        name=server_name,
                        connected=False,
                        last_check=datetime.now(),
                        tool_count=0,
                        error_count=1,
                        last_error="连接超时"
                    )
                except Exception as e:
                    print(f"  ❌ {server_name}: 连接失败 - {e}")
                    self.server_status[server_name] = ServerStatus(
                        name=server_name,
                        connected=False,
                        last_check=datetime.now(),
                        tool_count=0,
                        error_count=1,
                        last_error=str(e)
                    )
            
            # 加载所有工具并检测冲突
            await self.load_and_analyze_tools()
            
            print(f"✅ 多服务器管理器初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            raise e  # 直接抛出异常，不使用备用方案
    
    async def check_server_status(self, server_name: str) -> bool:
        """检查单个服务器状态"""
        try:
            # 检查客户端是否已初始化
            if not self.mcp_client:
                raise Exception("MCP客户端未初始化")
                
            # 尝试获取工具（添加超时）
            tools = await asyncio.wait_for(
                self.mcp_client.get_tools(), 
                timeout=15.0
            )
            
            # 从所有工具中过滤属于当前服务器的工具
            # 实际上，MultiServerMCPClient应该能够区分不同服务器的工具
            # 但为了简化处理，我们将工具按服务器配置顺序分配
            server_tools = []
            if tools:
                # 根据服务器在配置中的顺序来分配工具
                server_keys = list(self.server_configs.keys())
                if server_name in server_keys:
                    server_index = server_keys.index(server_name)
                    # 简单的工具分配策略：数学工具给数学服务器，文件工具给文件服务器
                    for tool in tools:
                        tool_name = getattr(tool, 'name', '')
                        if server_name == "math" and tool_name in ['add', 'subtract', 'multiply', 'divide', 'power', 'sqrt', 'factorial']:
                            server_tools.append(tool)
                        elif server_name == "file" and tool_name in ['read_file', 'write_file', 'list_directory']:
                            server_tools.append(tool)
            
            # 更新服务器状态
            self.server_status[server_name] = ServerStatus(
                name=server_name,
                connected=True,
                last_check=datetime.now(),
                tool_count=len(server_tools)
            )
            
            # 存储服务器的工具
            self.tools_by_server[server_name] = server_tools
            
            print(f"  ✅ {server_name}: {len(server_tools)} 个工具可用")
            return True
            
        except asyncio.TimeoutError:
            # 处理超时错误
            if server_name not in self.server_status:
                self.server_status[server_name] = ServerStatus(
                    name=server_name,
                    connected=False,
                    last_check=datetime.now(),
                    tool_count=0
                )
            
            self.server_status[server_name].connected = False
            self.server_status[server_name].error_count += 1
            self.server_status[server_name].last_error = "连接超时"
            
            print(f"  ⏰ {server_name}: 连接超时")
            return False
            
        except Exception as e:
            # 记录错误状态
            if server_name not in self.server_status:
                self.server_status[server_name] = ServerStatus(
                    name=server_name,
                    connected=False,
                    last_check=datetime.now(),
                    tool_count=0
                )
            
            self.server_status[server_name].connected = False
            self.server_status[server_name].error_count += 1
            self.server_status[server_name].last_error = str(e)
            
            print(f"  ❌ {server_name}: 连接失败 - {e}")
            return False
    
    async def load_and_analyze_tools(self):
        """加载所有工具并分析冲突"""
        print("\n🔍 分析工具和检测冲突...")
        
        # 收集所有工具
        self.all_tools = []
        tool_name_to_servers = defaultdict(list)
        
        for server_name, tools in self.tools_by_server.items():
            if server_name not in self.server_status or not self.server_status[server_name].connected:
                continue
                
            for tool in tools:
                self.all_tools.append(tool)
                tool_name = getattr(tool, 'name', f'unnamed_tool_{len(self.all_tools)}')
                tool_name_to_servers[tool_name].append(server_name)
        
        # 检测工具名冲突
        self.tool_conflicts.clear()
        for tool_name, servers in tool_name_to_servers.items():
            if len(servers) > 1:
                self.tool_conflicts[tool_name] = servers
                print(f"⚠️  工具名冲突: '{tool_name}' 存在于服务器 {servers}")
        
        print(f"📊 总计: {len(self.all_tools)} 个工具，{len(self.tool_conflicts)} 个冲突")
    
    async def demonstrate_server_management(self):
        """演示服务器管理功能"""
        print("\n" + "="*60)
        print("🖥️  多服务器管理演示")
        print("="*60)
        
        # 显示服务器状态
        await self.show_server_status()
        
        # 演示服务器健康检查
        await self.demonstrate_health_check()
        
        # 演示工具分类和路由
        await self.demonstrate_tool_routing()
    
    async def show_server_status(self):
        """显示所有服务器状态"""
        print("\n📊 服务器状态报告:")
        print("-" * 50)
        
        for server_name, status in self.server_status.items():
            status_icon = "🟢" if status.connected else "🔴"
            print(f"{status_icon} {server_name}:")
            print(f"   状态: {'在线' if status.connected else '离线'}")
            print(f"   工具数量: {status.tool_count}")
            print(f"   最后检查: {status.last_check.strftime('%H:%M:%S')}")
            
            if status.error_count > 0:
                print(f"   错误次数: {status.error_count}")
                if status.last_error:
                    print(f"   最后错误: {status.last_error}")
            print()
    
    async def demonstrate_health_check(self):
        """演示健康检查功能"""
        print("\n🏥 服务器健康检查演示:")
        print("-" * 40)
        
        print("正在执行健康检查...")
        
        health_results = []
        for server_name in self.server_configs.keys():
            print(f"检查 {server_name}...", end=" ")
            
            start_time = time.time()
            is_healthy = await self.check_server_status(server_name)
            check_time = time.time() - start_time
            
            health_results.append({
                "server": server_name,
                "healthy": is_healthy,
                "response_time": check_time
            })
            
            print(f"{'✅' if is_healthy else '❌'} ({check_time:.3f}s)")
        
        # 健康检查汇总
        healthy_count = sum(1 for r in health_results if r["healthy"])
        total_count = len(health_results)
        
        print(f"\n📈 健康检查结果: {healthy_count}/{total_count} 服务器正常")
    
    async def demonstrate_tool_routing(self):
        """演示智能工具路由"""
        print("\n🧭 智能工具路由演示:")
        print("-" * 40)
        
        # 按服务器分组显示工具
        for server_name, tools in self.tools_by_server.items():
            if not self.server_status[server_name].connected:
                continue
            
            print(f"\n🔧 {server_name} 服务器工具:")
            for tool in tools:
                tool_name = getattr(tool, 'name', 'unnamed_tool')
                tool_desc = getattr(tool, 'description', '无描述')
                print(f"  • {tool_name}: {tool_desc}")
        
        # 显示工具冲突处理
        if self.tool_conflicts:
            print(f"\n⚡ 工具冲突处理策略:")
            for tool_name, servers in self.tool_conflicts.items():
                print(f"  • {tool_name}: 优先使用 {servers[0]} (服务器优先级)")
    
    async def demonstrate_coordinated_operations(self):
        """演示协调操作"""
        print("\n" + "="*60) 
        print("🤝 多服务器协调操作演示")
        print("="*60)
        
        # 复合任务示例
        compound_tasks = [
            {
                "name": "数学计算+结果保存",
                "description": "计算数学表达式并将结果保存到文件",
                "steps": [
                    ("math", "multiply", {"a": 15, "b": 8}),
                    ("file", "write_file", {"file_path": "calculation_result.txt", "content": "计算结果待插入"})
                ]
            },
            {
                "name": "文件操作+数据分析",  
                "description": "读取数据文件并进行数学分析",
                "steps": [
                    ("file", "write_file", {"file_path": "numbers.txt", "content": "25\n16\n9\n4"}),
                    ("file", "read_file", {"file_path": "numbers.txt"}),
                    ("math", "sqrt", {"number": 25})
                ]
            }
        ]
        
        for task in compound_tasks:
            print(f"\n🎯 执行复合任务: {task['name']}")
            print(f"📝 描述: {task['description']}")
            
            results = []
            for server_name, tool_name, args in task["steps"]:
                print(f"\n🔧 步骤: {server_name}.{tool_name}({args})")
                
                try:
                    # 获取对应服务器的工具
                    if server_name not in self.tools_by_server:
                        print(f"❌ 服务器 {server_name} 不可用")
                        continue
                    
                    tool = next((t for t in self.tools_by_server[server_name] if getattr(t, 'name', '') == tool_name), None)
                    
                    if not tool:
                        print(f"❌ 工具 {tool_name} 在服务器 {server_name} 中未找到")
                        continue
                    
                    # 如果是写文件任务且依赖前面的计算结果
                    if tool_name == "write_file" and "计算结果待插入" in str(args.get("content", "")):
                        if results and "=" in results[-1]:
                            # 提取最后一个计算结果
                            last_result = results[-1]
                            args["content"] = f"计算结果: {last_result}"
                    
                    # 执行工具调用
                    result = await tool.ainvoke(args)
                    results.append(str(result))
                    print(f"✅ 结果: {result}")
                    
                except Exception as e:
                    print(f"❌ 执行失败: {e}")
                    results.append(f"错误: {e}")
                
                await asyncio.sleep(0.5)
            
            print(f"\n📊 任务 '{task['name']}' 完成，共执行 {len(results)} 个步骤")
    
    async def demonstrate_intelligent_routing(self):
        """演示基于LLM的智能路由"""
        if not self.llm:
            print("\n⚠️  跳过智能路由演示 - 未设置OPENAI_API_KEY")
            return
        
        print("\n" + "="*60)
        print("🤖 基于LLM的智能工具路由")
        print("="*60)
        
        # 绑定所有可用工具到LLM
        llm_with_tools = self.llm.bind_tools(self.all_tools)
        
        # 复杂任务示例
        complex_tasks = [
            "请计算 25 的平方根，然后将结果保存到名为 'sqrt_result.txt' 的文件中",
            "创建一个名为 'math_test' 的目录，然后在其中创建一个文件 'calculation.txt'，内容是 15 乘以 8 的结果",
            "计算 5 的阶乘，然后检查当前目录下有哪些文件"
        ]
        
        for task in complex_tasks:
            print(f"\n🎯 复杂任务: {task}")
            
            try:
                messages = [
                    SystemMessage(content="""你是一个智能助手，可以使用数学计算和文件操作工具。
当用户提出复杂任务时，请按逻辑顺序使用适当的工具来完成任务。
对于需要多步操作的任务，请逐步执行并将前一步的结果用于后续步骤。"""),
                    HumanMessage(content=task)
                ]
                
                # 让LLM分析任务并选择工具
                response = await llm_with_tools.ainvoke(messages)
                
                # 使用安全的属性访问检查工具调用
                tool_calls = getattr(response, 'tool_calls', None)
                if tool_calls:
                    print(f"🔧 LLM规划了 {len(tool_calls)} 个工具调用:")
                    
                    for i, tool_call in enumerate(tool_calls, 1):
                        # 兼容不同的tool_call格式
                        tool_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
                        tool_args = tool_call.get('args') if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
                        
                        if tool_name:
                            print(f"  {i}. {tool_name}({tool_args})")
                            
                            # 查找并执行工具
                            tool = next((t for t in self.all_tools if getattr(t, 'name', '') == tool_name), None)
                            if tool and isinstance(tool_args, dict):
                                try:
                                    result = await tool.ainvoke(tool_args)
                                    print(f"     ✅ 结果: {result}")
                                except Exception as e:
                                    print(f"     ❌ 执行失败: {e}")
                            else:
                                print(f"     ❌ 工具未找到或参数格式错误: {tool_name}")
                
                # 显示LLM的回复
                response_content = getattr(response, 'content', str(response))
                if response_content:
                    print(f"🤖 LLM说明: {response_content}")
                
            except Exception as e:
                print(f"❌ 智能路由失败: {e}")
            
            print("-" * 40)
            await asyncio.sleep(1)
    
async def demo_multi_server_coordination():
    """Challenge 2 主演示函数"""
    print("🚀 Challenge 2: 多服务器工具协调")
    print("="*60)
    
    # 创建多服务器管理器
    manager = MultiServerMCPManager()
    
    # 初始化服务器连接
    if not await manager.initialize_servers():
        print("❌ 无法初始化服务器，演示结束")
        return
    
    try:
        # 1. 服务器管理演示
        await manager.demonstrate_server_management()
        
        # 2. 协调操作演示
        await manager.demonstrate_coordinated_operations()
        
        # 3. 智能路由演示
        await manager.demonstrate_intelligent_routing()
        
        print("\n🎉 Challenge 2 演示完成！")
        print("\n📚 学习要点总结:")
        print("  ✅ 掌握了多MCP服务器的管理和协调")
        print("  ✅ 学会了工具冲突检测和解决策略")
        print("  ✅ 实现了服务器健康检查和状态监控")
        print("  ✅ 体验了智能工具路由和多服务器协调")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")

def main():
    """直接运行此Challenge的主函数"""
    asyncio.run(demo_multi_server_coordination())

if __name__ == "__main__":
    main()
