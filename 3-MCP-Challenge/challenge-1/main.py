# -*- coding: utf-8 -*-
"""
Challenge 1: 基础MCP工具连接

学习目标:
1. 理解MCP协议的基本概念
2. 掌握MultiServerMCPClient的使用方法
3. 学习如何连接到MCP服务器并加载工具
4. 实践基础的工具调用和错误处理

核心概念:
- MultiServerMCPClient: 多服务器MCP客户端
- Connection配置: 服务器连接参数
- Tool加载: 从MCP服务器获取可用工具
- Tool执行: 调用MCP工具并处理结果

实战场景:
创建一个简单的计算器客户端，连接到数学计算MCP服务器，
执行各种数学运算（加减乘除、乘方、开方、阶乘等）
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

# 添加项目根目录到Python路径，以便导入MCP服务器
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field
    from typing import Any, Dict, List, Optional, Union
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装必要的包:")
    print("pip install langchain-mcp-adapters langchain-openai")
    sys.exit(1)

class MathCalculatorDemo:
    """MCP数学计算器演示类"""
    
    def __init__(self):
        """初始化MCP客户端和LLM"""
        # MCP服务器配置
        self.server_configs = {
            "math": {
                "command": "python",
                "args": [str(project_root / "mcp_servers" / "math_server.py")],
                "transport": "stdio"
            }
        }
        
        # 初始化MCP客户端
        self.mcp_client = None
        self.available_tools: List[BaseTool] = []
        
        # 初始化LLM（用于智能工具调用）
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1
        ) if os.getenv("OPENAI_API_KEY") else None
    
    async def setup_mcp_client(self) -> bool:
        """设置MCP客户端连接"""
        try:
            print("🔧 初始化MCP客户端...")
            
            # 创建MCP客户端实例 - 使用类型转换避免类型检查错误
            from typing import cast, Any
            configs = cast(Any, self.server_configs)
            self.mcp_client = MultiServerMCPClient(configs)
            
            # 加载所有可用工具
            print("📡 连接到MCP服务器并加载工具...")
            self.available_tools = await self.mcp_client.get_tools(server_name="math")
            
            if not self.available_tools:
                raise Exception("无法从MCP服务器加载任何工具，请检查服务器配置")
            
            print(f"✅ 成功加载 {len(self.available_tools)} 个工具")
            
            # 显示可用工具
            print("\n🛠️  可用的数学工具:")
            for i, tool in enumerate(self.available_tools, 1):
                tool_name = getattr(tool, 'name', f'tool_{i}')
                tool_desc = getattr(tool, 'description', '无描述')
                print(f"  {i}. {tool_name}: {tool_desc}")
            
            return True
            
        except Exception as e:
            print(f"❌ MCP客户端设置失败: {e}")
            return False
    
    async def demonstrate_basic_tool_calls(self):
        """演示基础工具调用"""
        print("\n" + "="*60)
        print("🧮 基础数学运算演示")
        print("="*60)
        
        # 基础运算示例
        test_cases = [
            ("add", {"a": 15, "b": 25}, "加法运算"),
            ("subtract", {"a": 100, "b": 37}, "减法运算"), 
            ("multiply", {"a": 8, "b": 7}, "乘法运算"),
            ("divide", {"a": 144, "b": 12}, "除法运算"),
            ("power", {"base": 2, "exponent": 10}, "乘方运算"),
            ("sqrt", {"number": 81}, "平方根运算"),
            ("factorial", {"n": 5}, "阶乘运算")
        ]
        
        for tool_name, args, description in test_cases:
            print(f"\n📊 {description}: {tool_name}({args})")
            
            try:
                # 查找对应的工具 - 使用安全的属性访问
                tool = next((t for t in self.available_tools if getattr(t, 'name', '') == tool_name), None)
                
                if tool:
                    # 执行工具调用
                    result = await tool.ainvoke(args)
                    print(f"✅ 结果: {result}")
                else:
                    print(f"❌ 未找到工具: {tool_name}")
                    
            except Exception as e:
                print(f"❌ 调用失败: {e}")
            
            # 短暂延迟以便观察
            await asyncio.sleep(0.5)
    
    async def demonstrate_error_handling(self):
        """演示错误处理"""
        print("\n" + "="*60)
        print("🚨 错误处理演示")
        print("="*60)
        
        # 错误情况测试
        error_cases = [
            ("divide", {"a": 10, "b": 0}, "除零错误"),
            ("sqrt", {"number": -16}, "负数开方错误"),
            ("factorial", {"n": -3}, "负数阶乘错误"),
            ("add", {"a": "abc", "b": 5}, "参数类型错误")
        ]
        
        for tool_name, args, description in error_cases:
            print(f"\n🧪 {description}: {tool_name}({args})")
            
            try:
                tool = next((t for t in self.available_tools if getattr(t, 'name', '') == tool_name), None)
                
                if tool:
                    result = await tool.ainvoke(args)
                    print(f"📄 服务器响应: {result}")
                else:
                    print(f"❌ 未找到工具: {tool_name}")
                    
            except Exception as e:
                print(f"🛡️  客户端捕获异常: {e}")
    
    async def demonstrate_intelligent_calculation(self):
        """演示智能计算（使用LLM选择工具）"""
        if not self.llm:
            print("\n⚠️  跳过智能计算演示 - 未设置OPENAI_API_KEY")
            return
        
        print("\n" + "="*60)
        print("🤖 智能计算演示（LLM + MCP工具）")
        print("="*60)
        
        # 绑定工具到LLM
        llm_with_tools = self.llm.bind_tools(self.available_tools)
        
        # 复杂计算问题
        math_problems = [
            "计算 (15 + 25) × 3 的结果",
            "求 2的10次方除以4的值", 
            "计算 √(9×16) + 5!",
            "求 (100-37) × 2 + √81"
        ]
        
        for problem in math_problems:
            print(f"\n🤔 问题: {problem}")
            
            try:
                # 使用LLM分析问题并调用工具
                messages = [
                    SystemMessage(content="你是一个数学计算助手。请使用提供的数学工具来解决用户的数学问题。对于复杂的计算，请分步执行。"),
                    HumanMessage(content=problem)
                ]
                
                response = await llm_with_tools.ainvoke(messages)
                
                # 检查是否有工具调用 - 使用类型安全的方式
                tool_calls = getattr(response, 'tool_calls', None)
                if tool_calls:
                    print(f"🔧 LLM决定使用工具: {len(tool_calls)} 个调用")
                    
                    for tool_call in tool_calls:
                        # 兼容不同的tool_call格式
                        tool_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
                        tool_args = tool_call.get('args') if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
                        
                        if tool_name:
                            print(f"  - {tool_name}({tool_args})")
                            
                            # 执行工具调用
                            tool = next((t for t in self.available_tools if getattr(t, 'name', '') == tool_name), None)
                            if tool and tool_args:
                                try:
                                    # 确保tool_args是字典类型
                                    if isinstance(tool_args, dict):
                                        result = await tool.ainvoke(tool_args)
                                        print(f"    结果: {result}")
                                    else:
                                        print(f"    ❌ 工具参数格式错误: {type(tool_args)}")
                                except Exception as tool_error:
                                    print(f"    ❌ 工具执行失败: {tool_error}")
                            else:
                                print(f"    ❌ 未找到工具: {tool_name}")
                
                # 显示LLM的回复
                response_content = getattr(response, 'content', str(response))
                if response_content:
                    print(f"🤖 LLM回复: {response_content}")
                
            except Exception as e:
                print(f"❌ 智能计算失败: {e}")
            
            await asyncio.sleep(1)
    
    async def interactive_calculator(self):
        """交互式计算器模式"""
        print("\n" + "="*60)
        print("🎯 交互式计算器模式")
        print("="*60)
        print("输入数学表达式，我会帮你计算！")
        print("支持的操作: 加减乘除、乘方(^)、开方(√)、阶乘(!)") 
        print("输入 'quit' 退出，'help' 查看帮助")
        
        while True:
            try:
                user_input = input("\n🧮 请输入计算表达式: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 退出计算器模式")
                    break
                
                if user_input.lower() in ['help', 'h']:
                    self.show_calculator_help()
                    continue
                
                if not user_input:
                    continue
                
                # 简单的表达式解析和工具调用
                await self.parse_and_calculate(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 退出计算器模式")
                break
            except Exception as e:
                print(f"❌ 计算错误: {e}")
    
    def show_calculator_help(self):
        """显示计算器帮助信息"""
        print("\n📖 计算器使用帮助:")
        print("  • 加法: 5 + 3")
        print("  • 减法: 10 - 4")
        print("  • 乘法: 6 * 7") 
        print("  • 除法: 20 / 4")
        print("  • 乘方: 2 ^ 8")
        print("  • 开方: sqrt(25)")
        print("  • 阶乘: 5!")
    
    async def parse_and_calculate(self, expression: str):
        """解析表达式并调用相应工具"""
        expression = expression.replace(" ", "")
        
        try:
            # 简单的表达式解析（可以扩展为更复杂的解析器）
            if "+" in expression and len(expression.split("+")) == 2:
                parts = expression.split("+")
                a, b = float(parts[0]), float(parts[1])
                tool = next((t for t in self.available_tools if getattr(t, 'name', '') == "add"), None)
                if tool:
                    result = await tool.ainvoke({"a": a, "b": b})
                    print(f"✅ {result}")
            
            elif "-" in expression and len(expression.split("-")) == 2:
                parts = expression.split("-")
                a, b = float(parts[0]), float(parts[1])
                tool = next((t for t in self.available_tools if getattr(t, 'name', '') == "subtract"), None)
                if tool:
                    result = await tool.ainvoke({"a": a, "b": b})
                    print(f"✅ {result}")
            
            elif "*" in expression and len(expression.split("*")) == 2:
                parts = expression.split("*")
                a, b = float(parts[0]), float(parts[1])
                tool = next((t for t in self.available_tools if getattr(t, 'name', '') == "multiply"), None)
                if tool:
                    result = await tool.ainvoke({"a": a, "b": b})
                    print(f"✅ {result}")
            
            elif "/" in expression and len(expression.split("/")) == 2:
                parts = expression.split("/")
                a, b = float(parts[0]), float(parts[1])
                tool = next((t for t in self.available_tools if getattr(t, 'name', '') == "divide"), None)
                if tool:
                    result = await tool.ainvoke({"a": a, "b": b})
                    print(f"✅ {result}")
            
            elif "^" in expression and len(expression.split("^")) == 2:
                parts = expression.split("^")
                base, exp = float(parts[0]), float(parts[1])
                tool = next((t for t in self.available_tools if getattr(t, 'name', '') == "power"), None)
                if tool:
                    result = await tool.ainvoke({"base": base, "exponent": exp})
                    print(f"✅ {result}")
            
            elif expression.startswith("sqrt(") and expression.endswith(")"):
                num_str = expression[5:-1]  # 去掉 "sqrt(" 和 ")"
                num = float(num_str)
                tool = next((t for t in self.available_tools if getattr(t, 'name', '') == "sqrt"), None)
                if tool:
                    result = await tool.ainvoke({"number": num})
                    print(f"✅ {result}")
            
            elif expression.endswith("!"):
                num_str = expression[:-1]  # 去掉 "!"
                num = int(num_str)
                tool = next((t for t in self.available_tools if getattr(t, 'name', '') == "factorial"), None)
                if tool:
                    result = await tool.ainvoke({"n": num})
                    print(f"✅ {result}")
            
            else:
                print("❌ 不支持的表达式格式。输入 'help' 查看支持的格式")
                
        except ValueError:
            print("❌ 数字格式错误")
        except Exception as e:
            print(f"❌ 计算失败: {e}")

async def demo_basic_mcp_connection():
    """Challenge 1 主演示函数"""
    print("🚀 Challenge 1: 基础MCP工具连接")
    print("="*60)
    
    # 创建演示实例
    demo = MathCalculatorDemo()
    
    # 设置MCP客户端
    if not await demo.setup_mcp_client():
        print("❌ 无法设置MCP客户端，演示结束")
        return
    
    try:
        # 1. 基础工具调用演示
        await demo.demonstrate_basic_tool_calls()
        
        # 2. 错误处理演示
        await demo.demonstrate_error_handling()
        
        # 3. 智能计算演示（如果有OpenAI API）
        await demo.demonstrate_intelligent_calculation()
        
        # 4. 交互式计算器（可选）
        if input("\n🤔 是否进入交互式计算器模式？(y/N): ").lower().startswith('y'):
            await demo.interactive_calculator()
        
        print("\n🎉 Challenge 1 演示完成！")
        print("\n📚 学习要点总结:")
        print("  ✅ 学会了使用MultiServerMCPClient连接MCP服务器")
        print("  ✅ 掌握了工具的加载和调用方法")
        print("  ✅ 理解了MCP工具的错误处理机制")
        print("  ✅ 体验了LLM与MCP工具的结合使用")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")

def main():
    """直接运行此Challenge的主函数"""
    asyncio.run(demo_basic_mcp_connection())

if __name__ == "__main__":
    main()
