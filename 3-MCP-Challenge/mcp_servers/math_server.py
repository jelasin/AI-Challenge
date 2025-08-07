#!/usr/bin/env python3
"""
MCP数学计算服务器
提供基础数学运算工具的MCP服务器实现
"""

import asyncio
import math
from typing import Any, Dict, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
except ImportError:
    logger.error("请安装mcp包: pip install mcp")
    exit(1)

# 创建MCP服务器实例
server = Server("math-server")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """返回此服务器提供的工具列表"""
    return [
        Tool(
            name="add",
            description="加法运算：计算两个数字的和",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "第一个数字"
                    },
                    "b": {
                        "type": "number", 
                        "description": "第二个数字"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="subtract",
            description="减法运算：计算两个数字的差",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "被减数"
                    },
                    "b": {
                        "type": "number",
                        "description": "减数"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="multiply",
            description="乘法运算：计算两个数字的乘积",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "第一个数字"
                    },
                    "b": {
                        "type": "number",
                        "description": "第二个数字"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="divide",
            description="除法运算：计算两个数字的商",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "被除数"
                    },
                    "b": {
                        "type": "number",
                        "description": "除数（不能为0）"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="power",
            description="乘方运算：计算a的b次方",
            inputSchema={
                "type": "object",
                "properties": {
                    "base": {
                        "type": "number",
                        "description": "底数"
                    },
                    "exponent": {
                        "type": "number",
                        "description": "指数"
                    }
                },
                "required": ["base", "exponent"]
            }
        ),
        Tool(
            name="sqrt",
            description="开平方根运算",
            inputSchema={
                "type": "object", 
                "properties": {
                    "number": {
                        "type": "number",
                        "description": "要开方的数字（必须大于等于0）"
                    }
                },
                "required": ["number"]
            }
        ),
        Tool(
            name="factorial",
            description="阶乘运算：计算n!",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "要计算阶乘的非负整数"
                    }
                },
                "required": ["n"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """处理工具调用"""
    try:
        if name == "add":
            a = float(arguments["a"])
            b = float(arguments["b"])
            result = a + b
            return [TextContent(type="text", text=f"{a} + {b} = {result}")]
            
        elif name == "subtract":
            a = float(arguments["a"])
            b = float(arguments["b"])
            result = a - b
            return [TextContent(type="text", text=f"{a} - {b} = {result}")]
            
        elif name == "multiply":
            a = float(arguments["a"])
            b = float(arguments["b"])
            result = a * b
            return [TextContent(type="text", text=f"{a} × {b} = {result}")]
            
        elif name == "divide":
            a = float(arguments["a"])
            b = float(arguments["b"])
            if b == 0:
                return [TextContent(type="text", text="错误：除数不能为0")]
            result = a / b
            return [TextContent(type="text", text=f"{a} ÷ {b} = {result}")]
            
        elif name == "power":
            base = float(arguments["base"])
            exponent = float(arguments["exponent"])
            result = pow(base, exponent)
            return [TextContent(type="text", text=f"{base} ^ {exponent} = {result}")]
            
        elif name == "sqrt":
            number = float(arguments["number"])
            if number < 0:
                return [TextContent(type="text", text="错误：不能对负数开平方根")]
            result = math.sqrt(number)
            return [TextContent(type="text", text=f"√{number} = {result}")]
            
        elif name == "factorial":
            n = int(arguments["n"])
            if n < 0:
                return [TextContent(type="text", text="错误：阶乘的参数必须是非负整数")]
            result = math.factorial(n)
            return [TextContent(type="text", text=f"{n}! = {result}")]
            
        else:
            return [TextContent(type="text", text=f"错误：未知的工具名称 '{name}'")]
            
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"工具调用错误: {e}")
        return [TextContent(type="text", text=f"参数错误: {str(e)}")]
    except Exception as e:
        logger.error(f"意外错误: {e}")
        return [TextContent(type="text", text=f"计算错误: {str(e)}")]

async def main():
    """启动MCP服务器"""
    logger.info("启动数学计算MCP服务器...")
    
    # 使用stdio传输
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"服务器错误: {e}")
