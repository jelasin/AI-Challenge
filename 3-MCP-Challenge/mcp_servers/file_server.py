#!/usr/bin/env python3
"""
MCP文件系统服务器
提供文件操作工具和资源访问的MCP服务器实现
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 安全的工作目录（限制文件访问范围）
SAFE_WORK_DIR = Path(__file__).parent / "workspace"
SAFE_WORK_DIR.mkdir(exist_ok=True)

def safe_path(path_str: str) -> Path:
    """确保路径在安全目录内"""
    path = Path(path_str)
    if path.is_absolute():
        # 如果是绝对路径，检查是否在安全目录内
        try:
            resolved = path.resolve()
            SAFE_WORK_DIR.resolve().relative_to(resolved)
            return resolved
        except ValueError:
            raise ValueError(f"路径 {path} 不在安全工作目录内")
    else:
        # 相对路径，相对于安全目录
        full_path = (SAFE_WORK_DIR / path).resolve()
        try:
            full_path.relative_to(SAFE_WORK_DIR.resolve())
            return full_path
        except ValueError:
            raise ValueError(f"路径 {path} 试图访问安全目录外的文件")

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent, Resource
    import mcp.server.stdio
    from pydantic import AnyUrl
except ImportError:
    logger.error("请安装mcp包: pip install mcp")
    exit(1)

# 创建MCP服务器实例
server = Server("file-server")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """返回此服务器提供的工具列表"""
    return [
        Tool(
            name="read_file",
            description="读取文件内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "要读取的文件路径"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="write_file",
            description="写入文件内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "要写入的文件路径"
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的内容"
                    }
                },
                "required": ["file_path", "content"]
            }
        ),
        Tool(
            name="list_directory",
            description="列出目录中的文件和子目录",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": "要列出的目录路径，默认为当前工作目录",
                        "default": "."
                    }
                }
            }
        )
    ]

@server.list_resources()
async def list_resources() -> List[Resource]:
    """列出可用的文件资源"""
    resources = []
    
    try:
        # 遍历工作目录中的文件
        for file_path in SAFE_WORK_DIR.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(SAFE_WORK_DIR)
                resources.append(
                    Resource(
                        uri=AnyUrl(f"file://{relative_path}"),
                        name=file_path.name,
                        description=f"文件: {relative_path}",
                        mimeType="text/plain" if file_path.suffix in [".txt", ".py", ".md", ".json"] else "application/octet-stream"
                    )
                )
    except Exception as e:
        logger.error(f"列出资源时出错: {e}")
    
    return resources

@server.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """读取资源内容"""
    try:
        # 从URI中提取文件路径
        uri_str = str(uri)
        if uri_str.startswith("file://"):
            file_path = uri_str[7:]  # 移除 "file://" 前缀
            full_path = safe_path(file_path)
            
            if not full_path.exists():
                return f"错误：文件 {file_path} 不存在"
            
            if full_path.is_dir():
                return f"错误：{file_path} 是一个目录，不是文件"
            
            # 读取文件内容
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return content
        else:
            return f"错误：不支持的URI格式: {uri_str}"
            
    except Exception as e:
        logger.error(f"读取资源时出错: {e}")
        return f"读取资源时出错: {str(e)}"

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """处理工具调用"""
    try:
        if name == "read_file":
            file_path = arguments["file_path"]
            full_path = safe_path(file_path)
            
            if not full_path.exists():
                return [TextContent(type="text", text=f"错误：文件 {file_path} 不存在")]
            
            if full_path.is_dir():
                return [TextContent(type="text", text=f"错误：{file_path} 是一个目录，不是文件")]
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return [TextContent(type="text", text=f"文件 {file_path} 内容:\n\n{content}")]
            
        elif name == "write_file":
            file_path = arguments["file_path"]
            content = arguments["content"]
            full_path = safe_path(file_path)
            
            # 确保父目录存在
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return [TextContent(type="text", text=f"成功写入文件 {file_path}")]
            
        elif name == "list_directory":
            directory_path = arguments.get("directory_path", ".")
            full_path = safe_path(directory_path)
            
            if not full_path.exists():
                return [TextContent(type="text", text=f"错误：目录 {directory_path} 不存在")]
            
            if not full_path.is_dir():
                return [TextContent(type="text", text=f"错误：{directory_path} 不是一个目录")]
            
            items = []
            for item in full_path.iterdir():
                item_type = "目录" if item.is_dir() else "文件"
                items.append(f"{item_type}: {item.name}")
            
            if not items:
                return [TextContent(type="text", text=f"目录 {directory_path} 为空")]
            
            return [TextContent(type="text", text=f"目录 {directory_path} 内容:\n" + "\n".join(items))]
        
        else:
            return [TextContent(type="text", text=f"错误：未知的工具名称 '{name}'")]
            
    except ValueError as e:
        return [TextContent(type="text", text=f"路径安全错误: {str(e)}")]
    except Exception as e:
        logger.error(f"工具调用错误: {e}")
        return [TextContent(type="text", text=f"操作错误: {str(e)}")]

async def main():
    """启动MCP服务器"""
    logger.info("启动文件系统MCP服务器...")
    logger.info(f"安全工作目录: {SAFE_WORK_DIR.absolute()}")
    
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
