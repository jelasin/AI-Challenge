#!/usr/bin/env python3
"""
MCP Challenge 7: 企业级架构与多服务集成
HTTP 架构的 MCP 服务调用演示
"""

import asyncio
import sys
from pathlib import Path

# 导入本地的 HTTP MCP 客户端
try:
    from http_mcp_client import MCPHttpClient, EnterpriseHttpDemo
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保 http_mcp_client.py 文件存在于当前目录中")
    sys.exit(1)


async def main():
    """主函数 - Challenge 7 演示入口"""
    print("🎯 MCP Challenge 7: 企业级 HTTP 架构演示")
    print("=" * 60)
    
    gateway_url = "http://localhost:8000"
    
    # 确保 workspace 目录存在
    workspace_dir = Path(__file__).parent.parent / "mcp_servers" / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 工作空间: {workspace_dir}")
    
    # 检查服务器是否运行
    print("🔍 检查 MCP HTTP 服务器...")
    try:
        async with MCPHttpClient(gateway_url) as client:
            info = await client.get_gateway_info()
            if "error" not in info:
                print(f"✅ HTTP 服务器运行正常: {info.get('name', 'MCP Gateway')}")
                
                # 运行完整演示
                print("\n🚀 启动企业级演示...")
                demo = EnterpriseHttpDemo(gateway_url)
                await demo.run_demo()
                
                print("\n🎉 Challenge 7 演示完成!")
                print(f"🌐 网关地址: {gateway_url}")
                print(f"📚 API 文档: {gateway_url}/docs")
                print(f"📁 生产文件位置: {workspace_dir}")
                
            else:
                await show_server_instructions()
                
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        await show_server_instructions()


async def show_server_instructions():
    """显示服务器启动说明"""
    mcp_servers_dir = Path(__file__).parent.parent / "mcp_servers"
    workspace_dir = mcp_servers_dir / "workspace"
    
    # 确保 workspace 目录存在
    workspace_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("📋 请先启动 MCP HTTP 服务器")
    print("="*60)
    print("在新终端窗口中运行:")
    print(f"  cd {mcp_servers_dir}")
    print("  python enterprise_gateway_http.py")
    print()
    print("或使用快速启动脚本:")
    print("  python start_http_server.py")
    print()
    print(f"📁 生产文件将保存在: {workspace_dir}")
    print()
    print("服务器启动后，重新运行此演示:")
    print("  python main.py")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
