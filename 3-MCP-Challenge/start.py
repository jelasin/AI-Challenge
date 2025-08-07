# -*- coding: utf-8 -*-
"""
MCP (Model Context Protocol) 挑战系列 - 快速入门脚本
运行此脚本来快速体验各个挑战的核心功能
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

def check_requirements():
    """检查环境和依赖"""
    print("🔍 检查环境配置...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本需要3.8或更高")
        return False
    
    print(f"✅ Python版本: {sys.version}")
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  未设置OPENAI_API_KEY环境变量")
        print("请设置API密钥:")
        print("Windows: $env:OPENAI_API_KEY='your-key'")
        print("Linux/Mac: export OPENAI_API_KEY='your-key'")
        return False
    
    print("✅ API密钥已配置")
    
    # 检查MCP相关包
    required_packages = [
        "langchain",
        "langchain-openai",
        "langchain-core",
        "langgraph", 
        "langchain-mcp-adapters",
        "mcp",
        "fastmcp",
        "pydantic"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            # 处理包名中的连字符
            import_name = package.replace("-", "_")
            if import_name == "langchain_mcp_adapters":
                from langchain_mcp_adapters.client import MultiServerMCPClient
            elif import_name == "fastmcp":
                import fastmcp
            else:
                __import__(import_name)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def display_menu():
    """显示挑战菜单"""
    print("\n" + "="*60)
    print("🚀 MCP (Model Context Protocol) 学习挑战系列")
    print("="*60)
    print()
    print("选择要运行的挑战:")
    print()
    print("1️⃣  Challenge 1: 基础MCP工具连接 ⭐")
    print("    - 学习MCP客户端基础使用")
    print("    - 连接本地MCP服务器")
    print()
    print("2️⃣  Challenge 2: 多服务器工具协调 ⭐⭐")
    print("    - 管理多个MCP服务器")
    print("    - 工具冲突处理")
    print()
    print("3️⃣  Challenge 3: MCP资源管理和访问 ⭐⭐⭐")
    print("    - 动态资源加载")
    print("    - 文档处理系统")
    print()
    print("4️⃣  Challenge 4: MCP提示模板系统 ⭐⭐⭐⭐")
    print("    - 智能提示管理")
    print("    - 模板参数化")
    print()
    print("5️⃣  Challenge 5: LangGraph与MCP集成 ⭐⭐⭐⭐⭐")
    print("    - Agent工作流编排")
    print("    - 动态工具路由")
    print()
    print("6️⃣  Challenge 6: 自定义MCP服务器开发 ⭐⭐⭐⭐⭐⭐")
    print("    - 服务器端开发")
    print("    - 工具和资源提供")
    print()
    print("7️⃣  Challenge 7: 企业级MCP架构 ⭐⭐⭐⭐⭐⭐⭐")
    print("    - 分布式服务架构")
    print("    - 高可用性设计")
    print()
    print("8️⃣  Challenge 8: 综合应用：智能工作流引擎 ⭐⭐⭐⭐⭐⭐⭐⭐")
    print("    - 完整工作流系统")
    print("    - 多模态数据处理")
    print()
    print("0️⃣  运行所有挑战演示")
    print("Q  退出")
    print()

async def run_challenge_demo(challenge_num: int) -> bool:
    """运行指定挑战的演示"""
    challenge_dir = Path(f"challenge-{challenge_num}")
    main_file = challenge_dir / "main.py"
    
    if not main_file.exists():
        print(f"❌ Challenge {challenge_num} 尚未实现")
        return False
    
    print(f"\n🚀 运行 Challenge {challenge_num} 演示...")
    print("-" * 50)
    
    try:
        # 动态导入并运行挑战
        sys.path.insert(0, str(challenge_dir))
        
        if challenge_num == 1:
            from main import demo_basic_mcp_connection
            await demo_basic_mcp_connection()
        elif challenge_num == 2:
            from main import demo_multi_server_coordination
            await demo_multi_server_coordination()
        elif challenge_num == 3:
            from main import demo_resource_management
            await demo_resource_management()
        elif challenge_num == 4:
            from main import demo_prompt_templates
            await demo_prompt_templates()
        elif challenge_num == 5:
            from main import demo_langgraph_integration
            await demo_langgraph_integration()
        elif challenge_num == 6:
            from main import demo_custom_server_development
            await demo_custom_server_development()
        elif challenge_num == 7:
            from main import demo_enterprise_architecture
            await demo_enterprise_architecture()
        elif challenge_num == 8:
            from main import demo_workflow_engine
            await demo_workflow_engine()
        
        sys.path.pop(0)
        print(f"\n✅ Challenge {challenge_num} 演示完成!")
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        return False

async def run_all_demos():
    """运行所有挑战演示"""
    print("\n🎯 开始运行所有挑战演示...")
    
    for i in range(1, 9):
        success = await run_challenge_demo(i)
        if success:
            print(f"✅ Challenge {i} - 完成")
        else:
            print(f"⚠️  Challenge {i} - 跳过")
        
        if i < 8:
            print("\n" + "-"*30)
            await asyncio.sleep(1)  # 短暂暂停
    
    print("\n🎉 所有演示运行完成!")

def create_sample_mcp_servers():
    """创建示例MCP服务器文件（如果不存在）"""
    servers_dir = Path("mcp_servers")
    servers_dir.mkdir(exist_ok=True)
    
    # 这里可以添加创建示例服务器的代码
    # 但为了演示，我们会在各个挑战中按需创建
    pass

async def main():
    """主函数"""
    print("MCP Challenge Series - 启动中...")
    
    # 检查环境
    if not check_requirements():
        print("\n❌ 环境检查失败，请解决上述问题后重试")
        return
    
    # 创建必要的目录和文件
    create_sample_mcp_servers()
    
    while True:
        display_menu()
        
        try:
            choice = input("请选择 (1-8, 0, Q): ").strip().upper()
            
            if choice == 'Q':
                print("\n👋 感谢使用MCP挑战系列!")
                break
            elif choice == '0':
                await run_all_demos()
            elif choice in [str(i) for i in range(1, 9)]:
                challenge_num = int(choice)
                await run_challenge_demo(challenge_num)
            else:
                print("❌ 无效选择，请重试")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序已退出")
    except Exception as e:
        print(f"❌ 程序错误: {e}")
