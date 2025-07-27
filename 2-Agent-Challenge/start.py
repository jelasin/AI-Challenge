# -*- coding: utf-8 -*-
"""
LangGraph Agent 挑战系列 - 快速入门脚本
运行此脚本来快速体验各个挑战的核心功能
"""

import os
import sys
import subprocess
from pathlib import Path

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
    
    # 检查必要的包
    required_packages = [
        "langgraph",
        "langchain",
        "langchain-openai",
        "langchain-core"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def show_menu():
    """显示挑战菜单"""
    challenges = {
        1: "基础状态图Agent - 学习StateGraph基础概念和节点定义",
        2: "条件路由和工具调用 - 掌握条件边和工具集成技术", 
        3: "并行处理和子图 - 理解并行执行和子图设计模式",
        4: "持久化和断点续传 - 实现状态持久化和故障恢复",
        5: "人机交互和审批流程 - 构建Human-in-the-loop系统",
        6: "高级记忆和多Agent系统 - 多Agent协作和分层记忆",
        7: "流式处理和实时Agent - 掌握流式处理和实时响应技术",
        8: "企业级多Agent协作系统 - 构建完整的企业级Agent架构"
    }
    
    print("🤖 LangGraph Agent 挑战系列")
    print("=" * 60)
    print("选择一个挑战开始学习:")
    print()
    
    for num, desc in challenges.items():
        status = "✅" if Path(f"challenge-{num}/main.py").exists() else "❌"
        print(f"{status} {num}. {desc}")
    
    print("\n0. 退出")
    print("q. 快速演示所有挑战")
    print("=" * 60)

def run_challenge(challenge_num):
    """运行指定的挑战"""
    challenge_dir = f"challenge-{challenge_num}"
    main_file = Path(challenge_dir) / "main.py"
    
    if not main_file.exists():
        print(f"❌ 挑战 {challenge_num} 的main.py文件不存在!")
        print(f"📁 请检查目录: {challenge_dir}")
        return False
    
    print(f"🚀 启动挑战 {challenge_num}...")
    print(f"📁 工作目录: {challenge_dir}")
    print("-" * 40)
    
    try:
        # 切换到挑战目录
        original_dir = os.getcwd()
        os.chdir(challenge_dir)
        
        # 运行挑战
        result = subprocess.run([sys.executable, "main.py"], 
                              capture_output=False, 
                              text=True)
        
        # 恢复原目录
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print(f"\n✅ 挑战 {challenge_num} 完成!")
        else:
            print(f"\n❌ 挑战 {challenge_num} 执行出错!")
            
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print(f"\n⏸️  挑战 {challenge_num} 被中断")
        os.chdir(original_dir)
        return False
    except Exception as e:
        print(f"\n❌ 运行挑战时出错: {e}")
        os.chdir(original_dir)
        return False

def quick_demo():
    """快速演示所有挑战"""
    print("🚀 快速演示模式")
    print("=" * 40)
    
    for i in range(1, 9):  # 更新为8个挑战
        if Path(f"challenge-{i}/main.py").exists():
            print(f"\n📋 挑战 {i} 演示:")
            print("-" * 20)
            
            # 这里可以运行每个挑战的简化版本
            # 或者显示挑战的核心代码片段
            challenge_file = Path(f"challenge-{i}/main.py")
            try:
                with open(challenge_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 提取挑战描述
                    if '"""' in content:
                        desc_start = content.find('"""')
                        desc_end = content.find('"""', desc_start + 3)
                        if desc_end > desc_start:
                            description = content[desc_start+3:desc_end].strip()
                            print(description[:200] + "...")
            except Exception as e:
                print(f"无法读取挑战 {i}: {e}")
        else:
            print(f"❌ 挑战 {i} 不存在")

def start_application():
    """启动应用程序的主函数"""
    print("🚀 LangGraph Agent 挑战系列启动器")
    print("=" * 50)
    
    # 检查环境
    if not check_requirements():
        print("\n❌ 环境检查失败，请解决上述问题后重试")
        return
    
    print("\n✅ 环境检查通过!")
    print("=" * 50)
    
    # 处理命令行参数
    if len(sys.argv) > 1:
        try:
            challenge_num = int(sys.argv[1])
            if 1 <= challenge_num <= 8:
                run_challenge(challenge_num)
                return
            else:
                print("❌ 挑战编号必须在1-8之间")
                return
        except ValueError:
            if sys.argv[1].lower() in ['q', 'quick']:
                quick_demo()
                return
            else:
                print("❌ 无效的参数")
                return
    
    # 交互式菜单
    while True:
        print()
        show_menu()
        
        try:
            choice = input("\n请选择挑战 (0-8, q): ").strip().lower()
            
            if choice == '0':
                print("👋 再见!")
                break
            elif choice in ['q', 'quick']:
                quick_demo()
            elif choice.isdigit():
                challenge_num = int(choice)
                if 1 <= challenge_num <= 8:
                    run_challenge(challenge_num)
                else:
                    print("❌ 请选择1-8之间的数字")
            else:
                print("❌ 无效选择")
                
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 出错了: {e}")

if __name__ == "__main__":
    start_application()
