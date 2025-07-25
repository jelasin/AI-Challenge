"""
LangGraph Agent 挑战系列启动器

这个脚本帮助你快速开始任何一个LangGraph挑战。
每个挑战都专注于LangGraph的特定功能和概念。

使用方法:
python start.py [challenge_number]
"""

import sys
import os
import subprocess
from pathlib import Path

def show_menu():
    """显示挑战菜单"""
    challenges = {
        1: "基础状态图Agent - 学习StateGraph基础",
        2: "条件路由和工具调用 - 掌握条件边和工具集成", 
        3: "并行处理和子图 - 理解并行执行和子图设计",
        4: "检查点和状态持久化 - 实现故障恢复和持久化",
        5: "人机交互和审批流程 - 构建Human-in-the-loop系统",
        6: "高级记忆和多Agent系统 - 多Agent协作和高级记忆"
    }
    
    print("🤖 LangGraph Agent 挑战系列")
    print("=" * 50)
    print("选择一个挑战开始学习:")
    print()
    
    for num, desc in challenges.items():
        print(f"{num}. {desc}")
    
    print("\n0. 退出")
    print("=" * 50)

def run_challenge(challenge_num):
    """运行指定的挑战"""
    challenge_dir = f"challenge-{challenge_num}"
    main_file = Path(challenge_dir) / "main.py"
    
    if not main_file.exists():
        print(f"❌ 挑战 {challenge_num} 不存在!")
        return False
    
    print(f"🚀 启动挑战 {challenge_num}...")
    print(f"📁 工作目录: {challenge_dir}")
    print("-" * 30)
    
    try:
        # 切换到挑战目录并运行
        os.chdir(challenge_dir)
        result = subprocess.run([sys.executable, "main.py"], 
                              capture_output=False, 
                              text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        return False
    finally:
        # 返回原目录
        os.chdir("..")

def check_environment():
    """检查环境依赖"""
    try:
        import langgraph
        import langchain
        print("✅ LangGraph环境检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install langgraph langchain langchain-openai")
        return False

def main():
    """主函数"""
    print("🔍 检查环境...")
    if not check_environment():
        return
    
    # 如果提供了命令行参数
    if len(sys.argv) > 1:
        try:
            challenge_num = int(sys.argv[1])
            if 1 <= challenge_num <= 6:
                run_challenge(challenge_num)
            else:
                print("❌ 挑战编号必须在1-6之间")
        except ValueError:
            print("❌ 请提供有效的挑战编号")
        return
    
    # 交互式菜单
    while True:
        show_menu()
        try:
            choice = input("\n请选择挑战编号 (0-6): ").strip()
            
            if choice == "0":
                print("👋 再见!")
                break
            
            challenge_num = int(choice)
            if 1 <= challenge_num <= 6:
                success = run_challenge(challenge_num)
                if success:
                    print(f"\n✅ 挑战 {challenge_num} 完成!")
                else:
                    print(f"\n❌ 挑战 {challenge_num} 未能完成")
                
                input("\n按回车键继续...")
            else:
                print("❌ 请选择有效的编号 (0-6)")
                
        except (ValueError, KeyboardInterrupt):
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()
