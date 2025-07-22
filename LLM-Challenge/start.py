# -*- coding: utf-8 -*-
"""
LangChain Challenge 系列 - 快速入门脚本
运行此脚本来快速体验各个挑战的核心功能
"""

import os
import sys
import subprocess

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
    
    # 检查依赖包
    required_packages = [
        'langchain',
        'langchain-openai', 
        'langchain-community',
        'faiss-cpu',
        'pydantic'
    ]
    
    try:
        # 设置环境变量确保UTF-8编码
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # 获取已安装包列表
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, check=True,
                              env=env, encoding='utf-8', errors='replace')
        
        # 解析包列表，跳过表头
        lines = result.stdout.strip().split('\n')
        installed_packages = []
        for line in lines[2:]:  # 跳过前两行表头
            if line.strip():
                parts = line.split()
                if parts:
                    installed_packages.append(parts[0].lower())
        
        missing_packages = []
        for package in required_packages:
            package_lower = package.lower()
            if package_lower in installed_packages:
                print(f"✅ {package}")
            else:
                missing_packages.append(package)
                print(f"❌ {package} 未安装")
        
        if missing_packages:
            print("\n📦 安装缺失的依赖包:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 无法检查已安装包，请确保pip可用: {e}")
        return False
    except Exception as e:
        print(f"❌ 检查依赖包时发生错误: {e}")
        return False
    
    return True

def show_menu():
    """显示挑战菜单"""
    print("\n🎯 LangChain Challenge 系列")
    print("=" * 50)
    print("1. Challenge 1: 基础翻译器 (⭐)")
    print("2. Challenge 2: 工具调用系统 (⭐⭐)")  
    print("3. Challenge 3: 高级Prompt和Few-shot Learning (⭐⭐⭐)")
    print("4. Challenge 4: 文档处理和RAG (⭐⭐⭐⭐)")
    print("5. Challenge 5: LCEL和链组合 (⭐⭐⭐⭐)")
    print("6. Challenge 6: 智能Agent系统 (⭐⭐⭐⭐⭐)")
    print("7. Challenge 7: 流式处理和异步 (⭐⭐⭐⭐⭐)")
    print("8. Challenge 8: 综合企业级系统 (⭐⭐⭐⭐⭐⭐)")
    print("9. 运行所有挑战")
    print("0. 退出")
    print("=" * 50)

def run_challenge(challenge_num):
    """运行指定挑战"""
    challenge_dir = f"challenge-{challenge_num}"
    main_file = os.path.join(challenge_dir, "main.py")
    
    if not os.path.exists(main_file):
        print(f"❌ 找不到挑战文件: {main_file}")
        return False
    
    print(f"\n🚀 运行 Challenge {challenge_num}...")
    print("=" * 40)
    
    try:
        # 切换到挑战目录
        original_dir = os.getcwd()
        os.chdir(challenge_dir)
        
        # 执行挑战，指定UTF-8编码
        with open("main.py", "r", encoding="utf-8") as f:
            exec(f.read())
        
        # 切换回原目录
        os.chdir(original_dir)
        
        print(f"\n✅ Challenge {challenge_num} 执行完成!")
        return True
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        os.chdir(original_dir)
        return False

def run_all_challenges():
    """运行所有挑战"""
    print("\n🎯 运行所有挑战...")
    
    success_count = 0
    for i in range(1, 9):
        print(f"\n{'='*60}")
        print(f"开始 Challenge {i}")
        print(f"{'='*60}")
        
        if run_challenge(i):
            success_count += 1
        
        input("\n按回车键继续下一个挑战...")
    
    print(f"\n🎉 完成! 成功运行了 {success_count}/8 个挑战")

def show_learning_tips():
    """显示学习建议"""
    print("\n💡 学习建议:")
    print("-" * 30)
    print("1. 按顺序完成挑战，每个挑战都基于前面的知识")
    print("2. 仔细阅读代码注释和文档字符串")
    print("3. 尝试修改代码参数，观察不同的结果")
    print("4. 完成每个挑战末尾的练习任务")
    print("5. 遇到问题时查阅LangChain官方文档")
    print("\n📚 相关资源:")
    print("- 官方文档: https://python.langchain.com/")
    print("- GitHub: https://github.com/langchain-ai/langchain")

def main():
    """主函数"""
    print("🌟 欢迎来到 LangChain Challenge 系列!")
    print("这是一个全面学习LangChain v0.3的实战项目")
    
    # 检查环境
    if not check_requirements():
        print("\n❌ 环境检查失败，请先解决上述问题")
        return
    
    while True:
        show_menu()
        
        try:
            choice = input("\n请选择挑战 (0-9): ").strip()
            
            if choice == '0':
                print("👋 再见! 祝学习愉快!")
                break
            elif choice == '9':
                run_all_challenges()
            elif choice in [str(i) for i in range(1, 9)]:
                challenge_num = int(choice)
                run_challenge(challenge_num)
            else:
                print("❌ 无效选择，请输入0-9")
                continue
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
        
        # 显示学习提示
        if input("\n是否查看学习建议? (y/N): ").lower() == 'y':
            show_learning_tips()

if __name__ == "__main__":
    main()
