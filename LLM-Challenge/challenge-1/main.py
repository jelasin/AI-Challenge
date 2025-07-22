# -*- coding: utf-8 -*-
"""
Challenge 1: LangChain基础翻译器
==================================

学习目标：
1. 掌握LangChain的基本概念和架构
2. 学习使用ChatOpenAI模型
3. 理解PromptTemplate的作用和使用方法
4. 掌握结构化输出（Structured Output）
5. 学习错误处理和环境配置

核心知识点：
- ChatOpenAI: OpenAI模型的LangChain封装
- PromptTemplate: 提示词模板，支持变量替换
- Pydantic BaseModel: 数据验证和结构化输出
- with_structured_output(): 让模型输出结构化数据
- 环境变量管理和错误处理

使用方法：
1. 设置环境变量: $env:OPENAI_API_KEY='your-key'
2. 安装依赖: pip install langchain langchain-openai pydantic
3. 运行程序: python main.py
"""

def EasyTranslate():
    """
    智能翻译器主函数
    
    功能描述：
    - 接受用户输入的源语言、目标语言和待翻译文本
    - 使用OpenAI GPT模型进行翻译
    - 返回结构化的翻译结果，包含译文、语言识别和置信度评估
    
    技术实现：
    - 使用ChatOpenAI作为语言模型
    - 通过PromptTemplate构建标准化提示词
    - 利用Pydantic模型定义结构化输出格式
    - 实现完整的错误处理机制
    """
    # 导入必要的库和模块
    import os
    from langchain_openai import ChatOpenAI         # OpenAI模型的LangChain封装
    from langchain_core.prompts import PromptTemplate  # 提示词模板工具
    from langchain_core.messages import HumanMessage   # 消息类型定义
    from pydantic import BaseModel, Field          # 数据验证和结构化建模
    from typing import Optional                     # 类型注解支持

    # 定义翻译结果的结构化模型
    # 使用Pydantic BaseModel确保输出数据的结构和类型正确性
    class TranslationResult(BaseModel):
        """
        翻译结果的数据模型
        
        属性说明：
        - translated_text: 翻译后的文本内容
        - source_language: 自动检测的源语言
        - target_language: 用户指定的目标语言
        - confidence: 翻译质量的置信度评估
        """
        translated_text: str = Field(description="翻译后的文本")
        source_language: str = Field(description="检测到的源语言") 
        target_language: str = Field(description="目标语言")
        confidence: Optional[str] = Field(description="翻译置信度评估", default="high")
        
    # 步骤1: 检查OpenAI API密钥配置
    # API密钥是访问OpenAI服务的必要凭证
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 错误：请设置 OPENAI_API_KEY 环境变量")
        print("💡 您可以通过以下方式设置：")
        print("   Windows PowerShell: $env:OPENAI_API_KEY='your-api-key-here'")
        print("   Linux/Mac: export OPENAI_API_KEY='your-api-key-here'")
        return

    # 步骤2: 初始化ChatOpenAI模型
    # 配置模型参数以获得最佳翻译效果
    try:
        llm = ChatOpenAI(
            model="gpt-4o",        # 使用GPT-4 Omni模型，具有优秀的多语言能力
            temperature=0,         # 设置为0确保翻译结果的一致性和准确性
            streaming=False        # 禁用流式输出，确保与structured_output兼容
        ).with_structured_output(TranslationResult)  # 绑定结构化输出模型
        
        print("✅ ChatOpenAI模型初始化成功")
        
    except Exception as e:
        print(f"❌ 初始化 ChatOpenAI 时出错: {e}")
        print("💡 可能的解决方案：")
        print("   1. 检查API密钥是否有效")
        print("   2. 确认网络连接正常")
        print("   3. 验证langchain-openai库已正确安装")
        return

    # 步骤3: 创建提示词模板
    # PromptTemplate允许我们创建可复用的提示词模板，支持变量替换
    prompt = PromptTemplate.from_template(
        """你是一个专业的翻译助手。请将以下文本从{src_language}准确翻译成{dst_language}。

翻译要求：
1. 保持原文的语义和语调
2. 确保译文自然流畅
3. 如果遇到专业术语，请提供准确的对应翻译
4. 评估你对这次翻译质量的置信度

原文：{text_message}

请提供结构化的翻译结果。"""
    )

    # 步骤4: 获取用户输入
    # 收集翻译所需的基本信息
    print("\n🌍 欢迎使用LangChain智能翻译器")
    print("=" * 40)
    
    src_language = input("📝 请输入源语言 (如: 中文, English): ").strip()
    if not src_language:
        src_language = "自动检测"
        
    dst_language = input("🎯 请输入目标语言 (如: 英文, 中文): ").strip()
    if not dst_language:
        print("❌ 目标语言不能为空")
        return
        
    text_message = input("📄 请输入要翻译的文本: ").strip()
    if not text_message:
        print("❌ 翻译文本不能为空")
        return

    # 步骤5: 构建完整的提示词
    # 使用模板格式化方法将用户输入填入提示词模板
    try:
        message = prompt.format(
            src_language=src_language, 
            dst_language=dst_language, 
            text_message=text_message
        ) 
        
        print(f"\n🔄 正在调用GPT-4o进行翻译...")
        print(f"📊 文本长度: {len(text_message)} 字符")
        
    except Exception as e:
        print(f"❌ 构建提示词时出错: {e}")
        return

    # 步骤6: 调用模型进行翻译
    # 使用结构化输出确保返回的数据格式正确
    try:
        # 将格式化后的提示词包装成HumanMessage并发送给模型
        result = llm.invoke([HumanMessage(content=message)])
        
        # 兼容性处理函数：统一访问结果数据的方式
        # 处理可能的字典或对象返回格式
        def get_field(obj, field_name, default="未知"):
            """
            安全获取对象字段值的工具函数
            
            Args:
                obj: 结果对象（可能是dict或Pydantic模型）
                field_name: 字段名
                default: 默认值
                
            Returns:
                字段值或默认值
            """
            if isinstance(obj, dict):
                return obj.get(field_name, default)
            else:
                return getattr(obj, field_name, default)
        
        # 步骤7: 美观地展示翻译结果
        print("\n" + "=" * 60)
        print("🎉 翻译完成！结果如下：")
        print("=" * 60)
        print(f"🔤 源语言：{get_field(result, 'source_language')}")
        print(f"🎯 目标语言：{get_field(result, 'target_language')}")
        print(f"📄 原文：{text_message}")
        print(f"✨ 译文：{get_field(result, 'translated_text')}")
        print(f"📊 置信度：{get_field(result, 'confidence')}")
        print("=" * 60)
        print("💡 提示：置信度表示AI对翻译质量的自我评估")
        
    except Exception as e:
        print(f"❌ 翻译过程中出错: {e}")
        print("\n🛠 故障排除建议：")
        print("1. 检查OPENAI_API_KEY环境变量是否正确设置")
        print("2. 确认API密钥是否有效且有足够余额")
        print("3. 检查网络连接是否正常")
        print("4. 确认所需库已安装：pip install langchain langchain-openai pydantic")
        print("5. 尝试使用较短的文本进行测试")

def demo_advanced_features():
    """
    演示高级功能的示例函数
    
    展示更多LangChain功能：
    - 批量翻译
    - 多语言检测
    - 翻译质量评估
    """
    print("🚀 高级功能演示")
    print("这些功能将在后续的挑战中详细学习：")
    print("- Challenge 3: Few-shot Learning和示例选择")
    print("- Challenge 4: 文档处理和批量翻译")
    print("- Challenge 5: 链式处理和工作流")
    

if __name__ == "__main__":
    """
    程序入口点
    
    执行流程：
    1. 调用EasyTranslate()函数启动翻译器
    2. 用户交互式输入翻译需求
    3. 展示翻译结果
    4. 可选：演示高级功能
    """
    print("🎯 LangChain Challenge 1: 基础翻译器")
    print("学习目标：掌握ChatOpenAI、PromptTemplate和结构化输出")
    print("-" * 50)
    
    # 主功能：智能翻译
    EasyTranslate()
    
    # 可选：演示高级功能
    print("\n" + "=" * 60)
    user_choice = input("是否查看高级功能演示？(y/N): ").lower().strip()
    if user_choice in ['y', 'yes', '是']:
        demo_advanced_features()
    
    print("\n✅ Challenge 1 完成！")
    print("💡 下一步：尝试 Challenge 2 - 工具调用系统")
    print("🔗 继续学习：https://python.langchain.com/docs/concepts/")