def EasyTranslate():
    """
    EasyTranslate using langchain and openai
    轻松翻译，使用 langchain 和 openai
    """
    import os
    from langchain_openai import ChatOpenAI # openai for langchain
    from langchain_core.prompts import PromptTemplate # prompt template for langchain
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel, Field
    from typing import Optional

    # 定义翻译结果的结构化模型
    class TranslationResult(BaseModel):
        """翻译结果的结构化输出模型"""
        translated_text: str = Field(description="翻译后的文本")
        source_language: str = Field(description="检测到的源语言")
        target_language: str = Field(description="目标语言")
        confidence: Optional[str] = Field(description="翻译置信度评估", default="high")
        
    # 检查 API 密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：请设置 OPENAI_API_KEY 环境变量")
        print("您可以通过以下方式设置：")
        print("$env:OPENAI_API_KEY='your-api-key-here'")
        return

    try:
        llm = ChatOpenAI(
            model="gpt-4o", # gpt-4o for openai
            temperature=0, # 创作自由度，越高越自由，越低越严谨
            streaming=False # 禁用streaming以避免与structured_output冲突
        ).with_structured_output(TranslationResult)
    except Exception as e:
        print(f"初始化 ChatOpenAI 时出错: {e}")
        return

    # 更新 prompt template 以支持结构化输出
    prompt = PromptTemplate.from_template(
        """请将以下文本从{src_language}翻译成{dst_language}，并提供结构化的翻译结果：
        
        原文：{text_message}
        
        请确保翻译准确、自然，并评估翻译的置信度。"""
    )

    src_language = input("请输入源语言: ")
    dst_language = input("请输入目标语言: ")
    text_message = input("请输入要翻译的文本: ")

    # message for prompt template
    message = prompt.format(
        src_language=src_language, 
        dst_language=dst_language, 
        text_message=text_message,
    ) 

    print("\n正在翻译...\n")

    # 使用结构化输出调用
    try:
        result = llm.invoke([HumanMessage(content=message)])
        
        # 兼容处理：检查返回类型并统一访问方式
        def get_field(obj, field_name, default="未知"):
            if isinstance(obj, dict):
                return obj.get(field_name, default)
            else:
                return getattr(obj, field_name, default)
        
        # 结构化输出展示
        print("=" * 50)
        print("📝 翻译结果")
        print("=" * 50)
        print(f"🔤 源语言：{get_field(result, 'source_language')}")
        print(f"🎯 目标语言：{get_field(result, 'target_language')}")
        print(f"📄 原文：{text_message}")
        print(f"✨ 译文：{get_field(result, 'translated_text')}")
        print(f"🎯 置信度：{get_field(result, 'confidence')}")
        print("=" * 50)
        
    except Exception as e:
        print(f"翻译时出错: {e}")
        print("请检查：")
        print("1. OPENAI_API_KEY 环境变量是否正确设置")
        print("2. API 密钥是否有效")
        print("3. 网络连接是否正常")
        print("4. Pydantic 库是否已安装 (pip install pydantic)")

if __name__ == "__main__":
    EasyTranslate()