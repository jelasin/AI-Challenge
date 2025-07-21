def EasyTranslate():
    """
    EasyTranslate using langchain and openai
    轻松翻译，使用 langchain 和 openai
    """
    import os
    from langchain_openai import ChatOpenAI # openai for langchain
    from langchain_core.prompts import PromptTemplate # prompt template for langchain
    from langchain_core.messages import HumanMessage

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
        )
    except Exception as e:
        print(f"初始化 ChatOpenAI 时出错: {e}")
        return

    # prompt template
    prompt = PromptTemplate.from_template(
        "把下面这句话从{src_language}翻译成{dst_language} : {text_message}"
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

    print(message)

    # 使用简单的invoke调用，不使用structured_output
    try:
        result = llm.invoke([HumanMessage(content=message)])
        
        print(f"源语言：{src_language}")
        print(f"目标语言：{dst_language}")
        print(f"要翻译的文本：{text_message}")
        print(f"翻译后的文本：{result.content}")
    except Exception as e:
        print(f"翻译时出错: {e}")
        print("请检查：")
        print("1. OPENAI_API_KEY 环境变量是否正确设置")
        print("2. API 密钥是否有效")
        print("3. 网络连接是否正常")


if __name__ == "__main__":
    EasyTranslate()