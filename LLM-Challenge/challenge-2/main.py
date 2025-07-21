def calculate():
    from langchain_core.tools import tool
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
    from pydantic import BaseModel, Field
    from typing import List
    
    llm = init_chat_model(model="gpt-4o", model_provider="openai")

    # 加法运算参数描述
    class AdditionInput(BaseModel):
        a: int = Field(..., description="First number")
        b: int = Field(..., description="Second number")

    # 定义加法运算工具，绑定参数，函数描述不可少
    # 修饰器tool的第一个参数是工具名称，第二个参数是参数描述
    @tool("Addition", args_schema=AdditionInput)
    def Addition(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    # 乘法运算参数描述
    class MultiplyInput(BaseModel):
        a: int = Field(..., description="First number")
        b: int = Field(..., description="Second number")
    
    # 定义乘法运算工具，绑定参数，函数描述不可少
    # 修饰器tool的第一个参数是工具名称，第二个参数是参数描述
    @tool("Multiply", args_schema=MultiplyInput)
    def Multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    tools = [Addition, Multiply]
    llm_with_tools = llm.bind_tools(tools)

    question = "What is 21356 + 99487? Also what is 12347 * 12958?"
    messages: List[BaseMessage] = [HumanMessage(question)]

    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # 处理工具调用
    tool_calls = getattr(ai_msg, 'tool_calls', None)
    if tool_calls:
        tool_map = {"Addition": Addition, "Multiply": Multiply}
        
        for tool_call in tool_calls:
            selected_tool = tool_map[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"])
            tool_msg = ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            )
            messages.append(tool_msg)

    # 获取最终回答
    final_response = llm_with_tools.invoke(messages)
    # 如果需要结构化结果，可以使用Result类
    class Result(BaseModel):
        question: str = Field(..., description="Question asked by the user")
        answer: str = Field(..., description="Final answer from the AI")
    
    result = Result(
        question=question,
        answer=str(final_response.content)
    )
    
    print(f"问题: {result.question}")
    print(f"答案: {result.answer}")

if __name__ == '__main__':
    calculate()