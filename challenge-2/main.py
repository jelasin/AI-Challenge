def caclulate():
    from langchain_core.tools import tool
    from langchain.chat_models import init_chat_model
    from pydantic import BaseModel, Field
    
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

    from langchain_core.messages import HumanMessage

    question = "What is 2 + 3? Also what is 5 * 6?"
    messages = [HumanMessage(question)]

    ai_msg = llm_with_tools.invoke(messages)

    # tool_calls 调用工具，返回参数描述符号
    print(ai_msg)

    # If you want to inspect the AI message for tool calls, print its dict or content
    # For example:
    # print(ai_msg.__dict__)

    # If the AI message contains tool calls in a different attribute, update accordingly.
    # Example (uncomment and adjust if needed):
    # for tool_call in ai_msg.some_tool_calls_attribute:
    #     selected_tool = {"Addition": Addition, "Multiply": Multiply}[tool_call["name"]]
    #     tool_msg = selected_tool.invoke(tool_call)
    #     messages.append(tool_msg)
        
    class Result(BaseModel):
        question: str = Field(..., description="Question asked by the user")
        result: int = Field(..., description="Result of the calculation")

    
    print(messages)

if __name__ == '__main__':
    caclulate()