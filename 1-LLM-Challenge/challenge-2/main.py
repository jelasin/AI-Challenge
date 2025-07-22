# -*- coding: utf-8 -*-
"""
Challenge 2: LangChain工具调用系统
===================================

学习目标：
1. 理解Tool的概念和作用机制
2. 学习使用@tool装饰器创建自定义工具
3. 掌握Function Calling的基本原理
4. 理解工具绑定和调用流程
5. 学习处理多工具协调和结果集成

核心知识点：
- @tool装饰器: 将普通函数转换为LangChain工具
- args_schema: 使用Pydantic模型定义工具参数
- bind_tools(): 将工具绑定到语言模型
- tool_calls: 模型决定调用哪些工具及参数
- ToolMessage: 工具执行结果的消息格式

技术架构：
用户问题 → LLM分析 → 选择工具 → 执行工具 → 返回结果 → LLM总结

使用方法：
1. 设置环境变量: $env:OPENAI_API_KEY='your-key'
2. 安装依赖: pip install langchain langchain-openai
3. 运行程序: python main.py
"""

def calculate():
    """
    智能计算器主函数 - 工具调用系统演示
    
    功能描述：
    - 使用自然语言描述数学计算需求
    - AI自动选择和调用相应的数学工具
    - 支持多步骤计算和复杂表达式
    - 展示完整的工具调用生命周期
    
    工作流程：
    1. 创建数学运算工具(加法、乘法)
    2. 将工具绑定到语言模型
    3. 用户提出数学问题
    4. LLM分析问题并决定调用哪些工具
    5. 执行工具调用并收集结果
    6. LLM整合结果并给出最终答案
    
    关键技术点：
    - Function Calling: GPT模型的函数调用能力
    - Tool Schema: 工具的参数定义和验证
    - Message Flow: 消息在LLM和工具间的传递
    """
    # 导入必要的LangChain组件
    from langchain_core.tools import tool                    # 工具装饰器
    from langchain.chat_models import init_chat_model       # 通用模型初始化函数
    from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage  # 消息类型
    from pydantic import BaseModel, Field                   # 数据验证和建模
    from typing import List                                  # 类型注解
    import os                                               # 环境变量管理
    
    # 步骤1: 检查环境配置
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误：请设置 OPENAI_API_KEY 环境变量")
        print("💡 设置方法: $env:OPENAI_API_KEY='your-api-key'")
        return
    
    # 步骤2: 初始化支持工具调用的语言模型
    # init_chat_model是LangChain v0.3的新特性，提供统一的模型初始化接口
    print("🚀 初始化支持工具调用的GPT-4o模型...")
    try:
        llm = init_chat_model(model="gpt-4o", model_provider="openai")
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

    print("\n🛠 创建数学运算工具...")
    
    # 步骤3: 定义工具的输入参数模型
    # 使用Pydantic BaseModel确保参数类型和格式的正确性
    class AdditionInput(BaseModel):
        """
        加法运算的参数模型
        
        属性：
        - a: 第一个加数 (必需)
        - b: 第二个加数 (必需)
        """
        a: int = Field(..., description="第一个加数")
        b: int = Field(..., description="第二个加数")

    # 步骤4: 使用@tool装饰器创建加法工具
    # 装饰器将普通Python函数转换为LangChain工具
    @tool("Addition", args_schema=AdditionInput)
    def Addition(a: int, b: int) -> int:
        """
        执行两个整数的加法运算
        
        Args:
            a (int): 第一个加数
            b (int): 第二个加数
            
        Returns:
            int: 加法运算的结果
            
        示例:
            Addition(5, 3) -> 8
        """
        result = a + b
        print(f"   🔢 执行加法: {a} + {b} = {result}")
        return result

    # 定义乘法运算的参数模型
    class MultiplyInput(BaseModel):
        """
        乘法运算的参数模型
        
        属性：
        - a: 被乘数 (必需)
        - b: 乘数 (必需)
        """
        a: int = Field(..., description="被乘数")
        b: int = Field(..., description="乘数")
    
    # 使用@tool装饰器创建乘法工具
    @tool("Multiply", args_schema=MultiplyInput)
    def Multiply(a: int, b: int) -> int:
        """
        执行两个整数的乘法运算
        
        Args:
            a (int): 被乘数
            b (int): 乘数
            
        Returns:
            int: 乘法运算的结果
            
        示例:
            Multiply(6, 7) -> 42
        """
        result = a * b
        print(f"   🔢 执行乘法: {a} × {b} = {result}")
        return result

    # 步骤5: 创建工具集合并绑定到模型
    # 工具列表包含所有可供模型调用的工具
    tools = [Addition, Multiply]
    
    # bind_tools()方法告诉LLM哪些工具可用
    # 模型会根据用户问题自动选择合适的工具
    llm_with_tools = llm.bind_tools(tools)
    print(f"✅ 成功绑定 {len(tools)} 个工具到模型")

    # 步骤6: 准备测试问题
    # 这个问题需要调用两个不同的工具来解决
    question = "请帮我计算：21356 + 99487 等于多少？另外，12347 × 12958 的结果是什么？"
    print(f"\n❓ 用户问题: {question}")
    
    # 创建消息历史列表，用于跟踪对话流程
    messages: List[BaseMessage] = [HumanMessage(content=question)]

    # 步骤7: 第一次调用模型 - 分析问题并决定工具调用
    print("\n🧠 LLM分析问题并规划工具调用...")
    try:
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)  # 将AI的响应添加到消息历史
        
        # 检查AI是否决定调用工具
        tool_calls = getattr(ai_msg, 'tool_calls', None)
        if not tool_calls:
            print("⚠️  模型没有调用任何工具，直接给出了回答")
            print(f"回答: {ai_msg.content}")
            return
            
        print(f"🎯 模型计划调用 {len(tool_calls)} 个工具:")
        for i, tool_call in enumerate(tool_calls, 1):
            print(f"   {i}. 工具: {tool_call['name']}")
            print(f"      参数: {tool_call['args']}")
            
    except Exception as e:
        print(f"❌ 模型调用失败: {e}")
        return

    # 步骤8: 执行工具调用并收集结果
    print("\n⚙️  执行工具调用...")
    
    # 创建工具名称到工具对象的映射
    tool_map = {"Addition": Addition, "Multiply": Multiply}
    
    try:
        # 遍历所有工具调用请求
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            
            # 根据工具名称获取对应的工具对象
            selected_tool = tool_map[tool_name]
            
            # 执行工具调用
            print(f"   🔧 调用工具: {tool_name}")
            tool_output = selected_tool.invoke(tool_args)
            
            # 将工具执行结果包装成ToolMessage
            # ToolMessage用于将工具结果返回给LLM
            tool_msg = ToolMessage(
                content=str(tool_output),      # 工具执行结果
                tool_call_id=tool_call_id     # 工具调用的唯一标识符
            )
            messages.append(tool_msg)  # 添加到消息历史
            
    except Exception as e:
        print(f"❌ 工具执行失败: {e}")
        return

    # 步骤9: 第二次调用模型 - 整合工具结果生成最终答案
    print("\n🤖 LLM整合工具结果，生成最终答案...")
    try:
        final_response = llm_with_tools.invoke(messages)
        
        # 定义结果数据模型，用于结构化存储问答结果
        class CalculationResult(BaseModel):
            """
            计算结果的数据模型
            
            属性：
            - question: 用户提出的原始问题
            - answer: AI给出的最终答案
            - tools_used: 使用的工具列表
            - calculation_steps: 计算步骤
            """
            question: str = Field(description="用户提出的问题")
            answer: str = Field(description="AI的最终答案")
            tools_used: List[str] = Field(description="使用的工具列表")
        
        # 创建结构化结果
        result = CalculationResult(
            question=question,
            answer=str(final_response.content),
            tools_used=[call["name"] for call in tool_calls]
        )
        
        # 步骤10: 美观地展示最终结果
        print("\n" + "=" * 60)
        print("🎉 计算完成！结果如下：")
        print("=" * 60)
        print(f"❓ 问题: {result.question}")
        print(f"🛠  使用工具: {', '.join(result.tools_used)}")
        print(f"✅ 答案: {result.answer}")
        print("=" * 60)
        
        # 展示完整的消息流程（可选）
        print("\n💭 完整的对话流程:")
        for i, msg in enumerate(messages, 1):
            if isinstance(msg, HumanMessage):
                print(f"   {i}. 👤 用户: {msg.content[:100]}...")
            elif hasattr(msg, 'tool_calls') and getattr(msg, 'tool_calls', None):
                print(f"   {i}. 🤖 AI: 计划调用 {len(getattr(msg, 'tool_calls', []))} 个工具")
            elif isinstance(msg, ToolMessage):
                print(f"   {i}. 🔧 工具结果: {msg.content}")
            else:
                print(f"   {i}. 🤖 AI最终回答: {str(msg.content)[:100]}...")
                
    except Exception as e:
        print(f"❌ 生成最终答案失败: {e}")


def demo_advanced_tool_features():
    """
    演示高级工具功能
    
    展示更多工具相关的LangChain功能：
    - 工具错误处理
    - 条件工具调用
    - 工具链组合
    - 异步工具调用
    """
    print("\n🚀 高级工具功能预览")
    print("以下功能将在后续挑战中详细学习：")
    print("- Challenge 4: 文档处理工具和RAG系统")
    print("- Challenge 5: 工具链组合和LCEL")
    print("- Challenge 6: 智能Agent和工具集成")
    print("- 工具错误处理和重试机制")
    print("- 并行工具调用和结果聚合")
    print("- 自定义工具开发和工具市场")


if __name__ == '__main__':
    """
    程序入口点
    
    执行流程：
    1. 运行智能计算器演示
    2. 展示工具调用的完整生命周期
    3. 可选：预览高级工具功能
    """
    print("🎯 LangChain Challenge 2: 工具调用系统")
    print("学习目标：掌握@tool装饰器、Function Calling和工具集成")
    print("-" * 60)
    
    # 主功能：智能计算器
    calculate()
    
    # 可选：预览高级功能
    print("\n" + "=" * 70)
    user_choice = input("是否查看高级工具功能预览？(y/N): ").lower().strip()
    if user_choice in ['y', 'yes', '是']:
        demo_advanced_tool_features()
    
    print("\n✅ Challenge 2 完成！")
    print("🎓 你已经掌握了：")
    print("   - 工具创建和绑定")
    print("   - Function Calling机制")  
    print("   - 多工具协调处理")
    print("   - 结果集成和展示")
    print("\n💡 下一步：尝试 Challenge 3 - 高级Prompt和Few-shot Learning")
    print("🔗 深入学习：https://python.langchain.com/docs/concepts/tools/")