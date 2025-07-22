# -*- coding: utf-8 -*-
"""
Challenge 7: 流式处理和异步操作
难度：高级

学习目标：
1. 掌握LangChain的流式输出
2. 学习异步编程模式
3. 实现事件流处理
4. 构建实时聊天系统
5. 优化大规模并发处理
6. 实现流式数据管道

任务描述：
创建一个高性能的实时AI应用，能够：
1. 支持流式对话和实时响应
2. 处理大量并发请求
3. 实现事件驱动的处理流程
4. 构建流式数据处理管道
5. 监控和优化性能
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List, Dict, Any, Optional, AsyncIterator, Iterator, Callable
import asyncio
import aiohttp
import time
import json
import threading
from datetime import datetime
from dataclasses import dataclass
from collections import deque
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StreamEvent:
    """流事件数据类"""
    event_type: str
    timestamp: datetime
    data: Any
    session_id: Optional[str] = None

class StreamingCallbackHandler(BaseCallbackHandler):
    """流式回调处理器"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.events = deque(maxlen=1000)  # 保持最近1000个事件
        self.current_tokens = []
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """LLM开始时的回调"""
        event = StreamEvent("llm_start", datetime.now(), {"prompts": prompts}, self.session_id)
        self.events.append(event)
        print(f"🚀 [{self.session_id}] LLM开始处理...")
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """新token生成时的回调"""
        self.current_tokens.append(token)
        event = StreamEvent("new_token", datetime.now(), {"token": token}, self.session_id)
        self.events.append(event)
        print(f"📝 [{self.session_id}] 新token: '{token}'", end='', flush=True)
    
    def on_llm_end(self, response, **kwargs) -> None:
        """LLM结束时的回调"""
        full_text = ''.join(self.current_tokens)
        event = StreamEvent("llm_end", datetime.now(), {"response": full_text}, self.session_id)
        self.events.append(event)
        print(f"\\n✅ [{self.session_id}] LLM处理完成")
        self.current_tokens.clear()
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """LLM错误时的回调"""
        event = StreamEvent("llm_error", datetime.now(), {"error": str(error)}, self.session_id)
        self.events.append(event)
        print(f"❌ [{self.session_id}] LLM错误: {error}")

class AsyncStreamingCallbackHandler(AsyncCallbackHandler):
    """异步流式回调处理器"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.events = deque(maxlen=1000)
        self.current_tokens = []
        
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """异步LLM开始回调"""
        event = StreamEvent("async_llm_start", datetime.now(), {"prompts": prompts}, self.session_id)
        self.events.append(event)
        print(f"🚀 [ASYNC-{self.session_id}] LLM开始处理...")
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """异步新token回调"""
        self.current_tokens.append(token)
        event = StreamEvent("async_new_token", datetime.now(), {"token": token}, self.session_id)
        self.events.append(event)
        print(f"📝 [ASYNC-{self.session_id}] 新token: '{token}'", end='', flush=True)
    
    async def on_llm_end(self, response, **kwargs) -> None:
        """异步LLM结束回调"""
        full_text = ''.join(self.current_tokens)
        event = StreamEvent("async_llm_end", datetime.now(), {"response": full_text}, self.session_id)
        self.events.append(event)
        print(f"\\n✅ [ASYNC-{self.session_id}] LLM处理完成")
        self.current_tokens.clear()

class RealTimeChatSystem:
    """实时聊天系统"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.active_streams: Dict[str, bool] = {}
        
        # 检查API密钥
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
        
        # 创建LLM实例
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            streaming=True  # 启用流式输出
        )
        
        # 创建基础链
        prompt = ChatPromptTemplate.from_template(
            "你是一个友好的AI助手。请用轻松对话的方式回答用户的问题：{question}"
        )
        
        self.base_chain = prompt | self.llm | StrOutputParser()
    
    def create_session(self, session_id: str) -> None:
        """创建新的聊天会话"""
        self.sessions[session_id] = {
            "history": [],
            "created_at": datetime.now(),
            "callback_handler": StreamingCallbackHandler(session_id)
        }
        self.active_streams[session_id] = False
        print(f"📱 创建新会话: {session_id}")
    
    def stream_response(self, session_id: str, question: str) -> Iterator[str]:
        """流式响应生成器"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        session = self.sessions[session_id]
        callback_handler = session["callback_handler"]
        
        self.active_streams[session_id] = True
        
        try:
            # 使用流式处理
            for chunk in self.base_chain.stream(
                {"question": question}, 
                config={"callbacks": [callback_handler]}
            ):
                if self.active_streams[session_id]:  # 检查是否应该继续流式传输
                    yield chunk
                else:
                    break
            
            # 更新会话历史
            session["history"].append({"user": question, "assistant": "".join(chunk for chunk in self.base_chain.stream({"question": question}))})
            
        except Exception as e:
            yield f"错误: {str(e)}"
        finally:
            self.active_streams[session_id] = False
    
    def stop_stream(self, session_id: str) -> None:
        """停止指定会话的流式传输"""
        if session_id in self.active_streams:
            self.active_streams[session_id] = False
            print(f"⏹️ 停止会话 {session_id} 的流式传输")

class AsyncProcessingPipeline:
    """异步处理管道"""
    
    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
            
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            streaming=True
        )
        
        self.processing_queue = asyncio.Queue()
        self.results_cache: Dict[str, Any] = {}
        
    async def add_task(self, task_id: str, content: str, task_type: str = "general"):
        """添加任务到处理队列"""
        task = {
            "id": task_id,
            "content": content,
            "type": task_type,
            "timestamp": datetime.now(),
            "status": "queued"
        }
        await self.processing_queue.put(task)
        print(f"📥 任务已加入队列: {task_id}")
    
    async def process_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个任务"""
        try:
            task["status"] = "processing"
            print(f"🔄 开始处理任务: {task['id']}")
            
            # 根据任务类型选择不同的处理逻辑
            if task["type"] == "summarize":
                prompt = ChatPromptTemplate.from_template(
                    "请为以下内容生成一个简洁的摘要：\\n{content}"
                )
            elif task["type"] == "translate":
                prompt = ChatPromptTemplate.from_template(
                    "请将以下内容翻译成英文：\\n{content}"
                )
            elif task["type"] == "analyze":
                prompt = ChatPromptTemplate.from_template(
                    "请分析以下内容的主要观点和关键信息：\\n{content}"
                )
            else:
                prompt = ChatPromptTemplate.from_template(
                    "请处理以下内容：\\n{content}"
                )
            
            # 创建异步回调处理器
            callback_handler = AsyncStreamingCallbackHandler(task["id"])
            
            # 异步执行任务
            chain = prompt | self.llm | StrOutputParser()
            result = await chain.ainvoke(
                {"content": task["content"]}, 
                config={"callbacks": [callback_handler]}
            )
            
            task["status"] = "completed"
            task["result"] = result
            task["completed_at"] = datetime.now()
            
            # 缓存结果
            self.results_cache[task["id"]] = task
            
            print(f"✅ 任务处理完成: {task['id']}")
            return task
            
        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)
            task["failed_at"] = datetime.now()
            print(f"❌ 任务处理失败: {task['id']} - {e}")
            return task
    
    async def batch_process(self, max_concurrent: int = 5):
        """批量处理任务"""
        print(f"🔄 开始批量处理，最大并发数: {max_concurrent}")
        
        async def worker():
            """工作协程"""
            while True:
                try:
                    # 从队列获取任务
                    task = await self.processing_queue.get()
                    
                    # 处理任务
                    await self.process_single_task(task)
                    
                    # 标记任务完成
                    self.processing_queue.task_done()
                    
                except Exception as e:
                    print(f"Worker error: {e}")
        
        # 创建工作协程
        workers = [asyncio.create_task(worker()) for _ in range(max_concurrent)]
        
        try:
            # 等待所有任务完成
            await self.processing_queue.join()
        finally:
            # 取消工作协程
            for worker_task in workers:
                worker_task.cancel()
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务结果"""
        return self.results_cache.get(task_id)

class StreamingDataPipeline:
    """流式数据处理管道"""
    
    def __init__(self):
        self.processors: List[Callable] = []
        self.output_handlers: List[Callable] = []
        
    def add_processor(self, processor: Callable):
        """添加处理器"""
        self.processors.append(processor)
        
    def add_output_handler(self, handler: Callable):
        """添加输出处理器"""
        self.output_handlers.append(handler)
    
    async def process_stream(self, data_stream: AsyncIterator[Any]):
        """处理数据流"""
        async for data in data_stream:
            # 依次通过所有处理器
            processed_data = data
            for processor in self.processors:
                if asyncio.iscoroutinefunction(processor):
                    processed_data = await processor(processed_data)
                else:
                    processed_data = processor(processed_data)
            
            # 发送到所有输出处理器
            for handler in self.output_handlers:
                if asyncio.iscoroutinefunction(handler):
                    await handler(processed_data)
                else:
                    handler(processed_data)

def demo_basic_streaming():
    """演示基本流式处理"""
    print("🌊 基本流式处理演示")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 请设置 OPENAI_API_KEY 环境变量")
        return
    
    # 创建流式LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        streaming=True
    )
    
    prompt = ChatPromptTemplate.from_template("请详细介绍一下：{topic}")
    chain = prompt | llm | StrOutputParser()
    
    print("流式输出示例（介绍人工智能）:")
    print("-" * 30)
    
    # 流式处理
    for chunk in chain.stream({"topic": "人工智能的发展历程"}):
        print(chunk, end="", flush=True)
    
    print("\\n")

def demo_real_time_chat():
    """演示实时聊天系统"""
    print("💬 实时聊天系统演示")
    print("=" * 50)
    
    try:
        chat_system = RealTimeChatSystem()
        
        # 创建测试会话
        session_id = "demo_session_001"
        chat_system.create_session(session_id)
        
        # 测试问题
        questions = [
            "你好，请介绍一下你自己",
            "什么是机器学习？",
            "能给我讲个笑话吗？"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\\n👤 用户问题 {i}: {question}")
            print("🤖 AI回答: ", end="")
            
            # 流式响应
            for chunk in chat_system.stream_response(session_id, question):
                print(chunk, end="", flush=True)
            
            print("\\n")
            
            # 模拟用户思考时间
            time.sleep(1)
            
    except Exception as e:
        print(f"聊天系统错误: {e}")

async def demo_async_processing():
    """演示异步处理管道"""
    print("⚡ 异步处理管道演示")
    print("=" * 50)
    
    try:
        pipeline = AsyncProcessingPipeline()
        
        # 添加测试任务
        tasks = [
            ("task_001", "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。", "summarize"),
            ("task_002", "机器学习是人工智能的一个重要子领域。", "translate"),
            ("task_003", "深度学习、神经网络、自然语言处理是AI的关键技术。", "analyze"),
            ("task_004", "Python是AI开发中最流行的编程语言之一。", "summarize"),
            ("task_005", "数据科学和机器学习密不可分。", "translate")
        ]
        
        print(f"📋 添加 {len(tasks)} 个任务到处理队列...")
        
        # 添加任务
        for task_id, content, task_type in tasks:
            await pipeline.add_task(task_id, content, task_type)
        
        # 开始批量处理
        print("🔄 开始异步批量处理...")
        start_time = time.time()
        
        await pipeline.batch_process(max_concurrent=3)
        
        end_time = time.time()
        print(f"✅ 所有任务处理完成，用时: {end_time - start_time:.2f}秒")
        
        # 显示结果
        print("\\n📊 处理结果:")
        for task_id, _, _ in tasks:
            result = pipeline.get_result(task_id)
            if result:
                print(f"   {task_id}: {result['status']}")
                if result['status'] == 'completed':
                    print(f"     结果: {result['result'][:100]}...")
                elif result['status'] == 'failed':
                    print(f"     错误: {result['error']}")
        
    except Exception as e:
        print(f"异步处理错误: {e}")

async def demo_streaming_data_pipeline():
    """演示流式数据管道"""
    print("🔄 流式数据管道演示")
    print("=" * 50)
    
    # 创建数据管道
    pipeline = StreamingDataPipeline()
    
    # 添加处理器
    def text_cleaner(data):
        """文本清理器"""
        if isinstance(data, str):
            return data.strip().replace("\\n", " ")
        return data
    
    def text_analyzer(data):
        """文本分析器"""
        if isinstance(data, str):
            word_count = len(data.split())
            return {
                "original": data,
                "word_count": word_count,
                "length": len(data),
                "timestamp": datetime.now()
            }
        return data
    
    async def result_handler(data):
        """结果处理器"""
        print(f"📊 处理结果: 长度={data['length']}, 词数={data['word_count']}")
        print(f"   内容: {data['original'][:50]}...")
    
    pipeline.add_processor(text_cleaner)
    pipeline.add_processor(text_analyzer) 
    pipeline.add_output_handler(result_handler)
    
    # 创建测试数据流
    async def test_data_stream():
        test_texts = [
            "   人工智能正在改变我们的世界   \\n",
            "机器学习是AI的核心技术",
            "深度学习推动了AI的发展",
            "自然语言处理让机器理解人类语言",
            "计算机视觉让机器看懂图像"
        ]
        
        for text in test_texts:
            yield text
            await asyncio.sleep(0.5)  # 模拟数据到达间隔
    
    print("开始处理流式数据...")
    await pipeline.process_stream(test_data_stream())

def demo_performance_monitoring():
    """演示性能监控"""
    print("📈 性能监控演示")
    print("=" * 50)
    
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "avg_response_time": 0,
                "response_times": []
            }
        
        def record_request(self, success: bool, response_time: float):
            """记录请求"""
            self.metrics["total_requests"] += 1
            if success:
                self.metrics["successful_requests"] += 1
            else:
                self.metrics["failed_requests"] += 1
            
            self.metrics["response_times"].append(response_time)
            self.metrics["avg_response_time"] = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
        
        def get_stats(self) -> Dict[str, Any]:
            """获取统计信息"""
            return {
                **self.metrics,
                "success_rate": self.metrics["successful_requests"] / max(1, self.metrics["total_requests"]),
                "min_response_time": min(self.metrics["response_times"]) if self.metrics["response_times"] else 0,
                "max_response_time": max(self.metrics["response_times"]) if self.metrics["response_times"] else 0
            }
    
    # 创建性能监控器
    monitor = PerformanceMonitor()
    
    # 模拟一些请求
    import random
    for i in range(20):
        success = random.random() > 0.1  # 90%成功率
        response_time = random.uniform(0.1, 2.0)  # 0.1-2.0秒响应时间
        monitor.record_request(success, response_time)
    
    # 显示统计信息
    stats = monitor.get_stats()
    print(f"总请求数: {stats['total_requests']}")
    print(f"成功请求数: {stats['successful_requests']}")
    print(f"失败请求数: {stats['failed_requests']}")
    print(f"成功率: {stats['success_rate']:.2%}")
    print(f"平均响应时间: {stats['avg_response_time']:.3f}秒")
    print(f"最小响应时间: {stats['min_response_time']:.3f}秒")
    print(f"最大响应时间: {stats['max_response_time']:.3f}秒")

async def main():
    """主函数"""
    try:
        print("🌊 LangChain Challenge 7: 流式处理和异步操作")
        print("=" * 60)
        
        # 基本流式处理演示
        demo_basic_streaming()
        
        # 实时聊天演示
        demo_real_time_chat()
        
        print("\\n" + "=" * 60)
        print("⚡ 异步处理演示...")
        
        # 异步处理演示
        await demo_async_processing()
        
        print("\\n" + "=" * 60)
        print("🔄 流式数据管道演示...")
        
        # 流式数据管道演示
        await demo_streaming_data_pipeline()
        
        print("\\n" + "=" * 60)
        print("📈 性能监控演示...")
        
        # 性能监控演示
        demo_performance_monitoring()
        
        print("\\n" + "=" * 60)
        print("🎯 练习任务:")
        print("1. 实现WebSocket支持的实时聊天接口")
        print("2. 添加流式处理的速率限制和背压控制")
        print("3. 实现分布式异步处理系统")
        print("4. 添加更详细的性能监控和告警")
        print("5. 实现流式数据的持久化存储")
        print("6. 创建可视化的实时监控面板")
        print("7. 实现容错和故障恢复机制")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("\\n请确保:")
        print("1. 已设置 OPENAI_API_KEY 环境变量")
        print("2. 已安装所需的依赖包: pip install langchain langchain-openai aiohttp")

if __name__ == "__main__":
    asyncio.run(main())
