# -*- coding: utf-8 -*-
"""
Challenge 4: 文档处理和RAG（检索增强生成）
难度：中级到高级

学习目标：
1. 掌握Document Loader的使用
2. 学习Text Splitter的多种策略
3. 实现Embedding和Vector Store
4. 构建Retriever系统
5. 实现完整的RAG应用

任务描述：
创建一个智能文档问答系统，能够：
1. 处理多种格式的文档（PDF、TXT、Markdown、CSV等）
2. 使用多种文本切分策略
3. 构建向量数据库
4. 实现智能检索
5. 结合LLM生成准确答案
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter
)
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import tempfile
import json

class DocumentAnalysis(BaseModel):
    """文档分析结果"""
    total_documents: int = Field(description="文档总数")
    total_chunks: int = Field(description="文档块总数")
    average_chunk_length: float = Field(description="平均块长度")
    document_types: Dict[str, int] = Field(description="文档类型统计")
    key_topics: List[str] = Field(description="关键主题")

class QAResult(BaseModel):
    """问答结果"""
    answer: str = Field(description="回答")
    confidence: float = Field(description="置信度", ge=0.0, le=1.0)
    sources: List[str] = Field(description="来源文档")
    relevant_chunks: List[str] = Field(description="相关文档块")

def create_sample_documents():
    """创建示例文档用于测试"""
    docs = []
    
    # 创建AI相关文档
    ai_doc = """
    # 人工智能基础

    ## 机器学习
    机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下学习和改进。
    机器学习算法通过训练数据建立数学模型，以便对新数据进行预测或决策。

    ### 监督学习
    监督学习使用标记的训练数据来学习从输入到输出的映射函数。
    常见的监督学习算法包括：
    - 线性回归
    - 逻辑回归  
    - 决策树
    - 随机森林
    - 支持向量机

    ### 无监督学习
    无监督学习从未标记的数据中发现隐藏的模式或结构。
    主要类型包括：
    - 聚类分析
    - 降维
    - 关联规则学习

    ## 深度学习
    深度学习是机器学习的一个子集，使用神经网络来模拟人脑处理信息的方式。
    深度学习在图像识别、自然语言处理和语音识别等领域取得了突破性进展。

    ### 神经网络架构
    - 卷积神经网络（CNN）：主要用于图像处理
    - 循环神经网络（RNN）：适合序列数据
    - 长短期记忆网络（LSTM）：解决RNN的长期依赖问题
    - Transformer：革命性的注意力机制架构
    """
    
    # 创建编程相关文档
    programming_doc = """
    # Python编程指南

    ## 基础语法
    Python是一种高级、解释型的编程语言，以其简洁的语法和强大的功能而闻名。

    ### 变量和数据类型
    Python中的基本数据类型包括：
    - 整数（int）
    - 浮点数（float）
    - 字符串（str）
    - 布尔值（bool）
    - 列表（list）
    - 字典（dict）

    ### 控制结构
    Python提供了多种控制程序流程的结构：

    #### 条件语句
    ```python
    if condition:
        # 执行代码
    elif another_condition:
        # 执行其他代码
    else:
        # 默认执行
    ```

    #### 循环语句
    ```python
    # for循环
    for item in iterable:
        # 处理每个元素

    # while循环  
    while condition:
        # 重复执行
    ```

    ## 面向对象编程
    Python支持面向对象编程范式，允许创建类和对象。

    ### 类的定义
    ```python
    class MyClass:
        def __init__(self, value):
            self.value = value
        
        def method(self):
            return self.value * 2
    ```

    ## 常用库
    Python有丰富的标准库和第三方库：
    - NumPy：科学计算
    - Pandas：数据分析
    - Matplotlib：数据可视化
    - Requests：HTTP请求
    - Django：Web框架
    - Flask：轻量级Web框架
    """
    
    # 创建数据科学文档
    datascience_doc = """
    # 数据科学入门

    ## 什么是数据科学
    数据科学是一个跨学科领域，使用科学方法、过程、算法和系统从结构化和非结构化数据中提取知识和洞察。

    ## 数据科学流程
    典型的数据科学项目包括以下步骤：

    ### 1. 问题定义
    - 明确业务目标
    - 定义成功指标
    - 确定数据需求

    ### 2. 数据收集
    数据可能来自多个来源：
    - 数据库
    - API接口
    - 文件系统
    - 网页抓取
    - 传感器数据

    ### 3. 数据探索和清理
    - 数据质量检查
    - 处理缺失值
    - 异常值检测
    - 数据类型转换
    - 特征工程

    ### 4. 建模和分析
    - 选择合适的算法
    - 训练模型
    - 模型验证
    - 超参数调优

    ### 5. 结果展示
    - 数据可视化
    - 报告编写
    - 模型部署
    - 监控维护

    ## 常用工具
    - Python/R：编程语言
    - Jupyter Notebook：交互式环境
    - Pandas：数据处理
    - Scikit-learn：机器学习
    - TensorFlow/PyTorch：深度学习
    - Tableau/PowerBI：可视化工具
    """
    
    return [
        ("ai_basics.md", ai_doc),
        ("python_guide.md", programming_doc),
        ("data_science.md", datascience_doc)
    ]

def create_document_loaders():
    """创建多种文档加载器"""
    print("📁 创建示例文档...")
    
    # 创建临时目录和文件
    temp_dir = tempfile.mkdtemp()
    print(f"临时目录: {temp_dir}")
    
    # 创建示例文档
    sample_docs = create_sample_documents()
    file_paths = []
    
    for filename, content in sample_docs:
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        file_paths.append(file_path)
        print(f"创建文件: {filename}")
    
    # 创建CSV示例
    csv_content = """name,age,occupation,description
Alice,25,Data Scientist,专门从事机器学习和数据分析
Bob,30,Software Engineer,专注于Python和Web开发
Carol,28,Product Manager,负责AI产品的规划和管理
David,32,Research Scientist,在深度学习领域有丰富经验"""
    
    csv_path = os.path.join(temp_dir, "team.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(csv_content)
    file_paths.append(csv_path)
    print("创建文件: team.csv")
    
    return temp_dir, file_paths

def demo_text_splitters(documents: List[Document]):
    """演示不同的文本分割策略"""
    print("\n📝 文本分割策略演示")
    print("=" * 50)
    
    # 合并所有文档内容用于演示
    full_text = "\n\n".join([doc.page_content for doc in documents])
    
    # 1. 递归字符分割器（推荐）
    print("1. 递归字符分割器:")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    recursive_chunks = recursive_splitter.split_text(full_text)
    print(f"   生成块数: {len(recursive_chunks)}")
    print(f"   第一块长度: {len(recursive_chunks[0])}")
    
    # 2. Markdown头部分割器
    print("\n2. Markdown头部分割器:")
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"), 
            ("###", "Header 3"),
        ]
    )
    # 只处理Markdown文档
    md_documents = [doc for doc in documents if doc.metadata.get('source', '').endswith('.md')]
    if md_documents:
        md_chunks = md_splitter.split_text(md_documents[0].page_content)
        print(f"   生成块数: {len(md_chunks)}")
        if md_chunks:
            print(f"   第一块元数据: {md_chunks[0].metadata}")
    
    # 3. Token分割器
    print("\n3. Token分割器:")
    token_splitter = TokenTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    token_chunks = token_splitter.split_text(full_text)
    print(f"   生成块数: {len(token_chunks)}")
    print(f"   第一块长度: {len(token_chunks[0])}")
    
    return recursive_chunks

def create_rag_system():
    """创建RAG系统"""
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("请设置 OPENAI_API_KEY 环境变量")
    
    print("\n🔧 构建RAG系统...")
    
    # 创建示例文档
    temp_dir, file_paths = create_document_loaders()
    
    # 加载所有文档
    documents = []
    
    # 加载Markdown文件
    for file_path in file_paths:
        if file_path.endswith('.md'):
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = os.path.basename(file_path)
                doc.metadata['type'] = 'markdown'
            documents.extend(docs)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path, encoding='utf-8')
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = os.path.basename(file_path)
                doc.metadata['type'] = 'csv'
            documents.extend(docs)
    
    print(f"✅ 加载了 {len(documents)} 个文档")
    
    # 演示文本分割
    chunks = demo_text_splitters(documents)
    
    # 创建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    
    # 分割文档
    split_documents = text_splitter.split_documents(documents)
    print(f"✅ 文档分割完成，共 {len(split_documents)} 个块")
    
    # 创建嵌入模型
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # 使用较新的嵌入模型
    )
    
    # 创建向量存储
    print("🔄 创建向量数据库...")
    vectorstore = FAISS.from_documents(split_documents, embeddings)
    print("✅ 向量数据库创建完成")
    
    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # 检索最相似的4个块
    )
    
    # 创建LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1
    )
    
    # 创建RAG Prompt
    rag_prompt = ChatPromptTemplate.from_template("""
你是一个专业的AI助手，请基于以下上下文信息回答用户的问题。

上下文信息：
{context}

用户问题：{question}

请根据上下文信息提供准确、详细的答案。如果上下文中没有相关信息，请说明无法从提供的文档中找到答案。

回答：""")
    
    # 创建RAG链
    rag_chain = (
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever, vectorstore, len(split_documents)

def format_documents(docs):
    """格式化检索到的文档"""
    return "\n\n".join([f"来源: {doc.metadata.get('source', '未知')}\n内容: {doc.page_content}" for doc in docs])

def demo_advanced_retrieval(retriever, vectorstore):
    """演示高级检索功能"""
    print("\n🔍 高级检索功能演示")
    print("=" * 50)
    
    # 1. 相似性搜索
    print("1. 相似性搜索:")
    similar_docs = vectorstore.similarity_search("机器学习算法", k=3)
    for i, doc in enumerate(similar_docs, 1):
        print(f"   结果 {i}: {doc.page_content[:100]}...")
        print(f"           来源: {doc.metadata.get('source', '未知')}")
    
    # 2. 相似性搜索带分数
    print("\n2. 带分数的相似性搜索:")
    similar_docs_with_scores = vectorstore.similarity_search_with_score("Python编程", k=3)
    for i, (doc, score) in enumerate(similar_docs_with_scores, 1):
        print(f"   结果 {i} (分数: {score:.4f}): {doc.page_content[:80]}...")
    
    # TODO: 实现更多高级检索功能
    print("\n🎯 待实现的高级功能:")
    print("- 混合检索（关键词 + 向量）")
    print("- 重排序检索")
    print("- 多查询检索")
    print("- 上下文压缩检索")

def analyze_document_collection(documents, chunks_count):
    """分析文档集合"""
    print("\n📊 文档集合分析")
    print("=" * 50)
    
    doc_types = {}
    total_length = 0
    
    for doc in documents:
        doc_type = doc.metadata.get('type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        total_length += len(doc.page_content)
    
    avg_length = total_length / len(documents) if documents else 0
    
    analysis = DocumentAnalysis(
        total_documents=len(documents),
        total_chunks=chunks_count,
        average_chunk_length=avg_length,
        document_types=doc_types,
        key_topics=["人工智能", "机器学习", "Python编程", "数据科学"]  # 简化的主题提取
    )
    
    print(f"📋 文档统计:")
    print(f"   总文档数: {analysis.total_documents}")
    print(f"   总块数: {analysis.total_chunks}")
    print(f"   平均文档长度: {analysis.average_chunk_length:.0f} 字符")
    print(f"   文档类型: {analysis.document_types}")
    print(f"   关键主题: {', '.join(analysis.key_topics)}")

def main():
    """主函数"""
    try:
        print("🔍 LangChain Challenge 4: 文档处理和RAG")
        print("=" * 60)
        
        # 创建RAG系统
        rag_chain, retriever, vectorstore, chunks_count = create_rag_system()
        
        # 分析文档集合（这里我们需要重新加载documents来分析）
        temp_dir, file_paths = create_document_loaders()
        documents = []
        for file_path in file_paths:
            if file_path.endswith('.md'):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
        
        analyze_document_collection(documents, chunks_count)
        
        # 演示高级检索
        demo_advanced_retrieval(retriever, vectorstore)
        
        print("\n" + "=" * 60)
        print("💬 开始问答演示...")
        
        # 测试问题
        test_questions = [
            "什么是机器学习？它有哪些主要类型？",
            "Python中有哪些基本的数据类型？",
            "数据科学的典型流程是什么？",
            "深度学习和传统机器学习有什么区别？",
            "团队中有哪些成员？他们的职业是什么？"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n❓ 问题 {i}: {question}")
            print("-" * 40)
            
            # 获取答案
            answer = rag_chain.invoke(question)
            print(f"🤖 回答: {answer}")
            
            # 显示相关文档
            relevant_docs = retriever.invoke(question)
            print(f"\n📖 相关文档块 ({len(relevant_docs)} 个):")
            for j, doc in enumerate(relevant_docs, 1):
                print(f"   {j}. 来源: {doc.metadata.get('source', '未知')}")
                print(f"      内容: {doc.page_content[:150]}...")
            
            if i < len(test_questions):  # 不是最后一个问题
                print()
        
        print("\n" + "=" * 60)
        print("🎯 练习任务:")
        print("1. 实现多种文档格式的加载器（PDF、Word、Excel等）")
        print("2. 添加文档元数据增强（时间戳、作者、标签等）")
        print("3. 实现混合检索（BM25 + 向量检索）")
        print("4. 添加检索结果重排序功能")
        print("5. 实现文档更新和增量索引")
        print("6. 添加查询扩展和意图理解")
        print("7. 实现多轮对话的上下文管理")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("\n请确保:")
        print("1. 已设置 OPENAI_API_KEY 环境变量")
        print("2. 已安装所需的依赖包:")
        print("   pip install langchain langchain-openai langchain-community")
        print("   pip install faiss-cpu pypdf unstructured")

if __name__ == "__main__":
    main()
