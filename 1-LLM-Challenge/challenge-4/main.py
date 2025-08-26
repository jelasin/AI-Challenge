# -*- coding: utf-8 -*-
"""
Challenge 4: 文档处理和RAG（检索增强生成）v0.3
难度：中级到高级

学习目标：
1. 掌握Document Loader的使用
2. 学习Text Splitter的多种策略
3. 实现Embedding和Vector Store
4. 构建Retriever系统
5. 实现完整的RAG应用

任务描述：
创建一个智能文档问答系统，能够：
1. 处理多种格式的文档（TXT、Markdown、CSV等）
2. 使用多种文本切分策略
3. 构建向量数据库
4. 实现智能检索
5. 结合LLM生成准确答案
6. 记录历史对话
"""

from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter
)
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from typing import List,  Dict
import os
from pathlib import Path
from operator import itemgetter

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

def load_documents_from_dir(doc_dir: str) -> list[Document]:
    """从指定目录加载多种格式文档（md/txt/csv/pdf）。"""
    print(f"📂 从目录加载文档: {doc_dir}")
    documents: list[Document] = []

    # Markdown 与 TXT
    md_loader = DirectoryLoader(
        doc_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    txt_loader = DirectoryLoader(
        doc_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    # CSV（每行一条 Document）
    csv_loader = DirectoryLoader(
        doc_dir, glob="**/*.csv", loader_cls=CSVLoader, loader_kwargs={"encoding": "utf-8"}
    )
    for loader, dtype in (
        (md_loader, "markdown"),
        (txt_loader, "text"),
        (csv_loader, "csv"),
    ):
        try:
            docs = loader.load()
        except Exception as e:
            # 某些 loader 可能因依赖缺失失败，跳过并提示
            print(f"⚠️ 加载 {dtype} 文档时出错: {e}")
            docs = []
        for d in docs:
            d.metadata["source"] = os.path.relpath(d.metadata.get("source", d.metadata.get("file_path", "")), doc_dir) if d.metadata else ""
            d.metadata["type"] = dtype
        documents.extend(docs)

    print(f"✅ 共加载文档 {len(documents)} 条")
    return documents

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
    if recursive_chunks:
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
    if token_chunks:
        print(f"   第一块长度: {len(token_chunks[0])}")
    
    return recursive_chunks

def create_rag_system(doc_dir: str):
    """创建RAG系统，基于本地 doc 目录构建索引与检索。"""
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("请设置 OPENAI_API_KEY 环境变量")

    print("\n🔧 构建RAG系统...")

    # 从目录加载文档
    documents = load_documents_from_dir(doc_dir)
    if not documents:
        raise ValueError(f"在目录 {doc_dir} 下未发现可加载的文档（支持 md/txt/csv/pdf）。")
    
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
    
    # 创建LLM（v0.3 推荐：init_chat_model）
    llm = init_chat_model("gpt-4o", temperature=0.1)
    
    # 创建RAG Prompt
    rag_prompt = ChatPromptTemplate.from_template("""
你是一个专业的AI助手。请综合“最近对话历史”和“检索到的上下文”来回答当前用户问题；若上下文不包含相关信息，请明确说明无法从提供的文档中找到答案。

最近对话历史（最多5轮，若为空可忽略）：
{chat_history}

上下文信息：
{context}

用户问题：{question}

回答：""")
    
    # 创建RAG链
    # 链输入：{"question": str, "chat_history": str}
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_documents,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever, vectorstore, len(split_documents), documents

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
        # 文档目录（相对当前文件）
        docs_dir = str(Path(__file__).parent.joinpath("doc").resolve())

        # 创建RAG系统
        rag_chain, retriever, vectorstore, chunks_count, documents = create_rag_system(docs_dir)

        # 分析文档集合
        analyze_document_collection(documents, chunks_count)

        # 演示高级检索
        demo_advanced_retrieval(retriever, vectorstore)

        print("\n" + "=" * 60)
        print("💬 交互问答模式（输入内容后回车）：")
        print("   - 输入 exit / quit / q 或直接回车可退出。")

        # 交互式问答循环
        history: list[tuple[str, str]] = []  # 记录最近 5 轮 (question, answer)
        while True:
            try:
                question = input("\n❓ 你的问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 退出。")
                break

            if not question or question.lower() in {"exit", "quit", "q"}:
                print("👋 已退出问答模式。")
                break

            print("-" * 40)
            # 组装最近 5 轮对话历史
            if history:
                history_text = "\n".join([f"用户：{q}\n助手：{a}" for q, a in history[-5:]])
            else:
                history_text = "（无）"

            # 获取答案（传入问题与历史）
            payload = {"question": question, "chat_history": history_text}
            answer = rag_chain.invoke(payload)
            print(f"🤖 回答: {answer}")

            # 显示相关文档
            relevant_docs = retriever.invoke(question)
            print(f"\n📖 相关文档块 ({len(relevant_docs)} 个):")
            for j, doc in enumerate(relevant_docs, 1):
                print(f"   {j}. 来源: {doc.metadata.get('source', '未知')}")
                print(f"      内容: {doc.page_content[:150]}...")

            # 更新对话历史，最多保留 5 轮
            history.append((question, answer))
            if len(history) > 5:
                history = history[-5:]

    except Exception as e:
        print(f"❌ 错误: {e}")
        print("\n请确保:")
        print("1. 已设置 OPENAI_API_KEY 环境变量")
        print("2. 已安装所需的依赖包:")
        print("   pip install langchain langchain-openai langchain-community")
        print("   pip install faiss-cpu pypdf unstructured")

if __name__ == "__main__":
    main()
