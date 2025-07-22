# -*- coding: utf-8 -*-
"""
Challenge 8: 综合项目 - 智能知识管理系统
难度：专家级

学习目标：
1. 综合运用所有LangChain特性
2. 构建完整的企业级AI应用
3. 实现复杂的多模态处理
4. 集成多种数据源和服务
5. 实现高可用性和可扩展性
6. 添加监控、日志和安全控制

项目描述：
创建一个智能知识管理系统，整合：
- 多源文档处理和RAG
- 智能Agent和工具调用
- 实时问答和推理
- 知识图谱构建
- 多用户协作
- API服务和Web界面
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.document_loaders import TextLoader, DirectoryLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
import json
import os
import sqlite3
import hashlib
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================== 数据模型 ===========================

class DocumentMetadata(BaseModel):
    """文档元数据模型"""
    id: str
    title: str
    author: Optional[str] = None
    created_date: datetime
    modified_date: datetime
    file_path: str
    file_size: int
    content_hash: str
    tags: List[str] = []
    category: Optional[str] = None
    summary: Optional[str] = None
    language: str = "zh-CN"

class KnowledgeEntity(BaseModel):
    """知识实体模型"""
    id: str
    name: str
    entity_type: str  # person, organization, concept, technology
    description: str
    confidence: float
    source_documents: List[str]
    related_entities: List[str] = []

class UserSession(BaseModel):
    """用户会话模型"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    conversation_history: List[Dict[str, Any]] = []
    preferences: Dict[str, Any] = {}

class QueryResult(BaseModel):
    """查询结果模型"""
    query: str
    answer: str
    confidence: float
    sources: List[str]
    related_entities: List[str]
    suggestions: List[str]
    processing_time: float

# =========================== 数据库管理 ===========================

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str = "knowledge_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建文档表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                created_date TIMESTAMP,
                modified_date TIMESTAMP,
                file_path TEXT,
                file_size INTEGER,
                content_hash TEXT,
                tags TEXT,
                category TEXT,
                summary TEXT,
                language TEXT
            )
        ''')
        
        # 创建知识实体表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT,
                description TEXT,
                confidence REAL,
                source_documents TEXT,
                related_entities TEXT
            )
        ''')
        
        # 创建用户会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TIMESTAMP,
                last_activity TIMESTAMP,
                conversation_history TEXT,
                preferences TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("数据库初始化完成")
    
    def save_document(self, doc: DocumentMetadata):
        """保存文档元数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO documents 
            (id, title, author, created_date, modified_date, file_path, 
             file_size, content_hash, tags, category, summary, language)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            doc.id, doc.title, doc.author, doc.created_date, doc.modified_date,
            doc.file_path, doc.file_size, doc.content_hash, 
            json.dumps(doc.tags), doc.category, doc.summary, doc.language
        ))
        
        conn.commit()
        conn.close()
    
    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """获取文档元数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return DocumentMetadata(
                id=row[0], title=row[1], author=row[2],
                created_date=datetime.fromisoformat(row[3]),
                modified_date=datetime.fromisoformat(row[4]),
                file_path=row[5], file_size=row[6], content_hash=row[7],
                tags=json.loads(row[8]) if row[8] else [],
                category=row[9], summary=row[10], language=row[11]
            )
        return None
    
    def save_entity(self, entity: KnowledgeEntity):
        """保存知识实体"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO entities 
            (id, name, entity_type, description, confidence, source_documents, related_entities)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            entity.id, entity.name, entity.entity_type, entity.description,
            entity.confidence, json.dumps(entity.source_documents),
            json.dumps(entity.related_entities)
        ))
        
        conn.commit()
        conn.close()

# =========================== 文档处理器 ===========================

class DocumentProcessor:
    """智能文档处理器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\\n\\n", "\\n", "。", "！", "？", " "]
        )
        
        # 检查API密钥
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
        
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 创建向量存储目录
        self.vector_store_path = "vector_stores"
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # 初始化向量存储
        self.vector_store = None
        self.load_or_create_vector_store()
    
    def load_or_create_vector_store(self):
        """加载或创建向量存储"""
        vector_store_file = os.path.join(self.vector_store_path, "main_index")
        
        if os.path.exists(vector_store_file + ".faiss"):
            logger.info("加载现有向量存储")
            self.vector_store = FAISS.load_local(
                vector_store_file, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            logger.info("创建新的向量存储")
            # 创建一个空的向量存储
            from langchain_core.documents import Document
            dummy_doc = Document(page_content="初始化文档", metadata={"source": "init"})
            self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
            self.save_vector_store()
    
    def save_vector_store(self):
        """保存向量存储"""
        if self.vector_store:
            vector_store_file = os.path.join(self.vector_store_path, "main_index")
            self.vector_store.save_local(vector_store_file)
            logger.info("向量存储已保存")
    
    def calculate_content_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def extract_metadata(self, file_path: str, content: str) -> DocumentMetadata:
        """提取文档元数据"""
        file_stat = os.stat(file_path)
        content_hash = self.calculate_content_hash(content)
        
        # 生成标题（从内容的前100个字符或文件名）
        title = content[:100].strip() if content else os.path.basename(file_path)
        if len(title) > 50:
            title = title[:50] + "..."
        
        doc_id = hashlib.md5(file_path.encode()).hexdigest()
        
        return DocumentMetadata(
            id=doc_id,
            title=title,
            created_date=datetime.fromtimestamp(file_stat.st_ctime),
            modified_date=datetime.fromtimestamp(file_stat.st_mtime),
            file_path=file_path,
            file_size=file_stat.st_size,
            content_hash=content_hash
        )
    
    def generate_summary(self, content: str) -> str:
        """生成文档摘要"""
        if len(content) < 200:
            return content
        
        prompt = ChatPromptTemplate.from_template(
            "请为以下文档内容生成一个简洁的摘要（不超过200字）：\\n\\n{content}"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            summary = chain.invoke({"content": content[:2000]})  # 限制输入长度
            return summary
        except Exception as e:
            logger.error(f"生成摘要失败: {e}")
            return content[:200] + "..."
    
    def extract_entities(self, content: str, doc_id: str) -> List[KnowledgeEntity]:
        """提取知识实体"""
        prompt = ChatPromptTemplate.from_template("""
        从以下文本中提取重要的知识实体，包括人名、组织、概念、技术等。

        文本内容：{content}

        请以JSON格式返回结果，包含以下字段：
        {{
            "entities": [
                {{
                    "name": "实体名称",
                    "type": "实体类型（person/organization/concept/technology）",
                    "description": "实体描述"
                }}
            ]
        }}
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            result = chain.invoke({"content": content[:1500]})
            entities = []
            
            for entity_data in result.get("entities", []):
                entity_id = hashlib.md5(f"{entity_data['name']}_{entity_data['type']}".encode()).hexdigest()
                
                entity = KnowledgeEntity(
                    id=entity_id,
                    name=entity_data["name"],
                    entity_type=entity_data["type"],
                    description=entity_data["description"],
                    confidence=0.8,  # 简化的置信度
                    source_documents=[doc_id]
                )
                entities.append(entity)
            
            return entities
        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            return []
    
    def process_document(self, file_path: str) -> bool:
        """处理单个文档"""
        try:
            logger.info(f"开始处理文档: {file_path}")
            
            # 加载文档内容
            if file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path, encoding='utf-8')
            else:
                logger.warning(f"不支持的文件格式: {file_path}")
                return False
            
            documents = loader.load()
            if not documents:
                logger.warning(f"文档内容为空: {file_path}")
                return False
            
            content = documents[0].page_content
            
            # 提取元数据
            metadata = self.extract_metadata(file_path, content)
            
            # 检查是否已处理过（基于内容哈希）
            existing_doc = self.db_manager.get_document(metadata.id)
            if existing_doc and existing_doc.content_hash == metadata.content_hash:
                logger.info(f"文档未发生变化，跳过处理: {file_path}")
                return True
            
            # 生成摘要
            metadata.summary = self.generate_summary(content)
            
            # 分割文档
            chunks = self.text_splitter.split_documents(documents)
            
            # 更新向量存储
            if self.vector_store and chunks:
                # 为每个chunk添加文档ID
                for chunk in chunks:
                    chunk.metadata['doc_id'] = metadata.id
                    chunk.metadata['doc_title'] = metadata.title
                
                self.vector_store.add_documents(chunks)
                self.save_vector_store()
            
            # 提取知识实体
            entities = self.extract_entities(content, metadata.id)
            
            # 保存到数据库
            self.db_manager.save_document(metadata)
            for entity in entities:
                self.db_manager.save_entity(entity)
            
            logger.info(f"文档处理完成: {file_path}")
            logger.info(f"  - 生成了 {len(chunks)} 个文档块")
            logger.info(f"  - 提取了 {len(entities)} 个知识实体")
            
            return True
            
        except Exception as e:
            logger.error(f"处理文档失败 {file_path}: {e}")
            return False
    
    def process_directory(self, directory_path: str) -> Dict[str, Any]:
        """处理目录中的所有文档"""
        logger.info(f"开始处理目录: {directory_path}")
        
        results = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "skipped_files": 0
        }
        
        # 支持的文件类型
        supported_extensions = ['.txt', '.csv', '.md']
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                results["total_files"] += 1
                
                if file_ext not in supported_extensions:
                    results["skipped_files"] += 1
                    continue
                
                if self.process_document(file_path):
                    results["processed_files"] += 1
                else:
                    results["failed_files"] += 1
        
        logger.info(f"目录处理完成: {results}")
        return results

# =========================== 智能检索器 ===========================

class IntelligentRetriever:
    """智能检索器"""
    
    def __init__(self, document_processor: DocumentProcessor):
        self.doc_processor = document_processor
        self.llm = document_processor.llm
        
    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """语义检索"""
        if not self.doc_processor.vector_store:
            return []
        
        try:
            docs = self.doc_processor.vector_store.similarity_search_with_score(query, k=k)
            results = []
            
            for doc, score in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
            
            return results
        except Exception as e:
            logger.error(f"语义检索失败: {e}")
            return []
    
    def query_expansion(self, query: str) -> List[str]:
        """查询扩展"""
        prompt = ChatPromptTemplate.from_template("""
        为以下查询生成3-5个相关的扩展查询，以提高检索效果：

        原始查询：{query}

        请以JSON格式返回：
        {{"expanded_queries": ["扩展查询1", "扩展查询2", "扩展查询3"]}}
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            result = chain.invoke({"query": query})
            return result.get("expanded_queries", [query])
        except Exception as e:
            logger.error(f"查询扩展失败: {e}")
            return [query]
    
    def multi_query_retrieval(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """多查询检索"""
        expanded_queries = self.query_expansion(query)
        all_results = []
        seen_contents = set()
        
        for expanded_query in expanded_queries:
            results = self.semantic_search(expanded_query, k=k)
            
            for result in results:
                content = result["content"]
                if content not in seen_contents:
                    seen_contents.add(content)
                    all_results.append(result)
        
        # 按分数排序
        all_results.sort(key=lambda x: x["score"])
        return all_results[:k*2]  # 返回更多结果

# =========================== 智能工具 ===========================

class KnowledgeSystemTools:
    """知识系统工具集"""
    
    def __init__(self, db_manager: DatabaseManager, retriever: IntelligentRetriever):
        self.db_manager = db_manager
        self.retriever = retriever
    
    @tool
    def search_documents(self, query: str, limit: int = 5) -> str:
        """搜索相关文档"""
        # 这里需要访问实例变量，实际使用时需要调整
        return f"搜索结果：找到{limit}个相关文档关于'{query}'"
    
    @tool 
    def get_document_summary(self, doc_id: str) -> str:
        """获取文档摘要"""
        return f"文档{doc_id}的摘要"
    
    @tool
    def extract_key_concepts(self, content: str) -> str:
        """提取关键概念"""
        return f"从内容中提取的关键概念"

# =========================== 智能问答系统 ===========================

class IntelligentQASystem:
    """智能问答系统"""
    
    def __init__(self, db_manager: DatabaseManager, retriever: IntelligentRetriever):
        self.db_manager = db_manager
        self.retriever = retriever
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
        
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        
        # 创建RAG链
        self.create_rag_chain()
    
    def create_rag_chain(self):
        """创建RAG处理链"""
        self.rag_prompt = ChatPromptTemplate.from_template("""
        你是一个专业的知识管理助手。请基于以下检索到的相关文档回答用户的问题。

        相关文档：
        {context}

        用户问题：{question}

        请提供准确、详细的答案。如果文档中没有足够信息回答问题，请明确说明。
        同时，请指出答案的来源文档。

        回答：
        """)
        
        self.rag_chain = (
            RunnableParallel({
                "context": RunnableLambda(lambda x: self.get_context(x["question"])),
                "question": RunnablePassthrough()
            })
            | self.rag_prompt
            | self.llm 
            | StrOutputParser()
        )
    
    def get_context(self, question: str) -> str:
        """获取相关上下文"""
        # 使用多查询检索
        results = self.retriever.multi_query_retrieval(question, k=4)
        
        context_parts = []
        for i, result in enumerate(results, 1):
            doc_title = result["metadata"].get("doc_title", "未知文档")
            content = result["content"][:500]  # 限制长度
            context_parts.append(f"文档{i}（{doc_title}）：\\n{content}\\n")
        
        return "\\n".join(context_parts)
    
    def answer_question(self, question: str, session_id: Optional[str] = None) -> QueryResult:
        """回答问题"""
        start_time = time.time()
        
        try:
            # 生成答案
            answer = self.rag_chain.invoke({"question": question})
            
            # 获取相关文档
            results = self.retriever.multi_query_retrieval(question, k=3)
            sources = [result["metadata"].get("doc_title", "未知文档") for result in results]
            
            processing_time = time.time() - start_time
            
            return QueryResult(
                query=question,
                answer=answer,
                confidence=0.8,  # 简化的置信度
                sources=sources,
                related_entities=[],  # 可以进一步实现
                suggestions=[],      # 可以进一步实现
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"问答处理失败: {e}")
            return QueryResult(
                query=question,
                answer=f"处理问题时发生错误: {str(e)}",
                confidence=0.0,
                sources=[],
                related_entities=[],
                suggestions=[],
                processing_time=time.time() - start_time
            )

# =========================== 主系统 ===========================

class IntelligentKnowledgeSystem:
    """智能知识管理系统"""
    
    def __init__(self, data_directory: str = "data"):
        self.data_directory = data_directory
        os.makedirs(data_directory, exist_ok=True)
        
        # 初始化组件
        self.db_manager = DatabaseManager()
        self.doc_processor = DocumentProcessor(self.db_manager)
        self.retriever = IntelligentRetriever(self.doc_processor)
        self.qa_system = IntelligentQASystem(self.db_manager, self.retriever)
        
        logger.info("智能知识管理系统初始化完成")
    
    def add_documents_from_directory(self, directory_path: str) -> Dict[str, Any]:
        """从目录添加文档"""
        return self.doc_processor.process_directory(directory_path)
    
    def add_document_from_text(self, title: str, content: str, author: str = "") -> bool:
        """从文本添加文档"""
        # 创建临时文件
        temp_file = os.path.join(self.data_directory, f"temp_{title}.txt")
        
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            success = self.doc_processor.process_document(temp_file)
            
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return success
        except Exception as e:
            logger.error(f"添加文本文档失败: {e}")
            return False
    
    def ask_question(self, question: str, session_id: Optional[str] = None) -> QueryResult:
        """询问问题"""
        return self.qa_system.answer_question(question, session_id)
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索文档"""
        return self.retriever.multi_query_retrieval(query, k=limit)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        # 这里可以实现更详细的统计
        return {
            "total_documents": "N/A",  # 需要查询数据库
            "total_entities": "N/A",   # 需要查询数据库
            "vector_store_size": "N/A", # 需要检查向量存储
            "system_status": "running"
        }

# =========================== 演示功能 ===========================

def create_sample_documents(system: IntelligentKnowledgeSystem):
    """创建示例文档"""
    print("📚 创建示例文档...")
    
    sample_docs = [
        {
            "title": "人工智能发展史",
            "content": """
            人工智能（Artificial Intelligence，AI）的发展可以追溯到20世纪50年代。

            ## 早期发展（1950s-1960s）
            1950年，英国数学家艾伦·图灵提出了著名的图灵测试，为人工智能的评估标准奠定了基础。
            1956年，约翰·麦卡锡在达特茅斯学院组织了人工智能研讨会，正式提出了"人工智能"这一术语。

            ## 符号主义时期（1960s-1980s）
            这一时期的AI主要基于符号逻辑和知识表示。代表性成果包括专家系统的发展。

            ## 连接主义复兴（1980s-1990s）
            神经网络技术重新兴起，反向传播算法的提出推动了机器学习的发展。

            ## 现代AI时代（2000s-至今）
            深度学习、大数据和计算能力的提升，使AI在图像识别、自然语言处理等领域取得突破性进展。
            """,
            "author": "AI历史学家"
        },
        {
            "title": "机器学习基础概念",
            "content": """
            机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习和改进，而无需明确编程。

            ## 主要类型
            
            ### 监督学习
            - 使用标记数据进行训练
            - 包括分类和回归问题
            - 常见算法：线性回归、决策树、随机森林、支持向量机
            
            ### 无监督学习
            - 从未标记数据中发现隐藏模式
            - 包括聚类、降维、关联规则学习
            - 常见算法：K-means、主成分分析、层次聚类
            
            ### 强化学习
            - 通过与环境交互学习最优策略
            - 应用于游戏、机器人控制、推荐系统
            - 代表算法：Q-learning、策略梯度方法

            ## 关键概念
            - 特征工程：从原始数据中提取有用特征
            - 模型评估：使用交叉验证等方法评估模型性能
            - 过拟合与欠拟合：模型复杂度的平衡
            """,
            "author": "机器学习专家"
        },
        {
            "title": "深度学习和神经网络",
            "content": """
            深度学习是机器学习的一个分支，它基于人工神经网络进行学习和决策。

            ## 神经网络基础
            人工神经网络模仿人脑神经元的工作方式，由输入层、隐藏层和输出层组成。

            ### 核心组件
            - 神经元：基本计算单元
            - 权重和偏置：控制信号传递的参数
            - 激活函数：引入非线性特性

            ## 深度学习架构

            ### 卷积神经网络（CNN）
            - 主要用于图像处理
            - 包含卷积层、池化层和全连接层
            - 在计算机视觉领域表现优异

            ### 循环神经网络（RNN）
            - 适用于序列数据处理
            - LSTM和GRU解决了长期依赖问题
            - 广泛应用于自然语言处理

            ### Transformer架构
            - 基于注意力机制
            - GPT、BERT等模型的基础
            - 在自然语言处理领域取得突破

            ## 应用领域
            - 计算机视觉：图像识别、目标检测、图像生成
            - 自然语言处理：机器翻译、文本生成、情感分析
            - 语音技术：语音识别、语音合成
            """,
            "author": "深度学习研究员"
        }
    ]
    
    success_count = 0
    for doc in sample_docs:
        if system.add_document_from_text(doc["title"], doc["content"], doc["author"]):
            success_count += 1
            print(f"✅ 成功添加文档: {doc['title']}")
        else:
            print(f"❌ 添加文档失败: {doc['title']}")
    
    print(f"📊 成功添加 {success_count}/{len(sample_docs)} 个文档")

def demo_question_answering(system: IntelligentKnowledgeSystem):
    """演示问答功能"""
    print("\\n💬 智能问答演示")
    print("=" * 50)
    
    test_questions = [
        "什么是人工智能？",
        "机器学习有哪些主要类型？",
        "深度学习和传统机器学习有什么区别？",
        "Transformer架构的主要特点是什么？",
        "谁提出了图灵测试？",
        "CNN主要用于什么领域？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\\n❓ 问题 {i}: {question}")
        print("-" * 40)
        
        result = system.ask_question(question)
        
        print(f"🤖 回答: {result.answer}")
        print(f"📚 来源文档: {', '.join(result.sources)}")
        print(f"🎯 置信度: {result.confidence:.2f}")
        print(f"⏱️ 处理时间: {result.processing_time:.3f}秒")

def demo_document_search(system: IntelligentKnowledgeSystem):
    """演示文档搜索"""
    print("\\n🔍 文档搜索演示")
    print("=" * 50)
    
    search_queries = [
        "神经网络",
        "监督学习",
        "图像识别",
        "自然语言处理"
    ]
    
    for query in search_queries:
        print(f"\\n🔎 搜索: {query}")
        print("-" * 30)
        
        results = system.search_documents(query, limit=3)
        
        for i, result in enumerate(results, 1):
            doc_title = result["metadata"].get("doc_title", "未知文档")
            content_preview = result["content"][:150] + "..."
            score = result["score"]
            
            print(f"  {i}. {doc_title} (相似度: {score:.4f})")
            print(f"     {content_preview}")

def main():
    """主函数"""
    try:
        print("🧠 LangChain Challenge 8: 智能知识管理系统")
        print("=" * 60)
        
        # 初始化系统
        print("🚀 初始化智能知识管理系统...")
        system = IntelligentKnowledgeSystem()
        
        # 创建示例文档
        create_sample_documents(system)
        
        # 演示问答功能
        demo_question_answering(system)
        
        # 演示文档搜索
        demo_document_search(system)
        
        # 显示系统统计
        print("\\n📊 系统统计信息")
        print("=" * 50)
        stats = system.get_system_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\\n" + "=" * 60)
        print("🎯 系统功能特点:")
        print("✅ 多格式文档处理和索引")
        print("✅ 智能文档搜索和检索")
        print("✅ 基于RAG的问答系统")
        print("✅ 知识实体提取和管理")
        print("✅ 文档摘要自动生成")
        print("✅ 向量化存储和语义检索")
        print("✅ 数据库持久化存储")
        print("✅ 模块化和可扩展架构")
        
        print("\\n🚀 扩展建议:")
        print("1. 添加Web界面和API服务")
        print("2. 实现多用户权限管理")
        print("3. 集成更多文档格式支持")
        print("4. 添加知识图谱可视化")
        print("5. 实现分布式存储和计算")
        print("6. 添加实时协作功能")
        print("7. 集成外部数据源")
        
    except Exception as e:
        print(f"❌ 系统错误: {e}")
        print("\\n请确保:")
        print("1. 已设置 OPENAI_API_KEY 环境变量")
        print("2. 已安装所需的依赖包:")
        print("   pip install langchain langchain-openai langchain-community")
        print("   pip install faiss-cpu sqlite3")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import time
    main()
