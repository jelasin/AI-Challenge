# -*- coding: utf-8 -*-
"""
Challenge 3: MCP资源管理和访问

学习目标:
1. 掌握MCP资源的发现、加载和管理
2. 学习动态资源访问和URI处理
3. 实现资源缓存和性能优化策略
4. 理解结构化数据的资源处理

核心概念:
- Resource Discovery: 资源发现和枚举
- Dynamic Resource Loading: 动态资源加载
- Resource URI Handling: 资源URI处理和解析
- Resource Caching: 资源缓存机制
- Structured Data Processing: 结构化数据处理

实战场景:
构建一个智能文档分析系统，能够从MCP服务器动态发现和加载
各种类型的文档资源，进行内容分析、格式转换和智能摘要。
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib

# 添加类型转换函数
def cast(target_type: Any, value: Any) -> Any:
    """类型转换函数"""
    return value

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.resources import load_mcp_resources
    MCP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  MCP适配器不可用: {e}")
    MultiServerMCPClient = None
    MCP_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  OpenAI不可用: {e}")
    ChatOpenAI = None
    OPENAI_AVAILABLE = False

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.documents import Document
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  LangChain核心不可用: {e}")
    LANGCHAIN_CORE_AVAILABLE = False

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Pydantic不可用: {e}")
    BaseModel = object
    Field = lambda **kwargs: None
    PYDANTIC_AVAILABLE = False

class SimpleResource:
    """简单的资源类（替代Blob）"""
    def __init__(self, data: Union[str, bytes], path: str = "", name: str = ""):
        self.data = data
        self.path = path
        self.name = name or path.split('/')[-1] if path else "unnamed"

class ResourceMetadata:
    """资源元数据模型"""
    def __init__(self, uri: str, name: str, size: Optional[int] = None, 
                 mime_type: Optional[str] = None, created_at: Optional[datetime] = None,
                 accessed_at: Optional[datetime] = None, checksum: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        self.uri = uri
        self.name = name
        self.size = size
        self.mime_type = mime_type
        self.created_at = created_at or datetime.now()
        self.accessed_at = accessed_at
        self.checksum = checksum
        self.tags = tags or []

class ResourceCache:
    """资源缓存管理器"""
    
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, SimpleResource] = {}
        self.metadata: Dict[str, ResourceMetadata] = {}
        self.access_order: List[str] = []
        self.max_size = max_size
    
    def _evict_lru(self):
        """LRU缓存淘汰策略"""
        if len(self.cache) >= self.max_size and self.access_order:
            oldest_uri = self.access_order.pop(0)
            self.cache.pop(oldest_uri, None)
            self.metadata.pop(oldest_uri, None)
    
    def put(self, uri: str, resource: SimpleResource, metadata: ResourceMetadata):
        """添加资源到缓存"""
        self._evict_lru()
        self.cache[uri] = resource
        self.metadata[uri] = metadata
        
        # 更新访问顺序
        if uri in self.access_order:
            self.access_order.remove(uri)
        self.access_order.append(uri)
    
    def get(self, uri: str) -> Optional[SimpleResource]:
        """从缓存获取资源"""
        if uri in self.cache:
            # 更新访问顺序
            self.access_order.remove(uri)
            self.access_order.append(uri)
            
            # 更新访问时间
            self.metadata[uri].accessed_at = datetime.now()
            
            return self.cache[uri]
        return None
    
    def contains(self, uri: str) -> bool:
        """检查缓存中是否包含资源"""
        return uri in self.cache
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.metadata.clear()
        self.access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_size = sum(len(blob.data) if hasattr(blob, 'data') else 0 
                        for blob in self.cache.values())
        
        return {
            "count": len(self.cache),
            "max_size": self.max_size,
            "total_size_bytes": total_size,
            "hit_ratio": len([m for m in self.metadata.values() if m.accessed_at]) / max(len(self.metadata), 1)
        }

class MCPResourceManager:
    """MCP资源管理器"""
    
    def __init__(self):
        """初始化资源管理器"""
        # 服务器配置
        self.server_configs = {
            "file": {
                "command": "python",
                "args": [str(project_root / "mcp_servers" / "file_server.py")],
                "transport": "stdio"
            }
        }
        
        # 管理器组件
        self.mcp_client = None
        self.resource_cache = ResourceCache(max_size=50)
        self.discovered_resources: Dict[str, List[ResourceMetadata]] = {}
        
        # LLM用于内容分析
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY") and ChatOpenAI:
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1
            )
        else:
            self.llm = None
    
    async def initialize(self) -> bool:
        """初始化资源管理器"""
        print("🔧 初始化MCP资源管理器...")
        
        try:
            # 创建MCP客户端 - 使用类型转换避免类型错误
            if MCP_AVAILABLE and MultiServerMCPClient:
                try:
                    configs = cast(Any, self.server_configs)
                    self.mcp_client = MultiServerMCPClient(configs)
                    print("✅ MCP客户端初始化成功")
                except Exception as mcp_e:
                    print(f"⚠️  MCP客户端初始化失败: {mcp_e}")
                    self.mcp_client = None
            else:
                print("⚠️  MCP客户端不可用，使用备用模式")
                self.mcp_client = None
            
            # 确保工作目录存在并创建示例文件
            await self.setup_sample_resources()
            
            print("✅ 资源管理器初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            # 创建备用设置以确保演示可以继续
            await self._create_fallback_resources()
            return True
    
    async def _create_fallback_resources(self):
        """创建备用资源以确保演示可以继续"""
        try:
            # 创建示例文件
            sample_content = "This is a sample text file for demonstration."
            workspace_dir = project_root / "mcp_servers" / "workspace"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            
            sample_path = workspace_dir / "fallback_sample.txt"
            sample_path.write_text(sample_content)
            
            # 创建备用资源对象
            resource = SimpleResource(
                data=sample_content,
                path=str(sample_path),
                name="fallback_sample.txt"
            )
            
            # 创建元数据
            metadata = ResourceMetadata(
                uri=f"file://{sample_path}",
                name="fallback_sample.txt",
                size=len(sample_content),
                created_at=datetime.now(),
                checksum=hashlib.md5(sample_content.encode()).hexdigest()
            )
            
            # 添加到缓存
            self.resource_cache.put(metadata.uri, resource, metadata)
            print("✅ 创建备用资源完成")
            
        except Exception as e:
            print(f"⚠️ 创建备用资源失败: {e}")
    
    async def setup_sample_resources(self):
        """创建示例资源文件"""
        print("📁 创建示例资源文件...")
        
        workspace_dir = project_root / "mcp_servers" / "workspace"
        workspace_dir.mkdir(exist_ok=True)
        
        # 创建各种类型的示例文件
        sample_files = {
            "readme.txt": """# MCP资源管理演示

这是一个演示MCP资源管理功能的文本文件。
内容包含了项目介绍、使用说明和示例代码。

## 功能特性
- 资源发现和枚举
- 动态资源加载
- 缓存管理
- 内容分析

## 技术栈
- Python 3.8+
- LangChain MCP Adapters
- OpenAI GPT-4
""",
            "data.json": json.dumps({
                "project": "MCP Resource Demo",
                "version": "1.0.0",
                "resources": [
                    {"type": "text", "count": 15},
                    {"type": "json", "count": 3},
                    {"type": "markdown", "count": 8}
                ],
                "statistics": {
                    "total_size": "2.5MB",
                    "last_updated": "2024-01-01"
                }
            }, indent=2),
            "config.json": json.dumps({
                "cache_size": 100,
                "auto_discovery": True,
                "supported_formats": ["txt", "json", "md", "py"],
                "analysis_settings": {
                    "enable_summary": True,
                    "max_summary_length": 200
                }
            }, indent=2),
            "sample.py": '''"""
MCP资源管理示例代码
"""

def analyze_resource(content):
    """分析资源内容"""
    return {
        "length": len(content),
        "lines": content.count("\\n"),
        "words": len(content.split())
    }

class ResourceAnalyzer:
    def __init__(self):
        self.processed_count = 0
    
    def process(self, resource):
        self.processed_count += 1
        return analyze_resource(resource)
''',
            "notes.md": """# MCP资源管理笔记

## 核心概念

### 资源类型
- **静态资源**: 固定的文件和数据
- **动态资源**: 需要参数的资源
- **结构化数据**: JSON、XML等格式化数据

### 缓存策略
- LRU (Least Recently Used)
- 大小限制
- 自动清理

### 性能优化
- 异步加载
- 批量处理
- 智能预取

## 最佳实践
1. 合理设置缓存大小
2. 监控资源访问模式
3. 定期清理无用资源
"""
        }
        
        for filename, content in sample_files.items():
            file_path = workspace_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"✅ 创建了 {len(sample_files)} 个示例文件")
    
    async def discover_resources(self) -> Dict[str, List[ResourceMetadata]]:
        """通过MCP工具发现所有可用资源"""
        print("\n🔍 通过MCP工具发现服务器资源...")
        
        self.discovered_resources.clear()
        
        for server_name in self.server_configs.keys():
            print(f"📡 扫描服务器: {server_name}")
            
            mcp_resources = []
            
            # 优先使用MCP工具进行资源发现
            if self.mcp_client:
                try:
                    # 方法1: 使用MCP资源API
                    print(f"  🔧 使用MCP资源API扫描...")
                    try:
                        resources = await asyncio.wait_for(
                            cast(Any, self.mcp_client).get_resources(server_name),
                            timeout=3.0
                        )
                        
                        for resource in resources:
                            # 资源对象是LangChain的Blob对象
                            if hasattr(resource, 'metadata') and 'uri' in resource.metadata:
                                uri = str(resource.metadata['uri'])
                                # 移除URI中的结尾斜杠
                                uri = uri.rstrip('/')
                                name = uri.split('/')[-1] if '/' in uri else uri.split('://')[-1]
                                
                                metadata = ResourceMetadata(
                                    uri=uri,
                                    name=name,
                                    mime_type=self._guess_mime_type(Path(name).suffix),
                                    created_at=datetime.now()
                                )
                                mcp_resources.append(metadata)
                            else:
                                # 备用处理方式
                                uri = str(resource)
                                name = uri.split('/')[-1] if '/' in uri else uri
                                
                                metadata = ResourceMetadata(
                                    uri=uri,
                                    name=name,
                                    mime_type="application/octet-stream",
                                    created_at=datetime.now()
                                )
                                mcp_resources.append(metadata)
                        
                        print(f"  ✅ MCP资源API发现 {len(mcp_resources)} 个资源")
                    
                    except Exception as e:
                        print(f"  ⚠️  MCP资源API失败: {e}")
                    
                    # 方法2: 使用工具API获取可用工具
                    print(f"  🔧 获取MCP工具列表...")
                    try:
                        tools = await asyncio.wait_for(
                            cast(Any, self.mcp_client).get_tools(server_name=server_name),
                            timeout=3.0
                        )
                        
                        print(f"  🛠️  发现 {len(tools)} 个MCP工具:")
                        for tool in tools:
                            tool_name = getattr(tool, 'name', str(tool))
                            tool_desc = getattr(tool, 'description', '无描述')
                            print(f"    - {tool_name}: {tool_desc}")
                        
                        # 如果有list_directory工具，尝试使用它
                        list_dir_tool = None
                        read_file_tool = None
                        
                        for tool in tools:
                            if hasattr(tool, 'name'):
                                if tool.name == 'list_directory':
                                    list_dir_tool = tool
                                elif tool.name == 'read_file':
                                    read_file_tool = tool
                        
                        if list_dir_tool:
                            print(f"  📁 使用list_directory工具扫描文件...")
                            try:
                                # 直接调用工具而不是通过call_tool方法
                                tool_result = await list_dir_tool.ainvoke({"directory_path": "."})
                                
                                # 解析工具返回结果
                                content = str(tool_result) if tool_result else ""
                                
                                print(f"  📁 目录列表: {content[:200]}...")
                                
                                # 从返回内容中提取文件名
                                if "文件:" in content:
                                    lines = content.split('\n')
                                    for line in lines:
                                        if line.strip().startswith("文件:"):
                                            file_name = line.split("文件:")[-1].strip()
                                            if file_name and not any(r.name == file_name for r in mcp_resources):
                                                metadata = ResourceMetadata(
                                                    uri=f"file://{file_name}",
                                                    name=file_name,
                                                    mime_type=self._guess_mime_type(Path(file_name).suffix),
                                                    created_at=datetime.now()
                                                )
                                                mcp_resources.append(metadata)
                                
                                print(f"  ✅ 通过工具发现了更多文件")
                            
                            except Exception as e:
                                print(f"  ⚠️  list_directory工具调用失败: {e}")
                    
                    except Exception as e:
                        print(f"  ⚠️  MCP工具获取失败: {e}")
                
                except Exception as e:
                    print(f"  ❌ MCP客户端操作失败: {e}")
            
            # 备用方案：本地文件扫描
            if not mcp_resources:
                print(f"  🔄 使用备用本地扫描...")
                workspace_dir = project_root / "mcp_servers" / "workspace"
                
                if workspace_dir.exists():
                    for file_path in workspace_dir.glob("*"):
                        if file_path.is_file():
                            metadata = ResourceMetadata(
                                uri=f"file://{file_path.relative_to(workspace_dir)}",
                                name=file_path.name,
                                mime_type=self._guess_mime_type(file_path.suffix),
                                created_at=datetime.now(),
                                size=file_path.stat().st_size
                            )
                            mcp_resources.append(metadata)
                
                print(f"  ✅ 备用扫描发现 {len(mcp_resources)} 个资源")
            
            self.discovered_resources[server_name] = mcp_resources
        
        total_resources = sum(len(resources) for resources in self.discovered_resources.values())
        print(f"\n📊 通过MCP工具总计发现 {total_resources} 个资源")
        
        return self.discovered_resources
    
    def _guess_mime_type(self, suffix: str) -> str:
        """根据文件后缀猜测MIME类型"""
        mime_map = {
            ".txt": "text/plain",
            ".json": "application/json",
            ".md": "text/markdown",
            ".py": "text/x-python",
            ".js": "text/javascript",
            ".html": "text/html",
            ".css": "text/css",
            ".yaml": "text/yaml",
            ".yml": "text/yaml"
        }
        return mime_map.get(suffix.lower(), "application/octet-stream")
    
    async def load_resource_with_cache(self, server_name: str, uri: str) -> Optional[SimpleResource]:
        """通过MCP工具加载资源到缓存"""
        # 检查缓存
        if self.resource_cache.contains(uri):
            print(f"🎯 缓存命中: {uri}")
            return self.resource_cache.get(uri)
        
        print(f"📥 通过MCP工具加载: {uri}")
        
        # 从URI中提取文件路径
        file_path = uri
        if uri.startswith("file://"):
            file_path = uri[7:]  # 移除 "file://" 前缀
        
        # 优先使用MCP工具
        if self.mcp_client:
            try:
                print(f"  🔧 获取MCP工具列表...")
                
                # 获取可用工具
                tools = await asyncio.wait_for(
                    cast(Any, self.mcp_client).get_tools(server_name=server_name),
                    timeout=3.0
                )
                
                # 寻找read_file工具
                read_file_tool = None
                for tool in tools:
                    if hasattr(tool, 'name') and tool.name == 'read_file':
                        read_file_tool = tool
                        break
                
                if read_file_tool:
                    print(f"  🛠️  使用read_file工具读取: {file_path}")
                    
                    # 直接调用工具
                    tool_result = await asyncio.wait_for(
                        read_file_tool.ainvoke({"file_path": file_path}),
                        timeout=3.0
                    )
                    
                    # 解析工具返回结果
                    content = str(tool_result) if tool_result else ""
                    
                    # 如果内容包含"文件 xxx 内容:"前缀，去除它
                    if content.startswith(f"文件 {file_path} 内容:\n\n"):
                        content = content[len(f"文件 {file_path} 内容:\n\n"):]
                    elif "内容:\n\n" in content:
                        content = content.split("内容:\n\n", 1)[1]
                    
                    print(f"  ✅ MCP工具读取成功，内容长度: {len(content)} 字符")
                    
                    # 创建资源对象
                    resource = SimpleResource(
                        data=content,
                        path=uri,
                        name=file_path.split('/')[-1] if '/' in file_path else file_path
                    )
                    
                    # 创建元数据
                    metadata = ResourceMetadata(
                        uri=uri,
                        name=resource.name,
                        size=len(content),
                        checksum=hashlib.md5(content.encode()).hexdigest(),
                        created_at=datetime.now(),
                        mime_type=self._guess_mime_type(Path(file_path).suffix)
                    )
                    
                    # 添加到缓存
                    self.resource_cache.put(uri, resource, metadata)
                    return resource
                else:
                    print(f"  ⚠️  未找到read_file工具")
                    
            except asyncio.TimeoutError:
                print(f"  ⚠️  MCP工具超时")
            except Exception as e:
                print(f"  ⚠️  MCP工具调用失败: {e}")
        
        # 备用方案: 本地文件读取（作为最后手段）
        if uri.startswith("file://"):
            try:
                print(f"  🔄 尝试本地文件备用方案...")
                file_path_obj = Path(file_path)
                
                # 如果是相对路径，相对于workspace目录
                if not file_path_obj.is_absolute():
                    workspace_dir = project_root / "mcp_servers" / "workspace"
                    file_path_obj = workspace_dir / file_path_obj
                
                if file_path_obj.exists():
                    content = file_path_obj.read_text(encoding='utf-8')
                    
                    resource = SimpleResource(
                        data=content,
                        path=str(file_path_obj),
                        name=file_path_obj.name
                    )
                    
                    metadata = ResourceMetadata(
                        uri=uri,
                        name=file_path_obj.name,
                        size=len(content),
                        checksum=hashlib.md5(content.encode()).hexdigest(),
                        created_at=datetime.now(),
                        mime_type=self._guess_mime_type(file_path_obj.suffix)
                    )
                    
                    self.resource_cache.put(uri, resource, metadata)
                    print(f"  ✅ 本地文件备用方案成功")
                    return resource
                    
            except Exception as e:
                print(f"  ❌ 本地文件备用方案失败: {e}")
        
        print(f"  ❌ 所有加载方案均失败")
        return None
    
    async def demonstrate_resource_discovery(self):
        """演示资源发现功能"""
        print("\n" + "="*60)
        print("🔍 资源发现演示")
        print("="*60)
        
        # 发现所有资源
        discovered = await self.discover_resources()
        
        # 按服务器显示资源
        for server_name, resources in discovered.items():
            if resources:
                print(f"\n📂 {server_name} 服务器资源:")
                for i, resource in enumerate(resources, 1):
                    print(f"  {i}. {resource.name}")
                    print(f"     URI: {resource.uri}")
                    print(f"     类型: {resource.mime_type or '未知'}")
                    if resource.size:
                        print(f"     大小: {resource.size} 字节")
    
    async def demonstrate_resource_loading(self):
        """演示资源加载和缓存"""
        print("\n" + "="*60)
        print("💾 资源加载和缓存演示")
        print("="*60)
        
        # 选择一些资源进行加载测试
        test_uris = []
        
        for server_name, resources in self.discovered_resources.items():
            test_uris.extend([(server_name, r.uri) for r in resources[:3]])  # 取前3个
        
        if not test_uris:
            print("⚠️  没有可用资源进行测试")
            return
        
        print(f"📥 准备加载 {len(test_uris)} 个资源进行测试...")
        
        # 首次加载（从服务器）
        print(f"\n🔄 首次加载测试:")
        for server_name, uri in test_uris:
            resource = await self.load_resource_with_cache(server_name, uri)
            if resource:
                print(f"  ✅ {uri}: 加载成功")
            else:
                print(f"  ❌ {uri}: 加载失败")
        
        # 显示缓存统计
        cache_stats = self.resource_cache.stats()
        print(f"\n📊 缓存状态:")
        print(f"  • 缓存项数: {cache_stats['count']}")
        print(f"  • 总大小: {cache_stats['total_size_bytes']} 字节")
        print(f"  • 命中率: {cache_stats['hit_ratio']:.2%}")
        
        # 再次加载相同资源（从缓存）
        print(f"\n🎯 缓存测试 - 重新加载相同资源:")
        for server_name, uri in test_uris[:2]:  # 只测试前两个
            resource = await self.load_resource_with_cache(server_name, uri)
            if resource:
                print(f"  ✅ {uri}: 从缓存获取")
    
    async def demonstrate_mcp_tool_analysis(self):
        """演示通过MCP工具进行智能文件分析"""
        print("\n" + "="*60)
        print("🔧 MCP工具智能分析演示")
        print("="*60)
        
        if not self.mcp_client:
            print("⚠️  MCP客户端不可用，跳过工具分析演示")
            return
        
        # 首先获取可用工具
        print("🛠️  获取MCP工具列表...")
        
        try:
            # 获取第一个服务器的工具（通常是"file"服务器）
            server_name = list(self.server_configs.keys())[0] if self.server_configs else "file"
            
            tools = await asyncio.wait_for(
                cast(Any, self.mcp_client).get_tools(server_name=server_name),
                timeout=5.0
            )
            
            print(f"🔧 发现 {len(tools)} 个MCP工具:")
            for tool in tools:
                tool_name = getattr(tool, 'name', str(tool))
                tool_desc = getattr(tool, 'description', '无描述')
                print(f"  - {tool_name}: {tool_desc}")
            
            # 寻找list_directory工具
            list_dir_tool = None
            read_file_tool = None
            
            for tool in tools:
                if hasattr(tool, 'name'):
                    if tool.name == 'list_directory':
                        list_dir_tool = tool
                    elif tool.name == 'read_file':
                        read_file_tool = tool
            
        except Exception as e:
            print(f"❌ 获取工具列表失败: {e}")
            return
        
        # 使用list_directory工具获取工作目录结构
        if list_dir_tool:
            print("📁 使用list_directory工具分析工作目录结构...")
            
            try:
                tool_result = await asyncio.wait_for(
                    list_dir_tool.ainvoke({"directory_path": "."}),
                    timeout=5.0
                )
                
                content = str(tool_result) if tool_result else ""
                
                print(f"🗂️  目录结构分析结果:")
                print(f"   {content}")
                
                # 提取文件名进行进一步分析
                files_to_analyze = []
                if "文件:" in content:
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip().startswith("文件:"):
                            file_name = line.split("文件:")[-1].strip()
                            if file_name and file_name.endswith(('.txt', '.json', '.md', '.py')):
                                files_to_analyze.append(file_name)
                
                print(f"\n🎯 发现 {len(files_to_analyze)} 个可分析文件: {files_to_analyze}")
                
            except Exception as e:
                print(f"❌ list_directory工具调用失败: {e}")
                return
        else:
            print("⚠️  未找到list_directory工具，跳过目录分析")
            return
        
        # 对每个文件进行详细分析
        analysis_results = {}
        
        for file_name in files_to_analyze[:3]:  # 限制分析数量
            print(f"\n🔍 分析文件: {file_name}")
            
            try:
                # 使用read_file工具读取内容
                if read_file_tool:
                    read_result = await asyncio.wait_for(
                        read_file_tool.ainvoke({"file_path": file_name}),
                        timeout=5.0
                    )
                    
                    # 解析文件内容
                    file_content = str(read_result) if read_result else ""
                    
                    # 清理内容前缀
                    if file_content.startswith(f"文件 {file_name} 内容:\n\n"):
                        file_content = file_content[len(f"文件 {file_name} 内容:\n\n"):]
                    elif "内容:\n\n" in file_content:
                        file_content = file_content.split("内容:\n\n", 1)[1]
                    
                    # 基本统计分析
                    stats = {
                        "文件大小": len(file_content),
                        "行数": file_content.count('\n') + 1 if file_content else 0,
                        "字符数": len(file_content),
                        "单词数": len(file_content.split()) if file_content else 0
                    }
                    
                    # 内容类型分析
                    file_type = "未知"
                    if file_name.endswith('.json'):
                        try:
                            json.loads(file_content)
                            file_type = "有效JSON文档"
                        except:
                            file_type = "无效JSON文档"
                    elif file_name.endswith('.py'):
                        file_type = "Python源代码"
                    elif file_name.endswith('.md'):
                        file_type = "Markdown文档"
                    elif file_name.endswith('.txt'):
                        file_type = "纯文本文档"
                    
                    # 关键内容提取
                    key_content = file_content[:200] + "..." if len(file_content) > 200 else file_content
                    
                    analysis_results[file_name] = {
                        "类型": file_type,
                        "统计": stats,
                        "关键内容": key_content,
                        "通过工具加载": True
                    }
                    
                    print(f"  ✅ 成功分析: {file_type}")
                    print(f"  📊 统计: {stats['行数']} 行, {stats['字符数']} 字符, {stats['单词数']} 单词")
                    
                    # 如果有LLM，进行智能分析
                    if self.llm and file_content:
                        try:
                            print(f"  🤖 进行AI智能分析...")
                            
                            messages = [
                                SystemMessage(content="你是一个专业的文档分析助手。请简要分析文档的用途、结构和关键信息。"),
                                HumanMessage(content=f"文件名: {file_name}\n文件类型: {file_type}\n\n内容:\n{file_content[:500]}...")
                            ]
                            
                            ai_response = await asyncio.wait_for(
                                self.llm.ainvoke(messages),
                                timeout=10.0
                            )
                            
                            analysis_results[file_name]["AI分析"] = str(ai_response.content)[:200] + "..."
                            print(f"  🎯 AI分析: {str(ai_response.content)[:100]}...")
                            
                        except Exception as ai_e:
                            print(f"  ⚠️  AI分析失败: {ai_e}")
                else:
                    print(f"  ⚠️  未找到read_file工具")
                    analysis_results[file_name] = {"错误": "未找到read_file工具"}
                
                await asyncio.sleep(0.5)  # 避免过快调用
                
            except Exception as e:
                print(f"  ❌ 分析失败: {e}")
                analysis_results[file_name] = {"错误": str(e)}
        
        # 汇总分析结果
        print(f"\n📋 MCP工具分析汇总报告:")
        print("="*50)
        
        for file_name, result in analysis_results.items():
            print(f"\n📄 {file_name}:")
            if "错误" in result:
                print(f"   ❌ {result['错误']}")
            else:
                print(f"   类型: {result['类型']}")
                stats = result['统计']
                print(f"   大小: {stats['字符数']} 字符, {stats['行数']} 行")
                print(f"   摘要: {result['关键内容'][:100]}...")
                if "AI分析" in result:
                    print(f"   AI洞察: {result['AI分析'][:150]}...")
        
        print(f"\n✅ 通过MCP工具成功分析了 {len([r for r in analysis_results.values() if '错误' not in r])} 个文件")

    async def demonstrate_content_analysis(self):
        """演示内容分析功能"""
        if not self.llm:
            print("\n⚠️  跳过内容分析演示 - 未设置OPENAI_API_KEY")
            return
        
        print("\n" + "="*60)
        print("🔍 智能内容分析演示")
        print("="*60)
        
        # 选择一些资源进行内容分析
        analysis_targets = []
        
        for server_name, resources in self.discovered_resources.items():
            for resource in resources:
                if resource.mime_type in ["text/plain", None]:  # 只分析文本资源
                    analysis_targets.append((server_name, resource))
                    if len(analysis_targets) >= 3:  # 限制分析数量
                        break
            if len(analysis_targets) >= 3:
                break
        
        for server_name, resource_meta in analysis_targets:
            print(f"\n🔍 分析资源: {resource_meta.name}")
            
            try:
                # 加载资源内容
                resource = await self.load_resource_with_cache(server_name, resource_meta.uri)
                if not resource:
                    print("  ❌ 无法加载资源")
                    continue
                
                # 获取内容
                content = resource.data if hasattr(resource, 'data') else str(resource)
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='ignore')
                
                # 使用LLM分析内容
                messages = [
                    SystemMessage(content="""你是一个智能文档分析助手。请分析给定的文档内容，提供以下信息:
1. 文档类型和用途
2. 主要内容摘要
3. 关键信息点
4. 内容质量评估

请保持分析简洁明了。"""),
                    HumanMessage(content=f"请分析以下文档内容:\n\n```\n{content[:1000]}...\n```")  # 限制内容长度
                ]
                
                response = await self.llm.ainvoke(messages)
                
                print(f"🤖 分析结果:")
                print(f"{response.content}")
                
            except Exception as e:
                print(f"  ❌ 分析失败: {e}")
            
            await asyncio.sleep(1)
    
    async def demonstrate_batch_operations(self):
        """演示批量操作"""
        print("\n" + "="*60)
        print("⚡ 批量资源操作演示")
        print("="*60)
        
        # 批量加载资源 - 去重处理
        batch_uris = []
        seen_uris = set()
        
        for server_name, resources in self.discovered_resources.items():
            for resource in resources:
                # 只添加真实的文件URI，避免虚假的mcp_resource
                if resource.uri.startswith("file://") and resource.uri not in seen_uris:
                    batch_uris.append((server_name, resource.uri))
                    seen_uris.add(resource.uri)
        
        if not batch_uris:
            print("⚠️  没有可用资源进行批量测试")
            return
        
        print(f"⚡ 执行批量加载操作 ({len(batch_uris)} 个唯一资源):")
        
        # 并发加载
        start_time = asyncio.get_event_loop().time()
        
        tasks = []
        for server_name, uri in batch_uris:
            task = self.load_resource_with_cache(server_name, uri)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        # 统计结果
        success_count = sum(1 for r in results if not isinstance(r, Exception) and r is not None)
        error_count = sum(1 for r in results if isinstance(r, Exception))
        
        print(f"\n📊 批量操作结果:")
        print(f"  • 总耗时: {duration:.2f} 秒")
        print(f"  • 成功: {success_count}")
        print(f"  • 失败: {error_count}")
        print(f"  • 平均速度: {len(batch_uris)/duration:.2f} 资源/秒")
        
        # 最终缓存统计
        final_stats = self.resource_cache.stats()
        print(f"\n📈 最终缓存统计:")
        print(f"  • 缓存项: {final_stats['count']}/{final_stats['max_size']}")
        print(f"  • 总大小: {final_stats['total_size_bytes']} 字节")
        print(f"  • 命中率: {final_stats['hit_ratio']:.2%}")

async def demo_resource_management():
    """Challenge 3 主演示函数"""
    print("🚀 Challenge 3: MCP资源管理和访问")
    print("="*60)
    
    # 创建资源管理器
    manager = MCPResourceManager()
    
    # 初始化
    if not await manager.initialize():
        print("❌ 无法初始化资源管理器，演示结束")
        return
    
    try:
        # 1. 资源发现演示
        await manager.demonstrate_resource_discovery()
        
        # 2. MCP工具智能分析演示 (新增)
        await manager.demonstrate_mcp_tool_analysis()
        
        # 3. 资源加载和缓存演示
        await manager.demonstrate_resource_loading()
        
        # 4. 内容分析演示
        await manager.demonstrate_content_analysis()
        
        # 5. 批量操作演示
        await manager.demonstrate_batch_operations()
        
        print("\n🎉 Challenge 3 演示完成！")
        print("\n📚 学习要点总结:")
        print("  ✅ 掌握了通过MCP工具发现和枚举资源")
        print("  ✅ 学会了使用MCP工具进行文件操作和分析")
        print("  ✅ 实现了基于MCP工具的动态资源加载")
        print("  ✅ 体验了MCP工具与AI分析的结合应用")
        print("  ✅ 掌握了高效的资源缓存和批量处理")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")

def main():
    """直接运行此Challenge的主函数"""
    asyncio.run(demo_resource_management())

if __name__ == "__main__":
    main()
