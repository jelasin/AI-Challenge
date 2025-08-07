# -*- coding: utf-8 -*-
"""
Challenge 6: SQLite数据库MCP服务器集成

学习目标:
1. 掌握SQLite数据库与MCP的集成
2. 学习数据库CRUD操作工具的实现
3. 理解数据库资源的MCP暴露方式
4. 实现安全的数据库查询接口

核心概念:
- SQLite Database: SQLite轻量级数据库
- CRUD Operations: 创建、读取、更新、删除操作
- Database Resources: 数据库表数据资源化
- SQL Query Tools: 安全的SQL查询工具
- Database Schema: 数据库表结构管理

实战场景:
使用专门的SQLite MCP服务器进行用户管理、任务管理
和数据库查询操作，展示数据库与MCP的完美集成。
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # 客户端导入
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装必要的包:")
    print("pip install langchain-mcp-adapters langchain-openai")
    sys.exit(1)

class SQLiteMCPManager:
    """SQLite MCP管理器 - 专注于数据库操作"""
    
    def __init__(self):
        """初始化SQLite MCP管理器"""
        # SQLite MCP服务器配置
        self.server_configs = {
            "sqlite": {
                "command": "python",
                "args": [str(project_root / "mcp_servers" / "sqlite_server.py")],
                "transport": "stdio"
            }
        }
        
        self.client: Optional[MultiServerMCPClient] = None
        self.llm: Optional[ChatOpenAI] = None
        self.available_tools: List = []
        
        # 统计信息
        self.stats = {
            "tools_loaded": 0,
            "database_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "resources_accessed": 0,
            "sql_queries": 0
        }
    
    async def initialize(self) -> bool:
        """初始化SQLite MCP客户端和工具"""
        print("🔧 初始化SQLite MCP管理系统...")
        
        try:
            # 初始化MCP客户端
            print("📡 连接到SQLite MCP服务器...")
            self.client = MultiServerMCPClient(self.server_configs)  # type: ignore
            
            # 加载SQLite工具
            print("⚡ 加载SQLite数据库工具...")
            self.available_tools = await self.client.get_tools()
            self.stats["tools_loaded"] = len(self.available_tools)
            print(f"✅ 成功加载 {len(self.available_tools)} 个SQLite工具")
            
            # 显示可用工具
            print("\n🛠️  SQLite工具列表:")
            for tool in self.available_tools:
                print(f"  • {tool.name}: {tool.description}")
            
            # 初始化LLM（如果有API密钥）
            if os.getenv("OPENAI_API_KEY"):
                self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
                print("🤖 LLM助手已启用")
            else:
                print("⚠️  未设置OPENAI_API_KEY，将使用直接工具调用模式")
            
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False
    
    async def demonstrate_user_operations(self):
        """演示用户数据库操作"""
        print("\n" + "="*50)
        print("👥 用户数据库操作演示")
        print("="*50)
        
        try:
            # 1. 查看现有用户
            print("\n📋 1. 查看现有用户:")
            list_users_tool = self._get_tool("list_users")
            if list_users_tool:
                result = await list_users_tool.ainvoke({"limit": 5})
                print(result)
                self.stats["successful_operations"] += 1
            
            # 2. 创建新用户
            print("\n👤 2. 创建新用户:")
            create_user_tool = self._get_tool("create_user")
            if create_user_tool:
                result = await create_user_tool.ainvoke({
                    "name": "数据库管理员",
                    "email": f"db_admin_{datetime.now().strftime('%H%M%S')}@example.com"
                })
                print(result)
                self.stats["successful_operations"] += 1
            
            # 3. 根据ID获取用户详情
            print("\n🔍 3. 根据ID获取用户详情:")
            get_user_tool = self._get_tool("get_user_by_id")
            if get_user_tool:
                result = await get_user_tool.ainvoke({"user_id": 1})
                print(result)
                self.stats["successful_operations"] += 1
            
            # 4. 再次列出用户查看变化
            print("\n📊 4. 确认用户数据更新:")
            if list_users_tool:
                result = await list_users_tool.ainvoke({"limit": 10})
                print(result)
                self.stats["successful_operations"] += 1
            
            self.stats["database_operations"] += 4
            
        except Exception as e:
            print(f"❌ 用户操作演示失败: {e}")
            self.stats["failed_operations"] += 1
    
    async def demonstrate_task_operations(self):
        """演示任务数据库操作"""
        print("\n" + "="*50)
        print("📋 任务数据库操作演示")
        print("="*50)
        
        try:
            # 1. 创建新任务
            print("\n📝 1. 创建新任务:")
            create_task_tool = self._get_tool("create_task")
            if create_task_tool:
                result = await create_task_tool.ainvoke({
                    "title": "SQLite数据库性能优化",
                    "description": "分析和优化数据库查询性能，建立索引策略",
                    "user_id": 1
                })
                print(result)
                self.stats["successful_operations"] += 1
            
            # 2. 查看所有任务
            print("\n📋 2. 查看所有任务:")
            list_tasks_tool = self._get_tool("list_tasks")
            if list_tasks_tool:
                result = await list_tasks_tool.ainvoke({"limit": 10})
                print(result)
                self.stats["successful_operations"] += 1
            
            # 3. 查看特定用户的任务
            print("\n👤 3. 查看用户1的所有任务:")
            get_user_tasks_tool = self._get_tool("get_user_tasks")
            if get_user_tasks_tool:
                result = await get_user_tasks_tool.ainvoke({"user_id": 1})
                print(result)
                self.stats["successful_operations"] += 1
            
            # 4. 更新任务状态
            print("\n🔄 4. 更新任务状态为进行中:")
            update_task_tool = self._get_tool("update_task_status")
            if update_task_tool:
                result = await update_task_tool.ainvoke({
                    "task_id": 1,
                    "status": "in_progress"
                })
                print(result)
                self.stats["successful_operations"] += 1
            
            # 5. 按状态筛选任务
            print("\n🔍 5. 查看进行中的任务:")
            if list_tasks_tool:
                result = await list_tasks_tool.ainvoke({
                    "status": "in_progress",
                    "limit": 5
                })
                print(result)
                self.stats["successful_operations"] += 1
            
            self.stats["database_operations"] += 5
            
        except Exception as e:
            print(f"❌ 任务操作演示失败: {e}")
            self.stats["failed_operations"] += 1
    
    async def demonstrate_database_queries(self):
        """演示数据库查询功能"""
        print("\n" + "="*50)
        print("🔍 数据库查询功能演示")
        print("="*50)
        
        try:
            # 1. 执行统计查询
            print("\n📊 1. 统计每个用户的任务数量:")
            sql_tool = self._get_tool("execute_sql")
            if sql_tool:
                result = await sql_tool.ainvoke({
                    "query": "SELECT u.name, COUNT(t.id) as task_count FROM users u LEFT JOIN tasks t ON u.id = t.user_id GROUP BY u.id, u.name ORDER BY task_count DESC",
                    "description": "统计用户任务分布"
                })
                print(result)
                self.stats["successful_operations"] += 1
                self.stats["sql_queries"] += 1
            
            # 2. 查询任务状态统计
            print("\n📈 2. 任务状态统计:")
            if sql_tool:
                result = await sql_tool.ainvoke({
                    "query": "SELECT status, COUNT(*) as count FROM tasks GROUP BY status ORDER BY count DESC",
                    "description": "任务状态分布统计"
                })
                print(result)
                self.stats["successful_operations"] += 1
                self.stats["sql_queries"] += 1
            
            # 3. 查询最近创建的记录
            print("\n🕐 3. 查询最近创建的用户和任务:")
            if sql_tool:
                result = await sql_tool.ainvoke({
                    "query": "SELECT 'user' as type, name as title, created_at FROM users UNION ALL SELECT 'task' as type, title, created_at FROM tasks ORDER BY created_at DESC LIMIT 10",
                    "description": "最近活动记录"
                })
                print(result)
                self.stats["successful_operations"] += 1
                self.stats["sql_queries"] += 1
            
            # 4. 统计记录数量
            print("\n🔢 4. 统计各表记录数量:")
            count_tool = self._get_tool("count_records")
            if count_tool:
                for table in ["users", "tasks", "operation_logs"]:
                    result = await count_tool.ainvoke({"table_name": table})
                    print(f"  {result}")
                    self.stats["successful_operations"] += 1
            
            self.stats["database_operations"] += 6
            
        except Exception as e:
            print(f"❌ 数据库查询演示失败: {e}")
            self.stats["failed_operations"] += 1
    
    async def demonstrate_schema_operations(self):
        """演示数据库结构操作"""
        print("\n" + "="*50)
        print("🗃️ 数据库结构操作演示")
        print("="*50)
        
        try:
            # 1. 查看表结构
            print("📋 1. 查看数据库表结构:")
            schema_tool = self._get_tool("get_table_schema")
            if schema_tool:
                tables = ["users", "tasks", "operation_logs"]
                for table in tables:
                    print(f"\n🗂️ 表 '{table}' 结构:")
                    result = await schema_tool.ainvoke({"table_name": table})
                    # 简化显示，只显示关键信息
                    lines = str(result).split('\n')
                    key_lines = [line for line in lines if any(keyword in line for keyword in ['列名:', '类型:', '主键:', '非空:'])]
                    for line in key_lines[:12]:  # 限制显示行数
                        print(f"  {line}")
                    if len(key_lines) > 12:
                        print(f"  ... (显示了前12行，共{len(key_lines)}行)")
                    
                    self.stats["successful_operations"] += 1
            
            self.stats["database_operations"] += 3
            
        except Exception as e:
            print(f"❌ 数据库结构操作演示失败: {e}")
            self.stats["failed_operations"] += 1
    
    async def demonstrate_database_resources(self):
        """演示数据库资源访问"""
        print("\n" + "="*50)
        print("📁 数据库资源访问演示")
        print("="*50)
        
        try:
            if self.client:
                # 获取可用资源
                print("📂 1. 获取SQLite数据库资源列表:")
                resources = await self.client.get_resources("sqlite")  # type: ignore
                resource_names = []
                for r in resources:
                    if hasattr(r, 'name'):
                        resource_names.append(r.name)  # type: ignore
                print(f"可用资源: {resource_names}")
                
                # 访问用户数据资源
                print("\n👥 2. 访问用户数据资源:")
                user_data_result = await self.client.get_resources("sqlite", uris=["sqlite://users"])  # type: ignore
                if user_data_result and len(user_data_result) > 0:
                    user_data = str(user_data_result[0])
                    # 解析JSON并格式化显示
                    try:
                        user_list = json.loads(user_data)
                        print(f"用户数据 ({len(user_list)} 个用户):")
                        for user in user_list[:3]:  # 只显示前3个
                            print(f"  • ID: {user.get('id', 'N/A')}, 姓名: {user.get('name', 'N/A')}, 邮箱: {user.get('email', 'N/A')}")
                        if len(user_list) > 3:
                            print(f"  ... (还有 {len(user_list) - 3} 个用户)")
                    except json.JSONDecodeError:
                        display_data = user_data[:200] + "..." if len(user_data) > 200 else user_data
                        print(f"用户数据: {display_data}")
                
                # 访问任务数据资源
                print("\n📋 3. 访问任务数据资源:")
                task_result = await self.client.get_resources("sqlite", uris=["sqlite://tasks"])  # type: ignore
                if task_result and len(task_result) > 0:
                    task_data = str(task_result[0])
                    try:
                        task_list = json.loads(task_data)
                        print(f"任务数据 ({len(task_list)} 个任务):")
                        for task in task_list[:3]:  # 只显示前3个
                            status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "cancelled": "❌"}
                            emoji = status_emoji.get(task.get('status', ''), '📋')
                            print(f"  • {emoji} ID: {task.get('id', 'N/A')}, 标题: {task.get('title', 'N/A')}, 状态: {task.get('status', 'N/A')}")
                        if len(task_list) > 3:
                            print(f"  ... (还有 {len(task_list) - 3} 个任务)")
                    except json.JSONDecodeError:
                        display_data = task_data[:200] + "..." if len(task_data) > 200 else task_data
                        print(f"任务数据: {display_data}")
                
                # 访问数据库结构资源
                print("\n🗃️ 4. 访问数据库结构资源:")
                schema_result = await self.client.get_resources("sqlite", uris=["sqlite://schema"])  # type: ignore
                if schema_result and len(schema_result) > 0:
                    schema_data = str(schema_result[0])
                    try:
                        schema_info = json.loads(schema_data)
                        print("数据库表结构:")
                        for table, columns in schema_info.items():
                            print(f"  📋 {table} 表 ({len(columns)} 个字段):")
                            for col in columns[:2]:  # 只显示前2个字段
                                print(f"    - {col.get('name', 'N/A')} ({col.get('type', 'N/A')})")
                            if len(columns) > 2:
                                print(f"    ... (还有 {len(columns) - 2} 个字段)")
                    except json.JSONDecodeError:
                        display_data = schema_data[:200] + "..." if len(schema_data) > 200 else schema_data
                        print(f"结构数据: {display_data}")
                
                self.stats["resources_accessed"] += 4
                self.stats["successful_operations"] += 4
            
        except Exception as e:
            print(f"❌ 数据库资源访问演示失败: {e}")
            self.stats["failed_operations"] += 1
    
    async def demonstrate_intelligent_analysis(self):
        """演示智能数据分析"""
        print("\n" + "="*50)
        print("🤖 智能数据分析演示")
        print("="*50)
        
        if not self.llm:
            print("⚠️  需要OpenAI API密钥来启用智能数据分析")
            return
        
        try:
            # 使用LLM分析数据库统计信息
            print("🧠 1. LLM分析数据库状态:")
            
            if self.client:
                # 获取数据库统计信息
                user_data_result = await self.client.get_resources("sqlite", uris=["sqlite://users"])  # type: ignore
                task_data_result = await self.client.get_resources("sqlite", uris=["sqlite://tasks"])  # type: ignore
                
                if user_data_result and task_data_result:
                    user_data = str(user_data_result[0])
                    task_data = str(task_data_result[0])
                    
                    # 构建分析提示
                    analysis_prompt = f"""
                    作为数据库分析专家，请分析以下SQLite数据库数据并提供洞察：

                    用户数据:
                    {user_data[:500]}...

                    任务数据:
                    {task_data[:500]}...

                    请分析：
                    1. 数据库整体状况评估
                    2. 用户与任务的关联度分析
                    3. 任务完成效率评估
                    4. 数据质量和完整性检查
                    5. 数据库优化建议
                    6. 业务洞察和改进建议

                    请提供简洁但有深度的分析报告。
                    """
                    
                    messages = [HumanMessage(content=analysis_prompt)]
                    response = await self.llm.ainvoke(messages)
                    
                    print("📊 数据库智能分析报告:")
                    print(response.content)
                    
                    self.stats["successful_operations"] += 1
            
        except Exception as e:
            print(f"❌ 智能数据分析演示失败: {e}")
            self.stats["failed_operations"] += 1
    
    def _get_tool(self, tool_name: str):
        """获取指定名称的工具"""
        return next((tool for tool in self.available_tools if tool.name == tool_name), None)
    
    def display_sqlite_architecture(self):
        """显示SQLite MCP架构信息"""
        print("\n" + "="*60)
        print("🏗️  SQLite MCP服务器架构")
        print("="*60)
        
        architecture_info = {
            "服务器类型": "SQLite专用MCP服务器 (sqlite_server.py)",
            "数据库类型": "SQLite 轻量级数据库",
            "传输协议": "stdio",
            "工具数量": len(self.available_tools),
            "核心功能": [
                "👥 用户管理 - 创建、查询用户信息",
                "📋 任务管理 - 任务CRUD操作和状态管理", 
                "🔍 SQL查询 - 安全的SELECT查询执行",
                "📊 记录统计 - 表记录数量统计",
                "🗃️ 结构查询 - 数据库表结构获取",
                "📁 资源访问 - 数据库表内容资源化"
            ],
            "安全特性": [
                "🔒 SQL注入防护 - 仅允许SELECT查询",
                "📝 操作日志记录 - 所有操作可追踪",
                "🛡️ 参数验证 - 严格的输入验证",
                "⚡ 异常处理 - 优雅的错误处理机制"
            ],
            "数据库表": [
                "users - 用户信息表 (id, name, email, created_at)",
                "tasks - 任务管理表 (id, title, description, status, user_id, created_at, updated_at)",
                "operation_logs - 操作日志表 (id, operation, details, timestamp)"
            ],
            "资源类型": [
                "sqlite://users - 用户数据JSON资源",
                "sqlite://tasks - 任务数据JSON资源",
                "sqlite://logs - 操作日志JSON资源",
                "sqlite://schema - 数据库结构JSON资源"
            ]
        }
        
        for key, value in architecture_info.items():
            print(f"\n📌 {key}:")
            if isinstance(value, list):
                for item in value:
                    print(f"   {item}")
            else:
                print(f"   {value}")
    
    def display_statistics(self):
        """显示执行统计信息"""
        print("\n" + "="*50)
        print("📊 SQLite操作统计信息")
        print("="*50)
        
        success_rate = (self.stats["successful_operations"] / max(self.stats["database_operations"], 1)) * 100
        
        print(f"🛠️  工具加载数量: {self.stats['tools_loaded']} 个")
        print(f"💾 数据库操作总数: {self.stats['database_operations']}")
        print(f"✅ 成功操作数: {self.stats['successful_operations']}")
        print(f"❌ 失败操作数: {self.stats['failed_operations']}")
        print(f"📁 资源访问次数: {self.stats['resources_accessed']}")
        print(f"🔍 SQL查询次数: {self.stats['sql_queries']}")
        print(f"📈 操作成功率: {success_rate:.1f}%")

async def demo_sqlite_database_integration():
    """Challenge 6 主演示函数"""
    print("🚀 Challenge 6: SQLite数据库MCP服务器集成")
    print("="*60)
    
    # 创建SQLite MCP管理器
    sqlite_manager = SQLiteMCPManager()
    
    # 初始化系统
    if not await sqlite_manager.initialize():
        print("❌ 无法初始化SQLite MCP系统，演示结束")
        return
    
    try:
        # 显示服务器架构
        sqlite_manager.display_sqlite_architecture()
        
        # 演示各种数据库操作
        await sqlite_manager.demonstrate_user_operations()
        
        await sqlite_manager.demonstrate_task_operations()
        
        await sqlite_manager.demonstrate_database_queries()
        
        await sqlite_manager.demonstrate_schema_operations()
        
        await sqlite_manager.demonstrate_database_resources()
        
        await sqlite_manager.demonstrate_intelligent_analysis()
        
        # 显示统计信息
        sqlite_manager.display_statistics()
        
        print("\n🎉 Challenge 6 演示完成！")
        print("\n📚 学习成果总结:")
        print("  ✅ 掌握了SQLite数据库与MCP的深度集成")
        print("  ✅ 实现了完整的数据库CRUD操作工具")
        print("  ✅ 学会了安全的SQL查询接口设计")
        print("  ✅ 体验了数据库资源的MCP暴露方式")
        print("  ✅ 理解了数据库表结构的动态查询")
        print("  ✅ 掌握了数据库操作日志记录机制")
        print("  ✅ 实现了智能化的数据库分析功能")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """直接运行此Challenge的主函数"""
    asyncio.run(demo_sqlite_database_integration())

if __name__ == "__main__":
    main()
