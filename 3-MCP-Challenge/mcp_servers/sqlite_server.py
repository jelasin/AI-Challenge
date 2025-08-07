#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite MCP服务器

功能特性:
- 用户管理: 创建、查询用户信息
- 任务管理: 创建、更新、查询任务
- 数据查询: 安全的SQL查询执行
- 数据库资源: 表数据资源化访问
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    from mcp.types import (
        Tool, TextContent, Resource
    )
except ImportError as e:
    print(f"❌ MCP导入错误: {e}")
    print("请安装MCP包: pip install mcp")
    exit(1)

class DatabaseManager:
    """SQLite数据库管理器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库和表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        # 创建任务表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                user_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 创建日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS operation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                details TEXT,
                timestamp TEXT NOT NULL
            )
        ''')
        
        # 插入示例数据（如果表为空）
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            sample_users = [
                ("张三", "zhangsan@example.com", datetime.now().isoformat()),
                ("李四", "lisi@example.com", datetime.now().isoformat()),
                ("王五", "wangwu@example.com", datetime.now().isoformat()),
                ("赵六", "zhaoliu@example.com", datetime.now().isoformat())
            ]
            cursor.executemany(
                "INSERT INTO users (name, email, created_at) VALUES (?, ?, ?)",
                sample_users
            )
            
            # 添加示例任务
            sample_tasks = [
                ("完成项目设计", "设计系统架构图", 1, "pending", datetime.now().isoformat(), datetime.now().isoformat()),
                ("编写技术文档", "完善API文档", 2, "in_progress", datetime.now().isoformat(), datetime.now().isoformat()),
                ("代码审查", "审查核心模块代码", 3, "completed", datetime.now().isoformat(), datetime.now().isoformat()),
                ("数据库优化", "优化查询性能", 1, "in_progress", datetime.now().isoformat(), datetime.now().isoformat()),
                ("部署测试", "生产环境部署测试", 4, "pending", datetime.now().isoformat(), datetime.now().isoformat())
            ]
            cursor.executemany(
                "INSERT INTO tasks (title, description, user_id, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                sample_tasks
            )
        
        conn.commit()
        conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """执行SQL查询"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute(query, params)
            
            if query.strip().upper().startswith('SELECT'):
                results = [dict(row) for row in cursor.fetchall()]
            else:
                conn.commit()
                results = [{"affected_rows": cursor.rowcount}]
            
            return results
        finally:
            conn.close()
    
    def log_operation(self, operation: str, details: str):
        """记录操作日志"""
        query = "INSERT INTO operation_logs (operation, details, timestamp) VALUES (?, ?, ?)"
        params = (operation, details, datetime.now().isoformat())
        self.execute_query(query, params)

class SQLiteMCPServer:
    """SQLite专用MCP服务器"""
    
    def __init__(self):
        self.server = Server("sqlite-mcp-server")
        self.db_path = Path(__file__).parent / "workspace" / "sqlite_mcp.db"
        self.db_manager = DatabaseManager(str(self.db_path))
        self.setup_handlers()
    
    def setup_handlers(self):
        """设置服务器处理器"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """返回可用的SQLite工具列表"""
            return [
                Tool(
                    name="create_user",
                    description="创建新用户记录",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "用户姓名"},
                            "email": {"type": "string", "description": "用户邮箱"}
                        },
                        "required": ["name", "email"]
                    }
                ),
                Tool(
                    name="list_users",
                    description="查询用户列表",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "description": "限制返回数量", "default": 10}
                        }
                    }
                ),
                Tool(
                    name="get_user_by_id",
                    description="根据ID获取用户信息",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "integer", "description": "用户ID"}
                        },
                        "required": ["user_id"]
                    }
                ),
                Tool(
                    name="create_task",
                    description="创建新任务记录",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "任务标题"},
                            "description": {"type": "string", "description": "任务描述"},
                            "user_id": {"type": "integer", "description": "负责用户ID"}
                        },
                        "required": ["title", "user_id"]
                    }
                ),
                Tool(
                    name="update_task_status",
                    description="更新任务状态",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "integer", "description": "任务ID"},
                            "status": {"type": "string", "description": "新状态 (pending/in_progress/completed/cancelled)"}
                        },
                        "required": ["task_id", "status"]
                    }
                ),
                Tool(
                    name="get_user_tasks",
                    description="获取指定用户的所有任务",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "integer", "description": "用户ID"}
                        },
                        "required": ["user_id"]
                    }
                ),
                Tool(
                    name="list_tasks",
                    description="查询任务列表",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "description": "按状态筛选 (可选)"},
                            "limit": {"type": "integer", "description": "限制返回数量", "default": 20}
                        }
                    }
                ),
                Tool(
                    name="execute_sql",
                    description="执行安全的SQL SELECT查询",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL查询语句（仅支持SELECT）"},
                            "description": {"type": "string", "description": "查询描述"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_table_schema",
                    description="获取数据库表结构",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {"type": "string", "description": "表名 (users/tasks/operation_logs)"}
                        },
                        "required": ["table_name"]
                    }
                ),
                Tool(
                    name="count_records",
                    description="统计表记录数量",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {"type": "string", "description": "表名"},
                            "condition": {"type": "string", "description": "WHERE条件 (可选)"}
                        },
                        "required": ["table_name"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """处理工具调用"""
            
            if name == "create_user":
                return await self._create_user(arguments)
            elif name == "list_users":
                return await self._list_users(arguments)
            elif name == "get_user_by_id":
                return await self._get_user_by_id(arguments)
            elif name == "create_task":
                return await self._create_task(arguments)
            elif name == "update_task_status":
                return await self._update_task_status(arguments)
            elif name == "get_user_tasks":
                return await self._get_user_tasks(arguments)
            elif name == "list_tasks":
                return await self._list_tasks(arguments)
            elif name == "execute_sql":
                return await self._execute_sql(arguments)
            elif name == "get_table_schema":
                return await self._get_table_schema(arguments)
            elif name == "count_records":
                return await self._count_records(arguments)
            else:
                return [TextContent(type="text", text=f"❌ 未知工具: {name}")]
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """返回可用的数据库资源"""
            return [
                Resource(
                    uri="sqlite://users",  # type: ignore
                    name="用户数据表",
                    description="用户信息数据",
                    mimeType="application/json"
                ),
                Resource(
                    uri="sqlite://tasks",  # type: ignore
                    name="任务数据表",
                    description="任务信息数据",
                    mimeType="application/json"
                ),
                Resource(
                    uri="sqlite://logs",  # type: ignore
                    name="操作日志表",
                    description="系统操作日志",
                    mimeType="application/json"
                ),
                Resource(
                    uri="sqlite://schema",  # type: ignore
                    name="数据库结构",
                    description="数据库表结构信息",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri) -> str:  # type: ignore
            """读取数据库资源内容"""
            uri_str = str(uri)
            
            if uri_str == "sqlite://users":
                users = self.db_manager.execute_query("SELECT * FROM users ORDER BY created_at DESC")
                return json.dumps(users, indent=2, ensure_ascii=False)
            
            elif uri_str == "sqlite://tasks":
                tasks = self.db_manager.execute_query("""
                    SELECT t.*, u.name as user_name 
                    FROM tasks t 
                    LEFT JOIN users u ON t.user_id = u.id
                    ORDER BY t.created_at DESC
                """)
                return json.dumps(tasks, indent=2, ensure_ascii=False)
            
            elif uri_str == "sqlite://logs":
                logs = self.db_manager.execute_query(
                    "SELECT * FROM operation_logs ORDER BY timestamp DESC LIMIT 100"
                )
                return json.dumps(logs, indent=2, ensure_ascii=False)
            
            elif uri_str == "sqlite://schema":
                # 获取数据库表结构
                schema_info = {}
                tables = ["users", "tasks", "operation_logs"]
                
                for table in tables:
                    schema = self.db_manager.execute_query(f"PRAGMA table_info({table})")
                    schema_info[table] = schema
                
                return json.dumps(schema_info, indent=2, ensure_ascii=False)
            
            else:
                return f"❌ 未知资源: {uri_str}"
    
    # SQLite工具实现方法
    async def _create_user(self, args: Dict[str, Any]) -> List[TextContent]:
        """创建用户"""
        try:
            name = args["name"]
            email = args["email"]
            
            # 检查邮箱是否已存在
            existing = self.db_manager.execute_query("SELECT id FROM users WHERE email = ?", (email,))
            if existing:
                return [TextContent(type="text", text=f"❌ 邮箱 {email} 已存在")]
            
            query = "INSERT INTO users (name, email, created_at) VALUES (?, ?, ?)"
            params = (name, email, datetime.now().isoformat())
            
            result = self.db_manager.execute_query(query, params)
            self.db_manager.log_operation("create_user", f"创建用户: {name} ({email})")
            
            return [TextContent(
                type="text",
                text=f"✅ 成功创建用户 '{name}' (邮箱: {email})"
            )]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ 创建用户失败: {str(e)}")]
    
    async def _list_users(self, args: Dict[str, Any]) -> List[TextContent]:
        """列出用户"""
        try:
            limit = args.get("limit", 10)
            query = "SELECT * FROM users ORDER BY created_at DESC LIMIT ?"
            
            users = self.db_manager.execute_query(query, (limit,))
            self.db_manager.log_operation("list_users", f"查询用户列表 (限制: {limit})")
            
            if not users:
                return [TextContent(type="text", text="❌ 暂无用户数据")]
            
            result_text = f"👥 用户列表 (共 {len(users)} 个):\n\n"
            for user in users:
                result_text += f"🆔 ID: {user['id']}\n"
                result_text += f"👤 姓名: {user['name']}\n"
                result_text += f"📧 邮箱: {user['email']}\n"
                result_text += f"🕐 创建时间: {user['created_at']}\n"
                result_text += "-" * 40 + "\n"
            
            return [TextContent(type="text", text=result_text)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ 查询用户失败: {str(e)}")]
    
    async def _get_user_by_id(self, args: Dict[str, Any]) -> List[TextContent]:
        """根据ID获取用户"""
        try:
            user_id = args["user_id"]
            query = "SELECT * FROM users WHERE id = ?"
            
            users = self.db_manager.execute_query(query, (user_id,))
            self.db_manager.log_operation("get_user_by_id", f"查询用户ID: {user_id}")
            
            if not users:
                return [TextContent(type="text", text=f"❌ 未找到ID为 {user_id} 的用户")]
            
            user = users[0]
            result_text = f"👤 用户详情:\n\n"
            result_text += f"🆔 ID: {user['id']}\n"
            result_text += f"👤 姓名: {user['name']}\n"
            result_text += f"📧 邮箱: {user['email']}\n"
            result_text += f"🕐 创建时间: {user['created_at']}\n"
            
            return [TextContent(type="text", text=result_text)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ 查询用户失败: {str(e)}")]
    
    async def _create_task(self, args: Dict[str, Any]) -> List[TextContent]:
        """创建任务"""
        try:
            title = args["title"]
            description = args.get("description", "")
            user_id = args["user_id"]
            
            # 检查用户是否存在
            user_check = self.db_manager.execute_query("SELECT name FROM users WHERE id = ?", (user_id,))
            if not user_check:
                return [TextContent(type="text", text=f"❌ 用户ID {user_id} 不存在")]
            
            query = """
                INSERT INTO tasks (title, description, user_id, created_at, updated_at) 
                VALUES (?, ?, ?, ?, ?)
            """
            now = datetime.now().isoformat()
            params = (title, description, user_id, now, now)
            
            self.db_manager.execute_query(query, params)
            self.db_manager.log_operation("create_task", f"创建任务: {title} (用户ID: {user_id})")
            
            return [TextContent(
                type="text",
                text=f"✅ 成功创建任务 '{title}' (分配给: {user_check[0]['name']})"
            )]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ 创建任务失败: {str(e)}")]
    
    async def _update_task_status(self, args: Dict[str, Any]) -> List[TextContent]:
        """更新任务状态"""
        try:
            task_id = args["task_id"]
            status = args["status"]
            
            valid_statuses = ["pending", "in_progress", "completed", "cancelled"]
            if status not in valid_statuses:
                return [TextContent(
                    type="text",
                    text=f"❌ 无效状态。有效状态: {', '.join(valid_statuses)}"
                )]
            
            # 检查任务是否存在
            task_check = self.db_manager.execute_query("SELECT title FROM tasks WHERE id = ?", (task_id,))
            if not task_check:
                return [TextContent(type="text", text=f"❌ 任务ID {task_id} 不存在")]
            
            query = "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?"
            params = (status, datetime.now().isoformat(), task_id)
            
            self.db_manager.execute_query(query, params)
            self.db_manager.log_operation("update_task_status", f"更新任务 {task_id} 状态为 {status}")
            
            status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "cancelled": "❌"}
            
            return [TextContent(
                type="text",
                text=f"✅ 任务 '{task_check[0]['title']}' 状态已更新为 {status_emoji.get(status, '📋')} {status}"
            )]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ 更新任务状态失败: {str(e)}")]
    
    async def _get_user_tasks(self, args: Dict[str, Any]) -> List[TextContent]:
        """获取用户任务"""
        try:
            user_id = args["user_id"]
            
            query = """
                SELECT t.*, u.name as user_name 
                FROM tasks t 
                JOIN users u ON t.user_id = u.id 
                WHERE t.user_id = ?
                ORDER BY t.created_at DESC
            """
            
            tasks = self.db_manager.execute_query(query, (user_id,))
            self.db_manager.log_operation("get_user_tasks", f"查询用户 {user_id} 的任务")
            
            if not tasks:
                return [TextContent(type="text", text=f"❌ 用户ID {user_id} 没有任务")]
            
            user_name = tasks[0]['user_name']
            result_text = f"📋 {user_name} 的任务 (共 {len(tasks)} 个):\n\n"
            
            status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "cancelled": "❌"}
            
            for task in tasks:
                result_text += f"🆔 ID: {task['id']}\n"
                result_text += f"📌 标题: {task['title']}\n"
                result_text += f"{status_emoji.get(task['status'], '📋')} 状态: {task['status']}\n"
                result_text += f"📝 描述: {task['description'] or '无'}\n"
                result_text += f"🕐 创建: {task['created_at']}\n"
                result_text += f"🔄 更新: {task['updated_at']}\n"
                result_text += "-" * 40 + "\n"
            
            return [TextContent(type="text", text=result_text)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ 查询用户任务失败: {str(e)}")]
    
    async def _list_tasks(self, args: Dict[str, Any]) -> List[TextContent]:
        """列出任务"""
        try:
            status_filter = args.get("status")
            limit = args.get("limit", 20)
            
            if status_filter:
                query = """
                    SELECT t.*, u.name as user_name 
                    FROM tasks t 
                    LEFT JOIN users u ON t.user_id = u.id 
                    WHERE t.status = ?
                    ORDER BY t.created_at DESC LIMIT ?
                """
                params = (status_filter, limit)
            else:
                query = """
                    SELECT t.*, u.name as user_name 
                    FROM tasks t 
                    LEFT JOIN users u ON t.user_id = u.id
                    ORDER BY t.created_at DESC LIMIT ?
                """
                params = (limit,)
            
            tasks = self.db_manager.execute_query(query, params)
            filter_desc = f" (状态: {status_filter})" if status_filter else ""
            self.db_manager.log_operation("list_tasks", f"查询任务列表{filter_desc}")
            
            if not tasks:
                return [TextContent(type="text", text=f"❌ 没有找到任务{filter_desc}")]
            
            result_text = f"📋 任务列表{filter_desc} (共 {len(tasks)} 个):\n\n"
            
            status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "cancelled": "❌"}
            
            for task in tasks:
                result_text += f"🆔 ID: {task['id']}\n"
                result_text += f"📌 标题: {task['title']}\n"
                result_text += f"{status_emoji.get(task['status'], '📋')} 状态: {task['status']}\n"
                result_text += f"👤 负责人: {task['user_name'] or '未分配'}\n"
                result_text += f"🕐 创建时间: {task['created_at']}\n"
                result_text += "-" * 40 + "\n"
            
            return [TextContent(type="text", text=result_text)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ 查询任务列表失败: {str(e)}")]
    
    async def _execute_sql(self, args: Dict[str, Any]) -> List[TextContent]:
        """执行SQL查询"""
        try:
            query = args["query"]
            description = args.get("description", "自定义查询")
            
            # 安全检查 - 只允许SELECT查询
            if not query.strip().upper().startswith('SELECT'):
                return [TextContent(type="text", text="🔒 安全限制：仅允许SELECT查询")]
            
            results = self.db_manager.execute_query(query)
            self.db_manager.log_operation("execute_sql", f"{description}: {query}")
            
            if not results:
                return [TextContent(type="text", text="📊 查询无结果")]
            
            result_text = f"📊 SQL查询结果 (共 {len(results)} 行):\n"
            result_text += f"📝 描述: {description}\n"
            result_text += f"🔍 查询: {query}\n\n"
            result_text += json.dumps(results, indent=2, ensure_ascii=False)
            
            return [TextContent(type="text", text=result_text)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ 执行查询失败: {str(e)}")]
    
    async def _get_table_schema(self, args: Dict[str, Any]) -> List[TextContent]:
        """获取表结构"""
        try:
            table_name = args["table_name"]
            valid_tables = ["users", "tasks", "operation_logs"]
            
            if table_name not in valid_tables:
                return [TextContent(
                    type="text",
                    text=f"❌ 无效表名。可用表: {', '.join(valid_tables)}"
                )]
            
            schema = self.db_manager.execute_query(f"PRAGMA table_info({table_name})")
            self.db_manager.log_operation("get_table_schema", f"获取表结构: {table_name}")
            
            result_text = f"🗃️ 表 '{table_name}' 结构:\n\n"
            for column in schema:
                result_text += f"📋 列名: {column['name']}\n"
                result_text += f"🔤 类型: {column['type']}\n"
                result_text += f"🔑 主键: {'是' if column['pk'] else '否'}\n"
                result_text += f"❗ 非空: {'是' if column['notnull'] else '否'}\n"
                result_text += f"🔢 默认值: {column['dflt_value'] or '无'}\n"
                result_text += "-" * 30 + "\n"
            
            return [TextContent(type="text", text=result_text)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ 获取表结构失败: {str(e)}")]
    
    async def _count_records(self, args: Dict[str, Any]) -> List[TextContent]:
        """统计记录数量"""
        try:
            table_name = args["table_name"]
            condition = args.get("condition", "")
            
            valid_tables = ["users", "tasks", "operation_logs"]
            if table_name not in valid_tables:
                return [TextContent(
                    type="text",
                    text=f"❌ 无效表名。可用表: {', '.join(valid_tables)}"
                )]
            
            if condition:
                query = f"SELECT COUNT(*) as count FROM {table_name} WHERE {condition}"
            else:
                query = f"SELECT COUNT(*) as count FROM {table_name}"
            
            result = self.db_manager.execute_query(query)
            count = result[0]['count']
            
            self.db_manager.log_operation("count_records", f"统计 {table_name} 记录数")
            
            condition_desc = f" (条件: {condition})" if condition else ""
            return [TextContent(
                type="text",
                text=f"📊 表 '{table_name}'{condition_desc} 共有 {count} 条记录"
            )]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ 统计记录失败: {str(e)}")]
    
    async def run(self):
        """运行MCP服务器"""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="sqlite-mcp-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )

def main():
    """主函数"""
    server = SQLiteMCPServer()
    asyncio.run(server.run())

if __name__ == "__main__":
    main()
