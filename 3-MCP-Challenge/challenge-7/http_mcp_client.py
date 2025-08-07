#!/usr/bin/env python3
"""
企业级 MCP HTTP 客户端
通过 HTTP API 调用 MCP 服务
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
import aiohttp
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-http-client")


class MCPHttpClient:
    """MCP HTTP 客户端"""
    
    def __init__(self, gateway_url: str = "http://localhost:8000"):
        self.gateway_url = gateway_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """通用请求方法"""
        if not self.session:
            raise RuntimeError("客户端未初始化，请使用 async with 语句")
        
        url = f"{self.gateway_url}{path}"
        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.content_type == 'application/json':
                    return await response.json()
                else:
                    text = await response.text()
                    return {"text": text, "status": response.status}
                    
        except Exception as e:
            logger.error(f"请求失败 {method} {url}: {e}")
            return {"error": str(e), "success": False}
    
    async def get_gateway_info(self) -> Dict[str, Any]:
        """获取网关信息"""
        return await self._request("GET", "/")
    
    async def list_services(self) -> Dict[str, Any]:
        """列出所有服务"""
        return await self._request("GET", "/services")
    
    async def register_service(self, name: str, host: str = "localhost", 
                             port: int = 8000, description: str = "") -> Dict[str, Any]:
        """注册服务"""
        data = {
            "name": name,
            "host": host,
            "port": port,
            "description": description
        }
        return await self._request("POST", "/services/register", json=data)
    
    async def unregister_service(self, service_name: str) -> Dict[str, Any]:
        """注销服务"""
        return await self._request("DELETE", f"/services/{service_name}")
    
    async def discover_services(self, service_type: str = "") -> Dict[str, Any]:
        """服务发现"""
        params = {"service_type": service_type} if service_type else {}
        return await self._request("GET", "/services/discover", params=params)
    
    async def select_service(self, service_type: str = "", 
                           strategy: str = "round_robin") -> Dict[str, Any]:
        """选择服务"""
        data = {"type": service_type, "strategy": strategy}
        return await self._request("POST", "/services/select", json=data)
    
    async def health_check(self, service_name: str = "") -> Dict[str, Any]:
        """健康检查"""
        params = {"service_name": service_name} if service_name else {}
        return await self._request("GET", "/health", params=params)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        return await self._request("GET", "/metrics")
    
    async def call_tool(self, service: str, tool: str, 
                       arguments: Dict[str, Any] = {}) -> Dict[str, Any]:
        """调用工具"""
        data = {
            "service": service,
            "tool": tool,
            "arguments": arguments
        }
        return await self._request("POST", "/tools/call", json=data)
    
    async def get_logs(self, limit: int = 10) -> Dict[str, Any]:
        """获取请求日志"""
        params = {"limit": limit}
        return await self._request("GET", "/logs", params=params)
    
    async def generate_token(self, user_id: str = "anonymous", 
                           permissions: List[str] = []) -> Dict[str, Any]:
        """生成认证令牌"""
        data = {"user_id": user_id, "permissions": permissions}
        return await self._request("POST", "/auth/token", json=data)
    
    async def update_config(self, config_type: str, 
                          config_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新配置"""
        data = {"type": config_type, "data": config_data}
        return await self._request("PUT", "/config", json=data)
    
    async def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return await self._request("GET", "/config")


class EnterpriseHttpDemo:
    """企业级 HTTP MCP 演示"""
    
    def __init__(self, gateway_url: str = "http://localhost:8000"):
        self.gateway_url = gateway_url
        # 指向 mcp_servers/workspace 目录
        self.workspace_dir = Path(__file__).parent.parent / "mcp_servers" / "workspace"
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_demo(self):
        """运行演示"""
        print("🏢 企业级 MCP HTTP 架构演示")
        print("=" * 50)
        
        async with MCPHttpClient(self.gateway_url) as client:
            await self._test_gateway_connection(client)
            await self._demonstrate_service_management(client)
            await self._demonstrate_load_balancing(client)
            await self._demonstrate_tool_calling(client)
            await self._demonstrate_health_monitoring(client)
            await self._demonstrate_enterprise_features(client)
            await self._demonstrate_workspace_management(client)
    
    async def _test_gateway_connection(self, client: MCPHttpClient):
        """测试网关连接"""
        print("\n🔌 测试网关连接")
        print("-" * 30)
        
        try:
            info = await client.get_gateway_info()
            if "error" not in info:
                print(f"✅ 网关连接成功: {info.get('name', 'Unknown')}")
                print(f"📋 版本: {info.get('version', 'Unknown')}")
                print(f"📡 可用端点: {len(info.get('endpoints', {}))}")
            else:
                print(f"❌ 网关连接失败: {info.get('error')}")
                return False
                
        except Exception as e:
            print(f"❌ 网关连接异常: {e}")
            return False
        
        return True
    
    async def _demonstrate_service_management(self, client: MCPHttpClient):
        """演示服务管理"""
        print("\n📦 服务管理演示")
        print("-" * 30)
        
        # 列出当前服务
        services_result = await client.list_services()
        if "error" not in services_result:
            total = services_result.get("total", 0)
            print(f"📊 当前注册服务: {total} 个")
            
            services = services_result.get("services", [])
            for service in services[:3]:  # 显示前3个服务
                name = service.get("name", "Unknown")
                port = service.get("port", "Unknown")
                status = service.get("status", "Unknown")
                print(f"  🔹 {name} (端口 {port}, 状态: {status})")
            
            if len(services) > 3:
                print(f"  ... 还有 {len(services) - 3} 个服务")
        else:
            print(f"❌ 获取服务列表失败: {services_result.get('error')}")
        
        # 注册新服务演示
        register_result = await client.register_service(
            name="demo-service",
            port=8888,
            description="演示服务"
        )
        if register_result.get("success"):
            print(f"✅ 服务注册成功: {register_result.get('message')}")
        else:
            print(f"❌ 服务注册失败: {register_result.get('error')}")
    
    async def _demonstrate_load_balancing(self, client: MCPHttpClient):
        """演示负载均衡"""
        print("\n⚖️ 负载均衡演示")
        print("-" * 30)
        
        # 更新负载均衡配置
        config_result = await client.update_config(
            "load_balancer",
            {"strategy": "round_robin", "health_check_interval": 15}
        )
        if config_result.get("success"):
            print(f"⚙️ 负载均衡配置更新成功")
        
        # 演示服务选择
        for i in range(3):
            selection_result = await client.select_service("server", "round_robin")
            if selection_result.get("success"):
                selected = selection_result.get("selected_service", "Unknown")
                strategy = selection_result.get("strategy", "Unknown")
                print(f"🎯 请求 {i+1}: 选择服务 '{selected}' (策略: {strategy})")
            await asyncio.sleep(0.3)
    
    async def _demonstrate_tool_calling(self, client: MCPHttpClient):
        """演示工具调用"""
        print("\n🔧 工具调用演示")
        print("-" * 30)
        
        # 数学计算工具
        math_result = await client.call_tool("math-server", "add", {"a": 150, "b": 250})
        if math_result.get("success"):
            result_value = math_result.get("result")
            response_time = math_result.get("response_time", 0)
            print(f"🔢 数学计算: 150 + 250 = {result_value} ({response_time:.3f}s)")
        else:
            print(f"❌ 数学计算失败: {math_result.get('error')}")
        
        # 文件操作工具
        file_result = await client.call_tool(
            "file-server", 
            "list_files", 
            {"directory": str(self.workspace_dir)}
        )
        if file_result.get("success"):
            files_info = file_result.get("result", {})
            if isinstance(files_info, dict):
                file_count = files_info.get("total", 0)
                print(f"📁 文件列表: 发现 {file_count} 个文件")
            else:
                print(f"📁 文件列表: {files_info}")
        else:
            print(f"❌ 文件操作失败: {file_result.get('error')}")
        
        # 数据库操作工具
        db_result = await client.call_tool(
            "sqlite-server",
            "create_user",
            {"name": "HTTP演示用户", "email": "http.demo@company.com"}
        )
        if db_result.get("success"):
            user_info = db_result.get("result", {})
            if isinstance(user_info, dict):
                message = user_info.get("message", "用户创建成功")
                print(f"👤 数据库操作: {message}")
            else:
                print(f"👤 数据库操作: {user_info}")
        else:
            print(f"❌ 数据库操作失败: {db_result.get('error')}")
    
    async def _demonstrate_health_monitoring(self, client: MCPHttpClient):
        """演示健康监控"""
        print("\n💗 健康监控演示")
        print("-" * 30)
        
        # 全局健康检查
        health_result = await client.health_check()
        if "error" not in health_result:
            checked = health_result.get("checked_services", 0)
            message = health_result.get("message", "")
            print(f"🔍 全局健康检查: {message}")
        else:
            print(f"❌ 健康检查失败: {health_result.get('error')}")
        
        # 获取系统指标
        metrics_result = await client.get_metrics()
        if "error" not in metrics_result:
            total_services = metrics_result.get("total_services", 0)
            healthy_services = metrics_result.get("healthy_services", 0)
            total_requests = metrics_result.get("total_requests", 0)
            error_rate = metrics_result.get("error_rate", 0)
            
            print(f"📊 系统指标:")
            print(f"  - 服务总数: {total_services}")
            print(f"  - 健康服务: {healthy_services}")
            print(f"  - 总请求数: {total_requests}")
            print(f"  - 错误率: {error_rate:.2%}")
        else:
            print(f"❌ 获取指标失败: {metrics_result.get('error')}")
    
    async def _demonstrate_enterprise_features(self, client: MCPHttpClient):
        """演示企业级功能"""
        print("\n🏢 企业级功能演示")
        print("-" * 30)
        
        # 生成认证令牌
        token_result = await client.generate_token(
            "http_admin",
            ["gateway:manage", "services:monitor", "tools:execute"]
        )
        if token_result.get("success"):
            token = token_result.get("token", "")
            print(f"🔐 认证令牌生成成功: {token[:30]}...")
        else:
            print(f"❌ 令牌生成失败: {token_result.get('error')}")
        
        # 服务发现
        discovery_result = await client.discover_services("math")
        if "error" not in discovery_result:
            total = discovery_result.get("total_services", 0)
            query = discovery_result.get("query", "")
            print(f"🔍 服务发现: 查询 '{query}' 找到 {total} 个匹配服务")
        else:
            print(f"❌ 服务发现失败: {discovery_result.get('error')}")
        
        # 获取请求日志
        logs_result = await client.get_logs(5)
        if "error" not in logs_result:
            total_logs = logs_result.get("total_logs", 0)
            recent_logs = logs_result.get("recent_logs", [])
            print(f"📝 请求日志: 总计 {total_logs} 条，最近 {len(recent_logs)} 条")
            
            for log in recent_logs[-2:]:  # 显示最近2条
                timestamp = log.get("timestamp", "Unknown")[:19]
                service = log.get("service", "Unknown")
                tool = log.get("tool", "Unknown")
                success = "✅" if log.get("success") else "❌"
                print(f"  🕒 [{timestamp}] {service}.{tool} {success}")
        else:
            print(f"❌ 获取日志失败: {logs_result.get('error')}")
    
    async def _demonstrate_workspace_management(self, client: MCPHttpClient):
        """演示工作空间管理"""
        print("\n📂 工作空间管理演示")
        print("-" * 30)
        
        # 创建演示文件
        demo_file = self.workspace_dir / "http_architecture_demo.txt"
        with open(demo_file, 'w', encoding='utf-8') as f:
            f.write(f"企业级 MCP HTTP 架构演示\n")
            f.write(f"演示时间: {asyncio.get_event_loop().time()}\n")
            f.write(f"网关地址: {self.gateway_url}\n")
            f.write(f"工作空间: {self.workspace_dir}\n")
            f.write(f"架构类型: HTTP API 网关\n")
        
        print(f"✅ 创建演示文件: {demo_file.name}")
        
        # 通过文件服务创建文件
        create_result = await client.call_tool(
            "file-server",
            "create_file",
            {
                "path": str(self.workspace_dir / "http_created_file.txt"),
                "content": "通过 HTTP API 创建的文件内容"
            }
        )
        if create_result.get("success"):
            result_info = create_result.get("result", {})
            if isinstance(result_info, dict):
                message = result_info.get("message", "文件创建成功")
                print(f"📄 HTTP文件创建: {message}")
            else:
                print(f"📄 HTTP文件创建: {result_info}")
        
        # 列出工作空间文件
        if self.workspace_dir.exists():
            files = list(self.workspace_dir.iterdir())
            print(f"📋 工作空间文件列表 ({len(files)} 个文件):")
            for file_path in files:
                file_type = "📁" if file_path.is_dir() else "📄"
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    print(f"  {file_type} {file_path.name} ({file_size} bytes)")
                else:
                    print(f"  {file_type} {file_path.name}")
        
        print(f"\n📍 HTTP API 网关: {self.gateway_url}")
        print(f"📍 工作空间: {self.workspace_dir}")
        print(f"📍 API文档: {self.gateway_url}/docs")


async def demo_http_architecture():
    """HTTP架构演示函数"""
    demo = EnterpriseHttpDemo()
    await demo.run_demo()


async def main():
    """主函数"""
    print("启动 HTTP 架构演示...")
    await demo_http_architecture()


if __name__ == "__main__":
    asyncio.run(main())
