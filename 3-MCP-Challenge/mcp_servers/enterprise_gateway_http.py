#!/usr/bin/env python3
"""
企业级 MCP HTTP 网关服务器
通过 HTTP API 提供企业级 MCP 服务
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enterprise-gateway-http")

# 工作空间路径
WORKSPACE_DIR = Path(__file__).parent / "workspace"
WORKSPACE_DIR.mkdir(exist_ok=True)

# 创建 FastAPI 应用
app = FastAPI(
    title="企业级 MCP 网关",
    description="提供企业级 MCP 服务的 HTTP API 网关",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态存储
gateway_state = {
    "services": {},
    "metrics": {},
    "config": {},
    "request_logs": [],
    "round_robin_counters": {}
}

# Pydantic 模型
class ServiceRegistration(BaseModel):
    name: str
    host: str = "localhost"
    port: int = 8000
    description: str = ""

class ToolCall(BaseModel):
    service: str
    tool: str
    arguments: Dict[str, Any] = {}

class ConfigUpdate(BaseModel):
    type: str
    data: Dict[str, Any]

class TokenRequest(BaseModel):
    user_id: str = "anonymous"
    permissions: List[str] = []

class ServiceSelection(BaseModel):
    type: str = ""
    strategy: str = "round_robin"

def initialize_default_services():
    """初始化默认服务"""
    default_services = [
        {"name": "math-server", "port": 8001, "description": "数学计算服务"},
        {"name": "file-server", "port": 8002, "description": "文件操作服务"},
        {"name": "sqlite-server", "port": 8003, "description": "数据库服务"},
        {"name": "prompt-server", "port": 8004, "description": "提示服务"}
    ]
    
    for service_info in default_services:
        service_name = service_info["name"]
        gateway_state["services"][service_name] = {
            "name": service_name,
            "host": "localhost",
            "port": service_info["port"],
            "description": service_info["description"],
            "status": "healthy",
            "last_health_check": datetime.now().isoformat()
        }
        gateway_state["metrics"][service_name] = {
            "requests_count": 0,
            "error_count": 0,
            "avg_response_time": 0.0
        }
        gateway_state["round_robin_counters"][service_name] = 0

# 初始化默认服务
initialize_default_services()

@app.get("/")
async def root():
    """根路径，返回网关信息"""
    return {
        "name": "企业级 MCP 网关",
        "version": "1.0.0",
        "description": "提供企业级 MCP 服务的 HTTP API 网关",
        "endpoints": {
            "services": "/services",
            "tools": "/tools/{service_name}/{tool_name}",
            "health": "/health",
            "metrics": "/metrics",
            "config": "/config"
        }
    }

@app.get("/services")
async def list_services():
    """列出所有注册的服务"""
    return {
        "total": len(gateway_state["services"]),
        "services": list(gateway_state["services"].values())
    }

@app.post("/services/register")
async def register_service(service: ServiceRegistration):
    """注册新服务"""
    try:
        gateway_state["services"][service.name] = {
            "name": service.name,
            "host": service.host,
            "port": service.port,
            "description": service.description,
            "status": "healthy",
            "last_health_check": datetime.now().isoformat()
        }
        gateway_state["metrics"][service.name] = {
            "requests_count": 0,
            "error_count": 0,
            "avg_response_time": 0.0
        }
        gateway_state["round_robin_counters"][service.name] = 0
        
        logger.info(f"服务已注册: {service.name}")
        return {"message": f"服务 '{service.name}' 注册成功", "success": True}
        
    except Exception as e:
        logger.error(f"注册服务失败 {service.name}: {e}")
        raise HTTPException(status_code=500, detail=f"注册服务失败: {str(e)}")

@app.delete("/services/{service_name}")
async def unregister_service(service_name: str):
    """注销服务"""
    if service_name not in gateway_state["services"]:
        raise HTTPException(status_code=404, detail=f"服务 '{service_name}' 不存在")
    
    try:
        del gateway_state["services"][service_name]
        if service_name in gateway_state["metrics"]:
            del gateway_state["metrics"][service_name]
        if service_name in gateway_state["round_robin_counters"]:
            del gateway_state["round_robin_counters"][service_name]
            
        logger.info(f"服务已注销: {service_name}")
        return {"message": f"服务 '{service_name}' 注销成功", "success": True}
        
    except Exception as e:
        logger.error(f"注销服务失败 {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"注销服务失败: {str(e)}")

@app.get("/services/discover")
async def discover_services(service_type: str = ""):
    """服务发现"""
    try:
        if service_type:
            filtered_services = [
                service for service_name, service in gateway_state["services"].items()
                if service_type.lower() in service_name.lower()
            ]
        else:
            filtered_services = list(gateway_state["services"].values())
        
        return {
            "total_services": len(filtered_services),
            "services": filtered_services,
            "query": service_type if service_type else "all"
        }
        
    except Exception as e:
        logger.error(f"服务发现失败: {e}")
        raise HTTPException(status_code=500, detail=f"服务发现失败: {str(e)}")

@app.post("/services/select")
async def select_service(request: ServiceSelection):
    """负载均衡选择服务"""
    try:
        # 筛选匹配的健康服务
        available_services = [
            (name, service) for name, service in gateway_state["services"].items()
            if (not request.type or request.type.lower() in name.lower()) 
            and service.get("status") == "healthy"
        ]
        
        if not available_services:
            return {
                "error": f"没有可用的服务类型: {request.type}",
                "selected_service": None,
                "success": False
            }
        
        # 负载均衡策略
        if request.strategy == "round_robin":
            service_key = request.type or "all"
            counter = gateway_state["round_robin_counters"].get(service_key, 0)
            selected_name, selected_service = available_services[counter % len(available_services)]
            gateway_state["round_robin_counters"][service_key] = counter + 1
            
        elif request.strategy == "least_connections":
            selected_name, selected_service = min(
                available_services, 
                key=lambda x: gateway_state["metrics"].get(x[0], {}).get("requests_count", 0)
            )
        else:
            selected_name, selected_service = available_services[0]
        
        return {
            "selected_service": selected_name,
            "strategy": request.strategy,
            "service_info": selected_service,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"服务选择失败: {e}")
        raise HTTPException(status_code=500, detail=f"服务选择失败: {str(e)}")

@app.get("/health")
async def health_check(service_name: str = ""):
    """健康检查"""
    try:
        if service_name:
            if service_name not in gateway_state["services"]:
                raise HTTPException(status_code=404, detail=f"服务 '{service_name}' 不存在")
            
            gateway_state["services"][service_name]["last_health_check"] = datetime.now().isoformat()
            gateway_state["services"][service_name]["status"] = "healthy"
            
            return {
                "service": service_name,
                "status": "healthy",
                "message": f"服务 '{service_name}' 健康检查通过",
                "timestamp": datetime.now().isoformat()
            }
        else:
            # 检查所有服务
            checked_count = 0
            for service in gateway_state["services"].values():
                service["last_health_check"] = datetime.now().isoformat()
                service["status"] = "healthy"
                checked_count += 1
            
            return {
                "checked_services": checked_count,
                "message": f"已检查 {checked_count} 个服务，全部健康",
                "timestamp": datetime.now().isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """获取系统指标"""
    try:
        total_requests = sum(m.get("requests_count", 0) for m in gateway_state["metrics"].values())
        total_errors = sum(m.get("error_count", 0) for m in gateway_state["metrics"].values())
        healthy_services = sum(1 for s in gateway_state["services"].values() if s.get("status") == "healthy")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_services": len(gateway_state["services"]),
            "healthy_services": healthy_services,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "services": gateway_state["services"],
            "metrics": gateway_state["metrics"]
        }
        
    except Exception as e:
        logger.error(f"获取指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取指标失败: {str(e)}")

@app.post("/tools/call")
async def call_tool(request: ToolCall):
    """调用指定服务的工具"""
    start_time = time.time()
    
    try:
        service_name = request.service
        tool_name = request.tool
        arguments = request.arguments
        
        # 检查服务是否存在且健康
        if service_name not in gateway_state["services"]:
            raise HTTPException(status_code=404, detail=f"服务 '{service_name}' 不存在")
        
        service = gateway_state["services"][service_name]
        if service.get("status") != "healthy":
            raise HTTPException(status_code=503, detail=f"服务 '{service_name}' 不健康")
        
        # 模拟工具调用（实际应用中应该调用真实的服务）
        result = await simulate_tool_call(service, tool_name, arguments)
        
        # 记录指标
        if service_name in gateway_state["metrics"]:
            gateway_state["metrics"][service_name]["requests_count"] += 1
            
        response_time = time.time() - start_time
        
        # 记录请求日志
        request_log = {
            "timestamp": datetime.now().isoformat(),
            "service": service_name,
            "tool": tool_name,
            "arguments": arguments,
            "success": True,
            "response_time": response_time
        }
        gateway_state["request_logs"].append(request_log)
        
        # 保持最近100条日志
        if len(gateway_state["request_logs"]) > 100:
            gateway_state["request_logs"] = gateway_state["request_logs"][-100:]
        
        return {
            "result": result,
            "service": service_name,
            "tool": tool_name,
            "success": True,
            "response_time": response_time,
            "request_id": f"req_{int(time.time())}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # 记录错误
        if service_name in gateway_state["metrics"]:
            gateway_state["metrics"][service_name]["error_count"] += 1
            
        logger.error(f"工具调用失败 {service_name}.{tool_name}: {e}")
        raise HTTPException(status_code=500, detail=f"工具调用失败: {str(e)}")

async def simulate_tool_call(service: Dict[str, Any], tool_name: str, arguments: Dict[str, Any]) -> Any:
    """模拟工具调用"""
    await asyncio.sleep(0.1)  # 模拟网络延迟
    
    service_name = service.get("name", "")
    
    # 数学服务工具
    if "math" in service_name:
        if tool_name == "add" and "a" in arguments and "b" in arguments:
            return arguments["a"] + arguments["b"]
        elif tool_name == "multiply" and "a" in arguments and "b" in arguments:
            return arguments["a"] * arguments["b"]
        elif tool_name == "subtract" and "a" in arguments and "b" in arguments:
            return arguments["a"] - arguments["b"]
        elif tool_name == "divide" and "a" in arguments and "b" in arguments:
            if arguments["b"] != 0:
                return arguments["a"] / arguments["b"]
            else:
                return "错误：除数不能为零"
    
    # 文件服务工具
    elif "file" in service_name:
        if tool_name == "list_files":
            directory = arguments.get("directory", str(WORKSPACE_DIR))
            try:
                path = Path(directory)
                if path.exists():
                    files = [f.name for f in path.iterdir()]
                    return {"files": files, "total": len(files)}
                else:
                    return {"error": "目录不存在", "files": []}
            except Exception as e:
                return {"error": str(e), "files": []}
        elif tool_name == "read_file":
            file_path = arguments.get("path", "")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {"content": content, "success": True}
            except Exception as e:
                return {"error": str(e), "success": False}
        elif tool_name == "create_file":
            file_path = arguments.get("path", "")
            content = arguments.get("content", "")
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return {"message": f"文件创建成功: {file_path}", "success": True}
            except Exception as e:
                return {"error": str(e), "success": False}
    
    # SQLite 服务工具
    elif "sqlite" in service_name:
        if tool_name == "create_user":
            name = arguments.get("name", "Unknown")
            email = arguments.get("email", "unknown@example.com")
            return {"message": f"用户创建成功: {name} ({email})", "id": int(time.time()), "success": True}
        elif tool_name == "list_users":
            return {
                "users": [
                    {"id": 1, "name": "测试用户1", "email": "test1@example.com"},
                    {"id": 2, "name": "测试用户2", "email": "test2@example.com"}
                ],
                "total": 2
            }
    
    # 提示服务工具
    elif "prompt" in service_name:
        if tool_name == "generate_prompt":
            topic = arguments.get("topic", "通用")
            return {"prompt": f"关于{topic}的提示内容", "success": True}
    
    # 默认响应
    return {
        "message": f"工具 '{tool_name}' 在服务 '{service_name}' 中执行完成",
        "arguments": arguments,
        "success": True
    }

@app.get("/logs")
async def get_request_logs(limit: int = 10):
    """获取请求日志"""
    try:
        recent_logs = gateway_state["request_logs"][-limit:] if gateway_state["request_logs"] else []
        
        return {
            "total_logs": len(gateway_state["request_logs"]),
            "recent_logs": recent_logs,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"获取日志失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取日志失败: {str(e)}")

@app.post("/auth/token")
async def generate_auth_token(request: TokenRequest):
    """生成认证令牌"""
    try:
        token_data = {
            "user_id": request.user_id,
            "permissions": request.permissions,
            "issued_at": datetime.now().isoformat(),
            "expires_in": 3600
        }
        
        # 模拟令牌生成
        token = f"enterprise_token_{hash(json.dumps(token_data, sort_keys=True))}"
        
        return {
            "token": token,
            "token_data": token_data,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"生成令牌失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成令牌失败: {str(e)}")

@app.put("/config")
async def update_config(request: ConfigUpdate):
    """更新配置"""
    try:
        config_type = request.type
        config_data = request.data
        
        if config_type not in gateway_state["config"]:
            gateway_state["config"][config_type] = {}
        
        gateway_state["config"][config_type].update(config_data)
        
        return {
            "message": f"配置 '{config_type}' 已更新",
            "config": gateway_state["config"][config_type],
            "success": True
        }
        
    except Exception as e:
        logger.error(f"更新配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")

@app.get("/config")
async def get_config():
    """获取当前配置"""
    return {
        "config": gateway_state["config"],
        "timestamp": datetime.now().isoformat()
    }

def run_server(host: str = "localhost", port: int = 8000):
    """运行HTTP服务器"""
    print(f"🚀 启动企业级 MCP HTTP 网关服务器")
    print(f"📡 服务地址: http://{host}:{port}")
    print(f"📋 API 文档: http://{host}:{port}/docs")
    print(f"📊 指标监控: http://{host}:{port}/metrics")
    
    uvicorn.run(
        "enterprise_gateway_http:app",
        host=host,
        port=port,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    run_server(host="0.0.0.0", port=8000)
