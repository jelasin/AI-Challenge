"""IDA MCP 代理 (协调器客户端)  —— 中文文档

目的
====================
当外部 MCP 客户端 (如 IDE 插件 / LLM 工具) 只能通过“启动一个进程 + stdio/sse” 的形式接入时, 无法直接枚举多个 IDA 实例。本代理进程自身作为一个 FastMCP Server, 但内部并不执行逆向操作, 而是通过协调器 `/call` 将请求转发到目标 IDA 实例。

暴露工具
--------------------
    ping                   – 健康检查
    list_instances         – 获取当前所有已注册 IDA 实例 (由协调器返回)
    select_instance(port)  – 设置后续默认使用的实例端口 (若不指定自动选一个)
    list_functions         – 在当前选中实例上调用其 list_functions 工具

端口选择策略
--------------------
* 若未手动 select_instance, 自动优先选择 8765 (第一个常驻实例), 否则选择最早启动的实例。
* 切换实例只影响后续工具调用, 不影响协调器状态。

调用流程
--------------------
1. 客户端调用本代理的 tool (例如 list_functions)。
2. 代理确认/选择一个目标端口, 构造 body POST /call。
3. 协调器转发至对应 IDA 实例真正执行。
4. 返回的原始数据 (FunctionItem 列表等) 被协调器 JSON 化后再返回给客户端。

错误处理
--------------------
* 协调器不可达 / 超时: 返回 {"error": str(e)}。
* 没有实例: 返回 {"error": "No instances"}。
* 指定端口不存在: 返回 {"error": f"Port {port} not found"}。

可扩展点
--------------------
* 增加通用 forward(tool, params, port)
* 增加聚合/批量操作 (已根据需求删除 list_all_functions, 可随时恢复)
* 增加缓存/过滤/数据后处理

实现说明
--------------------
* 使用 urllib 标准库, 避免额外依赖。
* 超时严格 (GET 1 秒, CALL 5 秒) 防止阻塞。
* 内部维护 _current_port 作为默认目标。
"""
from __future__ import annotations
import json
import urllib.request
from typing import Optional, Dict, Any, List
from fastmcp import FastMCP

COORD_URL = "http://127.0.0.1:11337"
_current_port: Optional[int] = None

def _http_get(path: str) -> Any:
    try:
        with urllib.request.urlopen(COORD_URL + path, timeout=1) as r:  # type: ignore
            return json.loads(r.read().decode('utf-8') or 'null')
    except Exception:
        return None

def _http_post(path: str, obj: dict) -> Any:
    data = json.dumps(obj).encode('utf-8')
    req = urllib.request.Request(COORD_URL + path, data=data, method='POST', headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:  # type: ignore
            return json.loads(r.read().decode('utf-8') or 'null')
    except Exception as e:
        return {"error": str(e)}

def _instances() -> List[Dict[str, Any]]:
    data = _http_get('/instances')
    return data if isinstance(data, list) else []

def _choose_default_port() -> Optional[int]:
    inst = _instances()
    if not inst:
        return None
    for e in inst:
        if e.get('port') == 8765:
            return 8765
    inst_sorted = sorted(inst, key=lambda x: x.get('started', 0))
    return inst_sorted[0].get('port')

def _ensure_port() -> Optional[int]:
    global _current_port
    if _current_port and any(e.get('port') == _current_port for e in _instances()):
        return _current_port
    _current_port = _choose_default_port()
    return _current_port

def _call(tool: str, params: dict | None = None, port: int | None = None) -> Any:
    body = {"tool": tool, "params": params or {}}
    if port is not None:
        body['port'] = port
    elif _ensure_port() is not None:
        body['port'] = _ensure_port()
    return _http_post('/call', body)

server = FastMCP(name="IDA-MCP-Proxy", instructions="基于协调器的代理, 通过 /call 转发工具请求。")

@server.tool(description="健康检查 (返回 pong)")
def ping() -> str:  # type: ignore
    return "pong"

@server.tool(description="列出当前已注册的 IDA MCP 实例 (协调器视图)")
def list_instances() -> list[dict]:  # type: ignore
    return _instances()

@server.tool(description="选择后续调用所使用的实例端口 (不指定则自动选择)。")
def select_instance(port: int | None = None) -> dict:  # type: ignore
    global _current_port
    if port is None:
        port = _choose_default_port()
    if port is None:
        return {"error": "No instances"}
    if not any(e.get('port') == port for e in _instances()):
        return {"error": f"Port {port} not found"}
    _current_port = port
    return {"selected_port": port}

@server.tool(description="调用选中实例的 list_functions 工具 (经协调器转发)。")
def list_functions() -> Any:  # type: ignore
    p = _ensure_port()
    if p is None:
        return {"error": "No instances"}
    res = _call('list_functions', {}, port=p)
    return res.get('data') if isinstance(res, dict) else res

## 已移除 list_all_functions：按需求仅保留单实例查询，如需聚合可再恢复。

if __name__ == "__main__":
    # 直接运行: fastmcp 会自动选择 stdio/sse 传输方式 (默认 stdio)
    server.run()
