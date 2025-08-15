"""多实例协调器 (内存) + 工具转发 /call (中文文档)

设计目的
====================
在多个 IDA 实例并行时, 需要一种轻量方式发现彼此并统一转发工具调用。本模块提供一个**内存驻留**的小型 HTTP 服务, 仅占用本地 ``127.0.0.1:11337`` 端口。

角色与职责
--------------------
1. 第一个尝试注册的实例若发现 11337 空闲 → 直接绑定并成为协调器。
2. 其余实例仅向该协调器 POST /register 进行登记。
3. 无任何磁盘持久化; 退出后状态自动丢弃。

HTTP 接口
--------------------
* ``GET  /instances``  : 返回当前注册的全部实例列表 (数组)
* ``POST /register``   : 注册或刷新单个实例 (覆盖 pid 相同的旧记录)
* ``POST /deregister`` : 注销实例 (进程退出 / 插件关闭)
* ``POST /call``       : 将工具调用转发到指定实例 (通过 pid 或 port 识别)

实例结构字段示例
--------------------
```
{
    "pid": 1234,
    "port": 8765,
    "input_file": "/path/to/bin",
    "idb": "/path/to/db.i64",
    "started": 1730000000.123,   # 启动时间戳
    "python": "3.11.9"
}
```

转发机制 (/call)
--------------------
1. 客户端 (或代理) 提交: { tool, params, pid|port }
2. 协调器定位目标实例端口, 使用 fastmcp.Client 临时发起一次真实工具调用。
3. 对返回对象做 “可 JSON 序列化” 处理 (递归转普通结构) 后返回。

并发与线程
--------------------
* 采用 RLock 保护 _instances 列表。
* 协调器 HTTPServer 运行在守护线程, 不阻塞调用方。

扩展建议
--------------------
* 增加心跳(定期刷新时间戳) + 过期清理。
* 增加权限限制 (只允许本地请求 / 简单 token)。
* 支持广播调用 (例如对所有实例同步执行某工具)。

公开辅助函数
--------------------
* ``init_and_register`` : 保证协调器存在并注册当前实例。
* ``get_instances``     : 查询实例列表 (本地 or 远程)。
* ``deregister``        : 注销当前实例。
* ``call_tool``         : 调用 /call 进行一次转发。
"""
from __future__ import annotations
import threading
import json
import time
import socket
import http.server
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
import os
import atexit
import sys

COORD_HOST = "127.0.0.1"
COORD_PORT = 11337

_instances: List[Dict[str, Any]] = []
_lock = threading.RLock()
_is_coordinator = False
_server_thread: Optional[threading.Thread] = None
_self_pid = os.getpid()

class _Handler(http.server.BaseHTTPRequestHandler):  # pragma: no cover
    def log_message(self, format, *args):
        return
    def _send(self, code: int, obj: Any):
        data = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):  # type: ignore
        if self.path == '/instances':
            with _lock:
                self._send(200, _instances)
        else:
            self._send(404, {"error": "not found"})
    def do_POST(self):  # type: ignore
        length = int(self.headers.get('Content-Length', '0'))
        raw = self.rfile.read(length) if length else b''
        try:
            payload = json.loads(raw.decode('utf-8') or '{}')
        except Exception:
            payload = {}
        if self.path == '/register':
            needed = {'pid', 'port'}
            if not needed.issubset(payload):
                self._send(400, {"error": "missing fields"})
                return
            with _lock:
                pid = payload['pid']
                existing = [e for e in _instances if e.get('pid') != pid]
                _instances.clear()
                _instances.extend(existing)
                _instances.append(payload)
            self._send(200, {"status": "ok"})
        elif self.path == '/deregister':
            pid = payload.get('pid')
            if pid is None:
                self._send(400, {"error": "missing pid"})
                return
            with _lock:
                remaining = [e for e in _instances if e.get('pid') != pid]
                _instances.clear()
                _instances.extend(remaining)
            self._send(200, {"status": "ok"})
        elif self.path == '/call':
            # payload: { pid | port, tool, params }
            target_pid = payload.get('pid')
            target_port = payload.get('port')
            tool = payload.get('tool')
            params = payload.get('params') or {}
            if not tool:
                self._send(400, {"error": "missing tool"})
                return
            with _lock:
                target = None
                if target_pid is not None:
                    for e in _instances:
                        if e.get('pid') == target_pid:
                            target = e
                            break
                elif target_port is not None:
                    for e in _instances:
                        if e.get('port') == target_port:
                            target = e
                            break
            if target is None:
                self._send(404, {"error": "instance not found"})
                return
            port = target.get('port')
            if not isinstance(port, int):
                self._send(500, {"error": "bad target port"})
                return
            # Forward the tool call over SSE MCP (JSON-RPC) using fastmcp Client dynamically.
            try:
                from fastmcp import Client  # type: ignore
                import asyncio
                async def _do():
                    async with Client(f"http://127.0.0.1:{port}/mcp/") as c:  # type: ignore
                        resp = await c.call_tool(tool, params)
                        # Convert data into plain JSON serializable structures
                        def norm(x):
                            if isinstance(x, list):
                                return [norm(i) for i in x]
                            if isinstance(x, dict):
                                return {k: norm(v) for k, v in x.items()}
                            if hasattr(x, '__dict__'):
                                return norm(vars(x))
                            return x
                        return {"tool": tool, "data": norm(resp.data)}
                result = asyncio.run(_do())
                self._send(200, result)
            except Exception as e:  # pragma: no cover
                self._send(500, {"error": f"call failed: {e}"})
        else:
            self._send(404, {"error": "not found"})

def _start_coordinator():  # pragma: no cover
    global _server_thread
    if _server_thread and _server_thread.is_alive():
        return
    def run():
        try:
            httpd = http.server.HTTPServer((COORD_HOST, COORD_PORT), _Handler)
            httpd.serve_forever()
        except Exception:
            pass
    _server_thread = threading.Thread(target=run, name="IDA-MCP-Registry", daemon=True)
    _server_thread.start()

def _coordinator_alive() -> bool:
    try:
        with socket.create_connection((COORD_HOST, COORD_PORT), timeout=0.3):
            return True
    except OSError:
        return False

def init_and_register(port: int, input_file: str | None, idb_path: str | None):
    """确保协调器运行, 若不存在则当前进程抢占成为协调器, 然后注册本实例。

    参数:
        port: 当前实例监听的 MCP 端口
        input_file: 输入文件路径 (可能为 None)
        idb_path: IDB 路径 (可能为 None)
    逻辑:
        1. 尝试连接 11337; 若失败则尝试 bind -> 成为协调器并启动 HTTP 服务。
        2. 构造实例 payload 并 POST /register。
        3. 注册 atexit 钩子, 确保正常退出时自动注销。
    """
    global _is_coordinator
    if not _coordinator_alive():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((COORD_HOST, COORD_PORT))
            s.close()
            _is_coordinator = True
            _start_coordinator()
        except OSError:
            _is_coordinator = False
    payload = {
        'pid': _self_pid,
        'port': port,
        'input_file': input_file,
        'idb': idb_path,
        'started': time.time(),
        'python': sys.version.split()[0],
    }
    _post_json('/register', payload)
    atexit.register(deregister)

def _post_json(path: str, obj: Any):
    data = json.dumps(obj).encode('utf-8')
    req = urllib.request.Request(f'http://{COORD_HOST}:{COORD_PORT}{path}', data=data, method='POST', headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req, timeout=1)
    except Exception:
        pass

def get_instances() -> List[Dict[str, Any]]:
    if _is_coordinator:
        with _lock:
            return list(_instances)
    try:
        with urllib.request.urlopen(f'http://{COORD_HOST}:{COORD_PORT}/instances', timeout=1) as resp:  # type: ignore
            raw = resp.read()
            data = json.loads(raw.decode('utf-8') or '[]')
            if isinstance(data, list):
                return data
    except Exception:
        return []
    return []

def deregister():  # pragma: no cover
    _post_json('/deregister', {'pid': _self_pid})

def call_tool(pid: int | None = None, port: int | None = None, tool: str = '', params: dict | None = None) -> dict:
    body = json.dumps({"pid": pid, "port": port, "tool": tool, "params": params or {}}).encode('utf-8')
    req = urllib.request.Request(f'http://{COORD_HOST}:{COORD_PORT}/call', data=body, method='POST', headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # type: ignore
            raw = resp.read()
            return json.loads(raw.decode('utf-8') or '{}')
    except Exception as e:
        return {"error": str(e)}
