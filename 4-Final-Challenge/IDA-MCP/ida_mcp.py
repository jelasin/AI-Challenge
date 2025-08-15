"""IDA Pro MCP 插件 (SSE + 多实例协调器注册)  —— 中文文档

功能综述
====================
本插件为每个启动的 IDA 实例提供一个最小化 **FastMCP SSE** 服务, 暴露逆向分析能力给支持 MCP 的外部客户端。

核心特性:
    1. 启动/关闭采用“切换式”触发(再次运行插件即关闭)。
    2. 自动选择空闲端口 (从 8765 开始向上扫描), SSE 路径固定为 ``/mcp``。
    3. 首个成功启动的实例会在 ``127.0.0.1:11337`` 上创建一个 **内存型协调器(coordinator)**。
    4. 后续实例向协调器注册, 仅在内存维护实例列表, 不落盘 (避免文件锁 / 清理问题)。
    5. 工具最小化: 仅保留 ``list_functions`` 与 ``instances`` (实例列表)。
    6. 可配合独立进程型代理 ``ida_mcp_proxy.py`` 统一访问多个实例。

运行时架构
--------------------
``IDA 实例 (N 个)`` → 各自运行 uvicorn FastMCP (SSE) → 向协调器登记元信息(pid, port, input_file 等)。
``协调器`` 负责: 记录活跃实例; 接收代理或其他客户端的 /call 请求并转发至目标实例。

线程与生命周期
--------------------
* uvicorn 服务器在 **后台守护线程** 中运行, 便于主线程继续响应 IDA 事件。
* 关闭流程: 设置 ``_uv_server.should_exit = True`` → 等待线程退出 → 调用协调器注销。
* IDA 退出或插件终止时, 若仍在运行则自动停止并反注册。

端口选择策略
--------------------
* 若设置环境变量 ``IDA_MCP_PORT`` 且合法, 则使用该端口 (不再扫描)。
* 否则从 ``DEFAULT_PORT (=8765)`` 起向上扫描 (最大 50 次)。
* 允许多个 IDA 实例并行, 避免端口冲突。

环境变量 (可选)
--------------------
* ``IDA_MCP_PORT``: 指定固定端口。
* ``IDA_MCP_HOST``: 监听地址, 默认 ``127.0.0.1``。
* ``IDA_MCP_NAME``: MCP 服务名, 默认 ``IDA-MCP``。

主要内部变量
--------------------
* ``_server_thread``: 后台 uvicorn 线程对象。
* ``_uv_server``: uvicorn Server 实例 (用于发出停止信号)。
* ``_active_port``: 当前实例实际使用端口。
* ``_stop_lock``: 防止并发关闭竞争。

公共函数概览
--------------------
* ``start_server_async(host, port)``: 启动 MCP 服务器 (线程)。
* ``stop_server()``: 发送退出信号并等待线程结束, 注销协调器。
* ``is_running()``: 判断当前服务器线程是否存活。

扩展建议
--------------------
未来可在 ``ida_mcp/server.py`` 内增量添加更多工具 (反编译、交叉引用、数据段搜索等)。协调器 ``registry.py`` 已支持 /call 转发, 添加工具仅需在每个实例服务端注册, 代理端(可选)补一层转发包装。

使用方式
--------------------
1. 将本文件与 ``ida_mcp`` 目录复制到 IDA ``plugins/``。
2. 打开目标二进制, 分析完成后在菜单或快捷键中执行插件 (第一次执行 = 启动)。
3. 再次执行插件 = 停止并反注册。
4. 可启动多个 IDA 实例重复步骤 2, 通过协调器配合代理统一访问。

调试提示
--------------------
* 如果端口被占用, 会自动向上扫描; 如全部失败, 仍可能抛出绑定异常 (检查是否被防火墙或安全软件占用)。
* 服务器崩溃日志会打印堆栈; 若需更详细日志可将 uvicorn log_level 改为 info/debug。

本文件只包含逻辑入口与生命周期管理, 实际工具定义在 ``ida_mcp/server.py``。
"""

import threading
import os
import traceback
import socket

try:  # IDA imports (only available inside IDA)
    import idaapi  # type: ignore
    import ida_kernwin  # type: ignore
except Exception:  # pragma: no cover - outside IDA
    idaapi = None  # type: ignore
    ida_kernwin = None  # type: ignore

from ida_mcp.server import create_mcp_server, DEFAULT_PORT
from ida_mcp import registry

_server_thread: threading.Thread | None = None
_uv_server = None  # type: ignore
_stop_lock = threading.Lock()
_active_port: int | None = None


def _find_free_port(preferred: int, max_scan: int = 50) -> int:
    """端口扫描: 从 preferred 起向上尝试绑定, 返回第一个可用端口;
    若全部失败则返回 preferred (保底)。"""
    for i in range(max_scan):
        p = preferred + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
            return p
    return preferred


def _register_with_coordinator(port: int):
    """向协调器注册当前实例元信息。

    参数:
        port: 当前实例 FastMCP SSE 监听端口。
    说明:
        * 首个实例若发现协调器端口空闲会由 registry 内部启动协调器。
        * 注册内容包括: pid / port / 输入文件路径 / idb 路径 / Python 版本等。
    """
    if idaapi is None:
        return
    input_file = getattr(idaapi, 'get_input_file_path', lambda: None)()  # type: ignore
    idb_path = None
    if hasattr(idaapi, 'get_path'):
        try:
            idb_path = idaapi.get_path(idaapi.PATH_TYPE_IDB)  # type: ignore
        except Exception:
            idb_path = None
    registry.init_and_register(port, input_file, idb_path)


def is_running() -> bool:
    return _server_thread is not None and _server_thread.is_alive()


def stop_server():
    """停止服务器 (切换)。

    步骤:
        1. 设置 ``_uv_server.should_exit`` 触发 uvicorn 事件循环退出。
        2. join 后台线程 (最多 5 秒)。
        3. 若已注册协调器则执行注销。
    并发安全:
        使用 ``_stop_lock`` 以防多次同时调用。
    """
    global _uv_server, _server_thread
    with _stop_lock:
        if _uv_server is None:
            print("[IDA-MCP] Server not running.")
            return
        try:
            # Graceful shutdown
            _uv_server.should_exit = True  # type: ignore[attr-defined]
            print("[IDA-MCP] Shutdown signal sent.")
        except Exception as e:  # pragma: no cover
            print("[IDA-MCP] Failed to signal shutdown:", e)
        if _server_thread:
            _server_thread.join(timeout=5)
        global _active_port
        _server_thread = None
        _uv_server = None
        if _active_port is not None:
            registry.deregister()
        _active_port = None
        print("[IDA-MCP] Server stopped.")


def PLUGIN_ENTRY():  # IDA looks for this symbol
    return IDAMCPPlugin()


class IDAMCPPlugin(idaapi.plugin_t if idaapi else object):  # type: ignore
    flags = 0
    comment = "FastMCP SSE server for IDA"
    help = "Expose IDA features through Model Context Protocol"
    wanted_name = "IDA-MCP"
    wanted_hotkey = ""

    def init(self):  # type: ignore
        if idaapi is None:
            print("[IDA-MCP] Outside IDA environment; plugin inactive.")
            return idaapi.PLUGIN_SKIP if idaapi else 0
        # 不自动启动, 等待用户菜单/快捷方式显式触发。
        print("[IDA-MCP] Ready. ")
        return idaapi.PLUGIN_KEEP  # type: ignore

    def run(self, arg):  # type: ignore
        # 切换行为: 运行中 -> 停止; 否则启动。仅打印日志, 不弹出对话框。
        if not idaapi:
            print("[IDA-MCP] Not inside IDA.")
            return
        if is_running():
            print("[IDA-MCP] Stopping server (toggle)...")
            stop_server()
            return
        # 端口选择: 优先使用环境变量; 否则自动扫描以支持多实例。
        env_port = os.getenv("IDA_MCP_PORT")
        if env_port and env_port.isdigit():
            port = int(env_port)
        else:
            port = _find_free_port(DEFAULT_PORT)
        host = os.getenv("IDA_MCP_HOST", "127.0.0.1")
        print(f"[IDA-MCP] Starting SSE server http://{host}:{port}/mcp/ (toggle to stop).")
        start_server_async(host, port)

    def term(self):  # type: ignore
        print("[IDA-MCP] Plugin terminating.")
        if is_running():
            stop_server()


def start_server_async(host: str, port: int):
    """异步(线程)启动 uvicorn FastMCP 服务。

    设计要点:
        * 使用守护线程避免阻塞 IDA 主线程。
        * 通过保存 ``_uv_server`` 引用实现优雅关闭 (设置 should_exit)。
        * 线程启动后立即向协调器注册 (保持实例可发现性)。
    """
    global _server_thread, _uv_server
    if is_running():
        print("[IDA-MCP] Server already running.")
        return

    def worker():
        global _uv_server
        try:
            server = create_mcp_server()
            # 构建 ASGI 应用 (包含 SSE 端点), 挂载路径 '/mcp'
            app = server.http_app(path="/mcp")  # type: ignore[attr-defined]
            import uvicorn  # Local import to avoid overhead if never started
            # 使用 warning 日志级别并关闭 access log, 避免输出无意义的 CTRL+C 提示。
            config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
            _uv_server = uvicorn.Server(config)
            _uv_server.run()
        except Exception as e:  # pragma: no cover
            print("[IDA-MCP] Server crashed:", e)
            traceback.print_exc()
        finally:
            _uv_server = None
            print("[IDA-MCP] Server thread exit.")

    _server_thread = threading.Thread(target=worker, name="IDA-MCP-Server", daemon=True)
    _server_thread.start()
    # Record chosen port after thread start
    global _active_port
    _active_port = port
    _register_with_coordinator(port)

if __name__ == "__main__":  # 允许在非 IDA 环境下手动调试运行
    print("[IDA-MCP] Standalone mode: starting server.")
    start_server_async("127.0.0.1", DEFAULT_PORT)
    if _server_thread:
        _server_thread.join()
