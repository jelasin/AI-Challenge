"""最小版 IDA FastMCP 服务器。

提供工具:
    * list_functions – 枚举当前 IDA 中的全部函数
    * instances      – 通过协调器 (端口 11337) 获取所有已注册 IDA MCP 实例

设计说明:
    * 运行于每个独立的 IDA 实例内部; SSE 端点由插件代码启动的 uvicorn 提供。
    * 所有 IDA API 调用通过 ida_kernwin.execute_sync 切换到 IDA 主线程, 避免线程不安全。
    * 保持极简; 需要更多逆向相关工具时可按需增量扩展。
"""
from __future__ import annotations

import os
from typing import Callable, Any, List

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore

from fastmcp import FastMCP
try:
    from ida_mcp import registry  # type: ignore
except Exception:  # pragma: no cover
    registry = None  # type: ignore

try:  # Only the IDA bits actually needed for list_functions
    import idaapi  # type: ignore
    import idautils  # type: ignore
    import ida_kernwin  # type: ignore
    import ida_funcs  # type: ignore
    HAVE_IDA = True
except Exception:  # pragma: no cover
    HAVE_IDA = False

# Default TCP port exported for plugin entry (kept for compatibility)
DEFAULT_PORT = 8765


def _run_in_ida(fn: Callable[[], Any]) -> Any:
    """在 IDA 主线程执行回调并返回结果。若未处于 IDA 环境 (测试态) 则直接执行。"""
    if not HAVE_IDA:
        return fn()

    result_box: dict[str, Any] = {}

    def wrapper():  # type: ignore
        try:
            result_box["value"] = fn()
        except Exception as e:  # pragma: no cover
            result_box["error"] = repr(e)
        return 0

    ida_kernwin.execute_sync(wrapper, ida_kernwin.MFF_READ)  # type: ignore
    if "error" in result_box:
        raise RuntimeError(result_box["error"])
    return result_box.get("value")


class FunctionItem(BaseModel):  # type: ignore
    """函数条目结构 (显式声明避免出现通用 Root() 包装)。"""
    name: str  # type: ignore
    start_ea: int  # type: ignore
    end_ea: int  # type: ignore


def create_mcp_server() -> FastMCP:
    name = os.getenv("IDA_MCP_NAME", "IDA-MCP")
    mcp = FastMCP(name=name, instructions="通过 MCP 工具访问 IDA 反汇编/分析数据。")

    @mcp.tool(description="列出函数 (返回 FunctionItem 列表)。")
    def list_functions() -> List[FunctionItem]:  # type: ignore
        def logic():
            out: list[FunctionItem] = []
            for ea in idautils.Functions():  # type: ignore
                f = ida_funcs.get_func(ea)  # type: ignore
                if not f:
                    continue
                name = idaapi.get_func_name(ea)  # type: ignore
                out.append(FunctionItem(name=name, start_ea=int(f.start_ea), end_ea=int(f.end_ea)))
            return out

        return _run_in_ida(logic)

    @mcp.tool(description="获取所有已注册的 IDA MCP 实例 (通过协调器, 若存在)。")
    def instances() -> list[dict]:  # type: ignore
        if registry is None:
            return []
        try:
            return registry.get_instances()  # type: ignore
        except Exception as e:  # pragma: no cover
            return [{"error": str(e)}]

    return mcp
