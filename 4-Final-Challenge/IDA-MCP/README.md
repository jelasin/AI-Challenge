# IDA-MCP (FastMCP SSE + 多实例协调器)

最小但可扩展的 IDA Pro MCP 方案：

* 每个 IDA 实例启动一个 **SSE FastMCP** 服务器 (`/mcp`)
* 第一个实例占用 `127.0.0.1:11337` 作为 **协调器(coordinator)**，维持内存注册表并支持工具转发
* 后续实例自动注册到协调器；无需共享文件或手工配置端口
* 通过一个进程型 **代理 `ida_mcp_proxy.py`**（MCP 客户端可用 command/args 启动）统一访问 / 聚合各实例工具

## 当前工具

插件内置 (`server.py`):

* `list_functions` – 返回当前 IDA 数据库中全部函数 (name, start_ea, end_ea)
* `instances` – 返回所有已注册的 IDA 实例（来自协调器）

代理 (`ida_mcp_proxy.py`):

* `ping`
* `list_instances`
* `select_instance(port?)`
* `list_functions` (针对选中或自动选中实例；经协调器转发)

## 目录结构

```text
IDA-MCP/
  ida_mcp.py              # 插件入口：启动/停止 SSE server + 注册协调器
  ida_mcp/
    server.py             # FastMCP server 定义 (最小工具集)
    registry.py           # 协调器实现 / 多实例注册 & /call 转发
    __init__.py
  ida_mcp_proxy.py        # 进程型代理（附加 MCP server）
  mcp.json                # MCP 客户端配置 (含 proxy / sse)
  README.md
  requirements.txt        # fastmcp 依赖（若外部环境需要）
```

## 启动步骤

1. 将目录放入 IDA `plugins/` 或复制 `ida_mcp.py` + `ida_mcp` 文件夹。
2. 打开目标二进制，等待分析完成。
3. 菜单 / 快捷方式触发插件：首次启动会：
   * 选择空闲端口（从 8765 起）运行 SSE 服务 `http://127.0.0.1:<port>/mcp/`
   * 若 11337 空闲 → 启动协调器；否则向现有协调器注册
4. 再次触发插件 = 停止并注销实例。

## 代理使用

在 `mcp.json` 中选择 `ida-mcp-proxy`：

1. 启动若干 IDA 实例并开启插件。
2. 启动代理进程（由客户端自动执行)。
3. 调用 `list_instances` / `select_instance` / `list_functions`。

## Python 客户端示例（直连某一实例 SSE）

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8765/mcp/") as c:
        r = await c.call_tool("list_functions", {})
        print(len(r.data), "functions")

asyncio.run(main())
```

## 设计要点

| 关注点 | 说明 |
|--------|------|
| 线程安全 | 所有 IDA API 调用通过 `execute_sync` 在主线程执行 |
| 多实例发现 | 内存协调器 + HTTP (11337)，不写磁盘 |
| 端口冲突 | 自动向上扫描空闲端口 |
| 转发机制 | 协调器 `/call` 内部使用 fastmcp Client 发起工具调用并返回 JSON 序列化结果 |
| 最小化 | 只保留核心工具，便于逐步扩展 |

## 可扩展方向

* 添加更多原生工具：反编译、交叉引用、搜索、patch 等
* 在协调器中实现心跳/超时清理 stale 实例
* 在代理中增加通用 `call(tool, params, port)` 工具
* Resource / Streaming 支持（MCP resources 与 progress）
* 安全：白名单工具 / 认证 / 只读模式

## 依赖

仅需安装 `fastmcp`（如果当前 Python 环境未安装）：

```bash
python -m pip install fastmcp
```

## 问题排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 代理 `list_instances` 为空 | 插件未启动或协调器端口被占用 | 启动至少一个实例；确认 11337 未被防火墙阻断 |
| 工具调用超时 | 目标实例卡住/长时间分析 | 等待 IDA 完成分析或重启实例 |
| `call failed` 错误 | 实例下线但未注销 | 重新触发插件关闭、再开启使其重新注册 |

---
需要新增或修改的功能继续提出。欢迎增量扩展。
