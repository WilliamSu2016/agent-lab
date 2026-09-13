# 05 — Observability

本实验为 Agent 系统建立完整的 **Logs / Metrics / Traces** 三支柱可观测性，
使任何一次 request 事后都能被准确回答以下 10 个问题。

```
src/observability/
├── __init__.py    # 包说明 + 汇总导出
├── tracing.py      # ExecutionContext + Span 树 + Tracer（回答 1/2/3/4/7/8/9/10）
├── metrics.py      # MetricsRegistry（latency/token/failure-rate/retry/cost/success-rate，回答 5/6）
└── logging.py      # JsonFormatter + 结构化 JSON 日志（默认脱敏）

tests/test_observability.py   # 覆盖 10 个问题 + 7 个指标 + 强制字段传播 + 默认脱敏
```

## 0. 三支柱如何分别回答 10 个问题

| # | 问题 | 由谁回答 | 函数/方法 |
|---|---|---|---|
| 1 | 一个 Request 经过了哪些 Agent? | Tracing | `tracing.agents_visited(root)` |
| 2 | 调用了哪些 Tools? | Tracing | `tracing.tools_called(root)` |
| 3 | 每个 Tool 花了多久? | Tracing + Metrics | `tracing.tool_durations_ms(root)` / `MetricsRegistry.latency_stats(name, kind=TOOL)` |
| 4 | 每次 LLM call 消耗多少 tokens? | Tracing + Metrics | `tracing.llm_calls(root)` / `MetricsRegistry.token_usage_total()` |
| 5 | 哪个 Agent 最昂贵? | Metrics | `MetricsRegistry.most_expensive_agent()` |
| 6 | 哪个 Tool 最容易失败? | Metrics | `MetricsRegistry.least_reliable_tool()` |
| 7 | Agent loop 执行了多少次? | Tracing | `tracing.count_loop_iterations(root, agent_name)` |
| 8 | 为什么最终失败? | Tracing | `tracing.find_root_cause_failure(root)` |
| 9 | 用户是谁? | ExecutionContext | `ExecutionContext.user_id`（每个 Span/日志行都携带） |
| 10 | 使用了哪个 Agent version? | ExecutionContext | `ExecutionContext.agent_version`（每个 Span/日志行都携带） |

三支柱不是三套互相独立的埋点——**Metrics 是从 Trace 派生出来的**
(`MetricsRegistry.record_from_trace(root_span)`)，**Logs 是从当前绑定的
ExecutionContext 自动补全身份字段的**（`logging.current_execution_context()`）。
只有一处埋点入口（`tracer.span(...)`），却同时产生完整的 Trace 树、可聚合的
Metrics 记录、以及带正确身份字段的结构化日志。

## 1. 强制身份字段

`tracing.ExecutionContext`（不可变 dataclass）：

```python
@dataclass(frozen=True)
class ExecutionContext:
    request_id: str
    trace_id: str
    user_id: str
    session_id: str
    agent_version: str
    environment: str
```

- 通过 `new_execution_context(...)` 创建（`request_id`/`trace_id` 默认自动生成 UUID，
  但都可以显式传入，以便和上游 HTTP 请求头里的 trace id 拼接成一条更大的分布式 trace）。
- 通过 `bind_execution_context(ctx)`（`contextvars.ContextVar`）在一次请求的最外层绑定一次，
  之后同一线程/协程内的所有 `tracer.span(...)` 和所有 `logger.info(...)` 都会自动带上这 6 个字段，
  调用方不需要在每个调用点手工传递。
- 设计上和 `src/security/authorization.py::Identity` 同一个原则：**显式、不可变、绝不用全局可变状态**——
  见下方"已知限制"里关于 `contextvars` 跨线程的讨论。

## 2. Tracing：Span 树 = 一次 Request 的完整执行树

`tracing.Span`：

```python
@dataclass
class Span:
    span_id: str
    parent_span_id: str | None
    trace_id: str
    request_id: str
    kind: SpanKind          # WORKFLOW / AGENT / TOOL / LLM
    name: str
    start_time: float
    end_time: float | None
    status: str             # "ok" | "error"
    error: str | None
    attributes: dict[str, Any]
    children: list["Span"]
```

`Tracer.span(kind, name, *, context=None, parent=None, **attributes)` 是一个
context manager：

```python
with tracer.span(SpanKind.AGENT, "supervisor"):
    with tracer.span(SpanKind.TOOL, "search_web", retry_count=0):
        ...
    with tracer.span(SpanKind.LLM, "llm_call", agent_name="supervisor",
                      model="gpt-4", prompt_tokens=100, completion_tokens=50):
        ...
```

- 进入 `with` 块时记录 `start_time`；退出时记录 `end_time`；抛出异常时把
  `status` 置为 `"error"`、`error` 记录异常信息，然后**原样重新抛出**——
  永远不吞异常，只是顺带记录一份（和 `src/reliability/errors.py` 的
  "分类、不静默吞掉" 原则一致）。
- 父子关系默认通过 `contextvars` 隐式推断（"当前 span"），也支持
  `parent=` 显式传入，用于跨线程的并行 fan-out 场景（见下方已知限制）。
- 一次请求 = 一棵树，根节点是唯一没有 parent 的 Span（通常是一个
  `SpanKind.WORKFLOW`），`Tracer.get_trace(trace_id)` 取回整棵树。
- `jsonl_file_sink(path)` 把每一棵完整的树在结束时序列化成一行 JSON，
  追加写入文件——一行一条完整的嵌套树，事后可以只用 `grep`/`jq` 就还原出
  任意一次历史 request 的完整调用链，不需要依赖内存里的 `Tracer` 对象。

查询函数都是**纯函数**（输入一个 `root: Span`，返回一个值），因此既能在刚生成
的内存树上跑，也能在从 `jsonl_file_sink` 重新加载出来的历史记录上跑：
`agents_visited` / `tools_called` / `tool_durations_ms` / `llm_calls` /
`count_loop_iterations` / `find_root_cause_failure`。

`find_root_cause_failure` 的语义是"返回失败得最深的 span"——一个 Tool 抛出异常
后，通常会被外层 Agent span 捕获并重新包装/传播，如果只看"最外层失败的 span"，
看到的永远是最外层的 workflow/agent，而不是真正出错的那个操作；本实现按深度
排序，取最深的失败 span 作为根因。

## 3. Metrics：7 个指标 + 2 个衍生排名

`metrics.MetricsRegistry`：

| 需求指标 | 方法 | 说明 |
|---|---|---|
| latency | `latency_stats(name, kind=...)` | 返回 count/avg/p50/p95/p99/max（毫秒），无第三方依赖，纯 Python 实现的线性插值分位数 |
| token_usage | `token_usage_total(agent_name=None)` | 按 agent 过滤或全局汇总 prompt/completion/total tokens |
| tool_failure_rate | `tool_failure_rate(tool_name)` | 该 tool 失败调用数 / 总调用数 |
| agent_failure_rate | `agent_failure_rate(agent_name)` | 该 agent 失败运行数 / 总运行数 |
| retry_count | `retry_count(name)` | 该 tool/agent 累计重试次数（来自 Span 的 `retry_count` 属性） |
| cost | `total_cost_usd(agent_name=None)` | 见下方成本口径说明 |
| success_rate | `success_rate(kind=None, name=None)` | 全局或按 kind/name 过滤；无数据时返回 `1.0`（空场景下不应制造虚假的失败率） |

两个衍生排名直接回答 Q5/Q6：

- `most_expensive_agent()` — 按每个 agent 被归因的总成本（LLM 调用成本 + agent
  自身记录的成本）排序取最大。
- `least_reliable_tool()` — 按失败率排序，失败率相同时用调用次数打破平局
  （调用次数越多，这个失败率的可信度越高）。

`MetricsRegistry.record_from_trace(root_span)` 是从 Trace 派生 Metrics 的唯一入口：
遍历整棵 Span 树，把每个 TOOL span 转成 `ToolCallRecord`，每个 LLM span 转成
`LLMCallRecord`，每个 AGENT span 转成 `AgentRunRecord`。也支持
`record_tool_call`/`record_llm_call`/`record_agent_run` 直接喂入记录，用于不经过
完整 Tracing 就想单独测试/使用 Metrics 的场景。

**成本口径说明（重要，仅供实验演示）**：`metrics.DEFAULT_PRICE_PER_1K_TOKENS_USD`
是一张**示例性**的每千 token 单价表，不代表任何真实供应商当前定价——真实生产环境
的成本必须来自实际的计费协议，且按 provider/model 各不相同，不应把这张表当作权威
数据硬编码进决策逻辑。`estimate_cost_usd()` 也接受显式传入 `price_per_1k_tokens=`
覆盖默认值。

## 4. Logs：结构化 JSON + 默认脱敏

`logging.JsonFormatter`：

- 每一行日志都是一个 JSON 对象，始终包含 `timestamp`/`level`/`logger`/`message`
  以及 6 个强制身份字段（从当前绑定的 `ExecutionContext` 自动读取；未绑定时为
  `null`，例如进程刚启动、第一个请求到达之前）。
- **默认脱敏，且是格式化器（formatter）级别强制的，调用方无法绕过**：
  - `message` 本身会先过 `src/security/sanitization.py::redact_sensitive_data`
    （复用 Security 实验里同一份正则规则，而不是重新发明一套——同一类敏感数据，
    无论出现在 tool 返回值里还是日志行里，脱敏方式应当一致）。
  - 任何 `extra={...}` 传入的结构化字段也会被递归脱敏：既按**值**做正则匹配脱敏，
    也按**字段名**做脱敏（`SENSITIVE_FIELD_NAMES` 白名单，例如 `password`/
    `api_key`/`token`）——防止 `extra={"password": "hunter2"}` 这种值本身不匹配任何
    正则、但字段名一望而知敏感的信息被记录下来。这是纵深防御：值匹配防不住
    "看起来不像密钥的密钥"，字段名匹配防不住"混在长文本里的密钥"，两者互补。
- `configure_json_logging(logger_name=..., stream=...)` 是一个幂等的 setup
  helper（多次调用不会叠加重复 handler、不会重复打印同一行日志），默认写到
  `sys.stdout`——遵循"应用只管往 stdout 写，由容器平台/日志采集器决定落地到哪里"
  的生产惯例，而不是这个进程自己管理日志文件。

## 5. 已知限制（诚实边界）

- **`contextvars` 不会自动跨线程传播**：`tracer.span(...)` 默认通过
  `contextvars.ContextVar` 推断"当前 span"作为新 span 的 parent，这对本项目里
  顺序调用的场景（Supervisor -> Planner -> Worker -> Synthesizer）完全够用。但
  `contextvars` 只在 `asyncio`/`contextvars.copy_context()` 显式复制的场景下跨
  边界传播；一个直接用 `threading.Thread` 启动、没有走那套复制机制的线程（例如
  `src/reliability/timeout.py::run_with_timeout` 内部用的裸线程），看不到父线程
  的"当前 span"。对于真正的并行 fan-out（例如 `multi_agent_research` 里的
  research worker 并行执行），必须显式传入 `tracer.span(..., parent=parent_span)`，
  不能依赖 contextvar 自动推断——本模块的 API 同时支持隐式和显式两种方式，正是
  为了覆盖这个已知的传播边界。
- **`Tracer`/`MetricsRegistry` 都是纯内存实现，单进程、不持久化**：和本项目里
  `InMemoryIdempotencyStore`（`src/reliability/idempotency.py`）、`ApprovalStore`
  （`src/security/authorization.py`）是同一类文档化的已知限制——真实生产环境需要
  把 Trace/Metrics 推送到真正的后端（例如 OpenTelemetry Collector + Jaeger/Tempo
  存 Trace，Prometheus/StatsD 存 Metrics），本模块提供的是这些后端"背后应该有"的
  聚合/查询逻辑，且刻意保持独立、可单独测试，而不是直接绑定某一个具体后端 SDK。
- **成本估算是示例性的**，不是真实计费数据（见上文"成本口径说明"）。
- **本模块与已存在的 `src/tracing.py`（根目录）是两个独立模块**：后者是更早一个
  实验里专门给 OpenAI Agents SDK 用的、绑定 `agents.tracing` 的落盘导出器；本模块
  （`src/observability/tracing.py`）刻意做成框架无关，可以包裹 `multi_agent_research`/
  `durable`/`security` 里任何一个实验，不依赖任何具体 LLM/Agent 框架。两者不应混淆，
  也不应互相覆盖。

## 6. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_observability.py -q
```

覆盖：10 个必答问题各一个（或多个）测试、6 个强制身份字段的传播与绑定/重置、
默认脱敏（消息文本 + `extra` 字段值 + `extra` 字段名，含嵌套结构）、7 个指标各
一个测试、显式 `parent=` 的并行 fan-out 场景、`JsonFormatter` 的幂等 setup 与
直接使用。

另见 `docs/OBSERVABILITY-DASHBOARD-SPEC.md`：基于本模块数据源的
Production Agent Observability Dashboard Specification。
