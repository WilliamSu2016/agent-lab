# 09 — Failure Injection / Chaos Testing

本实验**不新增任何业务功能**。它只做一件事：针对本项目已经实现的失败处理机制
（`src/reliability/`、`src/specialists/llm.py`、`src/multi_agent_research/`、
`src/durable/`、`src/api/`）主动注入 12 种失败场景，并对"这些机制在失败发生时
到底做了什么"做出可验证、可重复的断言。凡是发现的真实缺口（例如 Planner 的
"invalid structured output"），本文档如实记录为**已知缺口**，而不是顺手把它修好
——修复属于新的业务功能变更，超出本实验范围。

```
tests/failure_injection/
├── __init__.py
├── conftest.py               # 共享 Settings 构造器 + openai 异常构造辅助函数
├── test_llm_failure.py        # #1 LLM timeout / #2 LLM 5xx / #7 Rate limit / #12 Invalid structured output
├── test_tool_failure.py       # #3 Tool timeout / #4 Tool 5xx / #5 Tool invalid response
├── test_mcp_failure.py        # #6 MCP unavailable
├── test_worker_crash.py       # #8 Worker crash / #9 Checkpoint failure
├── test_duplicate_request.py  # #10 Duplicate request
└── test_concurrency.py        # #11 Concurrent requests
```

与已存在、且与本实验无关的 `tests/test_failure_injection.py`（更早的 Reliability
实验遗留文件）完全独立，未做任何修改。

## 0. 设计原则：复用，不重新发明

每一个注入点都直接打到**真实的生产代码路径**，而不是重新实现一份 retry/timeout
逻辑去自我验证：

* LLM 失败：`monkeypatch.setattr(openai.resources.chat.completions.completions.Completions, "create", ...)`，
  让 `src.specialists.llm.build_openai_text_llm_call` 的真实调用链（`run_with_timeout`
  套 `retry_call`）完整跑一遍。
* Worker crash / Checkpoint failure：直接复用 `src/durable/recovery.py` 的
  `CrashInjector`/`run_or_crash`/`resume`，以及对 `SqliteSaver.put` 的包装。
* Duplicate request：直接复用 `src/reliability/idempotency.py` 的
  `InMemoryIdempotencyStore`/`idempotent()` 参考实现（`send_email`/`create_order`/
  `publish_report`），以及 `src/durable/graph.py` 里已经用 `idempotent()` 包装过的
  研究节点。
* Concurrent requests：直接对 `src.api.rate_limit.RateLimiter`、
  `src.reliability.idempotency` 和完整的 `src.api.app` FastAPI 应用发起真实的多
  线程并发调用（`concurrent.futures.ThreadPoolExecutor` + `threading.Barrier`
  让所有线程真正同时起跑），而不是顺序模拟。
* MCP unavailable：本仓库**没有任何真实 MCP 集成**，如实声明，并把"MCP 不可用"
  建模成与"LLM 连不上"完全相同的网络失败（`NETWORK` 分类、可重试），走同一套
  pipeline。

## 1. 12 种注入失败 × Expected Behavior

### #1 LLM timeout

```
持续超时 -> classify(TIMEOUT, retryable) -> retry (backoff) -> 达到 max_retries
        -> ToolInvocationError(category=timeout)   [从不裸抛/挂起]
偶发超时（某次重试之内恢复） -> 正常返回结果
```
验证于：`test_llm_failure.py::TestLLMTimeout`。真实机制：
`src.reliability.timeout.run_with_timeout`（daemon 线程 + `ToolTimeoutError`）
套在 `src.reliability.retry.retry_call` 里。

### #2 LLM 5xx

```
持续 5xx -> classify(SERVER_ERROR, retryable) -> retry (backoff) -> 耗尽重试
        -> ToolInvocationError(category=server_error)
偶发 5xx -> 在重试预算内恢复
非 5xx 的 4xx（如 400） -> classify(INVALID_PARAMETERS, non-retryable)
        -> 第一次尝试就立刻失败，不做无意义重试
```
验证于：`test_llm_failure.py::TestLLM5xx`。

### #3 Tool timeout / #4 Tool 5xx

本项目唯一真实的外部 HTTP 调用就是 #1/#2 覆盖的 LLM 调用。`retry_call`/
`run_with_timeout` 被文档明确写成"usable by *any* Agent Tool call"，所以
`test_tool_failure.py` 用一个通用（非 LLM）的工具调用形状复用同一套 pipeline，
验证其对任意未来外部工具调用同样成立，行为与 #1/#2 完全一致：
```
retry -> backoff -> retry limit -> ToolInvocationError
```

### #5 Tool invalid response

```
调用传输层成功，但响应内容不可用（缺字段/形状不对）
    -> 调用方校验 -> BusinessValidationError（non-retryable）
    -> 立刻结构化失败，不浪费重试预算在一个永远不会变"对"的响应上
```
验证于：`test_tool_failure.py::TestToolInvalidResponse`。与 #3/#4 的区别：
失败发生在**响应校验**阶段，而不是传输层。

### #6 MCP unavailable

**声明**：本仓库没有任何真实 MCP 集成。"MCP 不可用"被建模为一次连接失败
(`openai.APIConnectionError` / 通用 `ConnectionError`)，分类为 `NETWORK`
（可重试），走与 LLM 调用完全相同的 pipeline：
```
连接失败 -> classify(NETWORK, retryable) -> retry (backoff) -> 耗尽重试
        -> ToolInvocationError(category=network)
服务恢复（在重试预算内） -> 正常返回
```
验证于：`test_mcp_failure.py`。如果未来接入真实 MCP client，应当接入同一套
`src.reliability` pipeline，而不是另起一套处理逻辑。

### #7 Rate limit（LLM 端）

```
429 / RateLimitError -> classify(RATE_LIMIT, retryable) -> retry (backoff)
        -> 在预算内恢复，或耗尽重试 -> ToolInvocationError(category=rate_limit)
```
验证于：`test_llm_failure.py::TestRateLimit`。（并发场景下"多个调用方争抢同一个
限流 key"的角度见 #11。）

### #8 Worker crash

```
crash（未捕获异常逃出 graph.invoke，模拟进程崩溃）
    -> checkpoint（LangGraph 在每个已完成 superstep 后落盘）
    -> restart（全新的 graph 对象，指向同一个 checkpoint 文件）
    -> resume（graph.invoke(None, config)）
        -> 已经落盘的节点（Research A、Research B）绝不重新执行
        -> 只有真正没跑完的节点（Research C）继续
```
验证于：`test_worker_crash.py::TestWorkerCrash`（复用 `src/durable/recovery.py`
的 `CrashInjector`/`run_or_crash`/`resume`）。额外验证了"crash 发生在副作用**之
后**"的更细分支：Research C 的真实副作用已经执行过一次，resume 后节点会被
LangGraph 重新跑一遍（因为它从未被落盘确认完成），但幂等层保证副作用本身不会
被重复调用第二次。

### #9 Checkpoint failure

```
存储持续写入失败（模拟数据库故障/磁盘满/网络分区）
    -> graph.invoke() 抛出未被吞掉的异常（供上层监控/重启机制感知）
    -> 失败发生之前已经真正落盘的 checkpoint 保持完整、可读
    -> 存储恢复后，全新 graph 对象 resume() 成功完成
```
验证于：`test_worker_crash.py::TestCheckpointFailure`。

**一个诚实的发现**：LangGraph 会先跑完一个 superstep 的节点（含其副作用），再
尝试把该 superstep 落盘；这意味着"checkpoint 写入失败"完全可能发生在**副作用
已经执行之后**。本实验的 `_persistent_failure_from_step` 辅助函数刻意模拟
"存储从某个时间点起持续不可用"（而非仅一次性写入失败）——因为经过实测验证，单
次性的一次 `put()` 失败并不能可靠地阻止 LangGraph 之后的 superstep 各自独立地
成功落盘（`put()` 调用与节点执行之间并非严格的"必须等前一个写完成"关系）。
在"存储持续不可用"这个更贴近真实故障（数据库真的挂了）的建模下，行为是确定
的：失败点之前的 checkpoint 全部持久化成功，之后的一个都没有。

也因此得出与 #8 相同的重要结论：**真正防止副作用被重复执行的是幂等层
（`src.reliability.idempotency`），而不是 checkpointer 本身**——checkpoint 失败
只保证"未提交的进度会被安全地重新计算"，从不保证"副作用只运行一次"，那是幂等
层单独负责的正交保证。

### #10 Duplicate request

```
重复请求（相同 idempotency_key）
    -> idempotency store 命中已完成记录 -> 直接返回已缓存结果，不重新执行副作用
并发重复请求（相同 key，第一次调用仍在执行中）
    -> IdempotencyInProgressError（明确信号：稍后重试，而不是阻塞或二次执行）
key 相同但参数不同（冲突复用）
    -> IdempotencyConflictError（拒绝，而不是静默返回不相关的旧结果或静默重跑）
```
验证于：`test_duplicate_request.py`，直接使用
`src/reliability/idempotency.py` 的参考工具（`send_email`/`create_order`/
`publish_report`）以及 `src/durable/graph.py` 的研究节点。本项目的 HTTP 层
（`POST /runs`）刻意不接受调用方提供的 idempotency key（每次调用都铸造新的
`run_id`）——加上这个能力属于新业务功能，不在本实验范围内；已实现且可验证的
去重机制就是上面这一层，测试直接针对它。

### #11 Concurrent requests

```
N 个线程真并发地对同一个 idempotency key 发起调用
    -> 恰好一个真正执行了副作用，其余全部拿到相同结果或 IdempotencyInProgressError
N 个线程真并发地对同一个限流 key 发起调用
    -> 无论线程如何交错，通过数量永远不超过配置的上限
不同租户真并发地调用 HTTP API
    -> run_id 互不相同、互不可见（跨租户查询得到 404，与"不存在"不可区分）
```
验证于：`test_concurrency.py`，使用真实的
`concurrent.futures.ThreadPoolExecutor` + `threading.Barrier`（强制所有线程同时
起跑，制造真实竞争）而非顺序模拟，分别打向 `RateLimiter`、
`InMemoryIdempotencyStore`/`idempotent()`，以及完整的 `src.api.app` FastAPI 应用
（通过 `fastapi.testclient.TestClient`）。

### #12 Invalid structured output（已知缺口，如实记录，未修复）

```
Planner 的 LLM 回复不是可用 JSON
    -> _parse_aspects 容忍常见的"差一点对"的格式（markdown 代码块、编号列表）
    -> 但如果连编号列表都解析不出内容（彻底的乱码/空输出）
       -> aspects == []
       -> make_planner_node 抛出裸的 ValueError("Planner produced no research
          aspects; cannot continue.")
       -> 从 Planner 到 run_multi_agent_research 顶层之间，没有任何地方捕获
          这个异常 —— 它会让整个 graph.invoke() 调用崩溃
```
验证于：`test_llm_failure.py::TestInvalidStructuredOutput`（既验证了容忍路径，
也验证了会崩溃的路径）。**这是一个真实存在、当前未被处理的缺口**：Planner 层
没有对"结构化输出彻底无法解析"这种失败做重试/降级/结构化失败包装（对比 #1/#2/
#7 中 LLM 层本身的失败都被 `ToolInvocationError` 结构化包裹，这里是"LLM 调用本
身成功了，但返回内容不符合预期结构"，属于另一类失败，且目前没有对应的处理）。
按本实验"不新增业务功能"的约束，这里只记录、不修复；一个可能的未来修复方向是
比照 #5（Tool invalid response）的模式，把"aspects 为空"分类为
`BusinessValidationError` 并交给上层做结构化失败/降级处理，而不是裸 `raise`。

## 2. Failure → Detection → Recovery → User Experience

| # | Failure | Detection | Recovery | User Experience |
|---|---|---|---|---|
| 1 | LLM timeout | `ToolTimeoutError`（守护线程超时） | retry + backoff，耗尽后结构化失败 | 偶发延迟对用户透明；持续超时时该 worker 返回 `status="failed"`，其余 worker/研究任务不受影响 |
| 2 | LLM 5xx | `classify_openai_exception` -> `SERVER_ERROR` | retry + backoff，耗尽后结构化失败 | 同上；4xx（如 400）不重试，立即失败并保留原因 |
| 3 | Tool timeout | 同 #1（通用 `run_with_timeout`） | 同 #1 | 同 #1，适用于未来任意外部工具 |
| 4 | Tool 5xx | 同 #2（通用 `classify_exception`/`classify_generic_exception`） | 同 #2 | 同 #2 |
| 5 | Tool invalid response | 调用方响应校验 -> `BusinessValidationError` | 不重试，立即结构化失败（重试无法修复响应形状） | 明确的失败原因，而不是下游因为脏数据而莫名其妙崩溃 |
| 6 | MCP unavailable | `NETWORK` 分类（`APIConnectionError`/`ConnectionError`） | retry + backoff，耗尽后结构化失败 | 与 #1/#2 一致；无真实 MCP 集成时以文档声明代替实现 |
| 7 | Rate limit | `RATE_LIMIT` 分类 / `RateLimitExceededError`（API 层） | LLM 侧：retry + backoff；API 层：`429` + `Retry-After` | 客户端拿到明确的"何时可以重试"，而不是无提示地被拒绝或挂起 |
| 8 | Worker crash | 未捕获异常逃出 `graph.invoke()`（`WorkerCrash`/进程终止） | checkpoint -> restart（新 graph 对象）-> resume（跳过已完成节点） | 已完成的研究任务对用户不可见地被跳过重跑；只有真正未完成的部分继续，用户不需要重新提问 |
| 9 | Checkpoint failure | 存储写入异常从 `graph.invoke()` 逃出 | 失败前的 checkpoint 保持持久；存储恢复后 resume；幂等层防止副作用重复 | 用户可能观察到一次运行失败需要重试，但不会看到重复的副作用（不会收到两封邮件、不会被下两次单） |
| 10 | Duplicate request | idempotency key 命中已存在记录 | 直接返回缓存结果，不重跑副作用 | 重试一个"其实已经成功"的请求，用户拿到与第一次完全相同的结果，而不是第二次执行的新结果 |
| 11 | Concurrent requests | 锁/原子声明（`InMemoryIdempotencyStore`/`RateLimiter` 自带的线程锁） | 输家收到明确信号（`IdempotencyInProgressError`/`RateLimitExceededError`），而非被静默处理两次 | 高并发下用户看到的要么是唯一一致的结果，要么是明确的"请稍后重试"，从不会看到数据被搞乱 |
| 12 | Invalid structured output | 无 —— 当前未分类、未捕获 | 无 —— 整个运行崩溃（已知缺口） | 用户会看到一次不透明的失败（栈跟踪/500），而不是结构化的"研究规划失败，请重试"提示；见上文"已知缺口"说明 |

## 3. 运行

```powershell
.\.venv\Scripts\python.exe -m pytest tests\failure_injection -q
```

41 个测试，覆盖全部 12 种注入失败，全部通过；与既有 295 个测试（Durable
Execution / Security / Observability / Evaluation / Cost & Latency /
Deployment）合计 336 个测试，无回归。
