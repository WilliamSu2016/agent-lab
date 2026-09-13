# 02 — Reliability

本项目所有 Agent Tool 调用（目前唯一的真实外部工具是 `openai` LLM 客户端；未来任何 HTTP/MCP/业务工具都应遵循同样的模式）统一通过 `src/reliability/` 包实施可靠性策略：**分类 → 重试 / 降级 / 终止 → 结构化错误 → trace**，以及独立的超时、Agent 循环预算、幂等性机制。任何一次工具调用都**不允许**以“裸异常 → 应用崩溃”的方式结束。

## 0. 包结构

```
src/reliability/
├── __init__.py     # 汇总导出，四个子模块的角色说明
├── errors.py        # 错误分类 + 结构化 ToolError + AgentLoopGuard/ControlledFailure
├── retry.py          # 指数退避 + 最大重试次数，依赖 errors.py 的分类结果
├── timeout.py        # 唯一的超时实现（daemon thread），供所有外部调用复用
└── idempotency.py    # 幂等性框架 + 三个参考工具（send_email/create_order/publish_report）
```

`tests/test_errors.py`（47 个用例）、`tests/test_retry.py`（18 个）、`tests/test_timeout.py`（13 个）、`tests/test_failure_injection.py`（12 个，端到端组合场景）覆盖了以下全部设计。

## 1. Retry：指数退避 + 最大重试次数 + 可重试分类

### 1.1 分类表（`src/reliability/errors.py::ErrorCategory`）

| 类别 | 是否可重试 | 说明 |
|---|---|---|
| `TIMEOUT` | ✅ | 调用超过其超时预算 |
| `NETWORK` | ✅ | `ConnectionError`/`OSError`/`openai.APIConnectionError` 等临时网络故障 |
| `RATE_LIMIT` | ✅ | HTTP 429 / `openai.RateLimitError` |
| `SERVER_ERROR` | ✅ | HTTP 5xx |
| `INVALID_PARAMETERS` | ❌ | `ValueError`/`TypeError`/HTTP 400/404/409/422 等——参数本身有问题，重试不会改变结果 |
| `AUTHENTICATION` | ❌ | HTTP 401 / `openai.AuthenticationError` |
| `PERMISSION_DENIED` | ❌ | HTTP 403 / `PermissionError` / `openai.PermissionDeniedError` |
| `BUSINESS_VALIDATION` | ❌ | 显式抛出的 `BusinessValidationError`（业务规则拒绝，例如库存不足） |
| `UNKNOWN` | ❌（fail-safe 默认不重试） | 无法识别的异常；宁可不重试也不要对不理解的失败盲目重试 |

`classify_exception()` 是默认分类器（`BusinessValidationError` → openai 专用分类器 → 通用兜底分类器）；调用方也可以传入自定义 `classify=...`（例如 `classify_openai_exception`，`src/specialists/llm.py` 就是这样用的）。**分类逻辑是唯一的“可重试/不可重试”真相来源**，不允许在各个调用点各自判断。

> 备注：Configuration 实验中 `config.RetryPolicy.retry_on_status_codes`（可配置的 HTTP 状态码白名单）字段仍然保留在 `config/settings.py` 中作为历史/文档参考，但 `src/specialists/llm.py` 现在实际使用的是本模块固定的类别判断（上表），不再按可配置状态码列表来决定是否重试——这是本次 Reliability 实验按需求明确规定的“可重试/不可重试”分类表覆盖了此前更宽松的可配置版本。`config.RetryPolicy` 中 `max_retries`/`backoff_base_seconds`/`backoff_max_seconds` 三个数值仍然完整可配置，并在调用点转换为 `reliability.retry.RetryPolicy`。

### 1.2 退避算法（`src/reliability/retry.py::compute_backoff_seconds`）

第 `attempt` 次重试前的等待时间 = `min(backoff_base_seconds * 2^(attempt-1), backoff_max_seconds)`，再加上最多 `jitter_ratio`（默认 10%）的随机抖动，避免大量并发调用方在同一故障发生后同时重试（thundering herd）。

### 1.3 `retry_call()`：统一的调用管道

```python
from src.reliability.retry import RetryPolicy, retry_call
from src.reliability.errors import classify_openai_exception

result = retry_call(
    lambda: client.chat.completions.create(...),
    policy=RetryPolicy(max_retries=2, backoff_base_seconds=1.0, backoff_max_seconds=20.0),
    tool_name="openai_chat_completion",
    classify=classify_openai_exception,
    on_error=lambda tool_error: logger.warning(tool_error.to_trace_line()),
)
```

行为：每次异常都先被分类并包装为 `ToolError`（`on_error` 钩子在这一步被调用，即 trace）；如果可重试且预算未耗尽，退避后重试；否则抛出 `ToolInvocationError`（包装最后一次的 `ToolError`，`__cause__` 保留原始异常）。**原始异常永远不会直接传播给调用方**——这就是“不要 raise exception → 崩溃”的具体实现。

## 2. Timeout：每个 external tool 必须有 timeout

`src/reliability/timeout.py::run_with_timeout(func, timeout_seconds, tool_name=...)` 是**唯一**的超时实现，取代了此前分别存在于 `research_worker.py`、`graph.py`、`04_parallel_multi_agent.py` 里三份几乎相同的 `_run_with_timeout`（现在这三处都只是薄封装/直接调用这一个实现）。

- 使用 `threading.Thread(daemon=True)` 而不是 `concurrent.futures.ThreadPoolExecutor`：后者会注册一个全局 `atexit` 钩子，即使某个 executor 已经 `shutdown(wait=False)`，进程退出前仍会 join 它创建过的所有线程——一次真实的网络挂起会让整个进程退出被拖慢到远超配置的超时时间之后（这是本项目此前实验中实测到的问题）。daemon 线程没有这个钩子：超时一到就抛出 `ToolTimeoutError`（`TimeoutError` 的子类，兼容所有 `except TimeoutError` 的既有代码），线程本身由 OS 在进程退出时自然回收。
- `timeout_seconds=None` 表示不设超时（用于总运行预算等可选场景）；`<= 0` 直接 `ValueError`。
- **重要限制**：这只限定“调用方等待多久”，并不能真正取消已经在执行的底层调用——一个已经发出的真实网络请求，在超时之后仍可能在服务端继续执行/计费。需要真正取消语义的场景应优先使用客户端原生的超时/取消支持，而不是仅依赖这层包装。

`TimeoutPolicy` 数据类把 `tool_name`/`timeout_seconds` 绑定在一起，方便从 `config.Settings.limits` 为每个工具构建一份具名的超时预算。

`src/specialists/llm.py::build_openai_text_llm_call()` 现在用 `settings.limits.worker_timeout_seconds` 包住每一次 `client.chat.completions.create(...)` 调用，再交给 `retry_call` 重试——即真正做到“每个 external tool 调用都有 timeout，且 timeout 内部也会被分类为可重试的 `TIMEOUT`”。

## 3. Agent loop：max_iterations → controlled failure

`src/reliability/errors.py::AgentLoopGuard` 封装了“最多迭代 N 次，超过后进入受控失败”这一策略：

```python
guard = AgentLoopGuard(max_iterations=3)
while True:
    guard.enter_iteration()   # 超预算时抛出 ControlledFailureError（携带 ControlledFailure）
    ...
    if approved:
        break
```

`ControlledFailure` 是一个结构化的、**设计好的**终止状态（`reason`/`iterations_used`/`max_iterations`/`last_feedback`/`failure_id`/`occurred_at`），区别于单次工具调用失败的 `ToolError`：它描述的是整条 Planner↔Reviewer 循环“没有在预算内达成目标”这件事本身。

本项目已有的 `src/multi_agent_research/graph.py` 中 Planner↔Reviewer 循环（`reviewer.route_after_review`）此前就已经在 `iteration >= max_iterations` 时路由到 `finalizer` 并返回一个 best-effort 的 `final_answer`（保持这个向后兼容的行为不变）；本次改造在此基础上，为 `MultiAgentResearchOutcome` 增加了一个新的、默认 `None` 的 `controlled_failure` 字段：当 Reviewer 在预算耗尽时仍未通过审核，`run_multi_agent_research()` 会额外构造并附上一个 `ControlledFailure`，供调用方显式识别/记录/上报“这是一次受控失败”，而不是把它和正常通过（PASS）混为一谈，也不需要靠解析 `final_answer` 里的文字判断。

## 4. Error handling：分类 → 重试/降级/终止 → 结构化错误 → trace

统一管道（由 `retry_call` 驱动，`timeout`/`errors` 提供分类和结构化能力）：

```
工具调用抛出异常
   -> classify_exception(exc)               # errors.py：得到 ErrorCategory
   -> make_tool_error(...)                   # errors.py：包装为结构化 ToolError
   -> on_error(tool_error)                   # 调用方的 trace/日志钩子（每次都会触发）
   -> retryable? 且预算未耗尽？
        -> 是：退避后重试（retry.py）
        -> 否：raise ToolInvocationError(tool_error)  # 而不是裸异常/崩溃
```

- **retry**：类别属于 `RETRYABLE_CATEGORIES` 且未超过 `max_retries`。
- **fallback**：本项目目前没有真正的多工具降级路径（单一 LLM 客户端），但 `retry_call` 的 `classify=` 参数使得任何未来新增的工具都可以插入自己的降级/分类逻辑而不改变整体管道；`ToolInvocationError` 携带的 `ToolError` 足以让上层调用方决定是否切换到备用工具。
- **terminate**：不可重试，或重试耗尽——统一抛出 `ToolInvocationError`，绝不是裸的 `openai.*`/`TimeoutError`/`ValueError`。
- **trace**：`ToolError.to_trace_line()` / `to_dict()`（`ControlledFailure` 同样提供这两个方法）给出可直接打日志或序列化上报的结构化记录。

## 5. Idempotency：具有 side effect 的 Tool

需求明确列出的三个例子——`send_email`、`create_order`、`publish_report`——都在 `src/reliability/idempotency.py` 中给出了参考实现（**本项目当前没有真正的、会产生外部副作用的 Tool 层**，见 `docs/00-PRODUCTION-ARCHITECTURE.md` 的 P2-01 gap；这三个是可直接复制的参考实现，通过 `tests/test_failure_injection.py` 验证其行为，而不是接入到当前生产流水线中）。

| Tool | 幂等 key | 冲突检测 fingerprint | 设计要点 |
|---|---|---|---|
| `send_email` | 调用方提供的 `idempotency_key`（例如由业务事件派生，如 `"welcome-email-for-order-123"`），**绝不**在工具内部临时生成 | `to`/`subject`/`body` | 同一个 key 的重试（例如网络超时导致响应丢失后的重试）必须返回同一封邮件的结果，不会真的发第二封 |
| `create_order` | 调用方提供的 `idempotency_key`（例如结账会话 ID） | `customer_id`/`amount_cents` | 典型的“绝不能重复扣款”场景：请求已在服务端成功执行、但响应丢失的重试，必须返回同一个 `order_id`，而不是新建一个订单 |
| `publish_report` | 业务自然键 `f"{report_id}:{version}"` | `content` | 同一版本的重复发布（例如 Reviewer 重试、崩溃恢复重放、调度器重复触发）必须是无副作用的重复调用；发布**新版本**则是合法的新副作用 |

统一框架（`IdempotencyStore` 协议 + `idempotent()` 包装器）：

- **首次调用**：以 `key` 声明“进行中”，执行真正的副作用，记录结果为“已完成”。
- **同 key 同参数的后续调用**（已完成）：直接返回缓存结果，不重新执行副作用。
- **同 key 不同参数**：抛出 `IdempotencyConflictError`，绝不静默返回不相关的旧结果，也绝不用新参数再执行一次副作用。
- **同 key 且仍在进行中的并发重复调用**：抛出 `IdempotencyInProgressError`，调用方应等待/稍后重试，而不是重新执行副作用。
- **副作用执行失败**：释放该 key 的“进行中”声明（`store.fail(key)`），允许一次合法的后续重试真正执行。

`InMemoryIdempotencyStore` 是线程安全但**非持久化**的实现，仅用于测试/单进程演示；真实生产环境（尤其是多进程/多副本部署，或需要容忍进程重启）**必须**换成持久化存储（例如带唯一约束的数据库表，`INSERT ... ON CONFLICT DO NOTHING` 实现原子的“声明进行中”），这一点在模块 docstring 与本文档中都已明确标注为限制，不是本次改造遗漏的问题。

## 6. 本次改造的接线（wiring）总结

- `src/specialists/llm.py::build_openai_text_llm_call(settings)`：不再有自己的内联重试循环，而是用 `settings.retry` 构造 `reliability.retry.RetryPolicy`，用 `settings.limits.worker_timeout_seconds` 通过 `reliability.timeout.run_with_timeout` 包裹每次 `client.chat.completions.create(...)`，再交给 `reliability.retry.retry_call(..., classify=classify_openai_exception, on_error=...)`。
- `src/multi_agent_research/research_worker.py`、`src/multi_agent_research/graph.py`、`src/04_parallel_multi_agent.py`：各自的 `_run_with_timeout` 私有实现全部替换为对 `reliability.timeout.run_with_timeout` 的调用（保留原有的公开异常类型 `TimeoutError`/`MultiAgentResearchTimeoutError`/`ParallelResearchTimeoutError` 不变，因为 `ToolTimeoutError` 本身就是 `TimeoutError` 的子类）。
- `src/multi_agent_research/graph.py::run_multi_agent_research()`：新增 `MultiAgentResearchOutcome.controlled_failure`（默认 `None`），在 Reviewer 未通过且迭代预算耗尽时填充一个 `ControlledFailure`。
- 以上改造均已通过：`tests/test_config.py`（29）+ `tests/test_errors.py`（47）+ `tests/test_retry.py`（18）+ `tests/test_timeout.py`（13）+ `tests/test_failure_injection.py`（12），共 119 个测试全部通过；并对每个被修改的模块做了 AST 解析 + `import` 冒烟测试确认没有破坏既有代码。
