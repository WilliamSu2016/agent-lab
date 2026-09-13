# 07 — Cost & Latency Optimization

本实验为 Multi-Agent Research Agent（`src/multi_agent_research/`）建立成本模型，
并提供三种可切换的执行策略（Fast / Balanced / Quality），加上超预算时的**受控降级**
（controlled degradation）。

```
src/cost/
├── __init__.py      # 包说明 + 汇总导出
├── budget.py          # Budget / BudgetTracker / BudgetExceededError + 绝对硬上限 HARD_CEILING
├── policy.py           # ExecutionMode(Fast/Balanced/Quality) + 三套策略 + 受控降级 run_with_degradation
└── estimator.py         # 成本模型：回顾性（从 trace 计算实际花费）+ 前瞻性（从 policy 预测）

tests/test_cost.py      # Budget 校验、Tracker 越界检测、三策略参数关系、降级链路、成本模型
```

## 0. 成本模型统计维度

`src/cost/estimator.py::estimate_request_cost(root_span)` 直接消费
`src/observability/tracing.py` 产出的执行树（Observability 实验的 Span 树），
统计要求的全部维度：

| 统计维度 | 来源 |
|---|---|
| LLM calls | 树中 `SpanKind.LLM` 的数量 |
| input tokens | 每个 LLM span 的 `prompt_tokens` 属性求和 |
| output tokens | 每个 LLM span 的 `completion_tokens` 属性求和 |
| tool calls | 树中 `SpanKind.TOOL` 的数量 |
| subagent calls | 树中 `SpanKind.AGENT` 的数量（Planner/Worker/Synthesizer/Reviewer 每次被调用都算一次） |
| retry calls | 每个 span 的 `retry_count` 属性求和 |
| parallel workers | 同一个父节点下，名为 `research_worker` 的兄弟 AGENT span 的最大数量——即这次请求触发过的最宽一次 fan-out |

`RequestCostReport.cost_usd` / `.latency_ms` 直接给出 **cost per request** 和
**latency per request**：成本复用 `src/observability/metrics.py::estimate_cost_usd`
（同一张示例定价表，不重复发明），延迟直接是根 span 的 `duration_ms`（一次
`graph.invoke()` 的真实墙钟时间）。

没有真实 trace（还没跑起来）时，`estimator.py::predict_cost(policy)` 给出**前瞻性**
估算：基于 Multi-Agent Research 图固定的调用形状（每次迭代 = 1 次 Planner 调用 + 最多
`max_workers` 次 Worker 调用（并行，只计入成本、不重复计入延迟）+ 1 次 Synthesizer +
1 次 Reviewer，重复 `max_agent_iterations` 次），推算预计的 LLM 调用次数/token
数/成本/延迟，并且这些预测值永远被裁剪在对应 policy 自己的 Budget 上限之内。这让
调用方在真正发起一次 Quality 模式的请求之前，就能判断"这次请求大概会花多少钱、
要多久，我们负担得起吗"。

## 1. 三种执行策略

`src/cost/policy.py`：

| 策略 | max_agent_iterations | max_workers | max_tokens | max_cost_usd | timeout | 模型 | context |
|---|---|---|---|---|---|---|---|
| **Fast** | 1 | 2 | 6,000 | $0.02 | 30s | 低成本模型（`model_name_override`） | 2,000 chars（大幅裁剪） |
| **Balanced**（默认） | 3 | 6 | 40,000 | $0.20 | 120s | 沿用配置里的默认模型 | 8,000 chars |
| **Quality** | 6 | 10 | 120,000 | $0.80 | 300s | 沿用配置里的默认模型 | 24,000 chars（允许更完整的上下文） |

对应要求：

- **Fast**：减少 Agent calls（`max_agent_iterations=1`：Planner<->Reviewer 循环
  只跑一轮，不重新规划/复审）、减少 context（`max_context_chars` 最小）、低成本
  model（`model_name_override="gpt-4o-mini"`，调用方把这个偏好传给自己的
  `config.Settings`/模型选择逻辑，本模块从不直接持有/硬编码任何模型客户端）。
- **Balanced**：就是本项目 `src/multi_agent_research/state.py` 已有的默认值
  （`DEFAULT_MAX_WORKERS=6`、`DEFAULT_MAX_ITERATIONS=3`）——默认模式与既有系统的默认
  行为完全一致，这不是一个新引入的、行为不同的"第三选项"。
- **Quality**：允许更多 research / review 迭代（`max_agent_iterations=6`，是
  Balanced 的两倍）、更多并行 worker（`max_workers=10`）、更大的上下文窗口。

## 2. MAX_AGENT_ITERATIONS / MAX_WORKERS / MAX_TOKENS / MAX_COST / TIMEOUT

`src/cost/budget.py::Budget` 是这 5 个上限的载体；`src/cost/budget.py::HardCeiling`
（单例 `HARD_CEILING`）是一个**绝对的、与模式无关**的安全网——任何一个 `Budget`
（包括 Quality 模式自己的 Budget，也包括调用方临时构造的自定义 Budget）在构造时都会
校验不能超过这个硬上限（`Budget.__post_init__` 直接 `raise ValueError`）。这保证了
"Quality 模式"永远只是"比 Balanced 更宽松的上限"，而不是"无上限"。

`BudgetTracker` 是运行时强制这 5 个上限的唯一关口：

```python
tracker = BudgetTracker(policy.budget)
tracker.record_iteration()      # 超过 max_agent_iterations -> BudgetExceededError
tracker.record_workers(n)       # 超过 max_workers -> BudgetExceededError
tracker.record_tokens(n)        # 超过 max_tokens -> BudgetExceededError
tracker.record_cost(usd)        # 超过 max_cost_usd -> BudgetExceededError
tracker.check_timeout()         # 超过 timeout_seconds -> BudgetExceededError
```

任何一次越界都会抛出结构化的 `BudgetExceededError(dimension, used, limit)`，而不是
一个笼统的字符串异常——这是 `policy.py` 里受控降级判断"该往哪个方向降级、原因是
什么"所需要的最小信息。

## 3. 超预算 -> 受控降级（Controlled Degradation）

```
Quality  --(超预算/超时)-->  Balanced  --(超预算/超时)-->  Fast  --(超预算/超时)-->  Graceful Failure
```

`src/cost/policy.py::run_with_degradation(execute, start_mode=QUALITY)`：

```python
def execute(policy: ExecutionPolicy):
    # 用 policy.budget / policy.model_name_override / policy.max_context_chars
    # 实际去跑一次 Multi-Agent Research 请求；超预算时抛出
    # BudgetExceededError 或 TimeoutError。
    ...

outcome = run_with_degradation(execute, start_mode=ExecutionMode.QUALITY)
if outcome.succeeded:
    print(outcome.final_mode, outcome.result)
else:
    print(outcome.graceful_failure.to_trace_line())
```

行为：

1. 从 `start_mode`（默认 Quality）开始尝试 `execute(policy)`。
2. 一旦抛出 `BudgetExceededError`/`TimeoutError`，记录这次失败的尝试
   （`DegradationAttempt(mode, succeeded=False, error=...)`），沿着
   `DEGRADATION_ORDER = (QUALITY, BALANCED, FAST)` 降级到下一个更"便宜"的模式，
   重试。
3. 如果连 Fast 也失败，返回一个 `GracefulFailure`（不是抛异常、不是无限重试）——
   这是一个**设计好的、可观测的终止状态**，与
   `src/reliability/errors.py::ControlledFailure`（Planner<->Reviewer 迭代预算
   耗尽时的处理）同一设计原则，只是应用在"模式选择的预算"这一层。
4. `on_degrade(from_mode, to_mode, reason)` 是可选钩子，用于把每一次降级写进
   `src/observability/logging.py` 的结构化日志/`src/observability/tracing.py`
   的 trace，本模块本身不直接依赖任何具体日志后端。

`DegradationOutcome` 携带完整的尝试历史（`attempts`），即使最终成功了，也能看到
"这次请求实际上是从 Quality 降到 Balanced 才跑通的"——这本身就是一个值得写入
Observability 的信号（哪些请求经常触发降级，往往意味着预算设置得偏紧或者该请求
本身开销异常）。

## 4. 与既有实验的关系

- 成本/延迟的**统计**完全建立在 Observability 实验（`src/observability/`）的 Span
  树之上——`estimate_request_cost` 不重新发明一套埋点，Multi-Agent Research 只要
  已经用 `Tracer.span(...)` 包好了 Planner/Worker/Synthesizer/Reviewer/LLM 调用，
  这里就能直接算出成本报告。
- 美元成本的**定价表**复用 Observability 实验的
  `src/observability/metrics.py::estimate_cost_usd`——同一张"仅供参考"的定价表，
  在 Observability 里用来回答"哪个 Agent 最贵"，在这里用来回答"这次请求花了多少
  钱"，两者必须一致，否则同一个请求在两个模块里会算出不同的成本。
- `BudgetExceededError`/`GracefulFailure` 的设计哲学直接照抄 Reliability 实验
  （`src/reliability/errors.py`）里 `ToolError`/`ControlledFailure` 的"分类、
  结构化、绝不裸抛异常"原则，只是把它用在"选择哪个执行模式"这一层决策上。

## 5. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cost.py -q
```

覆盖：`Budget` 对 `HARD_CEILING` 的校验（超限拒绝构造）、`BudgetTracker` 对 5 个
维度分别的越界检测、降级后用 `check_all()` 重新核验历史用量、三个策略在
`max_agent_iterations`/`max_workers`/`max_context_chars`/`model_name_override`
上符合"Fast 更省、Quality 更宽松、Balanced 是默认"的关系、完整的降级链路（一次
成功 / 降一级成功 / 一路降到 Fast 成功 / 全部失败进入 GracefulFailure /
`on_degrade` 钩子被正确调用）、以及从一棵手工搭建的 Span 树计算出的
`RequestCostReport` 每一个字段是否与手工核算的期望值一致（LLM 调用数/输入输出
token/工具调用数/子代理调用数/重试次数/最大并行 worker 数/成本/延迟）。

## 6. 已知限制

- `predict_cost` 使用的平均 token 数/平均耗时是**示例性默认值**，明确标注为
  非权威——生产环境应该用 `src/observability/metrics.py::MetricsRegistry`
  积累的真实历史数据替换这些默认值（函数已经把它们做成可覆盖的关键字参数）。
- Fast 模式的"低成本 model"只是一个字符串偏好（`model_name_override`），本模块
  不直接持有任何 LLM 客户端，也不负责真正切换模型——真正生效需要调用方把这个值
  接入自己的 `config.Settings`/模型选择逻辑（例如构造
  `src/specialists/llm.py::build_openai_text_llm_call` 时使用这个模型名而不是
  `settings.model_name`）。
- `parallel_workers` 的统计基于 span **名字**（默认 `"research_worker"`），如果
  一个图给并行 worker 用了不同的命名约定，需要显式传入
  `estimate_request_cost(root, worker_span_name=...)`。
- 与 Observability/Reliability 实验一样，`BudgetTracker`/降级历史目前都是进程内
  状态，不跨进程/请求持久化；真实生产环境中，跨请求的预算（例如"本小时内全局成本
  上限"）需要一个共享的、持久化的计数器后端，这里提供的是单次请求内该如何强制这
  5 个上限的逻辑，而不是跨请求的全局限流器。
