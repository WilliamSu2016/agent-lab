# 11 — Production Ready: Multi-Agent Research Agent → Production Research Agent

本文总结如何依据 `docs/10-Final-Review.md` 的发现，把原本互不相连的多个原型
（`multi_agent_research` 业务图 + `durable`/`security`/`observability`/
`cost`/`api`/`reliability` 六个独立能力模块）组装成一个**单一、真实、
生产可用**的 Agent：`production-research-agent/`。

## 0. 核心变化（一句话）

> 之前：真正部署的图是不调用 LLM 的 `durable` demo，真正的研究 Agent
> （Planner → Worker → Synthesizer → Reviewer）从未接入 API/guardrails/
> observability/checkpoint。
>
> 现在：**只有一个图**（`production-research-agent/src/graph/graph.py::build_graph`），
> 它就是那个真实会思考、会调研的 Planner → Researcher(s) → Synthesizer →
> Reviewer → notify_action 流水线，而且 guardrails / checkpoint /
> observability / cost policy 全部**直接写在这条真实执行路径里**，不是挂
> 在旁边的独立模块。

## 1. 逐条 P0 问题的关闭方式

`docs/10-Final-Review.md` 的核心发现和评分基于六个角色视角。下面逐条说明
每个 P0/结构性问题在新实现里是如何关闭的。

| Final Review 发现 | 新实现里的关闭方式 |
|---|---|
| 部署的是 `durable` demo 图，不是真实业务图 | `langgraph.json` 的 entrypoint 直接指向真实图 `src/graph/entrypoint.py:graph`；FastAPI (`src/api/app.py`) 和所有测试共用同一个 `build_graph()`，不存在"两套图" |
| `src/security/guardrails.py` 在业务路径里零引用 | `supervisor_entry`（Layer 1 输入护栏）、每次 tool 调用（Layer 2/3）、`finalizer`（输出校验）全部直接调用 `AgentWorkflowGuardrail`/`ToolGuardrail`/`InputGuardrail`；`tests/security/`（68 个用例）针对真实图运行，不是针对独立模块的孤立单测 |
| HIGH 风险工具（`send_email`）没有"审批网关"落地，只在文档里描述 | 新增 `notify_action` 节点（`src/agents/supervisor.py::make_notify_node`），把 `finalizer → notify_action → END` 接入图；`ApprovalRequiredError` 未捕获、逼迫 LangGraph 在 checkpoint 处暂停，人工审批后 `resume()` 才能继续 —— `Agent → approval → Tool` 是可执行、被集成测试验证的事实，不是设计意图 |
| API 背后没有真实 checkpoint/durable 恢复 | `src/graph/checkpointer.py::sqlite_checkpointer` + `src/graph/recovery.py`（`run_or_crash`/`resume`/`get_execution_history`/`get_pending_tasks`）接入 `src/api/runs.py`；Research A/B/C 崩溃恢复场景在 `tests/failure_injection/test_worker_crash.py` 中对真实图验证通过 |
| `observability`/`cost` 只薄薄接在 HTTP 层，从未进入 Planner/Worker/Synthesizer/Reviewer 内部 | 每个 Agent 节点、每次 LLM 调用、每次工具调用都有 `tracer.span()`（含 token/cost/duration），`MetricsRegistry.record_from_trace()` 直接从真实执行树派生指标；解决了 `Send` 并行 fan-out 下 `contextvars` 不传播导致 trace 链路丢失的真实 bug |
| Evaluation 数据集与真实架构不匹配（specialist-router 词表） | 完全重写 20 个任务，使用这条图真实的路由词表（`completed`/`blocked_input_guardrail`/`blocked_authorization`/`notify_sent`/`notify_denied`）；针对真实图运行离线评估，得到可复现的 100% pass rate 基线（`evals/reports/baseline_report.md`） |
| `ApprovalStore` 存在但无可达 API | 新增 `GET /approvals/{id}` / `POST /approvals/{id}/decide`（`src/api/app.py`） |
| 无 Request ID / 无 graceful shutdown / 健康检查未区分 liveness vs readiness | `request_context_middleware` 注入 `X-Request-ID`；`GET /health`（liveness）与 `GET /ready`（readiness，关停期间自动 503）分离；FastAPI `lifespan` 在关停时先让 `/ready` 失败、再 `drain()` 等待在途 run 完成 |

## 2. Production Readiness Score（升级后）

| 维度 | 分数 | 说明 |
|---|---|---|
| Architecture | 9/10 | 单一真实图、三层护栏、durable checkpoint、cost policy 全部集成在同一条执行路径；MCP/RAG 仍未实现（诚实声明，非隐瞒） |
| Reliability | 9/10 | 真实验证的 crash→checkpoint→resume（A/B 不重跑，C 续跑）；retry/timeout/idempotency 全部接入真实图；`RunRegistry` 仍是进程内内存索引（已知限制） |
| Security | 9/10 | 三层护栏接入真实执行路径；HIGH 风险工具审批网关端到端可达并被集成测试覆盖；68 个安全测试覆盖 10+ 攻击场景 |
| Observability | 8/10 | 完整执行树 + token/cost/latency 指标 + 结构化日志（默认脱敏）；Dashboard 规格已写（`docs/observability.md`），但没有真实部署一套 Grafana/Prometheus 去验证 |
| Evaluation | 8/10 | 20 任务、11 维度、真实图上跑出 100% 基线、回归检测比较全部指标而非仅字符串；echo-stub LLM 验证的是管道正确性而非真实 LLM 质量，需要第二条真实 LLM 评估作为质量维度补充 |
| Scalability | 6/10 | 单机 LangGraph 运行时、SQLite 单写者、`RateLimiter`/`ApprovalStore`/`RunRegistry` 均为进程内内存态 —— 单实例生产可用，尚不支持多副本水平扩展（已知限制，非隐瞒） |
| Cost | 8/10 | Fast/Balanced/Quality 三档策略 + `MAX_AGENT_ITERATIONS`/`MAX_WORKERS`/`MAX_TOKENS`/`MAX_COST`/`TIMEOUT` 预算 + 自动降级（Quality→Balanced→Fast→graceful failure）均已实现并可在 trace/metrics 中观测 |
| UX | 7/10 | 非阻塞 `POST /runs` + 轮询式 `GET /runs/{id}/state`（含 execution history）+ resume + 人工审批；尚无真正的流式（SSE/WebSocket）进度推送 |

## 3. P0 / P1 / P2 / P3（升级后剩余问题）

**P0 — 必须在真正上生产前修复**
1. `RunRegistry`/`RateLimiter`/`ApprovalStore` 的进程内内存态需要迁移到共享
   存储（Redis/Postgres），否则多副本部署下限流被整除放大、审批/幂等结果
   互不可见、进程重启后 HTTP 可见的 run 状态丢失（底层 LangGraph
   checkpoint 本身没丢，但 `GET /runs/{id}` 查不到了）。
2. SQLite checkpointer 是单文件单写者，需要在生产多副本场景下替换为
   Postgres（LangGraph 官方支持的 `PostgresSaver`），否则无法水平扩展。

**P1 — 应尽快修复**
3. `notify_email`（HIGH 风险审批 demo）和 `memory_store`（回答缓存）目前
   只在 Python/graph 层可达，尚未作为 `POST /runs` 的请求字段暴露 —— 需要
   补上 API schema 字段和对应集成测试。
4. Evaluation 目前仅用 echo-stub LLM 验证管道正确性；需要补一条真实 LLM
   驱动的评估流水线（较低频率运行，例如每晚一次），用同样的 11 维度指标
   评判真实回答质量，而不只是管道是否正确路由。
5. 缺少真实部署的 Dashboard（Prometheus/Grafana 等）去验证
   `docs/observability.md` 里的规格是否真的可用；目前只有规格文档和内存态
   `MetricsRegistry`。

**P2 — 可以之后改进**
6. 补充真正的流式（SSE/WebSocket）进度推送，让长任务的 UX 从"轮询 state"
   升级为"实时进度"。
7. MCP client/tool-discovery 层仍未实现 —— 当前工具集合是硬编码的两个示例
   工具，扩展新工具需要改代码而非动态发现。
8. RAG/retrieval 层仍未实现 —— Researcher 依赖模型自身知识，无法验证引用
   是否真实存在；`evals` 里的 groundedness/citation 维度已经就绪，等待接入
   真实检索层。

**P3 — 锦上添花**
9. `RunRegistry`/`ApprovalStore`/`RateLimiter` 的内存实现可以先加一层简单
   的本地持久化（例如 SQLite 表）作为迁移到 Redis/Postgres 之前的过渡方案。
10. 补充跨进程/跨机器的 Researcher 任务分发（当前并行只是单进程内的多线程
    `Send` fan-out），为未来更大规模的并行研究做准备。

## 4. 最终回答

**这个 Agent 现在到底是不是 Production Ready？**

**对于单实例、中等流量的生产部署：是。** 三层护栏、durable
checkpoint+crash/resume、HIGH 风险工具的人工审批网关、完整 observability
（trace/metrics/结构化脱敏日志）、cost/latency 预算与自动降级、真实图上
100% 通过的评估基线与回归检测、可部署的 Docker/Compose + 认证/限流/超时/
健康检查/优雅关闭，都已经端到端集成并有测试证明（153 个测试全部通过，
覆盖 unit/integration/security/failure_injection 四类场景），不再是
"设计文档 vs 从未集成的原型"的状态。

**对于多副本水平扩展的生产部署：还不是。** 因为 P0 里的两个问题
（`RunRegistry`/`RateLimiter`/`ApprovalStore` 的进程内内存态、SQLite 单写者
checkpointer）会在多副本场景下产生真实的数据不一致/丢失问题——这不是"锦上
添花"，是会导致同一租户的两个请求被路由到不同副本时看到不一致结果的结构性
限制。

**如果不是（多副本场景），最关键的 5 个阻塞问题是什么？**

1. `RunRegistry` 进程内内存态 —— 进程重启/多副本下 HTTP 可见的 run 状态
   不一致或丢失（底层 checkpoint 本身没丢）。
2. `RateLimiter` 进程内内存态 —— 多副本下限流被整除放大，实际生效限流是
   配置值 × 副本数。
3. `ApprovalStore` 进程内内存态 —— 多副本下审批决定可能对发起审批的那个
   副本以外的副本不可见，导致 `resume()` 看不到已经批准的审批。
4. SQLite 单写者 checkpointer —— 无法在多副本间共享同一个写者，必须迁移到
   `PostgresSaver` 才能支撑水平扩展。
5. `notify_email`/`memory_store` 尚未暴露为 `POST /runs` 的请求字段 ——
   这两个已经在代码和集成测试里证明可工作的能力，目前对通过公开 HTTP API
   接入的真实用户来说是不可达的，需要补上 API 层的字段才能真正对外提供
   这两项能力。

单实例部署可以直接上生产；要支撑多副本水平扩展，需要先解决上面 5 个问题
（1-4 是共享状态迁移，5 是 API 完整性）。
