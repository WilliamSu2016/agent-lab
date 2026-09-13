# 10 — Production Agent Final Review

评审日期：2026-09-12
评审方式：只读代码评审（未运行真实 LLM 请求、未部署、未修改任何代码）。
评审假设：这个 Agent 明天要直接服务真实用户。

范围：`src/multi_agent_research`（唯一有业务价值的 Agent 主线）+
`src/durable`、`src/security`、`src/observability`、`src/cost`、`src/api`、
`src/reliability`（历次实验产出的生产能力模块）+ `evals/` + `deployment/`。

---

## 核心发现（贯穿所有角色的最大问题）

在展开六个角色的评审之前，有一个**结构性事实**决定了后面所有评分：

> **`src/api/main.py`（生产 API 入口）实际挂载的图是
> `src/durable/graph.py::build_durable_graph`—— 一个用占位符
> `_default_side_effect`（只拼字符串，不调用 LLM）演示"崩溃恢复"概念的教学图，
> 而不是真正会思考、会调研的 `src/multi_agent_research`（Planner → Worker →
> Synthesizer → Reviewer，真实调用 OpenAI）。**

也就是说：

- **真正有业务价值的 Agent（multi_agent_research）从未被部署过**——它没有
  API、没有鉴权、没有限流、没有 checkpoint、没有 observability、没有
  guardrails。
- **真正被部署的图（durable demo）没有业务价值**——它甚至不调用 LLM。
- `src/security/guardrails.py`（Prompt Injection 检测 / Tool 分级审批 / 输出
  校验）**在 `src/api`、`src/durable`、`src/multi_agent_research` 中零引
  用**——只在自己的单元测试里被调用过。`src/api/schemas.py` 的 docstring 甚至
  写着"upstream of `src.security.guardrails`"，但代码里根本没有 `import`。
  这是文档承诺与代码事实不符。
- `src/observability`、`src/cost` 同样只在 `src/api` 的 HTTP 层薄薄接了一层
  （request_id/日志/超时预算），从未进入 `src/multi_agent_research` 内部
  （没有对 Planner/Worker/Synthesizer/Reviewer 每次 LLM 调用做
  span/token/cost 记录）。

结论：**这不是"一个 Agent 有一些缺陷"，而是"六个高质量的独立能力模块
（Durable / Security / Observability / Evaluation / Cost / Deployment）从未
被组装成一个能力完整的 Agent"。** 下面的角色评审和分数都建立在这个事实之上。

---

## 1. AI Architect 视角

| 检查项 | 现状 |
|---|---|
| Agent orchestration | `multi_agent_research` 有完整的 Planner/Worker/Synthesizer/Reviewer + 审阅回路，设计良好；但从未接入任何对外服务入口 |
| State | `TypedDict` + 局部更新 + reducer，干净；但两套 State（`multi_agent_research.state` vs `durable.state`）完全独立，没有统一的"生产 State" |
| Tools | 无真实外部工具层。`security/tool_policy.py` 定义了 LOW/MEDIUM/HIGH 三级和 `send_email`/`update_record` 等**示例**，但项目里不存在任何真实被调用的工具（`multi_agent_research` 只是纯 LLM 文本生成，Worker 不能"调用工具"） |
| MCP | 完全未实现，`docs/09` 已如实声明；无 client/server/discovery 代码 |
| Memory | 无跨会话长期记忆；`InMemorySaver`/SQLite checkpointer 都只是"单次运行的执行状态"，不是"用户记忆" |
| RAG | 无 ingestion/embedding/index/retriever；Worker 靠模型自称"根据官方文档"，无法验证依据是否真实存在 |
| Multi-Agent | 设计合理（并行 Worker + 反馈回路），但完全是单机内 LangGraph 进程内调度，没有跨进程/跨机器的任务分发 |

**AI Architect 结论**：架构设计文档质量很高（`docs/00`），概念齐全，但"设计"
和"已集成实现"是两回事——目前是**多个互不相连的原型**，不是一个架构完整的
系统。

---

## 2. Backend Engineer 视角

- API 契约干净（`POST /runs` / `GET /runs/{id}` / `GET /runs/{id}/state` /
  `POST /runs/{id}/resume`），Pydantic 校验、统一错误结构、request_id 中间件
  都到位。
- **但 API 背后跑的是占位符 demo 图**，意味着这套精心设计的 API 目前对任何
  真实用户请求都只能返回"researching aspect 'A' of question=...”这种假
  文本，没有一行真实业务逻辑落地。
- `RunRegistry` 是纯内存索引（`self._records: dict`），进程重启后所有 run 的
  HTTP 可见状态（`status`/`final_report`）全部丢失，即使底层 checkpoint 还
  在——**恢复的是图执行，不是 API 语义**（`GET /runs/{id}` 在进程重启后直接
  查不到这个 run_id 了，因为它是内存 dict）。这与文档里"要支持恢复"的目标是
  矛盾的。
- SQLite 用 `check_same_thread=False` 单连接贯穿进程生命周期，配合
  `asyncio.to_thread` 起后台线程跑图——这是可行的单实例方案，但**完全不支持
  多副本水平扩展**（SQLite 单文件、单写者），`RateLimiter`/
  `IdempotencyStore`/`RunRegistry` 全部是进程内内存态，多副本部署下限流会被
  整除放大、幂等失效、run 注册表互不可见——这几点代码注释里其实**已经诚实
  写明了**，是好事，但也说明当前架构对"扩容"完全没有准备。

---

## 3. SRE 视角(Site Reliability Engineer,站点可靠性工程师)

**Reliability**

- Retry / Timeout / 分类错误 / 结构化失败：实现扎实，`src/reliability` 单元
  测试 + `tests/failure_injection` 已验证到位（LLM timeout/5xx/429、
  checkpoint 写失败、并发）。
- Checkpoint / Recovery：`src/durable` 的 crash→checkpoint→restart→resume
  循环有真实测试验证，是本项目质量最高的部分之一。
- Idempotency：正确实现且被验证为"防止副作用重复执行的真正机制"（甚至发现
  并记录了 checkpoint 失败时节点重放但副作用不重复执行的细节）。
- Concurrency：`RateLimiter`/`IdempotencyStore` 均有真实多线程测试验证正确
  性。
- **但以上全部只保护了那个"不调用 LLM 的 demo 图"**——真正的 Agent
  （`multi_agent_research`）没有 checkpointer、没有幂等包装，崩溃后无法恢
  复，Planner 抛出的裸 `ValueError`（结构化输出解析失败）会直接打穿整个
  `graph.invoke()`，这是一个已知且未修复的缺口（`docs/09` 第 12 条）。

**Operations**

- Health / Ready / Graceful shutdown：实现完整且经测试（`/health` 纯活性、
  `/ready` 检查 checkpointer + draining 状态、SIGTERM → drain in-flight
  runs）。
- Rate limiting：有效但进程内、单副本假设。
- **Scaling / Queue：完全空白**——没有队列、没有多副本一致性方案、没有对
  Postgres/Redis 的实际接入（只在架构文档里画了图）。
- 没有任何告警（Alerts）实现或配置——`docs/05` 只是 Dashboard Spec 文档，没有
  真实告警规则/阈值/接入 PagerDuty 等。

**SRE 结论**：可靠性工程手法（重试/超时/幂等/恢复）本身达到了不错的工程水
准，但**保护范围是错的对象**；生产运维意义上，"跑单实例、无扩容、无告警、只
有健康检查"离"服务真实用户"还差一大截。

---

## 4. Security Engineer 视角

- Layer 1（Prompt Injection 检测）、Layer 2（Workflow Guardrail：输出校验/
  敏感数据过滤）、Layer 3（Tool Guardrail：参数校验+HIGH风险审批）**三层防护
  全部有完整实现和至少 10 个攻击测试用例，工程质量高**。
- **但这三层防护没有被接入任何真实调用路径**——`src/api/app.py` 的
  `POST /runs` 直接把 `body.question` 传给 `registry.create_run`，中间没有
  任何 `guardrails.InputGuardrail`/`AgentWorkflowGuardrail`/`ToolGuardrail`
  的调用。这意味着：**如果明天真的把 `multi_agent_research` 接到这个 API
  上，用户输入会不经过 Prompt Injection 检测就直接进入 Planner 的 LLM 调
  用**。这是全部安全工作里最大的落地缺口，等同于"写了很好的门禁系统，但没有
  装在门上"。
- Authentication：Bearer token 静态白名单（`AuthConfig.valid_tokens`），没
  有 token 过期、没有多租户级别的 token 发放/吊销机制，生产环境级别偏弱
  （可接受作为 MVP，但需要标注）。
- Authorization / Tenant isolation：`Identity.tenant_id` 在 `RunRegistry`
  层做了资源级隔离，且用测试验证了跨租户查询返回 404 而非 403（防止租户存
  在性泄漏），设计正确。
- Secrets：`.env` 从不进镜像，`Settings.__repr__` 主动脱敏，`.gitignore` 覆
  盖 `.env`/`.venv`/`traces`，符合要求。
- PII / 敏感数据默认脱敏：`observability/logging.py` 有默认脱敏实现，但同样
  只在 API HTTP 层生效，**没有覆盖 Worker/Planner 生成内容里可能包含的
  PII**（因为图本身没有走 observability）。
- HIGH-risk 工具审批（Agent → approval → Tool）：`ApprovalStore` 实现完整、
  测试完整，但**没有任何 API 端点**（无 `/approvals`）暴露给真实运营/用户，
  只能在代码里手动调用——生产环境下无法真正走这个流程。

**Security Engineer 结论**：单项能力（每一层 guardrail）代码质量和测试覆盖
度都相当高，但**"纵深防御"的前提是防线真的部署在流量路径上**——目前它们是
并列在货架上的零件，还没有装到车上。这是一个 P0 级别的问题，不是"细节没做
好"，而是"核心防护完全没生效"。

---

## 5. AI Evaluation Engineer 视角

- `evals/` 目录结构完整：20 个任务的数据集（input/expected_behavior/
  success_criteria/risk_level）、`runner.py`（offline + regression）、
  `metrics.py`（11 个维度：质量/工具选择/参数/路由/终止/重试/安全/依据/引用/
  成本/延迟）、`baseline.py`/`regression.py`。评估方法论正确——不只比较最终
  字符串。
- `runner.py` 的 `agent_fn` 是可注入的，说明这套评估基础设施是
  **agent-agnostic 的骨架**，理论上能测任何 agent，但目前没有证据表明它曾
  经真正跑过 `multi_agent_research`（跑一次需要真实 API Key + 真实调用，属
  于"从未在 CI/发布流程里真正执行过"的状态）。
- 没有看到任何 CI 配置（`.github/workflows` 等）把"run eval → compare
  baseline → detect regression"接入到实际的代码/Prompt 变更流程里——文档承
  诺的"每次代码或 Prompt 修改都跑一次回归"目前是**手动、可选的**，不是强制
  门禁。
- Production feedback 闭环（线上真实失败案例回流到评估集）：未见任何实现或
  钩子。

**AI Evaluation Engineer 结论**：评估框架本身设计合理、维度全面，但**从未与
真实 Agent 或真实发布流程连接**，是"准备好了骨架，但从未真正运行过一次端到
端评估"的状态。

---

## 6. Product Engineer 视角

- Streaming：无。`POST /runs` 立即返回 202，用户只能轮询 `GET /runs/{id}`——
  对一个可能跑几十秒到几分钟的多智能体调研任务来说，缺少任何中间过程可见性
  （比如"Planner 已完成拆解，Worker A/B 完成，Worker C 进行中"这种进度体
  验），虽然 `GET /runs/{id}/state` 理论上能看到 `pending_tasks`/`trace`，
  但没有前端友好的进度模型或 WebSocket/SSE。
- Progress / Long-running task：设计上用轮询+超时降级 (`interrupted`) 处
  理，思路对，但对终端用户来说"interrupted"状态含糊——没有区分"临时超时可重
  试"和"永久失败"的用户可读提示。
- Failure message：`src/api/errors.py` 给出统一 `{request_id, error_code,
  message}` 结构，是好的基础，但目前"失败"这条链路对应的是 demo 图的失败，
  不是真实 Agent 失败（例如 Planner JSON 解析失败时的裸 `ValueError` 目前会
  变成什么样的用户可读错误？未经验证，因为这条路径根本没被 API 层触发过）。
- Resume：`POST /runs/{id}/resume` 实现且测试良好——这是本项目里最贴近"给
  用户一个可操作的恢复按钮"的功能。
- Human approval：`ApprovalStore` 存在，但没有 UI/API 暴露，用户侧完全感知
  不到"有一个高风险动作在等待审批"。

**Product Engineer 结论**：底层原语（resume、结构化错误、request_id）具备，
但缺少把它们组织成一个**用户能感知、能操作**的完整体验——尤其是进度可见性和
审批可见性两块基本是空白。

---

## Production Readiness Score

| 维度 | 分数 | 简述 |
|---|---|---|
| Architecture | **4/10** | 设计文档优秀，但真正服务的图（demo 图）与设计意图（multi_agent_research）脱节 |
| Reliability | **6/10** | 机制本身（retry/timeout/checkpoint/idempotency/concurrency）扎实且测试充分，但保护的不是真正的 Agent |
| Security | **3/10** | 三层 guardrail 完整实现但**零接入**真实请求路径；这是能直接导致"没有防护上线"的问题 |
| Observability | **3/10** | 日志/追踪基础设施质量高，但只覆盖 HTTP 外壳，Agent 内部（LLM 调用、token、每个节点耗时）完全不可观测 |
| Evaluation | **4/10** | 框架完整、维度正确，但从未针对真实 Agent 跑过、未接入发布门禁 |
| Scalability | **2/10** | 单实例、SQLite、纯内存限流/幂等/run 注册表，无法水平扩展 |
| Cost | **4/10** | Fast/Balanced/Quality 三档策略和预算模型设计良好，但只作用于 API 层超时，未真正控制 `multi_agent_research` 的 worker 数/迭代数/token 上限 |
| UX | **4/10** | 有 resume、有结构化错误，但无流式进度、无审批可见性 |

---

## P0 — 必须在上线前修复

1. **把真实 Agent（`multi_agent_research`）接入生产 API**，而不是让
   `src/api/main.py` 继续挂载不调用 LLM 的 durable demo 图——目前上线等于给
   用户提供一个假回答服务。
2. **把三层安全 Guardrail（Prompt Injection / Workflow / Tool）真正接入请
   求路径**（`POST /runs` 入口 + 每次 LLM 调用前后 + HIGH 风险动作前），而
   不是只停留在独立测试里——目前生产路径上没有任何 Prompt Injection 防护。
3. **给真正的 Agent 加上 durable checkpointer + 幂等包装**——目前
   `multi_agent_research` 一崩溃（进程重启/超时/异常）整个 run 全部丢失且不
   可恢复，Planner 的裸 `ValueError` 会直接打穿整条链路且无结构化失败兜底。
4. **RunRegistry 从纯内存改为可恢复存储**（或在启动时从 checkpointer 重
   建），否则 API 进程一重启，所有用户能看到的 run 状态（包括"是否完
   成"）全部丢失，即使底层执行数据还在。
5. **移除或修复单实例假设导致的扩容陷阱**：SQLite checkpointer + 进程内
   `RateLimiter`/`IdempotencyStore`/`RunRegistry` 意味着一旦水平扩容到 2 个
   副本，限流会翻倍失效、幂等会失效、run 状态会分裂——上线前至少要明确宣
   布"仅允许单副本部署"并在运维手册/健康检查里强制这一约束，否则运维扩容时
   会在不知情的情况下引入数据不一致。

## P1 — 应尽快修复

- 把 `src/observability`（tracing/metrics/token 统计）真正埋入
  `multi_agent_research` 每个节点和每次 LLM 调用，否则"哪个 Agent 最贵/哪个
  Tool 最容易失败"这类问题目前答不出来。
- 把 `src/cost` 的预算（MAX_AGENT_ITERATIONS/MAX_WORKERS/MAX_TOKENS/
  MAX_COST/TIMEOUT + 降级链路）真正接到 `multi_agent_research` 的
  Planner/Worker 调用参数上，而不是只作用于 API 层的整体超时。
- 为 Planner 的"结构化输出解析失败"补一个结构化失败/降级路径（分类为
  `BusinessValidationError` 并返回明确失败原因），而不是让裸 `ValueError`
  崩溃整条 Graph（`docs/09` 已记录该缺口）。
- 建立至少一条真实的告警规则（错误率/延迟/成本超阈值 → 通知），目前
  Dashboard Spec 只是文档，没有任何告警落地。
- 把 `evals/` 接入实际 CI/发布流程，让"改 Prompt/代码 → 强制跑一次 eval →
  对比 baseline"成为门禁而非可选步骤。

## P2 — 可以稍后改进

- 给 `POST /runs` 增加 SSE/流式进度端点，让长任务有真实的分阶段进度体验。
- 给 HIGH 风险工具审批增加真实的 `/approvals` API 端点，让审批流程可运营。
- Bearer token 认证升级为可吊销、可过期、按租户签发的方案。
- 把 `RunRegistry` 从内存索引升级为可从 checkpointer 全量重建（已在文档里
  承诺，尚未实现）。

## P3 — 有余力再做

- 把 `InMemorySaver`/SQLite 迁移到多副本可用的 Postgres/Redis 后端（架构文
  档已画出目标图，属于长期演进项）。
- 引入真实检索（RAG）为 Worker 的结论提供可验证依据，而不是让模型"自称有依
  据"。
- 探索真实 MCP 工具集成，替代当前"只有 LLM 文本生成、没有真实工具"的状态。

---

## 最终结论

### "这个 Agent 现在到底是不是 Production Ready？"

**不是。**

它更准确的状态是：**六套工程质量都不低的"生产能力组件"（Durable
Execution、Security Guardrails、Observability、Evaluation、Cost &
Latency、Deployment），从未被真正组装、接线到那个唯一有业务价值的 Agent
（`multi_agent_research`）身上**。今天如果直接把
`deployment/docker-compose.yml` 的 `agent-api` 服务发布给真实用户，他们得到
的是一个不调用任何 LLM、不做任何真实调研的占位符演示服务；而如果明天临时把
`multi_agent_research` 接上这套 API，它将在**没有 Prompt Injection 防护、没
有 checkpoint 恢复能力、没有 token/成本可观测、没有预算硬上限**的情况下直接
对外提供服务。

### 最关键的 5 个阻塞问题

1. **生产 API 挂载的不是真正的 Agent**——`src/api/main.py` 用的是不调用 LLM
   的 durable demo 图，`multi_agent_research` 从未上线。
2. **三层安全 Guardrail 完全没有接入请求路径**——Prompt Injection 检测、工
   具参数校验、HIGH 风险审批全部"束之高阁"，真实流量不会经过它们。
3. **真正的 Agent 没有崩溃恢复能力**——没有 checkpointer、没有幂等包装，
   Planner 的解析失败会直接崩溃整条链路且无结构化兜底。
4. **RunRegistry 是纯内存态**——进程一重启，用户能查询到的所有 run 状态全
   部丢失，与"支持恢复"的目标矛盾。
5. **架构不支持任何形式的水平扩容**——SQLite 单文件 checkpointer + 进程内
   限流/幂等/run 注册表，多副本部署会直接导致限流失效、幂等失效、状态分
   裂。
