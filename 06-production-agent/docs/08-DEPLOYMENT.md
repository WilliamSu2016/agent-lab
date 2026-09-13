# 08 — Deployment

本实验把当前 LangGraph Agent（`src/durable/` 的 Durable Execution 图，作为可部署
的参考图）包装成一个可部署应用：`deployment/langgraph.json` 定义本地开发用的
LangGraph CLI dev server，`src/api/` 是一套自研的生产级 FastAPI 应用，二者共享
同一个底层图和同一套 `src/durable` 持久化机制，但服务于两个不同的目的。

```
deployment/
├── langgraph.json     # LangGraph CLI dev server 配置：graph entrypoint / dependencies / env
├── Dockerfile          # 生产镜像：多阶段构建，非 root 运行，不含任何 secret
└── docker-compose.yml  # dev（langgraph dev）+ prod（本项目自研 API）两个 profile

src/api/
├── __init__.py
├── schemas.py     # Pydantic 请求/响应模型（Requirement 2：request validation）
├── auth.py        # Bearer token 认证 + identity propagation（Requirement 1）
├── rate_limit.py  # 进程内滑动窗口限流（Requirement 3）
├── errors.py      # 统一异常 -> 结构化 JSON 错误响应（Requirement 5）
├── runs.py         # run 注册表：thread_id 级生命周期，复用 src/durable
├── app.py          # FastAPI app 工厂：路由 + 中间件 + 生命周期
└── main.py         # 进程入口：python -m src.api.main

src/durable/graph_entrypoint.py  # langgraph.json 的 graph entrypoint 目标
tests/test_api.py                 # 15 个测试：crash/resume 全链路、认证、校验、限流、
                                   # 租户隔离、错误形状、request id、health/ready
```

## 0. Development vs Production：为什么是两套东西

| | Development | Production |
|---|---|---|
| 入口 | `langgraph dev`（LangGraph CLI 自带的本地 dev server），读取 `deployment/langgraph.json` | `python -m src.api.main`（本项目自研的 FastAPI 应用），Docker 化后由 `docker-compose.yml` 的 `agent-api` 服务运行 |
| 持久化 | CLI 自己管理的开发态存储（内存/临时文件），随进程重启可能丢失 | `src/durable/checkpointer.py::sqlite_checkpointer`，文件挂载在 `docker-compose.yml` 的具名 volume `checkpoints` 上，容器替换后依然存在 |
| 鉴权/限流/超时/... | 无——单开发者本机工具，从不对外暴露 | 完整 9 项要求（见第 2 节），面向真实流量 |
| 用途 | 快速迭代图本身的逻辑（`langgraph dev` 自带的可视化调试） | 真正对外提供服务的 API 契约（`POST /runs` 等） |

两者**共享同一份图逻辑**（`src/durable/graph.py::build_durable_graph`），只是
Development 通过 `src/durable/graph_entrypoint.py` 这个零参数模块级 `graph` 变量
（LangGraph CLI 的要求：它只能 import 一个现成的图对象，不能调用带参数的工厂函数）
拿到图，而 Production 通过 `src/api/main.py` 自己调用工厂函数、绑定真正的生产
checkpointer 连接。这不是两套逻辑，是同一套逻辑的两种装配方式。

## 1. `langgraph.json`

```json
{
  "python_version": "3.10",
  "dependencies": [".."],
  "graphs": { "durable_research": "../src/durable/graph_entrypoint.py:graph" },
  "env": "../.env",
  "dockerfile_lines": []
}
```

- **graph entrypoint**：`graphs` 里的键（`durable_research`）是 CLI 里这张图的名字，
  值是"文件路径:变量名"——指向 `graph_entrypoint.py` 里那个不带任何持久化的
  `graph`（见第 0 节：dev server 会替换/管理自己的持久化层）。
- **dependencies**：`[".."]` 表示"以仓库根目录作为一个可安装的本地依赖"——CLI 用它
  来定位 `requirements.txt`、`config/`、`src/` 等模块，路径都相对 `langgraph.json`
  自身所在目录（`deployment/`）解析。
- **environment variables**：`env` 指向仓库根的 `.env`（本地开发者自己创建、
  从不提交到 git，见 `.env.example`）——CLI 启动时会加载它，图内如果调用真正的
  LLM 需要 `OPENAI_API_KEY` 等变量时即可读到。

## 2. 生产 API：9 项交叉需求

`POST /runs` / `GET /runs/{id}` / `GET /runs/{id}/state` / `POST /runs/{id}/resume`
四个端点全部经过下面这 9 层——顺序即中间件/依赖的实际执行顺序：

| # | 需求 | 实现 | 位置 |
|---|---|---|---|
| 1 | Authentication | Bearer token，与配置的白名单做常数时间比较；`API_AUTH_TOKENS` 未配置时**拒绝启动**（不允许"忘记配置=无鉴权"） | `src/api/auth.py::AuthConfig` / `verify_bearer_token` |
| 2 | Request validation | 每个请求体都是 Pydantic 模型（`question` 非空、`mode` 只能是三选一）；不合法的请求在到达任何图节点之前就被 FastAPI 拒绝为 `422` | `src/api/schemas.py` |
| 3 | Rate limiting | 按 `user_id`（不是 IP）的滑动窗口计数器；超限返回 `429` + `Retry-After` | `src/api/rate_limit.py::RateLimiter` |
| 4 | Timeout | 每个 run 按其 `mode` 对应的 `src.cost.policy` 策略的 `budget.timeout_seconds` 复用 `src.reliability.timeout.run_with_timeout` 限时；超时的 run 状态变为 `"interrupted"`（可恢复），而不是让 HTTP 请求本身挂起 | `src/api/runs.py::RunRegistry._launch` |
| 5 | Error handling | 每种异常 -> 一种 `{request_id, error_code, message}` 响应；未预期的异常永远返回通用文案，完整细节只写入服务端结构化日志 | `src/api/errors.py` |
| 6 | Request ID | 中间件为每个请求生成/透传 `X-Request-ID`，绑定进 `Observability` 实验的 `ExecutionContext`，响应头和错误体里都能看到同一个值 | `src/api/app.py::request_context_middleware` |
| 7 | Health check | `GET /health`：永远快速返回、不查任何依赖（避免"数据库短暂不可用"被误判成"进程该被杀掉重启"） | `src/api/app.py` |
| 8 | Readiness check | `GET /ready`：真正探测 checkpointer 连接是否存活；关闭中的进程会先让它变 `503` | `src/api/app.py` / `src/api/main.py::readiness_probe` |
| 9 | Graceful shutdown | lifespan 的 shutdown 钩子：先标记"不再 ready"，再等待所有进行中的 run 任务收尾（`RunRegistry.drain`），最后才让进程退出；`docker-compose.yml` 给了 45s 的 `stop_grace_period` | `src/api/app.py::lifespan` / `src/api/runs.py::RunRegistry.drain` |

### 2.1 认证 vs 身份传播

“认证”（这是不是一个被允许调用本 API 的客户端）与“身份传播”（调用者到底是谁，
用于租户隔离/可观测性）是两个独立检查，不合并成一个：

- `Authorization: Bearer <token>` 只证明"这个客户端被允许说话"。
- `X-User-Id` / `X-Tenant-Id` / `X-User-Roles` 请求头显式携带调用者身份，构造成
  Security 实验里同一个 `src.security.authorization.Identity`（不是线程局部/全局
  的"当前用户"——见该模块文档：并发多租户场景下全局状态会串号）。

### 2.2 租户隔离体现在哪

`RunRegistry._resolve_owned`：查询一个不属于自己租户的 `run_id` 返回 `404`
（"不存在"），而不是 `403`（"存在但你无权访问"）——后者会向调用方泄露"这个 ID
确实是一个真实存在的、别的租户的 run"这一事实本身。

## 3. 四个端点与 Durable Execution 的对应关系

一次 HTTP "run" 就是 Durable Execution 实验里的一个 `thread_id`：

| 端点 | 复用的 `src.durable` API |
|---|---|
| `POST /runs` | `recovery.run_or_crash(graph, initial_state(question), thread_id)`，`thread_id` 就是新生成的 `run_id` |
| `GET /runs/{id}` | 读 `RunRegistry` 里缓存的 `RunRecord`（`status`/`final_report`/`error`） |
| `GET /runs/{id}/state` | `recovery.get_execution_history` + `recovery.get_pending_tasks`——完整 checkpoint 历史 + 恢复后会先跑哪个节点 |
| `POST /runs/{id}/resume` | `recovery.resume(graph, thread_id)`——已完成的节点（Research A/B）永远不会重跑，只继续没跑完的那个 |

`POST /runs` 立即返回 `202 Accepted`（状态 `"running"`），实际执行在后台线程
（`asyncio.to_thread`）里跑——这正是"支持 interrupted execution + 支持 resume"
这一对要求在 HTTP 层的样子：一个跑得比预期久的 run 不是让请求方的 HTTP 连接一直
挂着，而是让 run 本身进入一个可查询、可恢复的中间状态。

## 4. 测试场景：crash -> checkpoint -> restart -> resume，走 HTTP

`tests/test_api.py::test_crash_then_resume_does_not_rerun_completed_tasks`：

1. 给 `research_c` 军械一个"事后崩溃"（`CrashInjector.arm("C", "after")`）。
2. `POST /runs` 创建一次运行；轮询 `GET /runs/{id}` 直到状态离开 `"running"`——
   得到 `"interrupted"`。
3. `GET /runs/{id}/state` 确认 `pending_tasks == ["research_c"]`，并且三个任务的
   side-effect 调用计数是 `{"A": 1, "B": 1, "C": 1}`（C 的副作用在崩溃前已经真的
   执行过一次——这正是"崩溃发生在副作用执行之后、图提交结果之前"这个更难的场景）。
4. `POST /runs/{id}/resume`；轮询直到 `"completed"`。
5. 再次核对调用计数依然是 `{"A": 1, "B": 1, "C": 1}`——Research A/B 不重新执行，
   Research C 的副作用也没有因为节点被重新调度而重复执行一次（幂等性来自
   `src/durable/graph.py` 里 `reliability.idempotency.idempotent` 包装，checkpoint
   级别的"这个节点已完成"跳过是另一层，两层共同保证"不重复执行已经成功且具有
   副作用的操作"）。

## 5. Secrets 策略：镜像里永远没有 secret

- `deployment/Dockerfile` 从不使用 `ARG`/`ENV` 写入任何密钥——两者都会被烘焙进
  镜像层，`docker history`/`docker inspect` 能读到。
- `.dockerignore` 显式排除 `.env`/`.env.*`，即使未来某次改动把 `Dockerfile` 简化成
  `COPY . .`，`.env` 也永远不会进入 build context，更不会被拷进镜像。
- 真实密钥只通过 `docker run --env-file .env` 或 `docker-compose.yml` 的
  `env_file:` 在**容器启动时**注入——镜像本身，脱离运行环境去看，不含任何真实值。
- `config/settings.py::load_settings` 在进程启动的第一步就要求这些变量存在，
  缺失时立即 `ConfigurationError`（`src/api/main.py::main` 捕获后直接
  `SystemExit`，绝不会出现"容器启动成功了，但第一个真实请求才 500"的情况）。

## 6. 已知限制

- **`RunRegistry` 是进程内内存态**：一次真正的进程重启（不是本实验模拟的"新图对象"，
  而是整个容器被换掉）会丢失 `run_id -> RunRecord` 这份索引本身（尽管底层
  checkpoint 数据完好无损，因为它在 SQLite 文件里）。生产版本应当在启动时用
  `graph.get_state_history` 把所有已知 `thread_id` 重新枚举、重建索引——本实验
  没有实现这一步（同样的"进程内、非持久"限制也标注在
  `src.security.authorization.ApprovalStore`、`src.api.rate_limit.RateLimiter`
  里，是同一类问题）。
- **限流是单进程的**：多副本部署下每个副本各自维护自己的窗口计数，实际限额是
  "配置值 × 副本数"。真正的多副本部署需要把 `RateLimiter` 换成一个共享存储
  （如 Redis `INCR`+`EXPIRE`）实现，`src/api/app.py` 只依赖
  `RateLimiter.check(key)` 这一个接口，替换不需要改动其他任何文件。
- **SQLite checkpointer 是单机的**：与 Durable Execution 实验的结论一致——真正
  的多副本/多机部署需要把 `src/durable/checkpointer.py::sqlite_checkpointer`
  换成 `langgraph.checkpoint.postgres.PostgresSaver`（或其他共享数据库支持的
  `BaseCheckpointSaver`），`src/api/main.py` 只在一个地方（`graph_factory`
  内部）构造 checkpointer，替换同样是局部改动。
- **`langgraph dev` 未在本仓库里实际启动验证**（需要额外安装
  `langgraph-cli[inmem]`，且该工具版本/协议会独立于本项目演进）——
  `deployment/langgraph.json` 与 `src/durable/graph_entrypoint.py` 已经过
  静态验证（`graph_entrypoint.py` 可以被独立 import 并产出一个合法的
  `CompiledStateGraph`），但 docker-compose.yml 的 `langgraph-dev` 服务本身
  未在 CI/本次会话中实跑。
