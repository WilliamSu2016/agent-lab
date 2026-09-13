# Agent Lab

一个循序渐进的学习型 Agent 实验仓库：从最小 Agent Loop 开始，逐步学习 OpenAI Agents SDK、常见 Agent 架构模式、LangGraph、多 Agent 协作，以及生产环境所需的可靠性、安全性、可观测性、评测与部署能力。

## 学习路线

| 阶段 | 目录 | 重点内容 |
| --- | --- | --- |
| 01 | [`01-minimal-agent`](01-minimal-agent) | 手写最小 Agent Loop、工具调用、提示词与基础评测 |
| 02 | [`02-openai-agents-sdk`](02-openai-agents-sdk) | 使用 OpenAI Agents SDK 重构 Agent、Runner 与 Tracing |
| 03 | [`03-agent-patterns`](03-agent-patterns) | Prompt Chaining、Routing、Parallelization、Orchestrator-Workers、Evaluator-Optimizer |
| 04 | [`04-langgraph`](04-langgraph) | Graph、State、条件路由、Agent Loop、Persistence、Human-in-the-loop |
| 05 | [`05-multi-agent`](05-multi-agent) | Specialist Agents、Handoff、Supervisor、并行协作、共享状态与研究团队 |
| 06 | [`06-production-agent`](06-production-agent) | 配置、重试、超时、幂等、持久化、安全、观测、成本、评测与部署 |

每个阶段都包含自己的 `src/`、`tests/`、`requirements.txt`，并尽量配有 `prompt/` 与 `docs/` 学习材料。建议按编号顺序学习，但也可以直接进入感兴趣的主题。

## 环境要求

- Windows PowerShell
- Python 3.10.8（仓库中的阶段性开发约定）
- 一个 OpenAI 或 OpenAI-compatible API
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- 可选：`OPENAI_BASE_URL`，用于指定兼容 OpenAI Chat Completions 协议的网关

不要把密钥提交到版本库。将环境变量写入对应实验目录的 `.env` 文件，或在当前 PowerShell 会话中设置：

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_MODEL = "your-model-name"
# 使用兼容网关时设置；官方 OpenAI API 可省略
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
```

## 快速开始

以第一个实验为例：

```powershell
cd 01-minimal-agent
C:\Python310\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.main "What are the key ideas behind tool-using agents?"
```

其他阶段的通用运行方式相同：进入对应目录、创建并使用该目录下的 `.venv`，然后运行入口模块：

```powershell
cd ..\02-openai-agents-sdk
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.main "Your research question"
```

`03-agent-patterns` 和 `04-langgraph` 也可以从各自的 `src.main` 启动基础示例。更具体的模式示例位于对应目录的 `src/` 中，例如：

```powershell
# 在 03-agent-patterns 目录执行
.\.venv\Scripts\python.exe -m src.routing "Classify this question"

# 在 04-langgraph 目录执行
.\.venv\Scripts\python.exe -m src.01_minimal_graph
```

第 05 阶段的主要研究 Agent 入口为：

```powershell
# 在 05-multi-agent 目录执行
.\.venv\Scripts\python.exe -m src.multi_agent_research.main
```

## 测试

各实验的测试相互独立。安装对应目录的依赖后，在该目录执行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

第 01、02、03、04 阶段还提供 `tests/evaluations/` 下的离线评测用例；第 06 阶段包含更完整的单元测试、集成测试、安全测试、故障注入和评测工具。

## 生产 Agent

第 06 阶段将前面学习的能力整合为可运行的 Production Research Agent，包括：

- Planner → Researcher(s) → Synthesizer → Reviewer 的 LangGraph 工作流
- Checkpoint、恢复、超时、重试与幂等
- 输入/输出 Guardrail、工具策略、授权与高风险操作审批
- 结构化日志、Tracing、Metrics 与成本/延迟预算
- 离线评测、回归评测与故障注入
- FastAPI 服务、Docker 与 LangGraph 部署配置

推荐先阅读 [`06-production-agent/README.md`](06-production-agent/README.md) 和 [`06-production-agent/docs`](06-production-agent/docs)，再运行测试：

```powershell
cd 06-production-agent
C:\Python310\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

其中 [`06-production-agent/production-research-agent`](06-production-agent/production-research-agent) 是独立的、更加完整的生产研究 Agent 实现，拥有自己的依赖、API、部署配置和文档。

## 目录约定

```text
<stage>/
├── src/           # 实验代码与入口
├── tests/         # 测试与评测
├── prompt/        # 学习过程中的任务提示词
├── docs/          # 设计说明、复盘与架构文档
├── requirements.txt
└── .venv/         # 本地虚拟环境，不提交到版本库
```

## 推荐阅读顺序

1. 先运行 `01-minimal-agent`，理解一次 Agent 请求的完整生命周期。
2. 在 `02-openai-agents-sdk` 中对比 SDK 抽象与手写实现。
3. 在 `03-agent-patterns` 中学习不同任务对应的控制流。
4. 在 `04-langgraph` 中把控制流显式建模为 Graph 和 State。
5. 在 `05-multi-agent` 中学习角色拆分、交接和协作。
6. 最后阅读 `06-production-agent/docs`，理解如何将 Demo 演进为可运维系统。

## 注意事项

- 真实 API 调用可能产生费用；测试优先使用离线 mock 或 scripted model。
- 不同阶段的依赖版本可能不同，请在对应阶段的虚拟环境中安装依赖。
- `.env`、`.venv/`、Tracing 数据、缓存、SQLite 数据库和 Python 编译产物已加入根目录 `.gitignore`。
- 这是学习实验仓库，生产使用前仍需根据实际的身份认证、数据隔离、密钥管理、监控和部署环境进行审查。
