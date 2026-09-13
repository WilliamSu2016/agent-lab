# Pattern 4 — Orchestrator-Workers

代码：`src/orchestrator_workers.py`
测试：`tests/test_orchestrator_workers.py`（18 个测试，全部通过）

## 1. 结构

```
User Question
     │
     ▼
Orchestrator (LLM, 无工具)
     │  输出: ResearchPlan { tasks: [WorkerTask, ...] }
     ▼
_assert_within_worker_cap(tasks)   ← 代码强制的硬上限 (MAX_WORKERS = 5)
     │
     ▼
asyncio.gather(  Worker(task_1), Worker(task_2), ..., Worker(task_N)  )   ← N 由 Orchestrator 动态决定
     │
     ▼
Synthesizer (LLM, 无工具)
     │
     ▼
Final Answer
```

`Orchestrator`、`Worker`（单一通用模板）、`Synthesizer` 都是普通 `Agent`，彼此之间**没有** `handoffs`。谁在什么时候调用谁，完全由 `run_orchestrator_workers()` 这段 Python 代码显式决定（`Runner.run` 调用序列 + `asyncio.gather`），而不是靠 SDK 的 Handoff 机制或 Agent-as-a-tool 把这层逻辑藏进某个 LLM 的工具调用循环里。

## 2. 为什么这不是简单的 Parallelization？（Q1）

对比 `src/parallelization.py`：

| | Parallelization（Pattern 3） | Orchestrator-Workers（Pattern 4） |
|---|---|---|
| 子任务数量 | 固定为 3（Python/TS/Go），写死在代码里 | 由 Orchestrator 在运行时决定，本次真实运行产出了 **5** 个任务 |
| 子任务内容 | 固定（"研究 Python 作为 AI Agent 开发语言"...），代码里的字符串常量 | 由 Orchestrator 根据用户问题现场生成，本次运行产出的任务是"技术架构分层"、"模型与推理技术栈"、"应用开发与运行时组件"、"生产基础设施与工程治理"、"安全合规与技术选型" —— 这些主题从未出现在代码中 |
| Researcher/Worker 的构建方式 | 每个语言一个专属 builder（`build_python_researcher` 等） | 只有一个通用 `build_worker()` 模板，任务内容通过*运行时的输入消息*注入，而不是写死在 instructions 里 |
| 并发 | asyncio.gather，是该模式的**核心目的**之一（原本就要"同时研究三种语言"） | asyncio.gather，只是**实现细节**——即使把 worker 改成顺序 `for` 循环执行，这仍然是 Orchestrator-Workers 模式；但换成顺序执行后 Parallelization 就不再是 Parallelization 了 |

真实运行的证据（问题："全面分析 2026 年 AI Agent 开发技术栈。"）：

```
ORCHESTRATOR PLAN (5 task(s), cap=5)
- [task-1] 系统梳理 2026 年 AI Agent 的整体技术架构与分层模型...
- [task-2] 分析 2026 年 AI Agent 的模型与推理技术栈...
- [task-3] 分析 2026 年 AI Agent 的应用开发与运行时组件...
- [task-4] 评估 2026 年 AI Agent 的生产基础设施与工程治理技术栈...
- [task-5] 分析 2026 年 AI Agent 的安全、合规与技术选型决策...
```

如果换一个完全不同的问题（例如"比较三种数据库"），Orchestrator 大概率会生成完全不同数量、不同内容的任务——这正是"动态任务分解"，而 Parallelization 无法做到这一点，因为它的分解在写代码那一刻就已经固定了。

## 3. 谁决定 Worker 数量？（Q2）

**两方共同作用，但最终否决权在代码：**

1. **Orchestrator（LLM）提出建议**：它读取问题后，输出一个 `ResearchPlan`，其中 `tasks` 列表的长度就是它认为"应该有多少个 Worker"。这是模型的自由裁量。
2. **代码（`_assert_within_worker_cap`）做最终裁决**：无论 Orchestrator 想生成多少个任务，`src/orchestrator_workers.py` 都会在派发任何 Worker 之前调用这个函数检查 `len(tasks) <= MAX_WORKERS`（=5）。超出上限就直接抛出 `OrchestratorWorkersStepError(step_name="orchestrator", ...)`，一个 Worker 都不会被派发。

`tests/test_orchestrator_workers.py::WorkerCapEnforcementTest.test_pipeline_refuses_to_dispatch_workers_when_orchestrator_exceeds_the_cap` 验证了这一点：脚本化一个 6 任务的计划，管道在派发任何 worker 之前就失败，且失败被精确定位到 `"orchestrator"` 这一步。

## 4. 谁决定 Worker 任务？（Q3）

完全由 Orchestrator（LLM）决定，且**只能**由它决定——代码里没有任何硬编码的任务列表、任务模板或关键词分支。`build_worker()` 是唯一的、通用的 Worker 构建函数，它的 `instructions` 里只说"你会收到一个具体的研究任务，请完成它"，从不提及任何具体主题。任务内容全部通过运行时的输入消息传给 Worker（见 `_run_worker()`），这保证了：

* 换一个完全不同领域的问题，代码不用改一行，Worker 依然能拿到与新问题匹配的任务。
* 测试 `DynamicTaskCountTest` 里用 2 任务、1 任务、3 任务三种不同计划分别验证了 Worker 数量和任务内容都严格跟随 Orchestrator 的输出变化，而不是被写死。

`WorkerResult` 里的 `task_id`/`objective` 字段是**代码**从原始 `WorkerTask`（闭包捕获）里附加上去的，而不是让 LLM 在自己的输出里重复回显 `task_id`——避免了模型可能拼错或遗漏 ID 导致结果对不上号的问题。

## 5. 哪些控制流是 LLM 决定的？（Q4）

| 决策 | 谁决定 |
|---|---|
| 任务数量（提议值） | LLM（Orchestrator） |
| 任务内容/研究角度 | LLM（Orchestrator） |
| 每个 Worker 具体研究出的结论 | LLM（Worker） |
| 最终综合答案的措辞与结构 | LLM（Synthesizer） |
| 任务数量是否被接受（硬上限检查） | **代码**（`_assert_within_worker_cap`） |
| 空计划（0 个任务）是否被接受 | **代码 + Schema**（`min_length=1`） |
| Worker 何时/是否并发执行 | **代码**（`asyncio.gather`，固定的 Python 实现细节） |
| 整个四阶段的执行顺序（Orchestrator → cap check → Workers → Synthesizer） | **代码**（`run_orchestrator_workers()` 的函数体，永远是这个顺序，LLM 无法改变或跳过任何一步） |
| 每一步失败后如何报告 | **代码**（`OrchestratorWorkersStepError` 精确标注 `"orchestrator"` / `"worker-<task_id>"` / `"synthesizer"`） |

也就是说：LLM 决定"内容"（有多少个任务、每个任务是什么、每个任务的结论是什么、最终怎么措辞），而"流程骨架"（谁在什么时候被调用、上限如何强制、失败如何报告）永远由这个模块自己的 Python 代码决定，且不可被任何 Prompt 或模型输出绕过。

## 6. 为什么最大 Worker 数量必须由程序强制？（Q5）

这是本模式实现中刻意做出的一个设计决定：`ResearchPlan.tasks` 的 Pydantic 定义**故意没有** `max_length` 约束（只保留 `min_length=1`，那是"结构正确性"层面的约束，与"资源安全"层面的约束性质不同）。上限完全由独立的 `_assert_within_worker_cap()` 函数在代码里强制执行。原因：

1. **Prompt 指令不是保证**：即使在 `instructions` 里写"最多生成 5 个任务"，模型仍可能因为理解偏差、长问题、追求"更全面"而生成 6、8 甚至更多任务——这是自然语言指令的固有不可靠性。
2. **Schema 提示也不是保证**：本项目使用的是私有网关的 Chat Completions API（非原生 Structured Outputs / 非 OpenAI Responses API 的强约束 JSON Schema 强制模式），`output_type` 更多是"解析和验证"的手段，而不是能在生成阶段就物理限制模型输出长度的护栏。即使换成支持 JSON Schema `maxItems` 强约束的网关，也不该把"防止资源失控"这种安全关键的逻辑，完全托付给某个第三方推理服务对 schema 的实现细节和版本行为。
3. **失控 Worker 数量的实际代价是真实的**：每多一个 Worker 就多一次真实的模型调用（+ 可能的 `search_web` 网络调用），在真实场景中意味着更多的延迟、token 成本、以及对下游服务（如搜索 API）的请求量——这是一个必须由开发者能够 100% 确定其上界的资源边界，而不能依赖"模型大概率会听话"。
4. **代码作为唯一可审计、可测试、可强制回归的机制**：`_assert_within_worker_cap()` 是一个纯函数，可以在没有任何 LLM 参与的情况下用单元测试直接验证其边界行为（见 `WorkerCapEnforcementTest`），这种确定性是 Prompt/Schema 都无法提供的。

## 7. 什么情况下这个模式比单 Agent 更好？（Q6）

* **问题的分解方式无法提前预知**：用户可能问"全面分析 2026 年 AI Agent 技术栈"，也可能问"比较三个数据库"，也可能问一个非常窄的问题只需要 1 个角度。单 Agent（或 Parallelization 里那种写死 3 个子任务的方式）无法适应这种范围和结构都不确定的输入；Orchestrator-Workers 把"任务应该怎么拆"这个决策本身也交给 LLM，使系统能对**任意粒度、任意主题**的开放式研究问题做出合理的分解。
* **子任务之间相互独立、可并行**，且各自需要相对聚焦、单一职责的处理（如果直接把整个复杂问题丢给一个 Agent，它要在一次很长的推理里同时兼顾"分层架构"、"模型选型"、"生产运维"、"安全合规"等差异很大的关注点，容易出现内容混杂、深度不足、上下文过长等问题）。
* **需要一个硬性资源上限，但又不想放弃"任务数量应由内容驱动"的灵活性**：如果任务数量是固定的（比如永远 3 个），就应该用更简单的 Parallelization；只有当"应该有几个任务"本身也需要由 LLM 依据问题现场判断时，才需要 Orchestrator-Workers 这一整套"提议 + 硬上限校验 + 动态派发 + 汇总"的机制。

反过来，**不应该使用**这个模式的情况：问题本身范围很窄、只需要一个视角就能回答（此时单 Agent 或 Prompt Chaining 更便宜、更可预测、延迟更低）；或者子任务其实是固定已知的（此时应该用 Parallelization，省去一次 Orchestrator 调用和一层不确定性）。

## 8. 这是 Workflow 还是 Agent？

参考 `docs/PATTERN-TAXONOMY.md` 的定义：Orchestrator-Workers 仍然是一个 **Workflow**（预定义的代码路径：Orchestrator → cap check → Workers → Synthesizer，这个骨架本身不会因输入而改变），但它是五种模式里**最接近 Agent** 的一种——因为"需要做多少个子任务、每个子任务具体是什么"这个通常需要人类工程师提前想清楚并写进代码的决策，在这里被下放给了 LLM 在运行时动态产生。区别于真正的 Autonomous Agent 的地方在于：这个模块自己的代码始终掌握着"流程什么时候往下走"、"上限是多少"、"失败了该怎么报告"这些控制权，LLM 不能自己决定要不要再多派几个 Worker、要不要跳过 Synthesizer、或者递归地再造一层 Orchestrator——它只被允许在一次性的 `ResearchPlan` 输出里表达"内容层面"的意见。

## 9. 测试要点（`tests/test_orchestrator_workers.py`，18 tests）

* Agent 定义：Orchestrator 无工具 + `output_type=ResearchPlan`；Worker 唯一通用模板 + `search_web` 工具 + `output_type=WorkerFindings`；Synthesizer 无工具、纯文本输出；三者均无 `handoffs`。
* 硬上限：`MAX_WORKERS=5` 在边界值（5 个任务）被接受，超过（6 个）被 `_assert_within_worker_cap` 拒绝；空计划也被拒绝。
* 动态任务数量：分别用 1、2、3 个任务的计划验证实际派发的 Worker 数量精确匹配计划长度（证明数量不是写死的）。
* Worker 隔离：每个 Worker 的输入只包含自己的任务内容，不包含其他任务的关键词（通过 `ModelStep.respond` 断言实现）。
* 并发派发：人为给每个 Worker 加 0.15s 延迟，验证 3 个 Worker 总耗时远小于 `3 × 0.15s`，证明是真并发而非顺序执行。
* 失败定位：Orchestrator 输出非法 JSON → `step_name == "orchestrator"`；某个 Worker 输出非法 JSON → `step_name == "worker-<task_id>"`；Synthesizer 调用失败 → `step_name == "synthesizer"`；空问题被拒绝。

## 10. 真实运行记录

```
.\.venv\Scripts\python.exe -m src.orchestrator_workers "全面分析 2026 年 AI Agent 开发技术栈。"
```

Orchestrator 动态生成了 **5 个任务**（架构分层 / 模型与推理技术栈 / 应用开发与运行时组件 / 生产基础设施与工程治理 / 安全合规与技术选型），全部并发派发给 5 个 Worker（均调用了 `search_web`，本次搜索接口未返回有效结果，Worker 诚实地说明"基于公开架构规范和主流工程模式的系统归纳，具体产品能力仍需在落地前复核"，而不是编造），最终 Synthesizer 把 5 份结果汇总成一份包含架构分层、技术选型建议、演进方向和结论的完整报告。这组任务从未出现在 `src/orchestrator_workers.py` 的代码里，证明分解确实是由 Orchestrator 在运行时依据实际问题动态产生的。
