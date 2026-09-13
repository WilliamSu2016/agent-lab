# 实验 4：Parallel Multi-Agent Research（LangGraph）

## 0. 架构

```
                 Planner
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   Research A   Research B   Research C   (动态数量, fan-out)
        │           │           │
        └───────────┼───────────┘
                    ▼
               Synthesizer          (fan-in)
```

## 目标

任务："全面研究 AI Agent Framework。"

流程：

```
START
  ↓
Planner            —— 动态决定需要研究哪些方面（不写死）
  ↓ (conditional edge 返回 list[Send]，即 fan-out)
Worker × N          —— 每个 Worker 只负责自己的一个 Task，并行执行
  ↓ (LangGraph 等待所有已派发的 Worker 完成，即 fan-in)
Synthesizer         —— 汇总所有 ResearchResult，生成最终报告
  ↓
END
```

代码：`src/04_parallel_multi_agent.py`
测试：`tests/test_parallel_multi_agent.py`（21 个，全部离线，使用可注入的假 `TextLLMCall`，不需要网络/API Key）

## 实现要点

- **`StateGraph`**：`ParallelResearchState` 包含 `topic`、`tasks`、
  `results: Annotated[list[ResearchResult], operator.add]`（reducer 用于安全合并并发分支的结果）、
  `max_workers`、`per_worker_timeout_seconds`、`report`。
- **Planner 节点**：调用 LLM 提出研究方面（aspects），解析为 JSON 数组，解析失败时退化为按行拆分；
  然后 `capped = aspects[:max_workers]` 强制截断（Requirement 9）。
- **Fan-out**：`add_conditional_edges("planner", fan_out_to_workers, ["worker"])`，
  `fan_out_to_workers` 对每个 task 返回一个 `Send("worker", {"task": task, ...})`——
  每个 `Send` 只携带**它自己的** task，不携带完整 `tasks` 列表或其他 Worker 的信息。
- **Worker 节点**：用 `_run_with_timeout(...)`（daemon 线程 + `join(timeout)`）执行真正的
  LLM 调用并强制超时；无论超时还是普通异常都在节点内部捕获，返回 `status="timeout"/"failed"` 的
  `ResearchResult`，**从不向外抛出异常**（Requirement 11）。
- **Fan-in**：所有 `worker` 实例都有同一条静态边 `add_edge("worker", "synthesizer")`；
  因为 `results` 字段带 reducer，LangGraph 会在所有已派发的 `worker` 完成后，把结果合并成一份
  完整 state，只调用一次 `synthesizer`（不需要手写 join 逻辑）。
- **超时实现**：单 Worker 超时和整体超时都通过 `_run_with_timeout(func, timeout_seconds)`
  实现——它用 `threading.Thread(daemon=True)` + `thread.join(timeout_seconds)`，超时未完成则
  抛出 `TimeoutError`，**而不是** `concurrent.futures.ThreadPoolExecutor` + `future.result(timeout=...)`。
  原因见下面"实测证据 4"：这不是风格选择，而是修复了一个真实跑通后才暴露的 bug。
- **无 MCP / 无 Memory**：`compile()` 不传 checkpointer，state 只存在于一次 `invoke()` 调用中。

## 实测证据（本次会话中用独立脚本 + 真实 API 验证）

1. **并发性**：4 个 Worker 各 `time.sleep(1.0)`，用 `Send` 并行派发后总耗时 ~1.0s，而不是 ~4.0s——
   证明 LangGraph 自己的 Pregel 调度器并发执行同一 superstep 内的多个 `Send` 分支，
   不需要手写 `asyncio.gather` 或线程池来做编排。
2. **异常传播**：3 个并行分支中，只要有 1 个抛出未捕获异常，整个 `graph.invoke()` 都会失败，
   其余分支即便已经算完也拿不到结果——这直接证明了 Requirement 11（Worker 必须自己吞掉异常）
   不是可有可无的最佳实践，而是防止"一个 Worker 挂掉、全部白干"的硬性要求。
3. **Fan-in 只触发一次**：用假 LLM 调用跑通整个 graph，`synthesizer` 节点被调用的次数恒为 1，
   且能看到全部 Worker 的结果（`test_synthesizer_runs_exactly_once_after_all_workers_fan_in`）。
4. **`ThreadPoolExecutor` 超时的一个真实陷阱（用真实 API 跑通后才发现）**：最初用
   `ThreadPoolExecutor(max_workers=1)` + `future.result(timeout=...)` 实现超时，离线测试
   （用 `time.sleep` 模拟慢调用）全部通过。但用真实 OpenAI API 跑一次完整实验
   （`全面研究 AI Agent Framework。`）时，程序逻辑本身在预算内正确完成
   （单 Worker 30s 超时、整体 90s 超时都按预期触发/未触发），但**整个进程却卡了约 360 秒才退出**。
   原因：`concurrent.futures` 模块会为每个创建过的线程注册一个全局 `atexit` 钩子
   （`concurrent.futures.thread._python_exit`），解释器退出前会 join **所有**曾经创建过的
   worker 线程——即使某个 executor 实例已经被 `shutdown(wait=False)`。也就是说，只要有一个
   还在挂起的真实网络请求线程没结束，进程退出就会被卡住，超时逻辑返回的"正确结果"和
   "进程什么时候真正退出"是两回事。修复方法：改用 `threading.Thread(daemon=True)` 手写等待
   （`_run_with_timeout`），daemon 线程不会被 `atexit` 钩子等待，进程可以立刻退出。
   修复前 vs 修复后（同一条真实请求，进程整体耗时）：
   - 修复前：程序输出全部正确，但进程约 **360s** 后才退出。
   - 修复后：进程约 **44s** 就完整退出。
   这条经验同样适用于任何"用线程池模拟超时/取消"的场景：Python 线程无法被强制杀死，
   只要还残留非 daemon 线程，进程退出就可能被无限期拖住。

## 问题解答

### 1. 这和普通 Parallelization pattern 有什么不同？

普通 Parallelization（例如 `04-langgraph` 里的 `parallelization.py`）通常是：**任务集合是固定的**——
开发者提前写死"翻译成中文、日文、法文"这三个分支，Graph 结构（节点数量）在编译时就已确定。

这里的关键区别：**任务集合本身是运行时由 LLM 决定的**。Planner 先"思考"这次要研究哪些方面
（可能 3 个，可能 6 个，内容也完全由 LLM 决定，代码里不出现任何具体框架名），
然后再对这个**动态生成的列表**做 fan-out。也就是说：

- 传统 Parallelization：并行执行**已知、固定**的几个分支。
- 本实验：先用一个 Agent **决定要并行做什么**，再并行执行**未知数量、未知内容**的分支。

这一步"决定做什么"本身就是一次独立的智能决策，而不只是把预先设计好的任务铺开执行。

### 2. 为什么现在叫 Multi-Agent？

因为每一个 Worker 不是一个无状态的纯函数调用，而是一个**独立的 Research Agent 实例**：

- 它有自己的角色/指令（研究某一个具体方面）；
- 它独立调用 LLM，独立做决策、独立处理失败/超时；
- 它的输入输出是结构化的（`ResearchTask` → `ResearchResult`），与其他实例完全隔离；
- 多个这样的 Agent 实例同时存在、同时"思考"，彼此互不可见。

如果只是把一个固定的、非 LLM 的函数（比如"调用翻译 API"）并行跑几次，那只是普通并行计算。
这里并行的是**多个会自主决策的 Agent**，所以称为 Multi-Agent，而"并行"只是它们的调度方式。

### 3. Worker 是否共享同一个 State？

**不共享。** 每个 Worker 通过 `Send` 收到的 payload 只包含它自己的 `task`（以及一些只读的公共参数，
如 `per_worker_timeout_seconds`），既看不到完整的 `tasks` 列表，也看不到其他 Worker 的
`ResearchResult`。

`ParallelResearchState` 里唯一"共享"的字段是 `results`，但这种共享是**单向、事后的**：
每个 Worker 只**写入**自己那一条结果（一个只含单元素的 list），LangGraph 用
`operator.add` reducer 把所有分支产生的这些小 list 拼接成完整列表——这发生在 Worker 执行**之后**、
fan-in 时，Worker 执行期间彼此互不可见、互不干扰。

### 4. Worker 之间应该互相通信吗？

**不应该。** 这正是本模式追求的隔离性（isolation）：

- 通信意味着耦合，会重新引入竞态条件（谁先写、谁先读）；
- 通信也意味着一个 Worker 的延迟/失败可能级联影响另一个 Worker；
- 每个 Worker 的任务本身就应该是可以独立完成的子问题（否则 Planner 拆分任务的方式就有问题）。

如果某个 Worker 确实需要另一个 Worker 的结果，那说明这两个任务之间有依赖关系，
应该在 Planner 阶段就把它们规划成串行步骤，或者放到 Synthesizer 阶段统一处理，
而不是让并行执行中的 Worker 互相打听彼此的进度。

### 5. 为什么 Fan-in 是必要的？

两个原因：

1. **业务需求**：Synthesizer 的职责是"汇总所有方面研究结果，产出一份完整报告"——
   它必须等到**全部** Worker 完成（无论成功还是超时/失败）才能开始工作，否则报告会缺内容。
2. **并发安全**：多个 Worker 并发地往同一个 `results` 字段写入部分结果，如果没有 fan-in
   和 reducer 机制，就需要手写锁/合并逻辑来避免结果覆盖或丢失。LangGraph 的 fan-in
   （所有 `Send` 派发的节点共享同一条到下游的边 + reducer）把"等待所有分支完成再合并"
   这件事变成了框架保证的行为，而不是应用代码要自己操心的细节。

### 6. 如何防止 Worker 数量失控？

在 Planner 节点里做**强制截断**：

```python
capped = aspects[: state["max_workers"]]
```

无论 LLM 提出多少个研究方面（哪怕是 20 个），只有前 `max_workers`（默认 `MAX_WORKERS = 6`）个
会被转换成实际的 `Send` 派发，多出来的会被直接丢弃，不会产生对应的 Worker。
这一限制被放在 fan-out 之前（Planner 节点本身），而不是寄希望于 LLM"自觉"，
是因为 LLM 的输出不可信任、不可控——必须用确定性代码兜底。
`tests/test_parallel_multi_agent.py` 里的
`test_planner_caps_tasks_at_max_workers` / `test_more_aspects_than_max_workers_are_truncated_requirement_9`
专门验证了这一点。

## 小结对照表

| 维度 | Handoff（实验 2） | Agents-as-Tools（实验 3） | Parallel Multi-Agent（本实验） |
|---|---|---|---|
| 控制权 | 转移给被 Handoff 的 Agent | 始终在 Supervisor | 无中心控制者，Planner 决策后由框架调度并行执行 |
| 执行方式 | 串行、单路径 | 串行、Supervisor 决定调用顺序 | 并行 fan-out，之后 fan-in |
| Agent 数量 | 固定（Triage + 2 个 Specialist） | 固定（Supervisor + 3 个 Specialist） | 动态（由 Planner 决定，且有上限） |
| 状态共享 | 单一对话上下文顺延 | Specialist 输出返回给 Supervisor，不互相可见 | Worker 之间完全隔离，只在 fan-in 后共享结果 |
