# 03 — Durable Execution

本实验目标：让一次 Agent 图执行（LangGraph `StateGraph`）能够承受 **Worker 崩溃**而不丢失已完成的进度、不重复执行已经成功且具有副作用的操作，并支持**恢复（resume）**、**中断执行（interrupted execution）**与**查看执行历史（execution history）**。

```
crash -> checkpoint -> restart -> resume
```

代码：

```
src/durable/
├── __init__.py       # 包说明 + 汇总导出
├── state.py           # DurableResearchState：question / results / final_report / trace
├── graph.py            # research_a -> research_b -> research_c -> finalize（线性，便于验证"哪几步已提交检查点"）
├── checkpointer.py     # production-capable checkpointer 工厂（SQLite 文件持久化）
└── recovery.py         # CrashInjector/WorkerCrash + run_or_crash/resume + execution history

tests/test_recovery.py  # 9 个用例，覆盖本文档列出的全部场景
```

## 0. 为什么图是线性的 A → B → C，而不是像实验 4/5 那样并行 fan-out

需求给出的验证场景本身就是顺序的（"Research A 成功，Research B 成功，Research C crash"），并且要求"每个重要 graph step 都可以恢复"——用一条 `research_a -> research_b -> research_c -> finalize` 的直线，每一步都是**独立的 superstep**，可以用最少的机制把"哪几步已经提交了检查点"这件事说清楚、测清楚。并行 fan-out（实验 4）已经验证过 LangGraph 的并行调度本身；这里刻意不复用，避免把两个正交的问题（并行 vs 持久化）混在一起验证。

## 1. Checkpoint vs Memory vs Database vs Execution History

这四个词经常被混用，但在这个实验里分别对应完全不同的东西：

| 概念 | 是什么 | 生命周期 | 本实验中的实现 |
|---|---|---|---|
| **Checkpoint（检查点）** | LangGraph 在**每个 superstep 结束时**自动写入的、该线程（`thread_id`）在那一刻的**完整状态快照** + "接下来该跑哪个节点"的调度信息（`next`）。是 LangGraph 持久化机制的核心单元。 | 由 `BaseCheckpointSaver` 决定：`InMemorySaver` 随进程死亡而消失；`SqliteSaver`/`PostgresSaver` 随磁盘/数据库文件持久存在，可以跨进程重启存活。 | `graph.py` 里每个节点（`research_a`/`research_b`/`research_c`/`finalize`）返回后，LangGraph 自动为该 `thread_id` 落一个检查点；本模块不自己实现检查点逻辑，只依赖它。 |
| **Memory（内存 / InMemorySaver）** | 一种**checkpointer 的具体实现**：把检查点存进一个普通 Python `dict`，活在当前进程的堆里。它不是"长期记忆/RAG"意义上的 memory（那是完全不同的概念，参见 `docs/14-MULTI-AGENT-RESEARCH.md` 问题 9），这里特指 `langgraph.checkpoint.memory.InMemorySaver`。 | 进程退出（含崩溃）即彻底丢失；不能跨进程/跨副本共享。 | **明确禁止用于生产**（需求第 2 条）。`tests/test_recovery.py::TestInMemorySaverIsUnsuitableForCrashRecovery` 用它做反面对照：即使只是构造一个"新的" `InMemorySaver()` 实例（模拟"换一个进程"），之前崩溃线程的所有进度也彻底不可见——`get_state(...)` 返回空快照，而不是"回到 Research C 之前"。 |
| **Database（数据库 / 持久化后端）** | checkpointer 真正落地检查点数据的**存储介质**——本实验用文件形式的 SQLite（`langgraph-checkpoint-sqlite`），真实多副本生产环境应换成共享数据库（如 `langgraph-checkpoint-postgres`）。是"检查点"这个抽象概念之下的**具体持久化实现**。 | 独立于任何一个 Agent 进程：进程可以死、可以被替换，只要数据库文件/服务还在，检查点就还在。 | `checkpointer.py::sqlite_checkpointer(db_path)`：打开（或创建）一个 `.sqlite3` 文件；"重启"在测试里表现为对**同一个文件路径**打开一个全新的 `SqliteSaver`/`sqlite3.connect(...)` 连接（模拟真实重启时新进程重新连接同一个数据库）。 |
| **Execution History（执行历史）** | 某个 `thread_id` **所有历史检查点**按时间顺序组成的序列——不是"当前状态"，而是"这次运行是怎么一步步走到当前状态的"完整审计轨迹，即使从未崩溃过也存在、也可查。 | 与 Database 的生命周期相同（存在同一个持久化后端里），只是查询方式不同：`get_state(...)` 只给"现在"，`get_state_history(...)` 给"从现在到最早"的每一步。 | `recovery.py::get_execution_history()` 包装 `graph.get_state_history(config)`，返回每一步的 `next_tasks`（该检查点之后该跑哪个节点）+ `trace_so_far`/`results_so_far`（该检查点时刻的状态）。`tests/test_recovery.py::TestExecutionHistory` 验证了历史里 `results_so_far` 的键集合严格按 `{} -> {} -> {A} -> {A,B} -> {A,B,C} -> {A,B,C}` 递增（`__start__` 的一条 + `research_a` 的一条各贡献一次“尚未有结果”）。 |

一句话区分：**Checkpoint 是单位，Memory 是（不合格的）实现，Database 是（合格的）实现，Execution History 是对同一份持久化数据的"查全部/查最新"两种视图。**

## 2. Production-capable checkpointer：`SqliteSaver`（而不是 `InMemorySaver`）

`src/durable/checkpointer.py::sqlite_checkpointer(db_path)`：

```python
with sqlite_checkpointer("traces/durable_checkpoints.sqlite3") as checkpointer:
    graph = build_durable_graph(store, checkpointer=checkpointer)
    ...
```

- 底层是 `langgraph.checkpoint.sqlite.SqliteSaver`，包着一个真实的、写到磁盘文件的 `sqlite3.Connection`（`check_same_thread=False`，因为 LangGraph 的 Pregel 调度器可能从别的线程访问它；`SqliteSaver` 内部做了序列化保证这是安全的）。
- 测试里**从不**用 `sqlite3.connect(":memory:")`：内存态 SQLite 连接和 `InMemorySaver` 有一模一样的"随进程死亡而消失"的问题，用它测不出任何跨重启恢复的效果——所以每个测试用 `tmp_path` 下的真实文件。
- 生产多副本（多台机器/多个 worker 进程同时可能处理同一个 `thread_id`）场景应换成 `langgraph.checkpoint.postgres.PostgresSaver`（或任何共享数据库支持的 `BaseCheckpointSaver`）；`graph.py`/`recovery.py` 只依赖 `BaseCheckpointSaver` 接口本身，不依赖 SQLite，所以这个替换只需要改 `checkpointer.py` 一个文件。

## 3. `thread_id` 管理独立的 Agent executions

每次 `run_or_crash(graph, state, thread_id=...)` / `resume(graph, thread_id=...)` 都要求显式传入 `thread_id`；它被塞进 LangGraph 的 `config = {"configurable": {"thread_id": ...}}`。**同一个 checkpointer/数据库文件可以同时容纳任意多个互不干扰的 `thread_id`**——这正是"用 thread_id 管理独立 Agent executions"的落地方式：一个 `thread_id` 对应一次独立的、可恢复的执行会话（例如一次用户请求、一次后台任务），彼此的检查点互不覆盖、互不可见。

## 4. 崩溃模拟：`CrashInjector` / `WorkerCrash`

```python
crash_injector = CrashInjector()
crash_injector.arm("C", when="before")   # 或 "after"
```

- `WorkerCrash` 是一个**故意不在任何节点内部被捕获**的异常——真实的进程崩溃（OOM-kill、宿主机重启、`kill -9`）本身也不会被"捕获"，捕获它就失去了验证检查点恢复的意义。它会从 `graph.invoke(...)` 里原样抛出来。
- `"before"`：在该任务的副作用调用**之前**崩溃——模拟"worker 刚领到任务就死了，副作用完全没发生"。
- `"after"`：在副作用**已经执行完**、但节点返回值还没被 LangGraph 提交为检查点**之前**崩溃——模拟"副作用做完了，但确认/提交这一步丢失了"，这正是幂等性真正起作用的场景：节点重放时会重新调用同一个 `task_id` 的副作用，若不加幂等保护就会重复执行。
- 单次触发（single-shot）：崩溃触发一次后自动解除武装，代表"重启后的 worker 不会再次崩溃在同一个地方"。

## 5. 测试场景 → 需求逐条对照

`tests/test_recovery.py`：

| 需求 | 对应测试 |
|---|---|
| Research A 成功、Research B 成功、Research C crash | `TestCrashDuringResearchC::test_crash_before_c_side_effect_then_resume`：`effects.calls == {"A": 1, "B": 1, "C": 0}`，`get_pending_tasks(...) == ("research_c",)` |
| 恢复后 A 不重新执行、B 不重新执行 | 同上，`resume()` 之后 `effects.calls` 仍是 `{"A": 1, "B": 1, "C": 1}`（A/B 计数没有变化，只有 C 从 0 变成 1） |
| C 从合适的位置继续 | `resume()` 返回 `status == "completed"`，`results` 包含全部 `{"A","B","C"}`；同时 `test_a_and_b_checkpoints_are_committed_before_c_crashes` 直接断言崩溃那一刻的检查点里 `next_tasks == ("research_c",)` 且 `results_so_far` 已经有 `{A, B}` |
| 不重复执行已经成功且具有副作用的操作 | `test_crash_after_c_side_effect_then_resume_does_not_duplicate_side_effect`：C 的副作用在崩溃前已经真正执行过一次；节点重放后 `effects.calls["C"]` 仍然是 `1`（由 `reliability.idempotency` 去重，见下） |
| 支持 interrupted execution | `run_or_crash` 返回 `status="crashed"` 而不是让异常拖垮调用方；这就是"被中断的执行"的可观测状态——调用方可以据此决定何时/是否恢复 |
| 支持 resume | `recovery.resume(graph, thread_id)`：传入 `None` 作为输入，让 LangGraph 从最近一次检查点接着跑，而不是重新开始一次新的运行 |
| 支持查看 execution history | `TestExecutionHistory` 两个用例：`get_execution_history()` 返回按时间倒序的检查点列表，且这份历史本身是持久的——`test_history_survives_across_a_new_connection_to_the_same_file` 用全新连接重新打开同一个数据库文件依然能看到完整历史 |
| 使用 production-capable checkpointer；不再用 InMemorySaver 作为生产方案 | `TestInMemorySaverIsUnsuitableForCrashRecovery`（反面对照）+ 其余所有测试统一使用 `sqlite_checkpointer` |
| 每个重要 graph step 都可以恢复 | `TestExecutionHistory::test_history_shows_every_completed_step_in_order` 验证每一个 superstep（`__start__`/A/B/C/finalize）都各自产生了一条可查询的检查点 |

## 6. 端到端调用示例

```python
from src.durable import (
    sqlite_checkpointer, build_durable_graph, initial_state,
    CrashInjector, run_or_crash, resume, get_execution_history,
)
from src.reliability.idempotency import InMemoryIdempotencyStore

store = InMemoryIdempotencyStore()          # 生产环境应替换为持久化的幂等性存储
crash = CrashInjector()
crash.arm("C", when="after")                 # 仅用于演示/测试；生产代码不需要它

with sqlite_checkpointer("traces/demo.sqlite3") as checkpointer:
    graph = build_durable_graph(store, crash_injector=crash, checkpointer=checkpointer)
    outcome = run_or_crash(graph, initial_state("为什么天空是蓝色的？"), thread_id="demo-1")
    assert outcome.status == "crashed"       # Worker 崩溃

# "重启"：全新的 graph / checkpointer 对象，指向同一个数据库文件
with sqlite_checkpointer("traces/demo.sqlite3") as checkpointer2:
    graph2 = build_durable_graph(store, crash_injector=crash, checkpointer=checkpointer2)
    resumed = resume(graph2, thread_id="demo-1")
    assert resumed.status == "completed"     # 从检查点恢复，A/B 未重跑，C 正确完成

    for entry in get_execution_history(graph2, "demo-1"):
        print(entry.step, entry.next_tasks, list(entry.results_so_far))
```

## 7. 已知限制（与本项目其余 Reliability 文档一致的诚实边界）

- `InMemoryIdempotencyStore`（来自 `src/reliability/idempotency.py`）在这里被复用为副作用去重的存储，但它本身**不持久化**——真实生产环境中，副作用去重记录必须和检查点一样落在持久化存储里（例如数据库表 + 唯一约束），否则"worker 进程重启"会让幂等性存储和检查点一样清零，从而让 `after`-模式的崩溃恢复重新触发一次真实副作用。本实验的测试之所以能验证 `after` 场景的去重效果，是因为同一个 `store` 对象在"崩溃前"和"恢复后"两次 `with sqlite_checkpointer(...)` 之间被显式复用（模拟一个独立于 Agent 进程、持续存在的幂等性服务/数据库），而不是因为 `InMemoryIdempotencyStore` 本身是持久化的。
- 本实验没有实现真实的多进程/多机器崩溃（例如真正 `kill -9` 一个 Python 子进程）；`WorkerCrash` 是进程内异常注入，但由于 checkpointer 使用真实磁盘文件，"用全新对象重新打开同一个文件"这一步和真实进程重启在可观察行为上是等价的。
