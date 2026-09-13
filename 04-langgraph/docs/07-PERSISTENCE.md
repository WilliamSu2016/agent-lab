# 07 - LangGraph Persistence（Checkpointer）

对应代码：`src/07_persistence.py`
对应测试：`tests/test_persistence.py`

本实验在实验 5 的 Research Agent（`planner -> agent <-> tools -> finalize`，
带 Agent Loop 的 Conditional Edge）基础上，加上 LangGraph **当前推荐的
checkpointer 机制**，让 Agent 执行过程中的 `State` 被持久化保存下来。

Graph 结构完全不变：

```
START
  |
  v
planner
  |
  v
agent  <---------------------.
  |                          |
  v                          |
should_continue               |
  |-- "tools" --> tools ------'
  `-- "finalize" --> finalize --> END
```

变化只发生在**编译和调用方式**上：

```python
checkpointer = build_memory_checkpointer()          # InMemorySaver
graph = build_graph(text_llm_call, agent_llm_call, checkpointer=checkpointer)

config_a = thread_config("thread-a")                # {"configurable": {"thread_id": "thread-a"}}
result_a = graph.invoke(initial_state("问题 A"), config=config_a)

config_b = thread_config("thread-b")
result_b = graph.invoke(initial_state("问题 B"), config=config_b)
```

## 使用的机制

* **Checkpointer**：`from langgraph.checkpoint.memory import InMemorySaver`
  ——这是 LangGraph 目前推荐的 checkpointer 接口（`BaseCheckpointSaver`）
  的内存实现。它把每一次 super-step 后的完整 `State` 快照存进一个普通
  Python 字典里，按 `thread_id`（以及每个 thread 内部的 `checkpoint_id`）
  分区存储。不依赖 Redis、PostgreSQL 或任何外部数据库；进程退出后数据
  即丢失——这正是"先用内存实现"这一要求要验证的最小可行方案。若要换成
  持久化到磁盘/数据库的实现，只需把 `checkpointer=` 换成
  `PostgresSaver`/`SqliteSaver` 等，图的定义和 Node 代码完全不用改。
* **Thread ID**：`config={"configurable": {"thread_id": "..."}}`。每个不同
  的 `thread_id` 拥有自己独立的一条 checkpoint 历史；同一个
  `thread_id` 的多次调用会在同一条历史上继续追加。
* **`graph.get_state(config)`**：不重新执行图，只读取某个 thread 当前最新
  的 checkpoint（返回一个 `StateSnapshot`，包含 `.values`（State 内容）和
  `.next`（还有哪些 Node 待执行，若为空元组 `()` 表示已经跑完）。
* **`graph.get_state_history(config)`**：读取某个 thread 从头到尾的全部
  checkpoint 序列。
* **中断与恢复**：编译时传入 `interrupt_before=["tools"]`，图会在即将进入
  `tools` Node 之前暂停，把当前 State 作为一个普通 checkpoint 保存下来。
  恢复执行只需要 `graph.invoke(None, config=config)`——传入 `None` 作为
  输入，LangGraph 会自动加载该 `thread_id` 最新的 checkpoint 并从暂停点
  继续，**不会**从 `START` 重新跑一遍。

## 验证的四个要求

`tests/test_persistence.py` 对应覆盖：

1. **Thread A 与 Thread B 的 State 相互隔离**
   （`ThreadIsolationTest`）：两个 thread 共用同一个已编译的图和同一个
   checkpointer 实例，各自 `invoke` 后 `result["question"]` 互不影响；
   `get_state` 分别读回也只反映各自的 `question`；同一个 `thread_id`
   多次调用则会在同一条历史上累积 checkpoint。
2. **Agent 可以根据之前保存的 State 继续运行 / 能从中断位置继续**
   （`ResumeFromInterruptionTest`）：用 `interrupt_before=["tools"]`
   编译图后，第一次 `invoke` 只跑到 `agent` 就暂停（`state.next ==
   ("tools",)`，`tool_results` 仍为空），此时 `agent_llm_call` 只被调用
   了一次；第二次用 `graph.invoke(None, config=config)` 恢复执行，
   `tool_results` 才增加，`agent_llm_call` 的调用次数才变成两次，最终
   State 的 `next` 变回 `()`，证明是从暂停处继续，而不是重新跑了一遍。
3. **能够读取之前的 State**（`ReadPreviousStateTest`）：`graph.get_state
   (config)` 在图跑完之后依然能取到完整的最终 State，且 `.next == ()`
   表示没有待执行的 Node。
4. **无 checkpointer 时行为不受影响**（`NoCheckpointerTest`）：`build_graph`
   不传 `checkpointer` 时，图和实验 5 完全一样正常运行；但此时调用
   `get_state` 会抛 `ValueError`（LangGraph 明确要求：想要读取/恢复
   State，必须先配置 checkpointer）。

---

## 重点问题解答

### 1. State 和 Checkpoint 的区别？

`State` 是 **某一时刻** 图里流动的数据（本实验里是
`ResearchAgentState`：`question`/`plan`/`messages`/`tool_results`/...）
——它只是一份普通的 Python dict/TypedDict 数据，本身不具备"能否被读回、
能否恢复执行"的能力。

`Checkpoint` 是 checkpointer **把某一次 super-step 执行完之后的 State
拍下的一张快照**，连同一些元数据（属于哪个 `thread_id`、这是第几个
checkpoint、下一步该跑哪些 Node 等）一起保存下来。一次完整的图执行会
产生一连串 Checkpoint（`planner` 跑完一个、`agent` 跑完一个、`tools`
跑完一个……），构成一条历史。

打个比方：State 像是"程序运行时内存里的一个变量"，Checkpoint 像是"给这
个变量在某个时间点拍的一张照片并存进相册"。没有 checkpointer，State 只
在一次 `invoke` 调用的生命周期内存在，调用结束（或进程退出）就没了；有
了 checkpointer，每一张"照片"都被留存下来，可以在未来任意时刻取出来看，
或者以它为起点继续执行。

### 2. Thread ID 的作用？

`thread_id` 是 checkpointer 存储 Checkpoint 的**分区键（partition
key）**。它决定了："这次 `invoke`/`get_state` 调用应该读写哪一条独立的
Checkpoint 历史"。

* 同一个 `thread_id` 的多次调用，会在同一条历史上继续追加 Checkpoint,
  下一次调用能看到上一次留下的 State（这就是"记忆"/"继续对话"或者
  "从中断处恢复"的基础）。
* 不同的 `thread_id` 之间的 Checkpoint 历史完全独立、互不可见——这正是
  Thread A 和 Thread B 相互隔离的原因：即使它们共用同一个已编译的
  `graph` 对象和同一个 `checkpointer` 实例，只要 `thread_id` 不同，
  LangGraph 在内部就是查两张完全不同的"表"。

可以把 `thread_id` 理解成数据库里的"会话 ID"或者"用户 ID"：同一个图
（同一份代码逻辑）可以同时服务成千上万个相互独立的 thread，checkpointer
负责按 `thread_id` 把它们的执行状态分开存放。

### 3. 为什么 Agent 需要 Persistence？

* **可恢复**：真实场景中 Agent 的一次任务可能耗时很久（多轮工具调用、
  等待人工审批等），进程可能会重启、请求可能会超时。没有 Persistence，
  一旦中断就必须从 `START` 完全重跑；有了 checkpointer，可以从最近一个
  Checkpoint 继续，不丢失已经完成的工作（本实验的 `interrupt_before=
  ["tools"]` + 恢复测试直接验证了这一点）。
* **可观测/可调试**：`get_state`/`get_state_history` 能让开发者在图执行
  的任意阶段"暂停下来看一眼"当前 State 是什么样子，而不需要在代码里到
  处插 `print`。这也是为多步骤 Agent Loop 做故障排查的基础能力。
* **支持人机协作（Human-in-the-loop）的前提**：虽然本实验没有实现
  Human-in-the-loop，但 `interrupt_before` 展示的正是它的底层机制——图在
  某个关键 Node 之前暂停、把 State 落盘，等待外部（人工审核、审批、修改
  参数）介入后再恢复执行。没有 checkpointer，这种"暂停-等待-恢复"根本无
  法实现，因为暂停时的执行上下文（State）会直接消失。
* **多用户/多会话隔离**：`thread_id` 机制让一个部署的 Agent 服务可以同时
  安全地服务多个独立用户/独立任务，而不需要在业务代码里手写"给每个用户
  维护一份独立 State 字典"这类样板代码。

### 4. Persistence 与 Memory 有什么区别？

这里的 **Persistence（本实验实现的 Checkpointer）** 是 LangGraph 图执行
引擎自身的机制：它保存的是"这张图在某个 `thread_id` 下运行到哪一步、
`State` 长什么样"，服务于**恢复执行、故障排查、隔离并发任务**这些"执行
引擎"层面的问题。它是**结构化、按 super-step 记录、与图的拓扑强相关**
的——每个 Checkpoint 都对应图里某个确定的执行位置（"下一步该跑哪个
Node"）。

**Memory**（本次任务明确要求"不使用 Memory"）通常指的是更上层的、面向
"对话/推理内容"的能力，例如：跨越多次完全独立会话去总结、检索、注入
"用户过去说过什么/系统学到了什么"这样的语义信息，服务于**让 Agent 表现
得像记得住东西**这个产品层面的问题。它通常需要额外的存储和检索逻辑
（例如摘要、向量检索），并且不一定和某一次具体的图执行绑定。

简单说：Checkpointer 回答的是"这次任务执行到哪儿了，怎么接着跑"；Memory
回答的是"要不要、以及怎么把之前学到/说过的东西继续带到未来的新任务里"。
LangGraph 的 checkpointer 也可以被用作实现 Memory 的**底层存储手段之一**
（比如把过去所有 thread 的历史都存起来供检索），但两者的关注点和使用
方式并不相同——本实验只使用了前者，没有引入任何独立的 Memory 组件。
