# LangGraph 学习总结：架构回顾（Architecture Review）

本文档总结整个 LangGraph 学习阶段（`04-langgraph/`，对应 `src/01_minimal_graph.py`
到 `src/08_human_in_loop.py`，以及 `src/patterns/` 五种 Anthropic Pattern），
并与之前两个阶段做对比：

* `01-minimal-agent/`：手写 `while`/`for` 循环的最小 Agent（无框架）。
* `02-openai-agents-sdk/`：基于 OpenAI Agents SDK 的 Agent（`Agent` +
  `Runner` + `@function_tool`）。
* `03-agent-patterns/`：用 OpenAI Agents SDK 实现的 Anthropic 五种
  Agent Pattern（`src/patterns/` 是它们的 LangGraph 版本）。

本文档只做总结，不修改任何代码。

---

## Part 1 — 核心概念

| 概念 | 含义 | 本项目中的例子 |
|---|---|---|
| **State** | 在整张图里流动的、显式定义的数据结构（`TypedDict`）。每个 Node 读取它、返回一份"部分更新"，由 LangGraph 负责合并回主 State。 | `ResearchState`（`02_state.py`）、`ResearchAgentState`（`05_research_agent.py`）、`HumanReviewState`（`08_human_in_loop.py`） |
| **Node** | 一个普通函数 `(state) -> partial_state_update`。只负责自己该管的字段，不直接依赖全局变量、不直接持有 LLM/Tool 客户端（通过闭包注入）。 | `planner`/`researcher`/`analyst`/`writer`（`03_prompt_chaining_graph.py`）、`agent`/`tools`（`04_agent_loop.py`） |
| **Edge** | Node 之间固定、无条件的连接，`add_edge(a, b)`：`a` 跑完后必然轮到 `b`。 | `graph_builder.add_edge("planner", "researcher")` |
| **Conditional Edge** | 由一个"路由函数"在运行时决定下一个 Node 是谁（或动态调用多少次某个 Node），`add_conditional_edges(node, routing_fn, mapping)`。 | `should_continue`（`04_agent_loop.py`）、`route_after_review`（`08_human_in_loop.py`）、`assign_workers`（`src/patterns/orchestrator_workers.py`，用 `Send` 实现动态数量的分支） |
| **START** | 图的固定入口伪节点，`from langgraph.graph import START`；所有真正的第一个 Node 都从 `START` 引出一条边。 | `add_edge(START, "planner")` |
| **END** | 图的固定出口伪节点；走到 `END` 意味着这次 `invoke` 完成。 | `add_edge("writer", END)` |
| **Graph（`StateGraph`）** | 由 State Schema + 若干 Node + 若干 Edge/Conditional Edge 构成的、尚未编译的图定义对象。 | `graph_builder = StateGraph(ResearchAgentState)` |
| **Compile（`compile()`）** | 把 `StateGraph` 定义转换成一个可执行、经过校验（例如检测 Node 名和 State 字段冲突）的可运行对象；可以在这一步传入 `checkpointer`、`interrupt_before` 等运行时配置。 | `graph_builder.compile(checkpointer=checkpointer)` |
| **Invoke（`invoke()`）** | 同步地把图从 `START` 跑到 `END`（或跑到某个中断点），返回最终 State。 | `graph.invoke(initial_state(question), config=config)` |
| **Stream** | 与 `invoke` 对应的流式接口（`graph.stream(...)`），可以在每个 super-step 完成后就拿到中间 State/事件，而不是等整张图跑完才拿到一次性结果。本阶段的实验全部使用 `invoke`，`stream` 是同一套图定义可以直接切换的另一种消费方式，不需要改 Node/Edge 代码。 | （本阶段代码未使用，但 API 与 `invoke` 完全对称：`graph.stream(...)`） |
| **Checkpoint** | Checkpointer 在每个 super-step 结束后为某个 `thread_id` 保存的一份 State 快照（连同"下一步该跑哪些 Node"等元数据）。 | `InMemorySaver`（`07_persistence.py`、`08_human_in_loop.py`） |
| **Interrupt** | 图执行中途主动暂停的机制：`langgraph.types.interrupt(payload)` 在 Node 内部调用，把 payload 交给外部、暂停执行；靠 `Command(resume=...)` 从同一位置恢复。依赖 Checkpointer 才能真正生效（见 `docs/08-HUMAN-IN-THE-LOOP.md` Q1）。 | `human_review` Node（`08_human_in_loop.py`） |

这十二个概念可以归纳成三层：**State/Node/Edge/Conditional Edge/START/END**
是"图长什么样"（拓扑与数据契约）；**Graph/Compile/Invoke/Stream** 是"图怎么被
构建和跑起来"（执行引擎）；**Checkpoint/Interrupt** 是"图的执行状态怎么被保存、
暂停、恢复"（运行时持久化）。

---

## Part 2 — 与第⑤步比较：Minimal Agent（手写 `while` loop）VS LangGraph

`01-minimal-agent/src/agent.py` 的核心是一个 `for turn in range(1, max_turns
+ 1):` 循环（本质上是一个带上限的 `while True`）：

```python
state = {"messages": [...], "turns": 0, "tool_calls": [], ...}
for turn in range(1, max_turns + 1):
    response = _call_llm(llm, state["messages"], state["errors"])
    if not response.get("tool_calls"):
        state["final_answer"] = response["content"]
        return state          # <-- "done" 分支
    for tool_call in response["tool_calls"]:
        tool_result = _execute_web_search(tool_call, web_search)
        state["messages"].append(...)  # <-- 结果写回，continue 循环
```

这与 `04_agent_loop.py`（实验 4）里的 Graph 是**同一段逻辑的两种表达**：

| 手写 `while`/`for` 循环里的结构 | 对应的 LangGraph 结构 |
|---|---|
| 循环体本身（每一轮"问 LLM"） | `agent` Node |
| `if not tool_calls: return state`（终止分支） | `should_continue` 返回 `"end"` → 边指向 `END` |
| `for tool_call in tool_calls: execute...`（工具分支） | `should_continue` 返回 `"tools"` → 边指向 `tools` Node |
| 工具结果 `append` 进 `messages` 后，循环回到顶部再问一次 LLM | `tools` Node 执行完，`add_edge("tools", "agent")` 把控制权带回 `agent` |
| `for turn in range(1, max_turns+1)`（硬编码轮数上限） | `agent` Node 内部 `if steps > max_steps: ...` 提前终止，或 `config={"recursion_limit": ...}` |
| `state` 这个普通 dict，靠函数参数/返回值手动传递 | `ResearchAgentState`（`TypedDict`），由 LangGraph 在 Node 之间自动合并 |

**关键映射关系**：`while True` 本身就是一个"隐式的图"——它只有两个节点
（"问 LLM"和"执行工具"）和一条隐式的条件边（"有没有 tool_call"）。LangGraph
没有引入任何手写循环里没有的新概念，只是把这个隐式的图**显式化**：循环体
拆成 Node，`if/else` 分支拆成 Conditional Edge，`continue`/`return` 拆成
指向自身或指向 `END` 的边。好处是：这个图可以被 `get_graph().draw_mermaid()`
画出来、可以被 `get_state()` 检查、可以加 checkpointer 持久化——而这些能力
在手写循环里都需要自己重新实现（正如 `docs/08-HUMAN-IN-THE-LOOP.md` Q4 里
分析的那样）。

---

## Part 3 — 与第⑥步比较：OpenAI Agents SDK VS LangGraph

`02-openai-agents-sdk/src/agent.py` 用 `Agent(instructions=..., tools=[...])`
+ `Runner.run_sync(agent, question, max_turns=...)` 表达同一个研究 Agent；
`Runner` 内部封装了完整的"问 LLM → 检测 tool_call → 执行 → 回填 → 再问"循环，
业务代码只声明"Agent 是什么"，不声明"循环怎么跑"。

| 维度 | OpenAI Agents SDK | LangGraph |
|---|---|---|
| **Agent Runtime** | `Runner` 是一个黑盒运行时：Agent Loop、tool 分发、终止条件全部封装在 SDK 内部，开发者看不见、也不需要写循环本身。 | 没有预置的"Agent Runtime"概念——Agent Loop 只是 `StateGraph` 上 `agent ↔ tools` 之间的普通 Conditional Edge（`04_agent_loop.py`）。运行时是通用的图执行引擎，Agent 只是它能表达的众多拓扑之一。 |
| **Control Flow** | 由 `Agent.instructions` + LLM 的 tool-call 决策隐式决定；`max_turns` 是唯一暴露给开发者的显式控制点。 | 由 Node/Edge/Conditional Edge 显式声明；任何拓扑（线性链、路由、并行、循环、动态 fan-out）都用同一套 API 表达，参见 Part 4。 |
| **State** | `Runner` 内部维护对话历史（`messages`/`RunResult`），开发者不定义自己的 Schema，只能拿到 SDK 定义好的输出对象。 | 开发者显式定义 `TypedDict` State Schema（如 `ResearchAgentState`），每个字段的读写职责都是可审查的代码，而不是 SDK 内部结构。 |
| **Tool Calling** | `@function_tool` 装饰器自动从函数签名生成 JSON Schema，`Runner` 自动分发/回填结果，开发者几乎不写调度代码。 | 需要手写 `tools` Node（`04_agent_loop.py`/`05_research_agent.py`）：解析 `pending_tool_call`、调用真正的函数、把结果写回 State——控制力更强，但也更啰嗦。 |
| **Persistence** | SDK 提供 `Sessions`（对话历史持久化）等机制，但本仓库 `02-openai-agents-sdk` 未使用；且这类机制面向"对话历史"，不是"图执行到哪一步"。 | 内建 Checkpointer（`InMemorySaver`）+ `thread_id`，保存的是"图执行到了哪个 super-step、State 是什么"，可精确恢复到某个 Node 之前（`07_persistence.py`）。 |
| **Human-in-the-loop** | SDK 没有内建的"暂停等待人类审批"原语；要实现类似效果需要自己在 `Runner.run` 外部手写审批流程和状态存取。 | 原生支持：`interrupt()` + `Command(resume=...)`，暂停位置可以在任意 Node 内部任意一行（`08_human_in_loop.py`）。 |
| **Tracing** | 内建、开箱即用：每次 `Runner.run` 自动产生 agent-run/LLM-generation/tool-call span（`02-openai-agents-sdk/src/tracing.py` 只是把导出目的地换成本地文件）。 | 没有内建 tracing UI；`get_graph().draw_mermaid()` 提供的是**图结构**的可视化，不是**某一次运行的执行轨迹**；如需运行时可观测性通常接入 LangSmith 或自建日志（本阶段实验都用 `print`/测试断言代替）。 |
| **Long-running tasks** | 依赖 `max_turns` 和进程存活；进程重启或跨请求的长任务需要开发者自己接入 Sessions/外部存储。 | Checkpointer 让"长任务"天然可以跨调用、跨进程恢复——这正是 `07_persistence.py`/`08_human_in_loop.py` 存在的原因：一次审批可能间隔数小时，图可以安全地"挂起"在 `human_review` 之前。 |

**一句话对比**：OpenAI Agents SDK 优化的是"**尽快搭好一个能跑的 Agent**"
（Runtime/Tool Calling/Tracing 都是开箱即用的黑盒能力）；LangGraph 优化的是
"**把 Agent 的执行过程变成一个可检查、可持久化、可暂停恢复的显式图**"
（State/Control Flow/Persistence/Human-in-the-loop 都是需要自己声明、但也因此
完全可控的白盒能力）。两者并不互斥——LangGraph 的 Node 内部完全可以调用
OpenAI Agents SDK（本仓库没有这样做，是为了保持"只用 LangGraph"这一学习
目标的纯粹性）。

---

## Part 4 — 与第⑦步比较：Anthropic Patterns VS LangGraph

`03-agent-patterns/` 用 OpenAI Agents SDK（`Agent`/`Runner`/`asyncio.gather`/
手写 `while` 修订循环）实现了 Anthropic 的五种 workflow pattern；
`src/patterns/` 是它们完全对应的 LangGraph 版本。二者的"意图"完全相同，
"实现载体"不同：

| Anthropic Pattern | Graph Structure（LangGraph 版本） |
|---|---|
| **Prompt Chaining** | 一条线性链：`START -> planner -> researcher -> analyst -> writer -> END`。全是普通 `add_edge`，没有分支——因为步骤顺序在设计时已确定。 |
| **Routing** | 一次分类 + 三选一：`START -> router ->（Conditional Edge）-> {technical | business | general} -> END`。`route_after_router` 是路由函数，保证**只有**被选中的一个专家 Node 会执行。 |
| **Parallelization** | 静态 fan-out + 自动 fan-in：`START` 同时指向三个独立的 Researcher Node，它们并发写入同一个 `findings` 字段（靠 `Annotated[list[...], operator.add]` Reducer 合并），全部完成后自动汇入 `synthesizer -> END`。分支数量在设计时固定为 3。 |
| **Orchestrator-Workers** | 动态 fan-out：`orchestrator` 决定运行时产生的任务数量，`assign_workers` 路由函数返回 `list[Send("worker", {...})]`——分支数量在**运行时**才确定，靠同一个 Reducer 机制合并所有 `worker` 的结果，再汇入 `synthesizer -> END`。 |
| **Evaluator-Optimizer** | 一个环：`generator -> evaluator ->（Conditional Edge）-> {generator（revise，循环）| finalize（accept）} -> END`。`should_revise` 依据代码里写死的 `PASS_SCORE`/`MAX_ITERATIONS` 决定继续修订还是收尾。 |

**结构上的规律**：五种 Pattern 在 LangGraph 里对应的正是 Part 1 里那几个
基本概念的不同组合——普通 Edge 表达"顺序"，Conditional Edge 表达"分支"，
静态 fan-out/fan-in 表达"固定并行"，`Send` 表达"动态并行"，Conditional Edge
指回上游 Node 表达"循环"。没有为任何一个 Pattern 引入新的 API——这正是
`docs/06-PATTERNS-IN-LANGGRAPH.md` 里详细展开的"为什么 LangGraph 适合表达
这五种模式"的论据：一套统一的图语言，就能覆盖 Anthropic 定义的全部
基础工作流形状。

---

## Part 5 — 最重要的问题

### 1. LangGraph 是 Agent Framework 还是 Workflow Framework？

**本质上是 Workflow Framework，Agent 只是它能表达的一种特殊 Workflow。**
从整个学习路径看：实验 1-3（线性图）、Patterns 里的 Prompt
Chaining/Routing/Parallelization/Orchestrator-Workers 都是没有"LLM 自主决定
下一步做什么"的固定或半固定工作流；只有实验 4/5（`should_continue` 由 LLM
的 tool-call 决策驱动）和 Evaluator-Optimizer 才具备"Agent"的特征
（循环由运行时判断，而非设计时写死）。LangGraph 提供的 State/Node/Edge/
Conditional Edge 是通用的图计算原语，Agent Loop 只是用这套原语搭出来的
一种拓扑（`agent ↔ tools` 的环）。可以说：LangGraph 是一个**通用的、
有状态的图执行引擎**，Agent Framework 的能力（工具调用循环、暂停审批等）
是在这套引擎上"搭"出来的，而不是引擎本身内置的领域概念。

### 2. 为什么 LangGraph 可以不使用 LangChain？

因为 LangGraph 依赖的只是 LangChain 生态里**最底层、最通用**的几个东西
（`langchain-core` 提供的 `Runnable`/序列化协议等基础设施），而不是
LangChain 的 Agent 抽象（`AgentExecutor`、`create_react_agent` 等）。本仓库
从实验 1 到实验 8，Node 里调用 LLM 全部是通过手写的 `TextLLMCall`/
`AgentLLMCall` 这类**普通函数签名**（内部用裸 `openai` 客户端实现），从未
引入 `langchain.agents` 或任何 LangChain Chain/Agent 类。这证明 LangGraph
的核心能力——State、Node、Edge、Conditional Edge、Checkpointer、
Interrupt——是**独立于"LangChain 的 Agent 抽象"**的一套图计算模型，
可以只用 `StateGraph` + 普通 Python 函数就完整搭建 Agent，不需要接受
LangChain 那一整套 Prompt Template/Chain/Agent 的编程范式。

### 3. LangGraph 的核心价值是不是"让 LLM 更聪明"？

**不是。** 整个学习阶段里，每一个 Node 用的都是完全一样的、注入进去的
`TextLLMCall`/`AgentLLMCall`——LangGraph 从不参与 LLM 调用本身的质量
（不做 prompt 优化、不做模型选择、不做推理增强）。它解决的是"**LLM 调用
被组织成什么样的执行结构**"这个问题：调用顺序（Edge）、调用是否需要
重复（Conditional Edge 形成的循环）、调用之间的数据怎么传递（State）、
执行状态怎么保存和恢复（Checkpoint/Interrupt）。把 LangGraph 换成任何
其它 LLM（甚至实验 1-4 里完全不用 LLM），图的行为、可测试性、可持久化
能力都不会变——这本身就说明它的核心价值与"LLM 聪明与否"无关，而是
"**围绕 LLM 调用的编排、状态管理、持久化能力**"。

### 4. 为什么 State 是 LangGraph 的核心？

因为 LangGraph 里几乎所有其它概念都是"作用在 State 上的操作"：Node 是
"State 的一次局部更新"，Edge/Conditional Edge 是"State 更新完之后往哪走"，
Reducer 是"多个并发更新如何合并回 State"，Checkpoint 是"State 在某个时刻
的快照"，Interrupt 恢复时靠的也是从 Checkpoint 里取回 State 继续跑。
`docs/02-STATE.md` 里已经论证过：State 取代了手写循环里散落的局部变量/
闭包变量/全局变量，把"整个执行过程中会变化的数据"收敛成一个显式、可
检查、可序列化的 Schema。没有这个显式 State，Checkpointer 就无从"存什么"，
Reducer 就无从"合并什么"，Node 之间也无法解耦——State 是把"图的拓扑"和
"图的持久化能力"粘合在一起的那个基础数据结构。

### 5. Conditional Edge 为什么对 Agent 很重要？

因为 Agent 与普通 Workflow 的本质区别就在于："**下一步做什么是运行时才能
决定的**"（是否需要再搜索一次？分数够不够高需要重新生成？人类批准了没
有？）。普通 Edge 只能表达"设计时就知道下一步是谁"；Conditional Edge 把
"读取当前 State、决定走哪条路"这件事变成了一等公民（`should_continue`、
`route_after_review`、`should_revise` 都是这种模式）。没有 Conditional
Edge，Agent Loop（`agent ↔ tools`）、Evaluator-Optimizer 的修订循环、
Human-in-the-loop 的三路分支都无法表达——本质上，Conditional Edge 就是
"if/else 和 while 循环的图形式"，是 Agent 拥有"自主判断能力"在图结构上的
唯一落点。

### 6. Persistence 为什么对 Production Agent 很重要？

因为生产环境里的 Agent 任务往往不是"一次 `invoke` 调用几秒钟内跑完"这么
简单：进程可能重启、请求可能超时、审批可能要等几个小时甚至几天
（`08_human_in_loop.py` 里的 `human_review`）、同一时间有大量互不相关的
用户任务在并发进行（`07_persistence.py` 里的 Thread A/Thread B 隔离）。
没有 Persistence：一旦中断就必须从头重跑（浪费已完成的工作、可能重复
产生副作用如重复发布），且完全没有能力支持"等待外部输入"这种模式。有了
Checkpointer + `thread_id`：任务可以安全地跨调用、跨进程、跨时间挂起和
恢复，且不同任务之间的执行状态互不干扰——这是把一个"能跑通的 demo"变成
"能在真实生产环境里可靠运行的服务"所必需的能力。

### 7. 什么时候不应该使用 LangGraph？

* **任务本身就是一次性、无分支、无需重试的单次 LLM 调用**——比如"把这段
  文本翻译成英文"，直接调用 LLM API 即可，引入 State/Node/Edge 纯属过度
  设计。
* **不需要多步骤编排、也不需要持久化/暂停恢复的简单脚本**——如果整个
  逻辑三五行就能写清楚，用 `StateGraph` 反而增加了理解和维护成本（对比
  Part 2：手写 `while` 循环并不天然更差，只是可观测性和持久化能力较弱）。
* **需要的是"开箱即用的 Agent 产品能力"而不是"自己搭建执行引擎"**——例如
  需要成熟的对话历史管理、内建 tracing/评估平台、大量现成 Handoff/
  Multi-Agent 编排能力，OpenAI Agents SDK 或其它高层框架可能用更少代码
  达到目标（见 Part 3 对比，这些正是 LangGraph 没有内建、需要自己搭的
  部分）。
* **强依赖 LangChain 现有 Chain/Retriever/Agent 生态，且团队已经在用
  LangChain 的场景**——如果项目已经大量使用 LangChain 的 Prompt
  Template/Retriever/Chain 抽象，直接在那套体系上加 LangChain 的
  Agent 能力可能比"额外引入一套图执行引擎"更省心（当然也可以像本项目
  一样只用 LangGraph 的最小依赖）。
* **业务逻辑没有任何真正的分支/循环/并行结构**——如果流程永远是从头到
  尾直线执行，不需要 Conditional Edge，也大概率不需要持久化/中断，
  普通函数调用链（Part 2 对照的手写 `while`/顺序调用）足够清晰，
  没必要为了"用图"而用图。

---

## 一句话总结

> **LangGraph 真正解决的问题是：把"一次 LLM 驱动的多步骤执行过程"从一段
> 只能在内存里、从头跑到尾的命令式代码，变成一张显式的、State 驱动的、
> 可以被检查（Mermaid/`get_state`）、可以被持久化（Checkpoint）、可以被
> 暂停和恢复（Interrupt/Resume）的图——而不是让 LLM 本身变得更聪明。**
