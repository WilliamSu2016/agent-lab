# LangGraph 概念入门：它到底解决了什么问题？

> 资料访问日期：2026-09-10。以本次访问到的 **LangGraph 官方 Python 文档、官方 GitHub 源码和发布记录**为依据，不套用旧版教程。
>
> 本次官方发布页和核心包元数据均显示 `langgraph` 版本为 **1.2.11**。仓库中的 `langgraph-sdk`、checkpoint 扩展是独立包，版本号不能混用。在线文档与 GitHub `main` 会继续更新；本文对实验性接口另行标注。[L1][G1]
>
> 本文只讨论概念，并提供文档内的教学示例，不修改源代码、不安装依赖。当前项目的 `requirements.txt` 尚未声明 LangGraph；这些示例不表示当前环境已具备运行条件。项目 Python 仍遵循 **3.10.8 + 项目本地 `.venv`** 的约定。

## 先给出答案

**LangGraph 解决的核心问题，是如何把一个有状态、可能循环、分支、并行、暂停和恢复的 AI 工作过程，变成显式、可执行、可管理的程序。**

只有一次模型调用时，普通函数通常就够了。复杂性出现在这些问题上：

- 模型调用工具以后，谁决定再调用模型还是结束？
- 多个步骤如何共享结果，又不互相覆盖？
- 两条分支同时执行，什么时候汇总？
- 执行中断后，保存的是聊天记录，还是连“下一步做什么”也能恢复？
- 等人工审批时，如何暂停，并在稍后继续同一个任务？

LangGraph 不是让模型突然“更聪明”，也不是要求所有程序都画成图。它提供的是**状态与执行控制的基础设施**。模型、工具、业务规则仍需要开发者设计。

先记住这个关系：

```text
State             = 当前任务的数据与进展
Node              = 读取状态、执行工作、返回更新
Edge              = 后续执行关系
Conditional Edge  = 根据运行时结果选择后续执行
StateGraph        = 定义这些规则的图构建器
compile()         = 得到可执行图
invoke()/stream() = 驱动图执行
Checkpointer      = 保存图状态与恢复所需信息
interrupt()       = 暂停，等待外部输入
```

## 1. LangGraph 与 OpenAI Agents SDK 的核心区别是什么？

**核心区别是主要抽象层级，不是“一个能做 Agent，另一个不能”。**

| 角度 | OpenAI Agents SDK | LangGraph Graph API |
|---|---|---|
| 主要抽象 | `Agent`、`Runner`、tools、handoffs | `StateGraph`、State、Node、Edge |
| 通常从哪里开始 | 定义 Agent 的指令、工具和交接对象，再交给 Runner | 定义状态、步骤和流转规则，再编译执行 |
| Agent Loop | Runner 管理模型、工具、交接等运行循环 | 可以把模型、工具和终止条件显式写成图上的循环 |
| 控制流程 | 通过工具、handoffs、Agent-as-tool 和应用代码组合 | 用固定边、条件边、`Command`、`Send` 等直接表达 |
| 状态与恢复 | Sessions 管理会话历史；`RunState` 支持暂停运行的序列化与恢复 | Checkpointer 管理 thread 内的图状态、执行进度与历史 checkpoint |
| 人工介入 | 支持工具审批及审批后的继续运行 | 可在节点逻辑中动态 interrupt，并恢复或改变后续业务路径 |
| 常见选择理由 | 希望快速围绕 Agent 及其工具构建应用 | 希望明确掌握多步骤状态、分支、循环、汇合及恢复边界 |

这些是主要设计取向，**不是能力的互斥清单**。Agents SDK 也能通过代码实现路由、并行和复杂流程；LangGraph 也可以封装成简单的 Agent 调用。

尤其不要误解为：

> “Agents SDK 没有记忆、持久化或 Human-in-the-loop，因此需要 LangGraph。”

当前 Agents SDK 官方文档明确支持 Sessions、工具审批、`RunState` 序列化及恢复，也支持 handoffs。更准确的问题是：**我想以 Agent 运行循环为中心开发，还是想把整个业务状态机与执行拓扑显式建模？** [O1][O2][O3][O4]

## 2. LangGraph 为什么使用 Graph？

因为 Agent 的执行过程往往不是一条固定直线。

```text
顺序：  输入 -> 检索 -> 生成 -> 结束

分支：  分类 -> 技术支持
             -> 订单查询

循环：  模型 -> 工具 -> 模型 -> ... -> 结束

并行：  输入 -> 检索 A --+
             -> 检索 B --+-> 汇总
```

Graph 将“执行什么”放进节点，将“接下来去哪里”放进边，再用 State 连接各步骤的数据。

它不只是可视化：图结构参与实际调度、状态更新和恢复。LangGraph 支持环，因此**不能把它理解成只能执行 DAG 的流水线工具**。

底层运行采用 super-step 思路：同一执行步中被调度的节点可以并行运行，完成后合并更新，再推进后续执行。普通共享状态读写不是“多个节点随时修改同一个全局字典”；同一步中的兄弟节点不能依赖对方尚未提交的更新。[L2][L5][G4]

**Graph 管执行，LLM 管它被授权作出的决策。** 路由可以完全由确定性的 Python 规则决定，并不要求每一条边都让模型选择。

## 3. StateGraph 中 State 的作用是什么？

State 是**图运行过程中跨步骤传递、更新的数据模型**，例如：

```python
from typing import TypedDict


class TaskState(TypedDict):
    question: str
    answer: str
    attempts: int
```

它可以存消息历史、检索结果、工具输出、重试次数、审批结果等，不限于聊天记录。

`StateGraph(TaskState)` 用 schema 定义状态字段。官方支持 `TypedDict`、dataclass 和 Pydantic 等方式。`TypedDict` 主要提供类型描述，**不是自动运行时校验，也不会自动填充字段**。[L2][G2]

### 更新的是部分状态，不必返回全部状态

```python
def draft(state: TaskState):
    return {
        "answer": f"Draft for: {state['question']}",
        "attempts": state["attempts"] + 1,
    }
```

节点返回的是更新：这里不返回 `question`，并不意味着删除 `question`。应通过返回更新来表达变化，而不是依赖原地修改输入字典。

### Reducer 决定字段如何合并

默认没有 reducer 时，一个字段的新更新替换旧值。需要累积时，可以为字段指定 reducer：

```python
import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict):
    results: Annotated[list[str], operator.add]
```

此时节点返回 `{"results": ["new finding"]}`，表示追加这段结果。

两个关键边界：

- 同一 super-step 中多个节点写入同一个没有 reducer 的字段，会产生并发更新错误；**不是最后写入者获胜**。
- 追加 reducer 下应返回新增片段，而不是“旧列表 + 新内容”的完整列表，否则旧内容可能重复累计。返回空列表也不会清空已有列表。

对消息历史，通常优先理解官方的 `MessagesState`：

```python
from langgraph.graph import MessagesState
```

它的 `messages` 字段使用 `add_messages`：新消息 ID 追加，相同 ID 更新已有消息，并支持消息格式转换及删除消息的特殊处理。它不等同于普通列表相加。[L2][G3]

**State 也不是所有资源的容器。** 数据库连接、客户端等运行依赖不应直接当作可持久化业务状态塞进去；当前 Graph API 可用 `context_schema` 和节点运行上下文表达相关依赖。业务状态、运行上下文和 checkpoint 配置是不同概念。

## 4. Node 是什么？

Node 是图中的一个执行单元，通常是同步或异步 Python 函数。

基本心智模型是：

```text
State -> Node -> Partial State Update
```

一个节点可以调用模型、执行工具、检索资料、处理数据、校验结果，或者请求人工审批。**Node 不等于 Agent，也不等于一次 LLM 调用。**

```python
builder.add_node("draft", draft)
```

这里 `"draft"` 是图中的节点名，`draft` 是实际执行函数。支持的节点还可接收配置、运行上下文等；进阶情况下可返回 `Command`，同时表达状态更新与路由。[L2][G2]

划分节点时，优先考虑清晰的业务职责与恢复边界，而不是把每行代码拆成一个节点。

## 5. Edge 是什么？

普通 Edge 表达固定的后续执行关系：

```python
builder.add_edge("retrieve", "draft")
```

含义是：`retrieve` 完成后，调度 `draft`。

**边主要控制执行，不是把前一个节点的返回值当作下一个函数唯一的位置参数。** 节点更新先通过状态机制合并，后续节点再读取其可见状态。

一个节点也可以有多个后继，形成 fan-out。需要等待多个前驱全部完成时，使用明确的汇合声明：

```python
builder.add_edge(["search_a", "search_b"], "summarize")
```

这表示等待 `search_a` 和 `search_b` 都完成。不能把分别添加两条指向 `summarize` 的边，无条件视为相同的“等待全部”语义，尤其是分支长度不一致时。[L2][G2]

## 6. Conditional Edge 是什么？

Conditional Edge 是**由运行时路由函数决定目的地的边**。

```python
def route_after_review(state):
    return "retry" if state["needs_revision"] else "done"


builder.add_conditional_edges(
    "review",
    route_after_review,
    {"retry": "draft", "done": END},
)
```

上述为结构片段，假设已导入 `END`、定义对应状态，并注册 `review` 和 `draft` 节点。

运行逻辑是：

```text
review 完成 -> 合并状态更新 -> router 读取当前状态
                               |
                               +-> retry -> draft
                               +-> done  -> END
```

路由函数可以返回目标节点名，也可以返回由 `path_map` 映射的标签，还可以返回多个目的地。多个目的地意味着多路调度，不一定是单选。

建议使用显式 `path_map` 或 `Literal[...]` 返回类型，让路由意图和图可视化更明确。路由函数本身主要负责选择，**不是通过返回一个普通字典来更新 State**。[L2][G2]

## 7. START / END 是什么？

它们是 LangGraph 提供的特殊标记：

```python
from langgraph.graph import START, END

builder.add_edge(START, "first")
builder.add_edge("last", END)
```

| 标记 | 意义 |
|---|---|
| `START` | 虚拟入口，用于指定收到输入后从哪里开始 |
| `END` | 虚拟终点，表示该路径不再调度后继 |

不需要为它们编写业务函数，也不要用普通字符串 `"START"` / `"END"` 替代导入的常量。

有并行分支时，一条路径到达 `END` **不等于强制取消其他分支**。正常完成是图中没有待执行工作；暂停和异常则属于不同的运行结果。[L2][G2][G4]

## 8. compile() 做什么？invoke() 和 stream() 又是什么？

### compile()：从定义得到可执行图

`StateGraph` 是构建器，本身不是执行结果。`compile()` 将已声明的 schema、节点和边组织成 `CompiledStateGraph`，执行必要的图结构校验，并接入 checkpointer 等运行组件。

```python
graph = builder.compile()
```

它不是模型训练，也不是把 Python 编译成机器码；**不会在这一步执行整条业务流程**。结构校验也不能代替业务逻辑正确性校验。[L2][G2]

### 一个不依赖 LLM 的完整最小例子

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class GreetingState(TypedDict):
    name: str
    greeting: str


def greet(state: GreetingState):
    return {"greeting": f"Hello, {state['name']}!"}


builder = StateGraph(GreetingState)
builder.add_node("greet", greet)
builder.add_edge(START, "greet")
builder.add_edge("greet", END)
graph = builder.compile()

result = graph.invoke({"name": "Ada", "greeting": ""})
print(result["greeting"])  # Hello, Ada!
```

这已经是一个 LangGraph 程序，但不是 Agent。它说明**学 Graph 不需要先引入模型和复杂工具**。

### invoke()：运行并取得最终输出

通常的默认用法是：

```python
result = graph.invoke(input_state)
```

调用者等待运行完成，或者遇到 interrupt、错误等情况。默认返回最终输出；在上面的例子中，输出就是状态字典。若定义了独立 `output_schema`，输出则受该 schema 约束。

异步对应 `await graph.ainvoke(...)`。

### stream()：运行时逐步消费输出

接着上面的 `graph`，可改用：

```python
for part in graph.stream(
    {"name": "Ada", "greeting": ""},
    stream_mode="updates",
    version="v2",
):
    print(part["data"])
```

节点更新的数据形如：

```python
{"greet": {"greeting": "Hello, Ada!"}}
```

这是**一次新的执行**，不是读取前一次 `invoke()` 的日志；迭代 stream 才会驱动执行。对有外部副作用的图，不要为了观察输出而无意中运行两次。

| 常用 stream mode | 输出内容 |
|---|---|
| `updates` | 节点产生的状态增量 |
| `values` | 每一步之后的完整状态 |
| `messages` | LLM 消息片段及 metadata，不是任意节点返回字符串的逐字拆分 |
| `custom` | 节点主动发出的自定义进度数据 |
| `checkpoints` / `tasks` / `debug` | checkpoint、任务和诊断事件；前两者要求 checkpointer |

异步对应 `async for ... in graph.astream(...)`。[L7][G4]

### 当前版本特别需要区分的输出协议

**不要把不同方法的 `version` 参数混为一谈。**

| API | 本次官方文档与源码中的行为 |
|---|---|
| `stream(...)` 默认 v1 | 输出形状随单模式、多模式、子图选项变化 |
| `stream(..., version="v2")` | LangGraph >= 1.1；统一输出 `{"type": ..., "ns": ..., "data": ...}` |
| `invoke(...)` 默认 v1 | 本文最小例子使用的默认输出；动态中断放在 `__interrupt__` 中 |
| `invoke(..., version="v2")` | 默认 values 模式返回 `GraphOutput`，通过 `.value`、`.interrupts` 访问，不再直接等同于状态字典 |
| `stream_events(..., version="v3")` | v1.2 引入的 typed-projection 事件流；文档推荐新应用使用，但当前源码仍标注 experimental |

新版事件流示意：

```python
run_stream = graph.stream_events(
    {"name": "Ada", "greeting": ""},
    version="v3",
)
for state in run_stream.values:
    print(state)
final_state = run_stream.output
```

它提供 `messages`、`values`、`subgraphs`、`output`、`interrupts` 等独立投影。本文用 `stream(..., version="v2")` 讲清基础状态流，同时保留官方对 v3 的推荐及源码的实验性警告；不把新接口误写成旧版行为，也不承诺实验性 API 稳定。[L7][L8][G4]

## 9. LangGraph 如何表达 Agent Loop？

通过**回到先前节点的边 + 决定继续还是结束的条件边**。

官方 quickstart 中的工具调用循环可以概括为：

```text
START -> model
           |
           +-- 有 tool_calls --> tools --+
           |                            |
           +-- 无 tool_calls --> END    +--> model
```

对应的图构建片段如下。它不是完整 Agent；`model_node` 需调用已绑定工具的模型，`tool_node` 需执行工具并返回匹配 `tool_call_id` 的工具结果消息。

```python
from langgraph.graph import StateGraph, MessagesState, START, END


def route_after_model(state: MessagesState):
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else "finish"


builder = StateGraph(MessagesState)
builder.add_node("model", model_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "model")
builder.add_conditional_edges(
    "model", route_after_model, {"tools": "tools", "finish": END}
)
builder.add_edge("tools", "model")
graph = builder.compile()
```

这里假定模型节点最后写入的是包含 `tool_calls` 属性的 AI 消息。每一轮模型和工具都向 `messages` 返回新增消息，由 reducer 合并。

**模型决定要不要调用工具，图决定这个决定如何转化为下一步执行。**

必须设计终止条件，也可用 State 中的次数或预算限制业务循环，并用运行配置中的 `recursion_limit` 限制图执行步数作为保护。触及该限制会报错，并不等同于生成了合格答案。

仅仅有一个环不自动构成 Agent；环里的决策逻辑才决定它是确定性 workflow，还是模型驱动的 Agent。[L3]

## 10. LangGraph 如何表达 Routing？

Routing 是“这次请求下一步交给谁”，Conditional Edge 是最直接的表达方法。

```text
START -> classify -> technical_support -> END
                    -> order_lookup    -> END
                    -> general_answer  -> END
```

典型职责分工：

1. `classify` 节点计算并返回 `{"category": ...}`。
2. router 读取 `state["category"]`。
3. 条件边将分类结果映射到已注册的节点。

分类可以来自规则、结构化模型输出或人工选择。**使用 Graph 不意味着每次路由都要调用 LLM。**

还有一种当前 API：节点返回 `Command(update=..., goto=...)`，在同一个地方表达“更新状态 + 选择后继”：

```python
from typing import Literal
from langgraph.types import Command


def classify(state) -> Command[Literal["technical_support", "general_answer"]]:
    if "error" in state["question"].lower():
        return Command(
            update={"category": "technical"},
            goto="technical_support",
        )
    return Command(
        update={"category": "general"},
        goto="general_answer",
    )
```

这是路由结构示例；相应 State 需包含 `question`、`category`，图中需注册目标节点。

**`Command(goto=...)` 不会取消同一节点已有的静态边。** 不要一边添加无条件后继，一边期待 Command 将其覆盖；入门时为每个节点选择一种清晰的出边策略。[L2][G5]

## 11. LangGraph 如何表达 Parallelization？

### 固定数量的分支：fan-out / fan-in

下面是无需模型的完整示例：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class ParallelState(TypedDict):
    topic: str
    result_a: str
    result_b: str
    summary: str


def search_a(state: ParallelState):
    return {"result_a": f"Source A: {state['topic']}"}


def search_b(state: ParallelState):
    return {"result_b": f"Source B: {state['topic']}"}


def summarize(state: ParallelState):
    return {"summary": state["result_a"] + "; " + state["result_b"]}


builder = StateGraph(ParallelState)
builder.add_node("search_a", search_a)
builder.add_node("search_b", search_b)
builder.add_node("summarize", summarize)
builder.add_edge(START, "search_a")
builder.add_edge(START, "search_b")
builder.add_edge(["search_a", "search_b"], "summarize")
builder.add_edge("summarize", END)
graph = builder.compile()

result = graph.invoke({
    "topic": "LangGraph",
    "result_a": "",
    "result_b": "",
    "summary": "",
})
print(result["summary"])
```

两个检索节点可在同一 super-step 中并行执行；汇总节点等待两者完成。这里分别写不同字段，所以无需为检索结果设置 reducer。

若多个分支共同写 `results`，则应使用合适的 reducer。并行完成或输出到达的先后不应被当作稳定业务顺序；有顺序要求时保存标识，并在汇总时显式排序。

### 动态数量的分支：Send

当任务数由运行时输入决定，例如“为检索到的每篇文档各启动一个摘要任务”，可以让路由函数返回：

```python
from langgraph.types import Send


def dispatch(state):
    return [
        Send("summarize_one", {"document": document})
        for document in state["documents"]
    ]
```

这是动态 map 的结构片段：每个 `Send` 给目标节点一份定制输入，目标节点可以并行执行多次；其结果再通过图状态的 reducer 聚合。实际图还需定义 worker、结果字段和汇总/结束路径，并处理空任务集。

因此要区分：**固定拓扑分支用边，运行时数量不定的工作分发用 `Send`**。并行调度也不承诺无限资源，仍受并发配置、外部 API 限流及运行环境约束。[L2][G2][G5]

## 12. LangGraph 如何保存 Agent State？

**为编译后的图配置 checkpointer，并在执行时使用稳定的 `thread_id`。**

```text
thread_id
   +-> checkpoint 1: State + 执行信息
   +-> checkpoint 2: State + 执行信息
   +-> checkpoint 3: State + 执行信息
```

这里的 thread 是逻辑任务/会话，不是操作系统线程。一个 thread 可以跨多次 `invoke()`。

### 最小接入方式

以下接在一个已经定义好的 `builder` 后面：

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "task-001"}}

result = graph.invoke(initial_state, config)
snapshot = graph.get_state(config)
history = list(graph.get_state_history(config))
```

相同 `thread_id` 在同一 checkpoint 后端中可以找到已有状态；更换 ID 则使用另一条状态历史。`thread_id` 不是权限校验机制，应用仍需限制谁可以访问哪个任务。

`snapshot.values` 是状态值；`snapshot.next` 表示后续节点；快照还包含 metadata、任务和中断等相关信息。**Checkpoint 不只是把 messages 保存成 JSON。** [L4][L5]

### 内存保存不等于跨进程持久化

| 机制 | 用途与边界 |
|---|---|
| `InMemorySaver` | 同一进程内学习、调试；进程结束后数据丢失 |
| SQLite checkpointer | 本地文件持久化，需额外的 `langgraph-checkpoint-sqlite` 包 |
| Postgres checkpointer | 数据库持久化，可用于生产部署；需相应扩展、连接和初始化配置 |
| Store | 存储应用定义的跨 thread 数据，不等同于用于恢复执行的 checkpointer |

SQLite 的连接生命周期示意：

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
    graph = builder.compile(checkpointer=saver)
    result = graph.invoke(initial_state, config)
```

离开 `with` 后连接已关闭，不能继续用它执行图。本文不安装这个扩展。重启后恢复还需要重新建立兼容的图定义，并连接同一持久化后端；不是只凭一个 ID 就能恢复任意程序。[G6][G7]

### 保存的粒度与恢复边界

完整 checkpoint 通常位于 super-step 边界；同一步内还有成功节点的 pending writes，用于失败恢复时复用已完成工作的结果。

这**不是 Python 任意一行的内存/调用栈快照**。失败节点、被 interrupt 的节点可能重新执行；外部调用需要幂等设计，不能从“支持 checkpoint”推导出“外部操作恰好执行一次”。

当前支持的 durability 模式是：

| 模式 | 保存时机与取舍 |
|---|---|
| `sync` | 进入下一步前完成 checkpoint 写入 |
| `async` | 与下一步执行重叠写入；当前默认，崩溃时存在最近写入尚未完成的窗口 |
| `exit` | 退出运行时保存；不能保证中途崩溃后从每一步恢复 |

持久性还取决于后端：给 `InMemorySaver` 使用 `sync`，也不会让它变成磁盘存储。

需要人工修正图状态时，`graph.update_state(...)` 会按 reducer 规则建立新 checkpoint，而不是原地改掉历史；`as_node` 还会影响后续调度，不宜把它当普通字典赋值。[L5][G4]

## 13. LangGraph 如何支持 Human-in-the-loop？

核心组合是：

```text
interrupt(payload)
        +
checkpointer + thread_id
        +
Command(resume=human_answer)
```

**Interrupt 是暂停机制，Human-in-the-loop 是利用这个机制构建的业务交互。**

图可以提出审批或补充信息请求，把请求交还调用方；应用负责显示界面、确认操作者身份、接收并校验答案，再恢复图。LangGraph 不会自动替你创建审批页面或完成权限管理。

### 完整最小示例：只记录人工决定，不执行外部操作

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    proposal: str
    approved: bool


def approval(state: ApprovalState):
    decision = interrupt({
        "question": "Approve this proposal?",
        "proposal": state["proposal"],
    })
    if not isinstance(decision, bool):
        raise ValueError("Approval decision must be a boolean")
    return {"approved": decision}


builder = StateGraph(ApprovalState)
builder.add_node("approval", approval)
builder.add_edge(START, "approval")
builder.add_edge("approval", END)
graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "approval-001"}}

paused = graph.invoke({
    "proposal": "Publish the draft",
    "approved": False,
}, config)
print(paused["__interrupt__"])

# Demo input only; production must obtain an authorized human decision.
human_decision = False
finished = graph.invoke(Command(resume=human_decision), config)
print(finished["approved"])  # False
```

上面两次调用在同一进程内使用同一个内存 checkpointer；若要隔天或重启后审批，应使用持久化后端。

执行过程：

1. 第一次运行到 `interrupt(...)` 时暂停，payload 暴露给调用方。
2. 调用返回，让应用可以等待人工，不是必须一直占住一个 Python 调用。
3. 用同一个 `thread_id` 和 `Command(resume=...)` 再次调用。
4. `approval` 节点从开头重新执行；恢复值成为 `interrupt()` 的返回值。
5. 节点返回审批结果，后续图按业务规则继续。

若要在批准后执行操作，可以在审批节点之后增加条件边：批准进入执行节点，拒绝进入结束或撤销节点。**不要只记录 `approved=False`，却仍用无条件边执行待审批操作。** [L6][G5]

### 最容易误解的地方：不是从 Python 暂停行原地接着跑

**恢复会从包含 interrupt 的节点开头重跑。**

因此 interrupt 前面的发送邮件、扣款或写库等副作用可能重复执行。应把审批与执行职责分开，并对外部操作设计幂等键或可恢复任务；单纯移到 interrupt 后面也不能保证任何故障下都只执行一次。

还需要注意：

- 不要用宽泛 `try/except` 吞掉 interrupt 用来暂停的特殊异常。
- payload 和恢复值采用 JSON 可序列化数据。
- 同一节点多个 interrupt 的恢复值按调用顺序匹配，不要在恢复过程中随意改变调用顺序。
- 并行产生多个 interrupt 时，可用 `Command(resume={interrupt_id: answer, ...})` 分别提供答案。

### 不同“继续运行”方式不能混用

| 场景 | 典型调用 |
|---|---|
| 同一会话的新一轮输入 | `graph.invoke(new_input_dict, config)` |
| 动态 `interrupt()` 等待人工回答 | `graph.invoke(Command(resume=answer), config)` |
| 静态 `interrupt_before` / `interrupt_after` 断点继续 | `graph.invoke(None, config)` |
| 修复失败原因后，从已保存执行状态继续 | 通常 `graph.invoke(None, config)` |

静态断点适合调试或节点边界暂停，不是动态人工交互的首选。不要把旧教程中的 `invoke(None)` 直接替代 `Command(resume=...)` 来回答动态 interrupt。

普通多轮输入也不要写成 `Command(update=...)`：当前官方文档将输入端的 Command 用法限定在恢复 interrupt 的模式；`update`、`goto` 等主要用于节点返回值。[L6][G4][G5]

## 本阶段应该形成的心智模型

用一句话串起来：

> **用 State 记录任务进展，用 Node 做工作，用 Edge 控制执行，用条件边表达选择和循环，用并行分支与 reducer 聚合工作，用 checkpoint 保存进度，再用 interrupt 把人工决定接入同一个执行过程。**

理解 LangGraph 不等于立即开发复杂多 Agent 系统。先分清下面三层即可：

| 层次 | 负责什么 |
|---|---|
| 模型 / 工具 | 生成决策、查询信息、执行具体操作 |
| 图 / 状态 | 决定步骤如何衔接、哪些数据如何合并 |
| 持久化 / 人工交互 | 决定如何暂停、保存、恢复及接纳外部决定 |

如果需求只是“输入 -> 模型 -> 输出”，不用为了使用 Graph 而增加复杂性。当需要显式管理循环、路由、并行与恢复时，LangGraph 的价值才更加明显。

## 官方资料与源码索引

以下为本次研究依据；文中编号对应这些链接。源码链接指向持续更新的 `main`，不代表不可变的历史快照；包版本以访问日的发布页和元数据交叉确认。

### LangGraph 官方文档

- [L1] [官方 GitHub Releases](https://github.com/langchain-ai/langgraph/releases)：核对核心包及各扩展包版本。
- [L2] [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)：State、reducers、nodes、edges、Command、Send、编译。
- [L3] [Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)：模型与工具循环的官方示例。
- [L4] [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)：持久化整体概念。
- [L5] [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)：threads、super-steps、pending writes、state history、durability。
- [L6] [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)：动态暂停、恢复、审批、重执行注意事项。
- [L7] [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)：stream modes 与 v1/v2 输出差异。
- [L8] [Event streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming)：v3 typed projections。

### LangGraph 官方 GitHub 实现

- [G1] [核心包 pyproject.toml](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/pyproject.toml)：`langgraph` 版本及 Python 要求。
- [G2] [graph/state.py](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/state.py)：`StateGraph`、`add_edge`、`add_conditional_edges`、`compile`。
- [G3] [graph/message.py](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/message.py)：`MessagesState`、`add_messages`。
- [G4] [pregel/main.py](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/main.py)：运行时、`invoke`、`stream`、`stream_events`、输出版本与实验性说明。
- [G5] [types.py](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py)：`Send`、`Command`、`interrupt` 及示例。
- [G6] [InMemorySaver 实现](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/checkpoint/memory/__init__.py)：内存后端的适用范围。
- [G7] [SQLite checkpointer 实现](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/__init__.py)：连接上下文管理器。

### OpenAI Agents SDK 官方资料

- [O1] [官方 GitHub](https://github.com/openai/openai-agents-python)：Agent、Runner、工具、handoffs 等主要抽象。
- [O2] [Sessions](https://openai.github.io/openai-agents-python/sessions/)：会话历史的维护及存储后端。
- [O3] [Human-in-the-loop](https://openai.github.io/openai-agents-python/human_in_the_loop/)：审批、interruptions、暂停状态与恢复。
- [O4] [RunState 源码](https://github.com/openai/openai-agents-python/blob/main/src/agents/run_state.py)：运行状态的序列化与重建。
