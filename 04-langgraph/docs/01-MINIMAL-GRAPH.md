# 01 - 最小 LangGraph Graph

对应代码：`src/01_minimal_graph.py`
对应测试：`tests/test_minimal_graph.py`

本示例**不使用 LLM、不使用 Agent、不使用 Tool**，只使用 LangGraph 最核心的构建
模块，帮助理解 LangGraph 到底是什么。

流程：

```
START
  ↓
greet
  ↓
format
  ↓
END
```

## State（状态）

State 是在整个 Graph 中流动、被各个 Node 读取和更新的**共享数据结构**。

```python
class GraphState(TypedDict):
    name: str
    message: str
```

- 用 `TypedDict`（或 `pydantic.BaseModel`）明确声明字段和类型，这就是"定义明确
  的 State"。
- 每个 Node 接收当前 State，返回一个**部分更新**（只包含它想修改的字段），
  LangGraph 会把这个返回值合并（merge）回主 State。
- 本例中，`name` 从头到尾不变，`message` 先被 `greet` 写入，再被 `format`
  覆盖。

## Node（节点）

Node 就是一个普通的 Python 函数：**输入 State，输出（部分）State**。它不涉及
任何 LLM 调用、工具调用或决策逻辑，只是纯函数式的数据处理。

```python
def greet(state: GraphState) -> GraphState:
    return {"message": f"Hello, {state['name']}!"}

def format_message(state: GraphState) -> GraphState:
    return {"message": f">>> {state['message']} <<<"}
```

- `greet`：读取 `state["name"]`，生成 `"Hello, {name}!"`，写回 `message`。
- `format`：读取上一步产生的 `message`，把它转换成最终输出格式（这里用
  `>>> ... <<<` 包裹，代表"最终格式化"）。

用 `graph_builder.add_node("greet", greet)` 把函数注册为 Graph 中的一个具名
节点。

## Edge（边）

Edge 定义了 Node 之间的**执行顺序**——上一个 Node 执行完之后，控制流应该走向
哪一个 Node。本例是最简单的线性边（无分支、无条件）：

```python
graph_builder.add_edge(START, "greet")
graph_builder.add_edge("greet", "format")
graph_builder.add_edge("format", END)
```

即：`START -> greet -> format -> END`。

## START / END

- `START` 是 LangGraph 内置的特殊节点，代表 Graph 的**入口**。任何 Graph 都
  必须有一条从 `START` 出发的边，指向第一个真正执行的 Node。
- `END` 是内置的特殊节点，代表 Graph 的**出口**。当执行走到 `END`，说明这次
  `invoke()` 结束，最终 State 会被返回。

`START` 和 `END` 都从 `langgraph.graph` 导入：

```python
from langgraph.graph import END, START, StateGraph
```

## compile()

`StateGraph` 只是一个"构建器"（builder），描述了 Node 和 Edge 的静态结构。
调用 `.compile()` 会把这个结构编译成一个**可执行的 Runnable 图对象**：

```python
graph = graph_builder.compile()
```

编译之后的 `graph` 才能真正被调用、执行；编译前只是配置阶段。

## invoke()

`invoke(initial_state)` 是真正**运行**这个 Graph 的方式：给定一个初始 State，
LangGraph 按照 Edge 定义的顺序依次执行每个 Node，把每个 Node 的返回值合并进
State，直到到达 `END`，最后返回**最终的 State**。

```python
result = graph.invoke({"name": "LangGraph", "message": ""})
print(result)
# {'name': 'LangGraph', 'message': '>>> Hello, LangGraph! <<<'}
```

执行过程：

1. `START -> greet`：`greet` 读到 `name="LangGraph"`，把 `message` 设为
   `"Hello, LangGraph!"`。
2. `greet -> format`：`format` 读到上一步的 `message`，把它转换成
   `">>> Hello, LangGraph! <<<"`。
3. `format -> END`：Graph 结束，返回最终 State。

## 运行

```powershell
.\.venv\Scripts\python.exe -m src.01_minimal_graph
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_minimal_graph.py -v
```

## 小结：概念对照表

| 概念    | 是什么                          | 本例中的体现                                   |
|---------|----------------------------------|------------------------------------------------|
| State   | 在 Graph 中流动的共享数据结构    | `GraphState`（`name`, `message`）              |
| Node    | 接收/更新 State 的普通函数       | `greet`、`format_message`                      |
| Edge    | Node 之间的执行顺序              | `START->greet`, `greet->format`, `format->END` |
| START   | Graph 的入口特殊节点             | `add_edge(START, "greet")`                     |
| END     | Graph 的出口特殊节点             | `add_edge("format", END)`                      |
| compile | 把构建好的结构编译成可执行图     | `graph_builder.compile()`                      |
| invoke  | 用初始 State 运行整个 Graph      | `graph.invoke({...})`                          |

本示例完全不涉及 Agent 能力（没有 LLM 决策、没有工具调用、没有循环/分支），
只是纯粹的、确定性的线性 Graph，是理解后续更复杂 LangGraph 模式（Conditional
Edge、Agent Loop 等）的基础。
