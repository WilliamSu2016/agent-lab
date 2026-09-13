# 04 - Agent Loop 与 Conditional Edge

对应代码：`src/04_agent_loop.py`
对应测试：`tests/test_agent_loop.py`

本实验把手写的 Agent Loop：

```python
while True:
    response = LLM(...)
    if tool_call:
        execute_tool()
        continue
    return final_answer
```

用 LangGraph 的 **Conditional Edge** 重新表达为：

```
START
  |
  v
agent
  |
  v
should_continue        (routing function，不是 Node)
  |-- "tools" --> tools --> agent   (Tool 结果回到 agent，循环由 Edge 构成)
  `-- "end"   --> END
```

## State

```python
class AgentState(TypedDict):
    question: str
    messages: list[dict]          # 完整对话历史（user / assistant / tool）
    pending_tool_call: ToolCall | None
    final_answer: str
    steps: int                    # 已执行的 agent 步数
```

- `messages`：由 `agent` 追加 assistant 消息、由 `tools` 追加 tool 消息，
  两个 Node 都只**追加**，从不删除历史。
- `pending_tool_call`：`agent` 请求工具时写入，`tools` 执行完后清空——正是
  这个字段驱动了 `should_continue` 的路由决策。
- `final_answer`：只有 `agent` 判定"不再需要工具"时才写入，是否非空决定了
  Graph 是否结束。
- `steps`：`agent` 每次执行自增，用来实现"最大循环次数"这个安全阀。

## agent Node

```python
def agent(state: AgentState) -> AgentState:
    steps = state["steps"] + 1
    if steps > max_steps:                       # 需求 8：硬性上限
        return {"steps": steps, "pending_tool_call": None,
                "final_answer": "Stopped after reaching the maximum ..."}

    decision = llm_call(state["messages"])       # 需求 10/11：普通函数调用，不是 Agent SDK
    if "tool_call" in decision:
        return {..., "pending_tool_call": decision["tool_call"], "final_answer": ""}
    else:
        return {..., "pending_tool_call": None, "final_answer": decision["final_answer"]}
```

`llm_call` 是一个可注入的 `Callable[[messages], decision]`（生产环境用
`build_openai_llm_call()` 包装最基础的 `openai.chat.completions.create(..., tools=[...])`，
测试里用 `ScriptedLLM` 完全离线注入）。它只返回"工具调用"或"最终答案"二选一，
`agent` Node 据此决定写哪些字段——这一步完全对应原始 `while True` 里的
`response = LLM(...)` + `if tool_call: ... else: return`。

## tools Node

```python
def tools(state: AgentState) -> AgentState:
    tool_call = state["pending_tool_call"]
    try:
        result = search_web_call(tool_call["args"]["query"])
        tool_message = {"role": "tool", "content": result}
    except Exception as exc:                     # 需求 9：Tool 错误安全处理
        tool_message = {"role": "tool", "content": {"error": str(exc)}}
    return {"messages": state["messages"] + [tool_message], "pending_tool_call": None}
```

任何工具异常（网络失败、未知工具名等）都在这里被捕获，转换成一条
`{"role": "tool", "content": {"error": ...}}` 消息，**永远不会让异常穿透
到 Graph 之外**——`agent` 会在下一次循环里看到这条错误消息，可以据此改变
策略（换个查询词、或者直接放弃搜索给出答案），而不是让整个程序崩溃。

## should_continue：Conditional Edge 的路由函数

```python
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    if state["pending_tool_call"] is not None:
        return "tools"
    return "end"
```

```python
graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END},
)
graph_builder.add_edge("tools", "agent")   # 需求 6：Tool 结果回到 agent
```

`should_continue` 本身**不是 Node**——它不出现在 `add_node` 里，只是一个
"给定当前 State，返回下一步该走哪条边"的纯函数，通过 `add_conditional_edges`
注册给 `agent` 这个 Node 之后使用。`tools -> agent` 是一条普通、无条件的边，
这正是"循环"在图里的体现：`agent` 执行完后如果需要工具，就绕道 `tools`
再绕回 `agent`，如此往复，直到 `should_continue` 返回 `"end"`。

## 运行 & 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agent_loop.py -v
```

真实运行（需要 `OPENAI_API_KEY`/`OPENAI_MODEL`）：

```powershell
.\.venv\Scripts\python.exe -m src.04_agent_loop "What is LangGraph?"
```

---

## 重点问题解答

### 1. LangGraph 如何表示 Agent Loop？

手写 `while True` 循环把"要不要继续"这个判断和"具体做什么"这两件事混在同一
段命令式代码里。LangGraph 把它拆成三个独立、可分别测试的部分：

- **agent Node**：一次"决策"——读 State，返回"要工具"还是"要结束"。
- **tools Node**：一次"执行"——读 `pending_tool_call`，执行工具，把结果写回
  State。
- **Edge（`tools -> agent`）+ Conditional Edge（`agent -> tools`/`agent -> END`）**：
  循环本身，用图结构表达，而不是用 Python 的 `while`/`continue`/`break`。

`while True` 的每一次迭代，对应 Graph 里"进入 `agent` 一次"；`continue`
对应"从 `tools` 沿着边回到 `agent`"；`return final_answer` 对应
"`should_continue` 返回 `\"end\"`，边指向 `END`"。循环次数、当前处于哪一步、
历史消息，全部都是显式的 State 字段，而不是调用栈里稍纵即逝的局部变量。

### 2. Conditional Edge 与普通 if/else 有什么区别？

普通 `if/else` 是**函数内部**的控制流：它决定的是"接下来在这个函数里执行
哪几行代码"，分支逻辑和业务逻辑写在同一个作用域里，外部完全看不到"这里有
一个分支点"，除非去读函数体。

Conditional Edge 把"分支"提升成了**图结构的一部分**：

```python
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
```

- 路由函数 `should_continue` 只做一件事——根据 State 返回一个"边的名字"，
  不执行任何业务逻辑，因此可以完全独立于 `agent`/`tools` 的实现被单元测试
  （见 `RoutingFunctionTest`）。
- 分支的两个出口（`"tools"`/`"end"`）在构建图的时候就必须显式声明去哪个
  Node，这个映射关系是可以被读取、可视化、静态检查的（比如 LangGraph 能
  画出图结构），而 `if/else` 分支只有运行到那一行代码才"显形"。
- 因为路由决策和执行逻辑（`agent`/`tools` 的 Node 函数体）是分离的，想
  "改路由策略"（比如加一个 `"retry"` 分支）不需要改动 `agent`/`tools` 内部
  代码，只需要改 `should_continue` 和 `add_conditional_edges` 的映射表。

换句话说：`if/else` 是"藏在函数体里、命令式"的分支；Conditional Edge 是
"暴露在图结构里、声明式"的分支。

### 3. State 在循环过程中如何变化？

以"agent 请求一次工具搜索，再给出最终答案"为例，`invoke()` 期间 State 的
演变（打印语句会展示每一步的完整快照）：

| 阶段                | steps | pending_tool_call                  | messages 最后一条        | final_answer |
|---------------------|-------|-------------------------------------|---------------------------|---------------|
| 初始                | 0     | `None`                              | `{"role": "user", ...}`   | `""`          |
| `agent` 第 1 次执行后 | 1     | `{"name": "search_web", ...}`       | assistant（工具调用）      | `""`          |
| `should_continue` → `"tools"` | | | | |
| `tools` 执行后        | 1     | `None`                              | `{"role": "tool", ...}`   | `""`          |
| （沿 `tools -> agent` 边回去） | | | | |
| `agent` 第 2 次执行后 | 2     | `None`                              | assistant（最终答案）      | `"Paris ..."` |
| `should_continue` → `"end"`   | | | | |

关键点：

- `messages` 只增不减——每一次进出 `agent`/`tools`，State 都变得"更长"，
  完整保留了对话历史，这也是下一次 `agent` 调用能"看到"上一次工具结果的
  原因（LLM 输入永远是完整的 `messages`）。
- `steps` 单调递增，是循环终止条件之一。
- `pending_tool_call` 在"请求工具"和"工具已执行"之间来回切换（有值 ->
  `None`），正是它驱动了 `should_continue` 的分支。
- `final_answer` 从空字符串变为非空，是循环真正终止（走向 `END`）的信号。

### 4. 为什么 Graph 比 while loop 更容易观察？

`while True` 循环的"进度"只存在于调用栈和局部变量里：想知道"当前是第几轮、
上一轮做了什么、为什么还没结束"，唯一办法是加 `print`/`pdb` 断点去跟踪运行
中的那一个进程，循环结束后这些信息就随栈帧一起消失了。

LangGraph 的 Graph 把"进度"变成了**显式、可观察的数据和结构**：

- **每一步都是一次可数、可命名的事件**：`agent` 执行了几次、`tools` 执行了
  几次，直接对应 State 里的 `steps` 和 `messages` 长度，而不用额外埋点。
- **状态是可打印、可持久化的**：本例中 `_print_state()` 在每个 Node 前后
  打印完整 State（如上表），这在 `while` 循环里当然也能做到，但 LangGraph
  进一步允许（后续实验）把每一步的 State 做 Checkpoint，运行结束后仍然可以
  回放"第 N 步的 State 是什么样的"。
- **图结构本身可视化**：`agent -> tools -> agent` 的循环、`agent -> END`
  的终止条件，是构建阶段就声明好的静态结构，可以被 LangGraph 的可视化工具
  画成流程图；而 `while True` 里的分支/循环结构只存在于源代码的缩进里。
- **每个环节都能单独测试**：本实验的测试分别验证了 `agent` Node、`tools`
  Node、`should_continue` 路由函数、以及完整 Graph 的多轮循环——因为它们是
  相互独立的函数，而不是耦合在一个大 `while` 循环体内部的代码块。

简言之：`while` 循环把"发生了什么"埋在一次性的调用栈里，Graph 把"发生了
什么"变成了外部可读、可测试、可持久化的显式数据（State）和结构（Node +
Edge），这正是"更容易观察"的来源。
