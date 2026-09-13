# 03 - Prompt Chaining，重写为 LangGraph Graph

对应代码：`src/03_prompt_chaining_graph.py`
对应测试：`tests/test_prompt_chaining_graph.py`

本实验把 `src/prompt_chaining.py`（基于 OpenAI Agents SDK 的固定四步流水线：
`Planner -> Researcher -> Analyst -> Writer`）用 **LangGraph `StateGraph`**
重新实现为：

```
START
  ↓
planner        -- 只写 dimensions
  ↓
researcher     -- 只写 research_notes（内部调用 search_web tool）
  ↓
analyst        -- 只写 analysis
  ↓
writer         -- 只写 final_answer
  ↓
END
```

约束（与需求一一对应）：

- 使用 `StateGraph`，每一步是一个 Node，Node 之间用 `add_edge` 连接。
- 中间结果全部通过 `PromptChainState` 传递。
- **不使用** `openai-agents` SDK（没有 `Agent`/`Runner`/工具调用循环）。
- **不使用** LangChain 的 Agent 抽象（没有 `AgentExecutor`/`create_react_agent`）。
- **不使用** Multi-Agent（只有一条线性执行路径）。
- 保留 `search_web`（`src/tools.py`）这个 Tool，由 `researcher` Node 直接调用。
- 暂不使用 Conditional Edge / Persistence / Human-in-the-loop（后续实验再引入）。

## State

```python
class ResearchNote(TypedDict):
    dimension: str
    findings: str
    sources: list[str]

class PromptChainState(TypedDict):
    question: str
    dimensions: list[str]
    research_notes: list[ResearchNote]
    analysis: str
    final_answer: str
```

和实验 #2 一样，每个字段都由唯一一个 Node 负责写入，其余 Node 只读不写。

## LLM 调用方式：为什么不是 Agent，却仍然能调用模型？

`src/prompt_chaining.py` 里每一步都是一个 `agents.Agent`，真正的 LLM 调用、
JSON 解析、工具调用循环全部由 Agents SDK 的 `Runner` 负责。本实验刻意去掉了
这一整层：

```python
LLMCall = Callable[[str, str], str]   # (system_prompt, user_prompt) -> 文本
```

`LLMCall` 只是一个"系统提示 + 用户提示 -> 文本"的最小函数签名，生产环境下由
`build_openai_llm_call()` 用**最基础的** `openai.OpenAI().chat.completions.create()`
实现（不是 Agents SDK，也不是 LangChain 的 Agent），测试里则直接注入一个纯
Python 的 fake 函数，完全离线、无需网络和 API Key。

每个 Node 通过闭包持有这个 `llm_call`（以及 `researcher` 额外持有的
`search_web_call`），而不是读取全局变量或自己构造 SDK 客户端——这正是延续
实验 #1/#2 的"Node 是纯函数、依赖通过参数/闭包注入"的原则。

## Planner → Researcher → Analyst → Writer 如何映射成 Node → Node → Node → Node

| 原始 `prompt_chaining.py`                              | LangGraph 版本                                             |
|----------------------------------------------------------|--------------------------------------------------------------|
| `build_planner()` 返回一个 `Agent(output_type=ResearchPlan)` | `make_planner(llm_call)` 返回一个普通函数 `planner(state)`      |
| `Runner.run_sync(planner, question)`，SDK 负责调用模型 + 解析 JSON | `planner` Node 直接调用 `llm_call(...)`，自己用 `json.loads` 解析 |
| `build_researcher()` 是带 `search_web` 工具的 `Agent`，由 LLM 的工具调用循环决定何时搜索 | `researcher` Node 是**固定的 Python 代码**：对每个 dimension 都直接调用一次 `search_web`，工具调用不再由模型决定 |
| `build_analyst()` / `build_writer()` 是无工具、纯文本输出的 `Agent` | `analyst`/`writer` Node 直接调用 `llm_call(...)`，返回值就是纯文本 |
| 四个 `Runner.run_sync()` 调用，在 `run_prompt_chain()` 里按固定顺序手写串联，用 `try/except` 包一层 `PromptChainStepError` 报告是哪一步失败 | 四个 Node 通过 `add_edge` 串成 `planner -> researcher -> analyst -> writer`，顺序由 Graph 结构本身声明，而不是靠函数调用顺序 |
| 每一步的输入靠手写的字符串拼接（`f"..."`），显式地"只用上一步的输出" | 每一步的输入就是当前 `PromptChainState`，天然只包含之前 Node 写入过的字段 |

核心映射关系可以总结为：

- **Agent（一份 instructions + 一个模型 + 可选工具）→ Node（一个纯函数，
  通过闭包持有 `llm_call`，需要时直接调用工具）**。
- **`Runner.run_sync()` 的"调用模型 -> 解析结构化输出 -> 返回"这一整套逻辑 →
  Node 函数体自己完成**（调用 `llm_call`，自己 `json.loads`/字符串处理）。
- **`run_prompt_chain()` 里手写的顺序调用 → `StateGraph` 的
  `add_edge(START, "planner") -> ... -> add_edge("writer", END)`**：执行
  顺序从"命令式的函数调用序列"变成了"声明式的图结构"。
- **原本由 Agent 的工具调用循环决定"是否调用 `search_web`"的决策 →
  在 `researcher` Node 里变成了确定性的 Python `for` 循环**：因为这里没有
  Agent Loop，工具什么时候调用必须由固定的流水线代码显式决定，这也正是
  "Prompt Chaining"（区别于真正的 Agent）的定义——步骤和分支是预先写死的，
  不是模型在运行时决定的。

## 为什么 LangGraph 比普通函数调用更适合表达复杂 Workflow？

`src/prompt_chaining.py` 的 `run_prompt_chain()` 本质上就是四次
`Runner.run_sync()` 加上手写的 `try/except`/字符串拼接，对于四步的线性流水线
完全够用。但当 Workflow 变得更复杂时，"手写函数调用序列"这种方式会开始暴露
问题，而这些问题正是 LangGraph 从设计上要解决的：

1. **执行顺序是隐式的，还是显式的？**
   普通函数调用里，执行顺序 = 代码书写顺序，想知道"某一步之后会发生什么"
   必须读代码、跟踪调用栈。LangGraph 用 `add_edge`/`add_conditional_edges`
   把执行顺序显式声明成一张图，"下一步是什么"是图结构的一部分，可以在运行
   前被检查、可视化、甚至用工具自动生成流程图。

2. **要加分支/循环时，代码复杂度是线性增长还是爆炸增长？**
   本实验暂时只有线性流程，但下一步的实验会加 Conditional Edge（比如"如果
   `research_notes` 证据不足，就跳回 `researcher` 再搜一次，而不是直接进
   `analyst`"）。用普通函数调用实现这种分支/回环，往往要写一堆
   `if/else` + 手动维护"当前处于哪一步"的状态机；LangGraph 的
   Conditional Edge 让"路由到哪个 Node"变成一个返回 Node 名字的纯函数，图的
   结构本身就能表达循环、分支、汇合，而不需要在业务函数里混入控制流逻辑。

3. **中断/恢复、持久化怎么办？**
   普通函数调用一旦执行到一半崩溃，所有中间结果（局部变量）都丢失，重跑
   只能从头开始。LangGraph 的 State 是显式、可序列化的数据结构，配合
   Checkpoint（后续实验）可以在任意一个 Node 之间保存/恢复整个 Graph 的
   执行进度——这对"长流程、可能中断、需要人工介入"的 Workflow 至关重要，
   而这在手写函数调用里几乎无法优雅实现。

4. **谁拥有哪部分状态，边界是否清晰？**
   普通函数调用中，"哪个变量该传给下一个函数"完全靠开发者手动决定和维护
   （容易多传、少传、或者不小心共享了可变对象）。LangGraph 强制所有数据
   都通过一个显式类型的 State 流动，每个 Node 的输入输出边界因此是显式、
   可检查、可测试的（正如本实验和前两个实验的测试所展示的）。

5. **多个独立步骤能否并行执行？**
   如果 Researcher 需要对多个独立维度分别调用模型/工具，普通函数调用要么
   顺序执行（慢），要么手写线程池/`asyncio.gather`（增加复杂度）。
   LangGraph 支持在图结构层面声明并行分支（多个 Node 从同一个上游 Node
   出发，各自独立执行后再汇合），并行度是图结构的一部分，而不需要在业务
   代码里手写并发原语。

简而言之：四步都是线性、无分支、无中断需求时，普通函数调用完全够用，
`src/prompt_chaining.py` 的手写实现并不"错"。但一旦 Workflow 需要
**条件路由、循环、持久化/恢复、清晰的状态边界、或者并行执行**，把执行顺序
和状态管理"写死在函数调用里"就会变得脆弱、难以扩展；而 LangGraph 把这些
关注点都变成了图（Node + Edge + State）的一等公民，这正是它比"普通函数调用"
更适合表达复杂 Workflow 的原因。

## 运行 & 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_prompt_chaining_graph.py -v
```

运行真实流水线（需要设置 `OPENAI_API_KEY`、`OPENAI_MODEL`，可选
`OPENAI_BASE_URL`）：

```powershell
.\.venv\Scripts\python.exe -m src.03_prompt_chaining_graph "你的问题"
```
