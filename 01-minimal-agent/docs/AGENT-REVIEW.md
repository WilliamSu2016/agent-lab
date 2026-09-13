# Agent Review

This document reviews the current implementation in `src/` and answers a set of
design questions about why/how this codebase qualifies as an "AI Agent" rather
than a plain script that calls an LLM once.

## 1. 当前实现为什么可以称为 AI Agent？

因为它不是"一次性调用 LLM 拿答案"的脚本，而是具备了 Agent 的核心特征：

- **自主决策**：LLM 在每一轮自己决定"要不要调用工具、调用哪个工具、用什么参数"，
  这个决策不是由 Python 代码硬编码的（`src/agent.py` 的 `run_agent` 只是把 LLM
  的决策结果解析并执行，不替 LLM 做选择）。
- **与环境交互**：LLM 可以通过 `web_search`（`src/tools.py`）主动获取外部世界的
  实时信息，而不是只依赖训练数据里的静态知识。
- **观察—决策循环（loop）**：执行工具后，结果会被送回 LLM，LLM 再基于新证据决定
  是继续搜索还是给出最终答案（`src/agent.py` 第 47 行开始的 `for turn in
  range(...)` 循环）。
- **有边界、可终止的自主行为**：循环受 `max_turns` 约束，并有明确的状态机
  （`running` / `completed` / `limit_reached` / `failed`），符合"目标驱动、
  有限自主"的 Agent 定义。

也就是说，这是一个 **LLM + 工具 + 状态 + 控制循环** 的组合，而不是单纯的
"prompt → completion"调用，因此可以称为（一个最小化的）AI Agent。

## 2. 哪些代码体现了 Agent Loop？

`src/agent.py` 中 `run_agent()` 函数的主循环：

```python
for turn in range(1, max_turns + 1):
    state["turns"] = turn
    response = _call_llm(llm, state["messages"], state["errors"])
    ...
    tool_calls = response.get("tool_calls") or []
    content = response.get("content")
    if not tool_calls:
        ...
        return state          # 完成：给出最终答案

    state["messages"].append(
        {"role": "assistant", "content": content, "tool_calls": tool_calls}
    )
    for tool_call in tool_calls:
        tool_result = _execute_web_search(tool_call, web_search)
        state["tool_calls"].append(tool_result)
        state["messages"].append(
            {"role": "tool", "tool_call_id": tool_result["tool_call_id"], ...}
        )
    # 循环体结束，进入下一轮（下一次问 LLM）
```

这段代码就是标准的 **感知 (Observe) → 决策 (Decide) → 行动 (Act) → 观察结果
(Observe again)** 循环，反复执行直到给出答案或达到轮数上限。

## 3. 哪部分负责 Decision？

**LLM 本身 + `OpenAICompatibleClient.complete()` 的解析结果**是 Decision 的来源。

- `src/llm_client.py`：
  ```python
  return {
      "content": message.get("content"),
      "tool_calls": message.get("tool_calls", []),
  }
  ```
  这里把 LLM 的原始决策（"要不要调用工具、调用哪个、参数是什么"）从 API 响应中
  提取出来，原封不动地交给上层。Python 代码不参与"决定下一步做什么"，只是解析和
  转发。
- `src/agent.py` 中判断 `tool_calls` 是否为空的 `if not tool_calls:` 分支，是在
  **消费**这个决策（是继续调用工具还是结束），而不是在做决策。

一句话：**决策权完全在 LLM**；Python 代码只负责读取 LLM 的决策并据此路由执行。

## 4. 哪部分负责 Action？

`src/agent.py` 中的 `_execute_web_search()` 函数，以及它调用的
`src/tools.py` 中的 `web_search()`：

```python
def _execute_web_search(tool_call, web_search):
    ...
    result = web_search(query)   # 真正执行外部动作
    return {"tool_call_id": call_id, "name": "web_search",
            "status": "success", "content": result}
```

`web_search()`（`src/tools.py`）发起真实的 HTTP 请求（`urlopen`），是唯一让
Agent 真正"作用于外部世界"的代码——这就是 Action 的落地实现。

## 5. 哪部分负责 Observation？

Action 执行后的**结果被结构化并重新写回对话历史 `state["messages"]`**，这一步
就是 Observation：

```python
tool_result = _execute_web_search(tool_call, web_search)
state["tool_calls"].append(tool_result)
state["messages"].append(
    {
        "role": "tool",
        "tool_call_id": tool_result["tool_call_id"],
        "content": json.dumps(tool_result, ensure_ascii=False),
    }
)
```

这条 `role: tool` 消息就是 LLM 在下一轮能"看到"的观察结果（Observation）。
没有这一步，LLM 就无法感知工具执行的结果，Agent Loop 也就无法闭环。

## 6. Tool Result 如何回到 LLM？

流程是：

1. `_execute_web_search()` 返回结构化结果字典（含 `tool_call_id`、`status`、
   `content`/`error`）。
2. `run_agent()` 把这个结果序列化成 JSON 字符串，作为一条 `role: "tool"`
   的消息追加进 `state["messages"]`，并携带对应的 `tool_call_id`（用于让 LLM
   知道这个结果对应哪一次工具调用）。
3. 下一轮循环开始时，`_call_llm(llm, state["messages"], ...)` 把**包含该 tool
   消息在内的完整对话历史**重新发给 LLM API（`llm.complete(messages, tools)`）。
4. LLM 在生成下一个响应时，就能"看到"之前的工具调用结果，并据此决定下一步动作。

即：Tool Result 不是通过特殊通道传回 LLM，而是**作为对话历史的一部分**，随着
下一次完整的 `messages` 列表一起发送给 LLM。

## 7. Agent 如何判断任务已经完成？

判断依据是 **LLM 本轮响应中 `tool_calls` 是否为空，且 `content` 是否是非空
字符串**：

```python
if not tool_calls:
    if not isinstance(content, str) or not content.strip():
        ...                      # 视为异常，非正常完成
    state["messages"].append({"role": "assistant", "content": content})
    state["status"] = "completed"
    state["final_answer"] = content
    return state
```

也就是说，"任务完成"不是 Python 代码自己判断出来的，而是**由 LLM 决定**——当
LLM 认为证据已经足够、不再需要调用 `web_search` 时，它会直接返回纯文本答案
（没有 `tool_calls`），Python 代码据此把这个文本当作最终答案，状态置为
`completed`。系统提示词 `src/prompts.py` 中也明确告诉 LLM 这个规则：
"Do not call a tool when you are ready to provide the final answer."

## 8. Agent 如何防止无限循环？

通过硬性的**轮数上限 `max_turns`**（默认值 5，见 `src/agent.py` 函数签名
`max_turns: int = 5` 和 `src/main.py` 调用处 `run_agent(question, llm,
web_search, max_turns=5)`）：

```python
if max_turns < 1:
    raise ValueError("max_turns must be at least 1")
...
for turn in range(1, max_turns + 1):
    ...
state["status"] = "limit_reached"
state["final_answer"] = (
    "Research stopped because the maximum number of LLM turns was reached."
)
return state
```

只要循环跑满 `max_turns` 次仍未得到最终答案（即 LLM 一直请求工具），`for` 循环
自然结束，代码会强制返回 `limit_reached` 状态，不会无限执行下去。

此外还有两层次要的防护：
- `_call_llm()` 对 LLM API 的连续失败最多重试 2 次（`for attempt in
  range(1, 3)`），失败后返回 `None`，`run_agent` 立即以 `failed` 状态终止，
  不会无限重试。
- 无效/未知工具调用会被 `_execute_web_search()` 转成 `status: "error"` 的
  工具结果返回给 LLM，而不是让程序崩溃或死循环等待，LLM 仍然会消耗一个 `turn`
  额度。

## 9. 如果把 `web_search()` 删除，当前系统还是 Agent 吗？

**从工程功能上说会退化，但从 Agent 定义上说——勉强还是，但已经是"最弱形态"的
Agent，实用性很低。**

- 结构上：`run_agent()` 的循环、状态机、"LLM 决策 → 判断是否完成"逻辑都还在，
  仍然是"LLM 自主决定下一步"的控制流，理论上 Agent Loop 骨架不变。
- 但实际后果：`_execute_web_search()` 里 `if name != "web_search":` 会让所有
  工具调用都变成 `UNKNOWN_TOOL` 错误（因为工具已经不存在/无法真正执行），LLM
  实际上不再能对外部世界产生任何动作（Action 消失），只能靠自身知识直接回答。
  此时它退化为一个"每次都循环 1 轮就结束的、带复杂状态机的普通 LLM 问答
  封装"——Agent 的本质特征之一"能够采取行动改变/感知外部环境"已经丧失。

结论：**技术上循环结构还在，但"Agent 依赖工具与环境交互"这个核心价值已经不存在**，
可以说它退化成了一个"伪装成 Agent 的普通 LLM 调用器"。

## 10. 如果把 LLM 的 Decision 固定写死，它还是 Agent 吗？

**不是。** 如果把"是否调用工具、调用哪个工具"从 LLM 输出改成 Python 代码里
写死的固定逻辑（例如：`if turn == 1: always call web_search` /
`if turn == 2: always return canned answer`），那么：

- 决策权从 LLM 转移到了开发者手写的规则，`tool_calls` 不再来自
  `message.get("tool_calls", [])` 这种"读取 LLM 意图"的方式，而是硬编码的
  控制流。
- 系统就变成了传统的**确定性流程/脚本（if-else pipeline）**，即使表面上仍然
  调用 LLM 生成文本，也只是把 LLM 当作一个"文本生成子程序"使用，而不是让它
  自主判断"该做什么"。
- Agent 的核心定义——"根据观察自主决定下一步行动，以达成目标"——不再成立。

结论：**Decision 被写死后，它就退化为一个普通的自动化脚本 + LLM 文本生成
调用，而不再是 Agent。**（这也解释了为什么问题 1 强调"决策权完全在
LLM"是这个实现称为 Agent 的关键。）

## 11. 当前实现中哪些部分属于 Agent 本质？

这些是让系统"成为 Agent"而不是"普通程序"的关键部分：

| 部分 | 位置 | 为什么是 Agent 本质 |
|---|---|---|
| LLM 自主决策（是否调用工具/调用哪个/给什么参数） | `src/llm_client.py` 中 `message.get("tool_calls", [])`；`src/agent.py` 的 `if not tool_calls:` 分支 | 下一步行动由模型推理产生，而非硬编码规则 |
| Observe→Decide→Act 循环本身 | `src/agent.py` `run_agent()` 的 `for turn in ...` | 体现"持续与环境交互直至达成目标"的自主行为模式 |
| 工具结果回灌给 LLM（形成闭环） | `state["messages"].append({"role": "tool", ...})` | 使 LLM 能基于新证据调整后续决策，这是"感知—决策"闭环的核心 |
| 终止条件由 LLM 输出触发（而非外部强制） | `state["status"] = "completed"` 分支 | Agent 自己判断"目标已达成"，而不是被动地被程序告知结束 |
| 系统提示词定义的行为策略 | `src/prompts.py` | 引导 LLM 何时搜索、何时停止，是 Agent"目标与策略"的体现 |

## 12. 哪些部分只是普通软件工程？

这些部分与"是否是 Agent"无关，属于任何可靠软件都需要的工程实践：

| 部分 | 位置 | 说明 |
|---|---|---|
| HTTP 请求与错误处理 | `src/llm_client.py` 的 `urlopen`/`HTTPError`/`URLError` 处理 | 标准的网络调用容错，与 LLM API 换成任何其它 REST API 无区别 |
| 重试逻辑 | `_call_llm()` 中 `for attempt in range(1, 3)` | 通用的瞬时故障重试策略，任何分布式系统调用都会写 |
| 参数/JSON 校验 | `_execute_web_search()` 中对 `arguments`、`query` 的校验 | 普通的输入验证/防御性编程 |
| 轮数上限 (`max_turns`) 与状态机字段 | `state = {...}`，`status` 取值集合 | 工程上的资源保护与可观测性设计，不属于"智能"部分 |
| DuckDuckGo 结果解析 | `src/tools.py` 的 `web_search()` 中解析 `AbstractURL`/`RelatedTopics` | 纯粹的第三方 API 响应数据整形逻辑 |
| CLI 入口与环境变量加载 | `src/main.py` | 常规应用启动脚手架（argv 解析、`.env` 加载、错误提示） |
| 类型协议 `LLMClient(Protocol)` | `src/agent.py` | 纯 Python 类型系统/可测试性设计，与 Agent 智能行为无关 |

**总结**：Agent 的"智能"完全来自 LLM 的自主决策能力和 Observe-Decide-Act 循环
的闭环设计；其余的 HTTP 调用、重试、校验、状态管理、CLI 封装都是保证这个循环
能稳定、安全运行的**普通工程基础设施**，去掉它们不会让系统"不是 Agent"，只会
让它"不健壮"。
