# 从手写最小 AI Agent 到 OpenAI Agents SDK

## 阅读范围与版本依据

本文只做分析和职责映射，不迁移、不删除、不修改任何源代码或依赖。

查阅日期：**2026-09-08（UTC+08:00）**。依据为 OpenAI Agents SDK **Python** 官方在线文档及官方仓库，不混用 JavaScript SDK、Assistants API 或旧版 Swarm 的接口。

| 依据 | 本次读取结果 |
| --- | --- |
| GitHub 最新正式 Release | `v0.22.0`，发布于 `2026-08-19T13:44:38Z`，不是预发布版本 |
| 官方仓库主分支快照 | `cb808dfaee4f5f6bc256abc4f6f021a4693c397c` |
| 主分支 `pyproject.toml` | 包名 `openai-agents`，版本字段 `0.22.0`，Python `>=3.10` |
| Python 导入命名空间 | `agents`，例如 `from agents import Agent, Runner, function_tool` |
| 当前本地项目依赖 | `requirements.txt` 只有 `python-dotenv>=1.0,<2.0`，没有声明 Agents SDK |

**Release、主分支和在线文档不是同一个版本快照。** 主分支即使仍写着 `0.22.0`，也可能包含正式发布之后的更改。本文以当前官方文档解释概念，以固定主分支提交核对实现；另外直接核对了 `v0.22.0` 的 `Runner`、`function_tool` 和运行配置源码。默认 `max_turns=10`、支持 `max_turns=None`、未知工具默认报错等关键行为在该正式版本中也已存在。正式迁移时应明确锁定依赖版本，而不是将主分支的所有细节都假定为已发布能力。

本文中的“第⑤步”以当前工作目录中的手写实现为准；目录内没有单独的第⑤步讲义，不额外假设此前存在其他代码。[V1][V2][V3]

## 一、先看当前项目：我们已经手写了哪些 SDK 职责

当前实现不是“只请求一次模型”的聊天脚本，而是一个有工具反馈的有界循环：

```text
用户问题
  -> 初始化 state、system/user messages
  -> 调用 Chat Completions
  -> 有 tool_calls？
       有：解析工具名称与参数 -> 执行 web_search
           -> 追加 assistant tool_calls 和 tool message
           -> 再次调用模型
       无：检查回答内容 -> 返回 completed
  -> 模型调用失败：返回 failed
  -> 循环达到上限：返回 limit_reached
```

| 当前文件 / 符号 | 已经承担的职责 |
| --- | --- |
| `src\prompts.py`：`SYSTEM_PROMPT` | 研究助手角色、何时搜索、如何使用证据、何时继续搜索、何时回答 |
| `src\tools.py`：`WEB_SEARCH_TOOL` | 手写工具名称、描述和参数 JSON Schema |
| `src\tools.py`：`web_search()` | 查询校验、调用 DuckDuckGo Instant Answer endpoint、整理最多五条结果 |
| `src\llm_client.py`：`OpenAICompatibleClient.complete()` | 拼接 Chat Completions 请求、HTTP 调用、响应解析和异常转换 |
| `src\agent.py`：`run_agent()` | 初始化状态、执行 Agent Loop、记录消息、判断结束、控制轮次 |
| `src\agent.py`：`_execute_web_search()` | 工具 dispatch、参数解析与校验、执行工具、封装结果和错误 |
| `src\agent.py`：`_call_llm()` | 最多两次调用尝试、首次失败后等待一秒、记录模型错误 |
| `src\main.py`：`main()` | 加载环境变量、读取问题、组装依赖、启动运行、输出应用状态和答案 |

迁移的核心不是“让 SDK 替我们思考”，而是：**保留任务定义和业务能力，把通用的运行循环与工具协议交给 SDK。**

## 二、12 个核心问题

### 1. OpenAI Agents SDK 中 Agent 是什么？

`Agent` 是一个配置了模型、指令、工具及可选运行行为的智能体定义。

最小理解可以写成：

```text
Agent = 模型配置 + Instructions + 可用工具 + 可选行为配置
```

常用属性包括 `name`、`instructions`、`model`、`model_settings`、`tools`；还可配置 `output_type`、guardrails、handoffs 和 hooks。

**创建 `Agent(...)` 本身不会启动模型请求，也不会自己进入循环。** 它更接近“这个智能体是谁、能用什么、应该怎么做”的定义，而不是一项已经在执行的后台任务。

对当前项目而言，原本分散在 `SYSTEM_PROMPT`、`WEB_SEARCH_TOOL` 和 `llm` 配置中的定义会集中到 `Agent` 上。真正查询 DuckDuckGo 的实现仍是我们的 Python 函数。

它也不是天然携带永久会话记忆的对象：反复复用同一个 `Agent`，不等于自动保留前一次 `Runner.run()` 的历史。[D1][S1]

### 2. Runner 是什么？

`Runner` 是运行智能体的公开入口，负责把 `Agent` 的定义变成实际执行过程。

它接收起始 Agent、用户输入以及可选的 `context`、`session`、`max_turns`、`run_config` 等参数，协调模型调用、工具执行、结果回填和终止，并返回运行结果。

| 公开方法 | 调用方式与返回结果 |
| --- | --- |
| `Runner.run()` | 异步方法，使用 `await Runner.run(...)`，得到 `RunResult` |
| `Runner.run_sync()` | 同步封装，得到 `RunResult`；适合当前同步 CLI 入口 |
| `Runner.run_streamed()` | 返回 `RunResultStreaming`，再异步消费 `stream_events()`；不是写成 `await Runner.run_streamed(...)` |

因此，`Runner.run()` **不等于一次底层模型请求**。一次 run 可以包含多轮模型调用、多个工具调用，甚至 handoff 或等待审批。

当前项目的对应入口是 `run_agent(...)`，不是 `llm.complete(...)`。[D2][D4][S2]

### 3. Agent Loop 是由谁负责执行的？

**由 SDK 的 Runner 运行时负责。**

需要区分三种责任：

| 责任 | 承担者 |
| --- | --- |
| 决定下一步输出答案还是请求工具，以及生成工具参数 | 模型，在 instructions 和工具定义约束下生成输出 |
| 解释模型输出、执行下一步、维护循环和停止条件 | SDK Runner |
| 定义任务规则、提供工具能力、处理应用层结果 | 我们的应用 |

对于当前这种“普通文本输出 + 本地 function tool + 默认工具行为”的 Agent，循环是：

```text
Runner 收集 instructions、输入历史和工具定义
  -> 调用模型
  -> 如果有本地 function tool call：
       查找工具 -> 解析/校验参数 -> 执行函数
       -> 生成工具输出 item -> 加入后续模型输入
       -> 下一轮
  -> 如果产生可接受的最终文本且没有待执行工具调用：
       结束，返回 RunResult
  -> 如果下一轮会超过 max_turns：
       默认抛出 MaxTurnsExceeded
```

当前 `src\agent.py:49` 的 `for turn in range(1, max_turns + 1)` 及其内部通用调度分支，可以由该运行时替代。

这张图描述的是当前项目需要的默认路径，不是所有配置的唯一停止规则：结构化输出、工具直接作为最终输出、handoff、guardrails 和审批中断会影响执行路径。不能把它理解成 SDK 永远只检查一个字符串是否为空。[D1][D2][S2]

### 4. Tool Call 是由谁负责 dispatch 的？

**普通本地 function tool 的 dispatch 由 SDK Runner 及其工具执行层负责。**

模型返回工具调用请求，包含工具名称、参数和调用标识；模型并不直接运行 Python，也不会访问我们的 Python 函数注册表。

SDK 根据当前 Agent 可用的工具找到对应 `FunctionTool`，通过其调用包装层执行我们提供的函数。装饰器负责构建工具描述和调用适配；**触发 dispatch 的运行时是 Runner，而不是装饰器自己。**

当前项目的对应代码是 `_execute_web_search()` 中读取 `function.name`、判断 `name == "web_search"`、解析 `arguments`，以及调用 `web_search(query)` 的这一组基础设施。

有两个不能忽略的边界：

| 情况 | SDK 行为 / 应用责任 |
| --- | --- |
| 模型请求不存在的工具 | 当前默认抛出 `ModelBehaviorError`；可设置 `RunConfig(tool_not_found_behavior="return_error_to_model")`，将错误回传模型 |
| 工具需要访问外部服务或受保护资源 | SDK 负责调用；网络操作、鉴权、业务权限检查和副作用仍由工具实现负责 |

当前手写实现对未知工具返回 `UNKNOWN_TOOL` 并继续循环，因此 SDK 的默认未知工具行为与当前实现不完全一致。

这里讨论的是 `FunctionTool`。官方托管的 `WebSearchTool()` 等工具在服务端执行，不能笼统地说“所有 SDK 工具都是在我们的 Python 进程里执行”。[D2][D3][S3][S4]

### 5. Tool Result 如何重新进入 Agent Loop？

**工具返回值经 SDK 转换成工具输出 item，关联原始调用，再由 Runner 纳入下一次模型输入。**

默认本地 function tool 路径可以理解为：

```text
模型生成 function call，call_id = A
  -> Runner 执行对应 Python 函数
  -> 函数返回搜索结果
  -> SDK 构造工具输出，关联 call_id = A
  -> 保留本轮模型输出和工具输出
  -> 下一轮模型同时看到之前的问题、工具调用和工具结果
```

当前项目显式追加：

```python
{"role": "assistant", "content": content, "tool_calls": tool_calls}
{"role": "tool", "tool_call_id": call_id, "content": serialized_result}
```

SDK 使用 Responses 风格的运行 items；普通 function tool 的输出可表示为：

```python
{
    "type": "function_call_output",
    "call_id": call_id,
    "output": serialized_result,
}
```

`ToolCallOutputItem` 是 SDK 层的包装对象，包含工具输出及关联元数据。模型适配器再将运行 items 转成所选 API 需要的协议格式；使用 Chat Completions 适配器时，线上请求仍使用该 API 的消息格式。

所以，无须继续手写 `messages.append(...)` 来关联调用和结果，也不需要从工具内部递归调用 `Runner.run()`。

**“工具结果会回到模型”有配置前提：** `Agent.tool_use_behavior` 默认是 `"run_llm_again"`，符合当前“搜索后让模型综合证据”的行为。若使用 `"stop_on_first_tool"`、`StopAtTools` 或自定义工具终止策略，工具结果可以直接结束 run，不再经过下一次模型综合。

还要区分“SDK 生成协议信封”和“业务结果格式”：SDK 不承诺复刻目前 `status/content/error` 字典或 `json.dumps(..., ensure_ascii=False)` 的文本。如果需要稳定的 JSON 输出格式，应由工具或轻量适配层显式序列化。[D1][D3][D4][S4][S5]

### 6. max_turns 如何控制？

在 Runner 调用处设置：

```python
result = await Runner.run(agent, question, max_turns=5)
```

`max_turns` 是 Runner 参数，不是 `Agent.instructions` 中的一句提示词，也不是 `Agent` 的轮数属性。

| 项目 | 当前官方行为 |
| --- | --- |
| 未显式传值 | 默认 `10` |
| `max_turns=5` | 本次 run 最多允许五轮模型调用 |
| 一轮的含义 | 一次模型调用及该轮产生的工具处理，不是一次用户问答，也不是一个工具调用 |
| 一轮产生三个工具调用 | 仍是一轮，不会因为三个工具各执行一次而变成三轮 |
| 第五轮直接产生最终回答 | 可以正常返回 |
| 第五轮执行工具后仍需第六轮模型调用 | 默认抛出 `MaxTurnsExceeded`，不会额外赠送一次“总结调用” |
| `max_turns=None` | 关闭轮数上限；当前官方 API 明确支持，但本项目不建议使用 |

异常可从 `agents.exceptions` 导入：

```python
from agents.exceptions import MaxTurnsExceeded
```

当前实现通过 `status="limit_reached"` 和固定说明文字正常返回字典；SDK 默认通过异常表示超限。若保留当前 CLI 契约，需要应用边界捕获这个明确的异常，再转换成我们的状态和提示。不能假定超限后还会收到一个正常 `RunResult`。

`max_turns` 不控制 HTTP 重试次数、单工具超时、总运行时长、token 或费用，也不保证“最多五次搜索”。网络重试属于一次逻辑模型轮次内部的其他机制，不能把实际 HTTP 请求数与轮次简单等同。对当前单 Agent 项目，继续显式传 `5` 才能保留原有轮数预算。[D2][S2][S6]

### 7. State 在 SDK 中如何处理？

SDK 没有要求把当前整个 `state` 字典原封不动地塞进某个统一对象。需要拆开四类状态，再单独考虑业务记录。

#### 7.1 单次运行中的模型历史与执行状态

Runner 维护输入、生成的 items、当前 Agent、轮次等运行数据，工具结果也在这里进入下一轮。

我们不再手写单次 run 内的消息累积和循环状态机，但仍然要处理应用层的成功、异常或中断。

#### 7.2 本地应用 Context

通过 `Runner.run(..., context=app_context)` 传入自己定义的 Python 对象，例如 dataclass 或 Pydantic 对象。

工具或回调通过 `RunContextWrapper[T].context` 访问它。它适合承载用户身份、搜索服务实例、数据库依赖、业务计数器和策略等。

**Context 不会自动发送给模型，也不会因为传给 Runner 就自动持久化。** 若某项业务数据需要让模型知道，要通过 instructions、input 或工具返回值显式提供。

工具执行期间修改共享 context，也不等于模型已经知道了这个变化；更不等于业务事务自动提交了。并发工具对共享可变对象的访问仍需我们设计。

#### 7.3 多次运行之间的会话历史

可选择以下方式，而不是每次复用 `Agent` 就自动拥有记忆：

| 方式 | 责任边界 |
| --- | --- |
| `result.to_input_list()` | 获取可作为后续输入的历史；应用自己保存并附加下一条用户消息 |
| `session=SQLiteSession(...)` 等 | SDK 按 session 自动读取与保存会话历史；应用选择标识、后端和保留策略 |
| `conversation_id` / `previous_response_id` | 适用时使用 OpenAI Responses 服务端会话或响应链；不是通用 Chat Completions 能力 |

例如 `SQLiteSession("research-chat", "conversations.db")` 可以使用文件保存会话；只传 session ID 的默认内存使用方式不能被误解为进程重启后仍有持久记忆。

当前官方文档明确：同一次 run 的 `session` 不能与运行级 `conversation_id`、`previous_response_id` 或 `auto_previous_response_id` 组合使用。应选定一种延续方式，避免重复回放历史。

当前程序只处理一个命令行问题，并不需要为了“用 SDK”就立刻引入 Session。

#### 7.4 暂停与恢复的 RunState

`RunState` 是可恢复的运行快照，与“任意业务 state 字典”不是一个概念。

例如，工具要求人工批准时，`RunResult.interruptions` 暴露待处理项，`result.to_state()` 得到 `RunState`；应用批准或拒绝后，再把该状态传入 `Runner.run(...)` 恢复。

它服务于恢复执行，不等于永久知识记忆，也不会自动实现业务事务、可靠任务队列或任意 Python 依赖的持久化。若序列化快照，需关注 context 中的数据、凭据和不可序列化的客户端依赖。

#### 7.5 当前 state 字段的具体去向

| 当前字段 | SDK 对应表面 / 仍需保留的应用责任 |
| --- | --- |
| `run_id` | SDK trace 有自己的 `trace_id`；业务 run ID 若需要，仍由应用保留并关联，两者不自动相同 |
| `status` | 不存在与当前四种字符串状态完全等价的通用返回字段；应用根据返回、异常和 `interruptions` 映射 |
| `question` | Runner 的 `input`；如需审计原始问题，应用自己保留 |
| `messages` | Runner 的运行 items；运行后可用 `to_input_list()` 获取输入项视图，不保证逐项等于旧 Chat messages |
| `turns` | Runner 内部计数并执行上限；成功的当前单 Agent 普通 run 可通过 `raw_responses` 观察模型响应，但不要将它泛化成所有恢复/嵌套场景的业务计数器 |
| `tool_calls` | `new_items` 中的 `ToolCallItem` / `ToolCallOutputItem`，也可通过 hooks 做业务记录 |
| `errors` | 异常、工具错误输出、guardrail 结果和 tracing 等不同渠道；不会自动汇总成现有 `errors: list[str]` |
| `final_answer` | 通常对应 `RunResult.final_output`；错误提示或超限提示仍属于应用层输出 |

**总结：执行状态交给 Runner；模型历史用 items / Session；本地依赖用 Context；暂停恢复用 RunState；业务状态仍由应用定义。**[D4][D5][D6][D7]

### 8. Agent.instructions 对应我们第⑤步中的什么？

直接对应 `src\prompts.py` 的 `SYSTEM_PROMPT`，以及目前将它放入：

```python
{"role": "system", "content": SYSTEM_PROMPT}
```

的职责。迁移后的定义是：

```python
agent = Agent(
    name="Research assistant",
    instructions=SYSTEM_PROMPT,
    tools=[search_tool],
)
```

原有规则仍然要我们编写：什么时候需要实时信息、何时调用搜索、是否已有充分证据、如何引用 URL、何时停止继续搜索。

SDK 会将 instructions 作为模型指令传递；在不同模型适配器下，不必都表现为手工构造的 `role="system"` 消息。

`instructions` 也可使用同步或异步函数，接收 context wrapper 和 Agent，返回动态指令。这适合根据用户或任务提供指令，并不意味着 SDK 自动生成正确的业务策略。

当前提示词中“结果不足就继续搜索”的**策略内容**需要保留；手写 system message 的**协议拼装**可以移交 SDK。Instructions 是模型行为引导，不是权限校验或严格终止机制，轮次上限仍由 Runner 执行。[D1][D5][S1]

### 9. @function_tool 对应我们第⑤步中的什么？

它主要对应 **`WEB_SEARCH_TOOL` 的声明 + `_execute_web_search()` 中通用的参数调用适配**，而不是替代搜索业务函数。

从一个带类型标注和 docstring 的 Python 函数，SDK 会生成 `FunctionTool`：

| 当前手写内容 | SDK 提供的能力 |
| --- | --- |
| `"name": "web_search"` | 默认从 Python 函数名获取，可用 `name_override` 指定 |
| 工具 `description` | 从 docstring 获取，可用 `description_override` 指定 |
| 参数 `properties`、`required` 等 | 根据函数签名和类型生成 JSON Schema |
| 参数描述 | 可从 docstring 的参数说明提取 |
| `json.loads(arguments)` 与常规类型校验 | 调用包装层解析输入并使用 Pydantic 校验 |
| 将参数传给函数 | 包装层将已解析参数转成实际函数调用 |
| 工具失败转为模型可见错误 | 默认错误处理或自定义 `failure_error_function` |

还需要将工具注册到 `Agent(tools=[...])`；单独写装饰器不会启动执行，也不会自动将工具加入任何 Agent。

当前官方文档中的部分示例使用：

```python
from agents.decorators import tool
```

固定主分支源码明确写着 `tool = function_tool`。这不是另一个工具体系；`from agents import function_tool` 仍然是当前公开 API，本文按用户关心的名称解释。[S3][S7]

有三个迁移边界：

1. `query: str` 不会自动表达“去空白后不能为空”和“最多 300 字符”的全部业务规则。应保留 `web_search()` 内的校验，或明确声明相应约束，不能只依赖自动 schema。
2. `strict_mode=True` 是默认 schema 配置，但不能据此假设任意第三方模型返回的额外字段、类型转换等行为与当前 `set(arguments) == {"query"}` 判断逐字等价。
3. 普通 function tool 未指定 `failure_error_function` 时，默认把失败信息返回模型；显式设为 `None` 则让异常传播。默认不会保持我们的 `INVALID_ARGUMENTS`、`TOOL_EXECUTION_FAILED` 等业务错误码；需要该契约时仍应自行适配。

此外，本地 `web_search()` 和 SDK 的托管 `WebSearchTool()` 不是同一实现。换成后者会改变搜索服务、能力、返回信息及相关成本/隐私边界，不能把这种替换算作纯粹删除样板代码。[D3][S3]

### 10. Runner.run() 对应我们第⑤步中的什么？

对应 **`run_agent(question, llm, web_search, max_turns=5)` 的通用运行职责**：

```text
旧：
main()
  -> run_agent()
       -> _call_llm()
       -> _execute_web_search()
       -> messages.append(...)
       -> 再次调用模型
  -> state["final_answer"]

SDK：
main()
  -> await Runner.run(agent, question, max_turns=5)
       -> SDK 内部管理模型 / 工具 / 回填 / 下一轮
  -> result.final_output
```

如果保留当前同步的 `main()`，更直接的入口是 `Runner.run_sync(...)`；在已经运行事件循环的异步环境中，应使用 `await Runner.run(...)`，不要再套同步入口。

这不是签名层面的直接替换：SDK 接收已配置的 `Agent`，而不是我们当前的 `LLMClient` 协议和裸工具函数；返回的也是 `RunResult`，不是当前的 state 字典。

#### RunResult 应该怎样理解

它是“本次运行的结果和过程视图”，不是一条单纯的 assistant message，也不是模型返回的原始 JSON。

| 表面 | 用途 |
| --- | --- |
| `final_output` | 最终答案；未设置 `output_type` 时通常为字符串，设置后可为结构化对象；审批暂停时可能为 `None` |
| `new_items` | 本次运行新增的富元数据项，包括消息、工具调用和工具输出等 |
| `raw_responses` | 本次运行中的原始 `ModelResponse` 对象，不等于当前 `complete()` 返回的小字典 |
| `last_agent` | 最后运行的 Agent；多 Agent 场景中下一次交互常需考虑它 |
| `to_input_list()` | 将输入和运行 items 转为后续可使用的输入项列表 |
| `interruptions` / `to_state()` | 识别待批准操作并生成可恢复状态 |

不能只凭“Runner 返回了对象”就断言业务任务已完成；更不能把 `final_output=None` 无条件变成成功的空答案。对当前不配置审批和结构化输出的简单路径，正常返回后读取 `final_output` 即是主要用法。

同样，SDK 返回最终文本并不证明它已经满足“引用足够、内容真实、非空”等所有产品规则；这些仍是我们需要定义的输出要求。[D2][D4][S8]

### 11. 哪些代码可以被 SDK 删除？

以下是**将来迁移完成、确认行为契约后**可以删除的候选，不是本次已经删除。

| 当前代码 | 可由谁接管 | 删除前提 |
| --- | --- | --- |
| `WEB_SEARCH_TOOL` 手写 schema | `function_tool` | 用签名、docstring 或覆盖参数保留必要描述和输入约束 |
| `run_agent()` 的 `for` 循环 | Runner | 显式保留 `max_turns=5` |
| 通用“答案还是工具调用”的调度分支 | Runner | 接受 SDK 的完成语义，另外保留应用输出要求 |
| assistant tool call / tool result 消息拼装与回填 | Runner + 模型适配器 | 不再依赖旧消息列表的原始结构 |
| `_execute_web_search()` 的工具查找、JSON 解析、通用调用适配 | SDK 工具运行层 | 处理未知工具和参数校验行为的差异 |
| 为运行过程重复保存的工具调用列表 | `RunResult.new_items` / hooks | 如果业务外部仍需要固定审计格式，应保留转换层 |
| `OpenAICompatibleClient` 中通用 HTTP 请求和响应解析 | `AsyncOpenAI` + SDK 模型适配器 | 配置正确的 API 类型、地址、模型和超时 |
| `LLMClient` / `Tool` 等仅服务于旧执行器的协议或类型 | SDK 的模型与工具接口 | 没有其他调用者仍依赖旧接口 |
| `_call_llm()` 的自建重试循环 | 所选客户端 / SDK 的重试机制 | 先确定重试范围、次数和退避，不把默认值当作现有策略 |
| 仅用于观察循环过程的部分 `print` | tracing / hooks | 若 CLI 需要即时进度显示，应保留或重写输出 |

其中 `_tool_error()`、`LLMError` 及 `status/errors` 包装不能无条件删除：若它们是我们要保留的应用契约，就应缩小到应用边界，而不是让框架默认错误文本取代业务规范。

文件级别可以理解为：`src\agent.py` 中大部分“通用执行器”有机会消失，`src\llm_client.py` 中标准协议适配有机会被替代；但不能因为 SDK 有同类能力，就直接整文件删除且不处理调用方。[D2][D3][D8]

### 12. 哪些代码仍然需要我们自己编写？

| 必须继续由应用负责 | 当前对应内容 |
| --- | --- |
| 业务目标和策略 | `SYSTEM_PROMPT` 中的搜索、证据判断、来源引用要求 |
| 真正的工具逻辑 | `web_search()` 的 DuckDuckGo 请求和结果整理 |
| 业务参数约束 | 非空 query、300 字符上限、结果数量限制 |
| 工具的权限和数据边界 | 凭据、授权、网络访问范围、敏感数据处理 |
| 模型和服务配置 | API key、model、base URL、API 类型、超时和重试策略 |
| 应用入口 | 参数解析、`.env` 加载、空问题提示、输出格式、退出行为 |
| 应用状态和错误契约 | `completed` / `failed` / `limit_reached` 的映射、错误码、用户提示 |
| 记忆与持久化策略 | 是否需要 Session、用户/会话隔离、历史保留、业务数据存储 |
| 可观测性和隐私策略 | 是否启用 tracing、发往哪里、保留哪些字段、是否需要应用审计日志 |
| 业务正确性与成本控制 | 搜索质量、答案证据要求、工具异常、预算、时间和费用约束 |

SDK 还提供 guardrails、hooks、审批等扩展点，但**扩展点不等于已经实现了我们的业务规则**。对这个学习项目，先理解一个 Agent、一个本地工具和一个 Runner 即可，不需要为了使用框架就引入多 Agent 或复杂工作流。

## 三、Tracing：对应当前日志，但不是业务状态数据库

Tracing 是 SDK 内建的运行可观测性能力。Trace 表示一次工作流，span 表示其中一次模型生成、工具执行或其他操作，包含关联关系和时间信息。

当前官方文档列出的自动记录覆盖 Runner 运行、Agent、模型生成、function tool、guardrails 和 handoffs 等；当前主分支文档还描述了 task / turn spans。具体 span 层级会随 SDK 版本与配置变化，不应作为业务接口来依赖。

对应当前代码：

| 当前观察方式 | SDK 对应能力 |
| --- | --- |
| `run_id` 和 “started/stopping” 输出 | trace 关联及运行 spans |
| “turn N: asking LLM” | 模型调用相关 spans；需要精确终端进度时可使用 hooks |
| “executing web_search” 和结果状态输出 | function tool spans |
| API / 工具错误日志 | 相应运行诊断和错误信息；应用仍需处理并呈现失败 |

默认 tracing **开启**，默认处理器会将 traces/spans 批量导出到 OpenAI 后端。官方文档还指出，`trace_include_sensitive_data` 默认是 `True`，模型和工具输入输出可能被采集。

因此，迁移不能被当成只改变代码行数：它也可能改变数据流向。特别是使用第三方兼容模型 endpoint 时，**模型请求目的地与 tracing 导出目的地是两件事**。

常用控制：

```python
from agents import RunConfig

no_tracing = RunConfig(tracing_disabled=True)
less_content = RunConfig(trace_include_sensitive_data=False)
```

也可以使用环境变量 `OPENAI_AGENTS_DISABLE_TRACING=1` 全局关闭。`trace_include_sensitive_data=False` 只是减少特定内容采集，不等于完全禁用导出，也不自动清理我们自己写入 metadata 或日志中的敏感信息。

若设置自定义处理器，`add_trace_processor()` 是添加额外处理器，`set_trace_processors()` 才是替换默认处理器。不能为了避免默认导出而只“增加一个本地处理器”。

**Tracing 不负责继续 Agent Loop、不让模型自动获得记忆、不替代 `Session`、不替代业务状态表，也不保证复刻当前控制台日志。** 是否开启以及保留什么数据，由应用明确决定。[D7]

## 四、本项目迁移时最重要的兼容性边界

### 4.1 当前是 Chat Completions，SDK 默认是 Responses

`OpenAICompatibleClient` 请求的是 `/chat/completions`，而 SDK 对 OpenAI 模型默认走 Responses API。仅支持 Chat Completions 的兼容服务，不会因为换成 Agents SDK 就自动支持 Responses。

如果目标是尽量保留当前 endpoint，应考虑官方提供的：

```python
OpenAIChatCompletionsModel(
    model=model_name,
    openai_client=AsyncOpenAI(api_key=api_key, base_url=base_url),
)
```

不要把当前 `OpenAICompatibleClient` 直接作为 `Agent.model` 传入；它并未实现 SDK 的 `Model` 接口。

也不要假定项目约定的 `OPENAI_MODEL` 会被 SDK 自动当作模型名使用；应继续读取它并明确传入模型配置。第三方服务对工具 schema、模型设置和错误结构的兼容性仍需确认。[D8][S9]

### 4.2 下列行为不会自动保持不变

| 当前行为 | 迁移时要明确的选择 |
| --- | --- |
| 超限返回 `limit_reached` | SDK 默认抛异常，应用是否映射回原有状态 |
| 未知工具回传 `UNKNOWN_TOOL` 后继续 | SDK 默认报错，是否启用模型可见错误，以及是否保留错误码 |
| 工具错误固定为 `status/error` 字典 | 定义返回信封或错误 formatter；不要依赖 SDK 默认文本 |
| 对 query 和额外参数严格检查 | 明确 Pydantic/schema 与现有手写校验的差异 |
| 一个响应中的工具依次执行 | SDK 可并发执行多个本地 function calls；如果顺序具有业务意义，需要明确控制 |
| 模型最多两次尝试，固定等待一秒 | SDK/客户端默认策略不保证相同，不能叠加多层重试后仍声称预算没变 |
| LLM 请求超时 30 秒，搜索请求超时 10 秒 | 分别保留或重新配置；`max_turns` 不能替代超时 |
| 回答必须为非空字符串 | 保留应用输出要求，不把“存在 final_output”当作全部业务校验 |
| 只有本地控制台过程输出 | SDK 默认 tracing 有远程导出，须明确隐私配置 |

## 五、仅用于理解映射的代码示意

以下只存在于本文中，**不是已经实施的迁移，也不承诺完整保留当前 status、错误码、重试和日志契约**。示例选择 Chat Completions 适配器以贴近现有客户端，并关闭 tracing，避免默认新增遥测导出。

```python
import json

from openai import AsyncOpenAI
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
)

from src.prompts import SYSTEM_PROMPT
from src.tools import web_search as search_impl


@function_tool(name_override="web_search")
def search_web(query: str) -> str:
    """Search the web for current, factual information.

    Args:
        query: A focused web search query.
    """
    return json.dumps(search_impl(query), ensure_ascii=False)


async def research(
    question: str, api_key: str, model_name: str, base_url: str
) -> str:
    async with AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
        agent = Agent(
            name="Research assistant",
            instructions=SYSTEM_PROMPT,
            model=OpenAIChatCompletionsModel(
                model=model_name,
                openai_client=client,
            ),
            tools=[search_web],
        )
        result = await Runner.run(
            agent,
            question,
            max_turns=5,
            run_config=RunConfig(tracing_disabled=True),
        )
        if not isinstance(result.final_output, str):
            raise TypeError("Expected a text final output.")
        return result.final_output
```

这里仍复用真正的 `search_impl` 和 `SYSTEM_PROMPT`。消失的是手写 loop、tool dispatch 和消息回填，而不是业务能力。

示例的返回类型检查只说明应用如何读取结果；现有入口校验、非空回答要求、超限状态映射、重试策略和错误信封还需要在真正迁移时接回应用边界。同步 CLI 则可采用 `Runner.run_sync()` 的对应结构。

## 六、最终心智模型

```text
我们定义：
  Agent.instructions     -> 应该怎样研究和回答
  function tool 的函数体 -> 真实世界的搜索能力
  应用配置与策略         -> 模型、预算、权限、状态、持久化、隐私

模型生成：
  工具调用请求 / 最终回答

SDK Runner 执行：
  调用模型 -> 分发工具 -> 收集结果 -> 回填输入 -> 继续或停止

SDK 返回 / 观测：
  RunResult -> 本次运行的结果与过程
  RunState  -> 需要时恢复暂停的执行
  Tracing   -> 查看发生了什么
```

**不是“用了 SDK 就不需要 Agent Loop”，而是“Agent Loop 从我们自己的代码，转移到了 SDK 的运行时”。**

## 官方参考资料

### 版本信息

- [V1] [GitHub 最新 Release API](https://api.github.com/repos/openai/openai-agents-python/releases/latest)；[v0.22.0 Release](https://github.com/openai/openai-agents-python/releases/tag/v0.22.0)。
- [V2] [本次读取的主分支提交](https://github.com/openai/openai-agents-python/commit/cb808dfaee4f5f6bc256abc4f6f021a4693c397c)。
- [V3] [该提交的 pyproject.toml](https://github.com/openai/openai-agents-python/blob/cb808dfaee4f5f6bc256abc4f6f021a4693c397c/pyproject.toml)。

### 官方文档

- [D1] [Agents：配置、Instructions、Context、Tool use behavior](https://openai.github.io/openai-agents-python/agents/)。
- [D2] [Running agents：Runner、Agent Loop、max_turns、运行配置](https://openai.github.io/openai-agents-python/running_agents/)。
- [D3] [Tools：Function tools、错误处理、本地与托管工具](https://openai.github.io/openai-agents-python/tools/)。
- [D4] [Results：RunResult、new_items、final_output、状态恢复](https://openai.github.io/openai-agents-python/results/)。
- [D5] [Context management：本地 Context 与模型可见上下文](https://openai.github.io/openai-agents-python/context/)。
- [D6] [Sessions：跨 run 会话历史与存储](https://openai.github.io/openai-agents-python/sessions/)。
- [D7] [Tracing：默认采集、处理器、敏感数据和关闭方式](https://openai.github.io/openai-agents-python/tracing/)。
- [D8] [Models：Responses、Chat Completions 与模型配置](https://openai.github.io/openai-agents-python/models/)。

### 官方源码核对位置

以下主分支链接固定到本次读取的提交，内部文件仅用来解释实现，不建议应用直接导入 SDK 私有执行层。

- [S1] [Agent 定义](https://github.com/openai/openai-agents-python/blob/cb808dfaee4f5f6bc256abc4f6f021a4693c397c/src/agents/agent.py)。
- [S2] [Runner 入口与执行循环](https://github.com/openai/openai-agents-python/blob/cb808dfaee4f5f6bc256abc4f6f021a4693c397c/src/agents/run.py)；[v0.22.0 Runner](https://github.com/openai/openai-agents-python/blob/v0.22.0/src/agents/run.py)。
- [S3] [FunctionTool / function_tool 定义与参数调用适配](https://github.com/openai/openai-agents-python/blob/cb808dfaee4f5f6bc256abc4f6f021a4693c397c/src/agents/tool.py)；[v0.22.0 tool.py](https://github.com/openai/openai-agents-python/blob/v0.22.0/src/agents/tool.py)。
- [S4] [工具执行与 ToolCallOutputItem 创建](https://github.com/openai/openai-agents-python/blob/cb808dfaee4f5f6bc256abc4f6f021a4693c397c/src/agents/run_internal/tool_execution.py)；[下一轮输入准备](https://github.com/openai/openai-agents-python/blob/cb808dfaee4f5f6bc256abc4f6f021a4693c397c/src/agents/run_internal/run_loop.py)。
- [S5] [ItemHelpers.tool_call_output_item 与调用 ID 关联](https://github.com/openai/openai-agents-python/blob/cb808dfaee4f5f6bc256abc4f6f021a4693c397c/src/agents/items.py)。
- [S6] [DEFAULT_MAX_TURNS 与运行配置](https://github.com/openai/openai-agents-python/blob/cb808dfaee4f5f6bc256abc4f6f021a4693c397c/src/agents/run_config.py)；[v0.22.0 run_config.py](https://github.com/openai/openai-agents-python/blob/v0.22.0/src/agents/run_config.py)。
- [S7] [tool = function_tool 别名](https://github.com/openai/openai-agents-python/blob/cb808dfaee4f5f6bc256abc4f6f021a4693c397c/src/agents/decorators.py)；[公开导出](https://github.com/openai/openai-agents-python/blob/cb808dfaee4f5f6bc256abc4f6f021a4693c397c/src/agents/__init__.py)。
- [S8] [RunResult 定义](https://github.com/openai/openai-agents-python/blob/cb808dfaee4f5f6bc256abc4f6f021a4693c397c/src/agents/result.py)。
- [S9] [官方自定义 Chat Completions 模型示例](https://github.com/openai/openai-agents-python/blob/cb808dfaee4f5f6bc256abc4f6f021a4693c397c/examples/model_providers/custom_example_agent.py)。
