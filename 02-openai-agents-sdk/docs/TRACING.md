# 为本项目增加最基本的 Tracing

## 0. 做了什么改动

只用 OpenAI Agents SDK 自带的 tracing 能力（`agents.tracing`），没有引入任何第三方 observability
框架、没有 LangSmith、没有 OpenTelemetry、没有数据库，也没有改动任何 Agent 业务逻辑
（`src/agent.py` 里 Agent 的 instructions、tools、Runner 调用方式都不变）：

* `src/tracing.py`（新增）：实现 SDK 自带的 `TracingExporter` 扩展点
  `LocalFileExporter`，把 trace/span 写到本地 `traces/trace.jsonl` 文件（JSON Lines，一行
  一个 trace 或 span），再用 `set_trace_processors([BatchTraceProcessor(...)])`
  把它注册为唯一的 processor。
* `src/main.py`：在读取 `.env`、构建 Agent 之前调用一次 `enable_local_tracing()`。
* `src/agent.py`：`RunConfig(tracing_disabled=True)` 改为
  `RunConfig(workflow_name=AGENT_NAME)` —— tracing 本身保持 SDK 默认的**开启**状态，
  只是把导出目的地从 OpenAI 的公共 tracing 后端（`BackendSpanExporter`，会把请求/响应
  内容发到 `https://api.openai.com/v1/traces/ingest`）换成本地文件。这一点很重要：本项目
  实际连的是私有的 OpenAI 兼容网关（`OPENAI_BASE_URL`），如果保留 SDK 默认导出行为，
  即使模型本身不是 OpenAI 提供的，trace 内容也会被发到 OpenAI 的公网服务——这是不必要
  的数据外泄风险，所以用 SDK 自己的 processor/exporter 接口把目的地改本地即可，不需要
  额外的框架。

运行方式不变：

```powershell
python -m src.main "What is the current population of Japan? Please search for up-to-date sources."
```

每次运行会在 `traces/trace.jsonl` 追加一条完整 trace 的所有 span（该目录已加入
`.gitignore`，不会被提交，因为其中可能包含真实的用户问题、工具参数与模型输出）。

## 1. 实际跑一次需要 Tool Calling 的任务，Trace 里能看到什么

用上面的问题实跑一次（真实模型 + 真实 `web_search` 工具调用，非 mock），产生的
`trace.jsonl` 里一共 16 行（1 个 trace + 15 个 span）。span 之间通过 `parent_id`
形成一棵树，从外到内依次是：

```text
trace: workflow_name="Research assistant"
└─ span(custom/"task")                 # 整个 Runner.run_sync 调用
   └─ span(agent)  name="Research assistant", tools=["web_search"]   # Agent run
      ├─ span(custom/"turn" turn=1)                                  # 第 1 轮
      │  ├─ span(generation)  -> 模型决定调用 web_search              # LLM generation
      │  └─ span(function) name="web_search" input=... output=...    # Tool call + 结果
      ├─ span(custom/"turn" turn=2)                                  # Agent continuation
      │  ├─ span(generation)  -> 模型这次并行发起 3 个 web_search
      │  ├─ span(function) name="web_search" (call_1)
      │  ├─ span(function) name="web_search" (call_2)
      │  └─ span(function) name="web_search" (call_3)
      ├─ span(custom/"turn" turn=3)
      │  ├─ span(generation)  -> 又发起 1 次更具体的 web_search
      │  └─ span(function) name="web_search"
      └─ span(custom/"turn" turn=4)
         └─ span(generation)  -> 不再调用工具，直接给出 Final output
```

对应到问题里列出的六类信息，逐一都能在 trace 里找到：

| 关注点 | 在 trace 里的位置 | 摘录 |
| --- | --- | --- |
| **Agent run** | `span_data.type == "agent"`，记录 Agent 名字、可用工具列表、输出类型 | `{"type": "agent", "name": "Research assistant", "tools": ["web_search"], "output_type": "str"}` |
| **LLM generation** | 每轮里 `span_data.type == "generation"`，包含发给模型的完整 `input`（含 system prompt、历史消息、工具结果）和模型的 `output`（含 `tool_calls` 或最终文本）、`model`、`usage`（input/output/reasoning tokens） | 第 1 轮：`"output": [{"tool_calls": [{"function": {"name": "web_search", "arguments": "{\"query\":\"Japan current population ...\"}"}}]}]` |
| **Tool call** | `span_data.type == "function"`，`name` 是工具名，`input` 是 SDK 解析出的调用参数（JSON 字符串） | `{"type": "function", "name": "web_search", "input": "{\"query\":\"...\"}"}` |
| **Tool result** | 同一个 `function` span 的 `output` 字段，就是 `src/tools.py::web_search` 的返回值（本例中 DuckDuckGo 接口返回了空结果列表） | `"output": "{'query': '...', 'results': []}"` |
| **Agent continuation** | `span_data.type == "custom", name == "turn"`，每一轮一个，`turn` 字段递增（1→4）；下一轮 `generation` 的 `input` 里能看到上一轮的 `tool_calls` 和 `role: "tool"` 消息被自动拼接了回去 | turn=2 的 generation `input` 末尾包含 turn=1 的 `assistant` 消息（`tool_calls`）和 `role: "tool"` 消息 |
| **Final output** | 最后一轮（本例 turn=4）的 `generation.output[0].content`，且这一轮**没有** `tool_calls` | `"content": "Japan's population is approximately 123 million people. ..."` |

顶层还有一个 `task` span 汇总了整次运行的总量：`usage.requests=4`（即 4 次 LLM 调用/4
轮），`total_tokens=2429`。

## 2. 为什么没有框架的 minimal agent 很难观察 Agent Loop

一个手写的最小 Agent（比如本仓库姊妹目录 `01-minimal-agent` 那种“system/user message
→ 调用 Chat Completions → 解析 `tool_calls` → 手动 dispatch → 把结果拼回
messages → 再调一次”的 while 循环）本质上只是若干次同步函数调用叠在一起：

* **没有统一的“span”概念**：一次 LLM 调用、一次工具调用、一轮循环，在代码里就是几行
  `requests.post(...)` / `json.loads(...)` / `messages.append(...)`，彼此之间没有 ID、
  没有父子关系、没有开始/结束时间戳。事后除了 `print`/日志文本，没有结构化的数据可查。
* **状态是隐式的、易失的**：`messages` 列表在内存里被原地修改，循环结束后就地销毁；想知道
  “第 2 轮模型看到了哪些工具结果”，只能在代码里临时加 `print(messages)`，而且改一次代码
  才能补看一次。
* **循环边界不透明**：“这是第几轮”“为什么又调用了一次模型”“工具结果有没有真正喂回去”
  这些问题，答案散落在 `while` 循环体的各处逻辑里，必须读代码、加断点或加日志才能回答，
  没有现成的、按运行（run）维度分组的视图。
* **并发/并行工具调用更难看清**：一旦模型一次性发起多个 `tool_calls`（如上面例子的 turn=2
  并行 3 次搜索），手写代码里通常是一个 `for` 循环顺序处理，谁先谁后、各自耗时多少，
  同样只能靠零散日志推断。

简言之：手写 Agent 把“循环内部发生了什么”这件事完全交给了源码可读性和临时日志，
可观测性完全依赖开发者当时写了多少 `print`。

## 3. 本项目 SDK 如何帮助观察 Agent Loop

OpenAI Agents SDK 的 `Runner` 在**执行**循环的同时，天然会在关键节点开、关闭 span
（`agents/tracing/create.py` 里的 `agent_span`、`generation_span`、`function_span`、
`turn_span` 等），这些 span 由 SDK 自己在 `Runner.run`/`run_sync` 内部调用，不需要业务代码
插桩：

* **结构化、可编程消费**：每个 span 都是带 `id`/`parent_id`/`trace_id`/`started_at`/
  `ended_at`/`span_data` 的对象，`span_data.type` 明确区分 `agent`/`generation`/
  `function`/`custom`(turn/task) 等类型，可以直接 `json.loads` 后按类型过滤、按
  `trace_id` 聚合，而不必解析日志文本。
* **天然的父子层级 = 天然的调用栈**：`parent_id` 把“整个 run → Agent run → 第 N 轮 →
  这一轮的 generation/tool call”串成一棵树，不需要手写任何“当前在第几轮”的记录逻辑，
  遍历树就能重建循环的执行路径。
* **循环推进的证据是数据，不是猜测**：`turn` span 上直接标了 `turn: 1, 2, 3, 4`，
  下一轮 `generation` span 的 `input` 里能看到上一轮的 `tool_calls`
  和 `role: "tool"` 消息被自动拼接了回去——“工具结果有没有真正喂回模型”这个问题
  不用猜，直接读 span 数据就有答案。
* **扩展点是标准化的**：只要实现 SDK 自己定义的 `TracingExporter`/`TracingProcessor`
  接口（本项目的 `LocalFileExporter` 只有十几行），就能把这些 span 导出到任何地方
  （本地文件、控制台、自建后端），不需要引入 OpenTelemetry 或第三方 SDK；tracing 的
  开关（`tracing_disabled`）和目的地（`set_trace_processors`）都是 SDK 一等公民配置项。

## 4. Tracing 对 Debugging 有什么价值

* **把“循环内部发生了什么”从需要重新运行 + 加日志，变成直接读一份已经落盘的数据**：
  出问题后不需要复现现场、不需要临时改代码加 `print`，`traces/trace.jsonl` 里已经有
  完整的输入/输出/耗时/token 用量。
* **能回答“模型到底看到了什么”**：每个 `generation` span 的 `input` 是发给模型的
  完整消息列表（system prompt + 历史 + 工具结果），可以验证 prompt 拼接、历史截断、
  工具结果注入是否符合预期，而不是猜测“它应该看到了 XX”。
* **能定量分析成本和延迟**：`usage`（input/output/reasoning tokens）和
  `started_at`/`ended_at` 让你能算出每一轮、每次工具调用各花了多久、多少 token，
  定位是哪一轮/哪次调用拖慢或拖贵了整个 run。
* **能验证循环是否正确终止**：最后一个 `generation` span 没有 `tool_calls` 就是
  Runner 判定“可以停止循环”的依据，`max_turns` 是否触顶、循环有没有提前/过晚结束，
  都可以从 turn 计数和 span 结构直接确认。

## 5. 如果 Agent 错误调用 Tool，我们如何定位问题

按 span 层级从粗到细排查：

1. **先看 `function` span 的 `input`**：这是 SDK 从模型返回的 `tool_calls[i].function.arguments`
   解析出来的实际调用参数。如果参数本身就不对（字段缺失、拼写错误、单位不对等），
   说明是**模型生成的调用有问题**，往上一级看同一轮的 `generation` span 的
   `output`（模型的原始 `tool_calls`）以及该轮的 `input`（模型当时看到的 system
   prompt/历史/上一轮工具结果），确认是不是 prompt 没给够信息、还是上一轮工具结果
   本身就有误导性。
2. **再看 `function` span 的 `output`**：这是工具函数（`src/tools.py::web_search`）
   真实返回的数据。本次实跑中就出现了一个典型例子：3 轮里的每一次 `web_search`，
   `output` 都是 `"results": []`（DuckDuckGo instant-answer 接口对这类查询没有
   结构化摘要，返回空列表）——这说明**问题不在 Agent 的调用决策**（模型确实在需要时
   调用了工具、并且在没拿到有效结果后按 instructions 尝试了更具体的查询词），
   **而在工具本身/上游数据源**：换一个搜索 query 表达方式或换一个搜索接口才能解决。
   如果反过来是 `output` 里有正常结果、但**下一轮** `generation` 的 `output` 却没有
   利用这些结果（既没结合内容回答，也没有再调用工具去补充），那问题就出在模型这一步，
   该检查 instructions 是否明确要求“基于结果作答”。
3. **对照相邻 `turn` span 判断“错误调用”是不是重复发生**：多个 `turn` 里都出现同名
   `function` span 且参数高度相似（如本例 turn=1/2/3 反复搜同一件事的不同措辞），
   说明模型在“兜圈子”——要么是工具一直没返回有用结果导致模型不断重试（工具侧问题，
   参考第 2 点），要么是 instructions 没有给出清晰的“重试上限/放弃条件”（prompt 侧
   问题）。
4. **用 `trace_id` 串联同一次用户请求的全部 span**：当有多个并发用户请求时，先按
   `trace_id` 过滤出这一次请求的所有 span，再按上面 1–3 步排查，避免把不同请求的
   工具调用混在一起分析。
5. **如果是 Agent 调用了不该调用的工具（本项目里只有一个 `web_search`，多 Agent/多工具
   场景更常见）**：看触发该 `function` span 的同一 `generation` span 的 `input`，
   核对 system prompt 里关于“何时使用该工具”的措辞（本项目里是
   `"Use web_search when the question needs current, factual, or source-dependent
   information."`），判断是措辞不够明确导致模型误判，还是问题本身确实处于该措辞的
   模糊地带；这类 instructions 调整也正是 tracing 数据能直接支撑的、有据可依的
   prompt 迭代方式，而不是凭感觉改 prompt。
