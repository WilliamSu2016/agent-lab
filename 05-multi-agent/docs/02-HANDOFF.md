# 10. Handoff

> 实验范围：`src/02_handoff.py`（OpenAI Agents SDK）。
> TriageAgent 判断请求属于 `research` 还是 `coding`，通过 SDK 的 `handoffs`
> 机制把整个对话的控制权转移给 ResearchAgent 或 CodingAgent，由接管的专家
> Agent 直接给出最终答案。不使用 Supervisor、不使用 Agents-as-Tools、不手写
> 分发逻辑。

## 0. 架构：
                User
                  │
                  ▼
             Triage Agent
              /        \
             /          \
            ▼            ▼
       Research        Coding
        Agent           Agent

## 0. 本次实现了什么

```text
User -> TriageAgent --handoff (transfer_to_research)--> ResearchAgent -> Final Answer
User -> TriageAgent --handoff (transfer_to_coding)-->   CodingAgent   -> Final Answer
```

| Agent | 角色 | 是否有 `handoffs` | 是否生成最终答案 |
|---|---|---|---|
| TriageAgent | 分诊入口，只判断 research/coding | 有：`transfer_to_research`、`transfer_to_coding` | 否 |
| ResearchAgent | 技术调研专家 | 无 | 是 |
| CodingAgent | 代码方案专家 | 无 | 是 |

关键实现点（对应需求逐条）：

- **`agents.handoff(...)`**（`src/02_handoff.py` 的 `build_triage_agent`）把
  `tool_name_override` 显式设为 `"transfer_to_research"` /
  `"transfer_to_coding"`，只在 TriageAgent 上注册这两个 handoff。
- 每个专家 Agent 都设置了 `handoff_description`（需求 9），描述"什么请求应该
  转给我"，供 TriageAgent 的模型在决策时读取。
- TriageAgent 的 instructions 显式声明"只调用转接工具，不回答问题本身"（需求
  5），配合 `RECOMMENDED_PROMPT_PREFIX`（SDK 官方推荐的多 Agent 系统提示前缀）。
- Tracing 通过 `src/tracing.py`（沿用 02-openai-agents-sdk 的做法）路由到本地
  `traces/trace.jsonl`，SDK 默认的 agent-run / handoff / generation span 全部
  保留（需求 10），只是导出目的地从 OpenAI 公共后端改成本地文件。

## 1. 如何观察 Trace

跑一次真实请求（需要 `.env` 里的 API 配置）：

```powershell
.\.venv\Scripts\python.exe -m src.02_handoff "比较 LangGraph 和 OpenAI Agents SDK。"
.\.venv\Scripts\python.exe -m src.02_handoff "帮我写一个 LangGraph Agent。"
```

程序会打印从 `RunResult.new_items` 中提取出的 handoff 事件（`extract_handoff_trace`），
两个测试问题分别得到：

```text
HANDOFF: TriageAgent --transfer_to_research--> ResearchAgent
ANSWERED BY: ResearchAgent
```

```text
HANDOFF: TriageAgent --transfer_to_coding--> CodingAgent
ANSWERED BY: CodingAgent
```

同时 `traces/trace.jsonl` 里会出现一条 `"type": "handoff"` 的 span，直接记录了
`from_agent`/`to_agent`：

```json
{"span_data": {"type": "handoff", "from_agent": "TriageAgent", "to_agent": "ResearchAgent"}}
```

在 `tests/test_handoff.py` 里，同样的观察点被固化成断言：`result.new_items`
中必须依次出现 `HandoffCallItem`（`transfer_to_research`/`transfer_to_coding`
这次工具调用）和 `HandoffOutputItem`（记录 `source_agent`/`target_agent`），且
`result.last_agent.name` 必须是接管的专家 Agent，而不是 TriageAgent。

## 2. 为什么 Handoff 是"控制权转移"？

关键在于：Handoff 发生之后，**接下来负责"决定下一步做什么、什么时候结束、
给出什么最终答案"的，是另一个 Agent，而不是发起 Handoff 的那个 Agent。**

具体到 SDK 的行为（在 `tests/test_handoff.py` 里被断言验证）：

1. **当前 Agent 整体被替换，不只是多了一次调用。** TriageAgent 的模型只调用了
   一次 `transfer_to_research`；调用之后，`Runner` 把"当前活跃 Agent"从
   TriageAgent 切换成 ResearchAgent。这一步之后的所有事情——包括最终由谁的
   `model.get_response()` 产生 `final_output`——都改由 ResearchAgent 决定。
   `result.last_agent.name == "ResearchAgent"` 证明了这一点：**运行结束时"负责
   这次对话"的 Agent 变了。**

2. **TriageAgent 不会拿到结果、也不会再被调用一次来"收尾"。** 如果是普通的
   Tool Calling，工具执行完之后，结果会被喂回*发起调用的那个 Agent*，由它决定
   下一步（可能是直接回答，也可能是再调用别的工具）。而 Handoff 之后，
   TriageAgent 不会再收到 ResearchAgent 的回答去做二次处理——对话直接在
   ResearchAgent 那里结束（除非 ResearchAgent 自己又发起新的 Handoff，但本实验
   中专家 Agent 没有配置任何 `handoffs`）。这正是需求 5、6（"TriageAgent 不生成
   最终答案"、"Specialist Agent 接管后负责最终回答"）在机制层面成立的原因：
   不是靠约定和自觉，而是 Handoff 语义本身决定了控制权已经转移。

3. **Instructions（决策上下文）和身份边界一起转移。** 转移之后的每一轮模型
   调用，用的是 ResearchAgent 自己的 `instructions`（研究类职责边界）而不是
   TriageAgent 的分诊指令。"控制权"不仅指"谁来产出下一段文本"，也指"接下来
   由谁的职责定义、谁的完成标准来判断这次任务是否做完"。

## 3. 为什么这和普通 Tool Calling 不一样？

把 `transfer_to_research` 单独拿出来看，它确实和普通函数工具一样：都是模型
输出一个 `function_call` 输出项，携带工具名和参数。但两者在**结果如何被处理**
上有本质区别：

| | 普通 Tool Calling（如 `web_search`） | Handoff（如 `transfer_to_research`） |
|---|---|---|
| 谁执行 | 业务函数（`@function_tool` 包装的普通代码） | 触发一次 Agent 切换（SDK 内部逻辑：把 target agent 设为新的 current agent） |
| 结果去哪 | 工具返回值被序列化成 `tool` 消息，**喂回发起调用的同一个 Agent**，那个 Agent 的下一轮继续决定要不要再调用工具、要不要回答 | 没有"返回值喂回原 Agent"这一步；对话继续时使用的是目标 Agent 的 instructions/模型，原 Agent 不再参与 |
| 循环归属 | 属于同一个 Agent 的同一次"Agent Loop"（可能循环很多轮） | 结束了原 Agent 的回合，把"这是谁的任务"这件事本身转移给了另一个 Agent |
| 对应的 `new_items` 类型 | `ToolCallItem` + `ToolCallOutputItem`（结果作为普通工具输出） | `HandoffCallItem`（工具调用本身） + `HandoffOutputItem`（记录 `source_agent`/`target_agent`，没有"函数返回值"这个概念） |
| 职责边界 | 工具只是能力扩展，Agent 的身份、instructions、完成标准始终不变 | Agent 本身发生了替换，instructions、职责边界、"什么算完成"全部换成了目标 Agent 的定义 |

在本实现的 `tests/test_handoff.py` 中，这个区别直接体现为：

- `web_search` 之类的普通工具会产生 `ToolCallItem` + `ToolCallOutputItem`，
  最终答案仍由**同一个** Agent（比如 02-openai-agents-sdk 实验里的
  `Research assistant`）给出；
- 而这里的 `transfer_to_research`/`transfer_to_coding` 只产生
  `HandoffCallItem` + `HandoffOutputItem`，最终答案由**另一个** Agent（
  ResearchAgent/CodingAgent）给出，`result.last_agent` 发生了变化。

一句话总结：**普通 Tool Calling 是"让当前负责人去查一下资料/办一件事，再自己
拿主意"；Handoff 是"这件事换人负责了，新负责人从这里开始按自己的标准继续办"。**
这也是为什么 Handoff 必须搭配 `handoff_description`（让转出方知道"这件事该交给
谁"）而不是像普通工具那样只需要一份参数 Schema——它转移的不是一次调用的结果，
而是整个任务的所有权。
