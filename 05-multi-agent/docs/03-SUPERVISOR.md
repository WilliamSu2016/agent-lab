# 03. Supervisor / Manager（Agents-as-Tools）

> 实验范围：`src/03_supervisor.py`（OpenAI Agents SDK，`Agent.as_tool()`）。
> SupervisorAgent 把 ResearchAgent、CodingAgent、ReviewAgent 分别包装成
> `research_expert`、`coding_expert`、`review_expert` 三个普通函数工具，按需
> 调用零个、一个或多个，再由 Supervisor 自己写出最终答案。不使用 Handoff、
> 不使用 LangGraph、不使用 MCP。
## 0. 架构：
                   User
                     │
                     ▼
                Supervisor
               /     |     \
              ▼      ▼      ▼
          Research Coding Review
              │       │       │
              └───────┼───────┘
                      ▼
                 Supervisor
                      │
                      ▼
                    User

## 0. 本次实现了什么

```text
User -> SupervisorAgent -> [tool call] research_expert -> ResearchAgent -> tool result
                         -> [tool call] review_expert   -> ReviewAgent   -> tool result
                         -> Final Answer（由 SupervisorAgent 自己撰写）
```

| Agent | 角色 | 是否有 `tools` | 是否直接面对用户 |
|---|---|---|---|
| SupervisorAgent | 唯一决策者，调用专家、整合并撰写最终答案 | 有：`research_expert`、`coding_expert`、`review_expert` | 是（唯一一个） |
| ResearchAgent | 技术调研专家 | 无 | 否 |
| CodingAgent | 代码方案专家 | 无 | 否 |
| ReviewAgent | 审查专家 | 无 | 否 |

关键实现点（对应需求逐条）：

- **`Agent.as_tool(tool_name=..., tool_description=...)`**（需求 6）把每个
  专家 Agent 包装成一个 `FunctionTool`，加入 SupervisorAgent 的 `tools=[...]`
  列表——不是 `handoffs=[...]`（需求 7：完全不出现 Handoff）。
- 三个专家 Agent 各自没有 `tools`/`handoffs`（测试
  `test_specialists_have_no_tools_or_handoffs_of_their_own` 断言），指令里也
  明确写"你的输出不会直接展示给用户，会被 Supervisor 读取整合"。
- SupervisorAgent 的 instructions 显式允许"零个、一个或多个"专家调用，并要求
  "最终回复必须由你自己撰写、整合，不要原样粘贴专家输出"（需求 1、4、5）。
- Tracing 复用 `src/tracing.py`，本地 `traces/trace.jsonl` 中可以看到
  `SupervisorAgent` 的 agent-run span 下嵌套着 `research_expert`/
  `review_expert` 的 function span，其下又各自嵌套一个完整的
  `ResearchAgent`/`ReviewAgent` agent-run span（需求 10）。

## 1. 真实运行观察（测试问题）

```powershell
.\.venv\Scripts\python.exe -m src.03_supervisor "比较 LangGraph 和 OpenAI Agents SDK，并给出推荐。"
```

实测输出（节选）：

```text
SUPERVISOR CALLED: research_expert
SUPERVISOR CALLED: review_expert
ANSWERED BY: SupervisorAgent
```

即 Supervisor 依次调用了 `research_expert`（拿到调研结论）和 `review_expert`
（审查这份结论是否可靠），随后自己写出了包含明确推荐的最终回答——满足"至少调用
ResearchAgent 和 ReviewAgent"的要求。

从 `traces/trace.jsonl` 中提取的 span 类型/名称：

```text
('agent', 'ResearchAgent')
('agent', 'ReviewAgent')
('agent', 'SupervisorAgent')
('function', 'research_expert')
('function', 'review_expert')
```

三层结构清晰可见：`SupervisorAgent`（顶层 agent-run）→ `research_expert`/
`review_expert`（普通 function span）→ `ResearchAgent`/`ReviewAgent`（嵌套的
子 agent-run）。整个 trace 里没有任何 `handoff` 类型的 span——因为这次实验从
未使用 Handoff。

`tests/test_supervisor.py` 把这个真实场景固化成了离线可重复的测试
（`SupervisorCanCallMultipleExpertsTest.test_required_scenario_calls_research_then_review_experts`），
用 `ScriptedModel` 让 Supervisor 依次调用 `research_expert`、`review_expert`，
断言 `result.last_agent.name == "SupervisorAgent"` 且 `coding_expert` 未被调用。

## 2. Handoff vs Agents-as-Tools

SDK 自己在 `Agent.as_tool()` 的文档字符串里给出了权威定义（见
`agents/agent.py`）：

> This is different from handoffs in two ways:
> 1. In handoffs, the new agent receives the conversation history. In this
>    tool, the new agent receives generated input.
> 2. In handoffs, the new agent takes over the conversation. In this tool,
>    the new agent is called as a tool, and the conversation is continued by
>    the original agent.

对照本项目两个实验的实际代码，展开成下表：

| | Handoff（实验 2，`src/02_handoff.py`） | Agents-as-Tools（本实验，`src/03_supervisor.py`） |
|---|---|---|
| 注册方式 | `Agent(handoffs=[handoff(agent, ...)])` | `Agent(tools=[agent.as_tool(...)])` |
| 目标 Agent 收到什么输入 | **完整对话历史**（`HandoffInputData`，默认包含此前所有轮次） | **模型生成的一段输入文本**（工具调用的 `input` 参数），是一份被转述/整理过的材料，不是原始对话 |
| 调用之后谁继续主导对话 | **目标 Agent**：`Runner` 切换 `current_agent`，原 Agent 不再参与 | **发起调用的 Agent**（本实验里始终是 SupervisorAgent）：目标 Agent 跑完一次完整的嵌套 `Runner` 后，结果作为字符串工具结果，喂回原 Agent 继续它自己的这一轮 |
| 谁产出 `final_output` | 目标 Agent（`result.last_agent` 变成目标 Agent） | 发起方 Agent（`result.last_agent` 永远是 Supervisor，测试
`test_supervisor_is_always_the_last_agent` 断言这一点） |
| 对应的 run item 类型 | `HandoffCallItem` + `HandoffOutputItem`（无"返回值"概念） | 普通 `ToolCallItem` + `ToolCallOutputItem`（有明确的字符串返回值） |
| 能否同一次运行里调用多个目标 | 不适合：一次 Handoff 就转移了控制权，原 Agent 出局 | 天然支持：可以调用零个、一个或多个专家，任意顺序、任意次数（本实验 `test_supervisor_can_call_the_same_expert_more_than_once` 验证） |
| 专家是否需要知道自己"被谁调用" | 需要一定的意识（接过对话后要独立完整地回答用户） | 不需要：专家只是"一次函数调用"的执行者，甚至不知道自己在被 Supervisor 调用 |

## 3. 逐题回答

### 1. 谁控制最终答案？

**Agents-as-Tools 模式下，始终是 Supervisor。** 无论调用了几个专家、调用顺序
如何，`result.last_agent` 永远是 `SupervisorAgent`——因为专家 Agent 的整个
执行只是 Supervisor 那一次工具调用的内部实现细节（一次嵌套的、独立的
`Runner` 运行），执行完就把结果作为字符串"返回"给 Supervisor，Supervisor 的
Agent Loop 并未中断，仍由它决定"这个结果够不够，要不要再调用别的专家，最终怎么
组织语言回复用户"。这与 Handoff 相反：Handoff 里 `result.last_agent` 会变成
被转移到的那个 Agent（见 `docs/10-HANDOFF.md`）。

### 2. Specialist 是否直接面对用户？

**不直接面对。** 三个专家 Agent 的输出都以 `ToolCallOutputItem` 的形式出现在
`result.new_items` 里（本实验 `extract_expert_tool_results` 提取的就是这些
字符串），这些内容只被 Supervisor 的下一轮模型调用读取，从未作为
`result.final_output` 直接展示给用户。测试
`test_research_expert_output_is_fed_back_as_tool_result_not_final_output`
显式验证：专家原始输出（`"RAW-RESEARCH-NOTES"`）出现在工具结果里，但
`final_output` 是 Supervisor 自己整合后的文本，两者不相等。这与 Handoff 相反：
一旦 Handoff 发生，被转移到的 Agent 的下一次回复就会直接呈现给用户。

### 3. 谁决定下一步？

**Supervisor 的模型，在它自己的每一轮推理里决定。** 每次专家工具调用返回后，
执行权回到 Supervisor 的 Agent Loop，由 Supervisor 的模型基于"已经拿到的工具
结果"重新判断：是否已经足够回答、是否需要再调用别的专家（甚至重复调用同一个
专家）、还是可以收尾给出最终答案。整个决策链条自始至终没有离开 Supervisor 一步
——不像 Handoff 里，一旦转移，"下一步做什么"的决定权也随之转移给了目标 Agent。

### 4. 谁拥有 conversation control？

**Supervisor 独占，且贯穿整个运行。** "Conversation control" 这里指"谁的
Agent Loop 是当前活跃的、谁的 instructions 在起作用、谁能看到并处理原始用户
输入"。在 Agents-as-Tools 模式下，专家 Agent 甚至看不到原始的用户提问——它们
只看到 Supervisor 生成的、经过转述的工具调用参数（SDK 文档原话："the new
agent receives generated input"）。控制权从始至终没有转移，只是临时"借用"了
专家的能力来获取一段材料，这也是为什么这个模式被称为 Supervisor/Manager：
Supervisor 更像一个会安排下属去调研、但自己签发最终报告的经理，而不是把整个
会议移交给某个下属主持。

### 5. 哪种模式更适合 Customer Support？

**通常是 Handoff。** 客服场景的核心诉求是"专业对口"：用户的整个后续对话应该
无缝地由更合适的专家（退款专员、技术支持、账单专员）来处理，包括追问细节、
维持上下文、给出符合该领域规范的多轮回复。如果每次都要经过一个"经理"中转，会
增加不必要的往返和信息损耗，而且经理未必具备逐字转述所有细节的能力（Handoff
传递的是完整对话历史，比转述更不容易丢信息）。此外，客服场景通常不需要"同时
综合多个专家的意见"，而是"转给对的人，让他一路负责到底"，这正是 Handoff 的
设计初衷。

### 6. 哪种模式更适合 Research？

**通常是 Agents-as-Tools（Supervisor 模式）。** 研究任务的核心诉求恰恰相反：
往往需要综合多个来源/多个角度的信息才能得出一个可靠结论——比如本实验的测试
场景"比较 A 和 B，并给出推荐"，天然需要"先调研，再审查调研结论是否可靠，最后
自己权衡给出推荐"，这是一个需要留在同一个"大脑"里做综合判断的任务。如果用
Handoff，一旦转给 ResearchAgent，控制权就转移过去了，没有人再回来做"审查"和
"跨专家整合"这一步；而 Supervisor 模式下，Supervisor 可以按需调用
`research_expert`、`review_expert`（甚至反复调用），始终由同一个决策者把
多次调用的结果串联成一份连贯、有明确推荐的最终答案。
