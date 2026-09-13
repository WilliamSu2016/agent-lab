# 01. Specialist Agents

> 实验范围：`src/specialists/research_agent.py`、`coding_agent.py`、`review_agent.py`。
> 三个 Agent 均使用 LangGraph 的 `StateGraph` 实现，各自是一个只有一个 Node 的图
> （`START -> <node> -> END`）。三者之间**没有** Handoff、没有 Supervisor/Manager、
> 没有 Shared State、没有 MCP；每个 Agent 有独立的 `instructions`、独立的输入/输出
> `TypedDict` State、独立的测试文件。本文档只讨论"专业化"这一件事，不涉及后续实验
> （Handoff、Supervisor、Parallel、Shared State）。

## 0. 本次实现了什么

| Agent | 输入 | 输出 | 职责边界 |
|---|---|---|---|
| ResearchAgent | `topic: str` | `research_report: str`（摘要 / 关键要点 / 风险与不确定项） | 只研究技术资料，不设计代码，不做评审 |
| CodingAgent | `requirement: str` | `design_proposal: str`（方案概述 / 模块结构 / 关键实现 / 权衡与风险） | 只设计代码方案，不做背景调研，不做评审 |
| ReviewAgent | `artifact_type: "research_report" \| "code_design"`, `content: str` | `review_result: str`（结论 / 问题 / 缺口 / 改进建议） | 只审查给定材料本身，不重做研究，不重新设计方案 |

三个 Agent 的图结构完全一样（一个 Node），差异只在 `instructions` 和 State 的字段
含义上——这正是"专业化"应该发生的地方：不是图结构、不是工具集合，而是任务范围、
指令内容和输入输出契约。

---

## 1. 为什么 Specialist Agent 可能比一个万能 Agent 更好？

一个"万能 Agent"（同一份 instructions 里塞进"你既要研究、又要设计代码、还要审查"）
会遇到三类具体问题，而不是抽象的"更聪明/更笨"之分：

1. **注意力被稀释，输出契约模糊。** 当 instructions 同时描述三种任务的输出格式时，
   模型必须在一次回复里决定"现在该产出哪种格式"，容易出现格式漂移（比如该给
   "结论：通过/需要修改"时却给了一段研究摘要）。拆分之后，每个 Agent 的
   `INSTRUCTIONS` 只描述一种输出契约，格式更稳定，也更容易写断言式测试（见
   `tests/test_*_agent.py` 里对固定小节标题的检查）。

2. **职责边界不清会导致"既是运动员又是裁判"。** 如果研究和评审在同一个 Agent
   里，它评审自己刚做出的研究结论时，缺少真正独立的视角——本质上是同一次推理在
   自我确认。拆成 ResearchAgent 和 ReviewAgent 之后，ReviewAgent 只被给予
   `content`，没有生成这份内容时的推理过程或上下文，能提供更接近"独立复核"的
   检查（即便目前还没有真的把两者串起来运行，这个边界已经在 instructions 和
   State 契约里体现出来）。

3. **可以独立维护、独立测试、独立演化。** 三个 Agent 各自的 `INSTRUCTIONS` 和
   State 是独立文件、独立测试。改进 CodingAgent 的输出格式不需要重新验证
   ResearchAgent 的行为；反之亦然。这是工程上的收益，不是"更聪明"，而是"改动
   范围更小、回归风险更低"。

**代价同样要承认**：三个独立 Agent 意味着如果真要把它们的产出串起来（比如让
ReviewAgent 检查 ResearchAgent 的报告），需要额外的编排逻辑把一个 Agent 的输出
转成另一个的输入——这正是本实验刻意不做的事情（留给后续 Handoff/Supervisor
实验）。如果任务本身不需要这种独立评估或独立维护的收益，拆分只会增加没有回报
的协调成本。

## 2. Specialist Agent 的 instructions 应该如何设计？

本实现里每个 `INSTRUCTIONS` 都遵循同一个模板，模板本身就是设计要点：

1. **一句话身份声明**："你是 XAgent，一名只负责 Y 的专家。" 用"只负责"把职责
   边界写在第一句，避免模型把边界当作可选建议。

2. **"职责范围（只做这些）"清单。** 正面列出应该做的事情，具体到可验证的动作
   （"研究并总结事实、原理、权衡取舍"，而不是笼统的"帮助用户"）。

3. **"明确不做的事情"清单。** 这一节和第 2 节同等重要，甚至更重要：它显式排除
   相邻职责（CodingAgent 的 instructions 明确写"不做背景技术调研"、"不评审别人
   的方案"），防止模型在输入模糊时"顺手"越界去做别的 Agent 的工作。没有这一节，
   专业化只是名义上的——模型仍然可能因为"看起来有帮助"而去做研究之外的事。

4. **固定的输出格式（小节标题）。** 每个 Agent 的输出都用编号小节强制结构化
   （比如 ReviewAgent 的"结论 / 发现的问题 / 缺口 / 改进建议"）。这既是给模型
   的约束，也是给测试和未来编排逻辑的约束——固定格式才能被可靠地解析或断言。

5. **"信息不足时怎么办"的显式规则。** 三份 instructions 都要求"明确说明缺口，
   不要编造"。这是防止专业化产生虚假自信的关键：一个只被给了一小段材料的
   ReviewAgent，很容易在信息不足时"脑补"出材料应该包含的内容；显式规则把这种
   情况的正确行为（报告缺口）写进指令，而不是依赖模型自己判断。

**不应该做的事**：把 instructions 写成"你是某领域的专家，请尽力回答"这种笼统
声明。专业化的关键不是身份标签，而是第 2-5 点这些具体的、可测试的约束。

## 3. Specialist Agent 是否应该知道其他 Agent 的存在？

**在本实验（Specialist Agents，尚未引入 Handoff/Supervisor）的范围内：不应该。**

三点具体理由，对应到代码里的可验证约束：

- **模块层面互不引用**：`research_agent.py`、`coding_agent.py`、
  `review_agent.py` 互相之间没有 `import`；`tests/test_specialist_independence.py`
  用 AST 解析显式断言这一点。任何一个 Agent 的实现细节改变，都不会波及另外两个。

- **instructions 层面不提及对方**：每个 `INSTRUCTIONS` 都不出现另外两个 Agent
  的名字（同样有测试断言）。这不是"防止模型作弊"的技巧问题，而是契约设计问题：
  如果 ResearchAgent 的指令写"你的结果会被 ReviewAgent 检查"，它就可能因此改变
  行为（比如刻意留白让审查者去发现问题，或者因为"反正有人会查"而降低自我把关的
  标准）。要求"把你的产出当作会被直接使用的最终产出"，恰恰是为了让每个 Agent
  独立时的质量本身就是可靠的，不依赖下游还有别的 Agent 来兜底。

- **输出契约是"通用材料"而不是"专属消息"**：ReviewAgent 的输入是
  `artifact_type + content` 这种与来源无关的通用结构，不是"来自 ResearchAgent
  的消息"。这意味着 ReviewAgent 同样可以审查一段人工撰写的研究报告或代码方案，
  它的职责定义不依赖"上游是谁"。

**但这不是永久结论。** 一旦进入 Handoff 或 Supervisor 实验，编排层（Supervisor
或触发 Handoff 的 Agent）需要知道有哪些 Agent 可以调度——但即便在那种架构下，
更常见的做法也是"编排者知道 Agent 列表，Agent 本身仍然只描述自己的职责"，而不是
让 ResearchAgent 自己知道"审查完我要转给 ReviewAgent"。是否需要让 Agent 本身
感知彼此，取决于具体的协作模式（比如对等的多轮协商可能需要，简单的流水线不需要），
应在引入该模式时单独设计，而不是现在就预先埋点。

## 4. 一个 Agent 应该承担多少职责？

从本实现可以归纳出一个可操作的判断标准，而不是"越窄越好"这种空泛原则：

**一个 Agent 应该只承担"一种输出契约"能自洽描述的职责。**

具体检验方法：

1. **能否用一份固定格式的输出模板覆盖它的全部职责？** ResearchAgent 的"摘要 /
   关键要点 / 风险"三段式覆盖了"研究"这一件事；一旦职责扩展到"研究 + 给出代码
   建议"，就需要至少两套不兼容的输出模板，这是职责该拆分的信号。本实现中三个
   Agent 分别对应三套互不相同的小节模板，这不是巧合，而是拆分依据。

2. **"明确不做的事情"清单是否短且自然？** 如果一个 Agent 的"不做"清单开始变得
   很长、很牵强（需要不断补充"哦对了，也不要做 X、不要做 Y、不要做 Z"），说明
   它的"职责范围"部分定义得不够收敛，正在被期待去做太多事。本实现里每个 Agent
   的"不做"清单只有 2-3 条，且都是相邻职责（研究↔设计↔评审之间），是刻意保持
   小的信号。

3. **能否脱离另外两个 Agent 独立测试、独立验证？** 三个 `tests/test_*_agent.py`
   文件互相之间没有依赖，每个都能用假的 `TextLLMCall` 独立跑通并断言输出契约。
   如果某个职责必须结合另一个 Agent 的运行结果才能验证是否正确，说明这两个职责
   之间的耦合度已经超过"专业化拆分"应有的程度，可能本该是同一个 Agent 内部的
   两个步骤（Prompt Chaining），而不是两个 Specialist Agent。

4. **不要用"节点数"或"工具数"衡量职责大小。** 本实验里三个 Agent 都只有一个
   LangGraph Node、零工具调用，职责差异完全体现在 instructions 和 State 字段
   上。职责大小的度量单位是"输出契约的种类"，不是实现的复杂度。一个只有一个
   Node 的 Agent 完全可能职责过重（比如把研究和设计糅合进同一份 instructions 里
   还是一个 Node）；一个有多个 Node 的 Agent（比如 04-langgraph 里的
   ResearchAgent，含 planner/agent/tools/finalize 四个 Node）也完全可能职责很
   聚焦（这四个 Node 都服务于"完成一次研究"这一个职责）。

**经验法则**：如果为一个 Agent 写 instructions 时，"职责范围"和"明确不做的事情"
两节无法在几句话内讲清楚，或者输出格式需要用"如果是 A 情况输出 X 格式，如果是
B 情况输出 Y 格式"这种分支来描述，这通常意味着该拆成两个 Specialist Agent。
