# 01 — Prompt Chaining：Planner → Researcher → Analyst → Writer

对应代码：`src/prompt_chaining.py`、`tests/test_prompt_chaining.py`。
运行方式：`.\.venv\Scripts\python.exe -m src.prompt_chaining "比较 Python、TypeScript、Go 哪个更适合开发 AI Agent，并给出推荐。"`

已实际运行一次完整任务（见本文末尾"实际运行观察"），并非只是理论设计。

## 1. 这是什么

固定的 4 个步骤，由程序顺序调用，每一步是一个独立的 `agents.Agent`：

```text
用户问题
  -> Step 1 Planner   （output_type=ResearchPlan）
  -> Step 2 Researcher（tools=[search_web], output_type=ResearchBundle）
  -> Step 3 Analyst   （无工具，纯文本输出）
  -> Step 4 Writer    （无工具，纯文本输出）
最终答案
```

四个 `build_xxx()` 函数各自创建一个 `Agent`，`run_prompt_chain()` 按固定顺序调用四次 `Runner.run_sync`，每次调用的输入只由上一步已校验过的输出拼出。

## 2. 为什么这是 Prompt Chaining，而不是别的模式

- **固定顺序、固定步骤数**：`run_prompt_chain()` 里 Step 1→2→3→4 的顺序写死在 Python 代码里，任何一次运行都不会跳步、加步或改变顺序。
- **每步职责单一**：Planner 只出研究维度；Researcher 只查资料、不下结论；Analyst 只比较、不写终稿；Writer 只写终稿、不查资料。每个 `*_INSTRUCTIONS` 都显式写了"不要做什么"（如 Planner 的 instructions 明确写 "Do NOT answer the user's question yourself"）。
- **Gate 是结构化输出**：Step 1 用 `output_type=ResearchPlan`、Step 2 用 `output_type=ResearchBundle`，这些 Pydantic 模型起到了原文所说的"gate"作用——SDK 在把结果交给下一步之前，会先校验 JSON 是否满足 schema（例如 `dimensions` 至少 2 项）；不满足就直接抛错，不会把半成品悄悄传下去。
- **没有 Handoff**：四个 Agent 都没有设置 `handoffs`（见 `test_no_agent_declares_handoffs`），下一步永远由本模块的 Python 代码显式调用 `Runner.run_sync`，不是某个 Agent 在运行时把控制权"交给"另一个 Agent。
- **没有 Orchestrator，没有动态任务分解**：没有一个中央 LLM 决定"这次需要几步、分别做什么"。步骤数、步骤职责、步骤顺序全部写死；Planner 生成的只是"研究维度"这个数据，不是"下一步该跑哪个 Agent"这种控制流决策。

## 3. 哪些部分是 deterministic

| 确定性部分 | 在代码里的位置 |
|---|---|
| 4 个步骤的存在、顺序、且必须全部执行 | `run_prompt_chain()` 里硬编码的 4 个 `try/except` 块 |
| 每步输入必须来自上一步的已校验输出（不能跳步/回退） | `researcher_input`、`analyst_input`、`writer_input` 的拼接逻辑，只使用前一步的返回值 |
| 每步的工具集合固定 | `build_planner`/`build_analyst`/`build_writer` 不传 `tools`；只有 `build_researcher` 传 `tools=[search_web]` |
| 结构化输出是否合法 JSON、是否满足 schema（如 `dimensions` 数量 ≥2） | `output_type=ResearchPlan`/`ResearchBundle`，由 SDK 的 `AgentOutputSchema.validate_json` 执行 |
| 失败时报告的步骤名 | `PromptChainStepError(step_name, cause)`，四个 `except` 块分别标注 `"1-planner"`/`"2-researcher"`/`"3-analyst"`/`"4-writer"` |
| Step 3、4 的输出必须是纯文本（不是意外的结构化 JSON 或工具调用） | `isinstance(..., str)` 检查，不满足则 `TypeError` 并归为该步失败 |

这些是"过程规则"层面的确定性：给定同样的上一步输出，这一步会被同样地调用、同样地校验、同样地报错。它们**不代表**模型每次给出的文字内容是确定性的。

## 4. 哪些部分由 LLM 决定

| LLM 决定的部分 | 说明 |
|---|---|
| 研究维度具体是什么、有几个（3–6 个范围内） | Planner 的 `output_type` 只约束"是不是一个合法的字符串列表"，具体内容完全是模型生成的 |
| 是否调用 `search_web`、调用几次、每次的查询词是什么 | Researcher 的 instructions 建议"每个维度至少查一次"，但这是提示词层面的约束，不是代码强制；调不调、怎么查都是模型在工具调用循环里自行决定的（Runner 负责执行调用和把结果喂回去，但"要不要调"是模型决定的） |
| 每个维度的 findings 内容、是否承认证据不足 | Researcher 生成 |
| 具体的比较分析、哪个维度证据强/弱 | Analyst 生成 |
| 最终推荐哪个语言、理由怎么组织 | Writer 生成 |

## 5. 为什么不能叫它 Autonomous Agent

按 Anthropic 原文的定义，Agent 的关键特征是"LLM 动态地指挥自己的过程，并在执行中掌控如何完成任务"。这个实现不满足这一点：

- **没有模型决定"下一步做什么"**：四个步骤的存在和顺序是 `run_prompt_chain()` 里的 Python 代码决定的，不是任何一次 LLM 调用的输出决定的。即使 Researcher 决定不调用 `search_web`，Analyst 和 Writer 仍然会照常运行——步骤本身不受模型的意愿影响。
- **没有可变的终止条件**：流程总是跑满 4 步就结束（或者在某一步失败时报错退出），没有"模型判断已经完成、可以提前停止"或"模型判断还需要更多轮"的机制（这一点与 Evaluator-Optimizer 的循环形成对比）。
- **模型看不到、也管不了整个流程**：Planner 看不到 Researcher/Analyst/Writer 会怎么处理它的输出；Writer 也无法要求"回到 Researcher 重新查一次"。每个 Agent 的视野被限定在自己的一步之内。
- **唯一的工具局限在一步之内**：`search_web` 只挂在 Researcher 上，且 Researcher 用完工具产出结构化结果后就结束了这一步，不存在"模型持续与环境交互、根据反馈调整计划"的循环。

一句话：这里的 4 个 LLM 调用都是在**填内容**，不是在**定过程**。过程永远由这份 `.py` 文件里的代码决定。

## 6. 相比单 Agent 的优势和缺点

### 优势

- **每步任务更简单、更可控**：单个 Agent 要同时"规划维度 + 搜索 + 比较 + 写终稿"，容易在长回复里跑题或漏项；拆开后每一步只需要做好一件事，配合窄范围的 instructions 更容易稳定。
- **中间结果可检查、可复用**：`PromptChainResult` 把 `plan`/`research`/`analysis`/`final_answer` 都保留下来，可以单独检查 Planner 给的维度是否合理、Researcher 是否诚实报告了证据不足，而不必只看最终一段话。
- **失败定位精确**：`PromptChainStepError.step_name` 直接告诉你是哪一步坏的（比如结构化输出解析失败），不需要在一大段 Agent 轨迹里排查。
- **权限最小化**：只有 Researcher 能碰网络工具，其余三步完全没有工具访问权限，减少了意外行为的攻击面。

### 缺点

- **延迟和成本更高**：4 次独立的 LLM 调用（Researcher 内部还可能多次调用工具）比 1 次单 Agent 调用慢、贵得多。
- **错误会向下游传递**：如果 Planner 给出的研究维度本身有偏差或遗漏，后面三步都会在这个偏差之上继续工作，且这几步都无法反过来质疑或修正 Planner。
- **僵化，无法适应意外情况**：如果某个问题根本不需要 4 步（比如已经有现成答案），或者需要 Researcher 反复检索验证，这套固定流程既不能跳过步骤，也不能增加步骤。
- **上下文/证据在传递中变形**：Researcher 的结构化 `ResearchBundle` 被 `_format_research_notes()` 拍平成纯文本再喂给 Analyst，这中间可能丢失一些结构化信息（比如来源列表和正文摘要之间的对应关系）。

## 7. 实际运行观察

用真实网关跑了一次完整任务（问题就是本文档开头的中文问题），观察到：

- Planner 生成了 5 个研究维度（如"AI/LLM 与 Agent 框架及模型 SDK 生态""异步并发与流式处理及长任务执行能力"等）——数量在 instructions 允许的 3–6 之间，具体内容完全是模型自选的。
- Researcher **确实调用了 `search_web`**（trace 显示至少几十次工具调用 span，例如查询 `"Python TypeScript Go async concurrency streaming long running tasks AI agents official documentation"`），但这次真实调用的 DuckDuckGo instant-answer 接口对这类多关键词技术问题返回了 `results: []`（空结果）。
- 面对空结果，Researcher **没有编造事实**，而是按 instructions 要求老实地在 `findings` 里写"本轮尚未获得可核实的检索结果"——这正是"Do NOT skip... say so explicitly instead of guessing"这条规则起作用的证据，而不是工具调用失败或代码 bug。
- Analyst 相应地把每个维度都标注为"证据不足，无法判定优劣"，Writer 最终给出的是一个**如实说明证据不足、并给出替代性决策建议**（按团队熟悉度选型 + 用小型 PoC 验证）的回答，而不是在没有证据的情况下硬编造一个"Python 更好"的结论。

这次真实运行也印证了第 3/4 节的区分：`search_web` 是否被调用、返回什么内容、模型如何措辞是 LLM/环境决定的；但"这一步一定会跑""跑完必须是合法 JSON 才能进入下一步""失败要报告在哪一步"这些始终是代码保证的确定性行为。
