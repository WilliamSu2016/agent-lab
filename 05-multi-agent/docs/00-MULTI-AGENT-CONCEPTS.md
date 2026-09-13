# Multi-Agent Architecture：概念、模式与适用边界

> 研究日期：2026-09-11。
> 范围：OpenAI Agents SDK Python 当前官方文档、LangChain / LangGraph 当前官方文档，以及 Anthropic 的 Multi-Agent Research System 工程文章。
> 本文仅研究架构，不包含实现代码。框架行为以研究日读取的文档为准；历史工程案例不代表当前产品内部实现。

## 0. 先建立正确的认识

**Multi-Agent 的价值不是让更多角色“讨论”，而是让不同执行单元在明确的责任、上下文和控制边界内协作。**

优先从 Single Agent 开始。只有当上下文、能力边界、任务分解或并行需求形成可观察的瓶颈，且增加协调成本后仍有净收益，才升级为 Multi-Agent。OpenAI 明确区分由 LLM 决定流程和由代码编排流程；LangChain 明确提醒，合适的单 Agent、动态工具和提示往往已经足够。[O1][L1][A2]

本文区分两类内容：

- **官方机制与案例事实**：附来源编号，描述框架语义或文章报告的结果。
- **工程建议**：基于这些资料归纳的设计、预算和评估方法，不是框架保证或通用性能阈值。

### 0.1 什么算一个 Agent？

在本文的操作性定义中，Agent 是具有任务指令、可见上下文和可用能力，并能根据反馈决定后续行动的 LLM 驱动执行单元。典型行为是“判断下一步、调用工具、读取结果、继续或结束”。Anthropic 的 Research 文章采用自主循环使用工具的 LLM 这一描述。[A1]

但框架命名比这个定义更宽：OpenAI 的 `Agent` 也可以只生成一次回答，官方 Agents as Tools 示例就包含简单翻译 Agent。**因此应区分“框架中的 Agent 对象”与“架构上值得独立管理的自主工作单元”。** 不能仅凭对象数量证明系统获得了多 Agent 的收益。[O3]

多个 Agent：

- 可以使用同一个基础模型，甚至使用同一套能力，但处理不同任务、拥有不同运行上下文。
- 不必运行在不同机器或进程，也不必同时执行。
- 不等于多个模型；一个 Agent 也可以通过运行时配置选择模型。
- 不等于多个提示词、多个工具、多个图节点或多次 LLM 调用。

### 0.2 三个经常混淆的维度

| 维度 | 要回答的问题 | 不应混淆为 |
|---|---|---|
| 能力：Multi-Tool | 一个执行者能调用哪些工具？ | 工具多就是 Agent 多 |
| 流程：Multi-Step Workflow | 哪些步骤、分支、循环按什么规则执行？ | 步骤多就是 Agent 多 |
| 责任与上下文：Multi-Agent | 有哪些执行者，谁负责什么，怎样交接与协作？ | 有多个角色名就构成有效分工 |

三者可以同时成立：一个固定 workflow 可以包含多个 Agent，每个 Agent 又有多个工具；一个 Single Agent 也可以自主完成很多步骤。[O1][L5][L6]

---

## 1. 十个重点概念

这十项并不处于同一分类层级：Specialist 描述职责；Handoff 和 Agents as Tools 描述委派语义；Supervisor 和 Orchestrator-Workers 描述组织方式；Parallel 描述调度方式；State、Communication、Context 描述信息管理；Evaluation 判断整套设计是否有效。

### 1.1 Specialist Agents：把复杂能力变成有边界的执行单元

**解决的问题：一个通用 Agent 同时背负过多领域规则、工具和无关上下文，导致注意力分散、选工具错误或任务边界不清。**

专业化不只是给提示词加上“你是专家”，而是定义任务范围、领域指令、数据来源、可用工具、输出契约和完成标准。例如，研究系统可以将价格调查与技术兼容性调查分别交给专门执行者，再由主 Agent 汇总。[O1][L1][L2]

| 应定义的边界 | 研究场景示例 |
|---|---|
| 目标与非目标 | 只研究公开定价，不评价性能，不重复兼容性调查 |
| 所需上下文 | 产品列表、使用规模、地区、币种、时间范围 |
| 工具与来源 | 优先官方定价页；不把搜索摘要当作最终证据 |
| 输出契约 | 价格事实、来源、适用条件、未确认项目 |
| 完成与退出条件 | 覆盖指定产品；无法确认时报告缺口而非编造 |

这些是建议的任务契约，不是特定 SDK 的固定字段。

专业化的收益包括更聚焦的上下文、更小的工具选择空间，以及独立维护和评估能力。代价是委派、结果回传和跨领域信息整合。

**不必拆 Agent 的情况：** 如果只需要按需加载某个领域的知识或提示，单 Agent 加 Skills 就可能足够。当前 LangChain 将 Skills 明确描述为“同一个 Agent 保持控制，按需加载专业知识”。子 Agent 的另一种合理用途则纯粹是上下文隔离，并不要求它具有不同模型或工具。[L2][L7]

### 1.2 Handoff：转移当前处理权

**解决的问题：后续对话或流程应该由更合适的执行者接管，而不是让入口 Agent 永远充当传话人。**

概念流程：

```text
User -> Triage Agent --handoff--> Refund Agent -> User
```

例如，入口 Agent 识别到退款问题后，退款 Agent 接手询问订单、解释规则并继续处理；这不同于入口 Agent 只向退款专家咨询一次，再自己回答。

**OpenAI Agents SDK 的具体语义：**

- Handoff 对模型表现为工具，但运行时语义是切换到目标 Agent。
- 默认接收方看到先前的会话历史；可通过 `input_filter` 等机制改变其可见输入。
- `input_type` 定义交接工具的参数，例如原因或优先级；它不替代接收方的完整输入，也不自行改变目标 Agent。
- Handoff 发生在同一次 run 内。接收方可以继续交接；这不是调用专家后自动返回的函数调用语义。
- 若下一轮仍应由接收方处理，应用还需正确延续会话状态和活动 Agent，不能把一次交接理解为自动永久转接。[O1][O2]

**LangChain / LangGraph 的术语更宽：** 当前 Handoffs 文档将其描述为状态驱动的行为切换，可以切换多个 Agent，也可以只更新同一个 Agent 的工具和提示。因而“发生 handoff”不总能证明底层有多个独立 Agent。[L3]

好的交接至少传递目标、已知事实、已完成动作、未解决问题和约束。不要默认“转发全部历史”最好，也不要压缩到只剩一句“请继续处理”。

### 1.3 Manager / Supervisor：集中协调与最终责任

**解决的问题：多个专家的工作需要统一拆分、选择、补充和汇总，并且要有一个明确的最终回答负责人。**

```text
User -> Supervisor -> Specialist A -> Supervisor
                   -> Specialist B -> Supervisor -> User
```

Supervisor 一般负责选择专家、分配任务、读取结果、识别缺口、决定是否继续以及形成最终回答。它可以顺序或并行调用专家，不是天然只能串行。[O1][L2]

注意三个区别：

| 概念 | 主要职责 |
|---|---|
| Router | 分类并派发请求，通常不持续主持后续协作 |
| Supervisor | 根据结果持续决定下一步，对整体任务负责 |
| 确定性调度器 | 由代码按规则分配任务，不一定包含 LLM 决策 |

如果 Supervisor 只是机械地先调用 A、再调用 B、最后拼接结果，普通 workflow 可能更直接。真正需要 Supervisor 的地方，是下一步依赖中间发现，不能简单预设。[L4][L5]

Supervisor 的风险也很集中：拆错任务会影响全局；过度汇总可能丢失证据；反复规划会增加成本；同步等待可能被最慢的 worker 阻塞。**集中负责不等于全知，也不等于天然可靠。**

**当前文档提示：** 官方迁移指南说明 `langgraph-supervisor` 包已不再积极维护，推荐使用 `create_agent` 配合工具封装的 subagents。这是实现入口的变化，不是 Supervisor 架构失效。[L10]

### 1.4 Agents as Tools：把 Agent 能力封装为可调用子任务

**解决的问题：主 Agent 想利用另一 Agent 的自主能力，但不想交出用户对话的主导权。**

```text
Manager --task input--> Specialist Agent
Manager <--task result-- Specialist Agent
Manager -> Continue / Call another specialist / Answer user
```

对主 Agent 来说，这是一个工具接口；对被调用者来说，它可以运行自己的多步推理与工具使用流程。OpenAI 提供 `Agent.as_tool()`；当前 LangChain 的 Subagents 模式同样采用主 Agent 调用工具封装的子 Agent。[O3][L2]

这也解释了 Multi-Tool 与 Multi-Agent 的交集：如果一个工具内部封装完整 Agent，外层看起来仍然只是工具调用，但系统内部已经包含多个 Agent。反过来，一般数据库查询或计算工具没有独立的 Agent 控制循环。

当前 OpenAI 文档明确区分两种状态：

- 子 Agent run 不自动继承父 run 的会话历史；需要时显式配置输入或会话延续机制。
- 这不意味着应用状态隔离：嵌套 `Agent.as_tool()` 运行默认并不会获得应用 context 的独立副本。[O3][O4]

工程上应把它当作有成本和失败语义的服务调用：输入要完整、输出要可验证、耗时要受限、错误要明确。默认不让子 Agent直接承担用户对话，并不意味着所有框架都禁止它请求澄清或人工审批。[L2]

### 1.5 Orchestrator-Workers：动态拆分未知规模的工作

**解决的问题：事先不知道需要多少个子任务、每个子任务具体是什么，必须先分析输入，再生成工作分配。**

```text
Request -> Orchestrator -> Dynamic task plan
                              |-> Worker 1 --|
                              |-> Worker 2 --|-> Synthesis
                              |-> Worker N --|
```

例如，先分析一份大型调查请求，发现需要按公司、地区或议题拆分，再动态派发不同范围的研究任务。LangGraph 官方使用 `Send` 展示动态创建 worker 任务；每个 worker 有自己的输入状态，输出可汇集到共享结果字段。[L6]

与相邻概念的区别：

- **Supervisor** 强调“谁持续协调和负责”；**Orchestrator-Workers** 强调“如何动态分解和分配工作”。同一个系统可以同时采用两者。
- **静态 Parallel Workflow** 的分支预先定义；Orchestrator-Workers 的子任务内容或数量由输入决定。
- **Worker 不一定是 Agent**：LangGraph 的示例 worker 可以只是一次 LLM 调用；Anthropic Research 的 worker 则是自主多轮搜索的子 Agent。[L6][A1][A2]

因此，Orchestrator-Workers 是编排模式，不是“有 orchestrator 就必然是多 Agent”的判定规则。

### 1.6 Parallel Multi-Agent：并行独立工作，而不是并行制造依赖

**解决的问题：任务存在可以同时推进的独立部分，串行执行造成不必要的等待。**

两种常见用途：

| 用途 | 工作方式 | 主要收益与限制 |
|---|---|---|
| 分工覆盖 | 各 Agent 研究不同产品、资料源或问题维度 | 增加覆盖面；需要防止重复和遗漏 |
| 独立复核 | 多个执行者分别判断同一问题 | 可能发现错误；相关性偏差会削弱投票价值 |

适合并行的前提是：任务边界清楚、关键输入已经具备、对同一可变资源的写入不会互相冲突、结果能够合并。

简化的延迟模型：

```text
Sequential time ~= planning + sum(worker times) + synthesis
Parallel time   ~= planning + max(worker times) + synthesis + coordination
```

这是说明关系的估算，不是性能承诺。实际还受并发限制、排队、限流、重试、任务依赖和多轮派发影响；并发数受限时，worker 会分批完成。

**并行减少的是可能的墙钟等待时间，不会自动减少总 token 或总费用。** 也不要把单 Agent 并行调用三个搜索工具误称为三个 Agent。OpenAI 将并行 Agent 运行列为代码编排方式；Anthropic 的 Research 同时采用 Agent 级并行和工具级并行，两层不能混为一谈。[O1][A1]

### 1.7 Shared State：共享必要事实，而不是共享全部思考过程

**解决的问题：各执行者需要围绕同一任务进度、已确认事实、产物和约束保持一致。**

先区分四层：

| 层次 | 示例 | 是否自动成为模型输入？ |
|---|---|---|
| 模型可见上下文 | 指令、消息、工具结果 | 是，前提是实际传入此次模型请求 |
| 应用 / 图状态 | 任务列表、审批状态、结果索引 | 否，需要被选择性呈现 |
| 外部持久存储 | 报告、数据库记录、长期记忆 | 否，需要读取或检索 |
| 运行记录 | trace、耗时、错误和调用关系 | 通常不是业务提示上下文 |

OpenAI 的 `RunContextWrapper.context` 是提供给本地工具、hooks 等使用的应用对象，不自动发给 LLM。模型只有在数据经指令、输入或工具结果呈现后才能利用它。[O4]

LangGraph 用 state schema 描述状态，节点返回更新，reducer 决定如何合并字段更新。**Reducer 是合并机制，不是自动的业务正确性、去重或事务保证。** 并发结果收集也不意味着任何字段都能安全地被多个节点随意修改。[L8]

建议的共享内容包括任务标识、任务状态、已确认事实及来源、产物引用、预算、截止时间、审批状态和错误。每个字段应有明确写入者、更新规则和冲突处理方式。

对于会修改外部系统的任务，还需要幂等、并发控制和审计；仅仅把 `status` 写进共享字典，不能保证副作用只执行一次。

**共享状态与独立上下文不冲突：** 多个研究 Agent 可以只看到各自的资料，同时将带来源的结论写入同一份结果集合。没有必要互相读取全部中间历史。

### 1.8 Agent-to-Agent Communication：设计消息契约，而不是搭建聊天群

**解决的问题：任务、发现、澄清请求、失败和控制权如何跨越 Agent 边界。**

通信有不同形式：主 Agent 委派并接收结果、handoff 传递必要状态、共享状态更新、事件或队列消息、外部产物引用。它既可以发生在同一进程，也可以跨服务；不要求先引入专门的网络协议。[O2][L2][L8][A1]

本文的 Agent-to-Agent Communication 指广义通信，不特指名为 A2A 的互操作协议；跨框架协议选型不在本次范围内。

| 消息类型 | 建议包含的内容 |
|---|---|
| 任务委派 | 目标、范围、非目标、输入、所需输出、预算、截止条件 |
| 结果返回 | 完成状态、结论、证据、产物引用、未解决问题 |
| 交接 | 接收者、交接原因、已完成动作、继续处理所需上下文 |
| 失败 / 澄清 | 阻塞原因、已尝试事项、需要的信息、是否适合重试 |

这是一组工程建议，不是某个 SDK 的标准消息格式。重要的是区分“查无结果”“权限不足”“工具失败”和“尚未执行”，避免全部压成空字符串或成功形状的返回值。

LangGraph 的跨子图 handoff 还要求保持消息结构合法：工具调用消息与对应的 `ToolMessage` 需要正确配对，不能随意截断成失配历史。[L3]

避免无约束的全员广播。若每个 Agent 都与其他所有 Agent 建立双向关系，潜在关系数按 `n(n-1)/2` 增长；这是完全连接拓扑的关系数，不是所有多 Agent 系统的实际消息量。集中协调或分层通信可以缩小交互面，但会带来中心瓶颈。

另一个重要原则是：**子 Agent 输出是待核实的数据，不因来自“内部专家”就成为高优先级指令或自动可信事实。** 工具授权、资源权限和敏感信息过滤仍需在应用边界执行；角色隔离不能代替安全边界。

### 1.9 Independent Context：隔离探索历史，保留必要依赖

**解决的问题：主 Agent 不应承受所有专家的搜索噪声、失败尝试和长篇领域资料；不同探索也不应过早被同一个假设锁定。**

```text
Shared goal / constraints / evidence references
          |                    |
   Agent A context      Agent B context
   A's working history  B's working history
          |                    |
          +-- selected results +--> Coordinator context
```

独立上下文意味着每个执行者的模型输入和工作历史可以分开管理；不意味着完全没有共享信息，也不意味着独立进程、独立内存或天然权限隔离。

| 策略 | 好处 | 代价 |
|---|---|---|
| 仅传任务及必要资料 | 输入小、噪声少、边界清楚 | 缺少背景时容易重复调查或误解 |
| 继承父会话的相关历史 | 容易接续已有工作 | 上下文更大，也可能继承错误假设 |
| 返回摘要和证据引用 | 控制主上下文规模 | 摘要可能丢失限定条件 |
| 返回可检索产物引用 | 减少大输出复制，保留原始成果 | 需要产物存储、检索和访问管理 |

OpenAI 的 Handoff 默认转交历史，而 Agents as Tools 的父会话状态不自动继承；这说明“使用多个 Agent”本身并不保证上下文隔离，隔离取决于具体输入和状态策略。[O2][O3]

LangChain 的 Subagents 文档也区分隔离输入与继承父历史的方式，并允许显式定制输入、输出及持久化策略。不要把默认新上下文误写成“子 Agent 永远不能有记忆”。[L2]

独立上下文还不等于独立判断：同一模型、相同来源、相同提示或相同偏见仍会产生高度相关的错误。复核必须关注证据差异，不能只数赞成票。

### 1.10 Multi-Agent Evaluation：评估整体协作，而不只评价最终文笔

**解决的问题：单个专家看起来聪明，并不能证明整套协作更准确、更划算或更可靠。**

至少评估以下层次：

| 层次 | 核心问题 | 示例指标 / 方法 |
|---|---|---|
| 最终结果 | 是否解决真实任务？ | 正确率、完整性、引用支持率、最终业务状态 |
| 专家能力 | 每个执行者是否完成自己的契约？ | 子任务成功率、输出结构有效率、证据质量 |
| 协作过程 | 拆分、路由、交接和汇总是否有效？ | 遗漏率、重复劳动、错误交接、汇总丢失信息 |
| 效率 | 是否值得增加执行者？ | 总费用、token、工具调用、端到端 P50 / P95 延迟 |
| 可靠性与约束 | 出错后能否安全结束或恢复？ | 超时率、循环、越权、重复副作用、恢复成功率 |

#### 结果评估与轨迹评估应互补

开放式研究通常没有唯一正确路径。不能因为实际搜索顺序与参考轨迹不同就判失败；但授权检查、预算和关键业务前置条件不能因为最终答案正确就被忽略。

OpenAI 当前 Agent Evals 指南建议先用 traces 定位工具选择、handoff 和流程约束问题，再用数据集与 eval runs 做可重复比较。SDK tracing 能记录模型调用、工具、handoff 和 guardrails；**记录了 trace 不代表已经评价了质量。**[O5][O6]

LangChain 的 Agent Evals 提供严格顺序、无序匹配、子集 / 超集约束以及 LLM-as-judge 等轨迹评估方式，应按任务是否允许多条有效路径选择。[L9]

#### 建立有比较价值的实验

以下是建议流程：

1. 收集有代表性的真实任务，覆盖简单、复杂、多领域、信息不足和错误输入。
2. 建立经过合理优化的 Single Agent 基线，以及适用时的固定 Workflow 基线。
3. 比较 Multi-Agent 时，使用相同任务、数据访问范围和判分标准，并记录模型、提示、工具和版本。
4. 分别回答“同等预算下谁更好”以及“达到同等质量谁更便宜”，不要把更大的计算预算当成架构本身的贡献。
5. 对随机性明显的任务重复运行，报告分布与失败率；小样本适合早期排错，不足以证明微小提升。
6. 做消融实验：移除某个专家、关闭并行、替换 Supervisor 为规则路由，定位真正带来收益的部分。
7. 注入工具超时、部分结果缺失、冲突结论、历史过长和重复消息，观察恢复与退出行为。
8. 用人工抽样校准 LLM judge；将线上失败回收为离线回归案例。[L11][A1]

总费用应包含失败与重试。一个有用指标是“包含失败开销的总费用 / 成功任务数”，而不只是某次成功回答消耗了多少 token。

**运行时增加一个 Reviewer Agent，不等于完成了 Evaluation。** Reviewer 是系统中的一个组件，也可能漏判或与作者共享偏差；evaluation 是用外部标准和样本验证系统表现的活动。

记录与评估还应遵守数据最小化。OpenAI SDK 当前默认开启 tracing，敏感输入输出的记录有单独开关；不能未经考虑就把完整对话、工具参数或业务秘密导出到观测平台。[O5]

---

## 2. 十一个核心问题的直接回答

### 2.1 什么情况下一个 Agent 不够？

**当任务需要的有效上下文、独立探索能力或能力边界，已经超出一个执行单元能够可靠管理的范围。**

可观察信号包括：工具选择持续出错、领域规则互相干扰、长历史挤掉关键事实、多条独立调查路径串行造成明显延迟，以及不同能力需要独立维护。高价值、广度优先、可分解的研究尤其适合尝试 Multi-Agent。[L1][A1]

先排除更基础的问题：工具说明差、检索质量差、输入不足、没有结构化输出或停止条件。多 Agent 不能修复错误数据，也不能凭空创造缺失权限。工具数量本身没有一个通用的“超过多少就必须拆分”的阈值。

### 2.2 Multi-Agent 与 Multi-Tool 有什么区别？

**Multi-Tool 扩展一个执行者的能力；Multi-Agent 增加可以被独立委派和管理的执行者。**

单 Agent 调用搜索、数据库和计算器，仍然是单 Agent。主 Agent 调用一个会自主搜索、验证和整理结果的研究 Agent，则是 Agents as Tools 形式的多 Agent。区别不在函数名称，而在工具内部是否包含独立的 Agent 执行过程及其任务边界。

### 2.3 Multi-Agent 与 Multi-Step Workflow 有什么区别？

**Workflow 主要定义流程；Multi-Agent 主要定义分工。两者不是互斥选项。**

“提取信息、套用规则、生成报告”可以是三个普通节点，不需要三个 Agent。固定 workflow 也可以在某一步调用一个自主研究 Agent，或者编排多个 Agent。动态分支也不自动意味着 Multi-Agent：单 Agent 的工具循环同样是多步骤、动态的。[O1][L5][L6]

路径稳定、要求确定性和审计时，优先代码编排；下一步必须根据新发现重新判断时，才引入适度自主性。

### 2.4 Specialist Agent 解决什么问题？

**解决领域指令、工具和工作上下文过于混杂的问题，同时为任务提供可独立维护、验证的责任边界。**

它靠聚焦输入、合适工具和清晰完成标准发挥作用，不靠“专家”称号获得额外知识。若只需加载一段专业知识，Skills 或动态提示可能更轻量；若需要独立完成长任务并回传结果，Specialist Agent 更有意义。

### 2.5 Handoff 解决什么问题？

**解决“接下来应该由谁继续处理”的问题。**

适合客服转接、领域切换和状态驱动的阶段转换。交接必须保留继续处理所需的信息，并明确活动处理者。它不是简单地复制消息，也不是让专家完成子任务后默认返回入口 Agent。[O2][L3]

### 2.6 Supervisor 解决什么问题？

**解决“谁拆任务、谁决定下一步、谁检查覆盖范围、谁对最终输出负责”的问题。**

它适合多个专家需要多轮协作且输出必须整合为一个答案的场景。若只是单次分类，Router 更合适；若顺序固定，代码 workflow 通常更可控。Supervisor 自身也需要预算、停止条件和评估。

### 2.7 Agents as Tools 与 Handoff 有什么区别？

| 维度 | Agents as Tools | Handoff |
|---|---|---|
| 核心含义 | 请你完成这部分，然后把结果交给我 | 接下来由你负责处理 |
| 顶层责任 | 调用方保持协调与最终回答责任 | 活动处理者切换到接收方 |
| 典型返回路径 | 子任务结果返回调用方，由其继续 | 不隐含自动返回；回转需要再次交接或另行编排 |
| 用户交互 | 通常由主 Agent 统一负责 | 接收方通常直接继续处理用户请求 |
| OpenAI 默认历史行为 | 不自动继承父 run 会话状态 | 接收已有会话历史，可过滤 |
| 典型用途 | 并行调查、专业分析、整合多份结果 | 分诊、客服转接、对话责任切换 |

两者都可能以 tool call 形式呈现给模型，**真正区别是调用后的控制流，而不是有没有调用工具。** 也可以组合：先 handoff 给领域负责人，再由该负责人把其他 Agent 当工具调用。[O1][O2][O3]

### 2.8 为什么多个 Agent 不一定比一个 Agent 好？

**能力收益可能被协调损耗抵消，而且错误会沿着协作链传播。**

拆分可能错误；子任务可能重复；交接可能漏掉关键约束；汇总可能把有条件结论变成确定断言；多个相似 Agent 可能一致地犯错。高度耦合的任务还需要反复同步，失去并行优势。[A1]

更多 Agent 通常也意味着更多推理机会与 token 预算。只有在控制或明确报告预算、模型和工具差异后，才能判断收益是否真正来自架构，而非单纯“多花计算量”。

### 2.9 Multi-Agent 的主要成本是什么？

| 成本 | 来源 | 需要观察什么 |
|---|---|---|
| 模型与工具费用 | 分解、工作、交接、总结、复核、重试 | 全部执行者的输入输出 token、工具账单 |
| 延迟 | 额外路由、串行依赖、最慢 worker、汇总 | 端到端关键路径，而不只是单次模型响应 |
| 上下文成本 | 重复发送背景、保存历史、压缩与补充 | 输入规模、重复信息、摘要造成的缺失 |
| 工程成本 | 状态、并发、恢复、版本兼容、产物管理 | 实现与维护复杂度 |
| 评估与排障 | 随机路径、组件交互、难以归因的失败 | traces、回归集、人工评估投入 |
| 治理成本 | 权限、数据传递、审批与审计 | 跨执行者的数据暴露和副作用边界 |

可把总费用理解为所有实际模型调用费用、工具费用和运行基础设施费用之和，必须包含失败和重试。**并发不是费用折扣。** 另一方面，精心隔离上下文有时能减少重复处理长历史的 token；“多 Agent 总是更贵”也不应当成定律，需要实测。[L1][A1]

### 2.10 Multi-Agent 最容易出现什么问题？

**最核心的是任务边界、信息边界和控制边界不清。** 常见表现及对应工程对策如下：

| 问题 | 表现 | 建议对策 |
|---|---|---|
| 分解失误 | 重复调查、遗漏要求、把强依赖任务并行化 | 明确范围、非目标、依赖和覆盖清单 |
| 路由与交接错误 | 交给错误专家，来回转接，重复询问信息 | 交接条件、必要上下文、最大跳数与退出规则 |
| 上下文污染 / 缺失 | 全历史塞满上下文，或摘要丢掉关键限制 | 选择性输入，保留来源和未决事项 |
| 汇总失真 | 主 Agent 改写后失去证据或夸大确定性 | 可检索产物、事实与来源绑定、引用复核 |
| 状态冲突 | 并发覆盖、重复写入、使用旧结果 | 字段所有权、合并规则、版本与幂等设计 |
| 成本或循环失控 | 简单问题过度拆分，无休止复核 | 总预算、子任务预算、轮数、超时和取消 |
| 部分失败被掩盖 | 专家失败却输出看似完整的最终答案 | 显式报告缺失、区分失败类型、限定结论 |
| 相关性错误 | 多个 Agent 对同一错误达成一致 | 独立证据、外部核验、人工抽查 |
| 权限与信任混淆 | 把角色、内部消息或工具可见性当成授权 | 工具侧授权、最小权限、数据隔离和审批 |
| 无法复现与归因 | 最终失败但不知道从哪一步开始偏离 | 关联完整运行链、记录版本、组件与端到端评估 |

不是所有限制都能只靠提示执行。预算、授权、幂等和关键退出条件应有确定性的运行时约束。OpenAI 还明确说明：handoff 链中的输入 guardrail 只适用于首个 Agent，输出 guardrail 只适用于产出最终结果的 Agent，不能假设每次交接都会自动重新执行全部检查。[O2][O4]

### 2.11 什么情况下应该退回 Single Agent？

**当优化过的单 Agent 已满足要求，而多 Agent 无法提供稳定、值得付费的额外收益时。**

典型信号包括：子任务太小；任务强耦合且必须共享几乎全部历史；大部分时间用于互相解释；专家只是换提示重复处理同一资料；多 Agent 的费用、尾延迟、失败率或维护成本超出可接受范围。

退回并不意味着丢弃所有能力，可以保留工具、检索、Skills、按需上下文和确定性流程，只移除不必要的自主协调层。

决策应以业务约束为准：预先定义最低质量、费用上限、延迟目标和不可违反的规则；在相同任务集上比较。若 Multi-Agent 没有稳定改善这些目标，就保留更简单的方案。复杂任务可按条件启用多 Agent，简单任务继续走单 Agent 路径，不必全局二选一。[L1][A2]

---

## 3. Anthropic Multi-Agent Research System：案例能证明什么？

### 3.1 架构与收益来源

文章描述的 Research 系统采用 Orchestrator-Workers：lead agent 制定研究策略，派发多个并行 subagents；各子 Agent 在独立上下文内自主搜索，再把压缩后的结果交给 lead 汇总。[A1]

```text
User query
    |
Lead agent: strategy, decomposition, delegation
    |-> Subagent A: search loop + own context --|
    |-> Subagent B: search loop + own context --|-> Results -> Lead synthesis
    |-> Subagent C: search loop + own context --|
```

这不是“多个角色开会”的主要案例，而是**明确分工、独立搜索、上下文压缩和并行计算**的案例。它尤其适合广度优先问题，即需要同时探索多条相对独立方向的任务。

### 3.2 性能数字必须连同基线一起理解

| 原文报告 | 正确解释 | 不能推出 |
|---|---|---|
| 内部 research eval 上提升 90.2% | Opus 4 lead + Sonnet 4 subagents，对比单 Agent Opus 4 | 不是成功率 90.2%，不是提升 90.2 个百分点，也不是所有任务的通用收益 |
| agents 约用 4 倍 token | 基线是普通 chat interactions，来自其自身数据 | 不是所有单 Agent 请求固定贵 4 倍 |
| multi-agent 约用 15 倍 token | 基线同样是 chats | 不是相对 Single Agent 贵 15 倍，也不是固定美元成本倍数 |
| 复杂查询研究耗时最多减少 90% | 来自其 Agent 级与工具级并行改造 | 不是所有多 Agent 系统都能实现的延迟保证 |

原文没有在这些段落中提供完整内部评测集、绝对分数和等预算消融，因此不能把 90.2% 直接当作公平等成本的架构优越性证明。该数字也不是 BrowseComp 成绩；文章对 BrowseComp 中 token、工具调用和模型选择的分析是另一组结论。[A1]

### 3.3 更值得复用的工程经验

原文记录了简单问题生成 50 个子 Agent、无休止搜索不存在的来源、任务说明模糊导致重复调查，以及过多互相更新造成干扰。这些失败强调：委派必须包含目标、输出形式、工具 / 来源指导和清楚的边界，投入应与任务复杂度匹配。[A1]

文章所述版本按批次等待子 Agent 完成，造成慢 worker 阻塞和难以实时协调的问题。异步可以缓解等待，却会增加结果协调、状态一致性和错误传播的复杂度。这是文章中当时的实现描述，不是所有 Supervisor 必然同步，也不是对当前产品实现的断言。

另外两个重要经验是：对长任务保留恢复点，避免失败后全部重跑；让子 Agent 把大型成果保存为外部产物并返回引用，减少反复转述和信息损失。

**可迁移的是设计原则，不是照搬模型组合、Agent 数量或工具调用配额。**

---

## 4. 如何选择架构，而不是先选择框架？

| 实际需要 | 优先考虑 | 何时再增加复杂度 |
|---|---|---|
| 回答问题、查询资料、使用少量清晰工具 | Single Agent + Tools / Retrieval | 已有评估证明能力或上下文边界成为瓶颈 |
| 大量专业知识，但只有一个主要处理流程 | Single Agent + Skills / 动态上下文 | 专业任务需要独立长过程和结果契约 |
| 固定步骤、业务规则和审批关卡 | Multi-Step Workflow | 某些步骤需要自主探索时嵌入 Agent |
| 单次识别领域并派发 | Router | 后续需要多轮协调或持续活动角色 |
| 专家应直接接手用户问题 | Handoff | 不应仅为咨询一次就转移对话责任 |
| 多个专家的结果必须合成一个答案 | Supervisor + Agents as Tools | 中间发现确实需要持续重新决策 |
| 输入决定子任务内容或数量 | Orchestrator-Workers | 动态拆分的收益足以覆盖规划成本 |
| 多项低依赖调查需要同时完成 | Parallel Workers / Agents | Worker 需要自主工具循环时再使用完整 Agent |

这不是必须逐级升级的路线图。选择能满足质量、费用、延迟和治理要求的最简单组合即可。

学习时可以用同一个“比较三种数据库用于订单系统”的问题做纸面分析：一个 Agent 调用三类工具、固定流程分别收集资料、Supervisor 委派三个独立调查者，分别增加了什么、损失了什么。如果无法明确说明某个 Agent 的输入、输出、非目标、状态所有权和停止条件，暂时不要创建它。

**最终判断标准不是系统里有几个 Agent，而是这套分工是否以可接受的成本，可靠地解决了单 Agent 难以解决的问题。**

---

## 5. 官方参考资料与阅读定位

以下均为本次研究使用的官方来源。链接指向在线文档，后续内容可能更新。当前 LangGraph 的高层多 Agent 模式主要在 LangChain 文档介绍；LangGraph 文档负责图、状态、动态派发和执行机制，两者应结合阅读。

### OpenAI Agents SDK / Platform

- [O1：Agent orchestration][O1]：LLM / 代码编排，Agents as Tools 与 Handoffs 的核心选择。
- [O2：Handoffs][O2]：控制权、默认历史、输入过滤、交接参数和 guardrail 范围。
- [O3：Tools — Agents as tools][O3]：工具封装 Agent、嵌套运行输入与会话状态。
- [O4：Context management][O4]：本地应用 context 与 LLM 可见上下文的区别。
- [O5：Tracing][O5]：端到端 traces、spans 和敏感数据记录。
- [O6：Evaluate agent workflows][O6]：trace grading、数据集及可重复评估。

### LangChain / LangGraph / LangSmith

- [L1：Multi-agent][L1]：当前模式总览、使用动机及上下文工程。
- [L2：Subagents][L2]：Supervisor、工具封装、输入输出、上下文和状态策略。
- [L3：Handoffs][L3]：状态驱动切换、单 Agent 与多子图实现、消息交接。
- [L4：Router][L4]：路由、并行派发及与 Supervisor 的区别。
- [L5：Custom workflow][L5]：确定性逻辑与 Agent 节点组合。
- [L6：Workflows and agents][L6]：流程与 Agent 区别、并行、Orchestrator-Workers、`Send`。
- [L7：Skills][L7]：单 Agent 的按需专业化，避免不必要的子 Agent。
- [L8：Graph API overview][L8]：State、Nodes、Edges 和 reducers。
- [L9：Agent Evals][L9]：轨迹匹配与 LLM-as-judge。
- [L10：Migrate from langgraph-supervisor][L10]：当前推荐的 Supervisor 实现入口。
- [L11：Evaluation concepts][L11]：组件 / 系统、离线 / 在线评估及持续改进。

### Anthropic Engineering

- [A1：How we built our multi-agent research system][A1]：架构、内部评测、token 成本、委派失败、可靠性和产物传递。
- [A2：Building effective agents][A2]：简单优先、Workflow / Agent 区别、并行和 Orchestrator-Workers。该文属于较早的架构指导，本文引用其原则，不把其工具生态当作当前产品推荐。

[O1]: https://openai.github.io/openai-agents-python/multi_agent/
[O2]: https://openai.github.io/openai-agents-python/handoffs/
[O3]: https://openai.github.io/openai-agents-python/tools/#agents-as-tools
[O4]: https://openai.github.io/openai-agents-python/context/
[O5]: https://openai.github.io/openai-agents-python/tracing/
[O6]: https://developers.openai.com/api/docs/guides/agent-evals
[L1]: https://docs.langchain.com/oss/python/langchain/multi-agent
[L2]: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents
[L3]: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
[L4]: https://docs.langchain.com/oss/python/langchain/multi-agent/router
[L5]: https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow
[L6]: https://docs.langchain.com/oss/python/langgraph/workflows-agents
[L7]: https://docs.langchain.com/oss/python/langchain/multi-agent/skills
[L8]: https://docs.langchain.com/oss/python/langgraph/graph-api
[L9]: https://docs.langchain.com/oss/python/langchain/test/evals
[L10]: https://docs.langchain.com/oss/python/migrate/langgraph-supervisor
[L11]: https://docs.langchain.com/langsmith/evaluation-concepts
[A1]: https://www.anthropic.com/engineering/multi-agent-research-system
[A2]: https://www.anthropic.com/engineering/building-effective-agents
