# 02 — Routing：Router → 恰好一个 Specialist

对应代码：`src/routing.py`、`tests/test_routing.py`。
运行方式：`.\.venv\Scripts\python.exe -m src.routing "你的问题"`

已用三个要求的测试问题实际运行过（见本文末尾"实际运行观察"）。

## 1. 这是什么

```text
用户问题
  -> Router          （output_type=RouterDecision，只分类，不回答，不调用工具）
  -> 固定字典查表      （decision.category -> 对应的 Specialist builder）
  -> 恰好一个 Specialist（tools=[search_web]，纯文本输出）
最终答案
```

`build_router()` 创建一个只负责分类的 `Agent`；`run_routing()` 先调用 Router 拿到 `RouterDecision.category`，再用 Python 的 `_SPECIALIST_BUILDERS[category]` 字典查表，构建并调用**恰好一个** Specialist。另外两个 Specialist 连实例都不会被创建。

## 2. Router 与三个 Specialist 如何分工

| Agent | 职责 | 工具 | 输出类型 |
|---|---|---|---|
| Router | 只分类，判断 technical / business / general | 无 | `RouterDecision`（结构化） |
| Technical Researcher | 技术文档、GitHub、架构、实现细节、API/SDK 对比 | `search_web` | 纯文本 |
| Business Researcher | 市场、成本、竞争、商业模式、机会 | `search_web` | 纯文本 |
| General Researcher | 普通知识、概念解释 | `search_web`（可选使用） | 纯文本 |

三个 Specialist 的 `instructions`（`TECHNICAL_RESEARCHER_INSTRUCTIONS`/`BUSINESS_RESEARCHER_INSTRUCTIONS`/`GENERAL_RESEARCHER_INSTRUCTIONS`）内容完全不同，且各自明确写了"不要做什么"（例如 Technical 的 instructions 写"不要关注市场规模或定价"，Business 的 instructions 写"不要关注内部架构和实现细节"），以避免同一问题被泛泛地、不专业地处理。

## 3. 为什么这是 Routing，而不是别的模式

- **分类由一个独立步骤完成**：Router 是唯一负责分类的 Agent，`ROUTER_INSTRUCTIONS` 明确写"Do NOT answer the question yourself"，避免 Router 越权替 Specialist 干活。
- **路由表是程序代码，不是模型决定的**：`_SPECIALIST_BUILDERS` 是一个写死在 `routing.py` 里的 `dict[Category, Callable]`。Router 的输出只提供一个 key（`"technical"`/`"business"`/`"general"`），实际"选哪个函数来构建 Specialist"这件事由 Python 的字典查表完成，不是模型在运行时决定调用哪个 Agent。
- **只有三条预先定义好的路径**：不存在"Router 决定这次需要几个 Specialist"或"临时发明一个新类别"的情况——`Category` 是一个只有三个取值的 `Literal`，SDK 的结构化输出校验会拒绝任何不在这三个值里的分类结果。
- **恰好一个 Specialist 运行**（需求 7）：`run_routing()` 里只有一次 `Runner.run_sync(specialist, ...)` 调用，且 `specialist` 只可能是三者之一。测试 `test_technical_question_routes_to_technical_researcher_only` 等直接验证了这一点——脚本化的模型只准备了 Router + 一个 Specialist 的两轮对话，`model.assert_complete()` 确认没有多余的调用被脚本化，也没有调用被落下。
- **没有 Handoff**：所有四个 Agent 的 `agent.handoffs` 都是空列表（`test_no_agent_declares_handoffs`）。Router 不会把控制权"交给" Specialist；是 `run_routing()` 里的 Python 代码在拿到 Router 的分类结果之后，主动发起第二次、独立的 `Runner.run_sync` 调用。
- **没有 Orchestrator-Workers**：没有一个中央 LLM 把问题动态拆解成不确定数量的子任务分给多个 worker；只有一次分类决策，选择一条预先定义好的固定路径。
- **没有 Parallelization**：不会同时跑多个 Specialist 再合并结果；只跑一个。
- **没有 Evaluator-Optimizer**：Specialist 的第一次回答就是最终答案，没有第二个 LLM 去评价、要求它修改重写的循环。

## 4. 为什么 Routing 比单个万能 Agent 更好

- **提示词更专注、更不容易被"平均化"**：如果用同一个 Agent 处理"LangGraph 和 OpenAI Agents SDK 有什么区别"和"AI Agent SaaS 市场有哪些机会"，它的 instructions 要同时兼顾"关注架构细节"和"关注市场数据"，两种优化目标会互相稀释——原文明确指出"针对一种输入优化，可能损害另一种输入的表现"。拆开后 Technical Researcher 可以被写得非常"钻技术细节"，Business Researcher 可以被写得非常"重视市场数字和竞争格局"，互不干扰。
- **可以按需分配不同的模型或成本策略**（本实现未做，但架构上天然支持）：原文举的例子是"简单问题路由到便宜模型，复杂问题路由到强模型"；这里三个 Specialist 目前用同一个 `model` 参数，但因为它们是完全独立的 `Agent`，以后给 Technical Researcher 换一个更强的模型、给 General Researcher 换一个更便宜的模型，都不需要改动 Router 或彼此的 instructions。
- **可以单独调优、单独测试每个 Specialist**：`test_specialists_have_distinct_names_and_instructions` 这类测试可以针对某个 Specialist 的行为单独验证，不必每次都连带测试另外两个完全不相关领域的 prompt 改动有没有把它们带崩。
- **权限和范围可以按路径收紧**：如果以后 Business Researcher 需要接入付费的市场数据 API，而 Technical Researcher 不需要，这种"按路径分配工具/权限"的扩展在 Routing 结构下是自然的——每个 Specialist 已经是独立的 `Agent`，加工具只影响它自己。

## 5. 什么时候 Routing 反而会增加不必要的复杂度

- **类别边界模糊、大量问题横跨多个类别时**：如果实际提出的问题经常同时涉及技术和商业（比如"这个开源 Agent 框架的架构能不能支撑它的商业化定价模式？"），Router 被迫"选择单一最佳类别"，要么丢掉技术侧信息，要么丢掉商业侧信息；这时候可能一个能同时处理两个维度的 Agent，反而比强行拆开、只能覆盖一半问题的路由更合适。
- **各类别的处理方式其实差别不大时**：如果 Technical/Business/General 三个 Specialist 最终的 instructions 高度相似（比如都只是"搜索 + 总结"，没有真正差异化的关注点），那么多出来的 Router 分类步骤只是增加了一次额外的 LLM 调用和延迟，却没有带来专业化的实际收益。
- **问题量很小、且已知都属于同一类别时**：比如某个产品场景下 99% 的问题天然只会是技术问题，此时为了应对理论上存在的另外两类问题而维护一个三路由由系统，维护成本（三份 instructions、三份测试、路由表）可能超过收益，不如直接用一个 Technical Researcher 作为唯一 Agent。
- **分类本身容易出错、而错分类的代价很高时**：Router 是基于 LLM 的分类，存在误分类风险（例如把一个既懂技术又懂商业的边界问题分错类）。如果错分类会导致用户得到完全跑偏的答案（比如商业问题被当成技术问题处理，答非所问），而且没有办法在 Specialist 层面"发现自己分错了、退回去重新分类"，那么在准确率没有被充分验证之前，贸然上线一个不可逆的单路由系统是有风险的——原文本身也强调"分类要能被可靠地执行"才是 Routing 适用的前提。
- **需要多个角度共同参与判断时**：如果任务的本质是"我需要技术和商业两种视角都出来、再做综合判断"，这属于 Parallelization（多角度 sectioning）或 Orchestrator-Workers 的场景，而不是"选其中一个视角、放弃另一个"的 Routing——勉强用 Routing 会导致用户明明需要两种分析，却只拿到了一种。

## 6. 实际运行观察

用真实网关跑了三个要求的测试问题，Router 的分类结果和路由都符合预期：

| 问题 | Router 分类 | Router 给出的理由（节选） | 实际运行的 Specialist |
|---|---|---|---|
| "LangGraph 和 OpenAI Agents SDK 有什么区别？" | `technical` | "这是关于两个 AI 框架/SDK 的技术选型与功能差异比较。" | Technical Researcher（回答包含 LangGraph 的图/状态机模型 vs Agents SDK 的 Runner/Agent Loop/Handoff 模型的架构对比，并附带官方文档和 GitHub 链接） |
| "AI Agent SaaS 市场有哪些机会？" | `business` | "问题关注 AI Agent SaaS 市场中的商业机会，属于市场和商业策略分析。" | Business Researcher |
| "什么是 RAG？" | `general` | "这是一个关于 RAG 概念的普遍性知识问题，不涉及具体技术实现或商业策略。" | General Researcher |

每次运行都只触发了一次 Router 调用 + 一次对应 Specialist 调用，控制台输出里也只出现了被选中的那一个 Specialist 的回答，验证了需求 7（"不要让所有 Agent 都处理同一个问题"）在真实运行中确实成立，而不仅仅是脚本化测试里的断言。
