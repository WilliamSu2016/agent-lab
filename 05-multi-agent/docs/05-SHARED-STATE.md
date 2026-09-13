# 实验 5：Shared State（LangGraph）

## 0. 架构
             Supervisor
             /        \
            ▼          ▼
        Agent A      Agent B
            │          │
            └────┬─────┘
                 ▼
             Shared State
                 │
                 ▼
              Agent C

## 目标

流程：

```
START
  │
  ▼
ResearchAgent      (research 节点：只写 research_results)
  │
  ▼
AnalysisAgent      (analysis_agent 节点：只写 analysis、loop_count)
  │
  ▼
ReviewAgent        (review_agent 节点：只写 review；可以读 analysis 并挑错)
  │
  ▼ (条件边)
  ├── review.approved == False 且 loop_count < max_loops ──▶ 回到 AnalysisAgent
  │
  ▼ (否则)
Finalizer          (finalizer 节点：只写 final_answer)
  │
  ▼
END
```

代码：`src/05_shared_state.py`
测试：`tests/test_shared_state.py`（25 个，全部离线，使用可注入的假 `TextLLMCall`，不需要网络/API Key）

## `ResearchState` 定义

```python
class ReviewResult(TypedDict):
    approved: bool
    feedback: str

class ResearchState(TypedDict):
    question: str
    research_results: str
    analysis: str
    review: ReviewResult
    final_answer: str
    loop_count: int
    max_loops: int
    trace: Annotated[list[str], operator.add]   # append-only 审计日志
```

`question/research_results/analysis/review/final_answer` 是用户要求的最小字段集；
额外加了 `loop_count`（AnalysisAgent 自己的重试计数器）、`max_loops`（循环上限，默认 3）、
`trace`（可追踪的状态变化审计日志，带 `operator.add` reducer）。

## 实现要点

- **三个 Agent + 一个 Finalizer，全部通过 State 通信（Requirement 1）**：每个节点函数的入参
  都只是当前的 `ResearchState`，返回值是一个**局部更新字典**（partial update），LangGraph 负责
  把它合并进共享 state——节点之间没有任何直接调用、没有共享的 Python 对象引用。
- **没有任何全局变量（Requirement 2）**：`src/05_shared_state.py` 里不存在任何
  模块级可变变量。每个节点都是 `make_xxx_node(llm_call)` 工厂返回的闭包，闭包只捕获传入的
  `llm_call`（只读的函数引用），不捕获、不修改任何跨调用共享的状态。
  `tests/test_shared_state.py::OwnershipTest::test_two_independent_states_do_not_leak_into_each_other`
  用同一个节点函数处理两个完全独立的 `ResearchState`，验证互不干扰。
- **每个 Agent 只改自己的字段（Requirement 3）**：
  - `research` 节点只返回 `{"research_results": ..., "trace": [...]}`；
  - `analysis_agent` 节点只返回 `{"analysis": ..., "loop_count": ..., "trace": [...]}`；
  - `review_agent` 节点只返回 `{"review": ..., "trace": [...]}`；
  - `finalizer` 节点只返回 `{"final_answer": ..., "trace": [...]}`。
  `OwnershipTest` 对每个节点都用 `set(update.keys())` 精确断言返回字典只含它自己该有的键，
  尤其是 `test_review_node_never_writes_analysis_even_when_rejecting`：即使 Review 判定
  "不通过"，它的返回值里也绝不会出现 `analysis` 键——它只能通过 `review.feedback` **描述**问题，
  不能直接**修改**别人的产出。
- **ReviewAgent 能发现 AnalysisAgent 的问题（Requirement 4）**：`review_agent` 节点的 prompt
  里显式传入了 `state["analysis"]`（读取权限），指令要求它检查结论是否有研究资料支撑、逻辑是否
  自洽等，输出结构化 JSON `{"approved": bool, "feedback": str}`。它可以读别人的字段，但只能写
  自己的字段——"读所有、写自己"是这个模式里权限设计的核心。
- **Review 不通过则回到 AnalysisAgent（Requirement 5）**：`route_after_review` 这个条件边函数
  （不是节点）在 `review["approved"] is False` 且循环预算未耗尽时返回 `"analysis_agent"`；
  `analysis_agent` 节点重新被调用时会读取 `state["review"]["feedback"]` 和自己上一轮的
  `state["analysis"]`，把两者都写入新一轮的 prompt，要求 LLM 针对性修正
  （`test_analysis_sees_previous_feedback_when_retrying` 验证 feedback 确实被传给了下一次调用）。
- **最大循环次数为 3（Requirement 6）**：`route_after_review` 里 `state["loop_count"] >=
  state["max_loops"]` 时强制转到 `finalizer`，不管 Review 是否通过——保证不会无限循环。
  `test_max_loops_enforced_even_if_review_never_approves_requirement_6` 用一个"永远不通过"的
  假 Review 验证：即使一直被打回，循环也在第 3 次后停止，并把"未通过"的事实写进最终结果里，
  而不是假装通过。
- **所有状态变化都可追踪（Requirement 7）**：`trace` 字段是 `Annotated[list[str],
  operator.add]`——每个节点只 `return` 一条**自己的**新日志行（例如
  `"AnalysisAgent: produced analysis (attempt 2)"`），LangGraph 用 reducer 把每个节点贡献的
  条目按执行顺序拼接起来，运行结束后从 `final_state["trace"]` 就能重建完整的执行历史，
  不需要额外的日志系统。
- **使用 LangGraph persistence（Requirement 8）**：`build_graph(..., checkpointer=...)` 把
  `checkpointer`（例如 `langgraph.checkpoint.memory.InMemorySaver()`）传给
  `builder.compile(checkpointer=checkpointer)`。只要调用 `graph.invoke(state, {"configurable":
  {"thread_id": ...}})`，运行结束后依然可以用同一个 `thread_id` 调用
  `graph.get_state(config)`（拿到最终 state）或 `graph.get_state_history(config)`
  （拿到每一个 superstep 的历史快照，不仅仅是最终 `trace` 字段）。这比只有一份
  `trace` 列表的"事后追踪"更进一步：连每一步"当时的完整 state 长什么样"都能重新查到。
- **无 MCP、无 Supervisor、无 Handoff、无 Agents-as-tools**：这里只是普通的
  `StateGraph` + 节点 + 边，Agent 之间没有互相调用、没有转移控制权，也没有把彼此包装成 Tool。

## 一个真实的踩坑：节点名不能和 State 字段名重名

最初把节点命名为 `"analysis"` / `"review"`，与 `ResearchState` 里的 `analysis` / `review`
字段同名，`StateGraph.compile()` 直接报错：
`ValueError: 'analysis' is already being used as a state key`。
LangGraph 不允许节点名和 state key 撞名（它们共用同一个 Pregel channel 命名空间）。
修复：把节点名改成 `"analysis_agent"` / `"review_agent"`，State 字段名保持不变
（`analysis`/`review`）。这也从另一个角度提醒我们：**State 字段名和 Agent 名是两个不同的概念**——
字段描述"数据"，节点名描述"谁产生这份数据"，二者不应该混用同一个命名空间。

## 问题解答

### 1. Shared State 和 Agent Memory 有什么区别？

- **Shared State**：属于**一次运行（run/graph.invoke）内部**的、多个 Agent 之间的协作数据——
  本实验里的 `ResearchState` 只在一次 `graph.invoke()` 调用期间存在并在节点间传递，
  运行结束后（如果没有 checkpointer）就随对象一起被丢弃。它的作用是让**同一次任务**里的
  多个 Agent 能看到彼此的产出，是"任务执行期间的工作台"。
- **Agent Memory**：属于**跨多次运行/跨会话**的、单个（或多个）Agent 自己积累的长期知识——
  例如"这个用户上次问过什么"、"过去处理过的类似案例"。Memory 通常要额外做检索、摘要、
  写入策略，关注的是"时间跨度"和"个体的历史"，而不是"这一次任务内部谁该看到什么"。

简单说：Shared State 是**空间维度**的协作（同一时刻，多个 Agent 共享一份数据），
Agent Memory 是**时间维度**的持久化（同一个 Agent，跨越多次运行）。
本实验用的 `checkpointer`（Requirement 8）看起来像"记忆"，但它保存的是**同一次任务的执行
轨迹**（用于恢复/审计/追踪），不是 Requirement 13 那种"跨会话的长期记忆"（本实验也明确没有
使用长期记忆——`thread_id` 只是为了让这一次运行的中间状态可查询）。

### 2. Agent 之间是否应该共享完整 State？

**不应该毫无限制地共享。** 更准确的说法是"共享同一个 State 对象，但每个 Agent 只被允许
读取它需要读的字段、只被允许写它自己负责的字段"。本实验里：

- `ResearchAgent` 只需要读 `question`；
- `AnalysisAgent` 需要读 `question`、`research_results`，以及（重试时）自己上一轮的
  `analysis` 和 `review.feedback`；
- `ReviewAgent` 需要读 `analysis`（这是它的工作对象）；
- `Finalizer` 需要读 `analysis`、`review`、`loop_count`。

技术上，因为大家共用一个 `TypedDict`，"完整 State 对每个节点都可见"是 LangGraph 的实现细节
（节点函数的入参就是整个 state）；但**设计上**应该约束每个 Agent 的 prompt/逻辑只使用它
真正需要的字段，绝不能反过来利用"能看到全部字段"去做它职责之外的事情
（例如让 ReviewAgent 顺手把 `research_results` 也改了）。可见性和写权限是两回事，
这里刻意只用"写权限"（返回哪些键）来做硬约束，因为 LangGraph 没有原生的字段级读权限控制。

### 3. 为什么应该限制 Agent 能修改的 State？

1. **职责边界清晰、可推理**：如果任何 Agent 都能改任何字段，出了问题（比如最终答案不对）
   就无法在不读遍全部代码的情况下判断"是谁改坏的"——限制写权限之后，`analysis` 出问题
   只可能是 `AnalysisAgent` 的锅，排查范围直接缩小到一个节点。
2. **并发/顺序安全**：虽然本实验是严格串行的，但如果未来要把某些 Agent 改成并行执行
   （类似实验 4），"每个 Agent 只写自己的字段"是让并发安全变得可行的前提——多个并发分支
   同时写不相交的字段（或者带 reducer 的同一个字段）不会冲突，但如果任由它们互相改写
   同一个字段就会产生竞态。
3. **可测试性**：`OwnershipTest` 这种"断言返回字典的键集合"的测试之所以能写，正是因为
   限制了写权限——这变成了一个可以自动验证的契约，而不是"大家应该自觉遵守的约定"。
4. **防止 ReviewAgent 直接"顺手把分析改好"**：这是本实验 Requirement 4/5 的核心——
   Review 必须通过"打回去、让 AnalysisAgent 自己重新做"来纠正问题，而不能自己越权改写
   `analysis`。这保留了"谁生产数据，谁对数据负责"的边界，也让 Review 的意见变成了
   AnalysisAgent 输入的一部分（可追溯的 feedback），而不是一次不留痕迹的静默修改。

### 4. State Schema 为什么重要？

State Schema（这里是 `ResearchState` 这个 `TypedDict`）相当于多个 Agent 之间的**契约**：

- **明确输入输出的形状**：每个 Agent 不需要知道其他 Agent 的实现细节，只需要知道
  "`research_results` 是一段文本"、"`review` 是一个带 `approved`/`feedback` 的对象"，
  就可以正确使用它。
- **让"写权限"可以被静态检查/测试**：本实验大量测试（`OwnershipTest`）都是基于 Schema
  里字段名做断言的——没有 Schema，"每个 Agent 只改自己的字段"就无从谈起，也无法验证。
  `Annotated[list[str], operator.add]` 这种类型注解还直接决定了**多次写入同一字段时的合并
  行为**（reducer），这是纯字典没法表达的信息。
  这次实现还踩到了一个坑：**节点名不能与 Schema 字段名相同**（见上文），这恰恰说明 Schema
  不只是"文档"，而是 LangGraph 运行时真正依赖、会做校验的一等公民。
- **变更影响范围可控**：如果要新增一个字段（比如 `confidence_score`），只需要在 Schema 里加
  一行，并让某一个 Agent 负责写它——不需要改动其他 Agent 的代码，因为它们本来就不关心这个字段。
- **反面例子**：如果 Agent 之间传递的是没有 Schema 的自由格式字符串/字典
  （比如互相塞一堆没约定好的 key），任何一个 Agent 改了字段名或者字段的含义，都会在运行时
  才暴露问题，而且很难定位是谁改的、什么时候改的。

### 5. 如何避免 Agent A 修改 Agent B 的数据？

结合本实验的具体做法：

1. **节点函数只返回"局部更新字典"，而不是修改并返回整个 state**：每个节点的返回值只包含
   它自己拥有的键（例如 `analysis_agent` 从不在返回值里出现 `research_results` 或 `review`
   这些键）。LangGraph 用这个局部字典去**合并**进全局 state，而不是用它去**替换**全局 state，
   所以即使某个节点"忘了"返回某个字段，也不会意外把它清空——但反过来说，
   这也要求每个节点**绝不能**在返回值里放不属于自己的键，这是一条需要靠约定 + 测试来保证的规则
   （本实验用 `OwnershipTest` 系列测试来自动化验证这条规则，而不是仅仅依赖代码审查）。
2. **只读，不改写别人的字段**：像 `ReviewAgent` 那样，需要评价 `analysis` 时只读取它，
   把意见写进自己的 `review` 字段，绝不直接改 `analysis` 本身。
3. **需要修正时，通过控制流回退到"数据的主人"，而不是代替它修改**：Review 不通过 →
   条件边把控制权交回 `AnalysisAgent` 节点，由 `AnalysisAgent` 自己根据 feedback 重新生产
   一份新的 `analysis`——修改权始终留在数据的所有者手里。
4. **用测试而不是靠自觉**：`OwnershipTest` 对每一个节点单独调用、断言返回字典的键集合，
   是这条规则在本项目里唯一真正"生效"的强制手段——纯靠代码风格约定是不可靠的，
   写测试才能在未来有人不小心加错字段时立刻失败。

## 小结对照表

| 维度 | Parallel Multi-Agent（实验 4） | Shared State（本实验） |
|---|---|---|
| 通信方式 | Worker 之间完全隔离，只在 fan-in 后共享合并结果 | 显式共享同一个 `ResearchState`，按字段划分读写权限 |
| 执行方式 | 并行 fan-out / fan-in | 严格串行，带一个有界的重试循环 |
| 状态修改范围 | 每个 Worker 只写自己那一条 `results` 元素 | 每个 Agent 只写自己负责的具体字段 |
| 循环/重试 | 无（每个 Worker 只跑一次） | 有：AnalysisAgent ↔ ReviewAgent，上限 `max_loops` |
| 持久化 | 无（Requirement 13 明确不使用 Memory） | 有：`InMemorySaver` checkpointer，可查完整状态历史 |
