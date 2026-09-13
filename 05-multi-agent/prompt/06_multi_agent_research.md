现在实现本阶段最终项目：

Multi-Agent Research System。

使用：

LangGraph

目标：

用户输入任意复杂 Research Question。

例如：

“2026 年 AI Agent 开发生态有哪些值得 Solo Developer 关注的机会？”

系统自动：

1. 分析问题。
2. 动态拆分 Research Tasks。
3. 创建多个 Specialist Research Agents。
4. 并行执行 Research。
5. 汇总结果。
6. 进行 Cross-Agent Analysis。
7. Reviewer 检查结果。
8. 如果质量不够，重新研究。
9. 最终生成报告。

Architecture：

```text
User
↓
Supervisor
↓
Planner
↓
Dynamic Research Tasks
↓
Parallel Specialist Agents
↓
Synthesizer
↓
Reviewer
↓
Conditional
├── FAIL → Research
└── PASS → Final Answer
```

要求：

### Supervisor

负责整体任务协调。

### Planner

动态生成 Research Tasks。

### Research Workers

每个 Worker 负责一个独立研究主题。

### Synthesizer

整合所有 Research Results。

### Reviewer

检查：

* completeness
* factual consistency
* evidence quality
* logical consistency
* missing important aspects

### Loop

如果 Review 不通过：

Reviewer
→ Planner / Research
→ Synthesizer
→ Reviewer

最多 3 次。

### LangGraph

必须使用：

StateGraph
State
Nodes
Edges
Conditional Edges
Parallel Execution
Persistence

### Reliability

必须：

* 最大 Worker 数量
* 最大 Review Iterations
* Worker timeout
* Worker failure handling
* structured state
* structured worker result

### 不要加入

MCP
Memory/RAG
外部数据库
Human-in-the-loop

这些属于后续学习阶段。

创建：

src/multi_agent_research/

├── state.py
├── planner.py
├── research_worker.py
├── synthesizer.py
├── reviewer.py
├── graph.py
└── main.py

tests/multi_agent_research/

test_planner.py
test_workers.py
test_reviewer.py
test_graph.py

docs/

06-MULTI-AGENT-RESEARCH.md

同时生成：

Mermaid Architecture Diagram

以及一次完整运行的 Trace / Execution Log。

最后回答：

1. 为什么这个系统需要 Multi-Agent？
2. 如果改成 Single Agent，会发生什么？
3. 哪些 Agent 可以合并？
4. 哪些 Agent 必须保持独立？
5. 哪些任务适合并行？
6. 哪些任务存在依赖不能并行？
7. Supervisor 和 Worker 如何通信？
8. Shared State 如何设计？
9. 如何控制成本？
10. 如何控制 latency？
11. 如何防止 Agent 无限循环？
12. 如何评估整个 Multi-Agent 系统？
