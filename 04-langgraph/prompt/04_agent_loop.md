现在实现 LangGraph 的 Conditional Edge。

目标：

把手写 Agent Loop：

while True:
response = LLM(...)

```
if tool_call:
    execute_tool()
    continue

return final_answer
```

转换成 LangGraph。

Graph：

```text
START
↓
agent
↓
should_continue
├── "tools" → tools
│                 │
│                 ▼
│               agent
│
└── "end" → END
```

要求：

1. 使用 StateGraph。
2. agent 是 Node。
3. tools 是 Node。
4. should_continue 是 routing function。
5. 使用 add_conditional_edges。
6. Tool Result 必须回到 agent。
7. Agent 最终输出时进入 END。
8. 设置最大循环次数。
9. Tool 错误必须能够安全处理。
10. 不使用 OpenAI Agents SDK。
11. 不使用 LangChain Agent abstraction。

创建：

src/04_agent_loop.py

tests/test_agent_loop.py

docs/04-AGENT-LOOP.md

重点解释：

1. LangGraph 如何表示 Agent Loop？
2. Conditional Edge 与普通 if/else 有什么区别？
3. State 在循环过程中如何变化？
4. 为什么 Graph 比 while loop 更容易观察？
