现在为 Research Agent 增加 LangGraph Persistence。

目标：

Agent 执行过程中可以保存 State。

要求：

1. 使用 LangGraph 当前推荐的 checkpointer 机制。
2. 先使用内存实现。
3. 不使用 Redis。
4. 不使用 PostgreSQL。
5. 不使用外部数据库。

创建两个不同的 invocation：

Thread A：
用户开始研究问题。

Thread B：
另一个独立研究任务。

验证：

1. Thread A 和 Thread B 的 State 相互隔离。
2. Agent 可以根据之前保存的 State 继续运行。
3. 能够读取之前的 State。
4. 能够从中断位置继续。

创建：

src/07_persistence.py

tests/test_persistence.py

docs/07-PERSISTENCE.md

重点解释：

1. State 和 Checkpoint 的区别。
2. Thread ID 的作用。
3. 为什么 Agent 需要 Persistence？
4. Persistence 与 Memory 有什么区别？
