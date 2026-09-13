现在实现 Production Agent 实验 ：

Durable Execution。

使用 LangGraph。

要求：

1. 使用 production-capable checkpointer。
2. 不再使用 InMemorySaver 作为生产方案。
3. 使用 thread_id 管理独立 Agent executions。
4. 每个重要 graph step 都可以恢复。
5. 模拟 Worker crash。
6. 验证 Agent 可以从 checkpoint 恢复。
7. 不重复执行已经成功且具有副作用的操作。
8. 支持 interrupted execution。
9. 支持 resume。
10. 支持查看 execution history。

实现：

Crash
→ checkpoint
→ restart
→ resume

测试场景：

Research A 成功
Research B 成功
Research C crash

恢复后：

Research A 不重新执行
Research B 不重新执行
Research C 从合适的位置继续。

创建：

src/durable/

checkpointer.py
recovery.py

tests/test_recovery.py

docs/03-DURABLE-EXECUTION.md

重点解释：

Checkpoint
vs
Memory
vs
Database
vs
Execution History
