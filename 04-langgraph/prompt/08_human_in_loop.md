现在给 LangGraph Research Agent 增加 Human-in-the-loop。

模拟一个高风险 Tool：

publish_report(report)

要求：

Agent 生成最终 Research Report 后：

1. 不允许直接 publish。
2. Graph 必须暂停。
3. 使用 LangGraph interrupt。
4. 保存当前 State。
5. Human 可以：

   * approve
   * reject
   * request_changes
6. approve → publish
7. reject → END
8. request_changes → 返回 Generator

必须支持：

Graph interrupt
+
Persistence
+
Resume

创建：

src/08_human_in_loop.py

tests/test_human_in_loop.py

docs/08-HUMAN-IN-THE-LOOP.md

重点解释：

1. interrupt 为什么需要 persistence？
2. Graph 暂停后 State 在哪里？
3. resume 是怎么工作的？
4. 为什么传统 while loop 实现这种能力比较麻烦？
5. LangGraph 在 Agent Runtime 中解决了什么问题？
