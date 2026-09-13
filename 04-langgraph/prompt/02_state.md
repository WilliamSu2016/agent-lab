现在进行 LangGraph 第二个实验：

深入理解 State。

把之前的 Minimal Graph 改造成：

ResearchState：

{
question: str,
research_notes: list[str],
analysis: str,
final_answer: str
}

创建三个 Node：

research
analysis
finalize

要求：

1. 每个 Node 只负责修改自己需要负责的 State。
2. 不使用全局变量。
3. 不使用数据库。
4. 不使用 LLM。
5. 不使用 Tool。
6. 展示每一个 Node 执行前后的 State。
7. 添加测试验证 State 在 Node 之间正确传递。

创建：

src/02_state.py

tests/test_state.py

docs/02-STATE.md

重点回答：

1. State 为什么是 LangGraph 的核心？
2. Node 为什么不应该直接依赖全局变量？
3. State 与普通 Python function 参数有什么区别？
4. 如果两个 Node 同时修改同一个 State field，会发生什么？
5. Reducer 是解决什么问题的？
