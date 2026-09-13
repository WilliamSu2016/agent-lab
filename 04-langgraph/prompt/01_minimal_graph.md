现在开始实现第一个 LangGraph。

不要使用 LLM。

不要使用 Agent。

不要使用 Tool。

只使用 LangGraph。

实现：

START
↓
greet
↓
format
↓
END

State：

{
"name": str,
"message": str
}

greet：

根据 name 生成：

"Hello, {name}!"

format：

把 message 转换为最终输出格式。

要求：

1. 使用 StateGraph。
2. 定义明确的 State。
3. 创建两个 Node。
4. 使用 START。
5. 使用 END。
6. 使用 compile()。
7. 使用 invoke()。
8. 打印最终 State。

创建：

src/01_minimal_graph.py

以及：

tests/test_minimal_graph.py

创建：

docs/01-MINIMAL-GRAPH.md

重点解释：

Node
State
Edge
START
END
compile
invoke

不要增加任何 Agent 功能。
