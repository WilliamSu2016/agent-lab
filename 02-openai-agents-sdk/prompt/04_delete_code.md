现在进行一次架构重构。

目标：

让 OpenAI Agents SDK 完全负责 Agent Runtime。

请检查第⑤步的代码，并删除所有已经由 SDK 接管的 Runtime 逻辑。

重点检查：

1. 手写 LLM loop
2. 手写 tool dispatch
3. 手写 tool schema
4. 手写 tool-call detection
5. 手写 tool-result routing
6. 手写 continuation logic
7. 手写 Agent termination logic

保留：

1. Agent instructions
2. Business Tools
3. Domain logic
4. Application entry point
5. Evaluation tests

原则：

如果 SDK 已经负责，就不要在项目中重复实现。

不要为了“看起来完整”而保留重复 Runtime。

完成后：

1. 运行全部测试
2. 对比第⑤步代码
3. 统计删除了多少 Runtime code
4. 生成：

docs/SDK-RUNTIME-COMPARISON.md

其中明确说明：

第⑤步：
我们自己负责什么？

第⑥步：
OpenAI Agents SDK 负责什么？

不要增加新功能。
