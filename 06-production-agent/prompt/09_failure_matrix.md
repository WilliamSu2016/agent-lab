现在实现 Production Agent 实验 ：

Failure Injection / Chaos Testing。

不要增加新业务功能。

专门测试 Agent 在失败情况下的行为。

注入：

1. LLM timeout
2. LLM 5xx
3. Tool timeout
4. Tool 5xx
5. Tool invalid response
6. MCP unavailable
7. Rate limit
8. Worker crash
9. Checkpoint failure
10. Duplicate request
11. Concurrent requests
12. Invalid structured output

对于每一种 failure 定义：

Expected behavior

例如：

Tool timeout：

retry
→ backoff
→ retry limit
→ fallback
→ structured failure

Worker crash：

checkpoint
→ restart
→ resume

Duplicate request：

idempotency key
→ detect duplicate
→ return existing result

创建：

tests/failure_injection/

test_llm_failure.py
test_tool_failure.py
test_mcp_failure.py
test_worker_crash.py
test_duplicate_request.py
test_concurrency.py

创建：

docs/09-FAILURE-MATRIX.md

最终生成：

Failure
→ Detection
→ Recovery
→ User Experience
