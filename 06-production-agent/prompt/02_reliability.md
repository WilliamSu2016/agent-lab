现在实现 Production Agent 实验：

Reliability。

针对所有 Agent Tools 建立统一可靠性策略。

要求：

### Retry

实现：

* exponential backoff
* maximum retry count
* retryable exception classification

区分：

Retryable：

* timeout
* temporary network error
* rate limit
* HTTP 5xx

Non-retryable：

* invalid parameters
* authentication failure
* permission denied
* business validation error

### Timeout

每一个 external tool 必须有 timeout。

### Agent loop

设置：

max_iterations

超过后：

进入 controlled failure。

### Error handling

不要：

raise exception → application crash

而应该：

Tool failure
→ classify
→ retry / fallback / terminate
→ structured error
→ trace

### Idempotency

识别所有具有 side effect 的 Tool：

例如：

send_email
create_order
publish_report

为这些 Tool 设计 idempotency strategy。

创建：

src/reliability/

retry.py
timeout.py
errors.py
idempotency.py

tests/

test_retry.py
test_timeout.py
test_errors.py

docs/02-RELIABILITY.md

添加 failure injection tests。
