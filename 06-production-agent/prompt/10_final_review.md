现在进行 Production Agent Final Review。

不要修改代码。

假设这个 Agent 明天要服务真实用户。

从以下角色分别审查：

1. AI Architect
2. Backend Engineer
3. SRE
4. Security Engineer
5. AI Evaluation Engineer
6. Product Engineer

检查：

## Architecture

Agent orchestration
State
Tools
MCP
Memory
RAG
Multi-Agent

## Reliability

Retry
Timeout
Checkpoint
Recovery
Idempotency
Concurrency

## Security

Authentication
Authorization
Prompt Injection
Tool Abuse
Secrets
Data Isolation
PII

## Observability

Logs
Metrics
Traces
Alerts
Cost
Latency

## Evaluation

Offline eval
Regression
Safety eval
Tool eval
Routing eval
Production feedback

## Operations

Deployment
Scaling
Rate limiting
Queue
Health check
Graceful shutdown

## Cost

Token budget
Tool budget
Agent budget
Request budget

## User Experience

Streaming
Progress
Long-running task
Failure message
Resume
Human approval

输出：

### Production Readiness Score

Architecture: /10
Reliability: /10
Security: /10
Observability: /10
Evaluation: /10
Scalability: /10
Cost: /10
UX: /10

然后生成：

P0 — Must fix before production
P1 — Should fix soon
P2 — Can improve later
P3 — Nice to have

最后回答：

“这个 Agent 现在到底是不是 Production Ready？”

如果不是，明确说明：

“最关键的 5 个阻塞问题是什么？”

不要修改任何代码。把Final Review结果写到 docs/10-Final-Review.md
