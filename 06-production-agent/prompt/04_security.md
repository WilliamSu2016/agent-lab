现在实现 Production Agent 实验 ：

Security & Guardrails。

建立三层防护：

Layer 1：
Input Guardrail

Layer 2：
Agent / Workflow Guardrail

Layer 3：
Tool Guardrail

实现：

1. Prompt injection detection。
2. Tool argument validation。
3. Authorization check。
4. User identity propagation。
5. Tenant isolation。
6. Sensitive-data filtering。
7. High-risk action approval。
8. Output validation。

将 Tool 分级：

LOW:
read-only

MEDIUM:
write but reversible

HIGH:
external side effect

例如：

search_web → LOW

update_record → MEDIUM

send_email → HIGH

要求：

HIGH risk Tool：

Agent
→ approval
→ Tool

而不是：

Agent
→ Tool

创建：

src/security/

authorization.py
guardrails.py
tool_policy.py
sanitization.py

tests/security/

test_prompt_injection.py
test_authorization.py
test_tool_policy.py
test_data_isolation.py

docs/04-SECURITY.md

创建至少 10 个攻击测试用例。
