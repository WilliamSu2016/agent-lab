# Production Agent Observability Dashboard Specification

本规范定义一个基于 `src/observability/` 数据源（Traces / Metrics / Logs）构建的
生产 Dashboard，用于回答 `docs/05-OBSERVABILITY.md` 中列出的 10 个问题，并对 7 个
核心指标提供持续可见性、告警与下钻能力。本规范只描述**面板/数据源/交互设计**，
不绑定具体可视化产品（Grafana / Datadog / 自建 UI 均可实现）。

## 1. 目标与受众

| 受众 | 关注问题 |
|---|---|
| On-call / SRE | 当前是否健康？哪个 Tool/Agent 正在拖垮 SLA？(Q3, Q6, Q7, Q8) |
| Agent 开发者 | 某次失败的根因是什么？执行链路长什么样？(Q1, Q2, Q7, Q8) |
| 产品/成本负责人 | 哪个 Agent 最贵？总体 token/成本趋势如何？(Q4, Q5) |
| 安全/合规 | 谁在什么时间用了哪个 Agent version 做了什么？(Q9, Q10) |

## 2. 顶层布局

```
┌─────────────────────────────────────────────────────────────────┐
│  Global Filters: environment | agent_version | time range | user │
├───────────────┬───────────────┬───────────────┬─────────────────┤
│ Success Rate   │ P95 Latency   │ Total Cost     │ Active Requests │
│ (big number)   │ (big number)  │ (big number)   │ (big number)    │
├───────────────┴───────────────┴───────────────┴─────────────────┤
│  Panel A: Request Volume & Success Rate over Time (line chart)   │
├───────────────────────────────────┬───────────────────────────────┤
│  Panel B: Agent Cost Ranking      │  Panel C: Tool Failure Ranking │
│  (bar chart, Q5)                  │  (bar chart, Q6)               │
├───────────────────────────────────┼───────────────────────────────┤
│  Panel D: Tool Latency (p50/p95/p99, Q3) │ Panel E: Token Usage (Q4)│
├───────────────────────────────────┴───────────────────────────────┤
│  Panel F: Agent Execution Tree Explorer (single-request drilldown)│
│           answers Q1 / Q2 / Q7 / Q8 / Q9 / Q10 for one request     │
├─────────────────────────────────────────────────────────────────┤
│  Panel G: Structured Log Stream (filterable, redacted)            │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 面板详情

### 3.1 顶部 KPI 条（Big Number 卡片）

| 卡片 | 数据源 | 计算 |
|---|---|---|
| Success Rate | Metrics | `MetricsRegistry.success_rate()`，按选定时间窗口聚合 |
| P95 Latency | Metrics | 所有 WORKFLOW 根 span 的 `latency_stats(..., kind=WORKFLOW).p95_ms` |
| Total Cost | Metrics | `MetricsRegistry.total_cost_usd()`，按时间窗口聚合，标注"illustrative pricing" |
| Active Requests | Traces | 当前时间窗口内根 span 数（`Tracer.all_traces()` 或后端存储的等效计数） |

告警建议：Success Rate < 95%（5 分钟滚动）触发 P2；P95 Latency 超过基线 2 倍
持续 10 分钟触发 P2；Total Cost 单位时间环比增长 > 50% 触发预算告警（非 P1，
仅通知）。

### 3.2 Panel A — Request Volume & Success Rate over Time

- 类型：双轴折线图。左轴：每分钟 request 数（根 span 计数）；右轴：success_rate。
- 数据源：Metrics（按时间桶聚合 `success_rate(kind=None)`）。
- 交互：点击某个时间点 → 跳转到 Panel F，预筛选该时间窗口内失败的 request。

### 3.3 Panel B — 哪个 Agent 最昂贵？(Q5)

- 类型：水平条形图，按 `MetricsRegistry.most_expensive_agent()` 排序展开为
  全量排名（不仅仅是 top-1），即对每个已知 agent_name 调用
  `total_cost_usd(agent_name=name)` 排序。
- 附加列：`token_usage_total(agent_name=name)`、调用次数。
- 交互：点击某个 agent → 下钻到该 agent 的所有 LLM call 记录（model、
  prompt/completion tokens、单次 cost）。

### 3.4 Panel C — 哪个 Tool 最容易失败？(Q6)

- 类型：水平条形图，按 `tool_failure_rate(name)` 降序排列，副轴显示调用量
  （避免"只调用过一次且失败"的 tool 被误判为最不可靠——与
  `least_reliable_tool()` 平局打破逻辑一致，展示调用量帮助人工判断显著性）。
- 交互：点击某个 tool → 下钻到该 tool 最近 N 次失败调用的 trace（跳转 Panel F）。

### 3.5 Panel D — Tool Latency (Q3)

- 类型：按 tool 分组的箱型图或 p50/p95/p99 折线。
- 数据源：`MetricsRegistry.latency_stats(name, kind=TOOL)`。
- 告警建议：任一 tool 的 p95 超过其 7 天基线 3 倍，触发告警（可能是外部依赖降级）。

### 3.6 Panel E — Token Usage (Q4)

- 类型：堆叠面积图，按 agent_name 堆叠 prompt_tokens / completion_tokens 随时间变化。
- 数据源：LLM span 属性（`prompt_tokens`/`completion_tokens`），经
  `MetricsRegistry.token_usage_total(agent_name=...)` 按时间桶聚合。
- 次面板：模型分布饼图（按 `model` 属性分组的调用次数/成本占比）。

### 3.7 Panel F — Agent Execution Tree Explorer（单请求下钻，核心面板）

给定一个 `request_id`/`trace_id`（可从任意其它面板下钻进入，或直接搜索输入），
展示：

- 树状可视化：完整 Span 树（`Span.to_dict()`），WORKFLOW → AGENT → TOOL/LLM 逐层
  展开，每个节点标注 `name`/`duration_ms`/`status`。→ 直接回答 Q1（哪些 Agent，
  取树中所有 AGENT 节点名）、Q2（哪些 Tool）、Q7（某 Agent 节点重复出现次数 =
  loop 迭代次数）。
- 若该请求失败：高亮 `find_root_cause_failure(root)` 返回的节点（最深的失败
  span），并显示其 `error` 字段全文 → 直接回答 Q8。
- 请求元信息栏：显示该 trace 根 `ExecutionContext` 的全部 6 个字段
  （`request_id`/`trace_id`/`user_id`/`session_id`/`agent_version`/`environment`）
  → 直接回答 Q9（用户是谁）、Q10（agent version）。
- 数据源：优先查内存 `Tracer`（近实时）；历史请求从 `jsonl_file_sink` 落盘文件
  （或生产环境中对应的 Trace 后端，如 Jaeger/Tempo）按 `trace_id` 精确查询。

### 3.8 Panel G — Structured Log Stream

- 类型：可过滤的日志表格，列 = `timestamp`/`level`/`logger`/`message` + 6 个身份
  字段 + 任意 `extra` 字段。
- 过滤器：按 `request_id`/`user_id`/`agent_version`/`environment`/`level` 过滤，
  且默认与 Panel F 的当前选中 `request_id` 联动（选中一个 request 后，Panel G
  自动只显示该 request 的日志行）。
- **安全说明**：所有展示的日志行均已经过 `JsonFormatter` 的默认脱敏（值模式匹配 +
  字段名匹配），Dashboard 层不需要（也不应该）再做二次脱敏决策——脱敏发生在日志
  产生的那一刻，而不是展示的那一刻，避免任何查询路径绕过脱敏直接读到原始日志文件。

## 4. 数据源与刷新频率

| 面板 | 数据源 | 刷新频率 |
|---|---|---|
| KPI 卡片 / Panel A | Metrics（滚动窗口聚合） | 30s |
| Panel B / C / D / E | Metrics（近 1h / 24h / 7d 可切换窗口） | 1 分钟 |
| Panel F | Traces（按需查询，非轮询） | 用户触发 |
| Panel G | Logs（尾随/tail 模式可选） | 实时（或 5s 轮询） |

## 5. 告警汇总表

| 指标 | 阈值 | 严重级别 |
|---|---|---|
| success_rate（全局，5 分钟滚动） | < 95% | P2 |
| success_rate（单 agent，5 分钟滚动） | < 90% | P3 |
| tool_failure_rate（单 tool，30 分钟滚动） | > 20% | P3 |
| p95 latency（单 tool） | > 3x 七日基线 | P3 |
| total_cost_usd（每小时） | 环比增长 > 50% | 通知（非 P1/P2） |
| retry_count（单 tool，30 分钟滚动） | 异常升高（> 3x 基线） | P3（通常是 tool_failure_rate 的先行指标） |

## 6. 已知限制

- 本规范假设的 Traces/Metrics 存储为本实验的内存实现
  （`src/observability/tracing.Tracer`、`src/observability/metrics.MetricsRegistry`）
  或落盘的 `jsonl_file_sink` 文件；生产部署应将其替换为真正的可观测性后端
  （OpenTelemetry + Prometheus/Jaeger 等），本规范中的面板/查询设计在替换后端时
  应保持不变，因为查询逻辑（`agents_visited`/`most_expensive_agent`/...）是围绕
  数据模型而非具体存储实现设计的。
- 成本相关面板明确标注"illustrative pricing"，不作为真实财务对账依据。
