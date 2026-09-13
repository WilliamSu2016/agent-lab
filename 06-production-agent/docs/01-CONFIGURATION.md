# 01 — Configuration & Secrets

本项目所有配置与密钥统一通过 `config` 包加载：`config.load_settings()` 是**唯一**读取环境变量的地方；其余代码（`src/` 下每个 `main()`、每个 Agent 模块）只接收已校验好的 `config.Settings` 对象，绝不直接读 `os.environ`，也绝不在源码中硬编码任何密钥、模型名、超时或限流值。

## 1. 设计原则

1. **所有 API key 只从环境变量读取。** `config/settings.py` 里没有任何一处给 `OPENAI_API_KEY` 赋非空默认值。
2. **Secret 不出现在源码。** 全仓库搜索确认没有硬编码的 `sk-...`/`api_key="..."` 字面量；唯一的真实密钥只存在于本地 `.env`（不提交）。
3. **Secret 不出现在 Git。** `.env` 在 `.gitignore` 中；提供 `.env.example` 作为不含真实值的模板。
4. **Model name 可配置。** 通过 `OPENAI_MODEL` 环境变量；缺失时启动即失败，不使用任何硬编码模型名兜底。
5. **Agent limits 可配置。** `AGENT_MAX_WORKERS`、`AGENT_MAX_ITERATIONS`。
6. **Timeout 可配置。** `AGENT_WORKER_TIMEOUT_SECONDS`（单个 Worker）与 `AGENT_RUN_TIMEOUT_SECONDS`（整个运行的总预算，通过 `run_multi_agent_research(..., run_timeout_seconds=...)` 强制执行，超时抛出 `MultiAgentResearchTimeoutError`）。
7. **Max iterations 可配置。** 与 `AGENT_MAX_ITERATIONS` 共用同一个值（Planner↔Reviewer 循环上限）。
8. **Retry policy 可配置。** `LLM_RETRY_MAX_RETRIES`、`LLM_RETRY_BACKOFF_BASE_SECONDS`、`LLM_RETRY_BACKOFF_MAX_SECONDS`、`LLM_RETRY_STATUS_CODES`，由 `src/specialists/llm.py` 中的 `build_openai_text_llm_call()` 实际执行（指数退避 + 随机抖动；只重试可重试的失败类型，见下文）。
9. **Environment 至少区分** `development` / `test` / `production`，通过 `ENVIRONMENT` 变量指定；非法值直接报错。

## 2. 环境变量参考

| 变量 | 必需 | 默认值 | 说明 |
|---|---|---|---|
| `ENVIRONMENT` | 否 | `development` | 必须是 `development` / `test` / `production` 之一，其他值直接 `ConfigurationError` |
| `OPENAI_API_KEY` | **是（secret）** | 无 | 缺失/空白直接失败；错误信息不回显该值 |
| `OPENAI_MODEL` | **是** | 无 | 模型名/部署名；缺失直接失败 |
| `OPENAI_BASE_URL` | development/test 可选；**production 必需** | `https://api.openai.com/v1`（仅非 production） | production 下缺失会报错，避免私有网关流量默认发往公共端点 |
| `AGENT_MAX_WORKERS` | 否 | `6` | 整数 ≥ 1 |
| `AGENT_MAX_ITERATIONS` | 否 | `3` | 整数 ≥ 1 |
| `AGENT_WORKER_TIMEOUT_SECONDS` | 否 | `30.0` | 浮点数 > 0 |
| `AGENT_RUN_TIMEOUT_SECONDS` | 否 | `300.0` | 浮点数 > 0；整个 run 的硬性总预算 |
| `LLM_RETRY_MAX_RETRIES` | 否 | `2` | 整数 ≥ 0；`0` 表示禁用重试 |
| `LLM_RETRY_BACKOFF_BASE_SECONDS` | 否 | `1.0` | 浮点数 > 0 |
| `LLM_RETRY_BACKOFF_MAX_SECONDS` | 否 | `20.0` | 浮点数 > 0；必须 ≥ base，否则报错 |
| `LLM_RETRY_STATUS_CODES` | 否 | `408,409,429,500,502,503,504` | 逗号分隔整数列表 |

完整默认值和校验规则以 `config/settings.py` 中 `AgentLimits`/`RetryPolicy`/`load_settings()` 的实现为准；本表如与代码不一致，以代码为准。

## 3. 用法

```python
from dotenv import load_dotenv
from config import ConfigurationError, load_settings

load_dotenv()  # 唯一读取 .env 文件的地方；config 模块本身不接触文件系统

try:
    settings = load_settings()
except ConfigurationError as exc:
    raise SystemExit(f"Configuration error: {exc}") from exc

# settings.api_key / settings.model_name / settings.base_url
# settings.limits.max_workers / max_iterations / worker_timeout_seconds / run_timeout_seconds
# settings.retry.max_retries / backoff_base_seconds / backoff_max_seconds / retry_on_status_codes
```

所有 `src/` 下的入口（`multi_agent_research/main.py`、`specialists/*.py`、`02_handoff.py`、`03_supervisor.py`、`04_parallel_multi_agent.py`、`05_shared_state.py`）均已改为这个模式。`src/specialists/llm.py::build_openai_text_llm_call(settings)` 直接接收 `Settings`，不再接收裸的 `api_key`/`model_name`/`base_url` 参数。

## 4. Secret 管理规则

- **本地开发**：把真实值写进本地 `.env`（不提交），从 `.env.example` 复制而来。
- **绝不**把真实 secret 写入：源码、测试文件、文档、commit message、日志、trace 文件。`Settings.__repr__`/`__str__` 已经显式屏蔽 `api_key`（打印为 `***redacted***`），防止不小心 `print(settings)`/日志记录整个对象时泄漏。
- **CI / 生产**：secret 由平台的 secret manager（例如 GitHub Actions secrets、Key Vault、Vault 等）注入为环境变量，不落盘、不进镜像层。这是后续部署工作的前提条件；本次改造只保证代码侧不会以任何方式绕开环境变量读取 secret。
- 错误信息设计为**绝不回显** secret 值本身（即使该值被误设置为一个"像"真实密钥的字符串），只回显变量名和是否缺失/是否为空。

## 5. Environment 区分

`ENVIRONMENT` 目前影响的行为：

- 合法值：`development`、`test`、`production`；其他任何值都会在启动时报 `ConfigurationError`。
- **production** 额外要求显式设置 `OPENAI_BASE_URL`（不允许静默回退到公共 OpenAI 端点）——这是为了防止一个原本配置为私有网关的生产环境，因为遗漏配置而意外把请求发到公共端点（数据外泄风险，呼应 `src/tracing.py` 中关于 trace 导出目的地的同一类考量）。
- development/test 环境下 `OPENAI_BASE_URL` 可省略，默认指向公共 OpenAI 端点，方便本地快速上手。
- 三个环境目前共享同一套 `AgentLimits`/`RetryPolicy` 默认值和校验规则；如果未来需要按环境有不同的默认限流/超时策略，应在 `load_settings()` 中按 `environment` 分支扩展，而不是在各个调用方各自硬编码。

## 6. Retry Policy 的重试范围

`src/specialists/llm.py::_is_retryable()` 只把以下情况判定为可重试：

- `openai.APIConnectionError` / `openai.APITimeoutError` / `openai.RateLimitError`
- `openai.APIStatusError` 且其 `status_code` 落在 `retry.retry_on_status_codes` 中（默认 `408,409,429,500,502,503,504`）

其余异常（例如 401/403 鉴权失败、400 请求非法、内容策略拒绝等）**立即抛出，不重试**——重试一个必然失败的请求只会浪费预算、延长故障时间。重试之间使用指数退避（`backoff_base_seconds * 2^attempt`，封顶 `backoff_max_seconds`）加小幅随机抖动，避免多个并发 Worker 同时重试造成惊群效应。

`LLM_RETRY_MAX_RETRIES=0` 可以在 `test` 环境下使用，让失败立即冒泡，便于确定性测试（不必等待退避延迟）。

## 7. 自动化测试

`tests/test_config.py` 覆盖本次改造的核心要求，运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py -v
```

关键用例：

- **缺少必需 secret 时必须明确失败**：`test_missing_api_key_raises_configuration_error`（空/缺失/纯空白都必须失败，且错误信息包含变量名与"missing required secret"字样，而不是让程序继续运行到某个更深、更难定位的地方才失败）。
- 错误信息不泄漏 secret 值本身（即使该值“看起来像”真实密钥）。
- 模型名缺失同样明确失败；模型名可通过环境变量覆盖。
- 非法 `ENVIRONMENT` 值报错；三个合法环境值都能正确加载；production 缺少 `OPENAI_BASE_URL` 报错。
- Agent limits / timeout / max iterations 的默认值、覆盖值、非法值（非数字、非正数、越界）分别校验。
- Retry policy 的默认值、覆盖值、`max_retries=0`、非法状态码列表、`backoff_max < backoff_base` 等边界分别校验。
- `Settings` 是 frozen dataclass，`repr()`/`str()` 屏蔽 `api_key`。

所有测试都通过 `load_settings(env=<显式 dict>)` 注入配置，从不依赖也不修改真实进程环境变量或磁盘上的 `.env` 文件，因此在任何机器上运行结果都是确定性的。

## 8. 文件清单

| 路径 | 作用 |
|---|---|
| `config/__init__.py` | 对外导出 `Settings`/`AgentLimits`/`RetryPolicy`/`ConfigurationError`/`load_settings` |
| `config/settings.py` | 配置加载、校验与默认值的唯一实现 |
| `.env.example` | 安全模板，所有变量均为空值或非敏感默认值，可直接复制为 `.env` |
| `.env` | 本地真实配置（已在 `.gitignore` 中，不提交） |
| `tests/test_config.py` | 自动化测试，覆盖第 7 节所列用例 |
