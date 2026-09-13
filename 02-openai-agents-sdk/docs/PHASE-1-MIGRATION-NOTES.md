# Phase 1 迁移记录：只迁移 Agent 定义

本阶段严格按照要求执行：**只迁移 Agent 定义**，不迁移 `search_web` 工具，不使用 multi-agent、handoff、Sessions、guardrails、MCP、RAG。目标路径：

```
User -> Agent -> Runner -> Final Answer
```

> **更新说明（第二轮）**：本阶段最初是以"新增并行文件"的方式完成的（`src/agent_sdk.py`、`src/main_sdk.py`、`tests/test_agent_sdk.py`，`src/agent.py`/`src/main.py` 保持不变）。之后用户明确要求："agent_sdk.py 和 main_sdk.py 的作用是代替原来的两个代码文件，那么就直接覆盖，不需要创建新的文件"。因此第二轮把 SDK 实现**直接覆盖进** `src/agent.py` 和 `src/main.py`，删除了 `agent_sdk.py`/`main_sdk.py` 这两个并行文件。下文按覆盖后的最终状态描述。

## 一、环境与安装

### 1. 虚拟环境

在项目目录下新建了 `.venv`（Python **3.10.8**）。

### 2. 为什么不能用 Python 3.11.0

项目机器上原本能找到的另一个 Python 是 **3.11.0**（2022-10 发布的 3.11 系列最初版本，之后 3.11 还发布过多个补丁版本，但机器上没有装）。用它安装 `openai-agents==0.22.0` 后，仅仅 `import agents` 就报错：

```text
File "...\agents\tool.py", line 93, in <module>
    | ToolFunctionWithToolContext[ToolParams]
  File "<frozen _collections_abc>", line 468, in __getitem__
  File "...\typing.py", line 1439, in _determine_new_args
    subargs.append(new_arg_by_param[x])
KeyError: ~TContext
```

触发点是 `agents/tool.py` 里对包含 `ParamSpec` / `Concatenate` 的类型别名做下标 + `|` 联合运算：

```python
ToolFunction = (
    ToolFunctionWithoutContext[ToolParams]
    | ToolFunctionWithContext[ToolParams]
    | ToolFunctionWithToolContext[ToolParams]
)
```

官方 `pyproject.toml` 声明 `requires-python = ">=3.10"`，并且在分类器里列出了 3.10/3.11/3.12/3.13/3.14，并没有说不支持 3.11。这更像是 CPython 3.11.0 这个具体构建版本在 `typing` 泛型别名缓存上的已知缺陷，而不是 openai-agents 官方不支持 Python 3.11。

用机器上另一个已装的 **Python 3.10.8** 重建虚拟环境后，`from agents import Agent, Runner` 可以正常导入，问题不再出现。因此本阶段虚拟环境固定使用 Python 3.10.8。

### 3. 安装的依赖

`requirements.txt` 新增一行：

```
openai-agents==0.22.0
```

这是当前 PyPI 上的最新正式发布版本（对应 GitHub `v0.22.0` release）。

## 二、最终文件状态（覆盖后）

| 文件 | 状态 | 作用 |
| --- | --- | --- |
| `src\agent.py` | **已覆盖**（原手写 Agent Loop 实现被替换） | 用 SDK 定义 Agent：`build_model()` 构造指向现有网关的 `OpenAIChatCompletionsModel`；`build_agent()` 创建 `Agent(name=..., instructions=SYSTEM_PROMPT, model=...)`，**不带任何 tools**；`run_agent(question, agent, max_turns=5)` 用 `Runner.run_sync()` 执行并返回 `final_output` 文本 |
| `src\main.py` | **已覆盖**（原手写 CLI 被替换） | CLI 入口，UX 与覆盖前保持一致（读取 argv、加载 `.env`、校验环境变量、打印最终答案），内部调用新的 `src.agent`，走 `User -> Agent -> Runner -> Final Answer` 路径 |
| `tests\test_agent.py` | **已覆盖**（原测试针对手写工具调用循环，已删除） | 改为用 SDK 官方公开的 `agents.testing.ScriptedModel` + `assistant_message` 离线、确定性地验证：Agent 的 `instructions` 确实等于迁移后的 `SYSTEM_PROMPT`、Agent 没有注册任何工具、`Runner.run_sync()` 能正确跑完并返回预期的最终文本 |
| `src\agent_sdk.py`、`src\main_sdk.py`、`tests\test_agent_sdk.py` | **已删除** | 第一轮的并行文件，内容已合并进上面三个文件，不再需要重复维护 |
| `tests\test_evaluations.py` | **保留文件，但测试被显式禁用**（`@unittest.skip(...)`） | 原因见下方"评测用例为什么被禁用" |
| `tests\evaluations\run_evaluations.py`、`tests\evaluations\cases.json` | **未修改** | 按要求不修改 evaluation cases；`run_evaluations.py` 本身也未改动，仍是旧手写 `run_agent(question, llm, search_fn, max_turns)` 签名的调用方，只是暂时没有测试会执行它 |
| `src\llm_client.py`（`OpenAICompatibleClient`）、`src\tools.py`（`web_search`、`WEB_SEARCH_TOOL`）、`src\prompts.py`（`SYSTEM_PROMPT`） | **未修改** | 见下方说明 |

### 关于 instructions 迁移

`src\agent.py` 直接复用 `src\prompts.py` 里现有的 `SYSTEM_PROMPT` 常量，把它原样传给 `Agent(instructions=SYSTEM_PROMPT)`。没有改写提示词内容，符合"保持任务不变、不新增功能"的要求。

### 关于模型配置

项目当前 `.env` 里配置的 `OPENAI_BASE_URL` 指向一个企业内部的 OpenAI 兼容网关（IBM 的 `nextgen-beta.ica.ibm.com`），只支持 Chat Completions 协议，不是官方 OpenAI Responses API。SDK 对 OpenAI 模型默认走 Responses API，如果直接传字符串模型名会走错协议。因此这里显式使用：

```python
OpenAIChatCompletionsModel(
    model=model_name,
    openai_client=AsyncOpenAI(api_key=api_key, base_url=base_url),
)
```

来复刻现有 `OpenAICompatibleClient` 实际请求的协议（Chat Completions），保证连的是同一个网关、同一种请求格式。

### 关于 Tracing

SDK 默认开启 tracing，并会把请求/响应内容尝试导出到 OpenAI 官方的 tracing 后端——这与用哪个模型提供方无关。因为本项目实际请求的是企业内部网关，而不是官方 OpenAI API，默认 tracing 行为会把内部网关的请求内容尝试发往 OpenAI 的公共 tracing 服务（用的还是网关的 key，不是真正的 OpenAI 控制台 key）。为了不引入这个非预期的数据外发路径，`run_agent()` 显式设置：

```python
run_config=RunConfig(tracing_disabled=True)
```

这与覆盖前手写实现"没有任何遥测上报"的行为保持一致。

## 三、评测用例为什么被禁用

覆盖 `src/agent.py` 之后，`run_agent` 的函数签名从：

```python
run_agent(question, llm_client, search_fn, max_turns=...) -> dict  # 旧：手写实现
```

变为：

```python
run_agent(question, agent, max_turns=...) -> str  # 新：SDK 实现
```

`tests/evaluations/run_evaluations.py` 是专门为旧签名写的：它用 `ScriptedLLM` 脚本化工具调用序列，断言返回的 `state` 字典里有 `turns`/`status`/`tool_calls` 等字段。这些全部依赖"手写 Agent Loop + 工具调用"的实现细节，而新的 SDK 版本本阶段还没有工具（`search_web` 按要求未迁移），签名也完全不同，直接调用会报 `TypeError`（参数不匹配），不是断言失败。

经与用户确认，处理方式是：**暂时跳过（禁用）评测测试**，不删除、不修改 `run_evaluations.py` 和 `cases.json`（它们保留作为旧实现的参考），只在 `tests/test_evaluations.py` 里给测试方法加了 `@unittest.skip(...)`，并写明了重新启用的条件：等 `search_web` 迁移为 `@function_tool` 之后，需要把这个评测工具本身也改造成针对新 SDK Agent 打分（脚本化 `ScriptedModel` 而不是 `ScriptedLLM`），到那时再取消跳过。

## 四、测试验证

### 1. 离线单元测试（不需要网络/API key）

```
python -m unittest discover -s tests -v
```

结果：**3 个测试，2 个通过，1 个跳过**（`Ran 3 tests ... OK (skipped=1)`）：

- `tests/test_agent.py` 的 2 个测试（新，针对 SDK `Agent` + `Runner`）：全部通过
- `tests/test_evaluations.py` 的 1 个测试：**已跳过**（原因见上）

### 2. 真实端到端冒烟测试（连的是项目现有网关，同一凭据）

```
python -m src.main "What is the capital of France?"
```

输出：

```
Paris.
```

确认了覆盖后的 `User -> Agent -> Runner -> Final Answer` 这条链路在真实网关上也走得通，CLI 行为（打印最终答案）与覆盖前一致。

## 五、当前 Minimal Agent 项目哪些代码已经可以删除？

> 以下表格是 Phase 3（Runtime 清理）完成之后的最终状态；`llm_client.py` 和 `WEB_SEARCH_TOOL` 已经在 Phase 3 里被实际删除，详见第七节。

| 现有文件 | 现在能不能删 | 为什么 |
| --- | --- | --- |
| `src\agent.py` 里旧的 `run_agent` 手写循环、`_call_llm`、`_execute_web_search`、`_tool_error` | **已经被删除**（Phase 1 覆盖时直接替换掉了） | 已经没有任何测试或代码在引用这些函数（`test_agent.py` 已改为测 SDK 版本，`run_evaluations.py` 里对 `run_agent` 的调用虽然还在，但对应的测试已被禁用，不会在 CI/本地测试中触发） |
| `src\main.py` 里旧的、直接用 `OpenAICompatibleClient` + `web_search` 拼装 messages 的逻辑 | **已经被删除**（Phase 1 覆盖时直接替换掉了） | 同上，新的 CLI 完全走 SDK `Agent`/`Runner` |
| `src\llm_client.py`（`OpenAICompatibleClient`、`LLMError`，整个文件） | **已经被删除**（Phase 3） | 确认没有任何生产代码或测试再引用它（`src/agent.py` 用的是 SDK 的 `OpenAIChatCompletionsModel`；`run_evaluations.py` 的 `ScriptedLLM` 是独立定义的假对象，不依赖这个文件），属于已被 SDK 接管、纯粹重复的手写 LLM Runtime 代码 |
| `src\tools.py`（`web_search()`） | **不能删，而且现在正在被使用** | 已经把 `web_search()` 包装进了 `src/agent.py` 的 `@function_tool`，业务逻辑一个字节没改，是真正被复用、不是被替代 |
| `src\tools.py`（`WEB_SEARCH_TOOL` 手写 JSON Schema 常量） | **已经被删除**（Phase 3） | `@function_tool` 会自动从 `agent.py` 里 `web_search()` 包装函数的签名+docstring 生成等价 schema，这个手写常量已经没有任何代码引用，是重复的手写 tool schema |
| `src\prompts.py`（`SYSTEM_PROMPT`） | **不能删，而且以后也不会删** | 它是被"迁移"（复用），不是被替代。`src/agent.py` 直接导入并原样传给 `Agent.instructions`。它是业务内容，不是可以被框架吃掉的样板代码 |
| `tests\test_evaluations.py`、`tests\evaluations\run_evaluations.py`、`tests\evaluations\cases.json` | **暂不能删** | 按要求保留 Evaluation tests；`run_evaluations.py` 和 `cases.json` 保留作为旧实现的行为参考和未来"评测工具 SDK 化"的起点，即使当前测试被跳过 |

### 未来阶段的预告（不是本阶段的改动）

| 未来动作 | 前提条件 |
| --- | --- |
| 改造 `tests/evaluations/run_evaluations.py`，用 `ScriptedModel` 替代 `ScriptedLLM`，脚本化 SDK Agent 的工具调用，重新启用 `test_all_cases_pass_deterministic_assertions` | `search_web` 已迁移完成，具备改造条件，可以作为下一步任务；本次 Runtime 清理没有涉及评测工具，按要求原样保留 |

### 明确不会被删除的部分（不管迁移到哪个阶段）

- `SYSTEM_PROMPT` 的**内容**（研究策略、何时搜索、如何引用来源）——只会被复用，不会被框架取代。
- `web_search()` 里真正访问 DuckDuckGo 并整理结果的业务逻辑——SDK 不会替我们决定用哪个搜索源。
- `.env` 加载、命令行参数处理、"问题为空时报错退出"等应用入口逻辑。

## 六、Phase 2：迁移 search_web 为 SDK function tool

### 做了什么

`src/agent.py` 新增：

```python
from src.tools import web_search as _web_search

@function_tool
def web_search(query: str) -> dict[str, Any]:
    """Search the web for current, factual information.

    Args:
        query: A focused web search query.
    """
    return _web_search(query)
```

然后 `build_agent()` 里传入 `tools=[web_search]`。逐条对应要求：

| 要求 | 落实情况 |
| --- | --- |
| 保留原来的 `search_web()` Python function | `src/tools.py` 里的 `web_search()` 一个字节没改，业务逻辑（DuckDuckGo 请求、结果解析、参数校验）完全不变 |
| 使用 SDK 推荐的 function tool 机制 | 用官方 `@function_tool` 装饰器（`from agents import function_tool`），没有手写 JSON Schema、没有手写参数解析 |
| 不使用 MCP | 没有引入任何 MCP server/client |
| 不使用 Hosted Web Search Tool | 没有用 SDK 自带的托管 web search 工具，工具的搜索逻辑仍是我们自己的 `web_search()` |
| 不修改 search_web 的业务逻辑 | `src/tools.py` 未改动；`agent.py` 里新增的 `web_search()` 只是一层薄包装，调用 `_web_search(query)`（即 `src.tools.web_search`），没有增加/修改任何业务判断 |
| Tool 的输入参数保持不变 | 用 `@function_tool` 自动从函数签名 `query: str` + docstring 生成的 JSON Schema，和原来手写的 `WEB_SEARCH_TOOL` schema 完全一致：`{"query": {"type": "string", "description": "A focused web search query."}}`，`required: ["query"]` |
| Tool 的返回值保持兼容 | 包装函数直接原样返回 `_web_search(query)` 的 `dict`（`{"query": ..., "results": [...]}`），没有改变结构 |
| Agent 是否调用 Tool 仍由 LLM 决定 | `build_agent()` 只是把 `web_search` 注册进 `tools=[...]`，是否调用完全由模型根据 `Agent.instructions`（`SYSTEM_PROMPT`）自行判断；测试里通过让 `ScriptedModel` 分别脚本化"不调用"和"调用"两种场景来验证，不是我们代码里做 if/else 分支 |
| 不自己实现 Tool dispatch | 工具调用的 JSON 参数解析、Pydantic 校验、实际函数调用、结果序列化回传，全部由 SDK 的 `function_tool`/`Runner` 完成（见 `agents/tool.py` 里 `_on_invoke_tool_impl`），`agent.py` 里没有任何 `if tool_name == "web_search"` 之类的手写分发代码 |
| 不自己实现 Agent loop | 工具调用后是否继续问模型、什么时候停止，全部由 `Runner.run_sync()` 内部循环处理，`run_agent()` 只调用了一次 `Runner.run_sync(...)` |

### 验证结果

**离线确定性测试**（`tests/test_agent.py`，用官方 `agents.testing.ScriptedModel` + `function_call`/`assistant_message` 脚本化，不需要网络/API key）：

| Case | 测试方法 | 验证方式 |
| --- | --- | --- |
| Case 1：不需要搜索的问题 | `test_case1_agent_does_not_call_web_search_when_not_needed` | 脚本化模型只返回一个 `assistant_message`（不含任何 `function_call`），断言 `_web_search`（真实业务函数）从未被调用，且 `result.new_items` 里没有 `ToolCallItem` |
| Case 2：需要搜索的问题 | `test_case2_agent_can_call_web_search_when_needed` | 脚本化模型先返回一个 `function_call(name="web_search", ...)`，再返回最终 `assistant_message`；断言 `_web_search` 恰好被调用一次，且调用参数与脚本化的 `query` 一致 |
| Case 3：Tool 结果自动回到 Agent 继续执行 | `test_case3_tool_result_automatically_flows_back_into_the_agent_loop` | 同样两步脚本；断言 `result.new_items` 里同时存在 `ToolCallItem`（工具调用）和 `ToolCallOutputItem`（工具结果），且最终 `result.final_output` 是脚本第二步的文本——证明是 Runner 自己把工具结果喂回去、驱动了第二次模型调用，而不是测试代码手写的循环 |

全部测试跑一遍（`python -m unittest discover -s tests -v`）：**6 个测试，5 通过 1 跳过**（跳过的是 Phase 1 就已禁用的旧评测用例，原因见上文第三节，本次没有变化）。

**真实端到端冒烟测试**（连的是项目现有网关和模型，未使用任何脚本化/mock）：

- `python -m src.main "What is the capital of France?"` → 模型直接回答 `The capital of France is **Paris**.`，没有调用 `web_search`（验证了真实场景下 Case 1：不需要搜索时模型确实不会调用工具）
- `python -m src.main "Research the programming language Python and cite a source."` → 模型确实调用了 `web_search`（通过 SDK 的真实 dispatch，不是我们手写的），拿到 DuckDuckGo 的搜索结果后给出了带来源链接的完整回答（验证了 Case 2 + Case 3：工具被调用、工具结果确实自动回到 Agent 循环并驱动了最终回答）

一个额外发现（不是本次改动引入的问题，而是继承自现有 `web_search()` 业务逻辑本身的已知限制）：DuckDuckGo 的 Instant Answer API 对很多"人物/时事类"查询（例如"当前 OpenAI CEO 是谁"）经常返回空结果列表 `results: []`。真实测试时用这类问题，模型会反复换着关键词多次调用 `web_search`（因为每次都没搜到），如果恰好超过 `max_turns=5` 就会抛出 `MaxTurnsExceeded`。这与工具迁移无关——覆盖前的手写实现用的是同一个 `web_search()`、同一个 `max_turns=5` 默认值，遇到同样的空结果查询会有一样的行为。本次没有修改 `web_search()` 的业务逻辑或调整 `max_turns` 默认值（题目要求不修改业务逻辑），只是如实记录这个预先存在的限制。

## 七、Phase 3：架构重构——删除已被 SDK 接管的 Runtime 代码

目标：让 OpenAI Agents SDK 完全负责 Agent Runtime（LLM 调用循环、tool dispatch、tool schema、tool-call 检测、tool-result 路由、continuation、termination），项目里不再重复实现任何一块。

### 检查结果与删除项

| 检查项 | 项目里对应的代码 | 处理结果 |
| --- | --- | --- |
| 1. 手写 LLM loop | `src/llm_client.py` 的 `OpenAICompatibleClient.complete()`——拼 HTTP 请求、`urlopen`、解析 `choices[0].message` | **整个文件删除**（61 行）。已确认没有任何生产代码或测试再导入它：`src/agent.py` 用的是 SDK 自己的 `OpenAIChatCompletionsModel` + `AsyncOpenAI`；`tests/evaluations/run_evaluations.py` 里的 `ScriptedLLM` 是它自己独立定义的假对象，不 import `llm_client` |
| 2. 手写 tool dispatch | 无——Phase 1 覆盖 `src/agent.py` 时，旧的 `_execute_web_search()`/工具名判断分支已经被删除 | 本次检查确认没有残留，不需要再删 |
| 3. 手写 tool schema | `src/tools.py` 里的 `WEB_SEARCH_TOOL` 字典（手写 JSON Schema，18 行） | **已删除**。`@function_tool` 会从 `src/agent.py` 里 `web_search()` 包装函数的签名 + docstring 自动生成等价 schema，`WEB_SEARCH_TOOL` 已经没有任何代码引用 |
| 4. 手写 tool-call detection | 无——旧的“判断 assistant 消息里有没有 `tool_calls`”逻辑随 Phase 1 覆盖一起删除了 | 本次检查确认没有残留 |
| 5. 手写 tool-result routing | 无——旧的“把 tool 执行结果拼回 messages 数组再调用一次 LLM”的代码随 Phase 1 覆盖一起删除了 | 本次检查确认没有残留 |
| 6. 手写 continuation logic | 无——旧的 `for turn in range(max_turns)` 循环随 Phase 1 覆盖一起删除了 | 本次检查确认没有残留 |
| 7. 手写 Agent termination logic | 无——旧的“content 非空即停止”判断随 Phase 1 覆盖一起删除了 | 本次检查确认没有残留 |

结论：第 2-7 项对应的 Runtime 代码在 Phase 1 覆盖 `src/agent.py`/`src/main.py` 时已经被整体替换掉了（当时是为了迁移 Agent 定义顺带做的）。本次重构实际新删除的是此前一直保留、但已经查明确实是死代码的两块：**手写 LLM HTTP 客户端**（`src/llm_client.py`）和**手写 tool schema 常量**（`WEB_SEARCH_TOOL`）。

### 保留的部分（按要求逐条核对）

| 要求保留的类别 | 对应代码 | 状态 |
| --- | --- | --- |
| Agent instructions | `src/prompts.py`：`SYSTEM_PROMPT` | 未改动 |
| Business Tools | `src/tools.py`：`web_search()` | 未改动（业务逻辑一个字节没动，只删了它旁边已经没用的 schema 常量） |
| Domain logic | `web_search()` 内部 DuckDuckGo 请求/解析逻辑 | 未改动 |
| Application entry point | `src/main.py` | 未改动业务行为，只更新了两处过时的文档字符串（不再提"Phase 1"、不再提已删除的 `OpenAICompatibleClient`） |
| Evaluation tests | `tests/test_evaluations.py`、`tests/evaluations/run_evaluations.py`、`tests/evaluations/cases.json` | 未改动（沿用 Phase 2 里"评测用例暂时禁用"的既有状态，本次重构没有再动它们） |

### 删除统计

| 删除项 | 文件 | 行数 |
| --- | --- | --- |
| 手写 LLM HTTP 客户端（`OpenAICompatibleClient`、`LLMError`，整个文件） | `src/llm_client.py`（整个文件删除） | 61 行 |
| 手写 tool JSON Schema 常量（`WEB_SEARCH_TOOL`） | `src/tools.py` | 18 行 |
| **本次合计** | | **79 行 Runtime/schema 代码，1 个文件** |

（第 2-7 类 Runtime——手写 loop、dispatch、tool-call 检测、结果路由、continuation、termination——已经在 Phase 1 覆盖 `src/agent.py`/`src/main.py` 时被替换删除，不计入本次新增的删除量；这里只统计本次重构实际新删除的内容。）

### 测试结果

```
python -m unittest discover -s tests -v
```

**6 个测试，5 通过、1 跳过**（跳过的仍是 Phase 2 就已禁用的旧评测用例，本次没有变化）。删除 `llm_client.py` 和 `WEB_SEARCH_TOOL` 之后，用 `grep` 确认项目里（`src/`、`tests/`）不再有任何代码引用 `llm_client`、`OpenAICompatibleClient`、`WEB_SEARCH_TOOL`。真实端到端冒烟测试（`python -m src.main "What is the capital of France?"`）依旧正确返回 `The capital of France is **Paris**.`，确认清理没有破坏任何现有行为。

## 八、与既有分析文档的关系

本文件是 `docs/OPENAI-AGENTS-SDK-MAPPING.md` 里"哪些代码可以被 SDK 删除"这个问题的**阶段化落地版**：那份文档回答的是"最终形态下大致会怎样"，本文件回答的是"当前这一步实际做了什么、现在能不能删"。两者结论一致：删除永远滞后于迁移验证——本次之所以能直接删除旧的 `run_agent`/`main` 实现，是因为用户明确确认了覆盖方案，并接受"评测用例暂时禁用"作为过渡状态。
