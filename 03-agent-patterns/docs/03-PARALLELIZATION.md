# 03 — Parallelization：三个独立 Researcher 并发 → Synthesizer 汇总

对应代码：`src/parallelization.py`、`tests/test_parallelization.py`。
运行方式：`.\.venv\Scripts\python.exe -m src.parallelization "你的问题"`

已用真实网关实际运行过（见本文末尾"实际运行观察：Sequential vs Parallel latency"），不是只有理论设计。

## 1. 这是什么

```text
用户问题
  -> asyncio.gather(
         Python Researcher(仅研究 Python),
         TypeScript Researcher(仅研究 TypeScript),
         Go Researcher(仅研究 Go),
     )                                   # 三者并发执行，互不等待
  -> 全部完成后
  -> Synthesizer（汇总三份 findings，给出最终对比与推荐）
最终答案
```

`run_researchers_in_parallel()` 用 `asyncio.gather` 同时调度三个 `Runner.run`（SDK 的异步入口）协程；`run_parallelization()` 等三者全部完成后，再调用一次 `Synthesizer`。`run_researchers_sequentially()` 是另外提供的、仅用于对比测量的顺序版本，正式流程 (`run_parallelization`) 从不使用它。

## 2. 三个 Researcher 如何保持独立

| Agent | 职责 | 工具 | 输出类型 |
|---|---|---|---|
| Python Researcher | 只研究 Python 本身是否适合做 AI Agent | `search_web` | `LanguageFindings`（结构化） |
| TypeScript Researcher | 只研究 TypeScript 本身 | `search_web` | `LanguageFindings` |
| Go Researcher | 只研究 Go 本身 | `search_web` | `LanguageFindings` |
| Synthesizer | 汇总三份 findings，给出对比和推荐 | 无 | 纯文本 |

三个 Researcher 用同一个模板 `RESEARCHER_INSTRUCTIONS_TEMPLATE` 生成，但每份 instructions 明确写死"只研究 {language} 本身，绝不提及或比较其他语言"（`test_researcher_instructions_only_mention_their_own_language` 直接断言了这一点：Python Researcher 的 instructions 文本里不包含 "TypeScript" 或 "Go"）。每个 Researcher 的输入也只有"语言名 + 原始问题（仅作背景）"，不包含另外两个 Researcher 的任何信息——这是并发合法性的前提，见第 4 节。

## 3. 为什么这是 Parallelization，而不是别的模式

- **子任务预先固定、不由模型决定**：`LANGUAGES = ("Python", "TypeScript", "Go")` 写死在模块顶部；三个 Researcher 的存在、各自负责哪个语言，都是这份代码决定的，不是某个"协调者" LLM 在运行时决定"这次要拆成几份、分别研究什么"。这正是与 Orchestrator-Workers 的关键区别（见第 6 节）。
- **确实并发执行，不只是"逻辑上独立"**：`asyncio.gather(*(_run_researcher(...) for language in LANGUAGES))` 把三个 `Runner.run` 协程一起提交给事件循环。真实运行的 trace 证实了这一点——三个 Researcher 的 `started_at` 几乎同时（相差几毫秒），且彼此的执行区间在时间上重叠（见第 7 节的实测数据），这满足需求 2 和 4（"可以并行执行""用 asyncio 实现"）。
- **没有 Routing**：三个 Researcher 永远全部运行，不存在"分类后只选一个"的步骤。
- **没有 Orchestrator**：没有一个中央 LLM 先读问题、再决定"这次需要几个 worker、分别调查什么"；分工在写代码时就已经固定。
- **没有 Evaluator-Optimizer**：Synthesizer 的第一次输出就是最终答案，没有第二个 LLM 反过来评价、要求它修订重写的循环。
- **聚合由程序代码完成，不是模型自己判断"是否该合并"**：`run_parallelization()` 用 `await asyncio.gather(...)` 的 join 语义保证"三个 Researcher 全部完成之后"再启动 Synthesizer——这是原文所说"Aggregated programmatically"的具体体现。

## 4. 如果三个任务存在依赖关系，还能并行吗？

不能直接并行，或者说：**并行执行本身不会出错，但结果会是错的、或者根本无法并行**，原因分两种情况：

### 情况 A：真正的数据依赖（下游需要上游的输出内容）

如果 TypeScript Researcher 的任务被改成"参考 Python Researcher 的结论，指出 TypeScript 相对 Python 的优劣"，那么 TypeScript Researcher 在开始时根本**拿不到** Python Researcher 还未产出的结果——`asyncio.gather` 只是同时**发起**调用，不会让后发起的协程神奇地看到另一个还没跑完的协程的返回值。这种情况下，要么：

- 被迫改回顺序执行（Prompt Chaining 风格：先跑 Python，再把它的结果塞进 TypeScript 的输入里），
- 要么 TypeScript Researcher 会在缺少必要上下文的情况下运行，产出不完整或错误的比较（比如凭空猜测 Python 的结论）。

本实现刻意避免了这一点：每个 Researcher 的输入只有"语言名 + 原始问题（仅作背景）"，见 `_run_researcher()` 里拼的 prompt，任何一个 Researcher 都不引用另一个 Researcher 的输出，这才是三者能够被安全并行的前提。

### 情况 B：共享可变状态的竞争（本实现未触及，但值得说明）

如果三个"任务"不是各自独立的 LLM 调用，而是要读写同一份可变状态（比如同一个文件、同一个计数器、同一个数据库行），并发执行会引入竞态条件——谁先写、谁后写、覆盖了谁的结果是不确定的。`asyncio` 的单线程协作式调度不会自动帮你加锁；这需要额外的同步手段（锁、队列、不可变数据传递），而不是简单地套一层 `asyncio.gather`。本实现里三个 Researcher 之间互不共享任何可变对象（各自独立的 `Agent`、独立的输入字符串、独立的返回值），所以没有这个问题，但这是设计上刻意规避的，不是并行执行自动保证的。

**结论**：并行只对"彼此没有数据依赖、也不共享可变状态"的子任务安全有效。一旦存在真正的先后依赖，就应该换回 Prompt Chaining（顺序传递）或引入额外的同步机制，而不是继续用 `asyncio.gather` 硬凑并行。

## 5. Parallelization 和 Multi-Agent 是什么关系

"Multi-Agent" 是一个更宽泛的说法，泛指"系统里存在多个独立的 Agent"这件事本身；Parallelization 是一种**具体的多 Agent 编排方式**——它约束的是这些 Agent **如何被调度**，而不是"有没有多个 Agent"这件事。

- 本实现里确实有 4 个独立的 `Agent`（3 个 Researcher + 1 个 Synthesizer），所以它当然是"Multi-Agent"的一种；但反过来，"Multi-Agent" 不等于 "Parallelization"——Prompt Chaining（第 01 篇文档）里的 Planner/Researcher/Analyst/Writer 也是 4 个独立 Agent，却是顺序执行的，不是并行的；Routing（第 02 篇文档）里 Router + 3 个 Specialist 也是 Multi-Agent，但每次只运行其中一个。
- 换句话说：**Multi-Agent 描述的是"有几个 Agent"，Parallelization 描述的是"这些 Agent 之间的调度关系是不是同时发起、互不等待、结果最后合并"**。同一组 Agent，既可以按 Parallelization 的方式并发跑（像本实现），也可以按 Prompt Chaining 的方式顺序跑，还可以按 Routing 的方式只跑其中一个——这是编排策略的选择，不是 Agent 数量决定的。
- 本实现的 handoffs 检查（`test_no_agent_declares_handoffs`）也说明了一点：这里的"多 Agent"不依赖 SDK 的 `handoff` 机制（Agent 之间互相移交控制权），而是由 `run_parallelization()` 这个 Python 函数外部编排调用——是**代码在协调多个 Agent**，不是 Agent 之间在互相协调。

## 6. Parallelization 和 Orchestrator-Workers 有什么区别

两者拓扑结构看起来很像（都是"分派给多个下游 Agent，再汇总"），Anthropic 原文也明确指出这一点，但关键区别在于**子任务是谁决定的、什么时候决定的**：

| 维度 | Parallelization（本实现） | Orchestrator-Workers |
|---|---|---|
| 子任务数量和内容 | 写死在代码里（`LANGUAGES` 元组），对所有输入问题都一样：永远是"研究 Python / TypeScript / Go" | 由一个中央 LLM 在运行时根据具体输入动态决定，不同问题可能拆出完全不同数量、不同内容的子任务 |
| 谁分配任务 | 这份 `.py` 代码本身（`for language in LANGUAGES`） | Orchestrator Agent（一次 LLM 调用），决定"这次需要几个 worker、每个 worker 做什么” |
| 灵活性 | 低：换一个问题（比如"分析这个代码库需要改哪些文件"），子任务的形状不会变，因为形状是提前写死的 | 高：能处理"无法预先枚举子任务"的问题（比如原文的例子——一次代码改动可能涉及数量和位置都无法提前预知的文件） |
| 并发性 | 是核心特征——本实现的三个 Researcher 必须同时跑 | 不是必需特征——Orchestrator 分配的多个子任务既可以并发跑，也可以顺序跑，这由编排代码决定，不是 Orchestrator-Workers 这个模式本身规定的 |
| 适用场景（原文措辞） | "可分段的任务需要提速，或需要多个角度/多次尝试提高置信度" | "无法预测需要哪些子任务的复杂任务"（如代码改动涉及的文件数量和内容因任务而异） |

用一句话总结区别：**Parallelization 回答的是"已知的几件事，怎么同时做更快"；Orchestrator-Workers 回答的是"不知道具体要做哪几件事，先让一个模型想清楚要拆成什么，再去做"。** 本实现属于前者——"比较 Python/TypeScript/Go 这三个具体语言"是任务描述里已经给定的、固定的三件事，不需要任何模型去"想出"应该研究哪些语言。

## 7. 实际运行观察：Sequential latency vs Parallel latency

### 完整流程一次真实运行（3 个 Researcher 并发 + 1 个 Synthesizer）

用真实网关跑了一次完整流程（问题同前两篇文档），`TIMING` 输出的时间戳（相对进程内部时钟，单位秒）：

```text
- Go Researcher:         started=66971.668  ended=66993.523  duration=21.855s
- Python Researcher:     started=66971.665  ended=66993.538  duration=21.874s
- TypeScript Researcher: started=66971.667  ended=66997.886  duration=26.219s
- Synthesizer:           started=66997.886  ended=67013.545  duration=15.659s
- TOTAL wall-clock: 41.881s
```

三个 Researcher 的 `started_at` 几乎完全相同（相差不到 3 毫秒），且彼此的执行区间明显重叠（例如 Go 和 Python 几乎同时开始同时结束，TypeScript 稍慢但和另外两个的区间大部分重叠）——这直接证明了它们是并发执行，而不是排队顺序执行的。总耗时 41.881s ≈ 三者中最慢的 TypeScript（26.219s）+ Synthesizer（15.659s），而不是三者耗时相加。

### 专门对比 Sequential vs Parallel（同一批 Researcher，两种调度方式）

为了直接对比，另外用 `measure_sequential_vs_parallel()` 对同一个问题分别跑了一次"三个 Researcher 顺序执行"和一次"三个 Researcher 并发执行"（工具调用命中真实网络，两次调用不完全等价，但足以体现数量级差异）：

```text
=== SEQUENTIAL ===
- Python Researcher:     started=67035.933  ended=67042.937  duration=7.004s
- TypeScript Researcher: started=67042.937  ended=67047.767  duration=4.830s
- Go Researcher:         started=67047.767  ended=67052.573  duration=4.805s
- TOTAL wall-clock: 16.640s

=== PARALLEL ===
- Python Researcher:     started=67052.573  ended=67057.312  duration=4.739s
- Go Researcher:         started=67052.574  ended=67058.251  duration=5.677s
- TypeScript Researcher: started=67052.573  ended=67058.386  duration=5.813s
- TOTAL wall-clock: 5.814s
```

**Sequential latency ≈ 16.64s，Parallel latency ≈ 5.81s**（约 2.9 倍加速）。顺序执行时下一个 Researcher 的 `started_at` 精确等于上一个的 `ended_at`（比如 TypeScript 的 `started=67042.937` 正好等于 Python 的 `ended=67042.937`），验证了"顺序"确实是严格排队；并发执行时三者的 `started_at` 几乎相同（相差不到 1 毫秒），`TOTAL wall-clock` 约等于三者中最慢的那个（5.813s），而不是三者相加（4.739+5.677+5.813=16.229s）——这正是并行带来的延迟收益的直接证据，不是理论推测。
