# Pattern 5 — Evaluator-Optimizer

代码：`src/evaluator_optimizer.py`
测试：`tests/test_evaluator_optimizer.py`（18 个测试，全部通过）

## 1. 结构

```
Topic
  │
  ▼
Generator (LLM, 有 search_web 工具)  ──► Draft v1
  │
  ▼
Evaluator (LLM, 无工具)  ──► EvaluationResult { pass, score, feedback, missing_points }
  │
  ▼
代码判断: score >= PASS_SCORE(8) ?
  │                         │
  │ 否                       │ 是
  ▼                         ▼
Generator(v1 + feedback + missing_points) ──► Draft v2   停止，返回最佳版本
  │
  ▼
Evaluator ──► ... （最多 MAX_ITERATIONS=3 轮）
```

`Generator` 和 `Evaluator` 是两个独立的 `Agent`，彼此之间**没有** `handoffs`。是否再循环一轮、要不要停止、把哪个版本当作最终答案，全部由 `run_evaluator_optimizer()` 这段 Python 代码决定，不依赖 SDK 的任何自动机制。

## 2. 各角色职责

* **Generator**：唯一负责撰写/修订报告文本的角色，挂了 `search_web` 工具去查证据。它从不评价、也从不宣称自己的稿子"已经足够好"——那是 Evaluator 的职责。第二轮起，它会收到上一版草稿 + Evaluator 的 `feedback` + `missing_points`，instructions 明确要求"必须逐条处理，不能置之不理，也不能推倒重写"。
* **Evaluator**：唯一负责评价的角色，没有工具，`output_type=EvaluationResult`（结构化输出），**从不改写报告本身**——它的 instructions 明确写"Do not include any report text of your own"。评价标准直接对应用户给出的 6 条：是否回答问题、是否覆盖主要维度、是否有事实错误、是否有足够证据、是否结构清晰、是否遗漏内容。

关于 JSON 字段名：需求里要求返回 `{"pass": true/false, ...}`，但 `pass` 是 Python 保留字，不能直接作为属性名。`EvaluationResult` 用 `passed: bool = Field(alias="pass")` 实现——对模型和外部 JSON 而言，字段名仍然是 `"pass"`（`tests/test_evaluator_optimizer.py::EvaluationResultSchemaTest` 验证了这一点），只是 Python 代码里用 `passed` 访问。

## 3. Evaluator 是否真的改善了结果？（Q1）

**真实运行证据**（问题："Compare OpenAI Agents SDK, LangGraph, AutoGen/AG2, and CrewAI for building a production AI research agent, and recommend one."）：

```
Iteration 1: score=7, pass=False
Iteration 2: score=7, pass=False
Iteration 3: score=7, pass=False
STOPPED: max_iterations_reached | best iteration: 1 (score=7)
```

在这次真实运行里，Evaluator **提出了大量具体、可执行的反馈**（例如：给出精确版本号和评测日期、区分 AutoGen 与 AG2、加入定价和 TCO、加入决策权重矩阵、把"专家判断"和"可复现的量化评测"分开标注等），Generator 也确实根据反馈修订了内容（比如第 2 轮增加了打分矩阵和许可证章节，第 3 轮进一步拆分 AutoGen/AG2、补充成本模型）——**但最终打分并没有提高**，三轮都停在 7 分，report 因此判定为"最佳版本是第 1 轮"（`max()` 对并列最高分取第一个出现的），也就是**最简洁的那一版**。

这是一个诚实但重要的结论：Evaluator-Optimizer **不保证**分数会单调上升。它只保证"如果分数确实能被反馈改善，代码会持续把改进后的反馈喂给 Generator，并且始终记录并能取用目前为止分数最高的那一版"。是否真的改善，取决于：(a) Evaluator 给出的反馈是否具体可执行，(b) Generator 是否真的针对性修订而不是简单增加篇幅，(c) 打分标准本身是否足够稳定/可衡量（本例中 Evaluator 对"证据是否充分"这条要求极高，几乎要求可复现的量化评测，这本身很难在几轮修订内被完全满足）。

## 4. 如果 score 没有提高怎么办？（Q2）

这正是本次真实运行遇到的情况（7 → 7 → 7）。代码层面的处理方式：

1. **循环不会因为分数不涨而提前放弃或报错**——它仍然会跑满 `MAX_ITERATIONS` 轮（除非某一轮达到 `PASS_SCORE`），因为"这一轮没提高"不代表"下一轮改了也没用"。
2. **最终返回的不是最后一轮，而是历史最高分的那一轮**（`best = max(iterations, key=lambda r: r.evaluation.score)`）——`tests/test_evaluator_optimizer.py::BestIterationSelectionTest` 专门验证了"即使第 3 轮分数比第 2 轮低，也要返回第 2 轮"这一行为。本次真实运行中三轮同分，`max()` 对并列最高分保留第一个匹配项，所以返回了第 1 轮（也恰好是修订负担最小、最不容易"为了回应反馈而堆砌不必要复杂度"的一版）。
3. **`stopped_reason` 会诚实标注为 `"max_iterations_reached"`**，而不是伪装成"通过了"——调用方（`main()`、或任何上层业务代码）可以清楚地知道"这份报告并未达到发布门槛，是在预算耗尽后拿到的最好结果"，从而决定是否需要人工复核、放宽标准，或者换更强的模型/更多轮次重试。

## 5. Evaluator 自己判断错误怎么办？（Q3）

Evaluator 是一个 LLM，它自己给出的 `passed`（对应 JSON 的 `pass`）布尔值**本身也可能出错或者与它给出的 `score` 自相矛盾**——例如打了低分却说"pass: true"，或者打了高分却说"pass: false"。本模式的应对方式：

* **代码从不直接信任 `passed` 字段作为是否继续循环的依据**。真正决定"是否接受这一版"的，是 `run_evaluator_optimizer()` 里的一行代码：`accepted = evaluation.score >= pass_score`——`score` 是一个有范围约束（`ge=0, le=10`）的整数，比一个开放式布尔判断更适合作为程序可以稳定复算的唯一真相来源。
* **任何 `passed` 与 `accepted` 不一致的情况都会被显式记录**（`IterationRecord.score_pass_mismatch`），而不是被悄悄吞掉。`tests/test_evaluator_optimizer.py::ScorePassMismatchTest` 验证了两种矛盾场景：低分却标记 `pass:true`（不会被接受）、高分却标记 `pass:false`（依然会被接受，因为分数达标）。这让"Evaluator 自我判断和数值判分不一致"这种模型层面的错误变得**可观测、可审计**，而不是被静默地放大成一个错误的流程决策。
* Pydantic 的 `score: int = Field(ge=0, le=10)` 还提供了一层更基础的保护——如果 Evaluator 返回超出 0-10 范围的分数（比如幻觉出 15 分），结构化输出解析会直接失败并被包装成 `EvaluatorOptimizerStepError("evaluator-iteration-N", ...)`，而不是让一个明显不合理的数字进入判分逻辑。

换句话说：这一模式假设 Evaluator 的判断**可能出错**，因此把"接受/拒绝"这个关键决策收窄到一个范围受限、语义单一的数值字段上，并且始终保留可以人工复核的痕迹（`score_pass_mismatch`、完整的 `iterations` 历史），而不是把决策权完全交给模型自己的自然语言式自评。

## 6. 为什么必须设置最大 iteration？（Q4）

`MAX_ITERATIONS = 3` 是一个纯代码常量，不受 Evaluator 输出或 Prompt 内容影响。原因：

1. **没有收敛保证**：正如本次真实运行所示，分数完全可能长时间停滞不前（7/7/7）。如果没有硬上限，"分数不够就继续循环"这个条件在最坏情况下会导致无限循环——尤其是当 Evaluator 的标准本身极高、或者存在其无法被文本修订满足的诉求（比如"提供可复现的量化基准测试"，这不是靠改写文字就能做到的）。
2. **每一轮的真实成本是确定的**：每多循环一轮就多两次真实的模型调用（Generator 通常还会调用 `search_web`），对应真实的延迟和 token/调用成本——这是一个必须由开发者能完全把控上界的资源边界，不能寄希望于"模型迟早会给出满意的分数"。
3. **即使不设上限，收益也会递减**：从本次运行可以看到，Generator 确实认真回应了反馈（篇幅、结构和论证都在增加），但打分标准本身可能已经超出"文本修订"能解决的范畴（比如要求真实的基准数据），继续循环更多轮不一定能换来更高分数，反而持续消耗资源。
4. **必须保证"总能拿到一个结果"**：即使 3 轮都没达标，模块也不能"卡死不返回"——`stopped_reason="max_iterations_reached"` + 历史最高分版本，让上层调用方总能拿到一个明确、可用、且标注了"未完全通过质量门槛"的结果，而不是无限等待。

## 7. 这个 Pattern 与 Agent Loop 有什么关系？（Q5）

Anthropic 定义的 Autonomous Agent Loop 通常是：一个 Agent 自己决定"下一步做什么工具调用"、"什么时候算完成"，循环边界由模型自身判断，环境反馈直接喂回同一个 Agent 的上下文。

Evaluator-Optimizer 表面上很像一个"循环"，但与真正的 Agent Loop 有本质区别：

| | Autonomous Agent Loop | Evaluator-Optimizer（本实现） |
|---|---|---|
| 谁决定还要不要继续 | Agent 自己（同一个模型的自我判断） | **代码**（`score >= PASS_SCORE`，与 Evaluator 自己的 `passed` 判断相互独立） |
| 循环上限 | 通常也是代码里的 `max_turns`，但循环体内每一步"做什么"是模型自由决定的 | 循环体的结构固定为"生成→评价"两步，且执行者（Generator/Evaluator）在每一步都是各自独立、职责单一的调用，不存在一个模型自主决定"这一步该调用什么工具/该不该结束"的开放式决策空间 |
| 评价者与执行者 | 同一个 Agent 自己判断自己做得好不好（自我反思在同一上下文里发生） | Evaluator 是**独立的第二个 Agent**，看不到 Generator 的推理过程，只看最终产物——更接近"外部质检"而非"自我反思" |
| 失败可定位性 | 循环内部出错通常只知道"第几轮出错"，很难说是"决策"错还是"执行"错 | 每一步都有明确的 `step_name`（`generator-iteration-N` / `evaluator-iteration-N`），角色边界清晰 |

可以说，Evaluator-Optimizer 是"用一个受控的、职责分离的两阶段循环，去逼近 Agent Loop 里'自我评价、持续改进'这种能力"，但把"是否继续、循环多少次、谁说了算"这些控制权仍然留在代码里，而不是交给模型自身的自主判断——这也是它仍被归类为 **Workflow** 而非 Autonomous Agent 的核心原因（参见 `docs/PATTERN-TAXONOMY.md`）。

## 8. 它与普通 Evaluation（离线评测）有什么区别？（Q6）

"普通 Evaluation" 通常指：产出内容之后，**离线、旁路地**对结果打分/归档，用于监控质量、生成报表、决定是否需要人工复核或回归测试——评价结果不会自动反过来改变已经生成的内容，评价和生成是两个解耦的、时间上分离的阶段（生成在前，评价在"事后"）。

Evaluator-Optimizer 的关键区别在于：**评价结果被实时接入了生成过程本身**——`feedback` 和 `missing_points` 会被立即格式化进下一次 Generator 调用的输入里，驱动同一个任务在同一次运行内产出更好的版本，而不是只在事后被记录下来。也就是说：

* 普通 Evaluation：生成 → 评价 → 记录/上报（一次性，评价结果通常影响的是"未来某次"生成，比如用于后续微调或流程改进）。
* Evaluator-Optimizer：生成 → 评价 → **立即**反馈回同一个任务的下一次生成 → 再评价 → ...（评价结果直接影响"这一次"任务的最终产出，形成一个受限次数的闭环）。

两者并不互斥——实践中往往会把 Evaluator-Optimizer 产出的"最佳版本 + 完整迭代历史"再送入一个更广义的、旁路的评测/监控系统（比如离线评测某个 Pattern 在大量样本上的平均分数、平均迭代次数、达标率），后者才是真正意义上的"普通 Evaluation"。

## 9. 测试要点（`tests/test_evaluator_optimizer.py`，18 tests）

* Agent 定义：Generator 有 `search_web` 工具 + 纯文本输出；Evaluator 无工具 + `output_type=EvaluationResult`；两者均无 `handoffs`。
* `EvaluationResult` 的 JSON 别名（`pass` ↔ `passed`）正确解析；`score` 越界（<0 或 >10）被 Pydantic 拒绝。
* 达标即停：`score >= PASS_SCORE(8)` 时只跑 1 轮就停止，`stopped_reason="passed_threshold"`；`score` 恰好等于阈值也算通过。
* 硬上限：分数始终不达标时，精确跑满 `MAX_ITERATIONS=3` 轮后以 `"max_iterations_reached"` 结束。
* 最佳版本选择：分数序列 `[5, 7, 6]`（第 3 轮比第 2 轮低）时，最终返回的是第 2 轮（历史最高分），而不是最后一轮。
* 反馈确实传递：脚本化断言"第 2 次 Generator 调用的输入里必须包含第 1 轮的草稿原文和 Evaluator 的具体 feedback/missing_points 文本"。
* Evaluator 判断与分数矛盾：`pass:true` 但低分 → 不被接受，且 `score_pass_mismatch=True`；`pass:false` 但高分 → 仍被接受（因为分数达标），同样标记为 mismatch。
* 失败定位：Generator 调用失败 → `step_name == "generator-iteration-N"`；Evaluator 调用失败 → `step_name == "evaluator-iteration-N"`；空 topic 被拒绝。

## 10. 真实运行记录

```
.\.venv\Scripts\python.exe -m src.evaluator_optimizer "Compare OpenAI Agents SDK, LangGraph, AutoGen/AG2, and CrewAI for building a production AI research agent, and recommend one."
```

* Iteration 1: score=7（Evaluator 指出：过度依赖厂商文档、缺少版本/日期矩阵、缺少定价与 TCO、AutoGen 与 AG2 应分开评估等）
* Iteration 2: score=7（Generator 已经加入打分矩阵、许可证章节，但 Evaluator 指出打分矩阵本身不透明、仍缺乏可复现证据、遗漏开发者体验和生态治理风险等）
* Iteration 3: score=7（Generator 已进一步拆分 AutoGen/AG2、补充成本模型，但 Evaluator 仍要求精确版本号、可复现基准和更明确的决策阈值）
* `STOPPED: max_iterations_reached | best iteration: 1 (score=7)`——三轮打分持平，最终按代码规则返回并列最高分中最早出现的第 1 轮草稿。

这组真实数据直接印证了 Q1/Q2 的分析：Evaluator 给出的反馈是具体且被 Generator 认真回应的，但受限于"要求可复现的量化基准"这类难以通过单纯文本修订满足的评价标准，分数未能实际提升，管道也如预期地在触达硬上限后诚实地停下并说明原因，而不是无限重试或伪造一个"已通过"的结果。
