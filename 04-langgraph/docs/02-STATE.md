# 02 - 深入理解 State

对应代码：`src/02_state.py`
对应测试：`tests/test_state.py`

依然**不使用 LLM、不使用 Agent、不使用 Tool、不使用数据库、不使用全局变量**，
只用来加深对 LangGraph State 的理解。

流程：

```
START
  ↓
research      -- 只负责写 research_notes
  ↓
analysis      -- 只负责写 analysis
  ↓
finalize      -- 只负责写 final_answer
  ↓
END
```

## State

```python
class ResearchState(TypedDict):
    question: str
    research_notes: list[str]
    analysis: str
    final_answer: str
```

- `question`：调用方在初始 State 中提供，全程只读，没有任何 Node 修改它。
- `research_notes`：由 `research` 独占写入。
- `analysis`：由 `analysis` 独占写入。
- `final_answer`：由 `finalize` 独占写入。

## 三个 Node，各自只负责自己的字段

```python
def research(state: ResearchState) -> ResearchState:
    notes = [...]                 # 从 state["question"] 派生
    return {"research_notes": notes}          # 只返回自己负责的字段

def analysis(state: ResearchState) -> ResearchState:
    summary = ...                 # 从 state["research_notes"] 派生
    return {"analysis": summary}              # 只返回自己负责的字段

def finalize(state: ResearchState) -> ResearchState:
    return {"final_answer": f"Final answer -> {state['analysis']}"}  # 只返回自己负责的字段
```

每个 Node：

1. 只**读**它需要的字段（`research` 读 `question`；`analysis` 读
   `research_notes`；`finalize` 读 `analysis`）。
2. 只**写**它自己负责的字段，返回值是一个只包含该字段的局部更新（partial
   update），而不是整个 State。
3. 不修改、也不需要知道其它字段的存在。

LangGraph 在每个 Node 执行后，会把这个局部更新 `merge`（合并）进主 State，
再把合并后的完整 State 传给下一个 Node。

## 执行前后打印 State

`_print_state()` 是一个纯函数（只读 State、打印，不修改也不存储），在每个
Node 执行前后各调用一次，展示 State 的变化：

```
[research: before]  {'question': '...', 'research_notes': [], 'analysis': '', 'final_answer': ''}
[research: after]   {'question': '...', 'research_notes': [...], 'analysis': '', 'final_answer': ''}
[analysis: before]  {...}
[analysis: after]   {..., 'analysis': '...'}
[finalize: before]  {...}
[finalize: after]   {..., 'final_answer': '...'}
```

可以清楚看到：每一步之后，只有该 Node 负责的那个字段发生了变化，其余字段
原封不动地被带到下一步。

## 运行 & 测试

```powershell
.\.venv\Scripts\python.exe -m src.02_state
.\.venv\Scripts\python.exe -m pytest tests\test_state.py -v
```

---

## 重点问题解答

### 1. State 为什么是 LangGraph 的核心？

LangGraph 的本质是"一组 Node 按 Edge 定义的顺序/条件执行"，而 Node 之间**唯一**
的通信方式就是 State——Node 不能直接调用另一个 Node，也不能直接读取另一个
Node 的局部变量，它们只能通过读写共享的 State 来协作。

State 因此扮演了三个角色：

- **数据总线**：所有 Node 之间传递数据的唯一通道。
- **执行快照**：Graph 在任意时刻的进度都可以用"当前 State 长什么样"来完整
  描述，这也是 LangGraph 支持 checkpoint / 持久化 / 时间旅行（time travel）
  调试的基础。
- **契约（Schema）**：`TypedDict`/`BaseModel` 明确定义了每个字段的类型和
  含义，Node 之间的输入输出边界是显式、可检查的，而不是靠约定俗成。

没有 State，Graph 就只是一堆互相不知道彼此存在的函数；有了 State，这些函数
才能组成一个有明确数据流的"流水线"。

### 2. Node 为什么不应该直接依赖全局变量？

如果 `research`/`analysis`/`finalize` 通过读写模块级全局变量来传递数据（而
不是通过 State 参数），会带来几个问题：

- **不可组合/不可并发**：多次 `invoke()`（尤其是并发调用同一张图，或者
  LangGraph 的并行分支）会共享同一个全局变量，互相覆盖，State 就不再是
  "每次调用独立"的了。
- **不可测试**：Node 的行为依赖调用之前全局变量处于什么状态，单元测试时
  必须先手动设置/重置全局变量，测试之间容易互相污染（本例的测试之所以能
  对 `research`/`analysis`/`finalize` 逐个独立调用并断言，正是因为它们
  只依赖传入的 `state` 参数）。
- **不可持久化/不可恢复**：LangGraph 的 checkpoint 机制只保存 State，如果
  关键数据藏在全局变量里，中断后恢复执行时这些数据就会丢失。
- **数据流不透明**：全局变量让"谁在什么时候写了什么"变得难以追踪；而
  State 是显式参数，每个 Node 的输入输出都能一目了然地看到。

Node 应该是一个**纯函数**：`new_state_update = node(state)`，所有依赖都通过
参数传入，所有输出都通过返回值传出。

### 3. State 与普通 Python function 参数有什么区别？

普通函数参数是**调用者与被调用者之间一次性、局部的数据传递**：参数值在函数
返回后就不复存在（除非显式返回），调用链上每一层都需要手动把数据透传给下
一层。

LangGraph 的 State 则是**跨越整个 Graph 生命周期、被框架自动管理和合并**的
共享结构：

| 对比点     | 普通函数参数                          | LangGraph State                              |
|------------|----------------------------------------|-----------------------------------------------|
| 传递方式   | 调用者手动一层层传递                   | 框架自动在 Node 之间传递/合并                  |
| 修改方式   | 函数内部局部变量，返回值需手动组合      | 每个 Node 返回局部更新，框架自动 merge 回 State |
| 生命周期   | 函数调用期间                            | 整个 Graph 执行期间（可持久化、可恢复）        |
| 可见范围   | 只有直接调用者可见                      | Graph 中所有 Node 共享同一份 State             |
| 结构约束   | 无强制 Schema                           | 有明确的 State 类型定义（TypedDict/BaseModel） |

简单说：函数参数是"手动接力"，State 是"公共黑板 + 框架自动维护"。

### 4. 如果两个 Node 同时修改同一个 State field，会发生什么？

默认情况下（本例中的字段都没有声明 Reducer），LangGraph 对每个字段使用
"**后者覆盖前者**"（last-write-wins）的合并策略：如果两个 Node 的返回值里
都包含同一个字段，按照它们被处理的顺序，后处理的那个值会覆盖先处理的那个
值，先写入的数据就丢失了。

- 在**顺序执行**（如本例的 `research -> analysis -> finalize`）中，这通常
  是"预期行为"：下游 Node 本来就是要基于/替换上游的值。
- 但如果是**并行分支**中的两个 Node 同时写同一个字段（比如两个并行任务都
  往 `research_notes` 这种"列表"字段里塞新笔记），默认覆盖行为往往不是我们
  想要的——最终结果只保留了最后合并的那一次写入，另一次的数据凭空消失，
  且具体谁先谁后往往是不确定的（取决于执行/合并顺序），这会导致难以调试的
  "数据丢失"问题。

本例特意让每个字段只被唯一一个 Node 写入，从设计上完全避免了这种冲突。

### 5. Reducer 是解决什么问题的？

Reducer 解决的正是上一个问题：**当同一个 State 字段可能被多次写入（尤其是
并行分支、或者需要"追加"而不是"替换"的场景）时，如何明确定义这些写入应该
如何合并**，而不是依赖默认的"覆盖"行为。

在 LangGraph 中，可以给字段附加一个 Reducer 函数（例如用
`Annotated[list[str], operator.add]`），这样多个 Node（或同一个 Node 被
多次调用）对该字段的返回值就不再是互相覆盖，而是按 Reducer 定义的方式合并——
最典型的例子就是聊天消息列表用 `add_messages`/`operator.add` 做"追加"而不是
"覆盖"。

换句话说：

- **没有 Reducer**：字段更新 = 替换（后者覆盖前者）。
- **有 Reducer**：字段更新 = 按自定义规则合并（例如追加、求和、去重合并等）。

本例中三个字段都是"单一 Node 独占写入、下游只读"，天然没有合并冲突，所以
没有使用任何 Reducer；但如果未来要让 `research` 变成"多个并行子任务各自
往 `research_notes` 追加笔记"，就必须给 `research_notes` 声明一个类似
`operator.add` 的 Reducer，否则并行写入会互相覆盖、丢失数据。
