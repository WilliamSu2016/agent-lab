# Minimal Research Agent Design

## Goal

Build a minimal research agent in Python without an agent framework. The agent receives a research question, decides whether web search is needed, calls a Python search function when appropriate, evaluates returned evidence, optionally searches again, and produces a grounded final answer.

Example question:

> Which is more suitable for developing AI agents: Python or TypeScript?

The implementation should use only:

- An LLM API client
- Python functions
- A manually implemented agent loop

It must not use LangGraph, OpenAI Agents SDK, CrewAI, AutoGen, or another agent framework.

## 1. Minimal Components

The smallest useful research agent has six components:

| Component | Responsibility |
|---|---|
| User input | Supplies the research question. |
| System instructions | Defines the agent's role, available tools, evidence requirements, and stop conditions. |
| LLM client | Sends the current conversation and tool definitions to the LLM API; receives either a tool call or an answer. |
| Tool registry | Maps an allowed tool name to a local Python function. Initially this contains only `web_search`. |
| Web search tool | Accepts a query and returns structured search results. |
| Agent loop | Repeatedly asks the LLM what to do, executes valid requested tools, records results, and stops at an answer or a safety limit. |

The agent is not the LLM alone. It is the combination of an LLM, tools, persistent state for one run, and the control loop that coordinates them.

## 2. Agent Loop

The loop is a small state machine. Each iteration gives the LLM the accumulated context and lets it choose the next action.

```text
initialize state with system instructions and user question

for iteration in 1..max_iterations:
    response = call LLM with messages, tool definitions, and tool-choice policy

    if response is a final answer:
        return final answer

    if response contains tool call(s):
        validate each tool call
        execute each valid tool
        append assistant tool-call message to state
        append each tool result message to state
        continue

    record protocol error
    return safe failure response

return safe response explaining that the research limit was reached
```

For the initial version, allow at most one tool call per LLM response. This simplifies execution order, state recording, observability, and error handling. Multiple calls can be added later if there is a clear need.

## 3. LLM and Tool Interaction

The LLM never directly runs Python code or accesses the web. It only emits a structured request declaring its intended tool and arguments. The Python application remains the authority that:

1. Exposes a limited list of tool schemas to the LLM.
2. Receives and validates the LLM's requested tool call.
3. Invokes the corresponding local Python function.
4. Converts the tool output into a structured tool-result message.
5. Sends that result back to the LLM on the next loop iteration.

The interaction for a research question is:

```text
User question
    -> LLM decides: search or answer
    -> if search: LLM requests web_search(query)
    -> Python validates and executes web_search
    -> Python returns structured search results to the LLM
    -> LLM decides: search again or write final answer
```

The system instructions should tell the LLM to:

- Use `web_search` for factual, current, disputed, or source-dependent claims.
- Form a focused query that addresses an unresolved part of the question.
- Treat search snippets as evidence, not unquestionable truth.
- Search again only when current evidence is insufficient, conflicting, or incomplete.
- Cite or name the sources used in the final answer when source metadata is available.
- Produce a final answer only after it has enough evidence for the claims it makes.

## 4. State to Preserve

State belongs to one agent run. It must be maintained by the Python application rather than relying on the LLM to remember prior steps.

| State field | Why it is needed |
|---|---|
| `run_id` | Correlates logs and makes a run traceable. |
| `question` | Preserves the original user request. |
| `messages` | Holds the ordered conversation: system, user, assistant tool requests, and tool results. This is the primary LLM context. |
| `iteration_count` | Enforces the maximum loop count. |
| `max_iterations` | Defines the run's safety budget. |
| `tool_calls` | Records requested tool name, arguments, result status, duration, and errors for observability. |
| `sources` | Stores normalized sources gathered across searches for deduplication and final-answer attribution. |
| `final_answer` | Stores the answer once the run completes successfully. |
| `status` | Indicates `running`, `completed`, `limit_reached`, or `failed`. |
| `errors` | Preserves recoverable and terminal failures. |

Do not place hidden application state exclusively in a natural-language message. Counters, statuses, error records, and tool-call metadata should remain structured Python data. Only information the LLM needs for reasoning must be serialized into conversation messages.

## 5. Tool-Call Data Structure

Use a provider-neutral internal representation. The LLM API adapter converts its native response format into this structure before tool execution.

```json
{
  "id": "call_001",
  "name": "web_search",
  "arguments": {
    "query": "Python vs TypeScript for AI agent development ecosystem 2026"
  }
}
```

The corresponding tool result should preserve the call identifier:

```json
{
  "tool_call_id": "call_001",
  "name": "web_search",
  "status": "success",
  "content": {
    "query": "Python vs TypeScript for AI agent development ecosystem 2026",
    "results": [
      {
        "title": "Example source title",
        "url": "https://example.com/article",
        "snippet": "Relevant excerpt returned by the search provider."
      }
    ]
  }
}
```

The initial `web_search` input schema should be deliberately narrow:

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "A focused web search query."
    }
  },
  "required": ["query"],
  "additionalProperties": false
}
```

Before invoking the Python function, validate all of the following:

- The requested name is exactly `web_search`.
- Arguments are valid JSON/object data.
- `query` exists, is a non-empty string, and fits a configured length limit.
- No unexpected arguments are accepted.
- The call identifier exists and is unique within the current response.

## 6. When to Continue the Loop

Continue when the LLM has requested a valid tool call and the tool execution has produced a result that should be returned to the LLM.

The LLM should request another search only if at least one condition holds:

- The first search did not return relevant evidence.
- Evidence is insufficient to answer all material parts of the question.
- Important sources conflict and a targeted search could resolve the conflict.
- A claim requires current information and the available sources are stale or unclear.
- The question naturally requires comparison across separate dimensions, and one dimension lacks evidence.

The next query should be more specific than the previous one. For example, after a broad ecosystem search, a follow-up might target runtime performance, library maturity, deployment constraints, or a primary source.

Do not continue merely because more information could exist. The agent should optimize for sufficient, relevant evidence within its bounded budget.

## 7. When to End

End successfully when the LLM returns a normal assistant answer without a tool call and the answer directly addresses the question using the gathered evidence.

Also end without another tool call when:

- The question can be answered safely from stable, non-source-dependent knowledge under the configured policy.
- Search results establish that reliable evidence is unavailable; the final answer should state the limitation.
- The user asks for an opinion or recommendation and the available evidence is sufficient to explain the trade-offs.

End with a controlled non-success status when:

- The maximum iteration limit is reached.
- The LLM response cannot be parsed as either a final answer or a valid tool call.
- A required external dependency is unavailable and retry policy is exhausted.
- A safety or policy rule rejects the requested action.

## 8. Maximum Loop Count

Set a small default maximum, such as `max_iterations = 5`. Each LLM response consumes one iteration whether it requests a tool or provides a final answer.

This cap prevents:

- Infinite loops caused by repeated searches.
- Excessive LLM and search-provider cost.
- Long response latency.
- Tool misuse caused by malformed or circular reasoning.

Use both a hard cap and a practical search cap. For example:

| Limit | Initial value | Purpose |
|---|---:|---|
| LLM iterations | 5 | Bounds total decision-making turns. |
| Successful searches | 3 | Prevents unnecessary repeated research. |
| Search results per call | 5 | Keeps context focused and bounded. |
| Query length | 300 characters | Limits malformed or overly broad inputs. |

When a cap is reached, do not fabricate a complete answer. Return the best bounded answer possible, clearly state that the research limit was reached, and identify any unresolved uncertainty.

## 9. Failure Handling

Failures must be explicit, structured, and bounded. Never silently discard a failed tool request or pretend a failed search succeeded.

| Failure | Handling |
|---|---|
| Unknown tool name | Do not execute it. Append a tool error result stating that the tool is unavailable, then allow the LLM to choose another action if budget remains. |
| Invalid tool arguments | Do not execute it. Return a validation error to the LLM with the expected argument shape. |
| Search timeout or temporary provider error | Retry at most once for transient failures. If it still fails, return a structured tool error to the LLM. |
| Empty search results | Return a successful result with an empty list. The LLM may refine its query or answer with a limitation. |
| Search result malformed | Treat it as a tool failure, record diagnostic details internally, and return a safe error summary to the LLM. |
| LLM API transient error | Retry with bounded exponential backoff. If exhausted, terminate with an explicit failure response. |
| LLM response malformed | Attempt no unsafe interpretation. Terminate with a protocol failure or make one bounded retry, depending on the API contract. |
| Repeated identical calls | Detect duplicate `name + normalized arguments`; return a duplicate-call error or stop when it indicates looping. |
| Iteration limit reached | Stop and return a transparent partial answer or controlled failure. |

Tool errors should be represented as tool results so the LLM can recover when possible:

```json
{
  "tool_call_id": "call_002",
  "name": "web_search",
  "status": "error",
  "error": {
    "code": "SEARCH_TIMEOUT",
    "message": "The web search provider timed out. Refine the query or answer with the available evidence."
  }
}
```

Keep internal diagnostics, such as stack traces, credentials, provider-specific payloads, and request headers, out of messages sent to the LLM and out of user-facing answers.

## 10. Minimal Project Structure

```text
.
├── docs/
│   └── AGENT-DESIGN.md
├── src/
│   ├── main.py          # Starts one research-agent run.
│   ├── agent.py         # Owns the explicit agent loop and state transitions.
│   ├── llm_client.py    # Adapts one LLM API to the application's internal format.
│   ├── tools.py         # Defines web_search and tool schemas.
│   ├── models.py        # Defines typed state, tool-call, and tool-result structures.
│   └── prompts.py       # Holds the system instructions.
├── tests/
│   ├── test_agent.py    # Tests loop transitions with fake LLM and tool responses.
│   └── test_tools.py    # Tests tool argument validation and error mapping.
├── .env.example         # Documents required environment variable names only.
├── .gitignore
├── pyproject.toml
└── README.md
```

Keep API-provider concerns isolated in `llm_client.py`. Keep tool-provider concerns isolated in `tools.py`. `agent.py` should not know HTTP details for either dependency; it should only orchestrate typed requests, results, state transitions, and stopping rules.

## Initial Scope Boundary

The first implementation should intentionally remain small:

- One user question per run.
- One tool: `web_search`.
- One LLM provider adapter.
- Sequential tool execution.
- In-memory state.
- Bounded retries and loop count.
- Plain-text final answer with source URLs or names when available.

Do not add planning agents, memory databases, background workers, browser automation, multi-agent coordination, or framework abstractions until the basic loop is observable, testable, and reliable.
