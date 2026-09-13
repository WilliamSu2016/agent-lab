"""LangGraph implementations of Anthropic's five "Building effective agents"
workflow patterns.

Each submodule implements exactly one pattern using ``langgraph.graph.StateGraph``:

- ``prompt_chaining``      -- fixed, sequential steps (planner -> researcher -> analyst -> writer)
- ``routing``               -- classify, then dispatch to exactly one specialist
- ``parallelization``       -- independent subtasks run concurrently, then synthesized
- ``orchestrator_workers``  -- a dynamic number of tasks, decided at run time, run concurrently
- ``evaluator_optimizer``   -- generate -> critique -> (maybe) regenerate, until good enough

See ``docs/06-PATTERNS-IN-LANGGRAPH.md`` for the full Pattern -> Node/Edge/State
mapping and for why LangGraph is a good fit for expressing these patterns.
"""
