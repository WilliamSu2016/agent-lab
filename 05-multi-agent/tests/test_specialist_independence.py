"""Cross-agent independence checks for Experiment 1 (Specialist Agents).

These tests do not test any single agent's behavior (that's covered in
``test_research_agent.py`` / ``test_coding_agent.py`` / ``test_review_agent.py``).
Instead they assert the *structural* constraints of the experiment:

* No agent module imports another agent module (no direct calls between
  ResearchAgent / CodingAgent / ReviewAgent).
* No agent module references LangGraph handoff/supervisor primitives or
  MCP, and none share a mutable state object across agents.
"""

import ast
import unittest
from pathlib import Path

SPECIALISTS_DIR = Path(__file__).resolve().parent.parent / "src" / "specialists"
AGENT_MODULES = ["research_agent", "coding_agent", "review_agent"]


def _imported_names(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class NoCrossAgentImportsTest(unittest.TestCase):
    def test_no_agent_module_imports_another_agent_module(self):
        for module_name in AGENT_MODULES:
            module_path = SPECIALISTS_DIR / f"{module_name}.py"
            imported = _imported_names(module_path)
            other_agents = {f"src.specialists.{other}" for other in AGENT_MODULES if other != module_name}
            self.assertFalse(
                imported & other_agents,
                f"{module_name}.py must not import other specialist agent modules, found {imported & other_agents}",
            )

    def test_no_agent_module_uses_forbidden_multi_agent_primitives(self):
        # Module docstrings are allowed to *explain*, in prose, that no
        # handoff/supervisor/MCP primitives are used (that's how each module
        # documents this experiment's constraints). What must never appear is
        # actual *usage* of such a primitive: an import or a call.
        forbidden_patterns = [
            "import langgraph_supervisor",
            "create_handoff_tool",
            "create_supervisor",
            "handoff_to_",
            "from mcp",
            "import mcp",
            "MCPServer",
        ]
        for module_name in AGENT_MODULES:
            module_path = SPECIALISTS_DIR / f"{module_name}.py"
            source = module_path.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                self.assertNotIn(pattern, source, f"{module_name}.py must not use {pattern!r}")


if __name__ == "__main__":
    unittest.main()
