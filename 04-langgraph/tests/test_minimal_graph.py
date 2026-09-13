"""Tests for the minimal LangGraph graph (src/01_minimal_graph.py).

The module name starts with a digit, so it cannot be imported with a plain
``import`` statement; we load it dynamically with ``importlib``.
"""

import importlib
import unittest

minimal_graph = importlib.import_module("src.01_minimal_graph")

GraphState = minimal_graph.GraphState
greet = minimal_graph.greet
format_message = minimal_graph.format_message
build_graph = minimal_graph.build_graph


class NodeFunctionTest(unittest.TestCase):
    """Each node function should do exactly one thing, given a State dict."""

    def test_greet_builds_hello_message_from_name(self):
        state: GraphState = {"name": "World", "message": ""}
        result = greet(state)
        self.assertEqual(result["message"], "Hello, World!")

    def test_format_message_wraps_the_message(self):
        state: GraphState = {"name": "World", "message": "Hello, World!"}
        result = format_message(state)
        self.assertEqual(result["message"], ">>> Hello, World! <<<")


class GraphCompileAndInvokeTest(unittest.TestCase):
    """The compiled graph should run greet -> format in order via invoke()."""

    def test_build_graph_returns_a_compiled_graph_with_invoke(self):
        graph = build_graph()
        self.assertTrue(hasattr(graph, "invoke"))

    def test_invoke_runs_greet_then_format_and_returns_final_state(self):
        graph = build_graph()
        result = graph.invoke({"name": "Alice", "message": ""})

        self.assertEqual(result["name"], "Alice")
        self.assertEqual(result["message"], ">>> Hello, Alice! <<<")

    def test_invoke_with_a_different_name(self):
        graph = build_graph()
        result = graph.invoke({"name": "Bob", "message": ""})

        self.assertEqual(result["message"], ">>> Hello, Bob! <<<")


if __name__ == "__main__":
    unittest.main()
