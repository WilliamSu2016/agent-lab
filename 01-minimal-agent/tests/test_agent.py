import unittest

from src.agent import run_agent


class FakeLLM:
    def __init__(self):
        self.responses = [
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_001",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "Python AI agent libraries"}',
                        },
                    }
                ],
            },
            {
                "content": "Python is suitable because its AI ecosystem is mature.",
                "tool_calls": [],
            },
        ]

    def complete(self, messages, tools):
        return self.responses.pop(0)


class AgentLoopTest(unittest.TestCase):
    def test_agent_executes_tool_then_returns_final_answer(self):
        searched_queries = []

        def fake_search(query):
            searched_queries.append(query)
            return {"query": query, "results": [{"title": "Source", "url": "https://example.com"}]}

        state = run_agent(
            "Which language is better for AI agents?",
            FakeLLM(),
            fake_search,
            max_turns=5,
        )

        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["turns"], 2)
        self.assertEqual(searched_queries, ["Python AI agent libraries"])
        self.assertIn("Python is suitable", state["final_answer"])

    def test_agent_returns_tool_error_for_non_object_arguments(self):
        class InvalidArgumentsLLM:
            def complete(self, messages, tools):
                return {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "function": {"name": "web_search", "arguments": "[]"},
                        }
                    ],
                }

        state = run_agent(
            "Find information",
            InvalidArgumentsLLM(),
            lambda query: {"query": query, "results": []},
            max_turns=1,
        )

        self.assertEqual(state["status"], "limit_reached")
        self.assertEqual(state["tool_calls"][0]["status"], "error")
        self.assertEqual(state["tool_calls"][0]["error"]["code"], "INVALID_ARGUMENTS")


if __name__ == "__main__":
    unittest.main()
