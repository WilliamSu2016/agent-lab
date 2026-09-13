"""Tests for the Parallelization pattern: three independent Researchers run
concurrently (via asyncio), then a Synthesizer combines their results.

These use ``agents.testing.ScriptedModel`` to replay deterministic model output --
no network calls and no API key are required. Because the three Researchers run
*concurrently* against one shared ``ScriptedModel`` instance, tests use
``ModelStep.respond`` (instead of positional scripted steps) so each Researcher gets
the right scripted answer regardless of the non-deterministic order in which
``asyncio`` actually resolves the three coroutines.
"""

import asyncio
import time
import unittest

from agents.testing import ModelStep, ScriptedModel, assistant_message
from agents.testing.model import UnconsumedModelSteps

from src.parallelization import (
    LANGUAGES,
    ParallelizationStepError,
    build_researcher,
    build_synthesizer,
    measure_sequential_vs_parallel,
    run_parallelization,
)

QUESTION = "Compare Python, TypeScript, and Go for building AI agents, and give a recommendation."

FINDINGS_JSON = {
    "Python": '{"language": "Python", "findings": "Largest AI/agent library ecosystem.", "sources": []}',
    "TypeScript": '{"language": "TypeScript", "findings": "Strong web tooling via Node.js.", "sources": []}',
    "Go": '{"language": "Go", "findings": "Native goroutines give strong concurrency.", "sources": []}',
}

SYNTHESIZER_TEXT = (
    "Recommendation: Python is the best default choice for building AI agents today, "
    "given its ecosystem lead, though Go offers better raw concurrency and TypeScript "
    "offers strong web integration."
)


def _order_independent_researcher_script() -> ScriptedModel:
    """One scripted step per Researcher, resolved by inspecting the call's input text.

    Using ``ModelStep.respond`` (rather than positional steps) makes the script
    correct regardless of which order ``asyncio.gather`` actually invokes the three
    concurrent ``Runner.run`` calls in.
    """

    def make_responder(language: str):
        def _respond(call):
            assert language in str(call.input), (
                f"expected the {language} Researcher's input to mention '{language}'"
            )
            return [assistant_message(FINDINGS_JSON[language])]

        return _respond

    return ScriptedModel(
        [ModelStep.respond(make_responder(language)) for language in LANGUAGES]
        + [[assistant_message(SYNTHESIZER_TEXT)]]
    )


class StepDefinitionTest(unittest.TestCase):
    """Requirement 1 & 3: each Researcher is independent and covers only one language."""

    def test_each_researcher_has_its_own_name_and_output_type(self):
        names = set()
        for language in LANGUAGES:
            agent = build_researcher("placeholder-model", language)
            self.assertIn(language, agent.name)
            self.assertEqual([tool.name for tool in agent.tools], ["search_web"])
            self.assertIsNotNone(agent.output_type)
            names.add(agent.name)
        self.assertEqual(len(names), 3, "each Researcher must have a distinct name")

    def test_researcher_instructions_only_mention_their_own_language(self):
        """Each Researcher's instructions must not reference the other two languages."""
        for language in LANGUAGES:
            agent = build_researcher("placeholder-model", language)
            other_languages = [lang for lang in LANGUAGES if lang != language]
            for other in other_languages:
                self.assertNotIn(
                    other,
                    agent.instructions,
                    f"{language} Researcher's instructions must not mention {other}",
                )

    def test_no_agent_declares_handoffs(self):
        for language in LANGUAGES:
            agent = build_researcher("placeholder-model", language)
            self.assertEqual(agent.handoffs, [], f"{agent.name} must not declare handoffs")
        synthesizer = build_synthesizer("placeholder-model")
        self.assertEqual(synthesizer.handoffs, [])

    def test_synthesizer_has_no_tools_and_plain_text_output(self):
        agent = build_synthesizer("placeholder-model")
        self.assertEqual(agent.tools, [])
        self.assertIsNone(agent.output_type)


class ConcurrentExecutionTest(unittest.TestCase):
    """Requirements 2 & 4: the three Researchers actually run concurrently via asyncio."""

    def test_full_pipeline_runs_three_researchers_then_one_synthesizer(self):
        model = _order_independent_researcher_script()

        result = asyncio.run(run_parallelization(QUESTION, model))

        self.assertEqual(
            {f.language for f in result.findings}, {"Python", "TypeScript", "Go"}
        )
        self.assertEqual(result.final_answer, SYNTHESIZER_TEXT)

        # Exactly 4 steps ran in total: 3 Researchers + 1 Synthesizer.
        timing_names = [t.name for t in result.timing.steps]
        self.assertEqual(len(timing_names), 4)
        self.assertIn("Synthesizer", timing_names)
        model.assert_complete()

    def test_researchers_actually_overlap_in_wall_clock_time(self):
        """Prove concurrency: three artificially slow Researchers finish in ~1 slow-call
        duration, not ~3, because asyncio.gather runs them at the same time."""
        delay_seconds = 0.2

        async def slow_responder_factory(language: str):
            async def _respond(call):
                await asyncio.sleep(delay_seconds)
                return [assistant_message(FINDINGS_JSON[language])]

            return _respond

        async def build_script() -> ScriptedModel:
            steps = []
            for language in LANGUAGES:
                responder = await slow_responder_factory(language)
                steps.append(ModelStep.respond(responder))
            steps.append([assistant_message(SYNTHESIZER_TEXT)])
            return ScriptedModel(steps)

        async def scenario():
            model = await build_script()
            started = time.perf_counter()
            result = await run_parallelization(QUESTION, model)
            elapsed = time.perf_counter() - started
            return result, elapsed

        result, elapsed = asyncio.run(scenario())

        # If the three Researchers ran sequentially, this would take >= 3 * delay_seconds
        # (0.6s) before even reaching the Synthesizer. Concurrent execution keeps the
        # three Researcher calls' wall-clock spans overlapping, so total time stays well
        # under that even after adding the Synthesizer's own call.
        self.assertLess(elapsed, delay_seconds * 3)

        researcher_timings = [t for t in result.timing.steps if t.name != "Synthesizer"]
        self.assertEqual(len(researcher_timings), 3)
        # Overlap check: the combined wall-clock span of all three Researchers is
        # shorter than the sum of their individual durations -- which is only possible
        # if their execution windows genuinely overlapped in time.
        starts = [t.started_at for t in researcher_timings]
        ends = [t.ended_at for t in researcher_timings]
        total_span = max(ends) - min(starts)
        sum_of_durations = sum(t.duration_seconds for t in researcher_timings)
        self.assertLess(total_span, sum_of_durations)


class FailureReportingTest(unittest.TestCase):
    def test_researcher_failure_is_reported_with_its_language(self):
        def make_responder(language: str):
            def _respond(call):
                if language == "Go":
                    return [assistant_message("not valid json")]
                return [assistant_message(FINDINGS_JSON[language])]

            return _respond

        model = ScriptedModel(
            [ModelStep.respond(make_responder(language)) for language in LANGUAGES]
        )

        with self.assertRaises(ParallelizationStepError) as ctx:
            asyncio.run(run_parallelization(QUESTION, model))

        self.assertEqual(ctx.exception.step_name, "researcher-go")

    def test_synthesizer_failure_is_reported_as_synthesizer_step(self):
        def make_responder(language: str):
            def _respond(call):
                return [assistant_message(FINDINGS_JSON[language])]

            return _respond

        # Script only the three Researcher steps; the Synthesizer's call then hits an
        # exhausted script and fails, which must be reported as the "synthesizer" step.
        model = ScriptedModel(
            [ModelStep.respond(make_responder(language)) for language in LANGUAGES]
        )
        with self.assertRaises(ParallelizationStepError) as ctx:
            asyncio.run(run_parallelization(QUESTION, model))

        self.assertEqual(ctx.exception.step_name, "synthesizer")

    def test_rejects_empty_question(self):
        model = ScriptedModel([[assistant_message(FINDINGS_JSON["Python"])]])
        with self.assertRaises(ValueError):
            asyncio.run(run_parallelization("   ", model))
        with self.assertRaises(UnconsumedModelSteps):
            model.assert_complete()


class SequentialVsParallelTimingTest(unittest.TestCase):
    """Requirement: measure/compare sequential latency vs. parallel latency."""

    def test_parallel_run_is_not_slower_than_sequential_run(self):
        delay_seconds = 0.15

        async def _respond(call, _delay=delay_seconds):
            await asyncio.sleep(_delay)
            language = next(lang for lang in LANGUAGES if lang in str(call.input))
            return [assistant_message(FINDINGS_JSON[language])]

        async def scenario():
            # measure_sequential_vs_parallel runs 3 Researchers sequentially, then
            # the same 3 concurrently -- 6 model calls in total against one model.
            model = ScriptedModel([ModelStep.respond(_respond) for _ in range(6)])
            return await measure_sequential_vs_parallel(QUESTION, model)

        sequential_timing, parallel_timing = asyncio.run(scenario())

        self.assertGreaterEqual(sequential_timing.wall_clock_seconds, delay_seconds * 3)
        self.assertLess(parallel_timing.wall_clock_seconds, sequential_timing.wall_clock_seconds)


if __name__ == "__main__":
    unittest.main()
