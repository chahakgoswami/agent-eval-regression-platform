"""Tests for the trajectory comparator."""

from __future__ import annotations

import pytest

from agent_eval.comparator import (
    TrajectoryComparator,
    compare_steps,
    compare_trajectories,
    token_overlap_similarity,
)
from agent_eval.models import EvalResult, GoldenTrajectory, StepScore, TrajectoryStep
from agent_eval.store import ResultsStore


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_step(
    index: int = 0,
    action: str = "search",
    output: str = "the quick brown fox",
    latency_ms: float = 100.0,
) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=index,
        action=action,
        action_input={"q": "test"},
        output=output,
        latency_ms=latency_ms,
    )


def make_trajectory(
    tid: str = "traj-gold",
    steps: list[TrajectoryStep] | None = None,
) -> GoldenTrajectory:
    if steps is None:
        steps = [
            make_step(0, "search", "the quick brown fox", 100.0),
            make_step(1, "summarize", "summary result here", 200.0),
        ]
    return GoldenTrajectory(
        trajectory_id=tid,
        task_description="Test task",
        prompt="Do the task",
        steps=steps,
    )


# ---------------------------------------------------------------------------
# token_overlap_similarity
# ---------------------------------------------------------------------------

class TestTokenOverlapSimilarity:
    def test_identical_strings(self):
        assert token_overlap_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert token_overlap_similarity("alpha beta", "gamma delta") == 0.0

    def test_partial_overlap(self):
        sim = token_overlap_similarity("the cat sat", "the cat mat")
        # intersection={the,cat} union={the,cat,sat,mat} -> 2/4 = 0.5
        assert abs(sim - 0.5) < 1e-6

    def test_empty_strings(self):
        assert token_overlap_similarity("", "") == 1.0

    def test_one_empty(self):
        assert token_overlap_similarity("hello", "") == 0.0

    def test_case_insensitive(self):
        assert token_overlap_similarity("Hello World", "hello world") == 1.0

    def test_returns_float_in_range(self):
        sim = token_overlap_similarity("foo bar baz", "bar baz qux")
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# compare_steps
# ---------------------------------------------------------------------------

class TestCompareSteps:
    def test_perfect_match(self):
        golden = make_step(0, "search", "result text", 100.0)
        actual = make_step(0, "search", "result text", 100.0)
        score = compare_steps(golden, actual)
        assert score.action_match is True
        assert score.output_similarity == 1.0
        assert score.latency_delta_ms == 0.0

    def test_action_mismatch(self):
        golden = make_step(0, "search", "result", 100.0)
        actual = make_step(0, "lookup", "result", 100.0)
        score = compare_steps(golden, actual)
        assert score.action_match is False

    def test_output_similarity_partial(self):
        golden = make_step(0, "search", "the cat sat on the mat", 100.0)
        actual = make_step(0, "search", "the cat ran on the floor", 100.0)
        score = compare_steps(golden, actual)
        assert 0.0 < score.output_similarity < 1.0

    def test_latency_delta_positive(self):
        golden = make_step(0, "search", "x", 100.0)
        actual = make_step(0, "search", "x", 250.0)
        score = compare_steps(golden, actual)
        assert abs(score.latency_delta_ms - 150.0) < 1e-6

    def test_latency_delta_negative(self):
        golden = make_step(0, "search", "x", 200.0)
        actual = make_step(0, "search", "x", 50.0)
        score = compare_steps(golden, actual)
        assert abs(score.latency_delta_ms - (-150.0)) < 1e-6

    def test_step_index_preserved(self):
        golden = make_step(3, "search", "y", 100.0)
        actual = make_step(3, "search", "y", 100.0)
        score = compare_steps(golden, actual)
        assert score.step_index == 3

    def test_returns_step_score(self):
        score = compare_steps(make_step(), make_step())
        assert isinstance(score, StepScore)


# ---------------------------------------------------------------------------
# compare_trajectories
# ---------------------------------------------------------------------------

class TestCompareTrajectories:
    def test_perfect_match(self):
        golden = make_trajectory()
        actual = make_trajectory(tid="traj-actual")
        result = compare_trajectories(golden, actual, run_id="run-1")
        assert result.action_match_rate == 1.0
        assert result.avg_output_similarity == 1.0
        assert result.avg_latency_delta_ms == 0.0
        assert all(s.action_match for s in result.step_scores)

    def test_returns_eval_result(self):
        golden = make_trajectory()
        actual = make_trajectory(tid="a")
        result = compare_trajectories(golden, actual)
        assert isinstance(result, EvalResult)

    def test_step_count_matches_golden(self):
        golden = make_trajectory()
        actual = make_trajectory(tid="a")
        result = compare_trajectories(golden, actual)
        assert len(result.step_scores) == len(golden.steps)

    def test_action_mismatch_reduces_rate(self):
        golden_steps = [
            make_step(0, "search", "output", 100.0),
            make_step(1, "summarize", "output", 100.0),
        ]
        actual_steps = [
            make_step(0, "lookup", "output", 100.0),  # different action
            make_step(1, "summarize", "output", 100.0),
        ]
        golden = make_trajectory(steps=golden_steps)
        actual = make_trajectory(tid="a", steps=actual_steps)
        result = compare_trajectories(golden, actual)
        assert result.action_match_rate == 0.5

    def test_missing_actual_steps_penalised(self):
        """Actual trajectory with fewer steps should score 0 for missing ones."""
        golden_steps = [
            make_step(0, "search", "output", 100.0),
            make_step(1, "summarize", "output", 100.0),
        ]
        actual_steps = [
            make_step(0, "search", "output", 100.0),
            # step 1 missing
        ]
        golden = make_trajectory(steps=golden_steps)
        actual = make_trajectory(tid="a", steps=actual_steps)
        result = compare_trajectories(golden, actual)
        assert len(result.step_scores) == 2
        assert result.step_scores[1].action_match is False
        assert result.step_scores[1].output_similarity == 0.0

    def test_extra_actual_steps_ignored(self):
        """Actual trajectory with more steps than golden – extras are ignored."""
        golden_steps = [make_step(0, "search", "output", 100.0)]
        actual_steps = [
            make_step(0, "search", "output", 100.0),
            make_step(1, "lookup", "extra", 50.0),  # extra step
        ]
        golden = make_trajectory(steps=golden_steps)
        actual = make_trajectory(tid="a", steps=actual_steps)
        result = compare_trajectories(golden, actual)
        assert len(result.step_scores) == 1

    def test_run_id_assigned(self):
        result = compare_trajectories(
            make_trajectory(), make_trajectory(tid="a"), run_id="my-run"
        )
        assert result.run_id == "my-run"

    def test_run_id_auto_generated(self):
        result = compare_trajectories(make_trajectory(), make_trajectory(tid="a"))
        assert result.run_id  # non-empty

    def test_trajectory_id_matches_golden(self):
        golden = make_trajectory(tid="gold-123")
        actual = make_trajectory(tid="actual-456")
        result = compare_trajectories(golden, actual)
        assert result.trajectory_id == "gold-123"

    def test_metadata_contains_ids(self):
        golden = make_trajectory(tid="g")
        actual = make_trajectory(tid="a")
        result = compare_trajectories(golden, actual)
        assert result.metadata["golden_id"] == "g"
        assert result.metadata["actual_id"] == "a"

    def test_avg_output_similarity_aggregated(self):
        """Average similarity should be mean of per-step similarities."""
        golden_steps = [
            make_step(0, "search", "hello world", 100.0),
            make_step(1, "search", "completely different text here", 100.0),
        ]
        actual_steps = [
            make_step(0, "search", "hello world", 100.0),  # sim=1.0
            make_step(1, "search", "totally unrelated words now", 100.0),  # sim<1.0
        ]
        golden = make_trajectory(steps=golden_steps)
        actual = make_trajectory(tid="a", steps=actual_steps)
        result = compare_trajectories(golden, actual)
        expected_avg = (
            result.step_scores[0].output_similarity
            + result.step_scores[1].output_similarity
        ) / 2
        assert abs(result.avg_output_similarity - expected_avg) < 1e-6

    def test_empty_steps(self):
        golden = make_trajectory(steps=[])
        actual = make_trajectory(tid="a", steps=[])
        result = compare_trajectories(golden, actual)
        assert result.action_match_rate == 1.0
        assert result.avg_output_similarity == 1.0
        assert result.avg_latency_delta_ms == 0.0


# ---------------------------------------------------------------------------
# TrajectoryComparator (integration with ResultsStore)
# ---------------------------------------------------------------------------

class TestTrajectoryComparator:
    def test_compare_and_save_persists_result(self, tmp_path):
        comparator = TrajectoryComparator(store_dir=tmp_path)
        golden = make_trajectory()
        actual = make_trajectory(tid="actual")
        result = comparator.compare_and_save(golden, actual, run_id="r1")
        # Should be loadable from the store
        store = ResultsStore(store_dir=tmp_path)
        loaded = store.load(result.result_id)
        assert loaded.result_id == result.result_id
        assert loaded.trajectory_id == "traj-gold"

    def test_compare_and_save_returns_eval_result(self, tmp_path):
        comparator = TrajectoryComparator(store_dir=tmp_path)
        result = comparator.compare_and_save(
            make_trajectory(), make_trajectory(tid="a")
        )
        assert isinstance(result, EvalResult)

    def test_compare_many_and_save(self, tmp_path):
        comparator = TrajectoryComparator(store_dir=tmp_path)
        golden_a = make_trajectory(tid="gold-A")
        actual_a = make_trajectory(tid="actual-A")
        golden_b = make_trajectory(tid="gold-B")
        actual_b = make_trajectory(tid="actual-B")
        results = comparator.compare_many_and_save(
            [(golden_a, actual_a), (golden_b, actual_b)],
            run_id="batch-run",
        )
        assert len(results) == 2
        assert all(r.run_id == "batch-run" for r in results)

    def test_compare_many_persists_all(self, tmp_path):
        comparator = TrajectoryComparator(store_dir=tmp_path)
        pairs = [
            (make_trajectory(tid=f"gold-{i}"), make_trajectory(tid=f"actual-{i}"))
            for i in range(3)
        ]
        results = comparator.compare_many_and_save(pairs, run_id="r")
        store = ResultsStore(store_dir=tmp_path)
        all_stored = store.list_all()
        assert len(all_stored) == 3

    def test_explicit_result_id(self, tmp_path):
        comparator = TrajectoryComparator(store_dir=tmp_path)
        result = comparator.compare_and_save(
            make_trajectory(),
            make_trajectory(tid="a"),
            result_id="fixed-id-123",
        )
        assert result.result_id == "fixed-id-123"
        store = ResultsStore(store_dir=tmp_path)
        loaded = store.load("fixed-id-123")
        assert loaded.result_id == "fixed-id-123"

    def test_metadata_passed_through(self, tmp_path):
        comparator = TrajectoryComparator(store_dir=tmp_path)
        result = comparator.compare_and_save(
            make_trajectory(),
            make_trajectory(tid="a"),
            metadata={"env": "staging"},
        )
        assert result.metadata["env"] == "staging"
