"""Tests for the CI gate module."""

from __future__ import annotations

import pytest

from agent_eval.ci_gate import (
    CIGate,
    GateThresholds,
    MetricViolation,
    RunGateOutcome,
    evaluate_result,
    run_gate_for_run,
)
from agent_eval.models import EvalResult, StepScore
from agent_eval.store import ResultsStore


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_step_score(
    index: int = 0,
    action_match: bool = True,
    output_similarity: float = 0.9,
    latency_delta_ms: float = 10.0,
) -> StepScore:
    return StepScore(
        step_index=index,
        action_match=action_match,
        output_similarity=output_similarity,
        latency_delta_ms=latency_delta_ms,
    )


def make_result(
    result_id: str = "res-001",
    trajectory_id: str = "traj-001",
    run_id: str = "run-001",
    action_match_rate: float = 1.0,
    avg_output_similarity: float = 0.9,
    avg_latency_delta_ms: float = 10.0,
) -> EvalResult:
    return EvalResult(
        result_id=result_id,
        trajectory_id=trajectory_id,
        run_id=run_id,
        step_scores=[make_step_score()],
        action_match_rate=action_match_rate,
        avg_output_similarity=avg_output_similarity,
        avg_latency_delta_ms=avg_latency_delta_ms,
    )


DEFAULT_THRESHOLDS = GateThresholds(
    min_action_match_rate=0.8,
    min_avg_output_similarity=0.7,
    max_avg_latency_delta_ms=500.0,
)


# ---------------------------------------------------------------------------
# GateThresholds validation
# ---------------------------------------------------------------------------

class TestGateThresholds:
    def test_valid_defaults(self):
        t = GateThresholds()
        t.validate()  # should not raise

    def test_invalid_action_match_rate_too_high(self):
        t = GateThresholds(min_action_match_rate=1.5)
        with pytest.raises(ValueError, match="min_action_match_rate"):
            t.validate()

    def test_invalid_action_match_rate_negative(self):
        t = GateThresholds(min_action_match_rate=-0.1)
        with pytest.raises(ValueError, match="min_action_match_rate"):
            t.validate()

    def test_invalid_output_similarity(self):
        t = GateThresholds(min_avg_output_similarity=2.0)
        with pytest.raises(ValueError, match="min_avg_output_similarity"):
            t.validate()

    def test_none_latency_allowed(self):
        t = GateThresholds(max_avg_latency_delta_ms=None)
        t.validate()  # should not raise


# ---------------------------------------------------------------------------
# evaluate_result
# ---------------------------------------------------------------------------

class TestEvaluateResult:
    def test_all_pass(self):
        result = make_result(
            action_match_rate=1.0,
            avg_output_similarity=0.95,
            avg_latency_delta_ms=10.0,
        )
        detail = evaluate_result(result, DEFAULT_THRESHOLDS)
        assert detail.passed is True
        assert detail.violations == []

    def test_action_match_rate_violation(self):
        result = make_result(action_match_rate=0.5)
        detail = evaluate_result(result, DEFAULT_THRESHOLDS)
        assert detail.passed is False
        metrics = [v.metric for v in detail.violations]
        assert "action_match_rate" in metrics

    def test_output_similarity_violation(self):
        result = make_result(avg_output_similarity=0.3)
        detail = evaluate_result(result, DEFAULT_THRESHOLDS)
        assert detail.passed is False
        metrics = [v.metric for v in detail.violations]
        assert "avg_output_similarity" in metrics

    def test_latency_violation(self):
        result = make_result(avg_latency_delta_ms=1000.0)
        detail = evaluate_result(result, DEFAULT_THRESHOLDS)
        assert detail.passed is False
        metrics = [v.metric for v in detail.violations]
        assert "avg_latency_delta_ms" in metrics

    def test_multiple_violations(self):
        result = make_result(
            action_match_rate=0.1,
            avg_output_similarity=0.1,
            avg_latency_delta_ms=9999.0,
        )
        detail = evaluate_result(result, DEFAULT_THRESHOLDS)
        assert len(detail.violations) == 3

    def test_latency_disabled(self):
        thresholds = GateThresholds(max_avg_latency_delta_ms=None)
        result = make_result(avg_latency_delta_ms=999999.0)
        detail = evaluate_result(result, thresholds)
        metrics = [v.metric for v in detail.violations]
        assert "avg_latency_delta_ms" not in metrics

    def test_boundary_values_pass(self):
        """Exactly at the threshold should pass."""
        thresholds = GateThresholds(
            min_action_match_rate=0.8,
            min_avg_output_similarity=0.7,
            max_avg_latency_delta_ms=500.0,
        )
        result = make_result(
            action_match_rate=0.8,
            avg_output_similarity=0.7,
            avg_latency_delta_ms=500.0,
        )
        detail = evaluate_result(result, thresholds)
        assert detail.passed is True

    def test_violation_contains_description(self):
        result = make_result(action_match_rate=0.5)
        detail = evaluate_result(result, DEFAULT_THRESHOLDS)
        assert any("action_match_rate" in v.description for v in detail.violations)

    def test_detail_ids_match_result(self):
        result = make_result(result_id="xyz", trajectory_id="t-99", run_id="run-99")
        detail = evaluate_result(result, DEFAULT_THRESHOLDS)
        assert detail.result_id == "xyz"
        assert detail.trajectory_id == "t-99"
        assert detail.run_id == "run-99"


# ---------------------------------------------------------------------------
# CIGate.evaluate_run – all pass
# ---------------------------------------------------------------------------

class TestCIGateAllPass:
    def test_empty_results_passes(self, tmp_path):
        gate = CIGate(store_dir=tmp_path, require_confirmation=False)
        outcome = gate.evaluate_run(run_id="run-empty", results=[])
        assert outcome.exit_code == 0
        assert outcome.total == 0

    def test_single_passing_result(self, tmp_path):
        gate = CIGate(
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=False,
        )
        results = [make_result()]
        outcome = gate.evaluate_run(run_id="run-1", results=results)
        assert outcome.exit_code == 0
        assert outcome.passed == 1
        assert outcome.failed == 0
        assert outcome.confirmed_regression is None

    def test_multiple_passing_results(self, tmp_path):
        gate = CIGate(
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=False,
        )
        results = [
            make_result("r1", action_match_rate=1.0, avg_output_similarity=0.9),
            make_result("r2", action_match_rate=0.9, avg_output_similarity=0.8),
        ]
        outcome = gate.evaluate_run(run_id="run-multi", results=results)
        assert outcome.exit_code == 0
        assert outcome.passed == 2


# ---------------------------------------------------------------------------
# CIGate.evaluate_run – regression confirmed
# ---------------------------------------------------------------------------

class TestCIGateRegressionConfirmed:
    def test_failing_result_confirmed(self, tmp_path):
        gate = CIGate(
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=True,
            confirm_fn=lambda _prompt: True,  # human says yes
        )
        results = [make_result(action_match_rate=0.1)]
        outcome = gate.evaluate_run(run_id="run-fail", results=results)
        assert outcome.exit_code == 1
        assert outcome.failed == 1
        assert outcome.confirmed_regression is True

    def test_exit_code_1_on_confirmed_regression(self, tmp_path):
        gate = CIGate(
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=False,  # auto-confirm
        )
        results = [make_result(avg_output_similarity=0.1)]
        outcome = gate.evaluate_run(run_id="r", results=results)
        assert outcome.exit_code == 1

    def test_partial_failures_counted_correctly(self, tmp_path):
        gate = CIGate(
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=False,
        )
        results = [
            make_result("r1", action_match_rate=1.0, avg_output_similarity=0.9),  # pass
            make_result("r2", action_match_rate=0.1, avg_output_similarity=0.1),  # fail
        ]
        outcome = gate.evaluate_run(run_id="r", results=results)
        assert outcome.passed == 1
        assert outcome.failed == 1
        assert outcome.exit_code == 1


# ---------------------------------------------------------------------------
# CIGate.evaluate_run – regression denied by human
# ---------------------------------------------------------------------------

class TestCIGateRegressionDenied:
    def test_human_denies_regression(self, tmp_path):
        gate = CIGate(
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=True,
            confirm_fn=lambda _prompt: False,  # human says no
        )
        results = [make_result(action_match_rate=0.1)]
        outcome = gate.evaluate_run(run_id="run-denied", results=results)
        assert outcome.exit_code == 2
        assert outcome.confirmed_regression is False

    def test_denied_does_not_exit_1(self, tmp_path):
        gate = CIGate(
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=True,
            confirm_fn=lambda _: False,
        )
        outcome = gate.evaluate_run(
            run_id="r", results=[make_result(action_match_rate=0.0)]
        )
        assert outcome.exit_code != 1


# ---------------------------------------------------------------------------
# CIGate persists passed flags
# ---------------------------------------------------------------------------

class TestCIGatePersistence:
    def test_passed_flag_persisted_on_success(self, tmp_path):
        store = ResultsStore(store_dir=tmp_path)
        result = make_result()
        store.save(result)

        gate = CIGate(
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=False,
        )
        gate.evaluate_run(run_id="run-001", results=[result])

        loaded = store.load(result.result_id)
        assert loaded.passed is True

    def test_passed_flag_false_on_regression(self, tmp_path):
        store = ResultsStore(store_dir=tmp_path)
        result = make_result(action_match_rate=0.1)
        store.save(result)

        gate = CIGate(
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=False,
        )
        gate.evaluate_run(run_id="run-001", results=[result])

        loaded = store.load(result.result_id)
        assert loaded.passed is False

    def test_mixed_results_persisted_correctly(self, tmp_path):
        store = ResultsStore(store_dir=tmp_path)
        good = make_result("r1", action_match_rate=1.0, avg_output_similarity=0.9)
        bad = make_result("r2", action_match_rate=0.1, avg_output_similarity=0.1)
        store.save(good)
        store.save(bad)

        gate = CIGate(
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=False,
        )
        gate.evaluate_run(run_id="run-001", results=[good, bad])

        assert store.load("r1").passed is True
        assert store.load("r2").passed is False


# ---------------------------------------------------------------------------
# CIGate loads from store when results not passed explicitly
# ---------------------------------------------------------------------------

class TestCIGateLoadsFromStore:
    def test_loads_results_from_store(self, tmp_path):
        store = ResultsStore(store_dir=tmp_path)
        r = make_result(run_id="run-stored")
        store.save(r)

        gate = CIGate(
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=False,
        )
        outcome = gate.evaluate_run(run_id="run-stored")  # no explicit results
        assert outcome.total == 1

    def test_unknown_run_id_gives_empty(self, tmp_path):
        gate = CIGate(store_dir=tmp_path, require_confirmation=False)
        outcome = gate.evaluate_run(run_id="run-nonexistent")
        assert outcome.total == 0
        assert outcome.exit_code == 0


# ---------------------------------------------------------------------------
# run_gate_for_run helper
# ---------------------------------------------------------------------------

class TestRunGateForRun:
    def test_returns_0_on_pass(self, tmp_path):
        store = ResultsStore(store_dir=tmp_path)
        store.save(make_result(run_id="r"))

        import io
        exit_code = run_gate_for_run(
            run_id="r",
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=False,
            out=io.StringIO(),
        )
        assert exit_code == 0

    def test_returns_1_on_regression(self, tmp_path):
        store = ResultsStore(store_dir=tmp_path)
        store.save(make_result(run_id="r", action_match_rate=0.0))

        import io
        exit_code = run_gate_for_run(
            run_id="r",
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=False,
            out=io.StringIO(),
        )
        assert exit_code == 1

    def test_returns_2_on_denied(self, tmp_path):
        store = ResultsStore(store_dir=tmp_path)
        store.save(make_result(run_id="r", action_match_rate=0.0))

        import io
        exit_code = run_gate_for_run(
            run_id="r",
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=True,
            confirm_fn=lambda _: False,
            out=io.StringIO(),
        )
        assert exit_code == 2

    def test_summary_written_to_out(self, tmp_path):
        store = ResultsStore(store_dir=tmp_path)
        store.save(make_result(run_id="r"))

        import io
        buf = io.StringIO()
        run_gate_for_run(
            run_id="r",
            store_dir=tmp_path,
            thresholds=DEFAULT_THRESHOLDS,
            require_confirmation=False,
            out=buf,
        )
        summary = buf.getvalue()
        assert "CI Gate Summary" in summary
        assert "r" in summary
