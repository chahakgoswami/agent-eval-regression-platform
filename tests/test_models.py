"""Tests for core data models."""

from datetime import datetime

import pytest

from agent_eval.models import (
    EvalResult,
    GoldenTrajectory,
    StepScore,
    TrajectoryStep,
)


def make_step(index: int = 0) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=index,
        action="search",
        action_input={"query": "test"},
        output="some result",
        latency_ms=100.0,
    )


def make_trajectory(tid: str = "traj-001") -> GoldenTrajectory:
    return GoldenTrajectory(
        trajectory_id=tid,
        task_description="Test task",
        prompt="Do the test task",
        steps=[make_step(0), make_step(1)],
    )


class TestTrajectoryStep:
    def test_creation(self):
        step = make_step()
        assert step.step_index == 0
        assert step.action == "search"
        assert step.latency_ms == 100.0

    def test_defaults(self):
        step = make_step()
        assert step.action_input == {"query": "test"}
        assert step.metadata == {}


class TestGoldenTrajectory:
    def test_creation(self):
        traj = make_trajectory()
        assert traj.trajectory_id == "traj-001"
        assert len(traj.steps) == 2

    def test_serialization_roundtrip(self):
        traj = make_trajectory()
        json_str = traj.model_dump_json()
        restored = GoldenTrajectory.model_validate_json(json_str)
        assert restored.trajectory_id == traj.trajectory_id
        assert len(restored.steps) == len(traj.steps)

    def test_created_at_default(self):
        traj = make_trajectory()
        assert isinstance(traj.created_at, datetime)


class TestEvalResult:
    def test_creation(self):
        result = EvalResult(
            result_id="res-001",
            trajectory_id="traj-001",
            run_id="run-001",
            step_scores=[
                StepScore(
                    step_index=0,
                    action_match=True,
                    output_similarity=0.9,
                    latency_delta_ms=10.0,
                )
            ],
            action_match_rate=1.0,
            avg_output_similarity=0.9,
            avg_latency_delta_ms=10.0,
        )
        assert result.result_id == "res-001"
        assert result.passed is None
        assert result.action_match_rate == 1.0

    def test_step_score_bounds(self):
        with pytest.raises(Exception):
            StepScore(
                step_index=0,
                action_match=True,
                output_similarity=1.5,  # out of range
                latency_delta_ms=0.0,
            )
