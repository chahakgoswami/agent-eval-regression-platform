"""Tests for the local JSON-based trajectory store."""

import pytest

from agent_eval.models import GoldenTrajectory, TrajectoryStep
from agent_eval.store import TrajectoryStore, ResultsStore
from agent_eval.models import EvalResult, StepScore


def make_trajectory(tid: str = "traj-001") -> GoldenTrajectory:
    return GoldenTrajectory(
        trajectory_id=tid,
        task_description="Test task",
        prompt="Do the test task",
        steps=[
            TrajectoryStep(
                step_index=0,
                action="search",
                action_input={"query": "test"},
                output="some result",
                latency_ms=100.0,
            )
        ],
    )


def make_result(rid: str = "res-001", tid: str = "traj-001") -> EvalResult:
    return EvalResult(
        result_id=rid,
        trajectory_id=tid,
        run_id="run-001",
        step_scores=[
            StepScore(
                step_index=0,
                action_match=True,
                output_similarity=0.85,
                latency_delta_ms=5.0,
            )
        ],
        action_match_rate=1.0,
        avg_output_similarity=0.85,
        avg_latency_delta_ms=5.0,
    )


class TestTrajectoryStore:
    def test_save_and_load(self, tmp_path):
        store = TrajectoryStore(store_dir=tmp_path)
        traj = make_trajectory()
        store.save(traj)
        loaded = store.load("traj-001")
        assert loaded.trajectory_id == "traj-001"
        assert loaded.task_description == "Test task"

    def test_load_missing_raises(self, tmp_path):
        store = TrajectoryStore(store_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent")

    def test_list_all(self, tmp_path):
        store = TrajectoryStore(store_dir=tmp_path)
        store.save(make_trajectory("t1"))
        store.save(make_trajectory("t2"))
        all_trajs = store.list_all()
        ids = {t.trajectory_id for t in all_trajs}
        assert "t1" in ids
        assert "t2" in ids

    def test_delete(self, tmp_path):
        store = TrajectoryStore(store_dir=tmp_path)
        store.save(make_trajectory())
        assert store.exists("traj-001")
        deleted = store.delete("traj-001")
        assert deleted is True
        assert not store.exists("traj-001")

    def test_delete_nonexistent(self, tmp_path):
        store = TrajectoryStore(store_dir=tmp_path)
        assert store.delete("ghost") is False

    def test_exists(self, tmp_path):
        store = TrajectoryStore(store_dir=tmp_path)
        assert not store.exists("traj-001")
        store.save(make_trajectory())
        assert store.exists("traj-001")

    def test_roundtrip_steps(self, tmp_path):
        store = TrajectoryStore(store_dir=tmp_path)
        traj = make_trajectory()
        store.save(traj)
        loaded = store.load("traj-001")
        assert len(loaded.steps) == 1
        assert loaded.steps[0].action == "search"


class TestResultsStore:
    def test_save_and_load(self, tmp_path):
        store = ResultsStore(store_dir=tmp_path)
        result = make_result()
        store.save(result)
        loaded = store.load("res-001")
        assert loaded.result_id == "res-001"

    def test_load_missing_raises(self, tmp_path):
        store = ResultsStore(store_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent")

    def test_list_for_trajectory(self, tmp_path):
        store = ResultsStore(store_dir=tmp_path)
        store.save(make_result("r1", "traj-A"))
        store.save(make_result("r2", "traj-B"))
        store.save(make_result("r3", "traj-A"))
        results = store.list_for_trajectory("traj-A")
        assert len(results) == 2
        assert all(r.trajectory_id == "traj-A" for r in results)

    def test_list_for_run(self, tmp_path):
        store = ResultsStore(store_dir=tmp_path)
        r1 = make_result("r1")
        r1 = r1.model_copy(update={"run_id": "run-X"})
        r2 = make_result("r2")
        store.save(r1)
        store.save(r2)
        run_x = store.list_for_run("run-X")
        assert len(run_x) == 1
        assert run_x[0].result_id == "r1"
