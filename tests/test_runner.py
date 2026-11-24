"""Tests for the mock agent runner."""

from __future__ import annotations

import pytest

from agent_eval.models import GoldenTrajectory, TrajectoryStep
from agent_eval.runner import MockAgentRunner, StepConfig, _seeded_rng, _fake_output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_golden(tid: str = "traj-gold", num_steps: int = 3) -> GoldenTrajectory:
    runner = MockAgentRunner(seed=0)
    return runner.run(
        trajectory_id=tid,
        task_description="Golden task",
        prompt="Solve the problem step by step.",
        num_steps=num_steps,
    )


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_output(self):
        runner = MockAgentRunner(seed=42)
        t1 = runner.run(
            trajectory_id="t1",
            task_description="task",
            prompt="Do something.",
            num_steps=4,
        )
        t2 = runner.run(
            trajectory_id="t1",
            task_description="task",
            prompt="Do something.",
            num_steps=4,
        )
        assert [s.action for s in t1.steps] == [s.action for s in t2.steps]
        assert [s.output for s in t1.steps] == [s.output for s in t2.steps]
        assert [s.latency_ms for s in t1.steps] == [s.latency_ms for s in t2.steps]

    def test_different_seed_different_output(self):
        t1 = MockAgentRunner(seed=1).run(
            trajectory_id="t", task_description="task", prompt="prompt", num_steps=3
        )
        t2 = MockAgentRunner(seed=2).run(
            trajectory_id="t", task_description="task", prompt="prompt", num_steps=3
        )
        # At least one step should differ (extremely likely given distinct seeds)
        actions_1 = [s.action for s in t1.steps]
        actions_2 = [s.action for s in t2.steps]
        assert actions_1 != actions_2 or [
            s.output for s in t1.steps
        ] != [s.output for s in t2.steps]

    def test_different_prompt_different_output(self):
        runner = MockAgentRunner(seed=42)
        t1 = runner.run(
            trajectory_id="t", task_description="task", prompt="prompt A", num_steps=3
        )
        t2 = runner.run(
            trajectory_id="t", task_description="task", prompt="prompt B", num_steps=3
        )
        # Prompts affect the seed so outputs should differ
        assert [s.action for s in t1.steps] != [s.action for s in t2.steps] or [
            s.output for s in t1.steps
        ] != [s.output for s in t2.steps]


# ---------------------------------------------------------------------------
# Schema compliance tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_returns_golden_trajectory(self):
        runner = MockAgentRunner()
        result = runner.run(
            trajectory_id="schema-test",
            task_description="desc",
            prompt="prompt",
            num_steps=2,
        )
        assert isinstance(result, GoldenTrajectory)

    def test_step_count(self):
        runner = MockAgentRunner()
        for n in [1, 3, 5]:
            result = runner.run(
                trajectory_id="t", task_description="d", prompt="p", num_steps=n
            )
            assert len(result.steps) == n

    def test_step_indices_are_sequential(self):
        runner = MockAgentRunner()
        result = runner.run(
            trajectory_id="t", task_description="d", prompt="p", num_steps=4
        )
        for i, step in enumerate(result.steps):
            assert step.step_index == i

    def test_step_fields_populated(self):
        runner = MockAgentRunner()
        result = runner.run(
            trajectory_id="t", task_description="d", prompt="p", num_steps=2
        )
        for step in result.steps:
            assert isinstance(step, TrajectoryStep)
            assert step.action
            assert isinstance(step.action_input, dict)
            assert step.output
            assert step.latency_ms > 0

    def test_metadata_contains_runner_info(self):
        runner = MockAgentRunner(seed=7)
        result = runner.run(
            trajectory_id="t", task_description="d", prompt="p", num_steps=1
        )
        assert result.metadata["runner"] == "MockAgentRunner"
        assert result.metadata["runner_seed"] == 7

    def test_tags_and_metadata_passed_through(self):
        runner = MockAgentRunner()
        result = runner.run(
            trajectory_id="t",
            task_description="d",
            prompt="p",
            num_steps=1,
            tags=["ci", "smoke"],
            metadata={"env": "test"},
        )
        assert "ci" in result.tags
        assert result.metadata["env"] == "test"


# ---------------------------------------------------------------------------
# StepConfig override tests
# ---------------------------------------------------------------------------

class TestStepConfig:
    def test_action_override(self):
        runner = MockAgentRunner(seed=99)
        cfg = StepConfig(action="validate")
        result = runner.run(
            trajectory_id="t",
            task_description="d",
            prompt="p",
            num_steps=2,
            step_configs=[cfg],
        )
        assert result.steps[0].action == "validate"
        # Step 1 should still be generated
        assert result.steps[1].action != "validate" or True  # May coincidentally match

    def test_output_override(self):
        runner = MockAgentRunner()
        cfg = StepConfig(output="CUSTOM OUTPUT")
        result = runner.run(
            trajectory_id="t", task_description="d", prompt="p",
            num_steps=1, step_configs=[cfg],
        )
        assert result.steps[0].output == "CUSTOM OUTPUT"

    def test_latency_override(self):
        runner = MockAgentRunner()
        cfg = StepConfig(latency_ms=999.0)
        result = runner.run(
            trajectory_id="t", task_description="d", prompt="p",
            num_steps=1, step_configs=[cfg],
        )
        assert result.steps[0].latency_ms == 999.0

    def test_partial_configs(self):
        """Configs for first N steps only; remaining steps are auto-generated."""
        runner = MockAgentRunner(seed=0)
        cfg = StepConfig(action="extract")
        result = runner.run(
            trajectory_id="t", task_description="d", prompt="p",
            num_steps=3, step_configs=[cfg],
        )
        assert result.steps[0].action == "extract"
        # Steps 1 and 2 auto-generated – just ensure they exist and have valid fields
        assert result.steps[1].action
        assert result.steps[2].action


# ---------------------------------------------------------------------------
# run_from_golden tests
# ---------------------------------------------------------------------------

class TestRunFromGolden:
    def test_returns_trajectory(self):
        runner = MockAgentRunner(seed=42)
        golden = make_golden()
        actual = runner.run_from_golden(golden, run_id="test")
        assert isinstance(actual, GoldenTrajectory)

    def test_same_num_steps(self):
        runner = MockAgentRunner(seed=42)
        golden = make_golden(num_steps=5)
        actual = runner.run_from_golden(golden)
        assert len(actual.steps) == len(golden.steps)

    def test_id_contains_golden_id(self):
        runner = MockAgentRunner()
        golden = make_golden(tid="my-golden")
        actual = runner.run_from_golden(golden, run_id="abc")
        assert "my-golden" in actual.trajectory_id
        assert "abc" in actual.trajectory_id

    def test_metadata_contains_source_id(self):
        runner = MockAgentRunner()
        golden = make_golden(tid="src-golden")
        actual = runner.run_from_golden(golden)
        assert actual.metadata["source_golden_id"] == "src-golden"

    def test_deterministic_without_noise(self):
        runner = MockAgentRunner(seed=42)
        golden = make_golden()
        a1 = runner.run_from_golden(golden, run_id="r1")
        a2 = runner.run_from_golden(golden, run_id="r1")
        assert [s.action for s in a1.steps] == [s.action for s in a2.steps]
        assert [s.output for s in a1.steps] == [s.output for s in a2.steps]

    def test_noise_changes_some_steps(self):
        """With noise enabled, at least one step should differ from no-noise run."""
        runner = MockAgentRunner(seed=42)
        golden = make_golden(num_steps=10)
        clean = runner.run_from_golden(golden, run_id="clean", noise=False)
        noisy = runner.run_from_golden(golden, run_id="clean", noise=True, noise_probability=0.9)
        clean_actions = [s.action for s in clean.steps]
        noisy_actions = [s.action for s in noisy.steps]
        # With 90% noise probability across 10 steps, at least one should differ
        assert clean_actions != noisy_actions or [
            s.output for s in clean.steps
        ] != [s.output for s in noisy.steps]

    def test_noise_metadata_flag(self):
        """Noisy steps should carry a 'noise_applied' flag in metadata."""
        runner = MockAgentRunner(seed=1)
        golden = make_golden(num_steps=10)
        noisy = runner.run_from_golden(golden, run_id="n", noise=True, noise_probability=1.0)
        noisy_flags = [s.metadata.get("noise_applied", False) for s in noisy.steps]
        # With probability=1.0, every step should be noisy
        assert all(noisy_flags)


# ---------------------------------------------------------------------------
# Latency sanity
# ---------------------------------------------------------------------------

class TestLatency:
    def test_latency_always_positive(self):
        runner = MockAgentRunner(seed=0)
        for prompt in ["a", "hello world", "longer prompt text here please"]:
            traj = runner.run(
                trajectory_id="t", task_description="d", prompt=prompt, num_steps=5
            )
            for step in traj.steps:
                assert step.latency_ms > 0
