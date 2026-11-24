"""Mock agent runner that simulates an LLM-backed agent executing multi-step tasks.

Produces deterministic fake outputs based on seeded inputs and records actual
trajectories in the same schema as golden trajectories.
"""

from __future__ import annotations

import hashlib
import random
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from agent_eval.models import GoldenTrajectory, TrajectoryStep


# ---------------------------------------------------------------------------
# Deterministic fake LLM helpers
# ---------------------------------------------------------------------------

# Pool of plausible actions an agent might take
_ACTIONS: List[str] = [
    "search",
    "lookup",
    "summarize",
    "calculate",
    "retrieve",
    "classify",
    "extract",
    "validate",
    "format",
    "respond",
]

# Pool of output sentence fragments to build fake outputs
_OUTPUT_FRAGMENTS: List[str] = [
    "The result indicates a positive outcome.",
    "Based on the retrieved data, the answer is confirmed.",
    "Analysis complete. No anomalies detected.",
    "The query returned 5 relevant documents.",
    "Computation finished with value 42.",
    "Classification label: category_A with confidence 0.91.",
    "Extracted 3 key entities from the input.",
    "Validation passed. Schema is correct.",
    "Formatted output ready for downstream consumption.",
    "Task response generated successfully.",
    "Search yielded the following top result: item_7.",
    "Lookup succeeded; record found in database.",
    "Summary produced: the document covers three main topics.",
    "No further steps required; goal achieved.",
    "Intermediate result stored for next step.",
]


def _seeded_rng(seed_str: str) -> random.Random:
    """Return a fresh Random instance seeded deterministically from a string."""
    seed_int = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2 ** 31)
    return random.Random(seed_int)


def _fake_action(rng: random.Random) -> str:
    """Pick a deterministic action from the pool."""
    return rng.choice(_ACTIONS)


def _fake_action_input(action: str, rng: random.Random) -> Dict[str, Any]:
    """Generate plausible action input based on the action name."""
    if action == "search":
        queries = ["agent behavior", "regression test", "golden trajectory", "eval metrics"]
        return {"query": rng.choice(queries), "top_k": rng.randint(3, 10)}
    if action == "lookup":
        return {"key": f"record_{rng.randint(100, 999)}"}
    if action == "calculate":
        return {"expression": f"{rng.randint(1, 100)} + {rng.randint(1, 100)}"}
    if action == "summarize":
        return {"max_tokens": rng.choice([64, 128, 256])}
    if action == "retrieve":
        return {"document_id": f"doc_{rng.randint(1, 50)}"}
    if action == "classify":
        return {"text": "sample input text", "labels": ["A", "B", "C"]}
    if action == "extract":
        return {"field": rng.choice(["date", "name", "amount"])}
    if action == "validate":
        return {"schema": "default_v1"}
    if action == "format":
        return {"template": "json"}
    # respond / fallback
    return {"message": "proceeding with default parameters"}


def _fake_output(rng: random.Random, num_sentences: int = 2) -> str:
    """Build a deterministic fake output from sentence fragments."""
    sentences = rng.choices(_OUTPUT_FRAGMENTS, k=num_sentences)
    return " ".join(sentences)


def _fake_latency_ms(rng: random.Random, base_ms: float = 150.0) -> float:
    """Return a deterministic fake latency value in milliseconds."""
    jitter = rng.uniform(-50.0, 200.0)
    return round(max(10.0, base_ms + jitter), 2)


# ---------------------------------------------------------------------------
# Step configuration
# ---------------------------------------------------------------------------

class StepConfig:
    """Optional per-step override for the mock runner."""

    def __init__(
        self,
        action: Optional[str] = None,
        action_input: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        self.action = action
        self.action_input = action_input
        self.output = output
        self.latency_ms = latency_ms


# ---------------------------------------------------------------------------
# MockAgentRunner
# ---------------------------------------------------------------------------

class MockAgentRunner:
    """Simulates an LLM-backed agent that executes multi-step tasks.

    All outputs are deterministic given the same (prompt, seed, num_steps).
    No external calls are made.

    Parameters
    ----------
    seed:
        Master seed used to derive per-step RNG seeds.
    base_latency_ms:
        Approximate latency per step before jitter is applied.
    """

    def __init__(
        self,
        seed: int = 42,
        base_latency_ms: float = 150.0,
    ) -> None:
        self.seed = seed
        self.base_latency_ms = base_latency_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        trajectory_id: str,
        task_description: str,
        prompt: str,
        num_steps: int = 3,
        step_configs: Optional[Sequence[StepConfig]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GoldenTrajectory:
        """Execute a mock multi-step task and return the recorded trajectory.

        Parameters
        ----------
        trajectory_id:
            Identifier to assign to the resulting trajectory.
        task_description:
            Human-readable description of the task.
        prompt:
            The agent prompt (used as part of the RNG seed for determinism).
        num_steps:
            How many steps the agent should execute.
        step_configs:
            Optional sequence of StepConfig objects for per-step overrides.
            Entries beyond the available configs fall back to generated values.
        tags:
            Optional tags to attach to the trajectory.
        metadata:
            Optional metadata dict to attach to the trajectory.

        Returns
        -------
        GoldenTrajectory
            Recorded trajectory with the same schema as golden trajectories.
        """
        steps: List[TrajectoryStep] = []
        step_configs = step_configs or []

        for step_index in range(num_steps):
            # Derive a unique, deterministic seed per step
            step_seed_str = f"{self.seed}:{prompt}:{step_index}"
            rng = _seeded_rng(step_seed_str)

            # Allow per-step overrides
            cfg: Optional[StepConfig] = (
                step_configs[step_index]
                if step_index < len(step_configs)
                else None
            )

            action = cfg.action if (cfg and cfg.action) else _fake_action(rng)
            action_input = (
                cfg.action_input
                if (cfg and cfg.action_input is not None)
                else _fake_action_input(action, rng)
            )
            output = (
                cfg.output
                if (cfg and cfg.output is not None)
                else _fake_output(rng)
            )
            latency_ms = (
                cfg.latency_ms
                if (cfg and cfg.latency_ms is not None)
                else _fake_latency_ms(rng, self.base_latency_ms)
            )

            step = TrajectoryStep(
                step_index=step_index,
                action=action,
                action_input=action_input,
                output=output,
                latency_ms=latency_ms,
                metadata={"mock": True, "step_seed": step_seed_str},
            )
            steps.append(step)

        return GoldenTrajectory(
            trajectory_id=trajectory_id,
            task_description=task_description,
            prompt=prompt,
            steps=steps,
            created_at=datetime.utcnow(),
            tags=tags or [],
            metadata={
                "runner": "MockAgentRunner",
                "runner_seed": self.seed,
                "num_steps": num_steps,
                **(metadata or {}),
            },
        )

    def run_from_golden(
        self,
        golden: GoldenTrajectory,
        run_id: Optional[str] = None,
        noise: bool = False,
        noise_probability: float = 0.3,
    ) -> GoldenTrajectory:
        """Re-run the agent using a golden trajectory's prompt and task.

        Useful for regression testing: run the same task through the mock runner
        and compare the resulting trajectory against the golden one.

        Parameters
        ----------
        golden:
            The reference golden trajectory.
        run_id:
            Optional suffix appended to the trajectory ID to distinguish
            the actual run from the golden.  Defaults to a short UUID.
        noise:
            If True, randomly perturb some steps to simulate realistic drift.
        noise_probability:
            Fraction of steps to perturb when ``noise=True``.

        Returns
        -------
        GoldenTrajectory
            Recorded actual trajectory (same schema as the golden).
        """
        run_id = run_id or uuid.uuid4().hex[:8]
        actual_id = f"{golden.trajectory_id}__run_{run_id}"

        actual = self.run(
            trajectory_id=actual_id,
            task_description=golden.task_description,
            prompt=golden.prompt,
            num_steps=len(golden.steps),
            tags=golden.tags,
            metadata={"source_golden_id": golden.trajectory_id, "run_id": run_id},
        )

        if noise:
            actual = self._apply_noise(actual, golden, noise_probability)

        return actual

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_noise(
        self,
        actual: GoldenTrajectory,
        golden: GoldenTrajectory,
        probability: float,
    ) -> GoldenTrajectory:
        """Randomly perturb steps of *actual* to simulate agent drift."""
        rng = _seeded_rng(f"noise:{self.seed}:{golden.trajectory_id}")
        noisy_steps: List[TrajectoryStep] = []

        for step in actual.steps:
            if rng.random() < probability:
                # Perturb: pick a different random action and output
                new_action = rng.choice([a for a in _ACTIONS if a != step.action])
                new_output = _fake_output(rng, num_sentences=1)
                new_latency = step.latency_ms + rng.uniform(50.0, 300.0)
                noisy_steps.append(
                    TrajectoryStep(
                        step_index=step.step_index,
                        action=new_action,
                        action_input=step.action_input,
                        output=new_output,
                        latency_ms=round(new_latency, 2),
                        metadata={**step.metadata, "noise_applied": True},
                    )
                )
            else:
                noisy_steps.append(step)

        return actual.model_copy(update={"steps": noisy_steps})
