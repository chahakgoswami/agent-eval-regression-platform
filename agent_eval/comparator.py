"""Trajectory comparator that diffs actual vs golden trajectories step-by-step.

Scores on:
- action match (exact string equality)
- output similarity (token overlap / Jaccard on whitespace-tokenized outputs)
- latency delta (actual_ms - golden_ms)

Writes EvalResult objects to a ResultsStore.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from agent_eval.models import (
    EvalResult,
    GoldenTrajectory,
    StepScore,
    TrajectoryStep,
)
from agent_eval.store import ResultsStore, DEFAULT_STORE_DIR


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Tokenize text into a set of lowercase whitespace-split tokens."""
    return set(text.lower().split())


def token_overlap_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between token sets of two strings.

    Returns a float in [0.0, 1.0].  Returns 1.0 if both strings are empty.
    """
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    if not union:
        return 1.0
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Step-level comparison
# ---------------------------------------------------------------------------

def compare_steps(
    golden_step: TrajectoryStep,
    actual_step: TrajectoryStep,
) -> StepScore:
    """Compare a single actual step against its golden counterpart.

    Parameters
    ----------
    golden_step:
        The reference (golden) step.
    actual_step:
        The step produced by the agent under evaluation.

    Returns
    -------
    StepScore
        Scores for this step comparison.
    """
    action_match = golden_step.action == actual_step.action
    output_similarity = token_overlap_similarity(
        golden_step.output, actual_step.output
    )
    latency_delta_ms = actual_step.latency_ms - golden_step.latency_ms

    return StepScore(
        step_index=golden_step.step_index,
        action_match=action_match,
        output_similarity=round(output_similarity, 6),
        latency_delta_ms=round(latency_delta_ms, 4),
    )


# ---------------------------------------------------------------------------
# Trajectory-level comparison
# ---------------------------------------------------------------------------

def compare_trajectories(
    golden: GoldenTrajectory,
    actual: GoldenTrajectory,
    run_id: Optional[str] = None,
    result_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> EvalResult:
    """Diff an actual trajectory against a golden one, producing an EvalResult.

    Steps are paired by position (index).  If the actual trajectory has fewer
    steps than the golden, missing steps receive a score of 0 on similarity and
    action_match=False.  Extra actual steps beyond the golden length are ignored.

    Parameters
    ----------
    golden:
        The reference golden trajectory.
    actual:
        The trajectory produced by the agent run.
    run_id:
        Identifier for the eval run.  Defaults to a short UUID.
    result_id:
        Explicit result ID.  Defaults to a UUID.
    metadata:
        Extra metadata to attach to the EvalResult.

    Returns
    -------
    EvalResult
        Fully populated eval result with per-step scores and aggregate metrics.
    """
    run_id = run_id or uuid.uuid4().hex[:8]
    result_id = result_id or str(uuid.uuid4())

    step_scores: List[StepScore] = []
    num_golden_steps = len(golden.steps)

    for i, golden_step in enumerate(golden.steps):
        if i < len(actual.steps):
            actual_step = actual.steps[i]
        else:
            # Missing step – penalise fully
            actual_step = TrajectoryStep(
                step_index=golden_step.step_index,
                action="__missing__",
                action_input={},
                output="",
                latency_ms=0.0,
            )
        step_scores.append(compare_steps(golden_step, actual_step))

    # Aggregate metrics
    if step_scores:
        action_match_rate = sum(1 for s in step_scores if s.action_match) / len(step_scores)
        avg_output_similarity = sum(s.output_similarity for s in step_scores) / len(step_scores)
        avg_latency_delta_ms = sum(s.latency_delta_ms for s in step_scores) / len(step_scores)
    else:
        action_match_rate = 1.0
        avg_output_similarity = 1.0
        avg_latency_delta_ms = 0.0

    return EvalResult(
        result_id=result_id,
        trajectory_id=golden.trajectory_id,
        run_id=run_id,
        step_scores=step_scores,
        action_match_rate=round(action_match_rate, 6),
        avg_output_similarity=round(avg_output_similarity, 6),
        avg_latency_delta_ms=round(avg_latency_delta_ms, 4),
        evaluated_at=datetime.utcnow(),
        metadata={
            "golden_id": golden.trajectory_id,
            "actual_id": actual.trajectory_id,
            "num_golden_steps": num_golden_steps,
            "num_actual_steps": len(actual.steps),
            **(metadata or {}),
        },
    )


# ---------------------------------------------------------------------------
# High-level runner: compare and persist
# ---------------------------------------------------------------------------

class TrajectoryComparator:
    """Compares actual trajectories against golden ones and stores results.

    Parameters
    ----------
    store_dir:
        Directory for the ResultsStore.  Defaults to the global default.
    """

    def __init__(self, store_dir: Path = DEFAULT_STORE_DIR) -> None:
        self.results_store = ResultsStore(store_dir=Path(store_dir))

    def compare_and_save(
        self,
        golden: GoldenTrajectory,
        actual: GoldenTrajectory,
        run_id: Optional[str] = None,
        result_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """Compare trajectories, persist the result, and return it.

        Parameters
        ----------
        golden:
            Reference golden trajectory.
        actual:
            Trajectory from the agent run.
        run_id:
            Run identifier (auto-generated if omitted).
        result_id:
            Result ID (auto-generated if omitted).
        metadata:
            Extra metadata for the EvalResult.

        Returns
        -------
        EvalResult
            The persisted evaluation result.
        """
        result = compare_trajectories(
            golden=golden,
            actual=actual,
            run_id=run_id,
            result_id=result_id,
            metadata=metadata,
        )
        self.results_store.save(result)
        return result

    def compare_many_and_save(
        self,
        pairs: List[tuple[GoldenTrajectory, GoldenTrajectory]],
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EvalResult]:
        """Compare multiple (golden, actual) pairs under the same run_id.

        Parameters
        ----------
        pairs:
            List of (golden, actual) trajectory tuples.
        run_id:
            Shared run identifier (auto-generated if omitted).
        metadata:
            Shared extra metadata for all EvalResults.

        Returns
        -------
        List[EvalResult]
            All persisted evaluation results.
        """
        run_id = run_id or uuid.uuid4().hex[:8]
        results: List[EvalResult] = []
        for golden, actual in pairs:
            result = self.compare_and_save(
                golden=golden,
                actual=actual,
                run_id=run_id,
                metadata=metadata,
            )
            results.append(result)
        return results
