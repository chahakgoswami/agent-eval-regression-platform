"""CI gate module for the Agent Eval and Regression Platform.

Reads EvalResults, applies configurable pass/fail thresholds per metric,
enforces human-in-the-loop confirmation before marking any run as a
regression, and exits with a non-zero code on failure.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, TextIO

from agent_eval.models import EvalResult
from agent_eval.store import ResultsStore, DEFAULT_STORE_DIR


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

@dataclass
class GateThresholds:
    """Configurable pass/fail thresholds for CI gate evaluation.

    All thresholds are *minimum* acceptable values except
    ``max_avg_latency_delta_ms`` which is a *maximum*.

    Parameters
    ----------
    min_action_match_rate:
        Minimum fraction of steps where the action must match (0-1).
    min_avg_output_similarity:
        Minimum average token-overlap similarity across steps (0-1).
    max_avg_latency_delta_ms:
        Maximum allowed average latency increase vs golden (ms).
        Use ``None`` to disable the latency check.
    """

    min_action_match_rate: float = 0.8
    min_avg_output_similarity: float = 0.7
    max_avg_latency_delta_ms: Optional[float] = 500.0

    def validate(self) -> None:
        """Raise ValueError if thresholds are logically invalid."""
        if not 0.0 <= self.min_action_match_rate <= 1.0:
            raise ValueError(
                "min_action_match_rate must be in [0.0, 1.0], "
                f"got {self.min_action_match_rate}"
            )
        if not 0.0 <= self.min_avg_output_similarity <= 1.0:
            raise ValueError(
                "min_avg_output_similarity must be in [0.0, 1.0], "
                f"got {self.min_avg_output_similarity}"
            )


# ---------------------------------------------------------------------------
# Per-result evaluation
# ---------------------------------------------------------------------------

@dataclass
class MetricViolation:
    """Describes a single threshold violation."""

    metric: str
    threshold: float
    actual: float
    description: str


@dataclass
class GateResultDetail:
    """Detailed gate evaluation for a single EvalResult."""

    result_id: str
    trajectory_id: str
    run_id: str
    passed: bool
    violations: List[MetricViolation] = field(default_factory=list)


def evaluate_result(
    result: EvalResult,
    thresholds: GateThresholds,
) -> GateResultDetail:
    """Check a single EvalResult against thresholds.

    Parameters
    ----------
    result:
        The eval result to check.
    thresholds:
        Pass/fail thresholds to apply.

    Returns
    -------
    GateResultDetail
        Detailed gate outcome for this result.
    """
    violations: List[MetricViolation] = []

    if result.action_match_rate < thresholds.min_action_match_rate:
        violations.append(
            MetricViolation(
                metric="action_match_rate",
                threshold=thresholds.min_action_match_rate,
                actual=result.action_match_rate,
                description=(
                    f"action_match_rate {result.action_match_rate:.3f} "
                    f"< threshold {thresholds.min_action_match_rate:.3f}"
                ),
            )
        )

    if result.avg_output_similarity < thresholds.min_avg_output_similarity:
        violations.append(
            MetricViolation(
                metric="avg_output_similarity",
                threshold=thresholds.min_avg_output_similarity,
                actual=result.avg_output_similarity,
                description=(
                    f"avg_output_similarity {result.avg_output_similarity:.3f} "
                    f"< threshold {thresholds.min_avg_output_similarity:.3f}"
                ),
            )
        )

    if (
        thresholds.max_avg_latency_delta_ms is not None
        and result.avg_latency_delta_ms > thresholds.max_avg_latency_delta_ms
    ):
        violations.append(
            MetricViolation(
                metric="avg_latency_delta_ms",
                threshold=thresholds.max_avg_latency_delta_ms,
                actual=result.avg_latency_delta_ms,
                description=(
                    f"avg_latency_delta_ms {result.avg_latency_delta_ms:.1f} "
                    f"> threshold {thresholds.max_avg_latency_delta_ms:.1f}"
                ),
            )
        )

    return GateResultDetail(
        result_id=result.result_id,
        trajectory_id=result.trajectory_id,
        run_id=result.run_id,
        passed=len(violations) == 0,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Human-in-the-loop confirmation
# ---------------------------------------------------------------------------

def _default_confirm(
    prompt: str,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
) -> bool:
    """Prompt the user for yes/no confirmation via stdin.

    Returns True if the user confirms (types 'y' or 'yes'), False otherwise.
    """
    stdout.write(prompt + " [y/N]: ")
    stdout.flush()
    try:
        answer = stdin.readline().strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    return answer in ("y", "yes")


# ---------------------------------------------------------------------------
# CIGate
# ---------------------------------------------------------------------------

@dataclass
class RunGateOutcome:
    """Aggregated outcome for all results in a run."""

    run_id: str
    total: int
    passed: int
    failed: int
    details: List[GateResultDetail]
    confirmed_regression: Optional[bool]  # None if no regression; True/False if confirmed/denied
    exit_code: int  # 0 = success, 1 = regression confirmed / no human, 2 = regression denied

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


class CIGate:
    """Reads EvalResults for a run, applies thresholds, and enforces HITL confirmation.

    Parameters
    ----------
    store_dir:
        Directory for the ResultsStore.
    thresholds:
        Pass/fail thresholds.  Defaults to conservative defaults.
    require_confirmation:
        If True (default), prompt a human before marking a run as a regression.
        Set to False in fully-automated contexts where you want silent failure.
    confirm_fn:
        Callable that implements the confirmation UI.  Defaults to stdin prompt.
        Replace in tests to avoid blocking.
    """

    def __init__(
        self,
        store_dir: Path = DEFAULT_STORE_DIR,
        thresholds: Optional[GateThresholds] = None,
        require_confirmation: bool = True,
        confirm_fn: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.results_store = ResultsStore(store_dir=Path(store_dir))
        self.thresholds = thresholds or GateThresholds()
        self.thresholds.validate()
        self.require_confirmation = require_confirmation
        self._confirm_fn: Callable[[str], bool] = confirm_fn or _default_confirm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_run(
        self,
        run_id: str,
        results: Optional[List[EvalResult]] = None,
    ) -> RunGateOutcome:
        """Evaluate all results for a run and return the gate outcome.

        Parameters
        ----------
        run_id:
            The run to evaluate.
        results:
            Explicit list of EvalResult objects.  If omitted, results are
            loaded from the store by run_id.

        Returns
        -------
        RunGateOutcome
            Aggregate outcome including per-result details and exit code.
        """
        if results is None:
            results = self.results_store.list_for_run(run_id)

        if not results:
            # Nothing to evaluate → pass with exit code 0
            return RunGateOutcome(
                run_id=run_id,
                total=0,
                passed=0,
                failed=0,
                details=[],
                confirmed_regression=None,
                exit_code=0,
            )

        details: List[GateResultDetail] = [
            evaluate_result(r, self.thresholds) for r in results
        ]

        passed_count = sum(1 for d in details if d.passed)
        failed_count = len(details) - passed_count

        if failed_count == 0:
            # All results pass – update passed flag and return success
            self._persist_passed_flags(results, passed=True)
            return RunGateOutcome(
                run_id=run_id,
                total=len(details),
                passed=passed_count,
                failed=0,
                details=details,
                confirmed_regression=None,
                exit_code=0,
            )

        # There are failures – ask for human confirmation
        confirmed: Optional[bool] = None
        if self.require_confirmation:
            summary_lines = self._build_regression_summary(run_id, details)
            prompt = "\n".join(summary_lines)
            prompt += "\nMark this run as a REGRESSION and fail the CI gate?"
            confirmed = self._confirm_fn(prompt)
        else:
            # Automated mode: no prompt, treat as confirmed regression
            confirmed = True

        if confirmed:
            self._persist_passed_flags(results, passed=False, details=details)
            exit_code = 1
        else:
            # Human denied – do not mark as regression, but warn
            exit_code = 2

        return RunGateOutcome(
            run_id=run_id,
            total=len(details),
            passed=passed_count,
            failed=failed_count,
            details=details,
            confirmed_regression=confirmed,
            exit_code=exit_code,
        )

    def evaluate_results(
        self,
        results: List[EvalResult],
        run_id: str = "adhoc",
    ) -> RunGateOutcome:
        """Evaluate an explicit list of EvalResults (convenience method)."""
        return self.evaluate_run(run_id=run_id, results=results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_regression_summary(
        self,
        run_id: str,
        details: List[GateResultDetail],
    ) -> List[str]:
        """Build a human-readable summary of failing results."""
        lines: List[str] = [
            "",
            "=" * 70,
            f"  CI GATE: REGRESSION DETECTED  (run_id={run_id})",
            "=" * 70,
        ]
        failed = [d for d in details if not d.passed]
        lines.append(
            f"  {len(failed)} of {len(details)} trajectories FAILED gate thresholds:"
        )
        lines.append("")
        for d in failed:
            lines.append(f"  Trajectory: {d.trajectory_id}  (result: {d.result_id})")
            for v in d.violations:
                lines.append(f"    ✗ {v.description}")
        lines.append("")
        lines.append("  Thresholds applied:")
        lines.append(
            f"    min_action_match_rate      = {self.thresholds.min_action_match_rate}"
        )
        lines.append(
            f"    min_avg_output_similarity  = {self.thresholds.min_avg_output_similarity}"
        )
        if self.thresholds.max_avg_latency_delta_ms is not None:
            lines.append(
                f"    max_avg_latency_delta_ms   = {self.thresholds.max_avg_latency_delta_ms}"
            )
        else:
            lines.append("    max_avg_latency_delta_ms   = (disabled)")
        lines.append("=" * 70)
        return lines

    def _persist_passed_flags(
        self,
        results: List[EvalResult],
        passed: bool,
        details: Optional[List[GateResultDetail]] = None,
    ) -> None:
        """Update the ``passed`` field on each EvalResult and re-save it."""
        detail_map: Dict[str, GateResultDetail] = (
            {d.result_id: d for d in details} if details else {}
        )
        for result in results:
            # For mixed runs, determine per-result pass/fail
            if detail_map:
                result_passed = detail_map.get(result.result_id, None)
                new_passed = result_passed.passed if result_passed else passed
            else:
                new_passed = passed
            updated = result.model_copy(update={"passed": new_passed})
            self.results_store.save(updated)


# ---------------------------------------------------------------------------
# Standalone entry point (for use from GitHub Actions)
# ---------------------------------------------------------------------------

def run_gate_for_run(
    run_id: str,
    store_dir: Path = DEFAULT_STORE_DIR,
    thresholds: Optional[GateThresholds] = None,
    require_confirmation: bool = True,
    confirm_fn: Optional[Callable[[str], bool]] = None,
    out: TextIO = sys.stdout,
) -> int:
    """High-level helper: evaluate a run and return the exit code.

    Prints a summary to *out* and returns an integer exit code suitable for
    ``sys.exit()`` or GitHub Actions step failure detection.

    Parameters
    ----------
    run_id:
        Run to evaluate.
    store_dir:
        Path to the results store.
    thresholds:
        Gate thresholds (defaults if omitted).
    require_confirmation:
        Whether to prompt for HITL confirmation on regression.
    confirm_fn:
        Custom confirm callable (useful in tests).
    out:
        Output stream for the summary (defaults to stdout).

    Returns
    -------
    int
        0  – all results passed
        1  – regression confirmed (or auto-fail)
        2  – regression detected but human denied
    """
    gate = CIGate(
        store_dir=store_dir,
        thresholds=thresholds,
        require_confirmation=require_confirmation,
        confirm_fn=confirm_fn,
    )
    outcome = gate.evaluate_run(run_id=run_id)

    # Print summary
    out.write(f"\nCI Gate Summary  (run_id={run_id})\n")
    out.write(f"  Total results : {outcome.total}\n")
    out.write(f"  Passed        : {outcome.passed}\n")
    out.write(f"  Failed        : {outcome.failed}\n")

    if outcome.exit_code == 0:
        out.write("  Result        : PASS ✓\n")
    elif outcome.exit_code == 1:
        out.write("  Result        : FAIL ✗ (regression confirmed)\n")
    else:
        out.write("  Result        : WARN (regression detected but NOT confirmed)\n")

    return outcome.exit_code
