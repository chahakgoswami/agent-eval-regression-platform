"""Local JSON-based store for golden trajectories and eval results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from agent_eval.models import EvalResult, GoldenTrajectory


DEFAULT_STORE_DIR = Path(".agent_eval_store")
GOLDEN_SUBDIR = "golden"
RESULTS_SUBDIR = "results"


class TrajectoryStore:
    """Saves and loads GoldenTrajectory objects from a local JSON store."""

    def __init__(self, store_dir: Path = DEFAULT_STORE_DIR) -> None:
        self.store_dir = Path(store_dir)
        self.golden_dir = self.store_dir / GOLDEN_SUBDIR
        self.golden_dir.mkdir(parents=True, exist_ok=True)

    def save(self, trajectory: GoldenTrajectory) -> Path:
        """Persist a golden trajectory to disk. Returns the file path."""
        file_path = self.golden_dir / f"{trajectory.trajectory_id}.json"
        file_path.write_text(
            trajectory.model_dump_json(indent=2), encoding="utf-8"
        )
        return file_path

    def load(self, trajectory_id: str) -> GoldenTrajectory:
        """Load a golden trajectory by ID. Raises FileNotFoundError if missing."""
        file_path = self.golden_dir / f"{trajectory_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(
                f"No golden trajectory found with id '{trajectory_id}' "
                f"in {self.golden_dir}"
            )
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return GoldenTrajectory.model_validate(data)

    def list_all(self) -> List[GoldenTrajectory]:
        """Return all stored golden trajectories, sorted by created_at."""
        trajectories = []
        for json_file in sorted(self.golden_dir.glob("*.json")):
            data = json.loads(json_file.read_text(encoding="utf-8"))
            trajectories.append(GoldenTrajectory.model_validate(data))
        trajectories.sort(key=lambda t: t.created_at)
        return trajectories

    def delete(self, trajectory_id: str) -> bool:
        """Delete a golden trajectory by ID. Returns True if deleted, False if not found."""
        file_path = self.golden_dir / f"{trajectory_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def exists(self, trajectory_id: str) -> bool:
        """Check whether a golden trajectory with the given ID exists."""
        return (self.golden_dir / f"{trajectory_id}.json").exists()


class ResultsStore:
    """Saves and loads EvalResult objects from a local JSON store."""

    def __init__(self, store_dir: Path = DEFAULT_STORE_DIR) -> None:
        self.store_dir = Path(store_dir)
        self.results_dir = self.store_dir / RESULTS_SUBDIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: EvalResult) -> Path:
        """Persist an eval result to disk. Returns the file path."""
        file_path = self.results_dir / f"{result.result_id}.json"
        file_path.write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        return file_path

    def load(self, result_id: str) -> EvalResult:
        """Load an eval result by ID. Raises FileNotFoundError if missing."""
        file_path = self.results_dir / f"{result_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(
                f"No eval result found with id '{result_id}' in {self.results_dir}"
            )
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return EvalResult.model_validate(data)

    def list_all(self) -> List[EvalResult]:
        """Return all stored eval results, sorted by evaluated_at."""
        results = []
        for json_file in sorted(self.results_dir.glob("*.json")):
            data = json.loads(json_file.read_text(encoding="utf-8"))
            results.append(EvalResult.model_validate(data))
        results.sort(key=lambda r: r.evaluated_at)
        return results

    def list_for_trajectory(
        self, trajectory_id: str
    ) -> List[EvalResult]:
        """Return all eval results for a specific trajectory ID."""
        return [
            r for r in self.list_all() if r.trajectory_id == trajectory_id
        ]

    def list_for_run(self, run_id: str) -> List[EvalResult]:
        """Return all eval results belonging to a specific run."""
        return [r for r in self.list_all() if r.run_id == run_id]

    def delete(self, result_id: str) -> bool:
        """Delete an eval result by ID. Returns True if deleted, False if not found."""
        file_path = self.results_dir / f"{result_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
