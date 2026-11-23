"""Core data models for the Agent Eval and Regression Platform."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrajectoryStep(BaseModel):
    """A single step in an agent trajectory."""

    step_index: int = Field(..., description="Zero-based index of this step")
    action: str = Field(..., description="The action taken by the agent")
    action_input: Dict[str, Any] = Field(
        default_factory=dict, description="Input parameters for the action"
    )
    output: str = Field(..., description="Output produced by the action")
    latency_ms: float = Field(
        ..., description="Time taken to execute this step in milliseconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional step metadata"
    )


class GoldenTrajectory(BaseModel):
    """A golden (reference) trajectory for a task."""

    trajectory_id: str = Field(..., description="Unique identifier for this trajectory")
    task_description: str = Field(..., description="Description of the task being evaluated")
    prompt: str = Field(..., description="The prompt used to initiate the agent")
    steps: List[TrajectoryStep] = Field(
        default_factory=list, description="Ordered list of trajectory steps"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this trajectory was recorded"
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags for filtering and grouping trajectories"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional trajectory metadata"
    )


class StepScore(BaseModel):
    """Scores for a single step comparison."""

    step_index: int
    action_match: bool = Field(..., description="Whether the action matched the golden step")
    output_similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Token-overlap similarity score (0-1)"
    )
    latency_delta_ms: float = Field(
        ..., description="Difference in latency vs golden (actual - golden)"
    )


class EvalResult(BaseModel):
    """Result of evaluating an actual trajectory against a golden trajectory."""

    result_id: str = Field(..., description="Unique identifier for this eval result")
    trajectory_id: str = Field(..., description="ID of the golden trajectory evaluated against")
    run_id: str = Field(..., description="Identifier for the eval run this result belongs to")
    step_scores: List[StepScore] = Field(
        default_factory=list, description="Per-step evaluation scores"
    )
    action_match_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of steps with matching actions"
    )
    avg_output_similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Average output similarity across steps"
    )
    avg_latency_delta_ms: float = Field(
        ..., description="Average latency delta across steps"
    )
    passed: Optional[bool] = Field(
        None, description="Whether this result passed CI gates (None if not yet evaluated)"
    )
    evaluated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this eval was performed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional result metadata"
    )
