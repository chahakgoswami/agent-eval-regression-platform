# Agent Eval and Regression Platform

Golden trajectory eval platform with CI gates and quality dashboards for shipping agents like produc

**Domain:** Agentic AI
**Language:** python
**Demonstrates:** You ship agents like production software.

## 7-day build plan

- [ ] Day 1: Scaffold the project structure with a CLI entry point, define core data models (TrajectoryStep, GoldenTrajectory, EvalResult) using Pydantic, and implement a local JSON-based trajectory store that saves and loads golden trajectories from disk.
- [ ] Day 2: Build a mock agent runner that simulates an LLM-backed agent executing multi-step tasks, produces deterministic fake outputs based on seeded inputs, and records actual trajectories in the same schema as golden trajectories.
- [ ] Day 3: Implement the trajectory comparator that diffs actual vs golden trajectories step-by-step, scoring on action match, output similarity (token overlap), and latency delta, then writes EvalResult objects to a results store.
- [ ] Day 4: Add a CI gate module that reads EvalResults, applies configurable pass/fail thresholds per metric, enforces human-in-the-loop confirmation (stdin prompt) before marking any run as a regression, and exits with a non-zero code on failure so it can be called from GitHub Actions.
- [ ] Day 5: Create a prompt mutation tester that generates variants of agent prompts (paraphrase, truncation, adversarial injection) using rule-based transforms, runs each variant through the mock agent and comparator, and flags prompts that cause trajectory regressions.
- [ ] Day 6: Build a terminal quality dashboard using Rich that displays per-trajectory pass rates, metric trend sparklines across stored runs, top regressing prompts, and a summary health score, all rendered from the local results store with no external dependencies.
- [ ] Day 7: Wire everything into a single `eval` CLI with subcommands (record, run, gate, mutate, dashboard), add a pytest suite covering the comparator, CI gate logic, and prompt mutator with fully mocked data, and package the project with pyproject.toml and a sample golden trajectory fixture for instant demo runs.

_A comprehensive README with an architecture diagram is generated on Day 7._
