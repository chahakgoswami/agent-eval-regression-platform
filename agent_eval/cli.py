"""CLI entry point for the Agent Eval and Regression Platform."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_eval import __version__
from agent_eval.models import GoldenTrajectory, TrajectoryStep
from agent_eval.store import TrajectoryStore, DEFAULT_STORE_DIR


def cmd_list(args: argparse.Namespace) -> int:
    """List all stored golden trajectories."""
    store = TrajectoryStore(store_dir=Path(args.store_dir))
    trajectories = store.list_all()
    if not trajectories:
        print("No golden trajectories found.")
        return 0
    print(f"{'ID':<36}  {'Task':<40}  {'Steps':>5}  Created")
    print("-" * 100)
    for t in trajectories:
        print(
            f"{t.trajectory_id:<36}  "
            f"{t.task_description[:40]:<40}  "
            f"{len(t.steps):>5}  "
            f"{t.created_at.strftime('%Y-%m-%d %H:%M')}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show details of a golden trajectory."""
    store = TrajectoryStore(store_dir=Path(args.store_dir))
    try:
        trajectory = store.load(args.trajectory_id)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(json.loads(trajectory.model_dump_json()), indent=2))
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete a golden trajectory."""
    store = TrajectoryStore(store_dir=Path(args.store_dir))
    deleted = store.delete(args.trajectory_id)
    if deleted:
        print(f"Deleted trajectory '{args.trajectory_id}'.")
        return 0
    print(f"Trajectory '{args.trajectory_id}' not found.", file=sys.stderr)
    return 1


def cmd_version(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Print the version."""
    print(f"agent-eval {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Agent Eval and Regression Platform",
    )
    parser.add_argument(
        "--store-dir",
        default=str(DEFAULT_STORE_DIR),
        help="Path to the local store directory (default: .agent_eval_store)",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # version
    subparsers.add_parser("version", help="Show version and exit")

    # list
    subparsers.add_parser(
        "list", help="List all stored golden trajectories"
    )

    # show
    show_parser = subparsers.add_parser(
        "show", help="Show details of a golden trajectory"
    )
    show_parser.add_argument("trajectory_id", help="Trajectory ID to display")

    # delete
    delete_parser = subparsers.add_parser(
        "delete", help="Delete a golden trajectory"
    )
    delete_parser.add_argument("trajectory_id", help="Trajectory ID to delete")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "version": cmd_version,
        "list": cmd_list,
        "show": cmd_show,
        "delete": cmd_delete,
    }

    if args.command is None:
        parser.print_help()
        return 0

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
