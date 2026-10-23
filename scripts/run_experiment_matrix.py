#!/usr/bin/env python3
"""Run a resumable, config-driven sequence of Viola experiments."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from loader.config import deep_update, load_config  # noqa: E402


ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_state(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"runs": {}}
    with path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict) or not isinstance(state.get("runs", {}), dict):
        raise ValueError(f"Invalid experiment state file: {path}")
    state.setdefault("runs", {})
    return state


def _write_state(path: Path, state: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _load_matrix(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle) or {}
    if not isinstance(matrix, dict):
        raise ValueError("Matrix YAML root must be a mapping")
    experiments = matrix.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Matrix must define a non-empty 'experiments' list")
    ids = []
    for item in experiments:
        if not isinstance(item, dict):
            raise ValueError("Every experiment entry must be a mapping")
        run_id = str(item.get("id", ""))
        if not ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"Invalid experiment id: {run_id!r}")
        ids.append(run_id)
        if not isinstance(item.get("overrides", {}), dict):
            raise ValueError(f"overrides for {run_id} must be a mapping")
    if len(set(ids)) != len(ids):
        raise ValueError("Experiment ids must be unique within a matrix")
    return matrix


def _resolved_config(
    matrix: Dict[str, Any], experiment: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    base_path = _repo_path(str(matrix["base_config"]))
    cfg = load_config(base_path)
    deep_update(cfg, matrix.get("defaults", {}))
    deep_update(cfg, experiment.get("overrides", {}))
    cfg["seed"] = int(experiment.get(
        "seed", matrix.get("seed", cfg.get("seed", 42))))
    cfg["experiment"] = {
        "matrix": str(matrix.get("name", "unnamed")),
        "id": str(experiment["id"]),
        "title": str(experiment.get("title", experiment["id"])),
        "group": str(experiment.get("group", "unspecified")),
    }
    cfg.setdefault("train", {})["out_dir"] = str(output_dir)
    return cfg


def _filter_experiments(
    experiments: List[Dict[str, Any]], only: str | None
) -> List[Dict[str, Any]]:
    if not only:
        return experiments
    wanted = {value.strip() for value in only.split(",") if value.strip()}
    available = {str(item["id"]) for item in experiments}
    missing = wanted - available
    if missing:
        raise ValueError(
            f"Unknown experiment ids: {', '.join(sorted(missing))}")
    return [item for item in experiments if item["id"] in wanted]


def _status(matrix_path: Path, output_root: Path) -> int:
    state_path = output_root / "matrix_state.json"
    state = _read_state(state_path)
    print(f"matrix: {matrix_path}")
    print(f"output: {output_root}")
    print("experiment                         status       return code")
    for run_id, entry in state["runs"].items():
        print(
            f"{run_id:<34} {entry.get('status', 'unknown'):<12} {entry.get('returncode', '')}")
    if not state["runs"]:
        print("No runs recorded")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True,
                        help="Matrix YAML, relative to Viola root")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print runs without starting training")
    parser.add_argument("--status", action="store_true",
                        help="Show state for this matrix")
    parser.add_argument("--only", help="Comma-separated experiment ids to run")
    parser.add_argument("--wait-for-lock", action="store_true",
                        help="Wait for another matrix runner to finish")
    parser.add_argument("--rerun-complete", action="store_true",
                        help="Rerun completed experiments")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue after a failed experiment")
    args = parser.parse_args()

    matrix_path = _repo_path(args.matrix).resolve()
    matrix = _load_matrix(matrix_path)
    output_root = _repo_path(str(matrix["output_root"])).resolve()
    if args.status:
        return _status(matrix_path, output_root)

    experiments = _filter_experiments(matrix["experiments"], args.only)
    if args.dry_run:
        for experiment in experiments:
            run_id = str(experiment["id"])
            out_dir = output_root / run_id
            cfg = _resolved_config(matrix, experiment, out_dir)
            print(f"{run_id}: {experiment.get('title', run_id)}")
            print(
                f"  model={cfg.get('model', {}).get('name')} backbone={cfg.get('model', {}).get('backbone')}")
            print(
                f"  seed={cfg['seed']} focal_gamma={cfg.get('train', {}).get('focal_gamma', 0.0)}")
            print(f"  output={out_dir}")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "matrix_state.json"
    lock_path = output_root / ".matrix.lock"
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            lock_flags = fcntl.LOCK_EX
            if not args.wait_for_lock:
                lock_flags |= fcntl.LOCK_NB
            fcntl.flock(lock_handle.fileno(), lock_flags)
        except BlockingIOError:
            print(f"Another runner holds {lock_path}", file=sys.stderr)
            return 2

        state = _read_state(state_path)
        state["matrix"] = str(matrix_path)
        state["runner_pid"] = os.getpid()
        state["updated_at"] = _utc_now()

        for experiment in experiments:
            run_id = str(experiment["id"])
            out_dir = output_root / run_id
            history_path = out_dir / "history.json"
            previous = state["runs"].get(run_id, {})
            previous_history = Path(
                previous.get("history") or history_path
            )
            if (
                not args.rerun_complete
                and previous.get("status") == "complete"
                and previous_history.is_file()
            ):
                print(f"SKIP {run_id}: already complete")
                continue

            legacy_id = experiment.get("legacy_id")
            legacy_history = output_root / str(legacy_id) / "history.json"
            legacy_previous = state["runs"].get(str(legacy_id), {})
            if (
                legacy_id
                and not args.rerun_complete
                and legacy_previous.get("status") == "complete"
                and legacy_history.is_file()
            ):
                state["runs"][run_id] = {
                    **legacy_previous,
                    "title": str(experiment.get("title", run_id)),
                    "status": "complete",
                    "returncode": 0,
                    "history": str(legacy_history),
                    "adopted_from": str(legacy_id),
                }
                state["updated_at"] = _utc_now()
                _write_state(state_path, state)
                print(
                    f"SKIP {run_id}: reusing completed run {legacy_id}",
                    flush=True,
                )
                continue

            cfg = _resolved_config(matrix, experiment, out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            config_path = out_dir / "resolved_config.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False,
                               allow_unicode=True)

            command = [sys.executable, str(
                ROOT / "scripts" / "train.py"), "--config", str(config_path)]
            log_path = out_dir / "train.log"
            state["runs"][run_id] = {
                "title": str(experiment.get("title", run_id)),
                "group": str(experiment.get("group", "unspecified")),
                "status": "running",
                "started_at": _utc_now(),
                "config": str(config_path),
                "log": str(log_path),
                "command": command,
                "runner_pid": os.getpid(),
            }
            state["updated_at"] = _utc_now()
            _write_state(state_path, state)
            print(
                f"START {run_id}: {experiment.get('title', run_id)}", flush=True)

            with log_path.open("a", encoding="utf-8") as log_handle:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            complete = result.returncode == 0 and history_path.is_file()
            state["runs"][run_id].update({
                "status": "complete" if complete else "failed",
                "returncode": int(result.returncode),
                "finished_at": _utc_now(),
                "history": str(history_path) if history_path.is_file() else None,
            })
            state["updated_at"] = _utc_now()
            _write_state(state_path, state)
            print(
                f"{'DONE' if complete else 'FAILED'} {run_id}: "
                f"exit={result.returncode} history={history_path.is_file()}",
                flush=True,
            )
            if not complete and not args.continue_on_error:
                return int(result.returncode or 1)
            time.sleep(float(matrix.get("cooldown_seconds", 0)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
