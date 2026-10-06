#!/usr/bin/env python3
"""Capture software/runtime details from a live Viola training process."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def _process_python(pid: int) -> tuple[str, dict[str, str]]:
    proc = Path("/proc") / str(pid)
    if not proc.is_dir():
        raise ProcessLookupError(f"Training process {pid} is not present in /proc")
    raw_argv = (proc / "cmdline").read_bytes().split(b"\0")
    argv = [part.decode(errors="replace") for part in raw_argv if part]
    if not argv:
        raise RuntimeError(f"Cannot read command line for process {pid}")

    process_env = {}
    for entry in (proc / "environ").read_bytes().split(b"\0"):
        if b"=" in entry:
            key, value = entry.split(b"=", 1)
            process_env[key.decode(errors="replace")] = value.decode(errors="replace")
    candidate = argv[0]
    if not os.path.isabs(candidate):
        candidate = shutil.which(candidate, path=process_env.get("PATH")) or ""
    if not candidate or not Path(candidate).is_file():
        raise FileNotFoundError(
            f"Cannot resolve training interpreter from process {pid} argv[0]"
        )
    return candidate, process_env


def _run(command: list[str], env: dict[str, str] | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=45,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"UNAVAILABLE: {exc}"
    output = (result.stdout + result.stderr).strip()
    if result.returncode:
        return f"UNAVAILABLE (exit {result.returncode}): {output}"
    return output or "(no output)"


def _runtime_query() -> str:
    return (
        "import importlib.metadata as m, json, platform, sys\n"
        "print('python_executable=' + sys.executable)\n"
        "print('python_version=' + sys.version.replace('\\n', ' '))\n"
        "print('platform=' + platform.platform())\n"
        "try:\n"
        " import torch\n"
        " print('torch=' + torch.__version__)\n"
        " print('torch_cuda_runtime=' + str(torch.version.cuda))\n"
        " print('cuda_available=' + str(torch.cuda.is_available()))\n"
        " print('cudnn=' + str(torch.backends.cudnn.version()))\n"
        " if torch.cuda.is_available():\n"
        "  print('visible_gpu_count=' + str(torch.cuda.device_count()))\n"
        "  for i in range(torch.cuda.device_count()): print('visible_gpu_' + str(i) + '=' + torch.cuda.get_device_name(i))\n"
        "except Exception as e: print('torch_import_error=' + repr(e))\n"
        "print('installed_distributions_begin')\n"
        "for d in sorted(m.distributions(), key=lambda x: (x.metadata.get('Name') or '').lower()):\n"
        " name=d.metadata.get('Name') or '(unnamed)'\n"
        " print(name + '==' + (d.version or 'unknown'))\n"
        "print('installed_distributions_end')\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture versions using the exact interpreter of a live training process."
    )
    parser.add_argument(
        "--pid",
        type=int,
        help="PID of the running training process; its executable and environment are inspected.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional text output path. Defaults to stdout.",
    )
    args = parser.parse_args()

    if args.pid is None:
        python_executable = sys.executable
        process_env = dict(os.environ)
        pid_record = "not supplied; current process environment"
    else:
        python_executable, process_env = _process_python(args.pid)
        pid_record = str(args.pid)

    timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "VIOLA SOFTWARE / RUNTIME MANIFEST",
        f"captured_at={timestamp}",
        f"training_pid={pid_record}",
        f"interpreter={python_executable}",
        f"interpreter_command={shlex.join([python_executable, '-c', '<runtime query>'])}",
        f"CUDA_VISIBLE_DEVICES={process_env.get('CUDA_VISIBLE_DEVICES', '(unset)')}",
        f"VIRTUAL_ENV={process_env.get('VIRTUAL_ENV', '(unset)')}",
        f"CONDA_PREFIX={process_env.get('CONDA_PREFIX', '(unset)')}",
        "",
        "OPERATING SYSTEM",
        _run(["uname", "-a"], process_env),
        "",
        "GPU / DRIVER",
        _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            process_env,
        ),
        "",
        "PYTHON, ML RUNTIME, AND INSTALLED DISTRIBUTIONS",
        _run([python_executable, "-c", _runtime_query()], process_env),
        "",
        "CUDA TOOLKIT COMPILER (if installed)",
        _run(["nvcc", "--version"], process_env),
    ]
    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote runtime manifest: {args.output}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
