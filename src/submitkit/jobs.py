"""
Job / Run description and the function that executes a Job (on a node or locally).

Strings in Run.args, Run.env values and Run.done are formatted with:
    {results_root}  the profile's results dir
    {cpus}          cpus of the allocation (SLURM_CPUS_PER_TASK on a node)
    {<ENV_KEY>}     any key of the run's own env (e.g. {SEED})
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Run:
    script: str                                 # python script, relative to Job.cwd or absolute
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    done: str | None = None                     # if this path exists the run is skipped
    name: str | None = None                     # shown in logs; defaults to the env

    def label(self, env: dict[str, str] | None = None) -> str:
        return self.name or " ".join(f"{k}={v}" for k, v in (env or self.env).items()) or self.script


@dataclass
class Job:
    name: str
    runs: list[Run]
    cpus: int | None = None                     # None -> profile default
    mem_gb: int | None = None
    minutes: int | None = None
    gpus: int = 0
    cwd: str = field(default_factory=lambda: str(Path.cwd()))
    python: str = field(default_factory=lambda: sys.executable)
    threads: int = 1                            # OMP/MKL/torch threads per run (setdefault)
    keep_going: bool = True                     # continue with next run if one fails


def _fmt(s: str, ctx: dict[str, str]) -> str:
    try:
        return s.format(**ctx)
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(f"Cannot format '{s}' (available: {sorted(ctx)})") from e


def resolve_run(run: Run, results_root: Path, cpus: int) -> tuple[list[str], dict[str, str], Path | None]:
    ctx = {"results_root": str(results_root), "cpus": str(cpus)}
    env = {k: _fmt(str(v), ctx) for k, v in run.env.items()}
    ctx.update(env)
    args = [_fmt(str(a), ctx) for a in run.args]
    done = Path(_fmt(run.done, ctx)) if run.done else None
    return args, env, done


def is_done(run: Run, results_root: Path, cpus: int) -> bool:
    done = resolve_run(run, results_root, cpus)[2]
    return done is not None and done.exists()


def execute_job(job: Job, results_root: str, base_env: dict[str, str], cpus: int) -> dict[str, list[str]]:
    """Runs job.runs sequentially as subprocesses. Module-level so submitit/pickle can ship it."""
    results_root = Path(results_root)
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", cpus))
    env0 = {**os.environ, **base_env, "RESULTS_ROOT": str(results_root)}
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        env0.setdefault(k, str(job.threads))

    summary: dict[str, list[str]] = {"done": [], "skipped": [], "failed": []}
    print(f"[submitkit] job '{job.name}': {len(job.runs)} runs, cwd={job.cwd}, results_root={results_root}", flush=True)
    for run in job.runs:
        args, env, done = resolve_run(run, results_root, cpus)
        label = run.label(env)
        if done is not None and done.exists():
            print(f"===== SKIP  {label}  (exists: {done})", flush=True)
            summary["skipped"].append(label)
            continue
        print(f"\n===== START {label}", flush=True)
        t0 = time.time()
        rc = subprocess.call([job.python, run.script, *args], env={**env0, **env}, cwd=job.cwd)
        mins = (time.time() - t0) / 60
        if rc == 0:
            print(f"===== DONE  {label}  ({mins:.1f} min)", flush=True)
            summary["done"].append(label)
        else:
            print(f"===== FAIL  {label}  (exit {rc}, {mins:.1f} min)", flush=True)
            summary["failed"].append(label)
            if not job.keep_going:
                break

    print(f"[submitkit] job '{job.name}' finished: " + ", ".join(f"{k}={len(v)}" for k, v in summary.items()), flush=True)
    if summary["failed"]:
        raise RuntimeError(f"{len(summary['failed'])} run(s) failed in job '{job.name}': {summary['failed']}")
    return summary
