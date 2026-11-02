"""
`submit(jobs)` -- the one entry point a project's submit script calls.

    python my_submit.py                         # dry run: what would be submitted, done/todo counts
    python my_submit.py --submit                # submit with the default profile
    python my_submit.py --profile helix --submit --only 'guided_' --minutes 600
    python my_submit.py --profile local --submit --parallel 4   # run here, 4 jobs at a time
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Sequence

from submitkit.checks import git_state, preflight
from submitkit.jobs import Job, execute_job, is_done
from submitkit.profiles import Profile, load_profile

HEADLESS_ENV = {  # no display on compute nodes; harmless elsewhere
    "SDL_VIDEODRIVER": "dummy", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "MPLBACKEND": "Agg",
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Submit jobs (dry run unless --submit).")
    p.add_argument("--profile", help="cluster profile (default: $SUBMITKIT_PROFILE / user config / local)")
    p.add_argument("--submit", action="store_true", help="actually submit / run (default: dry run)")
    p.add_argument("--only", help="regex: keep jobs whose name matches")
    p.add_argument("--exclude", help="regex: drop jobs whose name matches")
    p.add_argument("--workspace")
    p.add_argument("--results-root")
    p.add_argument("--partition")
    p.add_argument("--minutes", type=int, help="override every job's time limit")
    p.add_argument("--cpus", type=int, help="override every job's cpus")
    p.add_argument("--mem-gb", type=int, help="override every job's memory")
    p.add_argument("--parallel", type=int, default=1, help="local profile: jobs run concurrently")
    p.add_argument("--no-checks", action="store_true", help="skip preflight errors (not recommended)")
    return p


def _resources(job: Job, profile: Profile) -> tuple[int, int, int]:
    return job.cpus or profile.cpus, job.mem_gb or profile.mem_gb, job.minutes or profile.minutes


def _submit_slurm(job: Job, profile: Profile, results_root, base_env, cpus, mem_gb, minutes) -> str:
    import submitit

    logs = results_root.parent / "submitit_logs" / job.name
    logs.mkdir(parents=True, exist_ok=True)
    ex = submitit.AutoExecutor(folder=str(logs), cluster="slurm")
    extra = {"export": "ALL", "chdir": job.cwd, **profile.slurm_extra}
    if job.gpus:
        extra["gres"] = f"gpu:{profile.gpu_type}:{job.gpus}" if profile.gpu_type else f"gpu:{job.gpus}"
    ex.update_parameters(
        name=job.name[:64], timeout_min=minutes, nodes=1, tasks_per_node=1,
        cpus_per_task=cpus, mem_gb=mem_gb,
        slurm_partition=profile.gpu_partition if job.gpus else profile.partition,
        slurm_additional_parameters=extra,
    )
    return ex.submit(execute_job, job, str(results_root), base_env, cpus).job_id


def submit(jobs: Sequence[Job], *, env: dict[str, str] | None = None,
           defaults: dict[str, dict[str, Any]] | None = None, requires: Sequence[str] = (),
           argv: Sequence[str] | None = None) -> None:
    """
    jobs:      what to run (see submitkit.jobs).
    env:       extra env vars for every run (e.g. WANDB_PROJECT).
    defaults:  project-level profile settings, {"helix": {"workspace": "my_ws"}, "*": {...}}.
    requires:  preflight checks to enforce, e.g. ["wandb"].
    """
    args = _parser().parse_args(argv)
    profile = load_profile(args.profile, defaults, workspace=args.workspace,
                           results_root=args.results_root, partition=args.partition)

    jobs = list(jobs)
    if args.only:
        jobs = [j for j in jobs if re.search(args.only, j.name)]
    if args.exclude:
        jobs = [j for j in jobs if not re.search(args.exclude, j.name)]
    if not jobs:
        sys.exit("[submitkit] no jobs selected")
    overrides = {k: v for k, v in (("cpus", args.cpus), ("mem_gb", args.mem_gb), ("minutes", args.minutes)) if v}
    jobs = [replace(j, **overrides) for j in jobs]

    commit, _ = git_state(jobs[0].cwd)
    base_env = {**HEADLESS_ENV, **profile.env, **(env or {})}
    if commit:
        base_env.setdefault("SUBMITKIT_GIT_COMMIT", commit)

    errors, warnings = preflight(jobs, profile, base_env, list(requires))
    for w in warnings:
        print(f"[submitkit] WARNING: {w}")
    for e in errors:
        print(f"[submitkit] ERROR: {e}")
    if errors and args.submit and not args.no_checks:
        sys.exit(f"[submitkit] {len(errors)} preflight error(s); nothing submitted.")
    try:
        results_root = profile.results_dir()
    except RuntimeError:
        results_root = None
    print(f"[submitkit] profile={profile.name} executor={profile.executor} "
          f"partition={profile.partition} results={results_root} commit={(commit or '-')[:8]}")

    todo_jobs = []
    for j in jobs:
        cpus, mem_gb, minutes = _resources(j, profile)
        n_done = sum(is_done(r, results_root, cpus) for r in j.runs) if results_root else 0
        state = "all done" if n_done == len(j.runs) else f"{len(j.runs) - n_done}/{len(j.runs)} todo"
        gpu = f" gpus={j.gpus}" if j.gpus else ""
        print(f"  {j.name:<60} cpus={cpus} mem={mem_gb}G time={minutes}m{gpu}  [{state}]")
        if n_done < len(j.runs):
            todo_jobs.append((j, cpus, mem_gb, minutes))

    if not args.submit:
        fix = f" Fix the {len(errors)} error(s) above first." if errors else ""
        print(f"[submitkit] dry run: {len(todo_jobs)} job(s) would run. Add --submit to go.{fix}")
        return
    if not todo_jobs:
        print("[submitkit] nothing to do.")
        return

    results_root.mkdir(parents=True, exist_ok=True)
    if profile.executor == "slurm":
        for j, cpus, mem_gb, minutes in todo_jobs:
            jid = _submit_slurm(j, profile, results_root, base_env, cpus, mem_gb, minutes)
            print(f"[submitkit] submitted {jid}  {j.name}")
        print(f"[submitkit] logs: {results_root.parent / 'submitit_logs'}")
    elif profile.executor == "local":
        def _one(item):
            j, cpus, _, _ = item
            try:
                execute_job(j, str(results_root), base_env, cpus)
                return None
            except Exception as e:  # report all, don't stop sibling jobs
                return f"{j.name}: {e}"
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            failures = [f for f in pool.map(_one, todo_jobs) if f]
        if failures:
            sys.exit("[submitkit] failed:\n  " + "\n  ".join(failures))
    else:
        sys.exit(f"[submitkit] unknown executor '{profile.executor}'")
