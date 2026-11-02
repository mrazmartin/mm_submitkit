"""Preflight checks run before anything is submitted, so mistakes fail on the login node."""
from __future__ import annotations

import netrc
import os
import subprocess
from pathlib import Path

from submitkit.jobs import Job
from submitkit.profiles import Profile


def wandb_configured(env: dict[str, str]) -> str | None:
    """None if WandB can log in non-interactively (or is disabled/offline), else the problem."""
    mode = env.get("WANDB_MODE", os.environ.get("WANDB_MODE", "online"))
    if mode in ("disabled", "offline", "dryrun"):
        return None
    if env.get("WANDB_API_KEY") or os.environ.get("WANDB_API_KEY"):
        return None
    try:
        host = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").split("://")[-1].split("/")[0]
        if netrc.netrc().authenticators(host):
            return None
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    return "no WandB API key (run `wandb login` once, or set WANDB_API_KEY / WANDB_MODE=offline)"


def git_state(cwd: str) -> tuple[str | None, bool]:
    """(commit hash, dirty) of the repo at cwd; (None, False) outside git."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=cwd, text=True).strip())
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def preflight(jobs: list[Job], profile: Profile, base_env: dict[str, str], requires: list[str]) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings)."""
    errors, warnings = [], []

    try:
        profile.base_dir()
    except RuntimeError as e:
        errors.append(str(e))

    if profile.executor == "slurm":
        if not profile.partition and any(j.gpus == 0 for j in jobs):
            errors.append(f"profile '{profile.name}' has no partition set")
        if any(j.gpus > 0 for j in jobs) and not profile.gpu_partition:
            errors.append(f"profile '{profile.name}' has no gpu_partition set but jobs request GPUs")
        if profile.max_minutes:
            for j in jobs:
                m = j.minutes or profile.minutes
                if m > profile.max_minutes:
                    errors.append(f"job '{j.name}' asks {m} min > partition limit {profile.max_minutes} min")

    for j in jobs:
        if not Path(j.python).exists():
            errors.append(f"job '{j.name}': python '{j.python}' not found")
        for r in j.runs:
            if not (Path(j.cwd) / r.script).exists():
                errors.append(f"job '{j.name}': script '{r.script}' not found under {j.cwd}")
                break

    if "wandb" in requires:
        problem = wandb_configured(base_env)
        if problem:
            errors.append(problem)

    cwds = sorted({j.cwd for j in jobs})
    for cwd in cwds:
        _, dirty = git_state(cwd)
        if dirty and profile.executor == "slurm":
            warnings.append(f"uncommitted changes in {cwd}: queued jobs run the working tree as it is "
                            f"when they START, not as it is now")
    return errors, warnings
