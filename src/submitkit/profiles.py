"""
Cluster profiles: where jobs go (partition, workspace) and their default resources.

Precedence, lowest to highest:
    built-in  <  ~/.config/submitkit/profiles.yaml  <  project defaults (submit(defaults=...))
              <  SUBMITKIT_* env vars  <  CLI flags

User file example (~/.config/submitkit/profiles.yaml):

    default: helix          # profile used when none is given
    helix:
      workspace: guided_dcrl_ws
    mycluster:              # new profile
      executor: slurm
      partition: cpuonly
      results_root: /scratch/me/runs
"""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

USER_CONFIG = Path(os.environ.get("SUBMITKIT_CONFIG", Path.home() / ".config" / "submitkit" / "profiles.yaml"))


@dataclass
class Profile:
    name: str
    executor: str = "local"               # "local" | "slurm"
    partition: str | None = None          # CPU jobs
    gpu_partition: str | None = None      # jobs with gpus > 0
    workspace: str | None = None          # resolved via `ws_find` (bwHPC workspaces)
    results_root: str | None = None       # explicit path; wins over workspace
    results_subdir: str = "runs"          # results go to <workspace or results_root>/<results_subdir>
    cpus: int = 4
    mem_gb: int = 16
    minutes: int = 60
    max_minutes: int | None = None        # partition walltime limit, checked before submitting
    gpu_type: str | None = None
    env: dict[str, str] = field(default_factory=dict)   # injected into every run
    slurm_extra: dict[str, Any] = field(default_factory=dict)  # extra sbatch options

    def updated(self, **kw) -> "Profile":
        known = {f.name for f in fields(self)}
        unknown = set(kw) - known
        if unknown:
            raise ValueError(f"Unknown profile keys {sorted(unknown)}; known: {sorted(known)}")
        kw = {k: v for k, v in kw.items() if v is not None}
        if "env" in kw:
            kw["env"] = {**self.env, **{k: str(v) for k, v in kw["env"].items()}}
        return replace(self, **kw)

    def base_dir(self) -> Path:
        """Workspace / results_root directory (not including results_subdir)."""
        if self.results_root:
            return Path(os.path.expanduser(self.results_root)).resolve()
        if self.workspace:
            try:
                out = subprocess.check_output(f"ws_find {shlex.quote(self.workspace)}",
                                              shell=True, text=True, stderr=subprocess.DEVNULL).strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                out = ""
            if not out:
                raise RuntimeError(f"Workspace '{self.workspace}' not found (ws_find). "
                                   f"Create it with `ws_allocate {self.workspace} <days>` or check `ws_list`.")
            return Path(out)
        if self.executor == "local":
            return Path.cwd()
        raise RuntimeError(f"profile '{self.name}' needs a workspace or results_root "
                           f"(--workspace / SUBMITKIT_WORKSPACE / ~/.config/submitkit/profiles.yaml)")

    def results_dir(self) -> Path:
        return self.base_dir() / self.results_subdir


BUILTIN: dict[str, dict[str, Any]] = {
    "local": dict(executor="local"),
    # bwForCluster Helix: https://wiki.bwhpc.de/e/Helix/Slurm (shared nodes, 64 cores, ~236 GB)
    "helix": dict(executor="slurm", partition="cpu-single", gpu_partition="gpu-single",
                  max_minutes=120 * 60),
}

_ENV_KEYS = {  # SUBMITKIT_<KEY> -> profile field (with type)
    "WORKSPACE": ("workspace", str), "RESULTS_ROOT": ("results_root", str),
    "PARTITION": ("partition", str), "GPU_PARTITION": ("gpu_partition", str),
    "CPUS": ("cpus", int), "MEM_GB": ("mem_gb", int), "MINUTES": ("minutes", int),
}


def _user_config() -> dict[str, Any]:
    if not USER_CONFIG.exists():
        return {}
    return yaml.safe_load(USER_CONFIG.read_text()) or {}


def load_profile(name: str | None = None, defaults: dict[str, dict[str, Any]] | None = None,
                 **overrides) -> Profile:
    """
    name:      profile name; falls back to $SUBMITKIT_PROFILE, then the user file's `default`, then "local".
    defaults:  project-level settings per profile name ({"helix": {"workspace": "x"}}, "*" = all profiles).
    overrides: highest-precedence field values (CLI flags); None values are ignored.
    """
    user = _user_config()
    name = name or os.environ.get("SUBMITKIT_PROFILE") or user.get("default") or "local"
    if name not in BUILTIN and name not in user:
        raise KeyError(f"Unknown profile '{name}'. Built-in: {sorted(BUILTIN)}; user ({USER_CONFIG}): "
                       f"{sorted(k for k in user if k != 'default')}")

    prof = Profile(name=name)
    defaults = defaults or {}
    for layer in (BUILTIN.get(name, {}), user.get(name, {}), defaults.get("*", {}), defaults.get(name, {})):
        prof = prof.updated(**layer)

    env_layer = {}
    for key, (fld, typ) in _ENV_KEYS.items():
        v = os.environ.get(f"SUBMITKIT_{key}", "").strip()
        if v:
            env_layer[fld] = typ(v)
    return prof.updated(**env_layer).updated(**overrides)
