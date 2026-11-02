# submitkit

Describe jobs once in a short Python script; run them locally or on a SLURM cluster. Built on
[submitit](https://github.com/facebookincubator/submitit).

- **Job** = one SLURM allocation. Its **Runs** execute one after another as `python <script> <args>`
  subprocesses with their own env vars (e.g. one Run per seed).
- **Profiles** hold the cluster specifics (partition, workspace, default resources).
- **Dry run by default**: lists jobs and how many runs are already done; `--submit` to go.
- **Resumable**: a Run with a `done` path is skipped when that file exists, so resubmitting
  the same script only reruns what is missing.
- **Preflight checks** on the login node: workspace exists, scripts exist, time limit fits the
  partition, WandB key present (`requires=["wandb"]`), warning on uncommitted changes.

## Install

```bash
uv add "submitkit @ git+https://github.com/<you>/submitkit"     # in the project using it
```

## Use

```python
# my_project/submit.py
from submitkit import Job, Run, submit

SEEDS = range(10)
jobs = [
    Job(name=f"exp__{cond}", cpus=8, mem_gb=32, minutes=12 * 60, runs=[
        Run(script="train.py",
            env={"COND": cond, "SEED": str(s), "N_ENVS": "{cpus}"},
            done="{results_root}/exp/{COND}/seed_{SEED}/final.zip")
        for s in SEEDS])
    for cond in ("baseline", "ours")
]

if __name__ == "__main__":
    submit(jobs, env={"WANDB_PROJECT": "my-project"}, requires=["wandb"],
           defaults={"helix": {"workspace": "my_ws"}})
```

```bash
python submit.py                                   # dry run
python submit.py --profile helix --submit          # submit
python submit.py --profile helix --submit --only ours --minutes 600
python submit.py --profile local --submit --parallel 4
```

Every run gets `RESULTS_ROOT` (the profile's results dir), `SUBMITKIT_GIT_COMMIT`, headless
rendering vars and single-threaded BLAS/torch defaults (`Job.threads`). Strings in `Run.env`,
`Run.args` and `Run.done` can use `{results_root}`, `{cpus}` and any key of the run's env.

Results go to `<workspace or results_root>/runs/`, logs to `<...>/submitit_logs/<job name>/`.

## Profiles

Built-in: `local`, `helix` (bwForCluster Helix: `cpu-single` / `gpu-single`, 120 h limit).
Add or adjust your own in `~/.config/submitkit/profiles.yaml`:

```yaml
default: helix
helix:
  workspace: my_ws
horeka:
  executor: slurm
  partition: cpuonly
  max_minutes: 4320
  workspace: my_ws
```

Precedence: built-in < user file < project `defaults=` < `SUBMITKIT_*` env vars
(`SUBMITKIT_PROFILE`, `_WORKSPACE`, `_RESULTS_ROOT`, `_PARTITION`, `_CPUS`, `_MEM_GB`, `_MINUTES`)
< CLI flags.

Note: queued jobs run the code in the working tree as it is when they *start*. Don't edit or
switch branches in that checkout while jobs are pending.
