"""
submitkit -- describe jobs once, run them locally or on a SLURM cluster.

    from submitkit import Job, Run, submit

    jobs = [
        Job(name=f"cond_{c}", minutes=600, cpus=8, mem_gb=32, runs=[
            Run(script="train.py", env={"COND": c, "SEED": str(s)},
                done="{results_root}/{COND}/seed_{SEED}/final.zip")
            for s in range(10)])
        for c in ("a", "b")
    ]
    submit(jobs)        # CLI: --profile helix --submit / --only REGEX / ...

A Job is one SLURM allocation; its Runs execute one after another as subprocesses.
"""
from submitkit.jobs import Job, Run
from submitkit.profiles import Profile, load_profile
from submitkit.cli import submit

__all__ = ["Job", "Run", "Profile", "load_profile", "submit"]
