#!/usr/bin/env python
import os
import subprocess

import click

from common import config as cfg
from common import itp, sing


@click.command()
@click.option("-t", "--target", type=click.Choice(["amlk8s", "sing"]), default="sing")
@click.option("-j", "--job_name", default="run1")
@click.option("-e", "--extra_args", default="")
def main(target, job_name, extra_args):
    if target == "amlk8s":
        yaml = cfg.experiment_root / "legacy.yaml"
        target, vc = itp.choose_target()
        extra_args += f" --target-name {target} --vc {vc} "
    else:
        yaml = cfg.experiment_root / "singularity.yaml"
        target = sing.choose_target()
        extra_args += f" --target-name {target} "
    extra_args += " --description None "
    cmd = f"amlt run {yaml} :train={job_name} {cfg.unique_id} "
    cmd += extra_args
    print("Execute:", cmd)
    subprocess.run(cmd, shell=True, env=os.environ)


if __name__ == "__main__":
    main()
