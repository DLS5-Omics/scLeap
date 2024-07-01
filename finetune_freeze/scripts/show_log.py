#!/usr/bin/env python
import sys
from pathlib import Path
import tempfile
import subprocess

import click

sys.path.append(str(Path(__file__).resolve().parent.parent))

from log_util import get_step_logs
from common import config as cfg


@click.command()
@click.option("--aml/--no-aml", default=False)
@click.option("-v", "--verbose", count=True)
def main(aml, verbose):
    if aml:
        prog = f"amlt log -f {cfg.unique_id}"
        subprocess.run(prog, shell=True)
    else:
        with tempfile.NamedTemporaryFile("w") as tp:
            lines = get_step_logs(verbose)
            tp.write("".join(lines))
            tp.flush()
            prog = subprocess.run("less -FRSXK +G " + tp.name, shell=True)


if __name__ == "__main__":
    main()
