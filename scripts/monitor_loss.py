#!/usr/bin/env python
from collections import defaultdict

import click

from log_util import get_step_logs, parse_step


@click.command()
@click.option("-s", "--start_step", default=-1, type=int)
@click.option("-e", "--end_step", default=-1, type=int)
@click.option("-v", "--verbose", count=True)
def main(start_step, end_step, verbose):
    lines = get_step_logs(verbose)
    if end_step == -1:
        end_step = parse_step(lines[-1])
    if start_step == -1:
        start_step = max(1, end_step - 2000)
    losses = defaultdict(list)
    for line in lines:
        parse_step(line)
        if start_step <= parse_step(line) <= end_step:
            for kv in line.split(","):
                if kv.count(":") != 1:
                    continue
                k, v = kv.split(":")
                if k.endswith("loss") or k.endswith("acc"):
                    if "nan" not in v:
                        try:
                            losses[k].append(float(v))
                        except:
                            pass
    out_str = "Step %i-%i" % (start_step, end_step)
    for k, v in losses.items():
        out_str += ",%s: %.6f" % (k, sum(v) / (len(v) + 1e-8))
    print(out_str)


if __name__ == "__main__":
    main()
