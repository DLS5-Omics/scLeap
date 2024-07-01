import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


from common import config as cfg


def parse_step(line):
    try:
        return int(re.search(r"Train Step: \[(\d+)", line).group(1))
    except:
        return -1


def get_logs(verbose=0):
    lines = []
    p = Path(cfg.log_dir)
    for path in p.glob("train_log.*"):
        if verbose > 0:
            print("Loading:", p / path)
        lines.extend(open(p / path).readlines())
    if len(lines) == 0:
        raise Exception("LogNotFoundException")

    return lines


def get_step_logs(verbose=0):
    lines = get_logs(verbose)
    lines = [_ for _ in lines if "Train Step" in _]
    return sorted(lines, key=lambda x: parse_step(x))
