#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from common import config as cfg


if __name__ == "__main__":
    print(cfg.unique_id)
