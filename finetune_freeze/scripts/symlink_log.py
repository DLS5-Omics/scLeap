#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from common import config as cfg

if __name__ == "__main__":
    src = cfg.saved_dir
    dst = cfg.experiment_root / "saved"
    if dst.is_symlink():
        dst.unlink(missing_ok=True)
    dst.symlink_to(src)
