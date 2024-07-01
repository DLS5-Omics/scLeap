import os
import hashlib
import functools
from pathlib import Path

import yaml
import numpy as np

from .util import ensure_dir, no_remote, is_remote_running
from .logger import Logger

__all__ = ["config"]


class Config:
    def __init__(self):
        # for data
        self.global_batch_size = 128
        self.local_batch_size = 1
        self.n_cls = 2058
        self.bins = np.linspace(0, 9.3, 65)

        # for training
        self.pretrain_step = 1000000
        self.mixed_precision = True
        self.nr_step = 30000
        self.warmup_step = 5000
        try:
            import torch

            self.n_gpu = torch.distributed.get_world_size()
        except:
            self.n_gpu = 1
        self.n_accumulate = (
            self.global_batch_size // self.n_gpu // self.local_batch_size
        )
        self.lr = 0.001

        # for checkpoint
        self.chk_time_interval = 3600
        self.chk_step_interval = [10000]

        # for job
        self._job = self._load_job_config()

    @property
    @functools.lru_cache(maxsize=1)
    def experiment_root(self):
        return Path(__file__).resolve().parent.parent

    @property
    @no_remote
    @functools.lru_cache(maxsize=1)
    def project_root(self):
        cur = Path(__file__).resolve().parent

        while True:
            if (cur / ".projectile").exists():
                return cur
            if cur == "/":
                raise Exception("ProjectRootNotFound")
            cur = cur.parent

    def _create_logger(self, path, **kwargs):
        return Logger(path, **kwargs)

    @property
    @functools.lru_cache(maxsize=1)
    def train_logger(self):
        return self._create_logger(self.log_dir / "train_log.txt")

    @property
    @functools.lru_cache(maxsize=1)
    def dataset_dir(self):
        path = [Path(f"/home/yan/ISPert/dataset_full")]
        for p in path:
            if p.exists():
                return p
        raise Exception("DatasetNotFoundError")

    @property
    @no_remote
    @functools.lru_cache(maxsize=1)
    def unique_id(self):
        relpath = self.experiment_root.relative_to(self.project_root)
        return hashlib.md5(str(relpath).encode()).hexdigest()

    @property
    def _job_config_path(self):
        return self.experiment_root / ".job.yaml"

    @no_remote
    def _generate_job_config(self):
        project_saved_dir = Path(f"/blob/r2pdev/experiments/ISPert-{os.getenv('USER')}/")
        relpath = self.experiment_root.relative_to(self.project_root)
        saved_dir = str(project_saved_dir / relpath)
        job = {
            "unique_id": self.unique_id,
            "saved_dir": saved_dir,
        }
        with open(self._job_config_path, "w") as fp:
            yaml.dump(job, fp)
        return job

    def _load_job_config(self):
        job = None
        try:
            with open(self._job_config_path, "r") as fp:
                job = yaml.safe_load(fp.read())
        except:
            job = self._generate_job_config()
        if not is_remote_running() and job.get("unique_id", "") != self.unique_id:
            return self._generate_job_config()
        return job

    @property
    @ensure_dir
    def saved_dir(self):
        return Path(self._job["saved_dir"])

    @property
    @functools.lru_cache(maxsize=1)
    @ensure_dir
    def log_dir(self):
        return self.saved_dir / "log"

    @property
    @functools.lru_cache(maxsize=1)
    @ensure_dir
    def checkpoint_dir(self):
        return Path("/tmp/checkpoint")

    @property
    @functools.lru_cache(maxsize=1)
    @ensure_dir
    def model_dir(self):
        return self.saved_dir / "model"

    @property
    @functools.lru_cache(maxsize=1)
    def pretrain_model_dir(self):
        return self.saved_dir.parent / "pretrain" / "model"


config = Config()
