import re
import time
import queue
import shutil
import threading
from collections import defaultdict

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from data_provider.train_loader import TrainLoader
from common import config as cfg
from model.finetune_model import FinetuneModel as model_fn


class Trainer:
    def __init__(self, local_rank, world_rank):
        self.local_rank = local_rank
        self.world_rank = world_rank

        self.device = torch.device("cuda:%i" % local_rank)
        self.dp = TrainLoader()

        self.model = model_fn().to(self.device)
        if world_rank == 0:
            self.logger = cfg.train_logger
            self.logger.info(str(self.model))
            # self.writer = SummaryWriter(cfg.log_dir)
        self.model = DDP(self.model, device_ids=[self.local_rank])

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=0.001, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01
        )

        self.step = 0
        self.load_checkpoint()

        if self.world_rank == 0:
            self.chk_worker = queue.Queue()
            threading.Thread(target=self.sync_checkpoint, daemon=True).start()

    def adjust_learning_rate(self):
        def inverse_sqrt_root_schedule(step, n_warmup, lr):
            factor = lr * n_warmup**0.5
            return factor * min(step**-0.5, step * n_warmup**-1.5)

        lr = inverse_sqrt_root_schedule(self.step + 1, cfg.warmup_step, cfg.lr)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def train(self):
        dp = iter(self.dp)
        cur_time = last_saved_time = start_time = time.time()
        n_trained = 0
        scaler = torch.cuda.amp.GradScaler()

        for i in range(self.step + 1, cfg.nr_step + 1):
            self.adjust_learning_rate()

            b_losses, b_metrics = defaultdict(list), defaultdict(list)

            def step_forward():
                nonlocal dp, scaler
                data = {k: v.to(self.device) for k, v in next(dp).items()}
                with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                    output = self.model(data)
                    losses = {k: v for k, v in output.items() if k.endswith("loss")}
                    metrics = {
                        k: v
                        for k, v in output.items()
                        if k.endswith("_acc") or k.startswith("token")
                    }
                    loss = output["update_loss"]
                    loss = loss / cfg.n_accumulate
                    scaler.scale(loss).backward()
                for k, v in losses.items():
                    b_losses[k].append(v.item())
                for k, v in metrics.items():
                    b_metrics[k].append(v)

            with self.model.no_sync():
                for j in range(cfg.n_accumulate - 1):
                    step_forward()
            step_forward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            n_trained += 1
            self.step += 1

            loss_str = ", ".join(
                ["%s: %.6f" % (k, sum(v) / len(v)) for k, v in b_losses.items()]
            )
            metric_str = ", ".join(
                ["%s: %.6f" % (k, sum(v) / len(v)) for k, v in b_metrics.items()]
            )

            speed = 1.0 / (time.time() - cur_time)
            passed_time = (time.time() - start_time) / 3600

            estimate_time = (cfg.nr_step - self.step) / n_trained * passed_time
            log_str = (
                f"Train Step: [{self.step}/{cfg.nr_step}], "
                f"{loss_str}, {metric_str}, "
                f"Speed: {speed:.3f} m/s, "
                f"Passed: {passed_time:.3f} h, "
                f"Estimate: {estimate_time:.3f} h"
            )
            if self.world_rank == 0:
                if i % 10 == 0:
                    self.logger.info(log_str)
                    # for k, v in b_losses.items():
                    #     self.writer.add_scalar(k, (sum(v) / len(v)), self.step)
                    # for k, v in b_metrics.items():
                    #     self.writer.add_scalar(k, (sum(v) / len(v)), self.step)

            if i % 1 == 0:
                print(log_str)

            cur_time = time.time()
            if (
                i % cfg.chk_step_interval[0] == 0
                or cur_time - last_saved_time > cfg.chk_time_interval
            ):
                self.save_checkpoint()
                last_saved_time = cur_time

    def save_checkpoint(self):
        if self.world_rank != 0:
            return
        state = {
            "step": self.step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        filename = cfg.checkpoint_dir / f"checkpoint-step-{self.step}.pth"

        self.logger.info(f"Saving checkpoint: {filename} ...")
        torch.save(state, filename)
        self.chk_worker.put(filename)

    def load_checkpoint(self):
        latest = -1
        for path in cfg.model_dir.iterdir():
            if path.stem.startswith("checkpoint-step-"):
                step = int(re.findall(r"\d+", path.stem)[0])
                latest = max(latest, step)
        if latest != -1:
            filename = cfg.model_dir / f"checkpoint-step-{latest}.pth"
            if self.world_rank == 0:
                self.logger.info(f"Loading checkpoint: {filename} ...")
            checkpoint = torch.load(filename, map_location="cpu")
            self.step = checkpoint["step"]
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.world_rank == 0:
                self.logger.info(f"Checkpoint '{filename}' (step {self.step}) loaded")
        else:
            pretrain_path = (
                cfg.pretrain_model_dir / f"checkpoint-step-{cfg.pretrain_step}.pth"
            )
            if self.world_rank == 0:
                self.logger.info(f"Loading pretrain checkpoint: {pretrain_path} ...")
            checkpoint = torch.load(pretrain_path, map_location="cpu")

            parsed_dict = {}
            for k, v in checkpoint["state_dict"].items():
                if k.startswith("module."):
                    k = k[7:]
                parsed_dict[k] = v
            self.model.module.pretrain.load_state_dict(parsed_dict)
            if self.world_rank == 0:
                self.logger.info(f"Pretrain checkpoint '{pretrain_path}' loaded")

    def sync_checkpoint(self):
        while True:
            path = self.chk_worker.get()
            print(f"Working on {path}")
            if cfg.checkpoint_dir != cfg.model_dir:
                shutil.copy(path, cfg.model_dir)
                path.unlink()
            print(f"Finished {path}")
            # cur_step = int(re.findall(r"\d+", path.stem)[0])
            # for it in cfg.chk_step_interval:
            #     if cur_step % it == 0:
            #         print(f"Clean checkpoint every {it} step")
            #         for chk in cfg.model_dir.iterdir():
            #             if chk.stem.startswith("checkpoint"):
            #                 chk_step = int(re.findall(r"\d+", chk.stem)[0])
            #                 if chk_step % it != 0:
            #                     chk_path = cfg.model_dir / chk
            #                     print(f"Remove {chk_path}")
            #                     chk_path.unlink()
            self.chk_worker.task_done()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.world_rank == 0:
            if hasattr(self.logger, "flush"):
                self.logger.flush()
            self.chk_worker.join()
