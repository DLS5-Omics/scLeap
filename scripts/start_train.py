import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch


def main():
    world_size = int(os.environ["WORLD_SIZE"])
    world_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    master_ip = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    master_uri = "tcp://%s:%s" % (master_ip, master_port)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=master_uri,
        world_size=world_size,
        rank=world_rank,
    )
    torch.cuda.set_device(local_rank)
    from trainer import Trainer

    with Trainer(local_rank, world_rank) as trainer:
        trainer.train()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
