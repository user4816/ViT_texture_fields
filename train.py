"""Base file for starting training (DDP version)"""

import torch
import wandb
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from mesh2tex import config
import matplotlib

matplotlib.use('Agg')

def main_worker(local_rank, world_size, args):
    # 초기화
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=local_rank)
    if local_rank == 0:
        import wandb
        wandb.login(key="0ab7a56f949924bd68ef4e0cea23f8cd23b19636")
        wandb.init(
            project="Coursework_CS570",
            job_type="train",
            sync_tensorboard=False,
            settings=wandb.Settings(_disable_stats=True),
            config=vars(args)
        )

    cfg = config.load_config(args.config, 'configs/default.yaml')

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    models = config.get_models(cfg, device=device)
    
    # 모든 모델을 DistributedDataParallel로 wrapping
    # for key in models:
    #     models[key] = DDP(models[key], device_ids=[local_rank], output_device=local_rank)

    for key in models:
        models[key] = DDP(models[key], device_ids=[local_rank],
                          output_device=local_rank,
                          find_unused_parameters=True)

    optimizers = config.get_optimizers(models, cfg)

    # 분산 데이터 로더 준비 (DistributedSampler 사용)
    train_loader = config.get_dataloader('train', cfg, distributed=True)
    val_loader = config.get_dataloader('val_eval', cfg, distributed=True)

    if cfg['training']['vis_fixviews'] is True:
        vis_loader = config.get_dataloader('val_vis', cfg, distributed=True)
    else:
        vis_loader = None

    trainer = config.get_trainer(models, optimizers, cfg, device=device)

    trainer.train(train_loader, val_loader, vis_loader,
                  exit_after=args.exit_after, n_epochs=None)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description='Train a Texture Field (DDP).'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--exit-after', type=int, default=-1,
                        help='Checkpoint and exit after specified '
                             'number of seconds with exit code 2.')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for DDP')

    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    if args.no_cuda or world_size < 1:
        raise ValueError("CUDA devices unavailable or explicitly disabled.")

    main_worker(args.local_rank, world_size, args)


if __name__ == "__main__":
    main()
