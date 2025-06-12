""" Training script for the diffusion model in the latent space of the pretrained AEKL model. """
import argparse
import warnings
from pathlib import Path
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from train_function import train_ddpm
from generative.losses.perceptual import PerceptualLoss
from functools import wraps
from util import get_dataloader, log_mlflow
from collections import OrderedDict

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="Random seed to use.")
    parser.add_argument("--run_dir", help="Location of model to resume.")
    parser.add_argument("--output_dir", default="/project/outputs/runs/", help="Base output directory for runs.")
    parser.add_argument("--training_folder", help="Location of training data folder.")
    parser.add_argument("--validation_folder", help="Location of validation data folder.")
    parser.add_argument("--config_file", help="Location of configuration file.")
    parser.add_argument("--batch_size", type=int, default=3, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of epochs to train.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs between evaluations.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--experiment", help="Mlflow experiment name.")
    parser.add_argument("--lr", type=float, help="Learning rate (overrides config file if specified).")
    parser.add_argument("--accumulation_steps", type=int, default=32, help="Gradient accumulation steps.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs.")

    args = parser.parse_args()
    return args


def main(args):
    # 设置随机种子
    set_determinism(seed=args.seed)
    print_config()

    # 加载配置文件
    print(f"Loading config from {args.config_file}")
    config = OmegaConf.load(args.config_file)
    print(f"Config loaded: {config}")

    # 从配置文件中读取参数（如果命令行未指定）

    if args.training_folder is None and "data" in config and "train_folder" in config["data"]:
        args.training_folder = config["data"]["train_folder"]
        print(f"Using training_folder from config: {args.training_folder}")

    if args.validation_folder is None and "data" in config and "val_folder" in config["data"]:
        args.validation_folder = config["data"]["val_folder"]
        print(f"Using validation_folder from config: {args.validation_folder}")

    if args.run_dir is None and "run_dir" in config:
        args.run_dir = config["run_dir"]
        print(f"Using run_dir from config: {args.run_dir}")

    if not args.training_folder:
        raise ValueError("Training folder (--training_folder) must be provided in command line or config file")

    if not args.validation_folder:
        raise ValueError("Validation folder (--validation_folder) must be provided in command line or config file")

    if not args.run_dir:
        raise ValueError("Run directory (--run_dir) must be provided")

    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 设置运行目录
    run_dir = output_dir / args.run_dir
    resume = run_dir.exists() and (run_dir / "checkpoint.pth").exists()
    run_dir.mkdir(exist_ok=True, parents=True)

    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # 设置TensorBoard记录器
    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    print("Getting data...")
    cache_dir = output_dir / "cached_data_diffusion"
    cache_dir.mkdir(exist_ok=True)

    # 加载数据
    train_loader, val_loader = get_dataloader(
        batch_size=args.batch_size,
        training_folder=args.training_folder,
        validation_folder=args.validation_folder,
        num_workers=args.num_workers,
        model_type="diffusion"
    )

    # 创建扩散模型
    print("Creating model...")
    perceptual_loss = PerceptualLoss(**config["perceptual_network"]["params"])
    perceptual_weight=config["perceptual_weight"]
    diffusion = DiffusionModelUNet(**config["ddpm"].get("params", dict()))
    scheduler = DDPMScheduler(**config["ddpm"].get("scheduler", dict()))

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        diffusion = torch.nn.DataParallel(diffusion)
        perceptual_loss = torch.nn.DataParallel(perceptual_loss)

    diffusion = diffusion.to(device)
    perceptual_loss= perceptual_loss.to(device)

    # 设置优化器
    optimizer = optim.AdamW(diffusion.parameters(), lr=config["ddpm"]["base_lr"])

    # 设置学习率调度器
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # 初始化训练变量
    best_loss = float("inf")
    start_epoch = 0
    no_improvement_count = 0

    # 如果有检查点，从检查点恢复
    if resume:
        print(f"Resuming from checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        diffusion.load_state_dict(checkpoint["diffusion"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        if "no_improvement_count" in checkpoint:
            no_improvement_count = checkpoint["no_improvement_count"]
        print(f"Resumed from epoch {start_epoch} with best loss {best_loss}")
    else:
        print(f"Starting fresh training")

    # 开始训练
    print(f"Starting Training")

    # 调用训练函数
    val_loss = train_ddpm(
        model=diffusion,
        perceptual_loss=perceptual_loss,
        scheduler=scheduler,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
        accumulation_steps=args.accumulation_steps,
        patience=args.patience,
        no_improvement_count=no_improvement_count,
        perceptual_weight=perceptual_weight,
    )

    # 记录到MLflow
    if args.experiment:
        log_mlflow(
            model=diffusion,
            config=config,
            args=args,
            experiment=args.experiment,
            run_dir=run_dir,
            val_loss=val_loss,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
