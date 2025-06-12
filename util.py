""" Utility functions for the project. """
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Union
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    RandRotate90d,
    RandFlipd,
    ToTensord,
)
from monai.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tensorboardX import SummaryWriter
from mlflow import start_run

def get_dataloader(
    batch_size,
    training_folder,
    validation_folder,
    num_workers,
    model_type=None,  # 添加这个参数
):
    """创建训练和验证数据加载器。"""
    # model_type参数在这里不使用，但保留它以保持调用兼容性
    
    # 定义预处理和数据增强的转换
    train_transforms = Compose(
        [
            LoadImaged(keys=["CTlung"]),
            EnsureChannelFirstd(keys=["CTlung"]),
            ScaleIntensityd(keys=["CTlung"]),
            Resized(keys=["CTlung"], spatial_size=(176, 176)),  # 根据需要调整大小
            RandRotate90d(keys=["CTlung"], prob=0.5),
            RandFlipd(keys=["CTlung"], prob=0.5),
            ToTensord(keys=["CTlung"]),
        ]
    )
    
    val_transforms = Compose(
        [
            LoadImaged(keys=["CTlung"]),
            EnsureChannelFirstd(keys=["CTlung"]),
            ScaleIntensityd(keys=["CTlung"]),
            Resized(keys=["CTlung"], spatial_size=(176, 176)),  # 根据需要调整大小
            ToTensord(keys=["CTlung"]),
        ]
    )
    
    # 创建训练和验证数据集
    train_files = [{"CTlung": os.path.join(training_folder, fname)} for fname in os.listdir(training_folder)]
    val_files = [{"CTlung": os.path.join(validation_folder, fname)} for fname in os.listdir(validation_folder)]
    
    # 使用常规Dataset
    train_ds = Dataset(
        data=train_files,
        transform=train_transforms,
    )
    
    val_ds = Dataset(
        data=val_files,
        transform=val_transforms,
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader


def recursive_items(dictionary, prefix=""):
    for key, value in dictionary.items():
        if type(value) in [dict, DictConfig]:
            yield from recursive_items(value, prefix=str(key) if prefix == "" else f"{prefix}.{str(key)}")
        else:
            yield (str(key) if prefix == "" else f"{prefix}.{str(key)}", value)


def log_mlflow(
    model,
    config,
    args,
    experiment: str,
    run_dir: Path,
    val_loss: float,
):
    """Log model and performance on Mlflow system"""
    config = {**OmegaConf.to_container(config), **vars(args)}
    print(f"Setting mlflow experiment: {experiment}")
    mlflow.set_experiment(experiment)

    with start_run():
        print(f"MLFLOW URI: {mlflow.tracking.get_tracking_uri()}")
        print(f"MLFLOW ARTIFACT URI: {mlflow.get_artifact_uri()}")

        for key, value in recursive_items(config):
            mlflow.log_param(key, str(value))

        mlflow.log_artifacts(str(run_dir / "train"), artifact_path="events_train")
        mlflow.log_artifacts(str(run_dir / "val"), artifact_path="events_val")
        mlflow.log_metric(f"loss", val_loss, 0)

        raw_model = model.module if hasattr(model, "module") else model
        mlflow.pytorch.log_model(raw_model, "final_model")


def get_figure(
    img: torch.Tensor,
    recons: torch.Tensor,
):
    img_npy_0 = np.clip(a=img[0, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
    recons_npy_0 = np.clip(a=recons[0, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
    img_npy_1 = np.clip(a=img[1, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
    recons_npy_1 = np.clip(a=recons[1, 0, :, :].cpu().numpy(), a_min=0, a_max=1)

    img_row_0 = np.concatenate(
        (
            img_npy_0,
            recons_npy_0,
            img_npy_1,
            recons_npy_1,
        ),
        axis=1,
    )

    fig = plt.figure(dpi=300)
    plt.imshow(img_row_0, cmap="gray")
    plt.axis("off")
    return fig


def log_reconstructions(
    image: torch.Tensor,
    reconstruction: torch.Tensor,
    writer: SummaryWriter,
    step: int,
    title: str = "RECONSTRUCTION",
) -> None:
    fig = get_figure(
        image,
        reconstruction,
    )
    writer.add_figure(title, fig, step)


@torch.no_grad()
def log_ddpm_sample(
    model: nn.Module,
    scheduler: nn.Module,
    spatial_shape: Tuple,
    writer: SummaryWriter,
    step: int,
    device: torch.device,
) -> None:
    """
    适用于标准无条件DDPM的采样和日志记录函数。
    """
    model.eval()

    # 1. 从一个符合图像维度的纯高斯噪声开始
    # spatial_shape 应该是 (channels, height, width), 例如 (1, 64, 64)
    image = torch.randn((1,) + spatial_shape).to(device)

    # 2. 设置采样步数 (如果你的scheduler需要的话)
    # 对于DDIMScheduler或PNDMScheduler等，这一步是必须的
    scheduler.set_timesteps(1000) # 或者你希望的推理步数

    # 3. 循环去噪
    for t in tqdm(scheduler.timesteps, ncols=70, desc="Sampling"):
        # 预测噪声 - 无需context
        noise_pred = model(x=image, timesteps=torch.asarray((t,), device=device))

        # 使用调度器计算上一步的图像状态
        # .prev_sample 会返回去噪后的图像
        image = scheduler.step(noise_pred, t, image).prev_sample

    # 4. 将最终生成的图像从 [-1, 1] 范围归一化到 [0, 1] 以便显示
    # 这是常见的做法，如果你的数据范围不同，请相应调整
    image = (image.clamp(-1, 1) + 1) / 2.0

    # 5. 提取图像数据并记录到TensorBoard
    # 假设是单通道灰度图
    img_np = image[0, 0, :, :].cpu().numpy()
    
    fig = plt.figure(dpi=300)
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")
    writer.add_figure("SAMPLE", fig, step)
    plt.close(fig) # 关闭图像以防占用过多内存

def log_ldm_sample_unconditioned(
    model: nn.Module,
    scheduler: nn.Module,
    spatial_shape: Tuple,
    writer: SummaryWriter,
    step: int,
    device: torch.device,
) -> None:
    latent = torch.randn((1,) + spatial_shape)
    latent = latent.to(device)
    for t in tqdm(scheduler.timesteps, ncols=70):
        noise_pred = model(x=latent, timesteps=torch.asarray((t,)).to(device))
        latent, _ = scheduler.step(noise_pred, t, latent)
    img_0 = np.clip(a=latent[0, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
    fig = plt.figure(dpi=300)
    plt.imshow(img_0, cmap="gray")
    plt.axis("off")
    writer.add_figure("SAMPLE", fig, step)
