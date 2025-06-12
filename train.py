""" Training functions for the diffusion model. """
from collections import OrderedDict
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
import torch
import torch.nn as nn
import torch.nn.functional as F
from pynvml_utils import nvidia_smi
from tensorboardX import SummaryWriter
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from util import log_ldm_sample_unconditioned, log_reconstructions


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def print_gpu_memory_report():
    if torch.cuda.is_available():
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        print("Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            print(f"gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")


def train_ddpm(
    model: nn.Module,
    perceptual_loss: nn.Module,
    scheduler: nn.Module,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    n_epochs: int,
    eval_freq: int,
    perceptual_weight: float,
    writer_train: SummaryWriter,
    writer_val: SummaryWriter,
    device: torch.device,
    run_dir: Path,
    accumulation_steps: int = 1,
    patience: int = 10,
    no_improvement_count: int = 0,
    
) -> float:
    scaler = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model

    # Initial evaluation
    val_loss = eval_ddpm(
        model=model,
        perceptual_loss=perceptual_loss,
        scheduler=scheduler,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
        perceptual_weight=perceptual_weight,
        sample=True,
    )
    print(f"Initial validation loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_ddpm(
            model=model,
            scheduler=scheduler,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler=scaler,
            perceptual_weight=perceptual_weight,
            accumulation_steps=accumulation_steps,
            perceptual_loss=perceptual_loss
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_ddpm(
                model=model,
                perceptual_loss=perceptual_loss,
                perceptual_weight=perceptual_weight,
                scheduler=scheduler,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                sample=True if (epoch + 1) % (eval_freq * 2) == 0 else False,
            )

            print(f"Epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            # Update learning rate scheduler
            lr_scheduler.step(val_loss)

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "diffusion": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "best_loss": best_loss,
                "no_improvement_count": no_improvement_count,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            # Check if we have a new best model
            if val_loss < best_loss:
                print(f"New best val loss {val_loss} (was {best_loss})")
                best_loss = val_loss
                no_improvement_count = 0

                # Save both raw model and full checkpoint for best model
                torch.save(raw_model.state_dict(), str(run_dir / "best_model.pth"))
                torch.save(checkpoint, str(run_dir / "best_checkpoint.pth"))
            else:
                no_improvement_count += 1
                print(f"No improvement for {no_improvement_count} evaluations")

                # Early stopping
                if patience > 0 and no_improvement_count >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))

    # Also save final full checkpoint
    final_checkpoint = {
        "epoch": epoch + 1,
        "diffusion": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "best_loss": best_loss,
        "no_improvement_count": no_improvement_count,
    }
    torch.save(final_checkpoint, str(run_dir / "final_checkpoint.pth"))

    return best_loss


def train_epoch_ddpm(
    model: nn.Module,
    perceptual_loss: nn.Module,
    scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    perceptual_weight: float,
    writer: SummaryWriter,
    scaler: GradScaler,
    accumulation_steps: int = 1,
    
) -> None:
    model.train()

    # Reset gradients at the beginning
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        # Move input data to device
        images = x["CTlung"].to(device)
        reports = x["report"].to(device) if "report" in x else None

        # Sample random timesteps
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        # Forward pass with gradient accumulation
        with autocast(device_type='cuda', enabled=True):
            # Add noise according to timestep
            noise = torch.randn_like(images).to(device)
            noisy_images = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

            # Forward the model - add condition if available
            if reports is not None and hasattr(model, "module") and hasattr(model.module, "with_conditioning") and model.module.with_conditioning:
                noise_pred = model(x=noisy_images, timesteps=timesteps, context=reports)
            elif reports is not None and hasattr(model, "with_conditioning") and model.with_conditioning:
                noise_pred = model(x=noisy_images, timesteps=timesteps, context=reports)
            else:
                noise_pred = model(x=noisy_images, timesteps=timesteps)

            # Compute loss based on prediction type
            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(images, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            else:
                raise ValueError(f"Unknown prediction type: {scheduler.prediction_type}")
            l1_loss= F.l1_loss(noise_pred.float(), target.float())
            p_loss = perceptual_loss(noise_pred.float(), target.float())
            loss = l1_loss+perceptual_weight*p_loss
            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
        # Scale loss for gradient accumulation
        scaled_loss = loss / accumulation_steps
        losses = OrderedDict(loss=loss,p_loss=p_loss,l1_loss=l1_loss)

        # Backward pass with scaling
        scaler.scale(scaled_loss).backward()

        # Update weights after accumulation steps
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Log training metrics
        writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)
        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        # Update progress bar
        pbar.set_postfix({
            "epoch": epoch + 1,
            "l1_loss": f"{losses['l1_loss'].item():.5f}",
            "p_loss": f"{losses['p_loss'].item():.5f}",
            "loss": f"{losses['loss'].item():.5f}",
            "lr": f"{get_lr(optimizer):.6f}"
        })


@torch.no_grad()
def eval_ddpm(
    model: nn.Module,
    perceptual_loss: nn.Module,
    scheduler: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    perceptual_weight: float,
    writer: SummaryWriter,
    sample: bool = False,
    scale_factor: float = 1.0,
    
) -> float:
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    total_losses = OrderedDict()
    num_items = 0

    for x in loader:
        # Move input data to device
        images = x["CTlung"].to(device)
        reports = x["report"].to(device) if "report" in x else None
        batch_size = images.shape[0]
        num_items += batch_size

        # Sample random timesteps
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()

        with autocast(device_type='cuda', enabled=True):
            # Add noise according to timestep
            noise = torch.randn_like(images).to(device)
            noisy_images = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

            # Forward the model with condition if available
            if reports is not None and hasattr(raw_model, "with_conditioning") and raw_model.with_conditioning:
                noise_pred = model(x=noisy_images, timesteps=timesteps, context=reports)
            else:
                noise_pred = model(x=noisy_images, timesteps=timesteps)

            # Compute loss based on prediction type
            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(images, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            else:
                raise ValueError(f"Unknown prediction type: {scheduler.prediction_type}")

            l1_loss= F.l1_loss(noise_pred.float(), target.float())
            p_loss = perceptual_loss(noise_pred.float(), target.float())
            loss = l1_loss+perceptual_weight*p_loss
            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
        # Accumulate losses
        losses = OrderedDict(
            loss=loss.item() * batch_size,
            l1_loss=l1_loss.item() * batch_size,
            p_loss=p_loss.item() * batch_size,
        )
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v

    # Calculate average losses
    for k in total_losses.keys():
        total_losses[k] /= num_items

    # Log validation metrics
    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)
    if sample:
        log_ldm_sample_unconditioned(
            model=raw_model,
            scheduler=scheduler,            
            spatial_shape=tuple(images.shape[1:]),
            writer=writer,
            step=step,
            device=device,
        )

    return total_losses["loss"]
