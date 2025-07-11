import os
import math
import random
import datetime
import types
from typing import Any

import hydra
import hydra.utils as hy_utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import wandb

from diffusion_policy.dataset.umi_pretrain_mae import TactileAutoencoderDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from .pretrain_mae_model import TactileAutoencoderModel

OmegaConf.register_new_resolver("eval", eval, replace=True)


# -----------------------------------------------------------------------------#
# Repro utils                                                                   #
# -----------------------------------------------------------------------------#
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------#
# Visual helpers                                                       #
# -----------------------------------------------------------------------------#
def _to_hwc(img: torch.Tensor) -> np.ndarray:
    img = img.float()
    if img.max() > 1:
        img = img / 255.0
    return img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def debug_plot_masked_input(
    rgb_batch: torch.Tensor,
    tactile_batch: torch.Tensor,
    out_dir: str,
    prefix: str = "",
    max_samples: int = 4,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if rgb_batch.ndim == 5 and rgb_batch.size(1) == 1:
        rgb_batch = rgb_batch[:, 0]

    rgb_batch = rgb_batch.cpu()
    tactile_batch = tactile_batch.cpu().numpy()

    n = min(max_samples, rgb_batch.shape[0])
    fig, axes = plt.subplots(n, 2, figsize=(6.5, 3 * n))

    for i in range(n):
        axes[i, 0].imshow(_to_hwc(rgb_batch[i]))
        axes[i, 0].set_title(f"Masked Image [{i}]")
        axes[i, 0].axis("off")

        im = axes[i, 1].imshow(tactile_batch[i], cmap="viridis", origin="upper")
        axes[i, 1].set_title(f"Masked Tactile [{i}]")
        axes[i, 1].axis("off")
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_masked_batch.png"))
    plt.close(fig)


def attention_to_heatmap(attn: torch.Tensor, patch_size: int, image_size: int) -> np.ndarray:
    h, s = attn.shape
    g = int(s**0.5)
    avg = attn.reshape(h, g, g).mean(0, keepdim=True).unsqueeze(0)
    return (
        F.interpolate(avg, (image_size, image_size), mode="bilinear", align_corners=False)
        .squeeze()
        .cpu()
        .numpy()
    )


def plot_attention_overlay(rgb: torch.Tensor, heat: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(_to_hwc(rgb))
    plt.imshow(heat, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def debug_plot_train_val(
    train_batch,
    train_pred,
    val_batch,
    val_pred,
    out_dir: str = "debug_plots",
    step: int = 0,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    val_bs = val_batch["rgb_cur"].shape[0]
    val_idx = np.random.choice(val_bs, size=min(10, val_bs), replace=False)
    rows = 1 + len(val_idx)
    fig, axes = plt.subplots(rows, 3, figsize=(16, 4 * rows))

    def _plot(b, p, row, tag, idx):
        axes[row, 0].imshow(_to_hwc(b["rgb_cur"][idx]))
        axes[row, 0].set_title(f"{tag}: Camera")
        axes[row, 0].axis("off")

        tact = b["tactile_cur"][idx].cpu().numpy()
        axes[row, 1].imshow(tact, cmap="viridis", origin="upper", vmin=tact.min(), vmax=tact.max())
        axes[row, 1].set_title(f"{tag}: Tactile (GT)")
        axes[row, 1].axis("off")

        if p is not None:
            c = p.shape[1]
            if c == 3:
                axes[row, 2].imshow(_to_hwc(p[idx]))
            else:
                axes[row, 2].imshow(p[idx].clamp(0, 1).cpu().squeeze(0), cmap="viridis", origin="upper")
            axes[row, 2].set_title(f"{tag}: Tactile (Pred)")
            axes[row, 2].axis("off")

    _plot(train_batch, train_pred, 0, "Train", 0)
    for r, i in enumerate(val_idx, start=1):
        _plot(val_batch, val_pred, r, f"Val[{i}]", i)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"debug_train_val_{step}.png"))
    plt.close(fig)


# -----------------------------------------------------------------------------#
# Eval helpers                                                                 #
# -----------------------------------------------------------------------------#
def eval_autoencoder(model, loader, device="cuda", max_batches: int = 5, debug_plot=False, is_val=True):
    model.eval()
    loss_sum = count = 0
    sb = sp = sg = None
    base = model.module if isinstance(model, AveragedModel) else model

    with torch.no_grad():
        for i, batch in enumerate(loader):
            rgb = batch["rgb_cur"].to(device)
            if rgb.ndim == 4:
                rgb = rgb.unsqueeze(1)
            tactile = batch["tactile_cur"].to(device)

            pred = base(rgb, tactile)
            gt = (
                base.convert_12x64_to_3x24x32(tactile, False)
                if base.predict_channel == 3
                else base.convert_12x64_to_1x24x32(tactile, False)
            )

            loss_sum += F.mse_loss(pred, gt).item()
            count += 1
            if debug_plot and i == 0:
                sb, sp, sg = batch, pred, gt
            if i >= max_batches - 1:
                break

    split = "Val" if is_val else "Train"
    print(f"[eval_autoencoder] {split} MSE over {count} mini-batches = {loss_sum / max(count,1):.6f}")
    return loss_sum / max(count, 1), sb, sp, sg


def attach_last_attn_hook(vit, which: int = -1) -> None:
    attn = vit.blocks[which].attn
    orig_fwd = attn.forward

    def new_fwd(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        a = (q @ k.transpose(-2, -1)) * self.scale
        a = a.softmax(-1)
        self.latest_attn = a.detach()
        x = (a @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

    attn.forward = types.MethodType(new_fwd, attn)


# -----------------------------------------------------------------------------#
# Training                                                                     #
# -----------------------------------------------------------------------------#
@hydra.main(version_base=None, config_path="../diffusion_policy/config", config_name="pretrain_mae.yaml")
def main(cfg: Any) -> None:
    # reproducibility
    seed_everything(cfg.training.seed)

    dataset_path = cfg.task.dataset_path
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"checkpoint_{os.path.basename(dataset_path.rstrip(os.sep))}_{ts}"
        f"_attn-{cfg.training.attention_type}"
        f"_ema-{cfg.training.ema_decay if cfg.training.ema_pretrain else 0:.4f}"
        f"_clipLR-{cfg.training.clip_lr:.0e}"
        f"_predCh-{cfg.training.predict_channel}"
    )

    run_dir = os.path.join("pretrain_checkpoints", run_name)
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))

    # data
    img_size = (224, 224)
    tf_cfg = cfg.transforms
    tfs = []
    for t in tf_cfg:
        c = OmegaConf.to_container(t, resolve=True)
        if isinstance(t, nn.Module):
            tfs.append(t)
        elif isinstance(c, dict) and "_target_" in c:
            tfs.append(hy_utils.instantiate(c))
        else:
            tfs.append(c)

    if tfs and not isinstance(tfs[0], nn.Module):
        assert tfs[0]["type"] == "RandomCrop"
        ratio = tfs[0]["ratio"]
        tfs = [
            torchvision.transforms.RandomCrop(int(img_size[0] * ratio)),
            torchvision.transforms.Resize(img_size[0], antialias=True),
        ] + tfs[1:]

    transform = nn.Identity() if not tfs else nn.Sequential(*tfs)

    dataset = TactileAutoencoderDataset(
        shape_meta=cfg.task.shape_meta,
        dataset_path=dataset_path,
        cache_dir=cfg.task.get("cache_dir"),
        val_ratio=cfg.training.val_ratio,
        train_ratio=1-cfg.training.val_ratio,
        seed=cfg.training.seed,
        transforms=transform,
    )
    val_dataset = dataset.get_validation_dataset()

    print(f"[DATA] Train ep  : {dataset.num_train_episodes}")
    print(f"[DATA] Val ep    : {dataset.num_val_episodes}")
    print(f"[DATA] Train len : {len(dataset)}")
    print(f"[DATA] Val len   : {len(val_dataset)}")

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.get("num_workers", 8),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.get("num_workers", 8),
        pin_memory=True,
    )

    wandb.init(
        project=cfg.logging.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
        tags=cfg.logging.tags,
    )

    # model & opt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TactileAutoencoderModel(
        embed_dim=cfg.training.embed_dim,
        tactile_patch_size=cfg.training.tact_patch_size,
        num_heads=cfg.training.num_heads,
        attention_type=cfg.training.attention_type,
        predict_channel=cfg.training.predict_channel,
        mask_ratio_min=cfg.training.mask_ratio_min,
        mask_ratio_max=cfg.training.mask_ratio_max,
    ).to(device)

    if cfg.training.ema_pretrain:
        decay = cfg.training.ema_decay

        def _ema(p_ema, p, _):
            return p_ema * decay + p * (1 - decay)

        ema_model = AveragedModel(model, avg_fn=_ema).to(device)
        attach_last_attn_hook(ema_model.module.clip_encoder)

        def _save(_, __, out):
            ema_model.module.cross_attn.attn_t2i.latest_attn = out[1]

        ema_model.module.cross_attn.attn_t2i.register_forward_hook(_save)
        for p in ema_model.parameters():
            p.requires_grad = False

    attach_last_attn_hook(model.clip_encoder)
    for p in model.clip_encoder.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(
        [
            {"params": model.clip_encoder.parameters(), "lr": cfg.training.clip_lr},
            {
                "params": [p for n, p in model.named_parameters() if not n.startswith("clip_encoder")],
                "lr": cfg.training.encoder_lr,
            },
        ],
        weight_decay=2e-3,
    )

    total_steps = cfg.training.num_epochs * len(train_loader)
    warmup = int(0.1 * total_steps)
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup, eta_min=0.0),
        ],
        milestones=[warmup],
    )

    def _save_attn(_, __, out):
        model.cross_attn.attn_t2i.latest_attn = out[1]

    model.cross_attn.attn_t2i.register_forward_hook(_save)

    # training
    global_step = 0

    def _cvt_12x64_to_3x24x32(t: torch.Tensor, mask=False):
        B = t.size(0)
        l, r = t[..., :32].clamp(0, 1), t[..., 32:].clamp(0, 1)
        li, ri = (l * 255).long().clamp(0, 255), (r * 255).long().clamp(0, 255)
        lc, rc = self.viridis_map[li], self.viridis_map[ri]  # type: ignore
        color = torch.cat([lc.permute(0, 3, 1, 2), rc.permute(0, 3, 1, 2)], 2)
        return self.mask_tactile_color_image(color) if mask else color  # type: ignore

    @torch.no_grad()
    def _cvt_12x64_to_1x24x32(t, _mask=False):
        left, right = t[..., :32], t[..., 32:]
        return torch.cat([left, right], 1).unsqueeze(1)

    def loss_batch(m, batch):
        rgb = batch["rgb_cur"].to(device)
        if rgb.ndim == 4:
            rgb = rgb.unsqueeze(1)
        tactile = batch["tactile_cur"].to(device)
        pred = m(rgb, tactile)
        gt = (
            _cvt_12x64_to_3x24x32(tactile, False)
            if cfg.training.predict_channel == 3
            else _cvt_12x64_to_1x24x32(tactile, False)
        )
        return F.mse_loss(pred, gt), pred, gt

    for epoch in range(cfg.training.num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} – Train"):
            optimizer.zero_grad()
            loss, pred_c, gt_c = loss_batch(model, batch)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()

            if cfg.training.ema_pretrain:
                ema_model.update_parameters(model)

            train_loss += loss.item()
            global_step += 1
            wandb.log({"train_loss": loss.item(), "epoch": epoch}, step=global_step)

            if global_step % 500 == 0:
                tgt = ema_model if cfg.training.ema_pretrain else model
                tl, tb, tp, _ = eval_autoencoder(tgt, train_loader, device, 3, True, False)
                vl, vb, vp, _ = eval_autoencoder(tgt, val_loader, device, 3, True, True)
                wandb.log({"debug/train_loss": tl, "debug/val_loss": vl}, step=global_step)
                debug_plot_train_val(tb, tp, vb, vp, os.path.join(run_dir, "debug_plots"), global_step)

                tgtm = tgt.module if isinstance(tgt, AveragedModel) else tgt
                attn = tgtm.clip_encoder.blocks[-1].attn.latest_attn
                if attn is not None:
                    heat = attention_to_heatmap(attn[0, :, 0, 1:], 16, 224)
                    plot_attention_overlay(
                        tb["rgb_cur"][0],
                        heat,
                        os.path.join(run_dir, "debug_attention", f"vit_self_attn_{global_step}.png"),
                    )

                if cfg.training.attention_type == "patch":
                    ca = tgtm.cross_attn.attn_t2i.latest_attn
                    if ca is not None:
                        heat = attention_to_heatmap(ca[0, 0], 16, 224)
                        plot_attention_overlay(
                            tb["rgb_cur"][0],
                            heat,
                            os.path.join(run_dir, "debug_attention", f"patch_cross_attn_{global_step}.png"),
                        )

        avg_train = train_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch} – Train MSE = {avg_train:.6f}")

        tgt = ema_model if cfg.training.ema_pretrain else model
        tgt.eval()
        val_loss = 0
        with torch.no_grad():
            for vb in val_loader:
                vl, *_ = loss_batch(tgt, vb)
                val_loss += vl.item()
        avg_val = val_loss / max(len(val_loader), 1)
        print(f"Epoch {epoch} – Val MSE = {avg_val:.6f}")
        wandb.log({"val_loss": avg_val, "epoch": epoch, "lr": scheduler.get_last_lr()[0]}, step=global_step)

        ckpt = os.path.join(run_dir, f"epoch_{epoch:04d}.pth")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict() if cfg.training.ema_pretrain else None,
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt,
        )
        print(f"[checkpoint] Saved → {ckpt}")


if __name__ == "__main__":
    main()
