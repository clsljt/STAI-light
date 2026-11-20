"""Training and testing of STAI under the CATCH / TAB protocol."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.data_loader import build_datasets, load_series
from metrics import evaluate
from models.STAI import Model


def random_patch_mask(x, ratio, patch_len):
    """Training mask: round(L * ratio / patch_len) patches of patch_len steps with uniform random starts (they may
    overlap); the same steps are masked in every channel."""
    batch, length, channels = x.shape
    mask = torch.zeros_like(x, dtype=torch.bool)
    patches = max(1, int(round(length * ratio / patch_len)))
    starts = torch.randint(max(1, length - patch_len + 1), (batch, patches), device=x.device)
    positions = (starts.unsqueeze(-1) + torch.arange(patch_len, device=x.device)).reshape(batch, -1).clamp(max=length - 1)
    mask.scatter_(1, positions.unsqueeze(-1).expand(-1, -1, channels), True)
    return mask


def fixed_patch_mask(x, pass_index, ratio, patch_len):
    """Deterministic mask: with P = 1 / ratio passes, pass p masks the patches whose index is p (mod P).
    Together the P passes mask every time step exactly once."""
    passes = max(1, round(1.0 / ratio))
    times = torch.arange(x.size(1), device=x.device)
    selected = ((times // patch_len + pass_index) % passes) == 0
    return selected.view(1, -1, 1).expand_as(x)


def spectral_error(reconstruction, target):
    """mean over bins and channels of |rFFT(rec) - rFFT(x)|^2 (unnormalised rFFT), per window."""
    diff = torch.fft.rfft(reconstruction.float(), dim=1) - torch.fft.rfft(target.float(), dim=1)
    return torch.abs(diff).pow(2).mean(dim=(1, 2))


class Exp_Anomaly_Detection:
    def __init__(self, args, config):
        self.args, self.cfg = args, config
        self.tc, self.mc = config["train"], config["model"]
        # Seed before the model is built: initialisation, shuffling, masks and dropout all follow this seed.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        self.series = load_series(args.root_path, args.data)
        self.train_set, self.valid_set, self.test_set, self.test_span = build_datasets(
            self.series, self.mc["seq_len"], self.tc["validation_ratio"], self.tc["train_stride"])
        self.model = Model(enc_in=self.series.channels, **self.mc).to(self.device)
        self.out_dir = Path(args.save_path) / args.data / f"seed_{args.seed}"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------------------------------------------
    def loss(self, reconstruction, target, mask):
        """masked MSE + w_u * MSE over the window + w_f * spectral error."""
        squared = (reconstruction - target).pow(2)
        spectral = torch.mean(torch.abs(torch.fft.rfft(reconstruction.float(), dim=1)
                                        - torch.fft.rfft(target.float(), dim=1)).pow(2))
        return (squared.masked_select(mask).mean()
                + self.tc["unmasked_loss_weight"] * squared.mean()
                + self.tc["frequency_loss_weight"] * spectral)

    @torch.no_grad()
    def validate(self, loader):
        """Training loss with the deterministic pass-0 mask, on every validation window."""
        self.model.eval()
        losses = []
        for x, _ in loader:
            x = x.to(self.device, non_blocking=True)
            mask = fixed_patch_mask(x, 0, self.tc["mask_ratio"], self.tc["mask_patch_len"])
            losses.append(self.loss(self.model(x.masked_fill(mask, 0.0)), x, mask).item())
        return float(np.mean(losses))

    def train(self):
        tc, pin = self.tc, self.device.type == "cuda"
        train_loader = DataLoader(self.train_set, batch_size=tc["batch_size"], shuffle=True, drop_last=True, pin_memory=pin)
        valid_loader = DataLoader(self.valid_set, batch_size=tc["batch_size"], shuffle=False, pin_memory=pin)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=tc["learning_rate"], epochs=tc["train_epochs"],
                                                        steps_per_epoch=len(train_loader), pct_start=0.2)
        amp = tc["use_amp"] and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        print(f"{self.args.data}: {len(self.train_set)} train / {len(self.valid_set)} validation / "
              f"{len(self.test_set)} test windows, {sum(p.numel() for p in self.model.parameters()):,} parameters")

        best, waiting, log = float("inf"), 0, []
        for epoch in range(1, tc["train_epochs"] + 1):
            self.model.train()
            start, train_losses = time.time(), []
            for x, _ in train_loader:
                x = x.to(self.device, non_blocking=True)
                mask = random_patch_mask(x, tc["mask_ratio"], tc["mask_patch_len"])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=amp):
                    loss = self.loss(self.model(x.masked_fill(mask, 0.0)), x, mask)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                train_losses.append(loss.item())
            valid_loss = self.validate(valid_loader)
            log.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "valid_loss": valid_loss,
                        "seconds": time.time() - start})
            print(f"Epoch {epoch:03d} | train {np.mean(train_losses):.6f} | valid {valid_loss:.6f}", flush=True)
            if valid_loss < best:                       # keep the epoch with the lowest validation loss
                best, waiting = valid_loss, 0
                torch.save(self.model.state_dict(), self.out_dir / "checkpoint.pth")
            else:
                waiting += 1
                if waiting >= tc["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break
        (self.out_dir / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")

    # ------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def score(self):
        """Point scores on the non-overlapping test windows (fp32).

        Each window is reconstructed P = 1 / mask_ratio times with the complementary deterministic masks; a time step
        is scored in the pass that masks it:  mean_c (rec - x)^2  +  w * spectral_error(window, pass).
        """
        tc, w = self.tc, self.cfg["test"]["frequency_score_weight"]
        self.model.eval()
        total = len(self.series.values)
        score_sum, count = np.zeros(total), np.zeros(total, dtype=np.int32)
        loader = DataLoader(self.test_set, batch_size=self.cfg["test"]["batch_size"], shuffle=False)
        for x, starts in loader:
            x = x.to(self.device, non_blocking=True)
            for p in range(max(1, round(1.0 / tc["mask_ratio"]))):
                mask = fixed_patch_mask(x, p, tc["mask_ratio"], tc["mask_patch_len"])
                rec = self.model(x.masked_fill(mask, 0.0))
                point = ((rec - x).pow(2).mean(dim=-1) + w * spectral_error(rec, x)[:, None]).cpu().numpy()
                selected = mask[:, :, 0].cpu().numpy()
                for row, s in enumerate(starts.numpy()):
                    idx = np.flatnonzero(selected[row]) + int(s)
                    score_sum[idx] += point[row, selected[row]]
                    count[idx] += 1
        begin, end = self.test_span
        if (count[begin:end] != 1).any():
            raise RuntimeError("Every test point must be scored exactly once")
        return score_sum[begin:end]

    def test(self):
        self.model.load_state_dict(torch.load(self.out_dir / "checkpoint.pth", map_location=self.device))
        scores = self.score()
        np.save(self.out_dir / "test_scores.npy", scores)
        begin, end = self.test_span
        labels = np.asarray(self.series.labels[begin:end])          # labels are read only after scoring
        result = evaluate(labels, scores)
        result.update(dataset=self.args.data, seed=self.args.seed, scored_points=int(end - begin),
                      unscored_tail=int(len(self.series.values) - end))
        (self.out_dir / "metrics.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return result
