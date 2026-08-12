"""Train the time-aware real-vs-synthetic classifier on masked sequences.

Per example per step: t ~ Uniform(t_min, t_max), fresh Bernoulli(t) mask,
BCE against the label (1=real, 0=synthetic; model outputs P(real)).
Every batch is exactly 50/50: each document contributes its real AND its
synthetic continuation. The prompt is never part of the input.

    python scripts/train_classifier.py [--smoke-test] [--fixed-t 0.5]
        [--resume auto] [--wandb]
"""
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.classifier import MaskedSeqClassifier, save_checkpoint
from src.config import REPO_ROOT, base_parser, load_config, setup_run
from src.data import PairedContinuations, collate_pairs, load_pairs
from src.masking import apply_random_mask, sample_t
from src.utils import derive_seed, device_auto, get_logger

log = get_logger("train_classifier")


def make_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(2B, L) inputs and (2B,) labels; first half real (1), second half synthetic (0)."""
    x = torch.cat([batch["real_ids"], batch["synth_ids"]]).to(device)
    y = torch.cat([torch.ones(len(batch["real_ids"])),
                   torch.zeros(len(batch["synth_ids"]))]).to(device)
    return x, y


@torch.no_grad()
def val_auc_binned(model, loader, cfg, device, epoch_tag: str) -> dict:
    """AUC binned by sampled t (bins of width 0.1). Fresh t and masks per pass."""
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(derive_seed(cfg.seed, "val", epoch_tag))
    recs = defaultdict(lambda: ([], []))  # bin -> (scores, labels)
    use_bf16 = cfg.train.bf16 and device.type == "cuda"
    for _ in range(cfg.train.val_passes):
        for batch in loader:
            x, y = make_batch(batch, device)
            if cfg.train.fixed_t is not None:
                t = torch.full((x.shape[0],), float(cfg.train.fixed_t), device=device)
            else:
                t = sample_t(x.shape[0], cfg.train.t_min, cfg.train.t_max,
                             device=device, generator=gen)
            xt = apply_random_mask(x, t[:, None], generator=gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(xt)
            p = torch.sigmoid(logits.float()).cpu().numpy()
            tb = np.clip(np.floor(t.cpu().numpy() * 10 + 1e-9), 0, 9) / 10
            for pi, yi, bi in zip(p, y.cpu().numpy(), tb):
                recs[bi][0].append(pi)
                recs[bi][1].append(yi)
    model.train()
    out = {}
    for b in sorted(recs):
        s, ys = np.array(recs[b][0]), np.array(recs[b][1])
        if len(np.unique(ys)) == 2:
            out[b] = float(roc_auc_score(ys, s))
    return out


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--fixed-t", type=float, default=None,
                        help="Train a dedicated classifier at a single mask ratio")
    parser.add_argument("--resume", type=str, default=None,
                        help="'auto' (latest ckpt for this run) or a checkpoint path")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args)
    if args.fixed_t is not None:
        cfg.train.fixed_t = args.fixed_t
    if args.resume is not None:
        cfg.train.resume = args.resume
    if args.wandb:
        cfg.train.wandb = True
    setup_run(cfg, "train_classifier")
    device = device_auto()

    run_name = cfg.train.run_name or (
        (f"fixed_t{cfg.train.fixed_t}" if cfg.train.fixed_t is not None else "uniform_t")
        + ("_nonmask" if cfg.model.pool == "nonmask" else ""))
    ckpt_dir = REPO_ROOT / cfg.paths.checkpoints / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log.info("run %s | device %s | backbone %s | pool %s",
             run_name, device, cfg.model.backbone, cfg.model.pool)

    train_pairs = load_pairs(cfg, ["train"])
    val_pairs = load_pairs(cfg, ["val"]).head(cfg.train.val_max_docs)
    dl_gen = torch.Generator()
    dl_gen.manual_seed(derive_seed(cfg.seed, "dataloader"))
    train_loader = DataLoader(
        PairedContinuations(train_pairs), batch_size=cfg.train.docs_per_batch,
        shuffle=True, generator=dl_gen, collate_fn=collate_pairs,
        num_workers=cfg.train.num_workers, drop_last=True)
    val_loader = DataLoader(
        PairedContinuations(val_pairs), batch_size=cfg.train.docs_per_batch,
        shuffle=False, collate_fn=collate_pairs, num_workers=0)

    model = MaskedSeqClassifier(cfg.model.backbone, cfg.model.pool).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.train.lr_backbone, cfg.train.lr_head),
        weight_decay=cfg.train.weight_decay)
    steps_per_epoch = len(train_loader) // cfg.train.grad_accum
    total_steps = max(1, cfg.train.epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < cfg.train.warmup_steps:
            return (step + 1) / max(1, cfg.train.warmup_steps)
        frac = (step - cfg.train.warmup_steps) / max(1, total_steps - cfg.train.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    if cfg.train.resume:
        if cfg.train.resume == "auto":
            # newest step first (by step number in filename); skip unloadable files
            candidates = sorted(ckpt_dir.glob("step_*.pt"),
                                key=lambda p: int(p.stem.split("_")[1]), reverse=True)
        else:
            candidates = [Path(cfg.train.resume)]
        for path in candidates:
            if not path.exists():
                continue
            try:
                ckpt = torch.load(path, map_location=device, weights_only=False)
            except Exception as e:
                log.info("skipping unloadable checkpoint %s (%s)", path.name, e)
                continue
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_step = ckpt["step"]
            log.info("resumed from %s at optimizer step %d", path, start_step)
            break
        else:
            log.info("no checkpoint found for --resume; starting fresh")

    wandb = None
    if cfg.train.wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=cfg.train.wandb_project, name=run_name,
                   config=cfg.to_plain(), resume="allow")

    mask_gen = torch.Generator(device=device)
    mask_gen.manual_seed(derive_seed(cfg.seed, "train-mask", start_step))
    use_bf16 = cfg.train.bf16 and device.type == "cuda"
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt_step = start_step
    micro = 0
    loss_acc = 0.0
    t_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    done = False
    for epoch in range(cfg.train.epochs):
        if done:
            break
        for batch in train_loader:
            if opt_step >= total_steps:
                done = True
                break
            x, y = make_batch(batch, device)
            if cfg.train.fixed_t is not None:
                t = torch.full((x.shape[0],), float(cfg.train.fixed_t), device=device)
            else:
                t = sample_t(x.shape[0], cfg.train.t_min, cfg.train.t_max,
                             device=device, generator=mask_gen)
            xt = apply_random_mask(x, t[:, None], generator=mask_gen)  # fresh mask
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(xt)
                loss = loss_fn(logits.float(), y)
            (loss / cfg.train.grad_accum).backward()
            loss_acc += loss.item()
            micro += 1
            if micro % cfg.train.grad_accum != 0:
                continue
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            if opt_step % cfg.train.log_every == 0:
                n = cfg.train.log_every * cfg.train.grad_accum
                log.info("epoch %d step %d/%d | loss %.4f | lr %.2e | %.2f s/step",
                         epoch, opt_step, total_steps, loss_acc / n,
                         scheduler.get_last_lr()[0],
                         (time.time() - t_start) / cfg.train.log_every)
                if wandb:
                    wandb.log({"loss": loss_acc / n,
                               "lr": scheduler.get_last_lr()[0]}, step=opt_step)
                loss_acc, t_start = 0.0, time.time()
            if opt_step % cfg.train.val_every == 0:
                aucs = val_auc_binned(model, val_loader, cfg, device, f"s{opt_step}")
                log.info("val AUC by t bin: %s",
                         " ".join(f"[{b:.1f}):{a:.3f}" for b, a in aucs.items()))
                if wandb:
                    wandb.log({f"val_auc_t{b:.1f}": a for b, a in aucs.items()},
                              step=opt_step)
            if opt_step % cfg.train.ckpt_every == 0:
                save_checkpoint(ckpt_dir / f"step_{opt_step:06d}.pt", model,
                                optimizer, scheduler, opt_step,
                                extra={"cfg": cfg.to_plain()})
                log.info("checkpoint at step %d", opt_step)

    save_checkpoint(ckpt_dir / "final.pt", model, optimizer, scheduler, opt_step,
                    extra={"cfg": cfg.to_plain()})
    aucs = val_auc_binned(model, val_loader, cfg, device, "final")
    log.info("FINAL val AUC by t bin: %s",
             " ".join(f"[{b:.1f}):{a:.3f}" for b, a in aucs.items()))
    log.info("saved %s", ckpt_dir / "final.pt")
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
