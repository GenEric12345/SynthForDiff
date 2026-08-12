"""Annotate every synthetic document with t_min: the smallest grid t at which
the classifier can no longer tell it from real data.

Per synthetic document and per grid t: mean P(real) over K=8 fresh masks.
t_min = smallest t with mean confidence >= tau (tau = 0.5 - eps, eps=0.05 by
default; both configurable). Documents that never cross get t_min = 1.0.

    python scripts/annotate_tmin.py [--smoke-test] [--ckpt ...]
        [--set annotate.eps=0.05] [--set annotate.splits=[test]]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.classifier import load_classifier
from src.config import REPO_ROOT, base_parser, load_config, setup_run
from src.data import load_pairs
from src.eval_utils import eval_mask_seed, score_documents
from src.plots import plot_tmin_histogram
from src.utils import device_auto, get_logger

log = get_logger("annotate_tmin")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args)
    setup_run(cfg, "annotate_tmin")
    device = device_auto()

    tau = cfg.annotate.tau if cfg.annotate.tau is not None else 0.5 - cfg.annotate.eps
    ckpt_path = Path(args.ckpt) if args.ckpt else (
        REPO_ROOT / cfg.paths.checkpoints / "uniform_t" / "final.pt")
    model, _ = load_classifier(ckpt_path, map_location=device)
    model.to(device).eval()

    pairs = load_pairs(cfg, list(cfg.annotate.splits))
    synth = np.stack([np.asarray(x, dtype=np.int64) for x in pairs["synth_ids"]])
    log.info("annotating %d synthetic docs (splits %s) | tau=%.3f | K=%d",
             len(pairs), list(cfg.annotate.splits), tau, cfg.eval.n_masks)

    t_grid = [float(t) for t in cfg.eval.t_grid]
    conf = np.zeros((len(pairs), len(t_grid)))
    for j, t in enumerate(t_grid):
        conf[:, j] = score_documents(model, synth, t, cfg.eval.n_masks,
                                     cfg.eval.batch_size, device,
                                     eval_mask_seed(cfg.seed, "annot", t))
        log.info("t=%.2f  mean P(real)=%.4f", t, conf[:, j].mean())

    crossed = conf >= tau
    t_min = np.full(len(pairs), 1.0)
    for i in range(len(pairs)):
        hits = np.nonzero(crossed[i])[0]
        if len(hits):
            t_min[i] = t_grid[hits[0]]

    out = pd.DataFrame({
        "doc_id": pairs["doc_id"].to_numpy(),
        "split": pairs["split"].to_numpy(),
        "t_min": t_min,
    })
    for j, t in enumerate(t_grid):
        out[f"conf_t{t:g}"] = conf[:, j]

    res_dir = REPO_ROOT / cfg.paths.results
    res_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(res_dir / "annotations.parquet", index=False)
    plot_tmin_histogram(t_min, t_grid, tau, res_dir / "tmin_histogram.png")
    log.info("wrote %s and %s | median t_min=%.2f | never-crossed=%.1f%%",
             res_dir / "annotations.parquet", res_dir / "tmin_histogram.png",
             float(np.median(t_min)), 100.0 * float((t_min == 1.0).mean()))


if __name__ == "__main__":
    main()
