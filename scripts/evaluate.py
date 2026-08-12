"""Evaluate the trained classifier on the test split over the t grid.

Per document and per t: K=8 independent mask realizations, averaged into one
P(real) score. AUC per t (real = positive class) with 95% bootstrap CI over
documents. Also saves per-document scores for reuse by run_controls.py.

    python scripts/evaluate.py [--smoke-test] [--ckpt checkpoints/uniform_t/final.pt]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.classifier import load_classifier
from src.config import REPO_ROOT, base_parser, load_config, setup_run
from src.data import load_pairs
from src.eval_utils import eval_mask_seed, paired_bootstrap_auc, score_documents
from src.plots import plot_auc_vs_t
from src.utils import derive_seed, device_auto, get_logger

log = get_logger("evaluate")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="default: checkpoints/uniform_t/final.pt")
    args = parser.parse_args()
    cfg = load_config(args)
    setup_run(cfg, "evaluate")
    device = device_auto()

    ckpt_path = Path(args.ckpt) if args.ckpt else (
        REPO_ROOT / cfg.paths.checkpoints / "uniform_t" / "final.pt")
    model, _ = load_classifier(ckpt_path, map_location=device)
    model.to(device).eval()
    log.info("loaded %s | device %s", ckpt_path, device)

    pairs = load_pairs(cfg, ["test"])
    real = np.stack([np.asarray(x, dtype=np.int64) for x in pairs["real_ids"]])
    synth = np.stack([np.asarray(x, dtype=np.int64) for x in pairs["synth_ids"]])
    log.info("test split: %d documents x 2 classes x %d tokens",
             len(pairs), real.shape[1])

    rows = []
    score_rows = []
    for t in cfg.eval.t_grid:
        s_real = score_documents(model, real, t, cfg.eval.n_masks,
                                 cfg.eval.batch_size, device,
                                 eval_mask_seed(cfg.seed, "eval-real", t))
        s_synth = score_documents(model, synth, t, cfg.eval.n_masks,
                                  cfg.eval.batch_size, device,
                                  eval_mask_seed(cfg.seed, "eval-synth", t))
        auc, lo, hi = paired_bootstrap_auc(
            s_real, s_synth, cfg.eval.n_boot, derive_seed(cfg.seed, "boot", f"{t}"))
        rows.append({"t": t, "auc": auc, "ci_lo": lo, "ci_hi": hi})
        log.info("t=%.2f  AUC=%.4f  [%.4f, %.4f]", t, auc, lo, hi)
        for doc_id, sr, ss in zip(pairs["doc_id"], s_real, s_synth):
            score_rows.append({"doc_id": doc_id, "t": t,
                               "score_real": sr, "score_synth": ss})

    res_dir = REPO_ROOT / cfg.paths.results
    res_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "auc_vs_t.csv", index=False)
    pd.DataFrame(score_rows).to_parquet(res_dir / "test_scores.parquet", index=False)
    plot_auc_vs_t(df, cfg.eval.threshold, res_dir / "auc_vs_t.png")
    log.info("wrote %s, %s, %s", res_dir / "auc_vs_t.csv",
             res_dir / "auc_vs_t.png", res_dir / "test_scores.parquet")


if __name__ == "__main__":
    main()
