"""Controls on the test split (run scripts/evaluate.py first):

1. Shuffled-token control: permute token order within each continuation BEFORE
   masking, score with the trained classifier. AUC barely dropping => the
   signal is bag-of-words.
2. Unigram baseline: per t, logistic regression on counts of surviving
   (unmasked) tokens, fit on TRAIN, AUC on test. Quantifies vocabulary skew.
3. Permutation null: 200 label shuffles of the saved per-document test scores
   per t; the 97.5th percentile is the chance band.

    python scripts/run_controls.py [--smoke-test] [--ckpt ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.classifier import load_classifier
from src.config import REPO_ROOT, base_parser, load_config, setup_run
from src.data import load_pairs
from src.eval_utils import (eval_mask_seed, permutation_null_auc, score_documents)
from src.masking import GPT2_VOCAB_SIZE, MASK_INDEX, apply_random_mask
from src.plots import plot_controls_overlay
from src.utils import derive_seed, device_auto, get_logger

log = get_logger("run_controls")


def shuffled_copy(tokens: np.ndarray, seed: int) -> np.ndarray:
    """Independently permute token order within each row (before masking)."""
    rng = np.random.default_rng(seed)
    out = tokens.copy()
    for i in range(len(out)):
        rng.shuffle(out[i])
    return out


def surviving_counts(tokens: np.ndarray, t: float, seed: int) -> sp.csr_matrix:
    """One fresh mask realization; bag-of-words counts over surviving tokens."""
    x = torch.from_numpy(np.ascontiguousarray(tokens))
    gen = torch.Generator()
    gen.manual_seed(seed)
    xt = apply_random_mask(x, t, generator=gen).numpy()
    rows_list, cols_list = np.nonzero(xt != MASK_INDEX)
    vals = xt[rows_list, cols_list]
    m = sp.csr_matrix((np.ones(len(vals), dtype=np.float32), (rows_list, vals)),
                      shape=(len(tokens), GPT2_VOCAB_SIZE))
    m.sum_duplicates()
    return m


def unigram_auc(train_real, train_synth, test_real, test_synth, t, cfg) -> float:
    seed = derive_seed(cfg.seed, "unigram", f"{t}")
    cap = cfg.controls.unigram_max_train_docs
    if cap:
        train_real, train_synth = train_real[:cap], train_synth[:cap]
    X = sp.vstack([surviving_counts(train_real, t, seed),
                   surviving_counts(train_synth, t, seed + 1)])
    y = np.concatenate([np.ones(len(train_real)), np.zeros(len(train_synth))])
    clf = LogisticRegression(C=cfg.controls.unigram_C, solver="liblinear",
                             max_iter=200)
    clf.fit(X, y)
    n_test = len(test_real)
    probs = np.zeros(2 * n_test)
    for k in range(cfg.controls.unigram_n_masks):
        Xt = sp.vstack([surviving_counts(test_real, t, seed + 100 + k),
                        surviving_counts(test_synth, t, seed + 200 + k)])
        probs += clf.predict_proba(Xt)[:, list(clf.classes_).index(1.0)]
    probs /= cfg.controls.unigram_n_masks
    y_test = np.concatenate([np.ones(n_test), np.zeros(n_test)])
    return float(roc_auc_score(y_test, probs))


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args)
    setup_run(cfg, "run_controls")
    device = device_auto()

    res_dir = REPO_ROOT / cfg.paths.results
    scores_path = res_dir / "test_scores.parquet"
    main_csv = res_dir / "auc_vs_t.csv"
    if not scores_path.exists() or not main_csv.exists():
        raise FileNotFoundError("run scripts/evaluate.py first (needs "
                                f"{scores_path.name} and {main_csv.name})")
    ckpt_path = Path(args.ckpt) if args.ckpt else (
        REPO_ROOT / cfg.paths.checkpoints / "uniform_t" / "final.pt")
    model, _ = load_classifier(ckpt_path, map_location=device)
    model.to(device).eval()

    test = load_pairs(cfg, ["test"])
    train = load_pairs(cfg, ["train"])
    te_real = np.stack([np.asarray(x, dtype=np.int64) for x in test["real_ids"]])
    te_synth = np.stack([np.asarray(x, dtype=np.int64) for x in test["synth_ids"]])
    tr_real = np.stack([np.asarray(x, dtype=np.int64) for x in train["real_ids"]])
    tr_synth = np.stack([np.asarray(x, dtype=np.int64) for x in train["synth_ids"]])

    sh_real = shuffled_copy(te_real, derive_seed(cfg.seed, "shuffle", "real"))
    sh_synth = shuffled_copy(te_synth, derive_seed(cfg.seed, "shuffle", "synth"))

    scores = pd.read_parquet(scores_path)
    rows = []
    for t in cfg.eval.t_grid:
        # 1) shuffled-token control (same trained classifier, same t grid)
        s_real = score_documents(model, sh_real, t, cfg.eval.n_masks,
                                 cfg.eval.batch_size, device,
                                 eval_mask_seed(cfg.seed, "ctrl-shuf-real", t))
        s_synth = score_documents(model, sh_synth, t, cfg.eval.n_masks,
                                  cfg.eval.batch_size, device,
                                  eval_mask_seed(cfg.seed, "ctrl-shuf-synth", t))
        y = np.concatenate([np.ones(len(s_real)), np.zeros(len(s_synth))])
        auc_shuffle = float(roc_auc_score(y, np.concatenate([s_real, s_synth])))

        # 2) unigram logistic-regression baseline
        auc_uni = unigram_auc(tr_real, tr_synth, te_real, te_synth, t, cfg)

        # 3) permutation null from the saved per-document classifier scores
        st = scores[np.isclose(scores["t"], t)]
        null_975, _ = permutation_null_auc(
            st["score_real"].to_numpy(), st["score_synth"].to_numpy(),
            cfg.controls.n_perm, derive_seed(cfg.seed, "perm", f"{t}"))

        rows.append({"t": t, "auc_shuffle": auc_shuffle, "auc_unigram": auc_uni,
                     "perm_null_975": null_975})
        log.info("t=%.2f  shuffle=%.4f  unigram=%.4f  null97.5=%.4f",
                 t, auc_shuffle, auc_uni, null_975)

    controls = pd.DataFrame(rows)
    controls.to_csv(res_dir / "controls.csv", index=False)
    main_df = pd.read_csv(main_csv)
    plot_controls_overlay(main_df, controls, cfg.eval.threshold,
                          res_dir / "controls_overlay.png")
    log.info("wrote %s and %s", res_dir / "controls.csv",
             res_dir / "controls_overlay.png")


if __name__ == "__main__":
    main()
