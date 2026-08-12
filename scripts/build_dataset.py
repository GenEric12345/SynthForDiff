"""Build the paired-prompt dataset from OpenWebText.

For every kept document (>= prompt_len + cont_len GPT-2 tokens):
  prompt = first 128 GPT-2 tokens, real continuation = next 960 tokens.
Deduplicated on the hash of the first 256 tokens; split BY DOCUMENT.

    python scripts/build_dataset.py [--smoke-test]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT, base_parser, load_config, setup_run
from src.data import (assign_splits, build_records, get_gpt2_tokenizer,
                      iter_openwebtext, save_dataset)
from src.utils import get_logger

log = get_logger("build_dataset")


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args)
    setup_run(cfg, "build_dataset")

    n_total = cfg.data.n_train + cfg.data.n_val + cfg.data.n_test
    tokenizer = get_gpt2_tokenizer()
    records = build_records(
        iter_openwebtext(cfg.data.hf_dataset),
        tokenizer,
        n_docs=n_total,
        prompt_len=cfg.data.prompt_len,
        cont_len=cfg.data.cont_len,
        dedup_prefix=cfg.data.dedup_prefix,
        tokenize_batch=cfg.data.tokenize_batch,
    )
    splits = assign_splits(len(records), cfg.data.n_train, cfg.data.n_val,
                           cfg.data.n_test, cfg.seed)
    for r, s in zip(records, splits):
        r["split"] = s
    save_dataset(records, REPO_ROOT / cfg.paths.dataset)
    counts = {s: splits.count(s) for s in ("train", "val", "test")}
    log.info("done: %s", counts)


if __name__ == "__main__":
    main()
    # The HF streaming reader (datasets 4.x) leaves a native background thread
    # that races with interpreter finalization: intermittent
    # "Fatal Python error: PyGILState_Release" aborts or exit hangs, even after
    # closing the iterator.  All artifacts are written by this point, so flush
    # and skip finalization entirely.
    import logging
    import os

    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
