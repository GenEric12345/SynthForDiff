"""Generate one synthetic continuation per dataset prompt.

Default backend is vLLM with Qwen/Qwen2.5-7B (a BASE model — intentionally not
Instruct), temperature 1.0, top_p 1.0, ignore_eos=True. Every generation is
re-tokenized with GPT-2 and truncated to exactly cont_len (960) tokens so both
classes have identical length; too-short generations are retried with a fresh
seed and a longer token budget (retry rate is logged).

Resumable: doc_ids already present in data/synthetic/<generator>/ are skipped.

    python scripts/generate_synthetic.py [--smoke-test] [--set generation.backend=hf]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import base_parser, load_config, setup_run
from src.data import (append_synthetic_shard, get_gpt2_tokenizer, load_dataset_df,
                      retokenize_truncate, synthetic_dir)
from src.utils import derive_seed, get_logger

log = get_logger("generate_synthetic")


class VllmBackend:
    def __init__(self, cfg):
        from vllm import LLM
        self.cfg = cfg
        self.llm = LLM(model=cfg.generation.model,
                       gpu_memory_utilization=cfg.generation.gpu_memory_utilization,
                       seed=cfg.seed)

    def generate(self, prompts: list[str], seeds: list[int], max_tokens: int) -> list[str]:
        from vllm import SamplingParams
        params = [SamplingParams(temperature=self.cfg.generation.temperature,
                                 top_p=self.cfg.generation.top_p,
                                 max_tokens=max_tokens,
                                 ignore_eos=True,
                                 seed=s) for s in seeds]
        outs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outs]


class HfBackend:
    """transformers.generate fallback for machines without vLLM (dev/smoke only).
    Mirrors vLLM semantics: sampling at temperature/top_p, EOS ignored (no eos
    stopping criterion; EOS tokens can still be sampled into the text)."""

    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(cfg.generation.model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(cfg.generation.model).to(self.device)
        self.model.eval()

    def generate(self, prompts: list[str], seeds: list[int], max_tokens: int) -> list[str]:
        import torch
        from transformers import GenerationConfig
        gen_cfg = GenerationConfig(
            do_sample=True,
            temperature=float(self.cfg.generation.temperature),
            top_p=float(self.cfg.generation.top_p),
            max_new_tokens=max_tokens,
            min_new_tokens=max_tokens,  # never stop early: forces full budget
            pad_token_id=self.tok.pad_token_id,
        )
        texts = []
        mb = self.cfg.generation.hf_micro_batch
        for i in range(0, len(prompts), mb):
            chunk = prompts[i:i + mb]
            torch.manual_seed(seeds[i])  # hf generate has no per-request seed
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**enc, generation_config=gen_cfg)
            new = out[:, enc["input_ids"].shape[1]:]
            texts.extend(self.tok.batch_decode(new, skip_special_tokens=False))
        return texts


def existing_doc_ids(cfg) -> set[str]:
    d = synthetic_dir(cfg)
    if not d.exists():
        return set()
    import pyarrow.parquet as pq
    ids: set[str] = set()
    for shard in sorted(d.glob("shard-*.parquet")):
        ids.update(pq.read_table(shard, columns=["doc_id"]).column("doc_id").to_pylist())
    return ids


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args)
    setup_run(cfg, "generate_synthetic")

    df = load_dataset_df(cfg)
    df = df[df["split"].isin(["train", "val", "test"])]
    done = existing_doc_ids(cfg)
    pending = df[~df["doc_id"].isin(done)]
    log.info("%d prompts total, %d already generated, %d pending",
             len(df), len(done), len(pending))
    if len(pending) == 0:
        log.info("nothing to do")
        return

    backend = (VllmBackend if cfg.generation.backend == "vllm" else HfBackend)(cfg)
    gpt2 = get_gpt2_tokenizer()
    cont_len = int(cfg.data.cont_len)
    n_retries = 0
    n_generated = 0
    t0 = time.time()

    chunk_size = int(cfg.generation.batch_size)
    rows_buf: list[dict] = []
    work = [(r.doc_id, r.prompt_text, 0) for r in pending.itertuples()]
    while work:
        chunk, work = work[:chunk_size], work[chunk_size:]
        max_attempt = max(a for _, _, a in chunk)
        max_tokens = int(cfg.generation.max_tokens
                         * cfg.generation.retry_len_factor ** max_attempt)
        seeds = [derive_seed(cfg.seed, "gen", doc_id, attempt)
                 for doc_id, _, attempt in chunk]
        texts = backend.generate([p for _, p, _ in chunk], seeds, max_tokens)
        token_lists = retokenize_truncate(texts, gpt2, cont_len)
        for (doc_id, prompt, attempt), seed, ids in zip(chunk, seeds, token_lists):
            if ids is None:
                if attempt + 1 >= cfg.generation.max_attempts:
                    raise RuntimeError(
                        f"doc {doc_id}: generation shorter than {cont_len} GPT-2 "
                        f"tokens after {cfg.generation.max_attempts} attempts; "
                        "raise generation.max_tokens")
                n_retries += 1
                work.append((doc_id, prompt, attempt + 1))
                continue
            assert len(ids) == cont_len
            rows_buf.append({
                "doc_id": doc_id,
                "generator_name": cfg.generation.model,
                "seed": seed,
                "attempt": attempt,
                "sampling_params": {
                    "backend": cfg.generation.backend,
                    "temperature": cfg.generation.temperature,
                    "top_p": cfg.generation.top_p,
                    "max_tokens": max_tokens,
                    "ignore_eos": True,
                },
                "synthetic_continuation_token_ids": ids,
            })
            n_generated += 1
        if rows_buf:
            path = append_synthetic_shard(cfg, rows_buf)
            log.info("wrote %s | %d/%d docs | retry rate %.3f | %.1f s elapsed",
                     path.name, n_generated, len(pending),
                     n_retries / max(1, n_generated + n_retries), time.time() - t0)
            rows_buf = []

    log.info("done: %d generated, %d retries (rate %.3f)",
             n_generated, n_retries, n_retries / max(1, n_generated + n_retries))


if __name__ == "__main__":
    main()
