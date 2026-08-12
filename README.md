# SynthForDiff

Does LLM-generated text become statistically indistinguishable from real web
text once corrupted by the forward process of a masked discrete diffusion
model? Discrete analog of Ambient Diffusion Omni (Daras et al.,
[arXiv:2506.10038](https://arxiv.org/abs/2506.10038)).

A time-aware classifier (backbone: [`kuleshov-group/mdlm-owt`](https://huggingface.co/kuleshov-group/mdlm-owt),
trained on masked OpenWebText at all mask ratios — exactly our input
distribution) is trained to separate real vs synthetic continuations under
Bernoulli(t) masking, producing an AUC-vs-t curve and a per-document
annotation `t_min` = smallest mask ratio at which the document is no longer
distinguishable from real data.

## Conventions (must not drift — verified against `third_party/mdlm`)

| Convention | Value | Source |
|---|---|---|
| Tokenizer (all classification inputs) | GPT-2 BPE (`gpt2`) | spec |
| Mask token id | **50257** (appended after GPT-2 vocab; model vocab 50258) | `third_party/mdlm/diffusion.py:85-90`; `mdlm-owt` config |
| Forward process | each token i.i.d. → [MASK] w.p. t, fresh randomness every access | `diffusion.py:575-586` (`q_xt`) |
| Schedule | log-linear: `move_chance = (1-ε)·t`, ε=1e-3 ≈ t; we use exactly t | `noise_schedule.py:126-144` |
| Classifier output | **P(real)**: label 1 = real, 0 = synthetic; AUC positive class = real | `src/classifier.py` |
| Lengths | prompt = first 128 GPT-2 tokens; continuation = next 960 (both classes, asserted) | spec |
| Prompt | never in any classifier input | spec |

`mdlm-owt` was trained with `time_conditioning: false`, so `timesteps=0` is
passed; the mask ratio is visible to the model via the mask-token fraction.

## Setup

```bash
git clone <this repo> && cd SynthForDiff
git clone https://github.com/kuleshov-group/mdlm third_party/mdlm   # reference impl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # GPU box; optional (pure-torch fallback exists)
pytest tests/                                  # ~1 min, CPU only
```

## Runbook (defaults: 30k/5k/5k docs, 1×A100 80GB)

One-command launchers (both log to `results/logs/` and fail fast):

```bash
./run_smoke_test.sh                   # pytest + all 6 steps with --smoke-test (< ~15 min)
./run_full_pipeline.sh                # full experiment; restartable after interruption
RUN_FIXED_T=1 ./run_full_pipeline.sh  # also trains the t=0.5 / t=0.9 control classifiers
```

Or step by step:

```bash
# 1) Build the paired dataset from OpenWebText (streaming).       ~30-60 min (CPU-bound)
python scripts/build_dataset.py

# 2) Generate one continuation per prompt with Qwen2.5-7B (BASE) via vLLM.
#    Resumable; re-run after interruption.                        ~4-8 h
python scripts/generate_synthetic.py

# 3) Train the time-aware classifier (t ~ U(0.05,0.95), fresh masks).
#    Logs val AUC binned by t every 500 steps.                    ~1.5-3 h/epoch
python scripts/train_classifier.py
#    Controls against capacity sharing (optional):
python scripts/train_classifier.py --fixed-t 0.5
python scripts/train_classifier.py --fixed-t 0.9

# 4) Test AUC over t ∈ {0.0..0.9, 0.95}, K=8 masks/doc, bootstrap CIs.   ~1-2 h
python scripts/evaluate.py
#    -> results/auc_vs_t.csv, results/auc_vs_t.png, results/test_scores.parquet

# 5) Controls: shuffled tokens, unigram LR baseline, permutation null.   ~2-3 h
python scripts/run_controls.py
#    -> results/controls.csv, results/controls_overlay.png

# 6) Annotate every synthetic doc (train+val+test) with t_min.           ~2-4 h
python scripts/annotate_tmin.py
#    -> results/annotations.parquet, results/tmin_histogram.png
```

### Smoke test (~500 docs, ~200 steps, < ~15 min total on one GPU)

Run the same six commands with `--smoke-test` appended, in order. Science
lengths (128/960) are unchanged; only counts/steps shrink.

### Config

One YAML (`configs/default.yaml`) + dot-notation overrides:

```bash
python scripts/generate_synthetic.py --set generation.model=meta-llama/Llama-3.1-8B
python scripts/train_classifier.py --set model.pool=nonmask        # pooling ablation
python scripts/annotate_tmin.py --set annotate.eps=0.1 --set annotate.splits=[test]
```

Every run seeds all RNGs from `seed` (masking/generation/splits use derived
per-purpose streams) and dumps its fully-resolved config to
`results/run_configs/`. `configs/local_cpu.yaml` is a dev-only shrunken config
for CPU machines (hf generation backend; not part of the experiment).

## Layout

```
src/        masking.py (forward process + [MASK]=50257), classifier.py,
            data.py, eval_utils.py, plots.py, flash_attn_shim.py, config.py
scripts/    build_dataset -> generate_synthetic -> train_classifier ->
            evaluate -> run_controls -> annotate_tmin
configs/    default.yaml, smoke.yaml, local_cpu.yaml
tests/      pytest suite (mask rate, retokenization, split hygiene, lengths,
            shim-vs-reference math, classifier pooling)
third_party/mdlm    reference implementation (cloned, read-only)
results/    figures, CSVs, annotations, resolved run configs
```

## Notes / caveats

* **Known asymmetry inherent to the protocol** (spec-mandated): real
  continuations are a slice of the full document's tokenization; synthetic
  continuations are re-tokenized standalone. A boundary artifact at token 0 is
  possible; the shuffled-token control does not remove it. Keep in mind when
  interpreting near-zero-t AUC.
* vLLM `ignore_eos=True` means EOS tokens can appear inside generations and
  survive re-tokenization; they count as ordinary tokens for both classes.
* `mdlm-owt` pretraining wrapped 1024-token blocks in `<|endoftext|>`
  (`third_party/mdlm/dataloader.py:277-300`); our inputs are bare 960-token
  continuations. Identical for both classes; fine-tuning absorbs the shift.
* `evaluate.py` must run before `run_controls.py` (the permutation null reuses
  its saved per-document scores).
* Bootstrap and permutation statistics resample DOCUMENTS (paired real+synth).
