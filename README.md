# Reproduce — rebuild everything from the prompts

The full chain behind *From Emotion Words to Steering Dials: A Prototype
Interface and Its Measured Limits*: extraction, validation, coherence edges,
steered generation, the superposition runs, analysis, figures, the interface
atlases and the verify package. Staged and resumable.

```
reproduce.py (per model, GPU)  ->  recompute_all.py (all nine)  ->  verify.py (laptop)
vectors, generations, atlas        every number, the figures,        every number,
                                   the verify package                checked
```

Most readers want only the last step: the
[verify package](https://github.com/Sinerror/Human-to-LLM-interface-verify)
re-derives every reported number from shipped data on a CPU. This repository
regenerates that data, which takes GPU days.

## Where reproducibility starts

**At the prompts, not the corpus.** The sentence set in `frozen_prompts.jsonl`
was produced by a sampling model and cannot be regenerated identically, so it
ships as a fixed, hashed artifact. From there the chain is forward passes with
fixed seeds.

On the same machine the chain reproduces closely but not always exactly. A full
rerun of gemma-3-1b-it reproduced all 70 transfer matrices in both arms with a
maximum difference of exactly 0.0. A rerun of gemma-3-1b-pt reproduced every
qualitative result — the same c*, every axis positive, the cross-alignment
diagonal maximal in all seven rows — but per-axis gains differed by up to 0.067.
Sampled generation is sensitive to the smallest change in a logit, and a single
flipped token changes everything after it. Across machines, bf16 accumulation
order varies with batch size and hardware, so expect vectors to agree to about
1e-3 and sampled text to diverge outright. Judge a fresh reproduction by the
paper's claims — signs, dominance, orderings, the superposition agreement — not
by every value falling within the verify package's tolerances, which are set for
the shipped data.

## Install

CUDA is required: generation needs a GPU, and the 4-bit Gemma-2-9B needs
`bitsandbytes` on CUDA. Install the torch build for your CUDA version first
(see pytorch.org), then:

```
pip install -r requirements.txt
```

## 1. Per model — `reproduce.py`

```
python reproduce.py --model google/gemma-3-1b-it --layer 17 --batch-rows 48 --work repro --code . --check
python reproduce.py --model google/gemma-3-1b-it --layer 17 --batch-rows 48 --work repro --code . --dry-run
python reproduce.py --model google/gemma-3-1b-it --layer 17 --batch-rows 48 --work repro --code .
```

`--check` confirms every input, script and package is present before any GPU
time is spent. `--dry-run` prints every command it would run.

| | stage | produces | needs |
|---|---|---|---|
| 1 | `extract` | corpus vectors, all layers | GPU |
| 2 | `heldout` | vectors for the pre-registered unseen words | GPU |
| 3 | `validate` | leave-one-word-out and unseen-word tests | CPU |
| 4 | `edge` | the coherence edge c\* for this model | GPU |
| 5 | `layers` | per-layer held-out AUC | CPU |
| 6 | `generate` | three dials at a time: raw, dual and control arms at c\* | GPU |
| 7 | `single` | the same trials, stems and c\*, one dial turned per cell: raw and dual (section 7.3) | GPU |
| 8 | `analyse` | per-model alignment and transfer matrices | CPU |
| 9 | `atlas` | the interface's token atlas | GPU |
| — | `commonmag` | optional (`--only commonmag`): the triple design at c = 0.08 for every model, behind the 54% of sections 3.5 and 7.2 | GPU |
| — | `package` | optional, single-model use; `recompute_all.py` packages all nine | CPU |

Each stage writes a receipt recording the sha256 of its inputs. A rerun skips any
stage whose receipt is present, whose inputs are unchanged, and whose outputs
still exist. **Change the corpus and every downstream stage invalidates** rather
than silently mixing new inputs with old outputs.

```
python reproduce.py ... --status           review
python reproduce.py ... --from generate    resume
python reproduce.py ... --force atlas      redo one stage
```

On failure it prints the tail of the output, writes the full log to
`<work>/<model>/run.log`, and names the `--from` to resume with.

### Nine models

| model | `--layer` | `--batch-rows` | c\* |
|---|---|---|---|
| google/gemma-3-1b-it | 17 | 48 | 0.06 |
| google/gemma-3-1b-pt | 25 | 48 | 0.12 |
| Qwen/Qwen2.5-1.5B | 25 | 48 | 0.32 |
| microsoft/phi-2 | 32 | 32 | 0.40 |
| state-spaces/mamba-2.8b-hf | 52 | 32 | 0.65 |
| EleutherAI/pythia-6.9b | 28 | 16 | 0.25 |
| mistralai/Mistral-7B-v0.3 | 32 | 16 | 0.40 |
| meta-llama/Meta-Llama-3-8B | 18 | 8 | 0.32 |
| unsloth/gemma-2-9b-bnb-4bit | 27 | 8 | 0.50 |

c\* is measured by the `edge` stage, not supplied; it is listed so a
reproduction can be checked against it. The layer is checked at startup: a run at
another model's layer measures a different basis. For Phi-2 and Mistral the
selected layer is the network's last, so the selection criterion reports the
boundary of its range rather than an interior optimum.

## 2. All nine — `recompute_all.py`

```
python recompute_all.py --code . --from-work repro --out results ^
  --labels frozen_emotionality_labels.csv --probes canonical_probes.json ^
  --predictions predictions.csv --cats27 canonical_probes_27d.json --ekman ekman.json
```

It first checks the primary data — all nine models present, every generation
folder made at its model's c\*, every arm present — and computes nothing if any
check fails. Then it derives the geometry, validation and shared-word analysis,
the transfer matrices, the linear baseline, the superposition test, the Appendix C
dimensionality, the common-magnitude matrices if present, the bootstrap, both
encoders' gains and all figures, and ends by building the verify package in
`repro/verify_package`. About an hour with a GPU.

## 3. Verify

Copy `verify.py` and `expected.json` from the verify repository into
`repro/verify_package` and run `python verify.py --full`.

## What you need beside the scripts

| file | role |
|---|---|
| `frozen_prompts.jsonl` | the deterministic starting point |
| `frozen_emotionality_labels.csv` | **the corrected file.** A superseded version differs by seven renamed keys and two additions, and using it shifts every geometry figure without raising an error |
| `canonical_probes.json` | the eighteen word lists |
| `canonical_probes_27d.json` | the 27 Cowen–Keltner categories as word lists (Appendix C) |
| `ekman.json` | Ekman's six as word lists: the atlas readout and Appendix C |
| `heldout_prompts.jsonl` | prompts for the pre-registered unseen words |
| `predictions.csv` | **the pre-registration**, `word,probe,side`, written before any extraction. No stage produces it and none should: regenerating it would destroy the property that makes it worth reporting |

## Data

The outputs of the full chain — vectors at every layer, all generations
(three-dial, single-dial and common-magnitude) and the nine atlases — are in the
Hugging Face dataset: https://huggingface.co/datasets/GhosTunes/H-to-LLM-BackData. With them the GPU stages can be skipped
entirely: arrange each model's files as `repro/<model>/<model>_vecs.npz`,
`repro/<model>/gens/` and `repro/<model>/gens_single/`, and
`recompute_all.py --from-work repro` reads them directly. (`reproduce.py` itself
skips a stage only on its own receipt, so it would regenerate them.)

## Memory

Models load straight to the GPU through a device map, never staged through CPU
RAM. Generation batches halve automatically on out-of-memory. If a model still
does not fit, `--device-map auto` shards across GPU and CPU; it runs, slowly, and
the loader warns which tensors landed on CPU.

## Related

- [Verify](https://github.com/Sinerror/Human-to-LLM-interface-verify) — every
  number in the paper, recomputed on a laptop.
- [Interface](https://github.com/Sinerror/Human-to-LLM-interface) — the control
  surface of section 4, in the browser.

## Licence

Code MIT; data CC BY-SA 4.0, because Wiktionary glosses are part of every sense
key and share-alike follows them. Files derived from the models — vectors,
generations and atlases — may also fall under each model's own licence. WordNet,
Wiktionary and model notices are in `LICENCE`. Model weights are referenced by
Hugging Face identifier and not redistributed.
