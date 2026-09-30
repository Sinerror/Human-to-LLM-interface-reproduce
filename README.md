# Reproduce — rebuild everything from the prompts

The full chain behind *A construct-defined steering basis for affect in language
models*: extraction, validation, coherence edges, steered generation, analysis,
and the interface atlas. Staged and resumable.

```
python reproduce.py --model google/gemma-3-1b-it --layer 17 --work repro --code . --check
python reproduce.py --model google/gemma-3-1b-it --layer 17 --work repro --code . --dry-run
python reproduce.py --model google/gemma-3-1b-it --layer 17 --work repro --code .
```

`--check` confirms every input, script and package is present before any GPU
time is spent. `--dry-run` prints every command it would run.

## Where reproducibility starts

**At the prompts, not the corpus.** The sentence set in `frozen_prompts.jsonl`
was produced by a sampling model and cannot be regenerated identically, so it
ships as a fixed, hashed artifact. From there the chain is forward passes with
fixed seeds.

On the same machine the chain reproduces closely but not always exactly. A full
rerun of gemma-3-1b-it reproduced all 70 transfer matrices in both arms with a
maximum difference of exactly 0.0. A rerun of gemma-3-1b-pt reproduced every
qualitative result — the same c\*, every axis positive, the cross-alignment
diagonal maximal in all seven rows — but per-axis gains differed by up to 0.067.
Sampled generation is sensitive to the smallest change in a logit, and a single
flipped token changes everything after it. Across machines, bf16 accumulation order
varies with batch size and hardware, so expect vectors to agree to about 1e-3 and
sampled text to diverge outright. The tolerances in the verify package exist to
absorb exactly that.

## Stages

| | stage | produces | needs |
|---|---|---|---|
| 1 | `extract` | corpus vectors, all layers | GPU |
| 2 | `heldout` | vectors for the pre-registered unseen words | GPU |
| 3 | `validate` | leave-one-word-out and unseen-word tests | CPU |
| 4 | `edge` | the coherence edge c* for this model | GPU |
| 5 | `layers` | per-layer held-out AUC, merged edges | CPU |
| 6 | `generate` | raw, dual and control arms at c* | GPU |
| 7 | `analyse` | alignment and transfer matrices | CPU |
| 8 | `atlas` | the interface's token atlas | GPU |
| 9 | `package` | the verify package's data tree | CPU |

Each stage writes a receipt recording the sha256 of its inputs. A rerun skips
any stage whose receipt is present, whose inputs are unchanged, and whose
outputs still exist. **Change the corpus and every downstream stage
invalidates** rather than silently mixing new inputs with old outputs.

```
python reproduce.py ... --status           review
python reproduce.py ... --from generate    resume
python reproduce.py ... --force atlas      redo one stage
```

On failure it prints the tail of the output, writes the full log to
`<work>/<model>/run.log`, and names the `--from` to resume with.

## What you need beside the scripts

| file | role |
|---|---|
| `frozen_prompts.jsonl` | the deterministic starting point |
| `frozen_emotionality_labels.csv` | **the corrected file.** A superseded version differs by seven renamed keys and two additions, and using it shifts every geometry figure without raising an error |
| `canonical_probes.json` | the eighteen word lists |
| `heldout_prompts.jsonl` | prompts for the pre-registered unseen words |
| `predictions.csv` | **the pre-registration**, `word,probe,side`, written before any extraction. No stage produces it and none should: regenerating it would destroy the property that makes it worth reporting |
| `ekman.json` | optional; the atlas's category readout |

Without `heldout_prompts.jsonl` and `predictions.csv` the chain still runs, but
the unseen-word test is skipped and the package reports that it cannot run.

## Nine models

Per-model settings used in the paper:

| model | `--layer` | `--batch-rows` | c* |
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

c* is measured by the `edge` stage, not supplied. It is listed so a
reproduction can be checked against it. Tolerance varies more than tenfold
across these models; a magnitude taken from one model either does nothing to
another or destroys its text.

For Phi-2 and Mistral the selected layer is the network's last, so the
selection criterion is reporting the boundary of its range rather than an
interior optimum.

After all nine:

```
python package_verify.py --work repro --models <nine model slugs> \
    --labels frozen_emotionality_labels.csv --probes canonical_probes.json \
    --predictions predictions.csv --out emotion-basis-verify
```

## Memory

Models load straight to the GPU through a device map, never staged through CPU
RAM. Generation batches halve automatically on out-of-memory. If a model still
does not fit, `--device-map auto` shards across GPU and CPU; it runs, slowly,
and the loader warns which tensors landed on CPU.

## What is also in this archive

`superseded/` holds runs made at hand-picked steering magnitudes before the
coherence-edge protocol existed. They are kept, labelled, because the paper
states that one of them produced an artefactual figure, and that claim is only
checkable with the data. `claims_log.md` records every retraction.

## Licence

Code MIT, data CC BY 4.0. WordNet and Wiktionary notices in `LICENCE`. Model
weights are referenced by Hugging Face identifier and not redistributed.
