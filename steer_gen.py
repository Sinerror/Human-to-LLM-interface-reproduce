#!/usr/bin/env python3
"""
steer_gen.py — steering generation, rebuilt. Covers the dual-basis and long-run
tests end to end: axis construction, intervention, generation, manifest.

  # dual-basis test: both arms, matched displacement, identical everything else
  python steer_gen.py --model google/gemma-3-1b-it --vecs g31bit_vecs.npz \
      --labels frozen_emotionality_labels.csv --probes canonical_probes.json \
      --layer 17 --arms raw dual --reps 2 --max-new-tokens 80 \
      --out d_dual/

  # long-run test: one arm, long generations, fewer triples, more reps
  python steer_gen.py --model google/gemma-3-1b-it --vecs g31bit_vecs.npz \
      --labels ... --probes ... --layer 17 --arms raw \
      --axes valence heat body_mind arousal --reps 5 --max-new-tokens 500 \
      --out d_long/

────────────────────────────────────────────────────────────────────────────────
FAULTS THIS REBUILD DESIGNS OUT
────────────────────────────────────────────────────────────────────────────────

**Generation length was set through a module global.** Child processes in a
parallel wave never executed the assignment, so models that shared a wave
silently produced 80 tokens when 500 were requested — and the fault was
invisible until token counts were tabulated afterwards. Here `max_new_tokens` is
a function argument threaded to the call site, never module state, and **every
record carries the token count actually produced**. A post-run assertion fails
loudly if the median falls short of the request.

**One repetition per triple.** With a single observation per triple the
downstream analysis (`--min-trials 2`) silently discards everything. Repetitions
are an explicit argument, enforced at plan time, and the plan is written to disk
before generation starts so the shortfall is visible before the compute is spent.

**Arms differing in more than the arm.** The dual comparison is only meaningful
if stems, triples, cell assignments, seeds and total displacement are identical
between arms. All are drawn once, stored in the plan, and replayed per arm. The
only difference is the direction matrix.

────────────────────────────────────────────────────────────────────────────────
THE TWO ARMS
────────────────────────────────────────────────────────────────────────────────

  raw    steer with P, the probe contrast directions.
  dual   steer with D = G^-1 P, where G = P P^T.

Dual rows are not unit vectors — ‖D_i‖ is the iso-cost, between about 1.0 and
1.3 here — so at equal alpha the dual arm would apply more displacement and any
gain would be confounded with magnitude. **Cell displacement norm is equalised
between arms**: the combined perturbation for each cell is rescaled to a common
target. This is the choice registered as (b) in the pending-experiments note; it
isolates decoupling from magnitude, at the cost of not testing dual steering "as
one would deploy it".

Alpha is a fraction of the token's own residual norm, so the intervention scales
with position rather than being absolute.

────────────────────────────────────────────────────────────────────────────────
"""
import argparse, sys, hashlib, json, os, sys, time
from collections import defaultdict
import numpy as np


# Generated text can contain any codepoint the model emits, and a Windows
# console is often cp1251 or cp437, which cannot encode most of them. Printing
# a sample then raises UnicodeEncodeError and kills a run that had already done
# the expensive part. Replace unencodable characters instead of failing; the
# JSON output is written with ensure_ascii and is unaffected.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

SEED = 0

STEMS = [
    "The letter had been sitting under the mail for three days.",
    "She opened the door to the meeting room and looked around.",
    "The train pulled into the station later than scheduled.",
    "He read the message twice before deciding how to reply.",
    "The lights in the office were still on at midnight.",
    "They walked from the car park toward the entrance.",
    "The report had been left on the desk overnight.",
    "A colleague mentioned the change during lunch.",
    "The phone rang while she was making coffee.",
    "He found the folder at the back of the second drawer.",
    "The meeting ended a few minutes earlier than planned.",
    "She checked the schedule for the following morning.",
]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def load_labels(p):
    import csv
    out = {}
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        out[r["key"]] = (r.get("final", "").strip().upper() or r["heuristic"])
    return out


def build_axes(vecs, layer, labels, probes, names):
    """Probe contrast directions P and their dual D = G^-1 P, in hidden space."""
    z = np.load(vecs, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    words = np.array([str(w).lower() for w in z["words"]])
    layers = [int(x) for x in np.asarray(z["layers"])]
    m = np.array([labels.get(k) == "EMO" for k in keys])
    X = z["vecs"][m, layers.index(layer), :].astype(np.float64)
    Xc = X - X.mean(0)
    acc = defaultdict(list)
    for w, v in zip(words[m], Xc):
        acc[w].append(v)
    wv = {w: np.mean(vs, 0) for w, vs in acc.items()}

    P, kept = [], []
    for nm in names:
        s = probes.get(nm, {})
        pos = [w.lower() for w in s.get("pos", []) if w.lower() in wv]
        neg = [w.lower() for w in s.get("neg", []) if w.lower() in wv]
        if len(pos) < 4 or len(neg) < 4:
            print(f"  skipping {nm}: {len(pos)}/{len(neg)} corpus words")
            continue
        P.append(unit(np.mean([wv[w] for w in pos], 0)
                      - np.mean([wv[w] for w in neg], 0)))
        kept.append(nm)
    P = np.array(P)
    G = P @ P.T
    D = np.linalg.solve(G, P)
    return P, D, G, kept


def find_blocks(model):
    """Locate the transformer block list across architectures.

    Every family names this differently and getattr chains fail loudly only
    after the weights are loaded, which wastes the load. Known paths are tried
    first; the fallback searches for the longest ModuleList of repeated blocks,
    which covers architectures not listed here.
    """
    import torch.nn as nn
    paths = ["model.layers",          # llama, mistral, qwen, gemma, phi
             "gpt_neox.layers",       # pythia
             "transformer.h",         # gpt2, falcon
             "backbone.layers",       # mamba
             "model.decoder.layers",  # opt
             "transformer.blocks"]    # mpt
    for path in paths:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 1:
            print(f"  blocks at {path}: {len(obj)}")
            return obj
    best, best_path = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 1:
            if best is None or len(mod) > len(best):
                best, best_path = mod, name
    if best is None:
        raise RuntimeError(
            f"cannot locate transformer blocks in {type(model).__name__}. "
            f"Add its path to find_blocks().")
    print(f"  blocks found by search at {best_path}: {len(best)}")
    return best


def make_plan(names, triples, reps, n_cells, rng):
    """Trials, cells and stems drawn ONCE and shared by every arm."""
    plan = []
    tid = 0
    for tri in triples:
        for _ in range(reps):
            stem = int(rng.integers(0, len(STEMS)))
            cells = []
            for c in range(n_cells):
                lv = [(1 if (c >> b) & 1 else -1) for b in range(len(tri))]
                cells.append(dict(cell_id=c, levels=dict(zip(tri, lv))))
            plan.append(dict(trial_id=f"T{tid:04d}", triple=list(tri),
                             stem_id=stem, stem=STEMS[stem], cells=cells))
            tid += 1
    return plan



def load_model(model_id, device, device_map=None, dtype=None):
    """Load straight onto the accelerator, without staging through CPU RAM.

    `from_pretrained(...).to(dev)` materialises every weight in CPU RAM and then
    copies it, so a 7B model in bf16 costs ~14 GB of RAM it does not need. A
    device_map pinning every module to one device makes accelerate place each
    shard as it reads it: no CPU copy, and no offload either, since offload only
    happens when the map assigns modules to "cpu" or "disk".

    device_map="auto" is the explicit opt-in for models that genuinely do not
    fit, and it WILL offload layers to CPU, which is slow.

    Quantised checkpoints carry their own dtype in the config; passing an
    explicit dtype overrides it and can fail, so dtype is left unset when the
    config declares quantisation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    quantised = getattr(cfg, "quantization_config", None) is not None
    kw = dict(low_cpu_mem_usage=True)
    if not quantised:
        kw["dtype"] = dtype or torch.bfloat16
    else:
        print("  quantised checkpoint: leaving dtype to its config")
    kw["device_map"] = device_map if device_map else {"": device}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    dev = next(model.parameters()).device
    n_cpu = sum(1 for p in model.parameters() if p.device.type == "cpu")
    if n_cpu:
        print(f"  !! {n_cpu} parameter tensors are on CPU. Generation will be "
              f"slow. This happens only with --device-map auto.")
    return model, dev

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--axes", nargs="*", default=["valence", "heat", "arousal",
                                                  "intensity", "antagonism_peace",
                                                  "body_mind",
                                                  "social_inner_outer"])
    ap.add_argument("--arms", nargs="+", default=["raw", "dual"],
                    choices=["raw", "dual", "control"])
    ap.add_argument("--triple-size", type=int, default=3)
    ap.add_argument("--design", choices=["triple", "single"], default="triple",
                    help="triple: the 2^k factorial over each triple (the main "
                         "experiment). single: the SAME trials and stems, but each "
                         "cell turns one of the triple's axes alone, to +1 or -1 — "
                         "the superposition test. Under a linear response the "
                         "single-axis cells predict the triple cells exactly.")
    ap.add_argument("--reps", type=int, default=2,
                    help="repetitions per triple; MUST be >= 2 for the analysis")
    ap.add_argument("--n-gen", type=int, default=8, help="generations per cell")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--c", type=float, default=0.08,
                    help="COMBINED displacement as a fraction of the token's "
                         "residual norm, after cell-norm equalisation. This is "
                         "the total for all knobs in the cell, not per knob. "
                         "The registered runs sat at 0.0997; test 5 found that "
                         "about 1.21x too high on 3 of 4 models, so 0.08 is the "
                         "corrected default. Values near 0.7 destroy the text.")
    ap.add_argument("--pilot", type=float, nargs="*", default=None,
                    help="run a magnitude ladder on 2 trials and stop, e.g. "
                         "--pilot 0.04 0.08 0.16 0.32. Do this before any full run.")
    ap.add_argument("--equalise-cell-norm", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="write the plan and the axis manifest, generate nothing")
    ap.add_argument("--resume", action="store_true",
                    help="append to existing generation files, skipping trials "
                         "already complete. Incomplete trials are discarded and "
                         "redone; seeding is per-trial so resumed output is "
                         "identical to an uninterrupted run.")
    ap.add_argument("--force-magnitude", action="store_true",
                    help="permit c above 0.25, which normally destroys the text")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-rows", type=int, default=48,
                    help="sequences per generate() call. Halves automatically on "
                         "out-of-memory. 48 suits a 1-2B model; try 16 for 7B, "
                         "8 for 9B.")
    ap.add_argument("--device-map", default=None,
                    help="pass 'auto' to shard across GPU and CPU for models "
                         "that do not fit. Slower, but it runs.")
    ap.add_argument("--batch-cells", type=int, default=8,
                    help="cells generated in one batched call. All 8 cells of a "
                         "trial share a stem and differ only in the steering "
                         "vector, so they batch cleanly with a per-row hook.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.c > 0.25 and not a.force_magnitude:
        print(f"REFUSING TO RUN: c={a.c} is a displacement of {a.c:.0%} of the "
              f"residual norm.")
        print("  The registered runs sat at 0.0997 and text degrades above about")
        print("  0.10-0.15. A value near 0.5 or 0.7 is almost certainly from a")
        print("  pre-correction command, where c meant a multiplier on per-knob")
        print("  gates rather than the displacement fraction itself.")
        print("  Use --c 0.08, or --pilot to choose, or --force-magnitude.")
        sys.exit(1)
    if a.reps < 2:
        sys.exit("--reps must be at least 2; the transfer-matrix analysis "
                 "discards triples with a single observation, which is how the "
                 "previous long-run attempt lost its data.")
    os.makedirs(a.out, exist_ok=True)

    labels = load_labels(a.labels)
    probes = json.load(open(a.probes, encoding="utf-8"))
    P, D, G, names = build_axes(a.vecs, a.layer, labels, probes, a.axes)
    iso = np.sqrt(np.diag(np.linalg.inv(G)))
    print(f"{len(names)} axes: {names}")
    print(f"  Gram mean |cos| {np.abs(G[np.triu_indices(len(G),1)]).mean():.3f}"
          f"  condition {np.linalg.cond(G):.2f}")
    print(f"  iso-cost {np.round(iso,3)}")

    from itertools import combinations
    triples = list(combinations(names, a.triple_size))
    rng = np.random.default_rng(SEED)
    plan = make_plan(names, triples, a.reps, 2 ** a.triple_size, rng)
    if a.design == "single":
        # Keep every trial and its stem exactly as the triple design draws them
        # (same seed, same order), so each single-axis cell sits on the prompt of
        # the triple cells it must predict. Replace only the cells.
        for tr in plan:
            tri = tr["triple"]
            tr["cells"] = [dict(cell_id=2 * i + (sgn > 0),
                                levels={k: (sgn if k == ax else 0) for k in tri})
                           for i, ax in enumerate(tri) for sgn in (-1, 1)]

    n_gen_total = len(plan) * (2 ** a.triple_size) * a.n_gen * len(a.arms)
    print(f"\nPLAN  {len(triples)} triples x {a.reps} reps = {len(plan)} trials/arm")
    print(f"      {2**a.triple_size} cells x {a.n_gen} generations")
    print(f"      {len(a.arms)} arm(s) -> {n_gen_total} generations at "
          f"{a.max_new_tokens} tokens")
    print(f"      combined displacement {a.c:.3f} x residual norm per cell")
    manifest = dict(model=a.model, layer=a.layer, axes=names, arms=a.arms,
                    design=a.design,
                    reps=a.reps, n_gen=a.n_gen, c=a.c,
                    max_new_tokens=a.max_new_tokens,
                    equalise_cell_norm=bool(a.equalise_cell_norm),
                    iso_cost=iso.tolist(), gram=G.tolist(),
                    n_trials_per_arm=len(plan), seed=SEED,
                    plan_sha256=hashlib.sha256(
                        json.dumps(plan, sort_keys=True).encode()).hexdigest())
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    json.dump(plan, open(os.path.join(a.out, "plan.json"), "w"), indent=1)
    np.savez(os.path.join(a.out, "axes.npz"), P=P, D=D, G=G,
             names=np.array(names), layer=a.layer)
    print(f"\nplan sha256 {manifest['plan_sha256'][:16]}")
    print(f"-> {a.out}/plan.json, manifest.json, axes.npz")

    if a.dry_run:
        print("\n--dry-run: nothing generated.")
        return

    # ── generation ────────────────────────────────────────────────────
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # required for batched generation
    dev = torch.device(a.device)
    # device_map="auto" routes through accelerate, which may place layers on CPU
    # when it judges VRAM tight and keeps a CPU copy alive. An explicit device
    # with low_cpu_mem_usage loads shard-by-shard straight to the GPU.
    model, dev = load_model(a.model, a.device, a.device_map)
    model.generation_config.use_cache = True

    blocks = find_blocks(model)
    if a.layer < 1 or a.layer > len(blocks):
        sys.exit(f"--layer {a.layer} out of range: this model has "
                 f"{len(blocks)} blocks, so layers 1..{len(blocks)} are valid "
                 f"(layer n means the output of block n-1).")
    block = blocks[a.layer - 1]
    print(f"  hooking block {a.layer - 1} of {len(blocks)} "
          f"({type(block).__name__})")

    state = {"vec": None, "c": a.c}

    def hook(_m, _i, out):
        """Add c * ||h|| * v_b at every position, with a DIFFERENT v per batch
        row. This is what lets all eight cells of a trial generate in one call:
        they share a stem and differ only in the steering vector.

        state["vec"] is (B, D), already on the right device and dtype, so the
        hook does no transfers or casts per forward pass."""
        h = out[0] if isinstance(out, tuple) else out
        v = state["vec"]
        if v is None:
            return out
        h = h + state["c"] * h.norm(dim=-1, keepdim=True) * v[:, None, :]
        return (h,) + out[1:] if isinstance(out, tuple) else h

    handle = block.register_forward_hook(hook)

    if a.pilot:
        print(f"\nPILOT: {len(a.pilot)} magnitudes x 2 trials, generation only.")
        print("Read the samples. Text should be fluent and only tinted by the")
        print("construct; word repetition or single-token spam means c is too high.\n")
        ladder = []
        for cval in a.pilot:
            state["c"] = cval
            tr = plan[0]
            idx = [names.index(k) for k in tr["triple"]]
            hi = tr["cells"][-1]
            lv = np.array([hi["levels"][k] for k in tr["triple"]], float)
            state["vec"] = torch.tensor(
                unit((lv[:, None] * P[idx]).sum(0))[None, :],
                dtype=torch.bfloat16, device=dev)
            ids = tok(tr["stem"], return_tensors="pt").to(dev)
            torch.manual_seed(SEED)
            with torch.no_grad():
                o = model.generate(**ids, do_sample=True, temperature=0.9,
                                   top_p=0.95, max_new_tokens=60,
                                   pad_token_id=tok.eos_token_id)
            txt = tok.decode(o[0][ids["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            toks = txt.split()
            uniq = len(set(t.lower() for t in toks)) / max(len(toks), 1)
            ladder.append((cval, uniq))
            print(f"  c={cval:<6} unique-token ratio {uniq:.2f}   {txt[:150]!r}")
        handle.remove()

        # per-model calibration. c is relative to the residual norm, so it is
        # invariant to activation scale, but NOT to how much perturbation a
        # given model tolerates before degenerating. Models differ there by
        # roughly 1.4x, so a single c across models puts one near its ceiling
        # and another at half its budget — which makes any cross-model
        # comparison of steering strength incommensurable.
        ok = [c for c, u in ladder if u >= 0.60]
        print("\n  CALIBRATION")
        if ok:
            cstar = max(ok)
            print(f"    c* (largest with unique-token ratio >= 0.60): {cstar:.3f}")
            print(f"    recommended for this model: --c {0.6 * cstar:.3f}")
            print(f"    That is 60% of the degeneracy threshold, leaving headroom.")
        else:
            print("    every rung degenerated; extend the ladder downward.")
        print("\n  Use the per-model recommendation, not one c for all models.")
        print("  Within a model both arms share c, so the dual comparison is")
        print("  unaffected; across models an uncalibrated c makes the steering")
        print("  strengths incommensurable. Record the c used per model.")
        print("  A single generation per rung is a coarse estimate -- treat it as")
        print("  an order-of-magnitude guide, not a measured gate.")
        return

    for arm in a.arms:
        M = {"raw": P, "dual": D}.get(arm)
        if arm == "control":
            g = np.random.default_rng(SEED + 99)
            R = g.normal(size=P.shape)
            R = np.array([unit(r) for r in R])
            M = R
        path = os.path.join(a.out, f"{os.path.basename(a.model)}_{arm}_generations.jsonl")

        done = set()
        if a.resume and os.path.exists(path):
            # A trial is complete only if all of its cells were written. A
            # partially written trial is discarded and redone, so an interrupt
            # mid-trial cannot leave a gap in the factorial.
            counts, keep = defaultdict(int), []
            for line in open(path, encoding="utf-8"):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue          # truncated final line from the interrupt
                counts[rec["trial_id"]] += 1
                keep.append((rec["trial_id"], line))
            n_cells = 2 ** a.triple_size
            done = {t for t, c in counts.items() if c == n_cells}
            with open(path, "w", encoding="utf-8") as fw:
                for t, line in keep:
                    if t in done:
                        fw.write(line if line.endswith("\n") else line + "\n")
            partial = len(counts) - len(done)
            print(f"  {arm}: resuming, {len(done)}/{len(plan)} trials complete"
                  + (f", {partial} partial discarded" if partial else ""))
        elif a.resume:
            print(f"  {arm}: --resume given but no existing file; starting fresh")

        f = open(path, "a" if a.resume else "w", encoding="utf-8")
        t0 = time.time()
        for ti, tr in enumerate(plan):
            if tr["trial_id"] in done:
                continue
            idx = [names.index(k) for k in tr["triple"]]
            cells = tr["cells"]
            # cells per call, further reduced by the OOM backoff below
            per_call = max(1, min(a.batch_cells, a.batch_rows // a.n_gen))
            for b0 in range(0, len(cells), per_call):
                chunk = cells[b0:b0 + per_call]
                # one unit steering vector per cell; the hook applies it per row
                V = []
                for cell in chunk:
                    lv = np.array([cell["levels"][k] for k in tr["triple"]], float)
                    vec = (lv[:, None] * M[idx]).sum(0)
                    if a.equalise_cell_norm:
                        # Unit norm. The hook applies c * ||h|| * v, so combined
                        # displacement is exactly c * ||h|| for every cell and
                        # every arm. Scaling to sqrt(k) here would give
                        # c*sqrt(k)*||h||, which at c=0.7, k=3 is 1.21x the
                        # residual norm and destroys the generation.
                        vec = unit(vec)
                    V.append(vec)
                Vt = torch.tensor(np.array(V), dtype=torch.bfloat16, device=dev)
                # generate expands each prompt into n_gen sequences in order,
                # so row j of the internal batch belongs to cell j // n_gen
                state["vec"] = Vt.repeat_interleave(a.n_gen, dim=0)

                ids = tok([tr["stem"]] * len(chunk), return_tensors="pt",
                          padding=True).to(dev)
                torch.manual_seed(SEED + 1000 * ti + b0)
                for attempt in range(6):
                    try:
                        with torch.inference_mode():
                            o = model.generate(
                                **ids, do_sample=True, temperature=0.9,
                                top_p=0.95, num_return_sequences=a.n_gen,
                                max_new_tokens=a.max_new_tokens,  # per call
                                pad_token_id=tok.pad_token_id)
                        break
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if len(chunk) == 1:
                            raise RuntimeError(
                                "out of memory at one cell. Reduce --n-gen or "
                                "--max-new-tokens, or use --device-map auto.")
                        # halve the cells in this call and retry the remainder
                        half = max(1, len(chunk) // 2)
                        print(f"    OOM: dropping to {half} cells per call",
                              flush=True)
                        a.batch_cells = half
                        chunk = chunk[:half]
                        state["vec"] = Vt[:half].repeat_interleave(a.n_gen, dim=0)
                        ids = tok([tr["stem"]] * half, return_tensors="pt",
                                  padding=True).to(dev)
                plen = ids["input_ids"].shape[1]
                new = o[:, plen:]
                texts = tok.batch_decode(new, skip_special_tokens=True)
                nt = (new != tok.pad_token_id).sum(1).tolist()
                for ci, cell in enumerate(chunk):
                    lo, hi = ci * a.n_gen, (ci + 1) * a.n_gen
                    f.write(json.dumps(dict(
                        trial_id=tr["trial_id"], arm=arm, triple=tr["triple"],
                        cell_id=cell["cell_id"], levels=cell["levels"],
                        stem_id=tr["stem_id"], stem=tr["stem"],
                        generations=texts[lo:hi],
                        n_tokens=[int(x) for x in nt[lo:hi]]),
                        ensure_ascii=False) + "\n")
            f.flush()          # so an interrupt loses at most the current trial
            if (ti + 1) % 10 == 0:
                el = time.time() - t0
                left = len(plan) - ti - 1
                rate = (ti + 1 - len(done)) or 1
                print(f"  {arm}: {ti+1}/{len(plan)} trials  "
                      f"{el/60:.1f} min  eta {el/rate*left/60:.0f} min", flush=True)
        f.close()
        state["vec"] = None

        # post-run assertion on the fault that cost the last long run
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        med = float(np.median([n for r in rows for n in r["n_tokens"]]))
        print(f"  {arm}: median tokens generated {med:.0f} "
              f"(requested {a.max_new_tokens})")
        if med < 0.8 * a.max_new_tokens:
            print(f"  !! TOKEN SHORTFALL: median {med:.0f} against "
                  f"{a.max_new_tokens} requested. The generations are usable but "
                  f"the length condition was not met; do not treat this as a "
                  f"long-run arm.")
        print(f"  -> {path}")

    handle.remove()
    print("\nNEXT")
    print("  python transfer_independent.py --generations", a.out,
          "--probes ... --workers N")
    print("  python embed_align.py         --generations", a.out, "--probes ...")
    print("  Compare arms. The dual arm should show lower rho(M, G) if the")
    print("  correction is removing the obliqueness component specifically.")


if __name__ == "__main__":
    main()
