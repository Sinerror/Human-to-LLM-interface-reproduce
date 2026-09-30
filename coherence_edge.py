#!/usr/bin/env python3
"""
coherence_edge.py — E1. Find c*, the largest steering magnitude a model
tolerates, so that every later measurement is taken at a comparable point.

  python coherence_edge.py --model google/gemma-3-1b-it \
      --vecs lat_outputs/google_gemma-3-1b-it_vecs.npz \
      --labels frozen_emotionality_labels.csv --probes canonical_probes.json \
      --layer 17 --out edge_g31bit.json

WHY THIS EXISTS

  Steering magnitude has been chosen by hand in every run so far, and models
  differ in how much displacement they tolerate before text degrades. A model
  measured at 0.08 and a model measured at 0.06 are not being asked the same
  question, so "model A is more controllable than model B" has meant nothing.

  c* is defined here as the largest combined displacement, as a fraction of the
  token's residual norm, at which generated text retains most of the model's
  OWN unsteered fluency. Everything downstream runs at each model's own c*.

  **The baseline matters and an absolute threshold does not work.** Base
  checkpoints are repetitive before any intervention: gemma-3-1b-pt passes a
  fixed 75% criterion at no level, including c = 0.02, because its unsteered
  output already fails it. An absolute threshold therefore measures a model's
  natural text quality and its steering tolerance together, and reports the sum
  as tolerance. c = 0 is measured first and c* is defined relative to it.

  The curve is reported, not only the threshold. Where the edge sits, and how
  sharply text falls off past it, is a property of the model worth having.

THE FLUENCY CRITERION, FIXED IN ADVANCE

  A generation passes when all three hold:
    MATTR-50                    >= 0.70 x the model's own baseline MATTR
    longest repeated token run  <= 3
    longest repeating cycle     <= 2   (period 1-4)

  **Lexical diversity is measured by MATTR, not by the raw type-token ratio.**
  TTR falls with length by construction — common words recur — so a threshold
  calibrated on 80-token text becomes stricter at 200 tokens without anyone
  changing it. MATTR averages the type-token ratio over a sliding 50-token
  window and is flat in length: identical prose truncated to 40, 80, 120 or 160
  words scores 0.80, 0.78, 0.78, 0.78. The two also separate the cases that
  matter — natural prose 0.78 against 0.51 for a generation that repeats "I
  hate it" through a paragraph without any adjacent cycle.

  Only alphabetic tokens are counted. Newlines, punctuation and markup are
  excluded, so a model that formats with blank lines is not penalised for it.

  c = 0 is measured and gives the baseline pass rate p0. A steered level passes
  when its pass rate is at least `--retention` x p0. c* is the largest passing
  level. With retention 0.70, a model whose unsteered text passes 66% of the
  time needs 46% under steering, and one that passes 89% needs 62%.

  Diagnostics are reported as medians as well as means, because a handful of
  fully collapsed generations drags a mean run-length to 10 while most of the
  sample is intact.

  These thresholds are stated here so they cannot be adjusted after seeing the
  curve. They are deliberately permissive: text at c* is expected to be tinted,
  not clean, and the criterion is meant to exclude collapse rather than to
  certify quality.

WHAT IS STEERED

  A random triple of axes at the all-high corner, since that is the largest
  combined displacement any cell in the main factorial applies. Measuring the
  edge at a milder cell would set c* too high for the corner cases.
"""
import argparse, sys, json, os, re, sys
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


def mattr(words, win=50):
    """Moving-average type-token ratio. Length-independent, unlike raw TTR."""
    if not words:
        return 0.0
    if len(words) <= win:
        return len(set(words)) / len(words)
    return float(np.mean([len(set(words[i:i + win])) / win
                          for i in range(len(words) - win + 1)]))


def fluent(text, min_mattr=0.38, max_run=3, max_cycle=2):
    """Fixed criterion. Returns (passes, diagnostics).

    `min_mattr` is a fallback floor; the caller normally supplies a threshold
    derived from the model's own unsteered baseline."""
    toks = re.findall(r"\S+", str(text))
    # lexical tokens only: newlines, punctuation and markup are not repetition
    low = [t.lower() for t in re.findall(r"[A-Za-z']+", str(text))]
    if len(low) < 12:
        return False, dict(reason="too short", n=len(low))
    uniq = mattr(low)
    run = best = 1
    for i in range(1, len(low)):
        run = run + 1 if low[i] == low[i - 1] else 1
        best = max(best, run)
    # longest run of an immediately repeating block, counted in REPETITIONS.
    # "the storm the storm the storm" is 3 repetitions of a period-2 block; the
    # earlier integer-division form scored it 2 and let it through.
    cyc = 1
    for per in range(1, 5):
        i = 0
        while i + per <= len(low):
            reps = 1
            while (i + per * (reps + 1) <= len(low)
                   and low[i + per * reps: i + per * (reps + 1)]
                   == low[i: i + per]):
                reps += 1
            cyc = max(cyc, reps)
            i += per if reps == 1 else per * reps
    ok = (uniq >= min_mattr) and (best <= max_run) and (cyc <= max_cycle)
    return ok, dict(uniq=round(uniq, 3), max_run=best, max_cycle=cyc)


def find_blocks(model):
    import torch.nn as nn
    for path in ["model.layers", "gpt_neox.layers", "transformer.h",
                 "backbone.layers", "model.decoder.layers", "transformer.blocks"]:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 1:
            return obj, path
    best, bp = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 1:
            if best is None or len(mod) > len(best):
                best, bp = mod, name
    if best is None:
        raise RuntimeError(f"no block list in {type(model).__name__}")
    return best, bp


def build_axes(vecs, layer, labels, probes, names):
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
        if len(pos) >= 4 and len(neg) >= 4:
            P.append(unit(np.mean([wv[w] for w in pos], 0)
                          - np.mean([wv[w] for w in neg], 0)))
            kept.append(nm)
    return np.array(P), kept



def generate_chunked(model, tok, prompts, n_gen, max_new, dev, state, vecs,
                     batch_rows, seed):
    """Generate in row-chunks with automatic backoff on out-of-memory.

    The whole level is one logical batch, but a 9B model in bf16 cannot hold 64
    sequences of 200 tokens at once. Rows are chunked, and on an OOM the chunk
    size halves and the attempt repeats, down to a single row. `state["vec"]` is
    re-sliced per chunk so each row keeps its own steering vector.
    """
    import torch
    out_texts = []
    rows = len(prompts) * n_gen
    bs = max(1, min(batch_rows, rows))
    i = 0
    while i < len(prompts):
        take = max(1, bs // n_gen)
        chunk = prompts[i:i + take]
        state["vec"] = vecs[i:i + take].repeat_interleave(n_gen, dim=0)
        ids = tok(chunk, return_tensors="pt", padding=True).to(dev)
        try:
            torch.manual_seed(seed + i)
            with torch.inference_mode():
                o = model.generate(**ids, do_sample=True, temperature=0.9,
                                   top_p=0.95, num_return_sequences=n_gen,
                                   max_new_tokens=max_new,
                                   pad_token_id=tok.pad_token_id)
            out_texts += tok.batch_decode(o[:, ids["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            i += take
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= n_gen:
                raise RuntimeError(
                    f"out of memory even at one prompt x {n_gen} sequences. "
                    f"Reduce --n-gen or --max-new-tokens, or pass "
                    f"--device-map auto to allow CPU offload.")
            bs = max(n_gen, bs // 2)
            print(f"    OOM: retrying at {bs} rows per call", flush=True)
    return out_texts


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
    ap.add_argument("--axes", nargs="*",
                    default=["valence", "heat", "arousal", "intensity",
                             "antagonism_peace", "body_mind",
                             "social_inner_outer"])
    ap.add_argument("--levels", nargs="*", type=float,
                    default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20,
                             0.25, 0.32, 0.40, 0.50, 0.65, 0.80, 1.00],
                    help="ladder of combined displacement fractions. Extends to "
                         "1.00, where the added vector matches the residual norm "
                         "and the intervention replaces rather than tilts the "
                         "state. Nothing is expected to survive there; the point "
                         "is to see the whole curve rather than assume its shape.")
    ap.add_argument("--n-triples", type=int, default=8,
                    help="axis triples sampled per level; each contributes "
                         "n-gen generations, so a level is scored on "
                         "n_triples * n_gen texts")
    ap.add_argument("--n-gen", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=200,
                    help="longer than the factorial's 80. Degeneration often "
                         "appears only after the first sentence or two, so a "
                         "short window overstates fluency and sets c* too high.")
    ap.add_argument("--n-samples", type=int, default=3,
                    help="example texts stored per level")
    ap.add_argument("--sample-chars", type=int, default=400)
    ap.add_argument("--retention", type=float, default=0.70,
                    help="fraction of the UNSTEERED pass rate a level must "
                         "retain. Relative rather than absolute, because base "
                         "checkpoints fail an absolute criterion before any "
                         "steering is applied.")
    ap.add_argument("--mattr-retention", type=float, default=0.70,
                    help="a generation's MATTR must reach this fraction of the "
                         "model's own unsteered median MATTR")
    ap.add_argument("--min-abs", type=float, default=0.25,
                    help="floor: a level must also pass this fraction outright, "
                         "so a model with a very poor baseline cannot qualify "
                         "at a magnitude where text is plainly destroyed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-rows", type=int, default=64,
                    help="sequences generated per call. Halves automatically on "
                         "out-of-memory, so this is a starting point rather than "
                         "a limit. 64 suits a 1-2B model at 200 tokens; try 16 "
                         "for 7B and 8 for 9B if the backoff is slow.")
    ap.add_argument("--device-map", default=None,
                    help="pass 'auto' to let accelerate shard across GPU and CPU "
                         "for models that do not fit. Slower, but it runs.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    labels = load_labels(a.labels)
    probes = json.load(open(a.probes, encoding="utf-8"))
    P, names = build_axes(a.vecs, a.layer, labels, probes, a.axes)
    print(f"{a.model}  layer {a.layer}  {len(names)} axes")

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dev = torch.device(a.device)
    model, dev = load_model(a.model, a.device, a.device_map)
    blocks, bpath = find_blocks(model)
    if a.layer < 1 or a.layer > len(blocks):
        sys.exit(f"--layer {a.layer} out of range: {len(blocks)} blocks at {bpath}")
    block = blocks[a.layer - 1]
    print(f"  hooking block {a.layer-1} of {len(blocks)} at {bpath}")

    state = {"vec": None, "c": 0.0, "mattr_floor": 0.0}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        v = state["vec"]
        if v is None:
            return out
        h = h + state["c"] * h.norm(dim=-1, keepdim=True) * v[:, None, :]
        return (h,) + out[1:] if isinstance(out, tuple) else h

    handle = block.register_forward_hook(hook)

    rng = np.random.default_rng(SEED)
    from itertools import combinations
    all_tri = list(combinations(range(len(names)), 3))
    tri_idx = [all_tri[i] for i in
               rng.choice(len(all_tri), min(a.n_triples, len(all_tri)),
                          replace=False)]
    # all-high corner: the largest displacement any cell in the factorial applies
    vecs = np.array([unit(P[list(t)].sum(0)) for t in tri_idx])
    Vt = torch.tensor(vecs, dtype=torch.bfloat16, device=dev)
    stems = [STEMS[i % len(STEMS)] for i in range(len(tri_idx))]

    levels = [0.0] + [c for c in a.levels if c > 0]
    results = []
    print(f"\n  scoring {a.n_triples * a.n_gen} generations per level, "
          f"{a.max_new_tokens} tokens each")
    print(f"\n  {'c':>6}{'pass':>7}{'uniq~':>7}{'run~':>6}{'cyc~':>6}"
          f"{'run avg':>7}{'cyc avg':>7}{'chars':>7}")
    print("  (~ = median; a few collapsed generations drag the averages)")
    for c in levels:
        state["c"] = float(c)
        texts = generate_chunked(model, tok, stems, a.n_gen, a.max_new_tokens,
                                 dev, state, Vt, a.batch_rows, SEED)
        checks = [fluent(t, min_mattr=state["mattr_floor"]) for t in texts]
        rate = float(np.mean([ok for ok, _ in checks]))
        dg = [d for _, d in checks if "uniq" in d]
        mu = float(np.mean([d["uniq"] for d in dg])) if dg else 0.0
        mr = float(np.mean([d["max_run"] for d in dg])) if dg else 0.0
        mc = float(np.mean([d["max_cycle"] for d in dg])) if dg else 0.0
        mlen = float(np.mean([len(t) for t in texts]))
        med_r = float(np.median([d["max_run"] for d in dg])) if dg else 0.0
        med_c = float(np.median([d["max_cycle"] for d in dg])) if dg else 0.0
        med_u = float(np.median([d["uniq"] for d in dg])) if dg else 0.0
        results.append(dict(c=float(c), pass_rate=rate, mean_uniq=mu,
                            median_uniq=med_u, mean_max_run=mr,
                            median_max_run=med_r, mean_max_cycle=mc,
                            median_max_cycle=med_c,
                            mean_chars=mlen, n_scored=len(texts),
                            samples=[t[:a.sample_chars] for t in texts[:a.n_samples]]))
        if c == 0.0:
            # threshold set from the model's own unsteered lexical diversity
            state["mattr_floor"] = a.mattr_retention * float(np.median(
                [d["uniq"] for d in dg])) if dg else 0.0
            checks = [fluent(t, min_mattr=state["mattr_floor"]) for t in texts]
            rate = float(np.mean([ok for ok, _ in checks]))
            results[-1]["pass_rate"] = rate
            print(f"         (baseline MATTR {np.median([d['uniq'] for d in dg]):.3f}"
                  f" -> floor {state['mattr_floor']:.3f})")
        tag = "  baseline" if c == 0.0 else ""
        print(f"  {c:>6.3f}{rate:>7.0%}{med_u:>7.2f}{med_r:>6.0f}{med_c:>6.0f}"
              f"{mr:>7.1f}{mc:>7.1f}{mlen:>7.0f}{tag}")

    handle.remove()
    p0 = results[0]["pass_rate"]
    need = max(a.retention * p0, a.min_abs)
    passing = [r["c"] for r in results
               if r["c"] > 0 and r["pass_rate"] >= need]
    # the edge is where the curve first breaks, not any later level that
    # recovers by chance or by the model shortening its output
    c_star = None
    for r in results:
        if r["c"] == 0.0:
            continue
        if r["pass_rate"] >= need:
            c_star = r["c"]
        else:
            break
    out = dict(model=a.model, layer=a.layer, axes=names,
               criterion=dict(min_uniq=0.55, max_run=3, max_cycle=2,
                              retention=a.retention, min_abs=a.min_abs),
               baseline_pass_rate=p0, required_pass_rate=need,
               levels=results, c_star=c_star,
               c_star_any=max(passing) if passing else None, seed=SEED)
    json.dump(out, open(a.out, "w"), indent=1)

    # print one full example at, just below, and just above the edge
    print("\n" + "=" * 74)
    print("SAMPLES")
    print("=" * 74)
    show = a.levels
    if c_star is not None:
        i = [r["c"] for r in results].index(c_star)
        show = [results[j]["c"] for j in
                sorted({max(0, i - 1), i, min(len(results) - 1, i + 1),
                        len(results) - 1})]
    for r in results:
        if r["c"] not in show:
            continue
        tag = "  <- c*" if c_star is not None and r["c"] == c_star else ""
        print(f"\nc = {r['c']:.3f}   pass {r['pass_rate']:.0%}{tag}")
        for t in r["samples"][:2]:
            print(f"    {t!r}")

    print()
    print(f"  baseline (c=0) pass rate {p0:.0%}")
    print(f"  required: {a.retention:.0%} of baseline = {need:.0%}")
    if c_star is None:
        print(f"  NO LEVEL PASSED. Even c={levels[1]} falls below {need:.0%}.")
        print(f"  Extend --levels downward, or the model does not tolerate")
        print(f"  steering at any magnitude this ladder covers.")
    else:
        print(f"  c* = {c_star}   (last level before the curve breaks)")
        if max(passing) > c_star:
            print(f"  note: level(s) above c* also clear the bar "
                  f"(up to {max(passing)}), but the curve broke first at c*. "
                  f"Recovery above a break usually means the model shortened "
                  f"its output rather than improved it — check mean chars.")
        if c_star == max(a.levels):
            print(f"  !! c* is the top of the ladder; the true edge is higher.")
            print(f"     Extend --levels upward or the protocol understates this")
            print(f"     model's tolerance.")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
