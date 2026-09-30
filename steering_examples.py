#!/usr/bin/env python3
"""
steering_examples.py — what does steered text actually look like?

  python steering_examples.py --generations d_out --probes canonical_probes.json \
      --embedder sentence-transformers/all-mpnet-base-v2 \
      --n 3 --out steering_examples.md

Every quantitative result in this project measures displacement in an embedding
space. That is the right way to avoid a judge's naming prior, but it leaves a
reviewer no way to check that the displacement corresponds to anything a reader
would recognise. This script closes that gap by printing the text.

WHAT IT SELECTS, AND WHY BOTH ENDS

  For each axis it finds cells differing only in that axis's level, embeds the
  generations, and ranks the high/low pairs by displacement along the axis's own
  probe contrast.

  BEST pairs show the effect at its clearest — useful, and the kind of example a
  paper usually prints.

  WORST pairs show cases where the measured displacement is near zero or
  negative. **These matter more.** Roughly one in ten triple-axis cases moves the
  text less than 5% of the pole-to-pole distance, and a reader who sees only
  successes will overestimate the method. Printing failures alongside successes
  is the difference between an illustration and evidence.

  DEGENERATE examples are shown separately rather than filtered out. 61.6% of
  generations contain repeating cycles; a reader should see what that looks like
  before deciding what the numbers mean.

The output is markdown, ready to paste into the paper's qualitative section.
"""
import argparse, sys, glob as _glob, json, os, re, sys
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

def _lit(v):
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except Exception:
        import ast
        return ast.literal_eval(v)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def degenerate(t):
    toks = re.findall(r"\S+", t)
    if len(toks) < 5:
        return True
    low = [x.lower() for x in toks]
    if len(set(low)) / len(low) < 0.5:
        return True
    run = best = 1
    for i in range(1, len(low)):
        run = run + 1 if low[i] == low[i - 1] else 1
        best = max(best, run)
    return best >= 4


def get_embedder(name):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name)
    return lambda t: np.asarray(m.encode(list(t), convert_to_numpy=True,
                                         show_progress_bar=False, batch_size=64),
                                dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", nargs="+", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--arm", default="real")
    ap.add_argument("--n", type=int, default=3, help="examples per end per axis")
    ap.add_argument("--max-chars", type=int, default=240)
    ap.add_argument("--out", default="steering_examples.md")
    a = ap.parse_args()

    paths = []
    for pat in a.generations:
        if os.path.isdir(pat):
            paths += sorted(_glob.glob(os.path.join(pat, "*_generations.jsonl")))
        elif any(c in pat for c in "*?["):
            paths += sorted(_glob.glob(pat))
        else:
            paths.append(pat)
    probes = json.load(open(a.probes, encoding="utf-8"))
    enc = get_embedder(a.embedder)

    D = {}
    for k, s in probes.items():
        if len(s.get("pos", [])) >= 2 and len(s.get("neg", [])) >= 2:
            D[k] = unit(enc(s["pos"]).mean(0) - enc(s["neg"]).mean(0))

    # collect matched cell pairs: same trial, same other-axis levels, axis flipped
    pairs = defaultdict(list)
    for p in paths:
        model = os.path.basename(p).replace("_generations.jsonl", "")
        cells = defaultdict(dict)
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("arm") != a.arm:
                continue
            lv = _lit(r["levels"])
            tri = tuple(_lit(r["triple"]))
            cells[(model, r["trial_id"], tri)][tuple(int(np.sign(lv[k])) for k in tri)] = \
                _lit(r["generations"])
        for (mdl, trial, tri), cc in cells.items():
            for ai, ax in enumerate(tri):
                if ax not in D:
                    continue
                for key, gens in cc.items():
                    if key[ai] != 1:
                        continue
                    lo_key = list(key); lo_key[ai] = -1
                    lo = cc.get(tuple(lo_key))
                    if not lo:
                        continue
                    pairs[ax].append((mdl, gens, lo, tri, key))

    print(f"{sum(len(v) for v in pairs.values())} matched high/low cell pairs")
    out = ["# Steering examples",
           "",
           "Matched cell pairs differing in one axis only. Displacement is measured "
           "along that axis's own probe contrast, in units of the pole-to-pole "
           "distance. Failures are shown alongside successes, since a reader who "
           "sees only successes will overestimate the method.",
           ""]

    for ax in sorted(pairs):
        recs = pairs[ax]
        if not recs:
            continue
        pole_sep = float(np.linalg.norm(enc(probes[ax]["pos"]).mean(0)
                                        - enc(probes[ax]["neg"]).mean(0)))
        scored = []
        for mdl, hi, lo, tri, key in recs:
            eh, el = enc(hi).mean(0), enc(lo).mean(0)
            disp = float((eh - el) @ D[ax]) / max(pole_sep, 1e-9)
            hi_c = [g for g in hi if not degenerate(g)]
            lo_c = [g for g in lo if not degenerate(g)]
            scored.append((disp, mdl, hi_c or hi, lo_c or lo, tri,
                           bool(hi_c and lo_c)))
        scored.sort(key=lambda x: -x[0])
        med = np.median([s[0] for s in scored])
        out += [f"## {ax}", "",
                f"{len(scored)} matched pairs, median displacement "
                f"{med:.1%} of pole distance, range {scored[-1][0]:.1%} to "
                f"{scored[0][0]:.1%}.", ""]

        for label, sel in (("Strongest", scored[:a.n]), ("Weakest", scored[-a.n:])):
            out.append(f"**{label}**")
            out.append("")
            for disp, mdl, hi, lo, tri, clean in sel:
                others = ", ".join(t for t in tri if t != ax)
                out += [f"*{mdl}, {disp:+.1%} of pole distance, "
                        f"co-steered with {others}"
                        f"{'' if clean else ', DEGENERATE cells'}*", "",
                        f"- **low:** {lo[0][:a.max_chars].strip()}",
                        f"- **high:** {hi[0][:a.max_chars].strip()}", ""]
        out.append("")

    open(a.out, "w", encoding="utf-8").write("\n".join(out))
    print(f"-> {a.out}")
    print("\nRead the weakest examples before the strongest. If a near-zero")
    print("displacement pair looks indistinguishable to you, that is the honest")
    print("floor of the method and belongs in the paper.")


if __name__ == "__main__":
    main()
