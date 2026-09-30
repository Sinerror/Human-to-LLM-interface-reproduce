#!/usr/bin/env python3
"""
bootstrap_crossalign.py — how stable is the cross-alignment diagonal count?

  python bootstrap_crossalign.py --generations d2_* --probes canonical_probes.json \
      --embedder sentence-transformers/all-mpnet-base-v2 --n-boot 300 \
      --out crossalign_boot.json

THE QUESTION

  Section 5.2 reports that each axis aligns more with its own construct than
  with any other. Counting rows where that holds gives 7 of 7 on a 400-per-group
  subsample and 5 of 7 using every generation, with `heat` and `arousal` losing
  to `valence` by 0.032 and 0.024.

  A count that moves with the sample is a fragile statistic. This resamples
  generations within each axis-level group, recomputes the whole matrix, and
  reports how often each row is diagonal-dominant. The output is a per-row
  probability rather than a single count, which is what a margin of 0.03
  actually supports.

  The structural claim does not depend on this. Mean diagonal against mean
  off-diagonal, and the negative entries that exclude a common affective shift,
  are stable regardless of which rows win. Run this to state the count
  honestly, not to rescue it.

COST
  Embedding happens once; the bootstrap resamples indices into the embedding
  matrix, so 300 iterations cost little beyond the initial encode. Expect
  roughly the runtime of one embed_align pass.
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

SEED = 0
AXES = ["valence", "heat", "arousal", "intensity", "antagonism_peace",
        "body_mind", "social_inner_outer"]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def dedup(text, max_period=4):
    toks = re.findall(r"\S+", str(text))
    n = len(toks)
    if n < 2:
        return text
    low = [t.lower() for t in toks]
    out, i = [], 0
    while i < n:
        bp, br = 1, 1
        for per in range(1, max_period + 1):
            if i + per > n:
                break
            reps = 1
            while (i + per * (reps + 1) <= n
                   and low[i + per * reps: i + per * (reps + 1)] == low[i: i + per]):
                reps += 1
            if reps > br:
                bp, br = per, reps
        out.extend(toks[i: i + bp])
        i += bp * br
    return " ".join(out)


def _lit(v):
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except Exception:
        import ast
        return ast.literal_eval(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", nargs="+", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--embedder",
                    default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--arm", default="raw")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--out", default="crossalign_boot.json")
    a = ap.parse_args()

    paths = []
    for pat in a.generations:
        if os.path.isdir(pat):
            paths += sorted(glob.glob(os.path.join(pat, "*_generations.jsonl")))
        elif any(c in pat for c in "*?["):
            for p in sorted(glob.glob(pat)):
                paths += (sorted(glob.glob(os.path.join(p, "*_generations.jsonl")))
                          if os.path.isdir(p) else [p])
        else:
            paths.append(pat)
    paths = [p for p in paths if a.arm in os.path.basename(p)]
    print(f"{len(paths)} file(s), arm={a.arm}")

    hi_txt = defaultdict(list)
    lo_txt = defaultdict(list)
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("arm") != a.arm:
                continue
            tri = _lit(r["triple"])
            lv = _lit(r["levels"])
            g = [dedup(x) for x in _lit(r["generations"])]
            for k in tri:
                if k not in AXES:
                    continue
                (hi_txt if lv[k] > 0 else lo_txt)[k].extend(g)

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(a.embedder)

    def enc(t):
        return np.asarray(st.encode(list(t), convert_to_numpy=True,
                                    show_progress_bar=False, batch_size=64),
                          dtype=np.float64)

    probes = json.load(open(a.probes, encoding="utf-8"))
    D = {k: unit(enc(probes[k]["pos"]).mean(0) - enc(probes[k]["neg"]).mean(0))
         for k in AXES if k in probes}
    ax = [k for k in AXES if k in D and hi_txt.get(k) and lo_txt.get(k)]
    print(f"  {len(ax)} axes; embedding "
          f"{sum(len(hi_txt[k]) + len(lo_txt[k]) for k in ax)} generations")

    H = {k: enc(hi_txt[k]) for k in ax}
    L = {k: enc(lo_txt[k]) for k in ax}

    def matrix(idx_h=None, idx_l=None):
        M = np.zeros((len(ax), len(ax)))
        for i, k in enumerate(ax):
            h = H[k][idx_h[k]] if idx_h else H[k]
            l = L[k][idx_l[k]] if idx_l else L[k]
            d = unit(h.mean(0) - l.mean(0))
            for j, o in enumerate(ax):
                M[i, j] = d @ D[o]
        return M

    M0 = matrix()
    dm0 = sum(int(np.argmax(M0[i]) == i) for i in range(len(ax)))
    print(f"\n  point estimate: diagonal is max in {dm0}/{len(ax)} rows")
    print(f"  mean diagonal {np.diag(M0).mean():+.3f}  "
          f"mean off-diagonal "
          f"{(M0.sum() - np.trace(M0)) / (M0.size - len(ax)):+.3f}")

    rng = np.random.default_rng(SEED)
    wins = np.zeros(len(ax))
    counts = []
    for b in range(a.n_boot):
        ih = {k: rng.integers(0, len(H[k]), len(H[k])) for k in ax}
        il = {k: rng.integers(0, len(L[k]), len(L[k])) for k in ax}
        M = matrix(ih, il)
        w = np.array([int(np.argmax(M[i]) == i) for i in range(len(ax))])
        wins += w
        counts.append(int(w.sum()))
        if (b + 1) % 50 == 0:
            print(f"    {b + 1}/{a.n_boot}", flush=True)

    p = wins / a.n_boot
    counts = np.array(counts)
    print(f"\n  {'axis':<22}{'own':>8}{'best other':>12}{'margin':>9}"
          f"{'P(dominant)':>13}")
    for i, k in enumerate(ax):
        others = [(M0[i, j], ax[j]) for j in range(len(ax)) if j != i]
        bv, bn = max(others)
        print(f"  {k:<22}{M0[i, i]:>+8.3f}{bv:>+12.3f}"
              f"{M0[i, i] - bv:>+9.3f}{p[i]:>13.0%}   {bn if bv > M0[i,i] else ''}")

    print(f"\n  diagonal-dominant rows across {a.n_boot} resamples:")
    for c in range(len(ax) + 1):
        n = int((counts == c).sum())
        if n:
            print(f"    {c}/{len(ax)}  {n / a.n_boot:>6.1%}  "
                  + "#" * int(40 * n / a.n_boot))
    print(f"\n  median {int(np.median(counts))}/{len(ax)}, "
          f"95% interval {int(np.percentile(counts, 2.5))}-"
          f"{int(np.percentile(counts, 97.5))}")

    json.dump(dict(axes=ax, matrix=M0.tolist(), point_count=dm0,
                   p_dominant={k: float(p[i]) for i, k in enumerate(ax)},
                   counts=counts.tolist(), n_boot=a.n_boot,
                   mean_diagonal=float(np.diag(M0).mean()),
                   mean_offdiagonal=float((M0.sum() - np.trace(M0))
                                          / (M0.size - len(ax)))),
              open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")
    print("\n  Report the per-row probability, not the count. A row at 50% is a")
    print("  coin flip and should be described as one; a row at 99% is stable.")


if __name__ == "__main__":
    main()
