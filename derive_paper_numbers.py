#!/usr/bin/env python3
"""
derive_paper_numbers.py — every figure in sections 3.3 and 4 that is derived
from the basis rather than read off verify's summary table.

  python derive_paper_numbers.py --vecs lat_outputs\\google_gemma-3-1b-it_vecs.npz \
      --layer 17 --labels frozen_emotionality_labels.csv \
      --probes canonical_probes.json --out paper_numbers.json

Seconds, no GPU. Run it on the CORRECTED label file: the point is that every
geometry figure in the draft came from the superseded one.
"""
import argparse, csv, json, sys
from collections import defaultdict
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

SEVEN = ["valence", "heat", "arousal", "intensity", "antagonism_peace",
         "body_mind", "social_inner_outer"]
ALT = ["valence", "drive", "arousal", "heat", "security_alarm", "desire",
       "antagonism_peace"]
SUPERSEDED_MARKER = "self-contempt::fallback"


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", default="paper_numbers.json")
    a = ap.parse_args()

    lab = {}
    for r in csv.DictReader(open(a.labels, newline="", encoding="utf-8")):
        lab[r["key"]] = (r.get("final", "").strip().upper() or r["heuristic"])
    if SUPERSEDED_MARKER in lab:
        sys.exit(f"{a.labels} is the SUPERSEDED label file; every number would "
                 f"be the old one. Use the corrected file.")

    z = np.load(a.vecs, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    words = np.array([str(w).lower() for w in z["words"]])
    X = z["vecs"].astype(np.float64)
    if X.ndim == 3:
        ls = [int(v) for v in np.asarray(z["layers"])]
        X = X[:, ls.index(a.layer), :] if X.shape[1] > 1 else X[:, 0, :]
    m = np.array([lab.get(k) == "EMO" for k in keys])
    Xc = X[m] - X[m].mean(0)
    acc = defaultdict(list)
    for w, v in zip(words[m], Xc):
        acc[w].append(v)
    wv = {w: np.mean(vs, 0) for w, vs in acc.items()}
    probes = json.load(open(a.probes, encoding="utf-8"))

    def basis(names):
        P, kept, poles = [], [], {}
        for nm in names:
            s = probes[nm]
            pos = [w.lower() for w in s["pos"] if w.lower() in wv]
            neg = [w.lower() for w in s["neg"] if w.lower() in wv]
            P.append(unit(np.mean([wv[w] for w in pos], 0)
                          - np.mean([wv[w] for w in neg], 0)))
            kept.append(nm)
            poles[nm] = (pos, neg)
        return np.array(P), kept, poles

    P, names, poles = basis(SEVEN)
    G = P @ P.T
    k = len(names)
    iu = np.triu_indices(k, 1)
    Gi = np.linalg.inv(G)
    iso = np.sqrt(np.diag(Gi))
    out = {}

    print(f"LABELS {a.labels}   EMO senses {int(m.sum())}\n")
    print("SECTION 4.1  Gram matrix, |cos|")
    print("  " + " " * 20 + "".join(f"{n[:6]:>8}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:<20}" + "".join(f"{abs(G[i, j]):>8.2f}" for j in range(k)))
    pairs = sorted(((abs(G[i, j]), names[i], names[j]) for i, j in zip(*iu)),
                   reverse=True)
    print(f"\n  mean |cos| {np.abs(G[iu]).mean():.3f}   max {pairs[0][0]:.3f}"
          f"   condition {np.linalg.cond(G):.2f}   det {np.linalg.det(G):.3f}")
    print(f"  most entangled:  " + ", ".join(f"{a_}-{b_} {v:.3f}"
                                            for v, a_, b_ in pairs[:3]))
    print(f"  least entangled: " + ", ".join(f"{a_}-{b_} {v:.3f}"
                                            for v, a_, b_ in pairs[-3:]))
    out["gram"] = dict(matrix=np.abs(G).round(3).tolist(), names=names,
                       mean=float(np.abs(G[iu]).mean()),
                       max=float(pairs[0][0]),
                       max_pair=[pairs[0][1], pairs[0][2]],
                       condition=float(np.linalg.cond(G)),
                       determinant=float(np.linalg.det(G)),
                       top3=[[a_, b_, float(v)] for v, a_, b_ in pairs[:3]],
                       bottom3=[[a_, b_, float(v)] for v, a_, b_ in pairs[-3:]])

    print("\nSECTION 4.2  leakage per unit requested, and iso-cost")
    leak = {}
    for i, n in enumerate(names):
        leak[n] = float(np.sqrt(sum(G[j, i] ** 2 for j in range(k) if j != i)))
    for n in sorted(leak, key=lambda x: -leak[x]):
        print(f"  {n:<22}leak {leak[n]:.3f}   iso-cost "
              f"{iso[names.index(n)]:.3f}")
    print(f"\n  worst leak {max(leak.values()):.3f} "
          f"({max(leak.values()):.0%} misdirected per unit requested)")
    print(f"  iso-cost {iso.min():.3f} to {iso.max():.3f}, mean {iso.mean():.3f}"
          f"  ->  {(iso.min()-1):.0%} to {(iso.max()-1):.0%} extra displacement")
    out["leakage"] = leak
    out["iso_cost"] = dict(zip(names, iso.round(3).tolist()))

    print("\nSECTION 4.3  Gram-Schmidt: cos(original, orthogonalised)")
    Q, _ = np.linalg.qr(P.T)
    gs = {n: float(abs(P[i] @ Q[:, i])) for i, n in enumerate(names)}
    for n in names:
        print(f"  {n:<22}{gs[n]:.3f}")
    worst = min(gs, key=gs.get)
    print(f"\n  worst: {worst} retains {gs[worst]:.3f} of its direction")
    out["gram_schmidt"] = gs

    print("\nSECTION 3.4 / 4.1  the alternative seven "
          "(drive, desire, security_alarm substituted)")
    try:
        Pa, na, _ = basis(ALT)
        Ga = Pa @ Pa.T
        iua = np.triu_indices(len(na), 1)
        pa = sorted(((abs(Ga[i, j]), na[i], na[j]) for i, j in zip(*iua)),
                    reverse=True)
        print(f"  mean |cos| {np.abs(Ga[iua]).mean():.3f}   condition "
              f"{np.linalg.cond(Ga):.2f}   worst pair {pa[0][1]}-{pa[0][2]} "
              f"{pa[0][0]:.3f}")
        out["alternative_seven"] = dict(mean=float(np.abs(Ga[iua]).mean()),
                                        condition=float(np.linalg.cond(Ga)),
                                        worst=[pa[0][1], pa[0][2],
                                               float(pa[0][0])])
    except KeyError as e:
        print(f"  a probe is missing: {e}")

    print("\nSECTION 3.3  leave-one-word-out, per axis")
    rng = np.random.default_rng(0)

    def lowo(pos, neg):
        c = t = 0
        for wp in pos:
            for wn in neg:
                d = unit(np.mean([wv[w] for w in pos if w != wp], 0)
                         - np.mean([wv[w] for w in neg if w != wn], 0))
                c += int(wv[wp] @ d > wv[wn] @ d)
                t += 1
        return c / t

    lw = {}
    for n in names:
        pos, neg = poles[n]
        acc_ = lowo(pos, neg)
        pool = pos + neg
        nl = [lowo(*(lambda s_: (s_[:len(pos)], s_[len(pos):]))(
            list(rng.permutation(pool)))) for _ in range(60)]
        lw[n] = dict(accuracy=acc_, null_p95=float(np.percentile(nl, 95)))
        print(f"  {n:<22}{acc_:.3f}   null {lw[n]['null_p95']:.3f}")
    lo = min(lw, key=lambda x: lw[x]["accuracy"])
    hi = max(lw, key=lambda x: lw[x]["accuracy"])
    print(f"\n  range {lw[lo]['accuracy']:.3f} ({lo}) to "
          f"{lw[hi]['accuracy']:.3f} ({hi})")
    out["lowo"] = lw

    # ---- shared words: how much of the Gram is the word lists themselves?
    print("\nSECTION 4.1  words shared between lists, and what they contribute")
    cnt = defaultdict(int)
    for nm in SEVEN:
        for side in ("pos", "neg"):
            for w in probes[nm][side]:
                cnt[w.lower()] += 1
    shared = sorted(w for w, c in cnt.items() if c > 1)
    def gram_of(lists):
        Q = np.array([unit(np.mean([wv[w] for w in ps], 0) - np.mean([wv[w] for w in ng], 0))
                      for ps, ng in lists])
        return Q @ Q.T
    full = [poles[nm] for nm in names]
    nos = [([w for w in ps if w not in shared], [w for w in ng if w not in shared]) for ps, ng in full]
    G_ns = gram_of(nos)
    rng2 = np.random.default_rng(0)
    null = []
    for _ in range(500):
        lists = []
        for (ps, ng), (ps2, ng2) in zip(full, nos):
            pair = []
            for pole, kept in ((ps, ps2), (ng, ng2)):
                k = len(pole) - len(kept)
                cand = [w for w in pole if w not in shared]
                drop = set(rng2.choice(cand, min(k, max(len(cand) - 2, 0)), replace=False)) if k else set()
                pair.append([w for w in pole if w not in drop])
            lists.append(tuple(pair))
        null.append(float(np.abs(gram_of(lists)[iu]).mean()))
    null = np.array(null)
    ch = sorted(((abs(G[i, j] - G_ns[i, j]), names[i], names[j], G[i, j], G_ns[i, j])
                 for i, j in zip(*iu)), reverse=True)
    print(f"  {len(shared)} words appear in more than one list: {', '.join(shared)}")
    print(f"  mean |cos| {np.abs(G[iu]).mean():.3f} -> {np.abs(G_ns[iu]).mean():.3f} without them "
          f"(condition {np.linalg.cond(G):.2f} -> {np.linalg.cond(G_ns):.2f})")
    print(f"  null, same count of random words removed: {null.mean():.3f} "
          f"(95% {np.percentile(null, 2.5):.3f}-{np.percentile(null, 97.5):.3f}); "
          f"draws at or below: {np.mean(null <= np.abs(G_ns[iu]).mean()):.3f}")
    for d, x, y, g0, g1 in ch[:5]:
        print(f"    {x}-{y}: {g0:+.3f} -> {g1:+.3f}")
    out["shared_words"] = dict(
        words=shared, mean_full=float(np.abs(G[iu]).mean()),
        mean_without=float(np.abs(G_ns[iu]).mean()),
        condition_without=float(np.linalg.cond(G_ns)),
        null_mean=float(null.mean()), null_p025=float(np.percentile(null, 2.5)),
        null_p975=float(np.percentile(null, 97.5)),
        null_at_or_below=float(np.mean(null <= np.abs(G_ns[iu]).mean())),
        largest_changes=[[x, y, float(g0), float(g1)] for d, x, y, g0, g1 in ch[:5]])

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}   send this file back")


if __name__ == "__main__":
    main()
