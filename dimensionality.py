#!/usr/bin/env python3
"""
dimensionality.py — effective dimensionality of every direction set in Appendix C,
computed one way, on the corrected labels.

  python dimensionality.py --vecs-dir lat_outputs ^
      --labels frozen_emotionality_labels.csv --probes canonical_probes.json ^
      --cats27 canonical_probes_27d.json --ekman ekman.json --out dimensionality.json

Seconds, CPU only, no generation.

WHAT IS COMPUTED, AND WHY IT IS ONE QUANTITY
  For a set of direction vectors, PR = (sum lambda)^2 / sum(lambda^2), where lambda
  are the eigenvalues of the Gram matrix G = M M^T of the UNIT-NORMALISED
  directions. PR is k for k orthogonal directions and 1 for k identical ones.

    27 categories   one direction per Cowen-Keltner category: the mean of that
                    category's word vectors, normalised
    6 categories    the same for Ekman's six
    seven axes      the paper's axes: mean(pos words) - mean(neg words), normalised
    corpus          PR of the squared singular values of the centred sense-by-
                    dimension matrix, with the single highest-variance coordinate
                    removed. This is a different object (data, not directions),
                    printed for the appendix's third row.

  Words are centred on the mean of all emotional senses, sense vectors averaged
  per word, exactly as in every other script here. On the superseded label file
  this reproduces the figures the paper printed before the correction: 12.16 and
  4.94 for Gemma-3-1B-it layer 17, 12.31 for the pt sibling, corpus 49.6 / 39.7.

THE LABEL FILE MATTERS
  Those printed figures came from the superseded labels (the ones containing
  `self-contempt::fallback`). This script refuses that file unless told
  --allow-superseded, which exists only to reproduce old numbers for comparison.
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

MODELS = {"google_gemma-3-1b-it": 17, "google_gemma-3-1b-pt": 25}
AXES = ["valence", "heat", "arousal", "intensity", "antagonism_peace",
        "body_mind", "social_inner_outer"]
MARKER = "self-contempt::fallback"
PAPER = {"google_gemma-3-1b-it": dict(c27=12.2, ekman=4.9, corpus="40-50"),
         "google_gemma-3-1b-pt": dict(c27=12.3, ekman=None, corpus="40-50")}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def pr(M):
    G = M @ M.T
    ev = np.maximum(np.linalg.eigvalsh(G)[::-1], 0)
    return float(ev.sum() ** 2 / (ev ** 2).sum()), G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--cats27", required=True)
    ap.add_argument("--ekman", required=True)
    ap.add_argument("--out", default="dimensionality.json")
    ap.add_argument("--allow-superseded", action="store_true")
    a = ap.parse_args()

    txt = open(a.labels, encoding="utf-8").read()
    old = MARKER in txt
    if old and not a.allow_superseded:
        sys.exit(f"{a.labels} is the SUPERSEDED label file (contains {MARKER}). "
                 f"Every number would be the old one. Use frozen_emotionality_labels.csv.")
    lab = {}
    for r in csv.DictReader(open(a.labels, newline="", encoding="utf-8")):
        lab[r["key"]] = (r.get("final", "").strip().upper() or r["heuristic"])
    probes = json.load(open(a.probes, encoding="utf-8"))
    c27 = json.load(open(a.cats27, encoding="utf-8"))
    ek = json.load(open(a.ekman, encoding="utf-8"))
    if old:
        print("!! SUPERSEDED LABELS — these numbers reproduce the old figures only.\n")

    out = {"labels_superseded": old, "models": {}}
    for model, layer in MODELS.items():
        vp = os.path.join(a.vecs_dir, f"{model}_vecs.npz")
        if not os.path.exists(vp):
            print(f"  {model}: vectors absent, skipped"); continue
        z = np.load(vp, allow_pickle=False)
        keys = [str(k) for k in z["keys"]]
        words = np.array([str(w).lower() for w in z["words"]])
        ls = [int(x) for x in np.asarray(z["layers"])]
        m = np.array([lab.get(k) == "EMO" for k in keys])
        X = z["vecs"][m, ls.index(layer), :].astype(np.float64)
        Xc = X - X.mean(0)
        acc = defaultdict(list)
        for w, v in zip(words[m], Xc):
            acc[w].append(v)
        wv = {w: np.mean(v, 0) for w, v in acc.items()}

        def category_dirs(spec):
            names, D = [], []
            for k, v in spec.items():
                ws = [x.lower().replace("_", " ") for x in v["pos"]
                      if x.lower().replace("_", " ") in wv]
                if len(ws) >= 2:
                    names.append(k); D.append(unit(np.mean([wv[x] for x in ws], 0)))
            return names, np.array(D)

        n27, M27 = category_dirs(c27)
        n6, M6 = category_dirs(ek)
        P = np.array([unit(np.mean([wv[w.lower()] for w in probes[k]["pos"] if w.lower() in wv], 0)
                           - np.mean([wv[w.lower()] for w in probes[k]["neg"] if w.lower() in wv], 0))
                      for k in AXES])
        pr27, G27 = pr(M27); pr6, G6 = pr(M6); pr7, G7 = pr(P)

        j = int(np.argmax(Xc.var(0)))
        s2 = np.linalg.svd(np.delete(Xc, j, axis=1), compute_uv=False) ** 2
        prc = float(s2.sum() ** 2 / (s2 ** 2).sum())

        def pairs(G, names, k):
            iu = np.triu_indices(len(names), 1)
            return sorted(((float(G[i, j_]), names[i], names[j_]) for i, j_ in zip(*iu)), reverse=True)[:k]
        f27, f6 = pairs(G27, n27, 6), pairs(G6, n6, 2)
        iu = np.triu_indices(len(n27), 1)
        min27 = float(G27[iu].min())
        js = float(G6[n6.index("joy"), n6.index("sadness")]) if {"joy", "sadness"} <= set(n6) else None

        rec = dict(layer=layer, n_senses=int(m.sum()), n27=len(n27), n6=len(n6),
                   pr_27=pr27, pr_ekman=pr6, pr_seven_axes=pr7, pr_corpus_rogue_removed=prc,
                   seven_axes_condition=float(np.linalg.cond(G7)),
                   fusions_27=f27, fusions_ekman=f6, min_offdiag_27=min27, joy_sadness=js)
        out["models"][model] = rec
        p = PAPER.get(model, {})
        print(f"{model}  (layer {layer}, {int(m.sum())} emotional senses)")
        print(f"   27 categories   PR {pr27:6.2f} of {len(n27):<3}  paper prints {p.get('c27')}")
        print(f"   6  categories   PR {pr6:6.2f} of {len(n6):<3}  paper prints {p.get('ekman')}")
        print(f"   seven axes      PR {pr7:6.2f} of 7    (condition {rec['seven_axes_condition']:.2f})")
        print(f"   corpus          PR {prc:6.1f}  (rogue coordinate removed)  paper prints {p.get('corpus')}")
        print(f"   strongest fusions, 27: " + "; ".join(f"{x} {y} {c:.2f}" for c, x, y in f27))
        print(f"   strongest fusions, Ekman: " + "; ".join(f"{x} {y} {c:.2f}" for c, x, y in f6))
        print(f"   most negative alignment among the 27: {min27:+.2f}    joy-sadness: {js:+.2f}\n")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
