#!/usr/bin/env python3
"""
linear_baseline.py — what would a perfectly LINEAR model show in our experiment?

  python linear_baseline.py --vecs-dir lat_outputs --labels frozen_emotionality_labels.csv \
      --probes canonical_probes.json --t2-raw t2_raw.json --t2-dual t2_dual.json

A few minutes on CPU. No generation.

WHY
  Section 7.3 attributes the dual basis's failure to a non-linear response. That
  is only evidence if the LINEAR prediction for our exact pipeline is known, and
  two linear effects stand between the textbook prediction and ours:

  1. Our harness rescales every cell's combined vector to unit length
     (steer_gen --equalise-cell-norm, always on). The applied displacement is
     then sum(l_i d_i) / ||sum(l_i d_i)||, whose scale depends on the products
     l_a * l_b through G. That is non-linear in the request even for a linear
     model, and it turns into real off-diagonal main effects in the 2^3 design.

  2. The readout projects generated text onto encoder directions built from the
     same word lists, and those directions overlap in the encoder's space. Text
     that moved purely along construct a still registers on construct j by the
     encoder's own cosine. No correction in the model's activations can remove
     that.

  The script computes, per model and over all 35 triples and 8 cells, the
  transfer matrices a linear model would produce under (i) the textbook,
  (ii) plus our harness, (iii) plus our readout, and prints the same statistics
  the paper reports beside the observed ones. Whatever gap remains after (iii)
  is evidence against linearity; whatever (ii) and (iii) already explain is not.

ASSUMPTIONS, STATED
  A linear model here means: the text readout of construct j equals the encoder
  overlap A_jk times the activation coordinate k, with equal gain per construct.
  Unequal gains would change magnitudes but not whether off-diagonals can vanish.
"""
import argparse, csv, itertools, json, os, sys
from collections import defaultdict
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

AXES = ["valence", "heat", "arousal", "intensity", "antagonism_peace",
        "body_mind", "social_inner_outer"]
LAYERS = {"google_gemma-3-1b-it": 17, "google_gemma-3-1b-pt": 25,
          "Qwen_Qwen2.5-1.5B": 25, "microsoft_phi-2": 32,
          "state-spaces_mamba-2.8b-hf": 52, "EleutherAI_pythia-6.9b": 28,
          "mistralai_Mistral-7B-v0.3": 32, "meta-llama_Meta-Llama-3-8B": 18,
          "unsloth_gemma-2-9b-bnb-4bit": 27}
L8 = np.array(list(itertools.product([-1, 1], repeat=3)), float)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def activation_gram(path, layer, labels, probes):
    z = np.load(path, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    words = np.array([str(w).lower() for w in z["words"]])
    ls = [int(x) for x in np.asarray(z["layers"])]
    m = np.array([labels.get(k) == "EMO" for k in keys])
    X = z["vecs"][m, ls.index(layer), :].astype(np.float64)
    X -= X.mean(0)
    acc = defaultdict(list)
    for w, v in zip(words[m], X):
        acc[w].append(v)
    wv = {w: np.mean(v, 0) for w, v in acc.items()}
    P = []
    for k in AXES:
        pos = [w.lower() for w in probes[k]["pos"] if w.lower() in wv]
        neg = [w.lower() for w in probes[k]["neg"] if w.lower() in wv]
        P.append(unit(np.mean([wv[w] for w in pos], 0) - np.mean([wv[w] for w in neg], 0)))
    P = np.array(P)
    return P @ P.T


def encoder_gram(probes, embedder):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(embedder)
    D = []
    for k in AXES:
        e = lambda ws: st.encode(ws, convert_to_numpy=True).mean(0)
        D.append(unit(e(probes[k]["pos"]) - e(probes[k]["neg"])))
    D = np.array(D)
    return D @ D.T


def main_effects(Y):
    return (Y.T @ L8) / 8.0


def predicted(G, A, harness):
    Gi = np.linalg.inv(G)
    R, D = [], []
    for idx in itertools.combinations(range(len(AXES)), 3):
        ix = np.ix_(idx, idx)
        G3, Gi3, A3 = G[ix], Gi[ix], A[ix]
        yr = np.array([A3 @ (G3 @ l) / (np.sqrt(l @ G3 @ l) if harness else 1) for l in L8])
        yd = np.array([A3 @ l / (np.sqrt(l @ Gi3 @ l) if harness else 1) for l in L8])
        R.append(main_effects(yr)); D.append(main_effects(yd))
    return R, D


def stats(R, D):
    iu = np.triu_indices(3, 1); il = np.tril_indices(3, -1)
    def parts(Ms):
        return (np.mean([np.abs(np.diag(X)).mean() for X in Ms]),
                np.mean([np.abs(((X + X.T) / 2)[iu]).mean() for X in Ms]),
                np.mean([np.abs(((X - X.T) / 2)[iu]).mean() for X in Ms]))
    def ratio(Ms):
        out = []
        for X in Ms:
            Xn = X / (np.abs(X).max(1, keepdims=True) + 1e-12)
            off = np.concatenate([np.abs(Xn[iu]), np.abs(Xn[il])]).mean()
            out.append(np.abs(np.diag(Xn)).mean() / max(off, 1e-9))
        return np.mean(out)
    a, b = parts(R), parts(D)
    ch = lambda i: (b[i] / a[i] - 1) if a[i] > 1e-9 else float("nan")
    rr, rd = ratio(R), ratio(D)
    return dict(diag=ch(0), sym=ch(1), anti=ch(2),
                sep=(rd / rr) if rr > 0 and np.isfinite(rd) and rd < 1e3 else float("inf"),
                raw_anti_share=a[2] / (a[1] + a[2]) if a[1] + a[2] > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--t2-raw", required=True)
    ap.add_argument("--t2-dual", required=True)
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--enc-identity", action="store_true",
                    help="skip the encoder; treat the readout as orthogonal (testing only)")
    ap.add_argument("--out", default="linear_baseline.json")
    a = ap.parse_args()

    labels = {}
    for r in csv.DictReader(open(a.labels, newline="", encoding="utf-8")):
        labels[r["key"]] = (r.get("final", "").strip().upper() or r["heuristic"])
    probes = json.load(open(a.probes, encoding="utf-8"))
    A = np.eye(len(AXES)) if a.enc_identity else encoder_gram(probes, a.embedder)
    iu = np.triu_indices(len(AXES), 1)
    print(f"encoder-space overlap of the seven probe directions: mean |cos| "
          f"{np.abs(A[iu]).mean():.3f}, max {np.abs(A[iu]).max():.3f}\n")

    tr = json.load(open(a.t2_raw, encoding="utf-8"))
    td = json.load(open(a.t2_dual, encoding="utf-8"))
    obs = {}
    for k in tr:
        dk = k.replace("_raw", "_dual")
        if dk in td:
            obs[k] = stats([np.array(p["M"]) for p in tr[k]["per_triple"]],
                           [np.array(p["M"]) for p in td[dk]["per_triple"]])

    rows, out = [], {"encoder_gram": A.tolist(), "models": {}}
    hdr = f"{'':<32}{'diag':>8}{'sym off':>9}{'anti':>8}{'separab.':>10}{'raw anti share':>16}"
    for slug, layer in LAYERS.items():
        vp = os.path.join(a.vecs_dir, f"{slug}_vecs.npz")
        ok = next((k for k in obs if k.replace("_raw", "") in slug), None)
        if not os.path.exists(vp) or ok is None:
            print(f"  {slug}: vectors or observed matrices missing, skipped")
            continue
        G = activation_gram(vp, layer, labels, probes)
        res = {"textbook": stats(*predicted(G, np.eye(len(AXES)), False)),
               "+harness": stats(*predicted(G, np.eye(len(AXES)), True)),
               "+harness +readout": stats(*predicted(G, A, True)),
               "OBSERVED": obs[ok]}
        out["models"][slug] = res
        print(slug); print("  " + hdr)
        for lab, r in res.items():
            f = lambda x: f"{x:+.0%}" if np.isfinite(x) else "  --"
            print(f"  {lab:<30}{f(r['diag']):>8}{f(r['sym']):>9}{f(r['anti']):>8}"
                  f"{('x%.2f' % r['sep']) if np.isfinite(r['sep']) else '   inf':>10}"
                  f"{r['raw_anti_share']:>15.0%}")
        print()
    if out["models"]:
        print("MEAN OVER MODELS")
        for lab in ("textbook", "+harness", "+harness +readout", "OBSERVED"):
            v = [m[lab] for m in out["models"].values()]
            g = lambda key: np.mean([x[key] for x in v if np.isfinite(x[key])])
            print(f"  {lab:<30}diag {g('diag'):+.0%}   sym off {g('sym'):+.0%}   "
                  f"separability x{g('sep'):.2f}   raw antisym share {g('raw_anti_share'):.0%}")
        print("\n  The gap between '+harness +readout' and OBSERVED is what a linear")
        print("  model cannot produce. Everything above that line is not evidence of")
        print("  non-linearity: it is our pipeline.")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
