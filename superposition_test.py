#!/usr/bin/env python3
"""
superposition_test.py — is emotion steering superposable?

  python superposition_test.py \
      --triple-gens d2_g31bit d2_qwen ...  --single-gens d3_g31bit d3_qwen ... \
      --vecs-dir lat_outputs --labels frozen_emotionality_labels.csv \
      --probes canonical_probes.json --out superposition.json

THE ARGUMENT
  Under a linear response, the effect of a sum of displacements is the sum of
  their effects. Our three-dial cells apply c||h|| * sum(l_i d_i)/||sum(l_i d_i)||;
  the single-dial cells of `steer_gen --design single` apply c||h|| * d_i/||d_i||
  on the same prompts. So under linearity every three-dial cell is predicted
  exactly by the single-dial responses:

      y(l) - y0  =  sum_i  l_i * (||d_i|| / ||sum_j l_j d_j||) * R_i

  with R_i the single-dial response of axis i. The per-cell rescaling is inside
  the formula, so it cannot masquerade as non-linearity; the readout is the same
  on both sides, so its obliquity cancels. What the prediction misses, beyond
  noise, is the response failing to superpose.

  Two questions are answered at once:
    1. Superposition. Do single-dial responses predict the three-dial transfer
       matrices, as well as one repetition of the experiment predicts the other?
    2. One at a time versus together. Does the dual basis improve separability
       when one dial is turned (the setting in which Pan et al. report gains),
       and does it when three are?

NOISE CEILING
  Each triple is run twice on different prompts. How well repetition 1 predicts
  repetition 2 is the best any prediction can do; superposition is judged against
  that, not against perfect agreement.

About the runtime of one embed_align pass; a GPU is recommended.
"""
import argparse, csv, glob, hashlib, json, os, re, sys
from collections import defaultdict
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

AXES = ["valence", "heat", "arousal", "intensity", "antagonism_peace",
        "body_mind", "social_inner_outer"]
LAYERS = {"google/gemma-3-1b-it": 17, "google/gemma-3-1b-pt": 25,
          "Qwen/Qwen2.5-1.5B": 25, "microsoft/phi-2": 32,
          "state-spaces/mamba-2.8b-hf": 52, "EleutherAI/pythia-6.9b": 28,
          "mistralai/Mistral-7B-v0.3": 32, "meta-llama/Meta-Llama-3-8B": 18,
          "unsloth/gemma-2-9b-bnb-4bit": 27}
L8 = np.array([[(1 if (c >> b) & 1 else -1) for b in range(3)] for c in range(8)], float)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def lit(v):
    return json.loads(v) if isinstance(v, str) else v


def dedup(text, max_period=4):
    toks = re.findall(r"\S+", str(text)); n = len(toks)
    if n < 2:
        return text
    low = [t.lower() for t in toks]; out, i = [], 0
    while i < n:
        bp, br = 1, 1
        for per in range(1, max_period + 1):
            if i + per > n:
                break
            reps = 1
            while i + per * (reps + 1) <= n and low[i + per * reps:i + per * (reps + 1)] == low[i:i + per]:
                reps += 1
            if reps > br:
                bp, br = per, reps
        out.extend(toks[i:i + bp]); i += bp * br
    return " ".join(out)


def load(dirs, arm):
    """model -> trial_id -> (triple, {levels-tuple: [texts]})"""
    out = {}
    for d in dirs:
        mf = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
        model = mf["model"]
        for fp in glob.glob(os.path.join(d, f"*_{arm}_generations.jsonl")):
            for line in open(fp, encoding="utf-8"):
                r = json.loads(line)
                if r.get("arm", arm) != arm:
                    continue
                tri = tuple(lit(r["triple"])); lv = lit(r["levels"])
                key = tuple(int(lv[k]) for k in tri)
                t = out.setdefault(model, {}).setdefault(r["trial_id"], (tri, {}))
                t[1].setdefault(key, []).extend(dedup(x) for x in lit(r["generations"]))
    return out


def activation_gram(vp, layer, labels, probes):
    z = np.load(vp, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    words = np.array([str(w).lower() for w in z["words"]])
    ls = [int(x) for x in np.asarray(z["layers"])]
    m = np.array([labels.get(k) == "EMO" for k in keys])
    X = z["vecs"][m, ls.index(layer), :].astype(np.float64); X -= X.mean(0)
    acc = defaultdict(list)
    for w, v in zip(words[m], X):
        acc[w].append(v)
    wv = {w: np.mean(v, 0) for w, v in acc.items()}
    P = np.array([unit(np.mean([wv[w] for w in probes[k]["pos"] if w.lower() in wv], 0)
                       - np.mean([wv[w] for w in probes[k]["neg"] if w.lower() in wv], 0))
                  for k in AXES])
    return P @ P.T


def sep(M):
    """Row-normalised diagonal / off-diagonal, rows = dial turned."""
    Mn = M / (np.abs(M).max(1, keepdims=True) + 1e-12)
    off = np.concatenate([np.abs(Mn[np.triu_indices(3, 1)]), np.abs(Mn[np.tril_indices(3, -1)])]).mean()
    return np.abs(np.diag(Mn)).mean() / max(off, 1e-9)


def anti_share(M):
    iu = np.triu_indices(3, 1)
    s, a = np.abs(((M + M.T) / 2)[iu]).mean(), np.abs(((M - M.T) / 2)[iu]).mean()
    return a / (s + a) if s + a > 0 else 0.0


def corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triple-gens", nargs="+", required=True)
    ap.add_argument("--single-gens", nargs="+", required=True)
    ap.add_argument("--vecs-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--out", default="superposition.json")
    a = ap.parse_args()

    labels = {}
    for r in csv.DictReader(open(a.labels, newline="", encoding="utf-8")):
        labels[r["key"]] = (r.get("final", "").strip().upper() or r["heuristic"])
    probes = json.load(open(a.probes, encoding="utf-8"))

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(a.embedder)
    cache = {}

    def mean_emb(texts):
        h = hashlib.sha1("\x00".join(texts).encode("utf-8", "replace")).hexdigest()
        if h not in cache:
            cache[h] = np.asarray(st.encode(list(texts), convert_to_numpy=True,
                                            show_progress_bar=False, batch_size=64),
                                  dtype=np.float64).mean(0)
        return cache[h]

    Denc = {k: unit(mean_emb(probes[k]["pos"]) - mean_emb(probes[k]["neg"])) for k in AXES}

    data = {arm: (load(a.triple_gens, arm), load(a.single_gens, arm)) for arm in ("raw", "dual")}
    models = sorted(set(data["raw"][0]) & set(data["raw"][1]))
    print(f"{len(models)} model(s) with both designs\n")
    res = {}
    for model in models:
        vp = os.path.join(a.vecs_dir, model.replace("/", "_") + "_vecs.npz")
        G = activation_gram(vp, LAYERS[model], labels, probes); Gi = np.linalg.inv(G)
        per = {}
        for arm in ("raw", "dual"):
            T, S = data[arm][0].get(model, {}), data[arm][1].get(model, {})
            rows = []
            for tid, (tri, cells) in T.items():
                if tid not in S or len(cells) < 8:
                    continue
                stri, scells = S[tid]
                if tuple(stri) != tuple(tri) or len(scells) < 6:
                    continue
                idx = [AXES.index(k) for k in tri]
                D3 = np.array([Denc[k] for k in tri])
                read = lambda texts: D3 @ mean_emb(texts)
                Y = np.array([read(cells[tuple(int(x) for x in l)]) for l in L8])
                M_obs = (L8.T @ Y) / 8.0                                   # rows: dial
                R = np.array([(read(scells[tuple(1 if j == i else 0 for j in range(3))])
                               - read(scells[tuple(-1 if j == i else 0 for j in range(3))])) / 2
                              for i in range(3)])                          # rows: dial
                K = G[np.ix_(idx, idx)] if arm == "raw" else Gi[np.ix_(idx, idx)]
                w = np.ones(3) if arm == "raw" else np.sqrt(np.diag(Gi)[idx])
                Yp = np.array([(l * w / np.sqrt(l @ K @ l)) @ R for l in L8])
                M_pred = (L8.T @ Yp) / 8.0
                rows.append(dict(trial=tid, triple=list(tri), M_obs=M_obs.tolist(),
                                 M_pred=M_pred.tolist(), single=R.tolist()))
            per[arm] = rows
        # noise ceiling: repetition 1 against repetition 2 of each triple
        def ceiling(rows):
            byt = defaultdict(list)
            for r in rows:
                byt[tuple(r["triple"])].append(np.array(r["M_obs"]))
            pairs = [v[:2] for v in byt.values() if len(v) >= 2]
            return corr([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else float("nan")
        st_ = {}
        for arm, rows in per.items():
            Mo = [np.array(r["M_obs"]) for r in rows]
            Mp = [np.array(r["M_pred"]) for r in rows]
            Ms = [np.array(r["single"]) for r in rows]
            st_[arm] = dict(
                n_trials=len(rows),
                agreement=corr(Mp, Mo), ceiling=ceiling(rows),
                sep_obs=float(np.mean([sep(m) for m in Mo])),
                sep_pred=float(np.mean([sep(m) for m in Mp])),
                sep_single=float(np.mean([sep(m) for m in Ms])),
                anti_obs=float(np.mean([anti_share(m) for m in Mo])),
                anti_pred=float(np.mean([anti_share(m) for m in Mp])),
                anti_single=float(np.mean([anti_share(m) for m in Ms])))
        if "raw" in st_ and "dual" in st_:
            for k in ("sep_obs", "sep_pred", "sep_single"):
                st_[f"dual_gain_{k[4:]}"] = st_["dual"][k] / st_["raw"][k]
        res[model] = dict(stats=st_, trials=per)
        r_, d_ = st_["raw"], st_.get("dual", {})
        print(f"{model}")
        print(f"   superposition: prediction vs observed r = {r_['agreement']:.2f}  "
              f"(repetition ceiling {r_['ceiling']:.2f})")
        print(f"   antisymmetric share: single-dial {r_['anti_single']:.0%}, predicted "
              f"{r_['anti_pred']:.0%}, observed {r_['anti_obs']:.0%}")
        if d_:
            print(f"   dual vs raw separability: one at a time x{st_['dual_gain_single']:.2f}, "
                  f"three predicted x{st_['dual_gain_pred']:.2f}, three observed x{st_['dual_gain_obs']:.2f}")
        print()

    S = [v["stats"] for v in res.values() if "dual_gain_obs" in v["stats"]]
    if S:
        g = lambda k: [s[k] for s in S]
        rr = lambda k: [s["raw"][k] for s in S]
        summ = dict(
            n_models=len(S),
            agreement_mean=float(np.mean(rr("agreement"))),
            ceiling_mean=float(np.nanmean(rr("ceiling"))),
            models_agreement_below_ceiling=int(sum(a_ < c for a_, c in zip(rr("agreement"), rr("ceiling")))),
            anti_single=float(np.mean(rr("anti_single"))),
            anti_pred=float(np.mean(rr("anti_pred"))),
            anti_obs=float(np.mean(rr("anti_obs"))),
            dual_gain_single=float(np.mean(g("dual_gain_single"))),
            dual_gain_pred=float(np.mean(g("dual_gain_pred"))),
            dual_gain_obs=float(np.mean(g("dual_gain_obs"))),
            models_single_improves=int(sum(x > 1 for x in g("dual_gain_single"))),
            models_three_improves=int(sum(x > 1 for x in g("dual_gain_obs"))))
        print("ACROSS MODELS")
        for k, v in summ.items():
            print(f"   {k:<34}{v:.3f}" if isinstance(v, float) else f"   {k:<34}{v}")
        print("""
  READING
   agreement near the ceiling       -> steering superposes; three-dial behaviour
                                       is predicted by single dials
   agreement well below the ceiling -> it does not; the gap is the non-linearity
   dual improves one at a time but
   not three at once                -> the correction's failure is a failure of
                                       superposition, not of geometry""")
        res["_summary"] = summ
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
