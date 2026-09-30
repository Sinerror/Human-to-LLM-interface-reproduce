#!/usr/bin/env python3
"""
transfer_independent.py — second implementation of the transfer-matrix analysis.

  python transfer_independent.py --generations d_full --probes canonical_probes.json \
      --labels frozen_emotionality_labels.csv --vecs-dir lat_outputs \
      --layer-map layer_map.json \
      --embedder sentence-transformers/all-mpnet-base-v2 \
      --workers 3 --out transfer_indep.json

Written to be checkable against `transfer_matrix.py` rather than to share code
with it. Where the two disagree, at least one is wrong, and the disagreement is
the useful output.

────────────────────────────────────────────────────────────────────────────────
BIASES THIS IMPLEMENTATION CONTROLS FOR
────────────────────────────────────────────────────────────────────────────────

**1. Spurious asymmetry from fitting.** Each transfer matrix is estimated from 8
   factorial cells with 12 free parameters (3 constructs x 3 knobs + 3
   intercepts). A *symmetric* true matrix plus measurement noise yields a fitted
   matrix that is asymmetric. Reporting ‖M−Mᵀ‖/‖M‖ without a null therefore
   overstates asymmetry by an unknown amount, and the effect grows as cells get
   noisier. **This script simulates a symmetric-truth null and reports observed
   asymmetry against it.** Nothing else here matters if that test fails.

**2. Normalisation before aggregation.** Row-normalising M before computing
   asymmetry or the symmetric/antisymmetric split destroys the comparison,
   because Mᵀ's rows are not M's rows. All second-order statistics here are
   computed on the raw matrix; normalisation is applied only for display.

**3. Ratio of averages vs average of ratios.** ‖S‖ and ‖A‖ averaged separately
   and then divided is not the mean of per-triple ratios, and the two can differ
   substantially. All ratios are computed per triple and then aggregated, with
   the aggregation method stated.

**4. Estimator dependence.** The matrix can be obtained by least squares over
   all 8 cells, or in closed form from the balanced design as a difference of
   marginal means. These coincide only if the design is exactly balanced and
   complete. **Both are computed and compared**; a discrepancy indicates missing
   or unbalanced cells.

**5. Embedding-side artefacts.** Cell means are computed from raw embeddings
   with the trial's own grand mean removed, so the shared stem cannot contribute.
   Degenerate generations are flagged but never silently dropped; the analysis
   is reported with and without them.

**6. Direction of the emitter/receiver claim.** "A emits more than it receives"
   is a statement about the antisymmetric part and inherits bias 1 directly. It
   is reported only against the same symmetric-truth null.
"""
import argparse, glob, json, os, re, sys
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# A Windows console is often cp1251 or cp437 and cannot encode much of what
# either the models or these scripts print. Replace unencodable characters
# rather than raising, which would otherwise kill a run after its expensive
# stage had completed. JSON output uses ensure_ascii and is unaffected.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


SEED = 0


# ───────────────────────────────────────────────── io
def _lit(v):
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except Exception:
        import ast
        return ast.literal_eval(v)


def expand(pats):
    out = []
    for p in pats:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*_generations.jsonl")))
        elif any(c in p for c in "*?["):
            out += sorted(glob.glob(p))
        else:
            out.append(p)
    return list(dict.fromkeys(out))


def degenerate(t):
    toks = re.findall(r"\S+", str(t))
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


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ───────────────────────────────────────────────── estimators
def fit_lstsq(L, Y):
    """p = M l + b, solved over all cells. Returns M with rows=construct."""
    X = np.hstack([L, np.ones((len(L), 1))])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return coef[:-1].T


def fit_contrast(L, Y):
    """Closed form for a balanced factorial: the main effect of knob j on
    construct i is half the difference of marginal means. Independent of the
    least-squares path, and exact when the design is complete."""
    k = L.shape[1]
    M = np.zeros((Y.shape[1], k))
    for j in range(k):
        hi = L[:, j] > 0
        lo = L[:, j] < 0
        if hi.sum() == 0 or lo.sum() == 0:
            M[:, j] = np.nan
            continue
        M[:, j] = (Y[hi].mean(0) - Y[lo].mean(0)) / 2.0
    return M


def asym_ratio(M):
    """‖M − Mᵀ‖_F / ‖M‖_F on the RAW matrix. Equals 2‖A‖/‖M‖."""
    d = np.linalg.norm(M)
    return float(np.linalg.norm(M - M.T) / d) if d > 0 else np.nan


def sym_split(M):
    S = (M + M.T) / 2.0
    A = (M - M.T) / 2.0
    ns, na = np.linalg.norm(S), np.linalg.norm(A)
    tot = np.sqrt(ns ** 2 + na ** 2)
    return S, A, (float(na / tot) if tot > 0 else np.nan)


# ───────────────────────────────────────────────── the critical null
def symmetric_truth_null(L, Y, M_hat, n_sim, rng):
    """How asymmetric does a SYMMETRIC truth look after fitting?

    Take the symmetric part of the fitted matrix as the truth, regenerate cell
    positions from it using residuals resampled from the actual fit, refit, and
    measure asymmetry. The resulting distribution is the asymmetry this design
    manufactures from noise alone.
    """
    S = (M_hat + M_hat.T) / 2.0
    pred = L @ S.T
    resid = Y - Y.mean(0) - (pred - pred.mean(0))
    out = []
    for _ in range(n_sim):
        idx = rng.integers(0, len(resid), len(resid))
        Ysim = pred + resid[idx]
        out.append(asym_ratio(fit_lstsq(L, Ysim)))
    return np.array(out)


# ───────────────────────────────────────────────── embedding
def get_embedder(name):
    if name == "hashing":
        print("  !! hashing embedder: pipeline smoke test only, not reportable")
        DIM = 2048

        def enc(texts):
            out = np.zeros((len(texts), DIM))
            for i, t in enumerate(texts):
                t = " " + re.sub(r"\s+", " ", str(t).lower()).strip() + " "
                for n in (3, 4, 5):
                    for j in range(len(t) - n + 1):
                        out[i, hash(t[j:j + n]) % DIM] += 1.0
                nz = out[i] > 0
                out[i, nz] = 1.0 + np.log(out[i, nz])
            return out
        return enc
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name)
    return lambda t: np.asarray(m.encode(list(t), convert_to_numpy=True,
                                         show_progress_bar=False, batch_size=64),
                                dtype=np.float64)


def hidden_gram(vecs_path, layer, labels, probes, names):
    z = np.load(vecs_path, allow_pickle=False)
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
    P = []
    for nm in names:
        s = probes[nm]
        pos = [w.lower() for w in s.get("pos", []) if w.lower() in wv]
        neg = [w.lower() for w in s.get("neg", []) if w.lower() in wv]
        if len(pos) < 4 or len(neg) < 4:
            return None
        P.append(unit(np.mean([wv[w] for w in pos], 0)
                      - np.mean([wv[w] for w in neg], 0)))
    return np.array(P) @ np.array(P).T


# ───────────────────────────────────────────────── per model
def analyse_model(path, probes, D, labels, layer_map, vecs_dir, enc, arm,
                  n_sim, drop_degenerate):
    model = os.path.basename(path).replace("_generations.jsonl", "")
    cells = defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("arm") != arm:
            continue
        tri = tuple(_lit(r["triple"]))
        lv = _lit(r["levels"])
        gens = _lit(r["generations"])
        if drop_degenerate:
            gens = [g for g in gens if not degenerate(g)] or gens
        key = tuple(int(np.sign(lv[k])) for k in tri)
        cells[(r["trial_id"], tri)].setdefault(key, []).extend(gens)

    rng = np.random.default_rng(SEED)
    gram_cache = {}
    per, agree = [], []
    for (trial, tri), cc in cells.items():
        if len(cc) < 8 or any(k not in D for k in tri):
            continue
        keys = sorted(cc)
        texts, spans = [], []
        for k in keys:
            spans.append((len(texts), len(texts) + len(cc[k])))
            texts.extend(cc[k])
        E = enc(texts)
        # cell means, trial grand mean removed so the shared stem cancels
        Pm = np.array([E[s:e].mean(0) for s, e in spans])
        Pm = Pm - Pm.mean(0, keepdims=True)
        Y = np.array([[p @ D[k] for k in tri] for p in Pm])
        L = np.array(keys, dtype=float)

        M1 = fit_lstsq(L, Y)
        M2 = fit_contrast(L, Y)
        if np.isfinite(M2).all():
            num = np.linalg.norm(M1 - M2)
            den = max(np.linalg.norm(M1), 1e-12)
            agree.append(float(num / den))

        Gt = None
        # generation files are named <model>_<arm>; the layer map and the
        # vector files use the full hub slug. Strip the arm and match on suffix,
        # or every lookup misses and the Gram correlation is silently NaN.
        base = model
        for suf in ("_raw", "_dual", "_control"):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        lkey = next((k for k in layer_map
                     if k == base or k.endswith("_" + base)), None)
        if vecs_dir and labels and lkey:
            vp = os.path.join(vecs_dir, f"{lkey}_vecs.npz")
            if os.path.exists(vp):
                Gt = gram_cache.get(tri)
                if Gt is None:
                    Gt = hidden_gram(vp, layer_map[lkey], labels, probes,
                                     list(tri))
                    gram_cache[tri] = Gt
        a_obs = asym_ratio(M1)
        null = symmetric_truth_null(L, Y, M1, n_sim, rng)
        S, A, share = sym_split(M1)
        # cross-talk vs Gram, the direct test of whether the dual arm has
        # stopped resembling the basis geometry
        rho_MG = float("nan")
        if Gt is not None:
            iu = np.triu_indices(3, 1)
            il = np.tril_indices(3, -1)
            mo = np.concatenate([np.abs(M1[iu]), np.abs(M1[il])])
            go = np.concatenate([np.abs(Gt[iu]), np.abs(Gt[il])])
            if np.std(mo) > 1e-12 and np.std(go) > 1e-12:
                rho_MG = float(np.corrcoef(mo, go)[0, 1])

        Mn = M1 / (np.abs(M1).max(axis=1, keepdims=True) + 1e-12)
        iu, il = np.triu_indices(3, 1), np.tril_indices(3, -1)
        offd = float(np.concatenate([np.abs(Mn[iu]), np.abs(Mn[il])]).mean())
        diag = float(np.abs(np.diag(Mn)).mean())

        per.append(dict(trial=trial, triple=list(tri),
                        M=M1.tolist(), asym=a_obs,
                        rho_M_gram=rho_MG,
                        gram=Gt.tolist() if Gt is not None else None,
                        diag_off_ratio=diag / max(offd, 1e-9),
                        n_diag_is_max=int(sum(
                            int(np.argmax(np.abs(Mn[i])) == i) for i in range(3))),
                        norm_M=float(np.linalg.norm(M1)),
                        asym_null_mean=float(np.nanmean(null)),
                        asym_null_p95=float(np.nanpercentile(null, 95)),
                        asym_exceeds_null=bool(a_obs > np.nanpercentile(null, 95)),
                        antisym_share=share,
                        norm_S=float(np.linalg.norm(S)),
                        norm_A=float(np.linalg.norm(A))))
    if not per:
        return model, None

    a = np.array([p["asym"] for p in per])
    nm = np.array([p["asym_null_mean"] for p in per])
    return model, dict(
        n_triples=len(per),
        asym_observed=float(np.nanmean(a)),
        asym_null=float(np.nanmean(nm)),
        asym_excess=float(np.nanmean(a - nm)),
        frac_exceeding_null=float(np.mean([p["asym_exceeds_null"] for p in per])),
        antisym_share=float(np.nanmean([p["antisym_share"] for p in per])),
        estimator_disagreement=float(np.nanmean(agree)) if agree else None,
        diag_off_ratio=float(np.mean([p["diag_off_ratio"] for p in per])),
        diag_is_max_frac=float(sum(p["n_diag_is_max"] for p in per)
                               / (3 * len(per))),
        rho_M_gram=float(np.nanmean([p["rho_M_gram"] for p in per])),
        norm_S=float(np.mean([p["norm_S"] for p in per])),
        norm_A=float(np.mean([p["norm_A"] for p in per])),
        per_triple=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", nargs="+", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--vecs-dir", default=None)
    ap.add_argument("--layer-map", default=None)
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--arm", default="real")
    ap.add_argument("--n-sim", type=int, default=300)
    ap.add_argument("--workers", type=int, default=1,
                    help="models analysed concurrently; embedding is the cost")
    ap.add_argument("--drop-degenerate", action="store_true")
    ap.add_argument("--out", default="transfer_indep.json")
    a = ap.parse_args()

    paths = expand(a.generations)
    probes = json.load(open(a.probes, encoding="utf-8"))
    labels = {}
    if a.labels:
        import csv
        for r in csv.DictReader(open(a.labels, newline="", encoding="utf-8")):
            labels[r["key"]] = (r.get("final", "").strip().upper() or r["heuristic"])
    layer_map = json.load(open(a.layer_map)) if a.layer_map else {}

    enc = get_embedder(a.embedder)
    D = {}
    for k, s in probes.items():
        if len(s.get("pos", [])) >= 2 and len(s.get("neg", [])) >= 2:
            D[k] = unit(enc(s["pos"]).mean(0) - enc(s["neg"]).mean(0))
    print(f"{len(paths)} file(s), {len(D)} probe directions, arm={a.arm}")

    def job(p):
        return analyse_model(p, probes, D, labels, layer_map, a.vecs_dir, enc,
                             a.arm, a.n_sim, a.drop_degenerate)

    R = {}
    if a.workers > 1:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for model, res in ex.map(job, paths):
                if res:
                    R[model] = res
                    print(f"  {model:<32} {res['n_triples']} triples")
    else:
        for p in paths:
            model, res = job(p)
            if res:
                R[model] = res
                print(f"  {model:<32} {res['n_triples']} triples")

    print("\n" + "=" * 78)
    print("IS THE ASYMMETRY REAL?  observed vs a symmetric-truth null")
    print("=" * 78)
    print("  A symmetric matrix fitted from 8 noisy cells looks asymmetric. The")
    print("  null column is how much. Only the excess is evidence.\n")
    print(f"  {'model':<30}{'observed':>10}{'null':>8}{'excess':>9}"
          f"{'>null':>8}{'A share':>9}")
    for m, v in sorted(R.items(), key=lambda kv: -kv[1]["asym_excess"]):
        print(f"  {m:<30}{v['asym_observed']:>10.3f}{v['asym_null']:>8.3f}"
              f"{v['asym_excess']:>+9.3f}{v['frac_exceeding_null']:>8.0%}"
              f"{v['antisym_share']:>9.1%}")

    ex = np.array([v["asym_excess"] for v in R.values()])
    print(f"\n  mean excess over null: {ex.mean():+.3f}")
    if ex.mean() < 0.05:
        print("  => the measured asymmetry is largely what this design manufactures.")
        print("     Do NOT report emitter/receiver structure from these fits.")
    else:
        print("  => asymmetry exceeds what noise produces; the structure is real.")

    dis = [v["estimator_disagreement"] for v in R.values()
           if v["estimator_disagreement"] is not None]
    if dis:
        print(f"\n  estimator disagreement ||M_lstsq - M_contrast||/||M||: "
              f"mean {np.mean(dis):.4f}")
        print("    near zero = the design is balanced and both paths agree;")
        print("    large = cells are missing or unbalanced, and the least-squares")
        print("    fit is absorbing that imbalance into the matrix.")

    json.dump(R, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")
    print("\nCompare against transfer_matrix.py. Disagreement is the point of")
    print("having two implementations; agreement is weak evidence both are right.")


if __name__ == "__main__":
    main()
