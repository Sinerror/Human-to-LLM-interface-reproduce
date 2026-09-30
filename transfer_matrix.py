#!/usr/bin/env python3
"""
transfer_matrix.py — does turning one knob turn only that knob?

  python transfer_matrix.py --generations d_out --probes canonical_probes.json \
      --vecs lat_outputs --layers-json layers.json \
      --embedder sentence-transformers/all-mpnet-base-v2 \
      --out transfer.json

WHAT THIS ASKS, AND WHY IT IS NOT WHAT embed_align.py ASKS

  embed_align measures MAIN EFFECTS: for knob K it pools every cell where
  K=+1 against every cell where K=-1, so the other knobs average out. That
  establishes each knob moves text toward its own construct *on average over
  the other knobs' settings*.

  It does NOT establish simultaneous control. Two knobs can each work alone and
  interfere when both are turned; a main-effect analysis cannot see it. An
  interface claim — several knobs, each moving one thing — needs the joint
  question, which is what this script measures.

THE TRANSFER MATRIX

  For a trial with knobs (A,B,C), each of the 8 cells has a level vector
  l = (+-1,+-1,+-1) and, after embedding, a position p in construct space,
  where p_j = <cell mean, D_probe_j>. Fit

      p = M l + b

  M is the 3x3 transfer matrix. Row i says what turning knob i does to every
  construct. **Diagonal dominance is the interface claim.**

THE PREDICTION THAT MAKES THIS DECISIVE

  Steering with the RAW directions d_i by amounts l_i displaces the residual
  stream by sum_i l_i d_i. Projecting onto d_j gives sum_i l_i <d_j,d_i> = (G l)_j.

      raw steering  =>  M is proportional to G, the Gram matrix.
      G's off-diagonal IS the cross-talk.

  Steering with the DUAL rows D = G^-1 P instead gives P D^T = P G^-1 P^T = I.

      dual steering  =>  M = I exactly, in the linear regime.

  So if measured M resembles G rather than I, cross-talk is fully explained by
  basis obliqueness, and the dual basis is the fix rather than a refinement.
  This is testable on generations that already exist, with no new steering.

  It is also the natural explanation for a judge doing poorly at naming which
  knob was turned while a human reading a dual-basis interface sees clean
  separation: they are looking at different bases.

CAVEAT STATED UP FRONT
  Steering happens in HIDDEN space; measurement happens in EMBEDDING space. The
  prediction M ~ G_hidden assumes the embedding readout is roughly linear in
  hidden displacement. G is therefore reported in both spaces, and the
  comparison of M's off-diagonal against each is the honest way to read it.
"""
import argparse, glob as _glob, json, os, re, sys
import sys
from collections import defaultdict
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


def get_embedder(name):
    if name == "hashing":
        print("  !! hashing embedder: SMOKE TEST ONLY, not reportable")
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
    """Gram matrix of the probe directions in HIDDEN space — what steering used."""
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
    P = np.array(P)
    return P @ P.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", nargs="+", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--labels", default=None,
                    help="needed for the hidden-space Gram comparison")
    ap.add_argument("--vecs-dir", default=None,
                    help="directory of *_vecs.npz, for the hidden-space Gram")
    ap.add_argument("--layer-map", default=None,
                    help='json {"model_name": layer}')
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--arm", default="raw",
                    help="generation arm to analyse. steer_gen.py writes raw, "
                         "dual and control; the older harness wrote real. A "
                         "mismatch produced an empty result with no error, so "
                         "the run now stops if nothing matches.")
    ap.add_argument("--min-trials", type=int, default=2,
                    help="minimum trials per triple to fit a transfer matrix")
    ap.add_argument("--out", default="transfer.json")
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
    labels = {}
    if a.labels:
        import csv
        for r in csv.DictReader(open(a.labels, newline="", encoding="utf-8")):
            labels[r["key"]] = (r.get("final", "").strip().upper() or r["heuristic"])
    layer_map = json.load(open(a.layer_map)) if a.layer_map else {}

    enc = get_embedder(a.embedder)
    print(f"embedding probe contrasts with {a.embedder}")
    D = {}
    for k, s in probes.items():
        pos, neg = s.get("pos", []), s.get("neg", [])
        if len(pos) >= 2 and len(neg) >= 2:
            D[k] = unit(enc(pos).mean(0) - enc(neg).mean(0))

    R = {"embedder": a.embedder, "arm": a.arm, "models": {}}

    for path in paths:
        model = os.path.basename(path).replace("_generations.jsonl", "")
        rows = [json.loads(l) for l in open(path, encoding="utf-8")
                if json.loads(l).get("arm") == a.arm]
        by_triple = defaultdict(list)
        for r in rows:
            by_triple[tuple(_lit(r["triple"]))].append(r)
        usable = {t: v for t, v in by_triple.items()
                  if len(v) >= 8 * a.min_trials and all(k in D for k in t)}
        print(f"\n{'=' * 76}\n{model}   {len(usable)} triple(s) with "
              f">= {a.min_trials} trials\n{'=' * 76}", flush=True)
        if not usable:
            continue

        per_triple, diag_ok, offd_all, gram_all = {}, 0, [], []
        for triple, recs in sorted(usable.items()):
            # cell -> mean embedding
            cells = defaultdict(list)
            for r in recs:
                lv = _lit(r["levels"])
                key = tuple(int(np.sign(lv[k])) for k in triple)
                cells[key].extend(_lit(r["generations"]))
            if len(cells) < 8:
                continue
            keys = sorted(cells)
            texts, spans = [], []
            for k in keys:
                spans.append((len(texts), len(texts) + len(cells[k])))
                texts.extend(cells[k])
            E = enc(texts)
            Pmat = np.array([E[s:e].mean(0) for s, e in spans])
            Pmat = Pmat - Pmat.mean(0, keepdims=True)     # centre across cells
            # position in construct space
            Y = np.array([[unit(p) @ D[k] for k in triple] for p in Pmat])

            # ---- MAGNITUDE, in units a reader can interpret -----------------
            # 1. pole fraction: how far along the probe's own pole-to-pole
            #    distance does steering move the text? The probe direction is
            #    normalised, so the separation must be recovered from the
            #    unnormalised pole means.
            # 2. Cohen's d: the same displacement in units of the spread of
            #    individual generations along that construct.
            mag = {}
            for ci, k in enumerate(triple):
                pos_e = enc(probes[k]["pos"]).mean(0)
                neg_e = enc(probes[k]["neg"]).mean(0)
                pole_sep = float(np.linalg.norm(pos_e - neg_e))
                hi = np.array([Pmat[i] for i, kk in enumerate(keys) if kk[ci] > 0])
                lo = np.array([Pmat[i] for i, kk in enumerate(keys) if kk[ci] < 0])
                disp = float((hi.mean(0) - lo.mean(0)) @ D[k])
                # SD must be on the SAME scale as `disp`: raw embedding space,
                # projected on the construct. Unit-normalising here (an earlier
                # bug) shrinks the denominator and inflates d by ~40x.
                gm = E.mean(0)
                gen_sd = float(np.std([(E[i] - gm) @ D[k] for i in range(len(E))]))
                # Two denominators, because they answer different questions.
                # pole_fraction: displacement as a share of the construct's own
                #   pole-to-pole distance. A tightly specified construct has a
                #   short pole distance, so this rewards good word lists.
                # corpus_fraction: the same displacement against the spread of
                #   the whole emotion corpus along that axis. This asks how far
                #   the text moved relative to the range the construct spans in
                #   natural language, independent of how the poles were written.
                corpus_sd = float(np.std([(E[i] - E.mean(0)) @ D[k]
                                          for i in range(len(E))]))
                corpus_rng = float(np.ptp([(E[i] - E.mean(0)) @ D[k]
                                           for i in range(len(E))]))
                mag[k] = dict(pole_fraction=disp / pole_sep if pole_sep > 0 else float("nan"),
                              corpus_fraction=disp / corpus_rng if corpus_rng > 0 else float("nan"),
                              displacement=disp, pole_separation=pole_sep,
                              corpus_range=corpus_rng, corpus_sd=corpus_sd,
                              cohens_d=disp / gen_sd if gen_sd > 0 else float("nan"))
            L = np.array(keys, dtype=float)
            # least squares p = M l + b  (b absorbed by centring L too)
            Lc = L - L.mean(0, keepdims=True)
            M, *_ = np.linalg.lstsq(Lc, Y, rcond=None)     # rows: knob, cols: construct
            M = M.T                                        # rows: construct, cols: knob
            # row-normalise so the diagonal is comparable across knobs
            Mn = M / (np.abs(M).max(axis=1, keepdims=True) + 1e-12)
            d_ok = int(sum(int(np.argmax(np.abs(Mn[i])) == i) for i in range(3)))
            diag_ok += d_ok

            G_emb = np.array([[D[i] @ D[j] for j in triple] for i in triple])
            G_hid = None
            if a.vecs_dir and labels and model in layer_map:
                vp = os.path.join(a.vecs_dir, f"{model}_vecs.npz")
                if os.path.exists(vp):
                    G_hid = hidden_gram(vp, layer_map[model], labels, probes, triple)

            iu = np.triu_indices(3, 1)
            off = np.abs(Mn[iu])
            offd_all.extend(off.tolist())
            if G_hid is not None:
                gram_all.extend(np.abs(G_hid[iu]).tolist())
            per_triple["|".join(triple)] = dict(
                M=Mn.tolist(), M_raw=M.tolist(), magnitude=mag,
                G_embed=G_emb.tolist(),
                G_hidden=G_hid.tolist() if G_hid is not None else None,
                diag_is_max=d_ok, n_cells=len(keys),
                mean_abs_offdiag=float(off.mean()),
                mean_diag=float(np.mean([abs(Mn[i, i]) for i in range(3)])))

        if not per_triple:
            continue
        nt = len(per_triple)
        md = np.mean([v["mean_diag"] for v in per_triple.values()])
        mo = np.mean([v["mean_abs_offdiag"] for v in per_triple.values()])
        print(f"  triples fitted            {nt}")
        print(f"  diagonal is max           {diag_ok}/{nt * 3} knob rows "
              f"({diag_ok / (nt * 3):.0%})")
        print(f"  mean |diagonal|           {md:.3f}")
        print(f"  mean |off-diagonal|       {mo:.3f}")
        print(f"  diag / off ratio          {md / max(mo, 1e-9):.2f}")
        if gram_all:
            from scipy.stats import spearmanr
            rho = spearmanr(offd_all[:len(gram_all)], gram_all).statistic
            print(f"  corr(|M offdiag|, |G_hidden offdiag|)  rho = {rho:+.3f}")
            print(f"    positive => cross-talk is explained by basis obliqueness,")
            print(f"    and the dual basis (M = I by construction) is the fix.")
            R["models"].setdefault(model, {})["rho_M_vs_Ghidden"] = float(rho)

        agg = defaultdict(lambda: defaultdict(list))
        for t in per_triple.values():
            for k, m_ in t.get("magnitude", {}).items():
                agg[k]["pole"].append(m_["pole_fraction"])
                agg[k]["corpus"].append(m_.get("corpus_fraction", float("nan")))
                agg[k]["d"].append(m_["cohens_d"])
        if agg:
            print(f"\n  MAGNITUDE -- how far does one knob actually move the text?")
            print(f"  {'knob':<22}{'pole frac':>11}{'corpus frac':>13}"
                  f"{"Cohen's d":>12}{'n':>5}")
            for k in sorted(agg, key=lambda x: -np.nanmean(agg[x]["pole"])):
                print(f"  {k:<22}{np.nanmean(agg[k]['pole']):>10.1%}"
                      f"{np.nanmean(agg[k]['corpus']):>13.1%}"
                      f"{np.nanmean(agg[k]['d']):>12.2f}{len(agg[k]['pole']):>5}")
            print("  pole fraction = displacement as a share of the probe's own")
            print("  positive-to-negative distance. Cohen's d = the same shift in")
            print("  units of the spread across individual generations.")
            R["models"].setdefault(model, {})["magnitude"] = {
                k: dict(pole_fraction=float(np.nanmean(v["pole"])),
                        corpus_fraction=float(np.nanmean(v["corpus"])),
                        cohens_d=float(np.nanmean(v["d"])), n=len(v["pole"]))
                for k, v in agg.items()}

        ex = sorted(per_triple.items(),
                    key=lambda kv: -kv[1]["mean_abs_offdiag"])[:2]
        for name, v in ex:
            print(f"\n  worst cross-talk: {name}")
            ks = name.split("|")
            print(f"    {'':<20}" + "".join(f"{k[:9]:>10}" for k in ks))
            for i, k in enumerate(ks):
                print(f"    {k:<20}" + "".join(f"{v['M'][i][j]:>+10.2f}"
                                               for j in range(3)))
            if v["G_hidden"]:
                print(f"    G_hidden off-diagonal: "
                      f"{[round(v['G_hidden'][i][j], 2) for i, j in zip(*np.triu_indices(3, 1))]}")

        R["models"].setdefault(model, {}).update(
            n_triples=nt, diag_is_max=diag_ok, diag_rows=nt * 3,
            mean_diag=float(md), mean_offdiag=float(mo), triples=per_triple)

    if not R.get("models"):
        sys.exit(f"\nNO GENERATIONS MATCHED arm={a.arm!r}. Nothing was analysed "
                 f"and {a.out} was not written. steer_gen.py writes 'raw', "
                 f"'dual' and 'control'; the older harness wrote 'real'.")
    json.dump(R, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")
    print("\nREAD IT LIKE THIS")
    print("  diag/off ratio near 1      knobs are entangled; not an interface")
    print("  diag/off ratio >> 1        knobs are separable as steered")
    print("  rho(M, G_hidden) positive  the entanglement is basis obliqueness,")
    print("                             so dual-basis steering should remove it")


if __name__ == "__main__":
    main()
