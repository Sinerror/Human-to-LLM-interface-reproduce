#!/usr/bin/env python3
"""
make_atlas.py — build the token atlas the interface embeds.

  python make_atlas.py --model google/gemma-3-1b-it \
      --vecs lat_outputs/google_gemma-3-1b-it_vecs.npz \
      --labels frozen_emotionality_labels.csv --probes canonical_probes.json \
      --layer 17 --c-star 0.06 --ekman ekman.json \
      --out atlas_gemma-3-1b-it.json

WHAT THE ATLAS IS

  Every token in the model's vocabulary, given a coordinate in the seven-axis
  construct space. The unembedding row for token t is projected onto each axis,
  so a token's coordinate says how strongly that token is promoted by moving
  along each construct.

  This is a logit-lens readout. It shows what the intervention does to the
  model's output distribution without generating anything, which is why the
  interface leads with it: instant, deterministic, and it sidesteps the
  degeneracy that makes short generated samples an unreliable demonstration.

WHAT IS STORED, AND WHY IT IS SMALL

  Coordinates are quantised to int8 with a per-axis scale. A 262k vocabulary at
  7 axes is 1.8 MB as int8 against 7.3 MB as float32, and the quantisation error
  is far below the noise in the underlying projections. Tokens are filtered to
  those that are printable and reasonably word-like, which removes most of the
  byte-fallback and reserved entries and cuts the vocabulary roughly in half.

  Also stored: the Gram matrix, the dual basis coefficients, c*, and the Ekman
  category positions in the same coordinates, so the interface can show cursor
  proximity to human categories without another lookup.

  The axes themselves are NOT stored. The atlas is a readout, not a steering
  artefact, and shipping D-dimensional directions would make this a model file.
"""
import argparse, sys, json, re, unicodedata
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
    P, kept, poles = [], [], {}
    for nm in names:
        s = probes.get(nm, {})
        pos = [w.lower() for w in s.get("pos", []) if w.lower() in wv]
        neg = [w.lower() for w in s.get("neg", []) if w.lower() in wv]
        if len(pos) >= 4 and len(neg) >= 4:
            P.append(unit(np.mean([wv[w] for w in pos], 0)
                          - np.mean([wv[w] for w in neg], 0)))
            kept.append(nm)
            poles[nm] = dict(pos=s.get("pos", [])[:8], neg=s.get("neg", [])[:8])
    return np.array(P), kept, poles, wv


def keepable(s):
    """Printable, word-like tokens. Drops byte fallbacks and reserved entries."""
    t = s.replace("\u2581", " ").replace("Ġ", " ").strip()
    if not (1 <= len(t) <= 20):
        return None
    if not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", t):
        return None
    if any(unicodedata.category(ch).startswith("C") for ch in t):
        return None
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--c-star", type=float, required=True,
                    help="from the coherence-edge sweep; baked in as the default")
    ap.add_argument("--ekman", default=None, help="ekman.json, optional")
    ap.add_argument("--axes", nargs="*",
                    default=["valence", "heat", "arousal", "intensity",
                             "antagonism_peace", "body_mind",
                             "social_inner_outer"])
    ap.add_argument("--top-per-axis", type=int, default=4000,
                    help="tokens kept per axis pole; the union is stored")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    labels = load_labels(a.labels)
    probes = json.load(open(a.probes, encoding="utf-8"))
    P, names, poles, wv = build_axes(a.vecs, a.layer, labels, probes, a.axes)
    G = P @ P.T
    D = np.linalg.solve(G, P)          # dual rows, for the raw/dual toggle
    print(f"{len(names)} axes: {names}")
    print(f"  Gram mean |cos| {np.abs(G[np.triu_indices(len(G),1)]).mean():.3f}"
          f"   condition {np.linalg.cond(G):.2f}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, low_cpu_mem_usage=True,
        device_map={"": a.device})
    model.eval()
    W = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    print(f"  unembedding {W.shape}")

    # token coordinates: project each unembedding row onto the seven axes
    coords = W @ P.T                                   # (V, k)
    del W

    ids, toks = [], []
    for i in range(coords.shape[0]):
        s = keepable(tok.convert_ids_to_tokens(i) or "")
        if s:
            ids.append(i); toks.append(s)
    ids = np.array(ids)
    print(f"  word-like tokens {len(ids)} of {coords.shape[0]}")

    # keep the extremes of every axis; the middle of the distribution is not
    # informative for a nearest-token readout and dominates the file size
    keep = set()
    for j in range(len(names)):
        col = coords[ids, j]
        order = np.argsort(col)
        keep.update(ids[order[:a.top_per_axis]].tolist())
        keep.update(ids[order[-a.top_per_axis:]].tolist())
    keep = np.array(sorted(keep))
    sel = coords[keep]
    lut = {int(v): k for k, v in enumerate(ids)}
    words_out = [toks[lut[int(i)]] for i in keep]
    print(f"  atlas tokens {len(keep)}")

    scale = np.abs(sel).max(axis=0) / 127.0
    q = np.clip(np.round(sel / scale), -127, 127).astype(np.int8)

    ek = {}
    if a.ekman:
        E = json.load(open(a.ekman, encoding="utf-8"))
        for cat, spec in E.items():
            ws = [w.lower() for w in spec["pos"] if w.lower() in wv]
            if len(ws) >= 2:
                c = np.mean([wv[w] for w in ws], 0)
                ek[cat] = (P @ c).tolist()
        print(f"  Ekman categories placed: {len(ek)}")

    out = dict(
        model=a.model, layer=a.layer, c_star=a.c_star, axes=names,
        poles=poles,
        gram=G.tolist(), dual=D.tolist() if D.shape[1] < 64 else None,
        gram_condition=float(np.linalg.cond(G)),
        iso_cost=np.sqrt(np.diag(np.linalg.inv(G))).tolist(),
        scale=scale.tolist(),
        tokens=words_out,
        coords=q.T.tolist(),           # axis-major: k lists of N int8
        ekman=ek,
        note="coords are int8; multiply by scale[axis] to recover the projection")
    json.dump(out, open(a.out, "w"))
    import os
    print(f"\n-> {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")
    print("  embed with:  python bake_interface.py --atlas <this> --out index.html")


if __name__ == "__main__":
    main()
