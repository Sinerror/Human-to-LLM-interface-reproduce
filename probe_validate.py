#!/usr/bin/env python3
"""
probe_validate.py — earn a knob's place in the panel without a VAD gate.

  python probe_validate.py --vecs M1.npz M2.npz ... --layers 17 25 ... \
      --labels frozen_emotionality_labels.csv --probes canonical_probes.json \
      [--knobs valence drive arousal heat security_alarm desire antagonism_peace \
               selfpain_excite engagement] \
      [--predictions preregistered_words.csv] --out probe_validation.json

WHY NOT THE SB GATE
  SB is the split-half reliability of a fitted direction — the same family of
  self-consistency statistic that tracks the eigengap rather than signal. On
  gemma-3-1b-it L17 it fails `heat` (pole coherence 0.326, the highest in the
  set) and passes `drive` (0.075, the lowest). It is reported here for every
  probe, and never used to exclude one.

WHY NOT A VAD GATE EITHER
  VAD is three dimensions of human affect theory. Gating on it would delete
  exactly the findings that make a model-side space worth building: an axis with
  no human category name (F4), and structure lying outside the frozen space
  (F7). A model-side space must be allowed to contain things human taxonomies
  do not.

THE THREE TESTS

  A  CROSS-MODEL AGREEMENT
     A direction that is an artifact of one model's geometry should not order
     the corpus the same way in another. Hidden spaces have different
     dimensions, so directions cannot be compared directly — instead each probe
     direction is used to SCORE all senses, and the score vectors are compared
     across models by rank correlation. Null: random directions in each model,
     scored and correlated the same way.
     Non-circular: the probe words are fixed; the models are independent.

  B  LEAVE-ONE-WORD-OUT
     Drop pole word w, rebuild the direction from the rest, then ask where w
     lands relative to the opposite pole's held-out words. If the construct is
     real, held-out members project to their own side. The word being tested
     never touches the direction it is scored against.
     Non-circular by holdout.

  C  PRE-REGISTERED UNSEEN WORDS (optional, --predictions)
     Words absent from the corpus, assigned an expected pole BEFORE extraction.
     This is the standard that validated the axes at AUC 0.88-1.00 (F2), applied
     to knobs. Format: csv with columns  word,probe,side  where side is pos|neg.
     Requires those words to be present in the vectors, so they must have been
     extracted; if a word is missing it is reported, never silently skipped.

WHAT PASSES
  Nothing here is a gate by default. The script reports each test and a
  SUMMARY line per probe. Use --require to turn chosen tests into a filter and
  write panel_axes.npy from the survivors.
"""
import argparse, csv, json, os, sys
import sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

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
OPTIMIZED = ["valence", "drive", "arousal", "heat", "security_alarm",
             "desire", "antagonism_peace", "selfpain_excite", "engagement"]


# ───────────────────────────────────────────────────────── loading
def load_labels(p):
    out = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["key"]] = (r.get("final", "").strip().upper() or r["heuristic"])
    return out


def wordvecs(npz_path, layer, labels, emo_only=True, center_with=None):
    """Returns (word -> mean centered vector, corpus mean).

    center_with: subtract this mean instead of the file's own. Required when
    projecting held-out words onto a direction built from the original corpus —
    centering held-out words on their own mean would move them into a different
    frame and silently change the projection."""
    z = np.load(npz_path, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    words = np.array([str(w).lower() for w in z["words"]])
    layers = [int(x) for x in np.asarray(z["layers"])]
    lab = np.array([labels.get(k, "UNLABELLED") for k in keys])
    m = (lab == "EMO") if emo_only else np.ones(len(keys), bool)
    X = z["vecs"][m, layers.index(layer), :].astype(np.float64)
    mu = X.mean(0, keepdims=True) if center_with is None else center_with
    Xc = X - mu
    acc = defaultdict(list)
    for w, v in zip(words[m], Xc):
        acc[w].append(v)
    return {w: np.mean(vs, 0) for w, vs in acc.items()}, mu


def build(wv, pos, neg):
    d = np.mean([wv[w] for w in pos], 0)
    if neg:
        d = d - np.mean([wv[w] for w in neg], 0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


def sb_score(wv, pos, neg, rng, reps=150):
    cs = []
    for _ in range(reps):
        ip = rng.permutation(len(pos)); inn = rng.permutation(len(neg))
        cs.append(float(build(wv, [pos[i] for i in ip[::2]], [neg[i] for i in inn[::2]])
                        @ build(wv, [pos[i] for i in ip[1::2]], [neg[i] for i in inn[1::2]])))
    r = float(np.mean(cs))
    return 2 * r / (1 + r) if r > -1 else float("nan")


def auc(pos_scores, neg_scores):
    """Rank-based AUC; no sklearn dependency, ties handled."""
    a = np.concatenate([pos_scores, neg_scores])
    order = a.argsort().argsort() + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, order)
    order = (sums / cnt)[inv]
    n1, n0 = len(pos_scores), len(neg_scores)
    return float((order[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def probe_words(spec, wv):
    return ([w.lower() for w in spec.get("pos", []) if w.lower() in wv],
            [w.lower() for w in spec.get("neg", []) if w.lower() in wv])


# ───────────────────────────────────────────────────────── TEST B
def _lowo_acc(wv, pos, neg):
    c = t = 0
    for wp in pos:
        for wn in neg:
            d = build(wv, [w for w in pos if w != wp], [w for w in neg if w != wn])
            c += int(float(wv[wp] @ d) > float(wv[wn] @ d)); t += 1
    return c / t, t


def test_b_lowo(wv, pos, neg, rng, n_null=60):
    """Hold out one word per pole, rebuild, score the held-out pair.

    NULL: permute the pole assignment among the SAME words and repeat. This is
    the null that matters. Substituting random corpus words instead only asks
    whether the direction points somewhere, and lands at 0.5 by construction —
    it does not ask whether THIS partition of these words is the meaningful one.
    Under the permuted-partition null the observed values still clear
    decisively (1.000 against a p95 of 0.75-0.87), so the test survives being
    made harder.
    """
    if len(pos) < 4 or len(neg) < 4:
        return None
    acc, trials = _lowo_acc(wv, pos, neg)
    pool = list(pos) + list(neg)
    nulls = []
    for _ in range(n_null):
        sh = list(rng.permutation(pool))
        nulls.append(_lowo_acc(wv, sh[:len(pos)], sh[len(pos):])[0])
    p95 = float(np.percentile(nulls, 95))
    mu, sd = float(np.mean(nulls)), float(np.std(nulls))
    return dict(accuracy=acc, n_pairs=trials,
                null_mean=mu, null_sd=sd, null_p95=p95,
                z=float((acc - mu) / sd) if sd > 1e-9 else float("inf"),
                clears_null=bool(acc > p95))


# ───────────────────────────────────────────────────────── TEST A
def test_a_cross_model(dirs_per_model, wv_per_model, shared_words, rng,
                      n_pos, n_neg, n_null=200):
    """Do the models order the same words the same way along this direction?

    Directions live in different hidden spaces and cannot be compared directly,
    so each is used to score the SHARED vocabulary and the score vectors are
    compared by Spearman correlation.

    NULL: arbitrary word contrasts of the SAME pole sizes, built in each model
    and correlated the same way. A random-*direction* null is far too weak here
    (p95 ~ 0.25) because any contrast between two word groups inherits lexical
    structure — frequency, length, part of speech, semantic field — that every
    language model encodes, so cross-model agreement is easy to obtain by
    accident. The matched null sits at p95 ~ 0.75, and most probes do not clear
    it. That is the honest bar.
    """
    models = list(dirs_per_model)
    if len(models) < 2:
        return None
    scores = {}
    for m in models:
        wv = wv_per_model[m]
        scores[m] = np.array([float(wv[w] @ dirs_per_model[m]) for w in shared_words])
    rhos, nulls = [], []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            rhos.append(float(spearmanr(scores[models[i]], scores[models[j]]).statistic))
    # The statistic is the MEAN over all model pairs, so each null draw must
    # also be a mean over all model pairs. Taking the p95 of SINGLE pairwise
    # rhos compares a mean against the spread of individuals and is invalid —
    # it put the bar at 0.91-0.93 and failed every probe.
    for _ in range(n_null):
        sample = list(rng.choice(shared_words, n_pos + n_neg, replace=False))
        rs = {}
        for m in models:
            wv = wv_per_model[m]
            d = build(wv, sample[:n_pos], sample[n_pos:])
            rs[m] = np.array([float(wv[w] @ d) for w in shared_words])
        draw = [abs(float(spearmanr(rs[models[i]], rs[models[j]]).statistic))
                for i in range(len(models)) for j in range(i + 1, len(models))]
        nulls.append(float(np.mean(draw)))
    p95 = float(np.percentile(nulls, 95))
    mu, sd = float(np.mean(nulls)), float(np.std(nulls))
    obs = float(np.mean(rhos))
    return dict(mean_rho=obs, min_rho=float(np.min(rhos)),
                max_rho=float(np.max(rhos)), n_pairs=len(rhos),
                null_mean=mu, null_sd=sd, null_p95=p95,
                z=float((obs - mu) / sd) if sd > 1e-9 else float("inf"),
                above_null=bool(obs > p95))


# ───────────────────────────────────────────────────────── TEST C
def test_c_prereg(wv, spec_pos, spec_neg, preds, rng, n_null=200, held=None):
    """Pre-registered unseen words.

    CIRCULARITY. If the prediction words sit in the same corpus the space was
    fitted on, they are not unseen: they helped define the mean, the covariance
    and therefore the direction. `held` supplies vectors extracted SEPARATELY
    and centered on the ORIGINAL corpus mean, so neither the fit nor the
    direction has met them. Without `held` this falls back to corpus words
    excluded from the probe definition, which is a weaker claim and is labelled
    as such in the output.
    """
    d = build(wv, spec_pos, spec_neg)
    src = held if held is not None else wv
    have_p = [w for w, s in preds if s == "pos" and w in src]
    have_n = [w for w, s in preds if s == "neg" and w in src]
    missing = [w for w, _ in preds if w not in src]
    if len(have_p) < 3 or len(have_n) < 3:
        return dict(status="insufficient", n_pos=len(have_p), n_neg=len(have_n),
                    missing=missing, held_out=held is not None)
    sp = np.array([float(src[w] @ d) for w in have_p])
    sn = np.array([float(src[w] @ d) for w in have_n])
    a = auc(sp, sn)
    pool = [w for w in src if w not in set(have_p) | set(have_n)]
    nulls = []
    for _ in range(n_null):
        s_ = rng.choice(pool, len(have_p) + len(have_n), replace=False)
        nulls.append(auc(np.array([float(src[w] @ d) for w in s_[:len(have_p)]]),
                         np.array([float(src[w] @ d) for w in s_[len(have_p):]])))
    p95 = float(np.percentile(nulls, 95))
    mu, sd = float(np.mean(nulls)), float(np.std(nulls))
    return dict(status="ok", auc=a, n_pos=len(have_p), n_neg=len(have_n),
                null_mean=mu, null_sd=sd, null_p95=p95,
                z=float((a - mu) / sd) if sd > 1e-9 else float("inf"),
                clears_null=bool(a > p95), missing=missing,
                held_out=held is not None)



# ───────────────────────────────────────────────────────── leakage
def leakage_report(prompt_path, probes, preds):
    """Do held-out prompts contain probe words from the axis they are scored on?

    TEST C is only clean if the held-out word's own prompts do not name the
    construct. A prompt for `boiling` that reads "He felt a boiling rage" hands
    the answer to `heat` directly, because `rage` is in heat's positive pole.

    Only SAME-PROBE, SAME-SIDE leaks bias the result. A `heat` word appearing in
    a prompt scored on `valence` is irrelevant, and is reported separately so the
    two are never confused.
    """
    import re
    rows = [json.loads(l) for l in open(prompt_path, encoding="utf-8")]
    vocab = defaultdict(list)
    for pb, spec in probes.items():
        for side in ("pos", "neg"):
            for w in spec.get(side, []):
                vocab[w.lower()].append((pb, side))
    side_of = {}
    for pb, lst in preds.items():
        for w, sd in lst:
            side_of[(w, pb)] = sd

    same, other, per_probe = [], 0, defaultdict(int)
    for r in rows:
        tgt = str(r["word"]).lower()
        toks = set(re.findall(r"[a-z']+", str(r["text"]).lower())) - {tgt}
        for t in toks & set(vocab):
            for pb, side in vocab[t]:
                pred = side_of.get((tgt, pb))
                if pred is None:
                    other += 1
                elif pred == side:
                    same.append(dict(word=tgt, leaked=t, probe=pb, side=side,
                                     text=str(r["text"])[:70]))
                    per_probe[pb] += 1
                else:
                    other += 1
    return dict(n_prompts=len(rows), n_same_side=len(same),
                n_other=other, per_probe=dict(per_probe), examples=same[:20])

# ───────────────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", type=int, required=True,
                    help="one layer per --vecs entry, same order")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--knobs", nargs="*", default=OPTIMIZED,
                    help="probe names to validate; default is the optimized set")
    ap.add_argument("--predictions", default=None,
                    help="csv word,probe,side for TEST C")
    ap.add_argument("--heldout-prompts", default=None,
                    help="heldout_prompts.jsonl — checked for probe-word leakage")
    ap.add_argument("--heldout-vecs", nargs="*", default=None,
                    help="separately extracted vectors for the TEST C words, "
                         "one per --vecs entry in the same order. Without these "
                         "TEST C falls back to corpus words and says so.")
    ap.add_argument("--require", nargs="*", default=[],
                    choices=["A", "B", "C"],
                    help="turn these tests into a filter and write panel_axes.npy")
    ap.add_argument("--out", default="probe_validation.json")
    a = ap.parse_args()
    if len(a.vecs) != len(a.layers):
        sys.exit("--vecs and --layers must have the same length")

    labels = load_labels(a.labels)
    probes = json.load(open(a.probes, encoding="utf-8"))
    knobs = [k for k in a.knobs if k in probes]
    for k in a.knobs:
        if k not in probes:
            print(f"[warn] '{k}' not in {a.probes} -- skipped")
    rng = np.random.default_rng(SEED)

    preds = defaultdict(list)
    if a.predictions:
        with open(a.predictions, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                preds[r["probe"]].append((r["word"].strip().lower(),
                                          r["side"].strip().lower()))
        print(f"pre-registered predictions: {sum(len(v) for v in preds.values())} "
              f"words over {len(preds)} probes")

    print(f"\nloading {len(a.vecs)} model(s)...")
    WV, MU, HELD, names = {}, {}, {}, []
    for path, L in zip(a.vecs, a.layers):
        nm = os.path.basename(path).replace("_vecs.npz", "")
        WV[nm], MU[nm] = wordvecs(path, L, labels)
        names.append(nm)
        print(f"  {nm:<34} L{L:<3} {len(WV[nm])} words")
    if a.heldout_vecs:
        if len(a.heldout_vecs) != len(a.vecs):
            sys.exit("--heldout-vecs needs one entry per --vecs, same order")
        print("  held-out vectors (centered on the ORIGINAL corpus mean):")
        for path, L, nm in zip(a.heldout_vecs, a.layers, names):
            HELD[nm], _ = wordvecs(path, L, labels, emo_only=False,
                                   center_with=MU[nm])
            print(f"    {nm:<32} {len(HELD[nm])} words")
    shared = sorted(set.intersection(*[set(WV[n]) for n in names]))
    print(f"  shared vocabulary across models: {len(shared)}")

    R = {"models": names, "layers": a.layers, "shared_words": len(shared),
         "probes": {}}

    for nm in knobs:
        spec = probes[nm]
        entry = {"per_model": {}}
        dirs = {}
        for m in names:
            wv = WV[m]
            pos, neg = probe_words(spec, wv)
            if len(pos) < 4 or len(neg) < 4:
                entry["per_model"][m] = dict(status="too few corpus words",
                                             n_pos=len(pos), n_neg=len(neg))
                continue
            dirs[m] = build(wv, pos, neg)
            e = dict(status="ok", n_pos=len(pos), n_neg=len(neg),
                     sb=round(sb_score(wv, pos, neg, np.random.default_rng(SEED)), 3))
            e["B"] = test_b_lowo(wv, pos, neg, np.random.default_rng(SEED))
            if nm in preds:
                e["C"] = test_c_prereg(wv, pos, neg, preds[nm],
                                       np.random.default_rng(SEED),
                                       held=HELD.get(m))
            entry["per_model"][m] = e
        if len(dirs) >= 2:
            sizes = [(e["n_pos"], e["n_neg"]) for e in entry["per_model"].values()
                     if e.get("status") == "ok"]
            np_, nn_ = int(np.median([x for x, _ in sizes])), \
                       int(np.median([y for _, y in sizes]))
            entry["A"] = test_a_cross_model(dirs, WV, shared,
                                            np.random.default_rng(SEED), np_, nn_)
        R["probes"][nm] = entry

    # ── report ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("TEST A -- cross-model agreement  (DESCRIPTIVE, NOT A GATE)")
    print("=" * 78)
    print("  Models are not obliged to agree for a space to be meaningful: the same")
    print("  words can carry different senses in different models, which is why the")
    print("  panel separates arousal, drive, intensity and desire at all. This")
    print("  measures whether a construct is SHARED or MODEL-SPECIFIC. Both answers")
    print("  are informative; neither disqualifies a knob.")
    print("\n  z = (observed - null mean) / null SD. A verdict says only 'above")
    print("  chance'; z says by how much, and is comparable across probes.")
    print(f"\n  {'probe':<22}{'mean rho':>10}{'null p95':>10}{'z':>8}  reading")
    for nm in knobs:
        A = R["probes"][nm].get("A")
        if not A:
            print(f"  {nm:<22}  (needs >= 2 models)"); continue
        print(f"  {nm:<22}{A['mean_rho']:>+10.3f}{A['null_p95']:>10.3f}"
              f"{A.get('z',float('nan')):>8.1f}  "
              f"{'shared' if A['above_null'] else 'model-specific'}")

    print("\n" + "=" * 78)
    print("TEST B -- leave-one-word-out (held-out pole members)")
    print("=" * 78)
    print(f"  {'probe':<22}{'accuracy':>10}{'null p95':>10}{'z':>8}  clears in")
    for nm in knobs:
        bs = [e["B"] for e in R["probes"][nm]["per_model"].values()
              if e.get("status") == "ok" and e.get("B")]
        if not bs:
            print(f"  {nm:<22}  (insufficient)"); continue
        zs = [b.get("z", float("nan")) for b in bs]
        print(f"  {nm:<22}{np.mean([b['accuracy'] for b in bs]):>10.3f}"
              f"{np.mean([b['null_p95'] for b in bs]):>10.3f}"
              f"{np.nanmean(zs):>8.1f}"
              f"  {sum(b['clears_null'] for b in bs)}/{len(bs)}")

    if a.predictions:
        print("\n" + "=" * 78)
        print("TEST C -- pre-registered unseen words")
        print("=" * 78)
        for nm in knobs:
            cs = [e["C"] for e in R["probes"][nm]["per_model"].values()
                  if e.get("C") and e["C"].get("status") == "ok"]
            if not cs:
                miss = next((e["C"]["missing"] for e in R["probes"][nm]["per_model"].values()
                             if e.get("C")), None)
                print(f"  {nm:<22}  no usable predictions"
                      + (f"; {len(miss)} words not in vectors" if miss else ""))
                continue
            ho = all(c.get("held_out") for c in cs)
            print(f"  {nm:<22} AUC {np.mean([c['auc'] for c in cs]):.3f}  "
                  f"null {np.mean([c['null_p95'] for c in cs]):.3f}  "
                  f"z {np.nanmean([c.get('z',float('nan')) for c in cs]):>5.1f}  "
                  f"{sum(c['clears_null'] for c in cs)}/{len(cs)}  "
                  f"{'held-out' if ho else 'CORPUS WORDS — weaker'}")

    if a.heldout_prompts and preds:
        print("\n" + "=" * 78)
        print("LEAKAGE -- do held-out prompts name the construct they are scored on?")
        print("=" * 78)
        lk = leakage_report(a.heldout_prompts, probes, preds)
        R["leakage"] = lk
        pct = lk["n_same_side"] / max(1, lk["n_prompts"])
        print(f"  prompts checked                      {lk['n_prompts']}")
        print(f"  SAME probe, SAME side (biasing)      {lk['n_same_side']}  ({pct:.1%})")
        print(f"  other-probe or opposite-side (inert) {lk['n_other']}")
        if lk["per_probe"]:
            print("\n  biasing leaks per probe:")
            for pb, n in sorted(lk["per_probe"].items(), key=lambda x: -x[1]):
                print(f"    {pb:<26}{n:>4}")
            print("\n  examples:")
            for e in lk["examples"][:6]:
                print(f"    [{e['word']}] leaked '{e['leaked']}' ({e['probe']}/{e['side']})")
                print(f"        {e['text']!r}")
        if pct > 0.05:
            print(f"\n  !! {pct:.1%} of prompts leak on the scored axis. Regenerate the")
            print("     affected words with the probe vocabulary excluded, or report")
            print("     TEST C both with and without them.")
        else:
            print(f"\n  {pct:.1%} biasing leakage. Each word has 8 prompts, so one leak")
            print("  moves 1/8 of its mean vector. Report the figure; do not ignore it.")

    print("\n" + "=" * 78)
    print("SB -- REPORTED, NOT A GATE")
    print("=" * 78)
    print("  SB is a split-half reliability of a fitted direction: the same family")
    print("  of self-consistency statistic shown to track the eigengap rather than")
    print("  signal. Shown for comparison with the tests above.")
    print(f"\n  {'probe':<22}{'mean SB':>9}{'min':>7}{'max':>7}{'>=0.70 in':>11}")
    for nm in knobs:
        ss = [e["sb"] for e in R["probes"][nm]["per_model"].values()
              if e.get("status") == "ok"]
        if not ss:
            continue
        print(f"  {nm:<22}{np.mean(ss):>9.3f}{min(ss):>7.3f}{max(ss):>7.3f}"
              f"{sum(1 for x in ss if x >= 0.70):>7}/{len(ss)}")

    # ── optional filter ───────────────────────────────────────────────
    if a.require:
        print("\n" + "=" * 78)
        print(f"FILTER -- requiring {', '.join(a.require)}")
        print("=" * 78)
        keep = []
        for nm in knobs:
            e = R["probes"][nm]
            ok = True
            if "A" in a.require:
                ok &= bool(e.get("A") and e["A"]["above_null"])
            if "B" in a.require:
                bs = [x["B"] for x in e["per_model"].values()
                      if x.get("status") == "ok" and x.get("B")]
                ok &= bool(bs) and all(b["clears_null"] for b in bs)
            if "C" in a.require:
                cs = [x["C"] for x in e["per_model"].values()
                      if x.get("C") and x["C"].get("status") == "ok"]
                ok &= bool(cs) and all(c["clears_null"] for c in cs)
            print(f"  {nm:<22}{'KEEP' if ok else 'drop'}")
            if ok:
                keep.append(nm)
        R["kept"] = keep
        print(f"\n  {len(keep)} of {len(knobs)} knobs survive: {keep}")
        for path, L in zip(a.vecs, a.layers):
            nm = os.path.basename(path).replace("_vecs.npz", "")
            wv = WV[nm]
            rows, kn = [], []
            for k in keep:
                pos, neg = probe_words(probes[k], wv)
                if len(pos) >= 4 and len(neg) >= 4:
                    rows.append(build(wv, pos, neg)); kn.append(k)
            if rows:
                np.save(f"panel_validated_{nm}.npy", np.vstack(rows))
                json.dump({"names": kn, "gate": a.require},
                          open(f"panelnames_validated_{nm}.json", "w"), indent=1)
                print(f"  -> panel_validated_{nm}.npy ({len(kn)} knobs)")

    json.dump(R, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
