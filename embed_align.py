#!/usr/bin/env python3
"""
embed_align.py — TEST E: did steering move the text toward THIS construct?

  python embed_align.py --generations gens.jsonl --probes canonical_probes.json \
      [--embedder sentence-transformers/all-mpnet-base-v2] \
      [--conditions raw dedup coherent] --out embed_align.json

WHAT IT MEASURES

  For knob K, embed the high-steered and low-steered generations with an
  embedder that is NOT one of the models under test, and take

      D_text  = mean(embed(high)) - mean(embed(low))
      D_probe = mean(embed(K.pos words)) - mean(embed(K.neg words))
      score   = cos(D_text, D_probe)

  No judge, no candidate list, no labels. Deterministic, so rerunning gives the
  same answer.

WHY A RANDOM CONTROL STILL WORKS HERE

  Steering along a random direction also produces a consistent D_text — text
  changes, and it changes the same way every time. What a random direction
  cannot do is produce a D_text that aligns with ONE SPECIFIC probe's pole
  contrast. That is the discriminator. The null is therefore built from random
  directions of matched norm, run through the same pipeline, not from shuffling
  the text.

  The off-diagonal is free and worth as much as the diagonal: cos(D_text of
  knob K, D_probe of knob J) for every J gives a non-discrete confusion matrix.
  If steering `heat` aligns with valence's probe contrast as strongly as with
  heat's own, that is visible here and invisible to a naming judge.

DEGENERATE OUTPUT — the two cases, which are not the same

  LOOPS ("rage rage rage rage")
    A mean-pooled embedder weights repetition. A looped generation embeds close
    to the repeated token, so D_text drifts toward "how often did the steered
    word repeat" rather than "did the meaning shift". This inflates the score
    for lexical injection specifically — the same confound the masked arm of
    TEST D was built to catch. It is a real threat and `dedup` addresses it by
    collapsing consecutive repeated n-grams to one instance.

  COLLAPSE ("\\n!\\n!\\n!")
    Mostly self-correcting. Punctuation degeneration carries little semantic
    content, so it embeds far from any probe pole contrast and pushes the
    cosine toward zero. It attenuates rather than inflates. It is still worth
    excluding, because a cell that produced only collapse tested nothing.

  The script therefore runs every condition you ask for and reports them side
  by side. Do not pick one afterwards — decide which is primary before looking,
  and report the others.

DEGENERACY STATS COME FREE
  TEST D's stated limitation was that no per-cell coherence screen was applied.
  This script computes one: repetition rate, unique-token ratio and
  punctuation share per cell. Use it to check whether alignment tracks
  coherence, which TEST D could not.

INPUT FORMAT — factorial cells, as produced by the TEST D harness
  JSONL, one CELL per line:
      {"trial_id": "R0000", "arm": "real"|"control"|"positive",
       "triple": "['body_mind','heat','valence']",
       "levels": "{'body_mind': -1, 'heat': -1, 'valence': +1}",
       "generations": "[...8 strings...]",
       "cell_id": "0", "stem": "...", "alphas": "{...}"}
  Nested fields may be JSON or Python-literal strings; both are parsed.

  A knob's high set is every generation from cells where that knob is +1; its
  low set is every generation where it is -1. Because the design is factorial
  and balanced, the other knobs take both levels equally often in each set, so
  they average out — this is what lets a per-knob contrast be read off a
  multi-knob design without re-running anything.
"""
import argparse, glob as _glob, json, os, re, sys
import sys
from collections import defaultdict, Counter
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


VERSION = "1.0-final"
SEED = 0



# ───────────────────────────────────────────────── input adaptation
def _lit(v):
    """Nested fields arrive as JSON or as Python literals depending on writer."""
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except Exception:
        import ast
        return ast.literal_eval(v)


def load_cells(path, model_name):
    """Factorial cells -> flat per-generation records with a knob and a level.

    Each cell contributes its generations once per knob in its triple, tagged
    high/low by that knob's level in the cell. The factorial balance is what
    makes this valid: for knob K's high set, every other knob is +1 in half the
    cells and -1 in the other half, so their contributions cancel in the mean.
    """
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        gens = _lit(r.get("generations", []))
        levels = _lit(r.get("levels", {}))
        if isinstance(gens, str):
            gens = [gens]
        for knob, lv in levels.items():
            if lv == 0:
                continue
            for g in gens:
                out.append(dict(model=model_name, knob=knob,
                                level="high" if lv > 0 else "low",
                                text=g, arm=r.get("arm", "real"),
                                cell=r.get("cell_uid", ""),
                                trial=r.get("trial_id", "")))
    return out

# ───────────────────────────────────────────────── degeneracy
def degeneracy(text):
    toks = re.findall(r"\S+", text)
    if not toks:
        return dict(n_tokens=0, uniq_ratio=0.0, punct_share=1.0,
                    max_run=0, looped=True, collapsed=True)
    uniq = len(set(t.lower() for t in toks)) / len(toks)
    punct = sum(1 for t in toks if not re.search(r"[A-Za-z0-9]", t)) / len(toks)
    run = best = 1
    for i in range(1, len(toks)):
        run = run + 1 if toks[i].lower() == toks[i - 1].lower() else 1
        best = max(best, run)
    # bigram repetition, catches "a b a b a b"
    low = [t.lower() for t in toks]
    bg = Counter(zip(low, low[1:]))
    top_bg = max(bg.values()) if bg else 0
    # longest run of a repeating cycle of period 1..4, which catches
    # "the storm the storm the storm" that a token-run counter misses
    cyc = 1
    for per in range(1, 5):
        run = 1
        for i in range(per, len(low)):
            run = run + 1 if low[i] == low[i - per] else 1
            cyc = max(cyc, run // per if per > 1 else run)
    return dict(n_tokens=len(toks), uniq_ratio=uniq, punct_share=punct,
                max_run=best, top_bigram=top_bg, max_cycle=cyc,
                looped=bool(best >= 4 or top_bg >= 3 or cyc >= 3 or uniq < 0.50),
                collapsed=bool(punct > 0.5 or len(toks) < 5))


def dedup(text, max_period=4):
    """Collapse immediately-repeating cycles of period 1..max_period to one copy.

    'rage rage rage'                -> 'rage'
    'the storm the storm the storm came' -> 'the storm came'
    Natural repetition ("furious ... furious") survives: it is not an adjacent
    cycle. Preserves 'the phrase occurred' while removing 'forty times', which
    is the part that hijacks a mean-pooled embedding.
    """
    toks = re.findall(r"\S+", text)
    n = len(toks)
    if n < 2:
        return text
    low = [t.lower() for t in toks]
    out, i = [], 0
    while i < n:
        best_per, best_reps = 1, 1
        for per in range(1, max_period + 1):
            if i + per > n:
                break
            reps = 1
            while i + per * (reps + 1) <= n and \
                    low[i + per * reps: i + per * (reps + 1)] == low[i: i + per]:
                reps += 1
            if reps > best_reps:
                best_per, best_reps = per, reps
        out.extend(toks[i: i + best_per])
        i += best_per * best_reps
    return " ".join(out)


# ───────────────────────────────────────────────── embedding
def get_embedder(name):
    """A sentence embedder that is NOT one of the models under test.

    `--embedder hashing` is a dependency-free fallback for smoke-testing the
    pipeline: character n-gram hashing with tf-idf-ish weighting. It is a real
    lexical-similarity measure, so the plumbing can be verified, but it has no
    semantic knowledge and MUST NOT be used for the reported result — it would
    reduce the test to word overlap, which is precisely the confound we are
    trying to exclude."""
    if name == "hashing":
        print("  !! --embedder hashing is for SMOKE TESTING ONLY. It has no")
        print("     semantic knowledge; results with it are not reportable.")
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

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("pip install sentence-transformers  (or use --embedder hashing "
                 "to smoke-test the pipeline only)")
    m = SentenceTransformer(name)

    def enc(texts):
        v = m.encode(list(texts), convert_to_numpy=True,
                     normalize_embeddings=False, show_progress_bar=False,
                     batch_size=64)
        return np.asarray(v, dtype=np.float64)
    return enc


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v



# ───────────────────────────────────────────────── statistics
def score_knob(E, hi_idx, lo_idx, d_probe, rng, n_boot=300, n_perm=300):
    """cos(D_text, D_probe) with a bootstrap CI and a permutation p-value.

    The permutation null shuffles the high/low labels among the SAME
    generations. It asks whether the split matters, holding the text fixed —
    a different and stricter question than "is this direction random", and it
    catches the case where every generation for a knob happens to sit somewhere
    unusual in embedding space.
    """
    d = unit(E[hi_idx].mean(0) - E[lo_idx].mean(0))
    obs = float(d @ d_probe)
    boots = []
    for _ in range(n_boot):
        a_ = rng.choice(hi_idx, len(hi_idx), replace=True)
        b_ = rng.choice(lo_idx, len(lo_idx), replace=True)
        boots.append(float(unit(E[a_].mean(0) - E[b_].mean(0)) @ d_probe))
    out = dict(cos=obs, n_high=int(len(hi_idx)), n_low=int(len(lo_idx)),
               ci95=[float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))])
    if n_perm:
        pool = np.concatenate([hi_idx, lo_idx]); nh = len(hi_idx)
        perms = []
        for _ in range(n_perm):
            s = rng.permutation(pool)
            perms.append(float(unit(E[s[:nh]].mean(0) - E[s[nh:]].mean(0)) @ d_probe))
        perms = np.array(perms)
        out["perm_p"] = float((np.sum(perms >= obs) + 1) / (n_perm + 1))
        out["perm_p95"] = float(np.percentile(perms, 95))
    return out


# ───────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", nargs="+", required=True,
                    help="one or more *_generations.jsonl (factorial-cell format)")
    ap.add_argument("--max-per-group", type=int, default=400,
                    help="cap generations per knob/level/arm; keeps embedding "
                         "time bounded on large runs")
    ap.add_argument("--probes", required=True)
    ap.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2",
                    help="MUST NOT be one of the models under test")
    ap.add_argument("--conditions", nargs="*", default=["raw", "dedup", "coherent"],
                    choices=["raw", "dedup", "coherent", "dedup_coherent"])
    ap.add_argument("--primary", default="dedup",
                    help="declare which condition is primary BEFORE looking")
    ap.add_argument("--arm", default="real",
                    help="which arm is the signal: real | positive. The d_c0 "
                         "positive-control run contains only arm='positive', "
                         "so the default of 'real' silently yields no rows.")
    ap.add_argument("--control-arm", default="control")
    ap.add_argument("--per-model", action="store_true", default=True)
    ap.add_argument("--no-per-model", dest="per_model", action="store_false")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--n-null", type=int, default=500)
    ap.add_argument("--out", default="embed_align.json")
    a = ap.parse_args()

    # Windows does not expand globs, so the shell hands us the literal
    # "dir/*_generations.jsonl". Expand here, and accept a directory too.
    paths = []
    for pat in a.generations:
        if os.path.isdir(pat):
            paths += sorted(_glob.glob(os.path.join(pat, "*_generations.jsonl")))
        elif any(c in pat for c in "*?["):
            paths += sorted(_glob.glob(pat))
        else:
            paths.append(pat)
    seen, ordered = set(), []
    for p_ in paths:
        if p_ not in seen:
            seen.add(p_); ordered.append(p_)
    if not ordered:
        sys.exit(f"no files matched: {a.generations}")
    print(f"{len(ordered)} file(s):")

    rows = []
    for gp in ordered:
        nm = os.path.basename(gp).replace("_generations.jsonl", "")
        # steer_gen writes <model>_<arm>_generations.jsonl, so the arm has to be
        # stripped from the model name. Left in place, each arm registers as a
        # separate model and the signal/control pairing silently fails within
        # every per-model scope while POOLED still works — which looks like a
        # result rather than a bug.
        for suffix in ("_raw", "_dual", "_control", "_positive"):
            if nm.endswith(suffix):
                nm = nm[: -len(suffix)]
                break
        got = load_cells(gp, nm)
        rows += got
        print(f"  {nm:<34}{len(got):>7} generations")
    probes = json.load(open(a.probes, encoding="utf-8"))

    arms = Counter(r["arm"] for r in rows)
    print(f"\narms present: {dict(arms)}")
    HAS_CTRL = arms.get(a.control_arm, 0) > 0
    if not HAS_CTRL:
        print(f"  !! NO '{a.control_arm}' ARM IN THIS INPUT.")
        print(f"     Scores are reported; DELTAS AND VERDICTS ARE NOT.")
        print(f"     The matched control is the load-bearing null: without it a")
        print(f"     raw cosine cannot be shown to exceed what steering along an")
        print(f"     arbitrary direction already produces. Treat this run as")
        print(f"     replication-of-ranking only.")
    if arms.get(a.arm, 0) == 0:
        sys.exit(f"no generations with arm='{a.arm}'. Present: {list(arms)}. "
                 f"For the positive-control run pass --arm positive.")

    # cap per group so runtime stays predictable
    rng0 = np.random.default_rng(0)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["knob"], r["level"], r["arm"])].append(r)
    capped = []
    for k, v in grouped.items():
        capped += (list(rng0.choice(v, a.max_per_group, replace=False))
                   if len(v) > a.max_per_group else v)
    rows = capped
    print(f"{len(rows)} generations after cap of {a.max_per_group}/group "
          f"| primary condition: {a.primary}")
    for r in rows:
        r["_deg"] = degeneracy(r["text"])

    # ── degeneracy report, the coherence screen TEST D lacked ──────────
    print("\n" + "=" * 76)
    print("DEGENERACY  (the per-cell coherence screen TEST D did not apply)")
    print("=" * 76)
    by = defaultdict(list)
    for r in rows:
        by[(r.get("model", "?"), r.get("knob", "?"), r.get("level", "?"))].append(r["_deg"])
    print(f"  {'model':<34}{'loop':>8}{'collapse':>10}{'clean':>8}")
    degstats = {}
    for m in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == m and r["arm"] == a.arm]
        if not sub:
            continue
        dl = float(np.mean([r["_deg"]["looped"] for r in sub]))
        dc = float(np.mean([r["_deg"]["collapsed"] for r in sub]))
        degstats[m] = dict(loop=dl, collapse=dc,
                           clean=float(np.mean([not r["_deg"]["looped"] and
                                                not r["_deg"]["collapsed"]
                                                for r in sub])), n=len(sub))
        print(f"  {m:<34}{dl:>7.0%}{dc:>10.0%}{degstats[m]['clean']:>8.0%}")
    print()
    lo = sum(1 for r in rows if r["_deg"]["looped"])
    co = sum(1 for r in rows if r["_deg"]["collapsed"])
    print(f"  looped     {lo:>5} / {len(rows)}  ({lo/len(rows):.1%})   inflates lexical injection")
    print(f"  collapsed  {co:>5} / {len(rows)}  ({co/len(rows):.1%})   attenuates toward zero")
    print(f"  clean      {sum(1 for r in rows if not r['_deg']['looped'] and not r['_deg']['collapsed']):>5} / {len(rows)}")
    worst = sorted(by.items(), key=lambda kv: -np.mean([d["looped"] for d in kv[1]]))[:6]
    if worst and np.mean([d["looped"] for d in worst[0][1]]) > 0:
        print("\n  worst cells by loop rate:")
        for k, ds in worst:
            lr = np.mean([d["looped"] for d in ds])
            if lr > 0:
                print(f"    {'/'.join(map(str,k)):<48}{lr:.0%}")

    enc = get_embedder(a.embedder)
    knobs = sorted({r["knob"] for r in rows if r.get("knob") in probes})
    print(f"\nembedding with {a.embedder}")

    # probe pole contrasts, embedded once
    D_probe = {}
    for k in knobs:
        pos, neg = probes[k].get("pos", []), probes[k].get("neg", [])
        if len(pos) < 2 or len(neg) < 2:
            continue
        D_probe[k] = unit(enc(pos).mean(0) - enc(neg).mean(0))

    R = dict(version=VERSION, embedder=a.embedder, seed=SEED, primary=a.primary,
             arm=a.arm, has_control=HAS_CTRL, files=ordered, arms=dict(arms),
             degeneracy=degstats, n_after_cap=len(rows), conditions={})

    for cond in a.conditions:
        sel = []
        for r in rows:
            if cond in ("coherent", "dedup_coherent") and \
               (r["_deg"]["looped"] or r["_deg"]["collapsed"]):
                continue
            t = dedup(r["text"]) if cond in ("dedup", "dedup_coherent") else r["text"]
            sel.append({**r, "_t": t})
        print(f"\n{'=' * 78}\nCONDITION {cond}   (n={len(sel)})"
              f"{'   <-- PRIMARY' if cond == a.primary else ''}\n{'=' * 78}", flush=True)

        uniq = sorted({r["_t"] for r in sel})
        pos = {t: i for i, t in enumerate(uniq)}
        print(f"  embedding {len(uniq)} unique texts...", flush=True)
        E = enc(uniq)

        models = sorted({r["model"] for r in sel})
        scopes = [("POOLED", None)] + ([(m, m) for m in models] if a.per_model else [])
        cond_out = {}

        for label, mf in scopes:
            sub = [r for r in sel if mf is None or r["model"] == mf]
            res, off = {}, {}
            for k in knobs:
                def pick(level, arm):
                    return np.array([pos[r["_t"]] for r in sub
                                     if r["knob"] == k and r["level"] == level
                                     and r["arm"] == arm], dtype=int)
                hi, lo_ = pick("high", a.arm), pick("low", a.arm)
                if len(hi) < 3 or len(lo_) < 3:
                    continue
                sc = score_knob(E, hi, lo_, D_probe[k],
                                np.random.default_rng(SEED), a.n_boot, a.n_perm)
                dt = unit(E[hi].mean(0) - E[lo_].mean(0))
                off[k] = {j: float(dt @ D_probe[j]) for j in knobs}
                if HAS_CTRL:
                    ch, cl = pick("high", a.control_arm), pick("low", a.control_arm)
                    if len(ch) >= 3 and len(cl) >= 3:
                        cs = score_knob(E, ch, cl, D_probe[k],
                                        np.random.default_rng(SEED),
                                        max(50, a.n_boot // 2), 0)
                        sc["control_cos"] = cs["cos"]
                        sc["control_ci95"] = cs["ci95"]
                        sc["delta"] = sc["cos"] - cs["cos"]
                        sc["separated"] = bool(sc["ci95"][0] > cs["ci95"][1])
                res[k] = sc
            cond_out[label] = dict(knobs=res, offdiag=off)

        # ---- pooled report
        P = cond_out["POOLED"]
        res, off = P["knobs"], P["offdiag"]
        hdr = f"  {'knob':<22}{'cos':>8}{'bootstrap 95% CI':>20}{'perm p':>9}"
        if HAS_CTRL:
            hdr += f"{'control':>9}{'delta':>8}{'CIs sep':>9}"
        print(hdr)
        for k in sorted(res, key=lambda x: -res[x].get("delta", res[x]["cos"])):
            sc = res[k]
            line = (f"  {k:<22}{sc['cos']:>+8.3f}"
                    f"   [{sc['ci95'][0]:+.3f}, {sc['ci95'][1]:+.3f}]"
                    f"{sc.get('perm_p', float('nan')):>9.4f}")
            if HAS_CTRL and "delta" in sc:
                line += (f"{sc['control_cos']:>+9.3f}{sc['delta']:>+8.3f}"
                         f"{('yes' if sc['separated'] else 'NO'):>9}")
            print(line)
        if not HAS_CTRL:
            print("\n  delta and separation omitted: no control arm in this input.")

        if off:
            ks = [k for k in knobs if k in off]
            print(f"\n  confusion -- rows steered, cols probe contrast")
            print(f"  {'':<22}" + "".join(f"{j[:8]:>9}" for j in ks))
            dm = 0
            for k in ks:
                row = "".join(f"{off[k].get(j, 0):>+9.2f}" for j in ks)
                best = max(off[k], key=off[k].get)
                dm += best == k
                print(f"  {k:<22}{row}   "
                      f"{'diag' if best == k else 'MAX=' + best[:10]}")
            M = np.array([[off[i].get(j, 0) for j in ks] for i in ks])
            n = len(ks)
            rp = np.random.default_rng(SEED)
            pd_ = float(sum(1 for _ in range(2000)
                            if sum(int(np.argmax(rp.permutation(M[i])) == i)
                                   for i in range(n)) >= dm) / 2000)
            print(f"\n  diagonal is the maximum for {dm}/{n}"
                  f"   (permutation p = {pd_:.4f})")
            print(f"  mean diagonal {np.diag(M).mean():+.3f} | "
                  f"mean off-diagonal {(M.sum() - np.trace(M)) / (M.size - n):+.3f}")
            P["diag_max"] = int(dm); P["diag_perm_p"] = pd_

        # ---- per-model consistency: the question pooling hides
        if a.per_model and len(models) > 1:
            key = "delta" if HAS_CTRL else "cos"
            print(f"\n  PER-MODEL {key} -- uniform effect, or carried by a few?")
            print(f"  {'knob':<20}" + "".join(f"{m[:8]:>9}" for m in models) + "   >0")
            for k in knobs:
                vals = [cond_out[m]["knobs"].get(k, {}).get(key) for m in models]
                got = [v for v in vals if v is not None]
                if not got:
                    continue
                cells = "".join(f"{v:>+9.2f}" if v is not None else f"{'--':>9}"
                                for v in vals)
                print(f"  {k:<20}{cells}   {sum(1 for v in got if v > 0)}/{len(got)}")

        R["conditions"][cond] = cond_out

    json.dump(R, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")
    print("\nReport every condition. The gap between `raw` and `dedup` is the size")
    print("of the lexical-injection confound; the gap between `dedup` and")
    print("`coherent` is how much degenerate output was carrying the signal.")


if __name__ == "__main__":
    main()
