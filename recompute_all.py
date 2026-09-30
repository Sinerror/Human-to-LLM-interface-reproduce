#!/usr/bin/env python3
"""
recompute_all.py — rebuild every number and figure from the primary data.

  1. Edit the PRIMARY block below once: where the nine generation folders,
     the vectors, the labels and the probes are.
  2. python recompute_all.py --code C:\\path\\to\\scripts --out C:\\path\\to\\new_empty_folder
  3. If interrupted:  same command plus --resume

About an hour on a GPU; the second encoder is the longest step. Each step
prints its own time.

WHAT IS TREATED AS TRUE
  Only primary data: the steered generations and the extracted vectors, plus the
  word lists and labels that define the axes. Every derived file — every JSON
  the figures and the paper's numbers come from — is recomputed here, into an
  empty folder, so nothing from an earlier run can be picked up. The primary
  data is checked for structure only (each folder carries its model's c*, every
  arm is present, counts are complete), never against a derived number, so the
  check cannot be circular.

WHAT YOU GET IN --out
  every derived JSON, the figures and tables, summary.json with the headline
  numbers beside what the paper currently prints, and provenance.json with the
  sha256 of every input, script and output.
"""
import argparse, shutil, hashlib, json, os, subprocess, sys, time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

# ============================================================== PRIMARY — edit once
# Point each model at the folder holding the generations the paper uses.
PRIMARY = {
    "generations": {
        "google/gemma-3-1b-it":        r"C:\Users\1\Desktop\test_D\final solution\d2_g31bit",
        "google/gemma-3-1b-pt":        r"C:\Users\1\Desktop\test_D\final solution\d2_g31bpt",
        "Qwen/Qwen2.5-1.5B":           r"C:\Users\1\Desktop\test_D\final solution\d2_qwen",
        "microsoft/phi-2":             r"C:\Users\1\Desktop\test_D\final solution\d2_phi2",
        "state-spaces/mamba-2.8b-hf":  r"C:\Users\1\Desktop\test_D\final solution\d2_mamba",
        "EleutherAI/pythia-6.9b":      r"C:\Users\1\Desktop\test_D\final solution\d2_pythia",
        "mistralai/Mistral-7B-v0.3":   r"C:\Users\1\Desktop\test_D\final solution\d2_mistral",
        "meta-llama/Meta-Llama-3-8B":  r"C:\Users\1\Desktop\test_D\final solution\d2_llama",
        "unsloth/gemma-2-9b-bnb-4bit": r"C:\Users\1\Desktop\test_D\final solution\d2_gemma2",
    },
    # optional: single-dial runs (steer_gen --design single) and the archived
    # common-magnitude runs (c = 0.08), model -> folder; empty skips those steps
    "single": {},
    "common": {},
    "vecs_dir": r"C:\Users\1\Desktop\test_D\final solution\lat_outputs",
    "labels":   r"C:\Users\1\Desktop\test_D\final solution\frozen_emotionality_labels.csv",
    "probes":   r"C:\Users\1\Desktop\test_D\final solution\canonical_probes.json",
}
# ====================================================================================

PAPER_LAYERS = {
    "google/gemma-3-1b-it": 17, "google/gemma-3-1b-pt": 25, "Qwen/Qwen2.5-1.5B": 25,
    "microsoft/phi-2": 32, "state-spaces/mamba-2.8b-hf": 52,
    "EleutherAI/pythia-6.9b": 28, "mistralai/Mistral-7B-v0.3": 32,
    "meta-llama/Meta-Llama-3-8B": 18, "unsloth/gemma-2-9b-bnb-4bit": 27,
}
PAPER_CSTAR = {
    "google/gemma-3-1b-it": 0.06, "google/gemma-3-1b-pt": 0.12, "Qwen/Qwen2.5-1.5B": 0.32,
    "microsoft/phi-2": 0.40, "state-spaces/mamba-2.8b-hf": 0.65,
    "EleutherAI/pythia-6.9b": 0.25, "mistralai/Mistral-7B-v0.3": 0.40,
    "meta-llama/Meta-Llama-3-8B": 0.32, "unsloth/gemma-2-9b-bnb-4bit": 0.50,
}
AXES = ["valence", "heat", "arousal", "intensity", "antagonism_peace",
        "body_mind", "social_inner_outer"]
SUPERSEDED_MARKER = "self-contempt::fallback"
MPNET = "sentence-transformers/all-mpnet-base-v2"
BGE = "BAAI/bge-large-en-v1.5"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def slug(m):
    return m.replace("/", "_")


# ------------------------------------------------------------------ primary checks
def check_primary():
    print("PRIMARY DATA — structure only, no derived numbers involved\n")
    bad = []
    lab = PRIMARY["labels"]
    if not os.path.exists(lab):
        bad.append(f"labels not found: {lab}")
    elif SUPERSEDED_MARKER in open(lab, encoding="utf-8").read():
        bad.append(f"{lab} is the SUPERSEDED label file (contains {SUPERSEDED_MARKER})")
    else:
        print(f"  labels   ok  {lab}")
    if not os.path.exists(PRIMARY["probes"]):
        bad.append(f"probes not found: {PRIMARY['probes']}")

    import numpy as np
    for m, layer in PAPER_LAYERS.items():
        vp = os.path.join(PRIMARY["vecs_dir"], f"{slug(m)}_vecs.npz")
        if not os.path.exists(vp):
            bad.append(f"vectors missing: {vp}")
            continue
        z = np.load(vp, allow_pickle=False)
        if layer not in [int(x) for x in np.asarray(z["layers"])]:
            bad.append(f"{vp} lacks layer {layer}")

    print(f"\n  {'model':<30}{'c* manifest':>12}{'paper':>7}   arms and generations")
    for m, d in PRIMARY["generations"].items():
        if not os.path.isdir(d):
            bad.append(f"generation folder missing: {d}")
            continue
        mf = os.path.join(d, "manifest.json")
        c = json.load(open(mf, encoding="utf-8")).get("c") if os.path.exists(mf) else None
        counts = {}
        for fn in os.listdir(d):
            if fn.endswith("_generations.jsonl"):
                for line in open(os.path.join(d, fn), encoding="utf-8"):
                    r = json.loads(line)
                    g = r["generations"]
                    g = json.loads(g) if isinstance(g, str) else g
                    counts[r.get("arm", "?")] = counts.get(r.get("arm", "?"), 0) + len(g)
        flag = ""
        if c is None:
            flag = "  !! no manifest c"
        elif abs(float(c) - PAPER_CSTAR[m]) > 1e-9:
            flag = f"  !! generated at c={c}, paper c*={PAPER_CSTAR[m]}"
            bad.append(f"{d}: generated at c={c}, not the model's c* {PAPER_CSTAR[m]}")
        missing = {"raw", "dual", "control"} - set(counts)
        if missing:
            bad.append(f"{d}: arms missing {sorted(missing)}")
        print(f"  {m[:30]:<30}{str(c):>12}{PAPER_CSTAR[m]:>7}   "
              + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + flag)
    if bad:
        print("\nPRIMARY DATA NOT CONSISTENT — nothing will be computed:")
        for b in bad:
            print(f"  - {b}")
        sys.exit(1)
    print("\n  primary data consistent\n")


# ------------------------------------------------------------------ steps
def step(name, cmd, out, resume, log):
    if resume and os.path.exists(out):
        print(f"  {name:<34} skip (exists)")
        return
    t0 = time.time()
    print(f"  {name:<34} running ...", flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 70}\n{' '.join(cmd)}\n{'=' * 70}\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc or not os.path.exists(out):
        sys.exit(f"  {name} FAILED (exit {rc}); see {log}. Fix, then rerun with --resume.")
    print(f"  {'':<34} done {time.time() - t0:.0f}s")


def summary(o):
    """Headline numbers from the fresh files only, beside expected.json."""
    import numpy as np
    from scipy.stats import spearmanr, ttest_1samp
    J = lambda f: json.load(open(os.path.join(o, f), encoding="utf-8"))
    S = {}
    pn = J("paper_numbers.json")
    S["gram_mean"], S["gram_max"], S["gram_condition"] = \
        pn["gram"]["mean"], pn["gram"]["max"], pn["gram"]["condition"]
    lw = pn["lowo"]
    S["lowo_min"] = min(v["accuracy"] for v in lw.values())

    def per_model(E):
        C = E["conditions"]["dedup"]
        mods = [k for k in C if k != "POOLED"]
        return {m: {k: C[m]["knobs"][k]["delta"] for k in C[m]["knobs"]
                    if "delta" in C[m]["knobs"][k]} for m in mods}
    A, B = per_model(J("ea2_mpnet.json")), per_model(J("ea2_bge.json"))
    ax = sorted({k for d in A.values() for k in d})
    S["gain_by_axis"] = {k: float(np.mean([A[m][k] for m in A if k in A[m]])) for k in ax}
    pmm = [np.mean(list(A[m].values())) for m in A]
    S["gain_model_mean"] = float(np.mean(pmm))
    S["gain_t"] = float(ttest_1samp(pmm, 0).statistic)
    S["cells_positive"] = sum(v > 0 for d in A.values() for v in d.values())
    S["bge_model_mean"] = float(np.mean([np.mean(list(B[m].values())) for m in B]))
    bmm = [np.mean(list(B[m].values())) for m in B]
    S["bge_t"] = float(ttest_1samp(bmm, 0).statistic)
    S["bge_cells_positive"] = sum(v > 0 for d in B.values() for v in d.values())
    S["encoder_sign_agreement"] = sum((A[m][k] > 0) == (B[m][k] > 0)
                                      for m in A if m in B for k in A[m] if k in B[m])
    od = J("ea2_bge.json")["conditions"]["dedup"]["POOLED"].get("offdiag")
    if od:
        ks = [k for k in od]
        Mb = np.array([[od[i].get(j, np.nan) for j in ks] for i in ks])
        S["bge_crossalign_diag_max"] = int(sum(np.nanargmax(Mb[i]) == i for i in range(len(ks))))
    bax = {k: np.mean([B[m][k] for m in B if k in B[m]]) for k in ax}
    S["encoder_rank_rho"] = float(spearmanr([S["gain_by_axis"][k] for k in ax],
                                            [bax[k] for k in ax]).statistic)
    bo = J("crossalign_boot.json")
    M = np.array(bo["matrix"])
    S["crossalign_diag_max"] = int(sum(np.argmax(M[i]) == i for i in range(len(M))))
    S["crossalign_mean_diag"], S["crossalign_mean_off"] = \
        bo["mean_diagonal"], bo["mean_offdiagonal"]

    def dom(v):
        a = b = 0
        for p in v["per_triple"]:
            X = np.array(p["M"]); Xn = X / (np.abs(X).max(1, keepdims=True) + 1e-12)
            a += sum(int(np.argmax(np.abs(Xn[i])) == i) for i in range(3)); b += 3
        return a / b
    r, d = J("t2_raw.json"), J("t2_dual.json")
    S["diag_dominant_per_repetition"] = float(np.mean([dom(v) for v in r.values()]))
    tc = J("transfer_cstar.json")["models"]
    S["diag_dominant_pooled"] = float(np.mean([v["diag_is_max"] / v["diag_rows"]
                                               for v in tc.values()]))

    def parts(Ms):
        iu = np.triu_indices(3, 1)
        return (np.mean([np.abs(np.diag(X)).mean() for X in Ms]),
                np.mean([np.abs(((X + X.T) / 2)[iu]).mean() for X in Ms]),
                np.mean([np.abs(((X - X.T) / 2)[iu]).mean() for X in Ms]))
    ch = []
    for k in r:
        dk = k.replace("_raw", "_dual")
        if dk in d:
            a_ = parts([np.array(p["M"]) for p in r[k]["per_triple"]])
            b_ = parts([np.array(p["M"]) for p in d[dk]["per_triple"]])
            ch.append([b_[i] / a_[i] - 1 for i in range(3)])
    if ch:
        ch = np.array(ch)
        S["dual_diag_change"], S["dual_sym_change"], S["dual_antisym_change"] = \
            map(float, ch.mean(0))
    mag = {}
    for v in tc.values():
        for k, mm in v.get("magnitude", {}).items():
            mag.setdefault(k, []).append(mm["pole_fraction"])
    S["pole_fraction"] = {k: float(np.mean(v)) for k, v in mag.items()}
    lbp = os.path.join(o, "linear_baseline.json")
    if os.path.exists(lbp):
        lb = J("linear_baseline.json")["models"].values()
        pick = lambda lab, key: float(np.mean([m[lab][key] for m in lb]))
        S["linear_pred_separability"] = pick("+harness +readout", "sep")
        S["linear_pred_sym_change"] = pick("+harness +readout", "sym")
        S["linear_pred_antisym_share"] = pick("+harness +readout", "raw_anti_share")
        S["observed_antisym_share"] = pick("OBSERVED", "raw_anti_share")
        S["separability_below_linear"] = sum(m["OBSERVED"]["sep"] < m["+harness +readout"]["sep"] for m in lb)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="folder with the analysis scripts")
    ap.add_argument("--out", required=True, help="an EMPTY or new folder")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary-only", action="store_true",
                    help="recompute summary.json from the files already in --out; seconds")
    ap.add_argument("--n-boot", type=int, default=1200)
    ap.add_argument("--from-work", default=None,
                    help="a reproduce.py work tree (work/<model>/...); replaces the "
                         "PRIMARY block, and ends by building the verify package")
    ap.add_argument("--labels", default=None, help="with --from-work")
    ap.add_argument("--probes", default=None, help="with --from-work")
    ap.add_argument("--predictions", default=None, help="with --from-work: predictions.csv")
    ap.add_argument("--cats27", default=None, help="canonical_probes_27d.json (Appendix C)")
    ap.add_argument("--ekman", default=None, help="ekman.json (Appendix C)")
    a = ap.parse_args()

    o = os.path.abspath(a.out)
    if a.summary_only:
        Sm = summary(o)
        json.dump(Sm, open(os.path.join(o, "summary.json"), "w"), indent=1)
        print(json.dumps(Sm, indent=1))
        return
    if os.path.isdir(o) and os.listdir(o) and not a.resume:
        sys.exit(f"{o} is not empty. Use a new folder, or --resume to continue a run.")
    os.makedirs(o, exist_ok=True)
    if a.from_work:
        W = os.path.abspath(a.from_work)
        slugs = sorted(d for d in os.listdir(W)
                       if os.path.exists(os.path.join(W, d, f"{d}_vecs.npz")))
        if not slugs:
            sys.exit(f"{W}: no <model>/<model>_vecs.npz — is this a reproduce.py work tree?")
        vd = os.path.join(o, "vectors")
        os.makedirs(vd, exist_ok=True)
        for d in slugs:                       # one folder of vectors, as every script expects;
            src = os.path.join(W, d, f"{d}_vecs.npz")      # hard links cost no disk
            dst = os.path.join(vd, f"{d}_vecs.npz")
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
        hub = {m.replace("/", "_"): m for m in PAPER_CSTAR}   # slug -> hub id, the
        unknown = [d for d in slugs if d not in hub]            # keys the checks use
        if unknown:
            sys.exit(f"  models not in the paper's table: {unknown}")
        sub = lambda name: {hub[d]: os.path.join(W, d, name) for d in slugs
                            if os.path.isdir(os.path.join(W, d, name))}
        PRIMARY.update(generations=sub("gens"), single=sub("gens_single"),
                       common=sub("gens_common"), vecs_dir=vd,
                       labels=a.labels, probes=a.probes)
        print(f"  work tree {W}: {len(slugs)} models; single-dial runs for "
              f"{len(PRIMARY['single'])}, common-magnitude runs for {len(PRIMARY['common'])}")
    check_primary()

    code = os.path.abspath(a.code)
    py = sys.executable
    S = lambda n: os.path.join(code, n)
    O = lambda n: os.path.join(o, n)
    log = O("recompute.log")
    gens = list(PRIMARY["generations"].values())
    lm = O("layer_map.json")
    json.dump({slug(m): l for m, l in PAPER_LAYERS.items()}, open(lm, "w"), indent=1)
    base = ["--probes", PRIMARY["probes"]]
    trf = ["--labels", PRIMARY["labels"], "--vecs-dir", PRIMARY["vecs_dir"], "--layer-map", lm]
    vit = os.path.join(PRIMARY["vecs_dir"], "google_gemma-3-1b-it_vecs.npz")
    T0 = time.time()
    print("RECOMPUTING into", o)

    step("geometry and validation", [py, S("derive_paper_numbers.py"), "--vecs", vit,
         "--layer", "17", "--labels", PRIMARY["labels"], "--probes", PRIMARY["probes"],
         "--out", O("paper_numbers.json")], O("paper_numbers.json"), a.resume, log)
    step("transfer, per repetition (raw)", [py, S("transfer_independent.py"),
         "--generations", *gens, *base, *trf, "--arm", "raw",
         "--out", O("t2_raw.json")], O("t2_raw.json"), a.resume, log)
    step("transfer, per repetition (dual)", [py, S("transfer_independent.py"),
         "--generations", *gens, *base, *trf, "--arm", "dual",
         "--out", O("t2_dual.json")], O("t2_dual.json"), a.resume, log)
    step("linear baseline", [py, S("linear_baseline.py"),
         "--vecs-dir", PRIMARY["vecs_dir"], "--labels", PRIMARY["labels"],
         "--probes", PRIMARY["probes"], "--t2-raw", O("t2_raw.json"),
         "--t2-dual", O("t2_dual.json"), "--embedder", MPNET,
         "--out", O("linear_baseline.json")], O("linear_baseline.json"), a.resume, log)
    singles = list(PRIMARY.get("single", {}).values())
    if singles:
        step("superposition (section 7.3)", [py, S("superposition_test.py"),
             "--triple-gens", *gens, "--single-gens", *singles,
             "--vecs-dir", PRIMARY["vecs_dir"], "--labels", PRIMARY["labels"],
             "--probes", PRIMARY["probes"], "--out", O("superposition.json")],
             O("superposition.json"), a.resume, log)
    if a.cats27 and a.ekman:
        step("dimensionality (Appendix C)", [py, S("dimensionality.py"),
             "--vecs-dir", PRIMARY["vecs_dir"], "--labels", PRIMARY["labels"],
             "--probes", PRIMARY["probes"], "--cats27", a.cats27, "--ekman", a.ekman,
             "--out", O("dimensionality.json")], O("dimensionality.json"), a.resume, log)
    commons = list(PRIMARY.get("common", {}).values())
    if commons:
        step("common magnitude, c = 0.08 (sections 3.5, 7.2)", [py, S("transfer_independent.py"),
             "--generations", *commons, *base, *trf, "--arm", "raw",
             "--out", O("common_magnitude_transfer.json")],
             O("common_magnitude_transfer.json"), a.resume, log)
    step("transfer pooled + magnitude", [py, S("transfer_matrix.py"),
         "--generations", *gens, *base, *trf, "--arm", "raw",
         "--out", O("transfer_cstar.json")], O("transfer_cstar.json"), a.resume, log)
    step(f"cross-alignment bootstrap ({a.n_boot})", [py, S("bootstrap_crossalign.py"),
         "--generations", *gens, *base, "--embedder", MPNET, "--n-boot", str(a.n_boot),
         "--out", O("crossalign_boot.json")], O("crossalign_boot.json"), a.resume, log)
    # uncapped, so per-axis gains equal what verify.py computes
    for tag, emb in (("mpnet", MPNET), ("bge", BGE)):
        step(f"steering gains ({tag}, uncapped)", [py, S("embed_align.py"),
             "--generations", *gens, *base, "--embedder", emb,
             "--conditions", "raw", "dedup", "coherent", "--primary", "dedup",
             "--max-per-group", "100000000", "--arm", "raw",
             "--out", O(f"ea2_{tag}.json")], O(f"ea2_{tag}.json"), a.resume, log)

    # figure 3 reads per-model gains; the uncapped mpnet run is that source
    E = json.load(open(O("ea2_mpnet.json"), encoding="utf-8"))["conditions"]["dedup"]
    json.dump({"gain_per_model": {m: {k: v["delta"] for k, v in E[m]["knobs"].items()
                                      if "delta" in v} for m in E if m != "POOLED"}},
              open(O("gains_per_model.json"), "w"), indent=1)
    step("figures and tables", [py, S("make_figures.py"),
         "--embed", O("ea2_mpnet.json"), "--embed2", O("ea2_bge.json"),
         "--verified", O("gains_per_model.json"),
         "--transfer", O("transfer_cstar.json"), "--transfer-indep", O("t2_raw.json"),
         "--crossalign", O("crossalign_boot.json"),
         "--paper-numbers", O("paper_numbers.json"),
         "--vecs", vit, "--labels", PRIMARY["labels"], "--probes", PRIMARY["probes"],
         "--layer", "17", "--outdir", O("figures"),
         *(["--superposition", O("superposition.json"),
            "--linear-baseline", O("linear_baseline.json")]
           if os.path.exists(O("superposition.json")) else [])],
         O(os.path.join("figures", "fig4_simultaneous.png")), a.resume, log)

    if a.from_work:
        pkg = os.path.join(W, "verify_package")
        atlas = os.path.join(W, "google_gemma-3-1b-it", "atlas_google_gemma-3-1b-it.json")
        step("verify package (all models)", [py, S("package_verify.py"),
             "--work", W, "--models", *slugs, "--labels", PRIMARY["labels"],
             "--probes", PRIMARY["probes"], "--predictions", a.predictions or "predictions.csv",
             "--single-gens", *singles,
             *(["--atlas", atlas] if os.path.exists(atlas) else []),
             "--out", pkg], os.path.join(pkg, "data", "MANIFEST.json"), a.resume, log)
        if os.path.exists(O("common_magnitude_transfer.json")):
            shutil.copy2(O("common_magnitude_transfer.json"),
                         os.path.join(pkg, "data", "common_magnitude_transfer.json"))
        print(f"\n  verify package: {pkg}\n  next: copy verify.py and expected.json into it "
              f"and run  python verify.py --full")

    Sm = summary(o)
    json.dump(Sm, open(O("summary.json"), "w"), indent=1)
    prov = {"primary": {}, "scripts": {}, "outputs": {}}
    for m, d in PRIMARY["generations"].items():
        for fn in sorted(os.listdir(d)):
            prov["primary"][f"{m}/{fn}"] = sha(os.path.join(d, fn))
    for k in ("labels", "probes"):
        prov["primary"][k] = sha(PRIMARY[k])
    prov["primary"]["vectors_gemma-3-1b-it"] = sha(vit)
    for fn in ("derive_paper_numbers.py", "transfer_independent.py", "transfer_matrix.py",
               "linear_baseline.py",
               "bootstrap_crossalign.py", "embed_align.py", "make_figures.py"):
        prov["scripts"][fn] = sha(S(fn))
    for root, _, fs in os.walk(o):
        for fn in fs:
            if fn not in ("provenance.json", "recompute.log"):
                p = os.path.join(root, fn)
                prov["outputs"][os.path.relpath(p, o)] = sha(p)
    json.dump(prov, open(O("provenance.json"), "w"), indent=1)

    print(f"\nHEADLINE NUMBERS (fresh)  —  total {(time.time() - T0) / 60:.0f} min")
    for k, v in Sm.items():
        if isinstance(v, dict):
            print(f"  {k}")
            for kk, vv in v.items():
                print(f"      {kk:<22}{vv:.3f}")
        else:
            print(f"  {k:<32}{v:.3f}" if isinstance(v, float) else f"  {k:<32}{v}")
    print(f"\n-> {o}\\summary.json, figures\\, provenance.json")
    print("These are now the paper's numbers. Send summary.json; any figure the draft")
    print("quotes differently gets changed in the draft, not here.")


if __name__ == "__main__":
    main()
