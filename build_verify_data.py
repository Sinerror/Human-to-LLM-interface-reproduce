#!/usr/bin/env python3
"""
build_verify_data.py — assemble the derived files the verify package needs.

  python build_verify_data.py --edges edge_*.json --vecs-dir lat_outputs \
      --labels frozen_emotionality_labels.csv --probes canonical_probes.json \
      --layer-map layer_map.json --out emotion-basis-verify/data/

Produces three files that do not exist as run outputs and have to be assembled:

  edges.json           the nine coherence ladders merged into one file, keyed by
                       model, each with its baseline, criterion and c*
  layer_selection.csv  per-layer held-out AUC for every model. The package ships
                       single-layer vectors, so without this the layer choice is
                       unauditable: a reader can see which layer was used but not
                       why. Recomputing it needs all layers, so it is produced
                       here, once, from the full vectors.
  probes.json          the word lists, annotated with which seven are used and
                       why each of the other eleven is not

Everything else in data/ is copied or subset from existing artifacts.
"""
import argparse, csv, glob, json, os, sys
from collections import defaultdict
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

USED = ["valence", "heat", "arousal", "intensity", "antagonism_peace",
        "body_mind", "social_inner_outer"]

# why each unused probe is not in the seven. Recorded so the choice is on the
# record rather than looking like selection after the fact.
UNUSED_REASON = {
    "drive": "usable axis; excluded to keep the basis well conditioned "
             "(drive-desire cos 0.585)",
    "desire": "usable axis; excluded for the same reason as drive",
    "security_alarm": "usable axis; excluded to keep the seven-axis basis at "
                      "condition 6.01",
    "power": "fewer than four corpus words on one pole",
    "body_feeling": "fewer than four corpus words on one pole",
    "ego_altruism": "fewer than four corpus words on one pole",
    "emotion_simple_complex": "not an emotional property",
    "emotion_simple_abstract": "not an emotional property",
    "passive_active": "not carried through the full validation",
    "social_person_group": "not carried through the full validation",
    "desire_greed_social": "not carried through the full validation",
}


def load_labels(p):
    out = {}
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        out[r["key"]] = (r.get("final", "").strip().upper() or r["heuristic"])
    return out


def merge_edges(paths, out):
    merged = {}
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        key = d["model"].replace("/", "_")
        merged[key] = dict(
            model=d["model"], layer=d["layer"], c_star=d["c_star"],
            baseline_pass_rate=d["baseline_pass_rate"],
            required_pass_rate=d["required_pass_rate"],
            criterion=d["criterion"],
            levels=[{k: v for k, v in L.items() if k != "samples"}
                    for L in d["levels"]])
    json.dump(merged, open(out, "w"), indent=1)
    cs = {k: v["c_star"] for k, v in merged.items()}
    print(f"  edges.json          {len(merged)} models, "
          f"c* {min(cs.values())} to {max(cs.values())} "
          f"({max(cs.values())/min(cs.values()):.1f}x spread)")
    return merged


def layer_auc(vecs, labels, max_layers=None):
    """Held-out AUC per layer: the criterion that selected the layer."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    z = np.load(vecs, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    layers = [int(x) for x in np.asarray(z["layers"])]
    lab = np.array([labels.get(k, "U") for k in keys])
    m = (lab == "EMO") | (lab == "NONEMO")
    y = (lab[m] == "EMO").astype(int)
    out = []
    for li, L in enumerate(layers):
        X = z["vecs"][m, li, :].astype(np.float64)
        if X.std() < 1e-9:
            out.append((L, float("nan")))
            continue
        aucs = []
        for seed in range(3):
            for tr, te in StratifiedKFold(5, shuffle=True,
                                          random_state=seed).split(X, y):
                mu = X[tr].mean(0)
                p = PCA(n_components=min(9, len(tr) - 1),
                        random_state=0).fit(X[tr] - mu)
                clf = LogisticRegression(max_iter=3000).fit(
                    p.transform(X[tr] - mu), y[tr])
                aucs.append(roc_auc_score(
                    y[te], clf.predict_proba(p.transform(X[te] - mu))[:, 1]))
        out.append((L, float(np.mean(aucs))))
    return out


def _write_layers(rows, out):
    with open(os.path.join(out, "layer_selection.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "layer", "heldout_auc",
                                          "selected"])
        w.writeheader()
        w.writerows(rows)
    sel = [r for r in rows if r["selected"]]
    print(f"\n  layer_selection.csv {len(rows)} rows, {len(sel)} models")
    disagree = 0
    for r in sel:
        best = max((x for x in rows if x["model"] == r["model"]),
                   key=lambda x: x["heldout_auc"])
        ok = best["layer"] == r["layer"]
        disagree += not ok
        flag = "" if ok else f"   !! argmax is layer {best['layer']} " \
                             f"at {best['heldout_auc']}"
        print(f"    {r['model']:<32}layer {r['layer']:<4}"
              f"AUC {r['heldout_auc']}{flag}")
    if disagree:
        print(f"\n  {disagree} model(s) where the selected layer is not the "
              f"argmax of this table.")
        print("  Either the table was computed differently from the selection,")
        print("  or the selection was not strict argmax. Resolve before shipping:")
        print("  a reader will check exactly this.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", nargs="+", required=True)
    ap.add_argument("--vecs-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--layer-map", required=True)
    ap.add_argument("--from-reproduce", default=None,
                    help="reproduce_results.json. PREFERRED: takes the per-layer "
                         "AUC from the run that actually chose the layers, "
                         "rather than recomputing it. Recomputation drifts if "
                         "the label file or the estimator differs by even a "
                         "little, and then the shipped table disagrees with the "
                         "decision it is meant to document.")
    ap.add_argument("--skip-auc", action="store_true",
                    help="skip layer_selection.csv; it is the slow part")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    labels = load_labels(a.labels)
    lm = json.load(open(a.layer_map, encoding="utf-8"))
    print(f"building into {a.out}\n")

    paths = []
    for pat in a.edges:
        paths += sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
    merge_edges(paths, os.path.join(a.out, "edges.json"))

    pr = json.load(open(a.probes, encoding="utf-8"))
    ann = {}
    for name, spec in pr.items():
        ann[name] = dict(pos=spec.get("pos", []), neg=spec.get("neg", []),
                         used=name in USED)
        if name not in USED:
            ann[name]["excluded_because"] = UNUSED_REASON.get(
                name, "not carried through the full validation")
    json.dump(ann, open(os.path.join(a.out, "probes.json"), "w"), indent=1)
    print(f"  probes.json         {len(ann)} lists, {sum(1 for v in ann.values() if v['used'])} used")

    if a.skip_auc:
        print("  layer_selection.csv skipped (--skip-auc)")
        return

    rows = []
    if a.from_reproduce:
        R = json.load(open(a.from_reproduce, encoding="utf-8"))["models"]
        for model, layer in lm.items():
            if model not in R:
                print(f"  !! {model} absent from {a.from_reproduce}")
                continue
            for r in R[model]["layers"]:
                rows.append(dict(model=model, layer=r["layer"],
                                 heldout_auc=round(r["heldout_auc"], 4),
                                 selected=int(r["layer"] == layer)))
        print(f"  layer AUC taken from {os.path.basename(a.from_reproduce)}")
        _write_layers(rows, a.out)
        return

    print("  !! recomputing layer AUC. This must use the SAME label file and")
    print("     estimator as the run that selected the layers, or the table")
    print("     will contradict the choice it documents. --from-reproduce is")
    print("     the safer path.")
    for model, layer in lm.items():
        vp = os.path.join(a.vecs_dir, f"{model}_vecs.npz")
        if not os.path.exists(vp):
            print(f"  !! {vp} absent; skipped")
            continue
        print(f"  layer AUC  {model} ...", flush=True)
        for L, auc in layer_auc(vp, labels):
            rows.append(dict(model=model, layer=L, heldout_auc=round(auc, 4),
                             selected=int(L == layer)))
    _write_layers(rows, a.out)


if __name__ == "__main__":
    main()
