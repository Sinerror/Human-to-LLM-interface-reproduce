#!/usr/bin/env python3
"""
package_verify.py — assemble the verify data tree from reproduce work dirs.

  python package_verify.py --work repro --models google_gemma-3-1b-it ... \
      --labels frozen_emotionality_labels.csv --probes canonical_probes.json \
      --predictions predictions.csv --out emotion-basis-verify

Turns what `reproduce.py` produced into what `verify.py` reads. The two have
different shapes on purpose: reproduce keeps everything per model and at full
size, verify ships one tree at a twenty-seventh of it.

WHAT IT DOES, AND WHY EACH STEP

  Single-layer vector subsets. A full extraction is 46 MB per model and the
  package needs 1.7. Only the selected layer is kept, which is why
  layer_selection.csv must also ship: the choice stays auditable even though it
  is no longer recomputable from what remains.

  Gzipped generations. The expensive artifact, three-fold smaller as .gz, and
  verify reads them compressed.

  Merged edges.json. Nine per-model ladders into one file.

  MANIFEST.json last, over the finished tree.

  predictions.csv is copied, never generated. It is a pre-registration.
"""
import argparse, csv, glob, gzip, hashlib, json, os, shutil, sys
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def subset_layer(src, dst, layer):
    z = np.load(src, allow_pickle=False)
    X = z["vecs"]
    if X.ndim == 3:
        layers = [int(v) for v in np.asarray(z["layers"])]
        if layer not in layers:
            return None
        X = X[:, layers.index(layer):layers.index(layer) + 1, :]
    np.savez_compressed(dst, vecs=X, keys=z["keys"], words=z["words"],
                        layers=np.array([layer]))
    return os.path.getsize(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--predictions", default="predictions.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D = os.path.join(a.out, "data")
    for sub in ("vectors_layer", "heldout_vectors", "generations"):
        os.makedirs(os.path.join(D, sub), exist_ok=True)
    print(f"packaging into {D}\n")

    layer_rows, edges, total_before, total_after = [], {}, 0, 0
    for model in a.models:
        md = os.path.join(a.work, model)
        if not os.path.isdir(md):
            print(f"  !! {md} absent; skipped")
            continue
        led = os.path.join(md, "ledger.json")
        layer = None
        if os.path.exists(led):
            L = json.load(open(led, encoding="utf-8"))
            layer = (L.get("extract", {}).get("inputs", {}) or {}).get("layer")
        if layer is None:
            print(f"  !! {model}: no layer in ledger; skipped")
            continue

        src = os.path.join(md, f"{model}_vecs.npz")
        if os.path.exists(src):
            before = os.path.getsize(src)
            after = subset_layer(src, os.path.join(D, "vectors_layer",
                                                   f"{model}.npz"), layer)
            if after:
                total_before += before
                total_after += after
                print(f"  {model:<32}vectors {before/1e6:>6.1f} -> "
                      f"{after/1e6:>5.2f} MB  (layer {layer})")

        hsrc = os.path.join(md, f"{model}_heldout_vecs.npz")
        if os.path.exists(hsrc):
            subset_layer(hsrc, os.path.join(D, "heldout_vectors",
                                            f"{model}.npz"), layer)
        else:
            print(f"  {model:<32}no held-out vectors; test C will not run")

        for g in sorted(glob.glob(os.path.join(md, "gens", "*.jsonl"))):
            dst = os.path.join(D, "generations",
                               os.path.basename(g) + ".gz")
            with open(g, "rb") as fi, gzip.open(dst, "wb", 9) as fo:
                shutil.copyfileobj(fi, fo)

        ep = os.path.join(md, "edge.json")
        if os.path.exists(ep):
            e = json.load(open(ep, encoding="utf-8"))
            edges[model] = dict(
                model=e["model"], layer=e["layer"], c_star=e["c_star"],
                baseline_pass_rate=e["baseline_pass_rate"],
                required_pass_rate=e["required_pass_rate"],
                criterion=e["criterion"],
                levels=[{k: v for k, v in L_.items() if k != "samples"}
                        for L_ in e["levels"]])

        lp = os.path.join(md, "verify_data", "layer_selection.csv")
        if os.path.exists(lp):
            layer_rows += list(csv.DictReader(open(lp, newline="",
                                                   encoding="utf-8")))

    if edges:
        json.dump(edges, open(os.path.join(D, "edges.json"), "w"), indent=1)
        cs = {k: v["c_star"] for k, v in edges.items()}
        print(f"\n  edges.json          {len(edges)} models, c* "
              f"{min(cs.values())} to {max(cs.values())}")
    if layer_rows:
        with open(os.path.join(D, "layer_selection.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(layer_rows[0]))
            w.writeheader()
            w.writerows(layer_rows)
        print(f"  layer_selection.csv {len(layer_rows)} rows")

    shutil.copy(a.labels, os.path.join(D, "labels.csv"))
    shutil.copy(a.probes, os.path.join(D, "probes.json"))
    if os.path.exists(a.predictions):
        shutil.copy(a.predictions, os.path.join(D, "predictions.csv"))
    else:
        print(f"  !! {a.predictions} absent. It is a pre-registration and "
              f"cannot be regenerated; test C will not run.")

    man = {}
    for root, _, files in os.walk(D):
        for f in files:
            if f == "MANIFEST.json":
                continue
            fp = os.path.join(root, f)
            man[os.path.relpath(fp, D).replace(os.sep, "/")] = sha(fp)
    json.dump(man, open(os.path.join(D, "MANIFEST.json"), "w"), indent=1)

    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(D) for f in fs)
    print(f"\n  MANIFEST.json       {len(man)} files")
    print(f"  package size        {size/1e6:.1f} MB")
    if total_before:
        print(f"  vector reduction    {total_before/1e6:.0f} -> "
              f"{total_after/1e6:.1f} MB  ({total_before/total_after:.0f}x)")
    print(f"\n  next: copy verify.py, expected.json, README.md, LICENCE and")
    print(f"  requirements.txt alongside data/, then run verify.py against a")
    print(f"  CLEAN CHECKOUT rather than this tree.")


if __name__ == "__main__":
    main()
