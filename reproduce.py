#!/usr/bin/env python3
"""
reproduce.py — the full chain, staged and resumable.

  python reproduce.py --model google/gemma-3-1b-it --layer 17 --work repro/
  python reproduce.py --model ... --work repro/ --status
  python reproduce.py --model ... --work repro/ --from edge
  python reproduce.py --model ... --work repro/ --only extract atlas

Runs one model end to end: extraction, coherence edge, factorial generation,
analysis, atlas. Every stage writes its output and a receipt; a rerun skips any
stage whose receipt is present and whose inputs are unchanged. Interrupt it,
run out of disk, hit an OOM on stage four — start it again and it continues.

WHY STAGED

  The chain is long and the middle is expensive. Extraction takes minutes,
  generation takes hours, and a crash in the last stage should not cost the
  first four. Each stage is also independently useful: the coherence edge is a
  result on its own, and the atlas feeds the interface without needing anything
  downstream.

  Stages record the sha256 of their inputs. If the corpus changes, every stage
  downstream of it invalidates and says so rather than silently mixing a new
  corpus with old vectors — the failure mode that cost this project a week.

STAGES

  1  extract    prompts + model         -> vectors               (GPU, minutes)
  2  heldout    heldout prompts + model -> heldout vectors       (GPU, minutes)
  3  validate   vectors + probes        -> axis validation       (CPU, seconds)
  4  edge       vectors + model         -> c*                    (GPU, ~10 min)
  5  layers     vectors + edge          -> per-layer AUC, edges  (CPU, minutes)
  6  generate   axes + model + c*       -> three arms            (GPU, hours)
  7  analyse    generations             -> alignment, transfer   (CPU, ~20 min)
  8  atlas      model + axes            -> interface atlas       (GPU, minutes)
  9  package    everything              -> verify data tree      (CPU, minutes)

  Stages 2 and 3 exist because the verify package needs artifacts the main
  chain does not produce. Test C projects pre-registered words that were never
  in the corpus, so they need their own extraction. And shipping single-layer
  vectors makes the layer choice unrecomputable, so the table that made the
  choice has to be written while all layers are still available.

  predictions.csv is not produced by any stage. It is a pre-registration,
  written by hand before extraction, and regenerating it would destroy the
  property that makes it worth reporting.
"""
import argparse, hashlib, json, os, subprocess, sys, time
import sys
from datetime import datetime, timezone

# A Windows console is often cp1251 or cp437 and cannot encode much of what
# either the models or these scripts print. Replace unencodable characters
# rather than raising, which would otherwise kill a run after its expensive
# stage had completed. JSON output uses ensure_ascii and is unaffected.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


# order matters: `layers` merges the edge file, so it cannot precede `edge`.
# An earlier ordering put it third and it silently wrote an empty edges.json.
# The layers used in the paper, each the held-out-AUC argmax for its model.
# Checked at startup: running a model at another model's layer silently measures
# a different basis. A gemma-3-1b-pt reproduction run at 17 (the -it layer)
# instead of 25 produced c* = 0.08 rather than 0.12 and was caught only after
# generation, by the per-layer table.
PAPER_LAYERS = {
    "google/gemma-3-1b-it": 17, "google/gemma-3-1b-pt": 25,
    "Qwen/Qwen2.5-1.5B": 25, "microsoft/phi-2": 32,
    "state-spaces/mamba-2.8b-hf": 52, "EleutherAI/pythia-6.9b": 28,
    "mistralai/Mistral-7B-v0.3": 32, "meta-llama/Meta-Llama-3-8B": 18,
    "unsloth/gemma-2-9b-bnb-4bit": 27,
}

STAGES = ["extract", "heldout", "validate", "edge", "layers", "generate",
          "single", "analyse", "atlas"]
# Run only when asked (--only commonmag): the archived common-magnitude
# comparison of sections 3.5 and 7.2, every model at c = 0.08, raw arm only.
OPTIONAL = ["commonmag", "package"]   # package: single-model use; recompute_all packages all nine
NEEDS_GPU = {"extract", "heldout", "edge", "generate", "single", "commonmag", "atlas"}


def sha(path, limit=None):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        read = 0
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            read += len(b)
            if limit and read >= limit:
                break
    return h.hexdigest()[:16]


def slug(model):
    return model.replace("/", "_")


class Ledger:
    """Receipts on disk. A stage is done when its receipt exists and the
    recorded input digests still match."""

    def __init__(self, path):
        self.path = path
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def save(self):
        json.dump(self.d, open(self.path, "w"), indent=1)

    def done(self, stage, inputs):
        r = self.d.get(stage)
        if not r or r.get("status") != "ok":
            return False, "not run"
        for k, v in inputs.items():
            if r["inputs"].get(k) != v:
                return False, f"input changed: {k}"
        for f in r.get("outputs", []):
            if not os.path.exists(f):
                return False, f"output missing: {os.path.basename(f)}"
        return True, r.get("finished", "")

    def record(self, stage, inputs, outputs, seconds, cmd):
        self.d[stage] = dict(status="ok", inputs=inputs, outputs=outputs,
                             seconds=round(seconds, 1), command=cmd,
                             finished=datetime.now(timezone.utc)
                             .strftime("%Y-%m-%d %H:%M UTC"))
        self.save()

    def fail(self, stage, cmd, code):
        self.d[stage] = dict(status="failed", command=cmd, exit_code=code,
                             finished=datetime.now(timezone.utc)
                             .strftime("%Y-%m-%d %H:%M UTC"))
        self.save()


def run(cmd, log_path):
    print(f"    $ {' '.join(cmd)}")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 70}\n{datetime.now(timezone.utc)}\n"
                  f"{' '.join(cmd)}\n{'=' * 70}\n")
        log.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        tail = []
        for line in p.stdout:
            log.write(line)
            tail.append(line.rstrip())
            if len(tail) > 12:
                tail.pop(0)
        p.wait()
    if p.returncode:
        print("    " + "\n    ".join(tail))
    return p.returncode


SCRIPTS = {
    "extract":  "extract_vectors.py",
    "validate": "probe_validate.py",
    "edge":     "coherence_edge.py",
    "generate": "steer_gen.py",
    "analyse":  ["transfer_independent.py", "embed_align.py"],
    "atlas":    "make_atlas.py",
}


def preflight(a, code):
    """Everything the chain needs, checked before any GPU time is spent.

    The prompt file is the boundary of reproducibility: it was produced by a
    sampling model and cannot be regenerated identically, so it ships as a
    fixed, hashed artifact. From there the chain is a forward pass with no
    sampling and is deterministic up to bf16 accumulation order, which varies
    with batch size and hardware. Expect agreement to about 1e-3 in the
    vectors and to the stated tolerances in every downstream statistic, not
    bit-identity.
    """
    print("preflight\n")
    ok = True

    print("  inputs")
    for label, path, why in (
            ("prompts", a.corpus, "frozen; the deterministic starting point"),
            ("labels", a.labels, "EMO/NONEMO per sense"),
            ("probes", a.probes, "the bipolar word lists"),
            ("ekman", a.ekman, "optional; atlas category readout")):
        exists = os.path.exists(path)
        if not exists and label == "ekman":
            print(f"    {label:<10}{'absent':<10}{path}  (optional)")
            continue
        ok &= exists
        d = sha(path) if exists else "—"
        print(f"    {label:<10}{('ok' if exists else 'MISSING'):<10}"
              f"{path}  {d}")
        if not exists:
            print(f"               needed for: {why}")

    print("\n  scripts")
    for stage, names in SCRIPTS.items():
        for n in ([names] if isinstance(names, str) else names):
            f = os.path.join(code, n)
            e = os.path.exists(f)
            ok &= e
            print(f"    {stage:<10}{('ok' if e else 'MISSING'):<10}{n}")

    print("\n  environment")
    try:
        import torch
        cu = torch.cuda.is_available()
        print(f"    {'torch':<22}{torch.__version__}  "
              f"cuda {'yes' if cu else 'NO'}")
        if cu:
            p_ = torch.cuda.get_device_properties(0)
            print(f"    {'gpu':<22}{p_.name}  {p_.total_memory/1e9:.0f} GB")
        else:
            print("               stages extract, edge, generate and atlas "
                  "need a GPU")
    except ImportError:
        print(f"    {'torch':<22}MISSING")
        ok = False
    for mod in ("transformers", "sentence_transformers", "scipy", "sklearn"):
        try:
            m = __import__(mod)
            print(f"    {mod:<22}{getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"    {mod:<22}MISSING")
            ok = False

    print("\n  " + ("ready" if ok else "NOT READY -- resolve the above first"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--work", default="repro")
    ap.add_argument("--corpus", default="frozen_prompts.jsonl")
    ap.add_argument("--heldout-prompts", default="heldout_prompts.jsonl",
                    help="prompts for the pre-registered unseen words")
    ap.add_argument("--labels", default="frozen_emotionality_labels.csv")
    ap.add_argument("--probes", default="canonical_probes.json")
    ap.add_argument("--ekman", default="ekman.json")
    ap.add_argument("--predictions", default="predictions.csv",
                    help="the pre-registration, word,probe,side")
    ap.add_argument("--code", default=".", help="directory holding the scripts")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--n-gen", type=int, default=6)
    ap.add_argument("--batch-rows", type=int, default=48)
    ap.add_argument("--c-star", type=float, default=None,
                    help="skip stage 3 and use this value")
    ap.add_argument("--only", nargs="*", choices=STAGES + OPTIONAL)
    ap.add_argument("--from", dest="start", choices=STAGES)
    ap.add_argument("--force", nargs="*", choices=STAGES + OPTIONAL, default=[])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--other-layer", action="store_true",
                    help="permit a --layer different from the paper's for this "
                         "model; the run then reproduces something else")
    ap.add_argument("--check", action="store_true",
                    help="verify every input and script is present, then stop")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    want = PAPER_LAYERS.get(a.model)
    if want is not None and a.layer != want and not a.other_layer:
        sys.exit(f"--layer {a.layer} but the paper uses layer {want} for "
                 f"{a.model}.\n  A run at another layer measures a different "
                 f"basis and will not reproduce the paper's numbers.\n  Pass "
                 f"--layer {want}, or --other-layer if this is deliberate.")

    S = slug(a.model)
    W = os.path.abspath(a.work)
    D = os.path.join(W, S)
    os.makedirs(D, exist_ok=True)
    led = Ledger(os.path.join(D, "ledger.json"))
    log = os.path.join(D, "run.log")
    code = os.path.abspath(a.code)
    PY = sys.executable

    P = {
        "vectors": os.path.join(D, f"{S}_vecs.npz"),
        "heldout": os.path.join(D, f"{S}_heldout_vecs.npz"),
        "layers": os.path.join(D, "layer_auc.csv"),
        "validation": os.path.join(D, "validation.json"),
        "edge": os.path.join(D, "edge.json"),
        "gens": os.path.join(D, "gens"),
        "gens_single": os.path.join(D, "gens_single"),
        "gens_common": os.path.join(D, "gens_common"),
        "transfer_raw": os.path.join(D, "transfer_raw.json"),
        "transfer_dual": os.path.join(D, "transfer_dual.json"),
        "align": os.path.join(D, "align.json"),
        "atlas": os.path.join(D, f"atlas_{S}.json"),
    }

    base_in = {"corpus": sha(a.corpus), "labels": sha(a.labels),
               "probes": sha(a.probes), "model": a.model, "layer": a.layer}
    LAYER_FREE = {"extract", "heldout"}   # these extract every layer

    if a.check:
        sys.exit(0 if preflight(a, code) else 1)

    if a.status:
        print(f"{a.model}  layer {a.layer}\n  {D}\n")
        for st in STAGES:
            r = led.d.get(st, {})
            mark = {"ok": "done", "failed": "FAILED"}.get(r.get("status"), "—")
            extra = ""
            if r.get("status") == "ok":
                extra = f"  {r['finished']}  {r['seconds']:.0f}s"
            elif r.get("status") == "failed":
                extra = f"  exit {r.get('exit_code')}"
            print(f"  {st:<10}{mark:<8}{extra}")
        cs = led.d.get("edge", {}).get("c_star")
        if cs:
            print(f"\n  c* = {cs}")
        return

    todo = a.only or STAGES
    if a.start:
        todo = STAGES[STAGES.index(a.start):]
    todo = [s for s in STAGES + OPTIONAL if s in todo]

    print(f"reproduce  {a.model}  layer {a.layer}")
    print(f"  work dir {D}")
    print(f"  stages   {' '.join(todo)}\n")

    c_star = a.c_star or led.d.get("edge", {}).get("c_star")

    for st in todo:
        inputs = dict(base_in)
        if st in LAYER_FREE:
            inputs.pop("layer", None)
        if st in ("generate", "atlas"):
            inputs["c_star"] = c_star
        if st == "heldout":
            inputs["heldout_prompts"] = sha(a.heldout_prompts)
        if st == "validate":
            inputs["predictions"] = sha(a.predictions)
        if st in ("analyse",):
            inputs["gens"] = sha(os.path.join(P["gens"], "done.txt"))

        if st not in a.force:
            ok, why = led.done(st, inputs)
            if ok:
                print(f"  {st:<10} skip     ({why})")
                continue

        if st in NEEDS_GPU:
            print(f"  {st:<10} run      [GPU]")
        else:
            print(f"  {st:<10} run")

        cmds = []
        outs = []
        if st == "extract":
            cmds = [[PY, os.path.join(code, "extract_vectors.py"),
                     "--prompts", a.corpus, "--model", a.model,
                     "--out-dir", D]]
            outs = [P["vectors"]]
        elif st == "heldout":
            if not os.path.exists(a.heldout_prompts):
                print(f"           skip     ({a.heldout_prompts} absent; "
                      f"test C will not be reproducible)")
                continue
            # --tag names the output file. --suffix is something else entirely:
            # it APPENDS TEXT TO EVERY PROMPT, which with final-token pooling
            # moves the pooled position onto the appended text. An earlier
            # version passed --suffix here; extract_vectors' overwrite guard
            # refused because the output name did not change, and that refusal
            # is the only reason held-out vectors were not silently extracted
            # at the token "heldout" instead of at the word.
            cmds = [[PY, os.path.join(code, "extract_vectors.py"),
                     "--prompts", a.heldout_prompts, "--model", a.model,
                     "--tag", "heldout", "--out-dir", D]]
            outs = [P["heldout"]]
        elif st == "layers":
            json.dump({S: a.layer}, open(os.path.join(D, "layer_map.json"), "w"))
            if not os.path.exists(P["edge"]):
                sys.exit("  layers needs edge.json; run the edge stage first")
            cmds = [[PY, os.path.join(code, "build_verify_data.py"),
                     "--edges", P["edge"], "--vecs-dir", D,
                     "--labels", a.labels, "--probes", a.probes,
                     "--layer-map", os.path.join(D, "layer_map.json"),
                     "--out", os.path.join(D, "verify_data")]]
            outs = [os.path.join(D, "verify_data", "layer_selection.csv")]
        elif st == "validate":
            cmd = [PY, os.path.join(code, "probe_validate.py"),
                   "--vecs", P["vectors"], "--layers", str(a.layer),
                   "--labels", a.labels, "--probes", a.probes,
                   "--out", P["validation"]]
            # test C needs the pre-registration and its separately extracted
            # vectors; without both, probe_validate falls back to corpus words
            # and labels the result as the weaker claim it is
            if os.path.exists(a.predictions) and os.path.exists(P["heldout"]):
                cmd += ["--predictions", a.predictions,
                        "--heldout-vecs", P["heldout"]]
            elif os.path.exists(a.predictions):
                print("           note     predictions present but held-out "
                      "vectors are not; run the heldout stage first")
            cmds = [cmd]
            outs = [P["validation"]]
        elif st == "edge":
            if a.c_star:
                print("           skip     (--c-star supplied)")
                continue
            cmds = [[PY, os.path.join(code, "coherence_edge.py"),
                     "--model", a.model, "--vecs", P["vectors"],
                     "--labels", a.labels, "--probes", a.probes,
                     "--layer", str(a.layer),
                     "--batch-rows", str(a.batch_rows), "--out", P["edge"]]]
            outs = [P["edge"]]
        elif st == "generate":
            if not c_star:
                if a.dry_run:
                    c_star = "<c* from the edge stage>"
                else:
                    sys.exit("  no c*: run the edge stage first, or pass "
                             "--c-star if you already know it")
            os.makedirs(P["gens"], exist_ok=True)
            cmds = [[PY, os.path.join(code, "steer_gen.py"),
                     "--model", a.model, "--vecs", P["vectors"],
                     "--labels", a.labels, "--probes", a.probes,
                     "--layer", str(a.layer), "--arms", "raw", "dual", "control",
                     "--reps", str(a.reps), "--n-gen", str(a.n_gen),
                     "--max-new-tokens", "80", "--c", str(c_star),
                     "--batch-rows", str(a.batch_rows),
                     "--force-magnitude", "--out", P["gens"]]]
            outs = [os.path.join(P["gens"], "manifest.json")]
        elif st in ("single", "commonmag"):
            # single: the SAME trials, stems and c* as "generate", each cell turning
            # one of its triple's dials alone (section 7.3, superposition test).
            # commonmag: the triple design at one common magnitude, c = 0.08.
            if st == "single" and not c_star:
                if a.dry_run:
                    c_star = "<c* from the edge stage>"
                else:
                    sys.exit("  no c*: run the edge stage first, or pass --c-star")
            out_dir = P["gens_single"] if st == "single" else P["gens_common"]
            os.makedirs(out_dir, exist_ok=True)
            cmds = [[PY, os.path.join(code, "steer_gen.py"),
                     "--model", a.model, "--vecs", P["vectors"],
                     "--labels", a.labels, "--probes", a.probes,
                     "--layer", str(a.layer),
                     "--arms", *(["raw", "dual"] if st == "single" else ["raw"]),
                     "--reps", str(a.reps), "--n-gen", str(a.n_gen),
                     "--max-new-tokens", "80",
                     "--c", str(c_star if st == "single" else 0.08),
                     "--batch-rows", str(a.batch_rows), "--force-magnitude",
                     *(["--design", "single"] if st == "single" else []),
                     "--out", out_dir]]
            outs = [os.path.join(out_dir, "manifest.json")]
        elif st == "analyse":
            lm = os.path.join(D, "layer_map.json")
            json.dump({S: a.layer}, open(lm, "w"))
            for arm, out in (("raw", P["transfer_raw"]),
                             ("dual", P["transfer_dual"])):
                cmds.append([PY, os.path.join(code, "transfer_independent.py"),
                             "--generations", P["gens"], "--probes", a.probes,
                             "--labels", a.labels, "--vecs-dir", D,
                             "--layer-map", lm, "--arm", arm, "--out", out])
            cmds.append([PY, os.path.join(code, "embed_align.py"),
                         "--generations", P["gens"], "--probes", a.probes,
                         "--conditions", "raw", "dedup", "coherent",
                         "--primary", "dedup", "--arm", "raw",
                         "--out", P["align"]])
            outs = [P["transfer_raw"], P["transfer_dual"], P["align"]]
        elif st == "package":
            cmds = [[PY, os.path.join(code, "package_verify.py"),
                     "--work", W, "--models", S,
                     "--labels", a.labels, "--probes", a.probes,
                     "--predictions", a.predictions,
                     "--single-gens", P["gens_single"], "--atlas", P["atlas"],
                     "--out", os.path.join(W, "verify_package")]]
            outs = [os.path.join(W, "verify_package", "data", "MANIFEST.json")]
        elif st == "atlas":
            if not c_star:
                if a.dry_run:
                    c_star = "<c* from the edge stage>"
                else:
                    sys.exit("  no c*: the atlas bakes it in as the default "
                             "magnitude, so the edge stage must run first")
            cmd = [PY, os.path.join(code, "make_atlas.py"),
                   "--model", a.model, "--vecs", P["vectors"],
                   "--labels", a.labels, "--probes", a.probes,
                   "--layer", str(a.layer), "--c-star", str(c_star),
                   "--out", P["atlas"]]
            if os.path.exists(a.ekman):
                cmd += ["--ekman", a.ekman]
            cmds = [cmd]
            outs = [P["atlas"]]

        if a.dry_run:
            for c in cmds:
                print(f"    $ {' '.join(c)}")
            continue

        t0 = time.time()
        failed = False
        for c in cmds:
            rc = run(c, log)
            if rc:
                led.fail(st, " ".join(c), rc)
                print(f"  {st:<10} FAILED   exit {rc}; see {log}")
                print(f"             fix, then: --from {st}")
                failed = True
                break
        if failed:
            sys.exit(1)

        if st == "edge":
            c_star = json.load(open(P["edge"]))["c_star"]
            if c_star is None:
                sys.exit("  no level passed the fluency criterion; extend the "
                         "ladder downward before generating")
            led.d.setdefault("edge", {})["c_star"] = c_star
            print(f"             c* = {c_star}")
        if st == "generate":
            open(os.path.join(P["gens"], "done.txt"), "w").write("ok")

        led.record(st, inputs, outs, time.time() - t0,
                   " ; ".join(" ".join(c) for c in cmds))
        if st == "edge":
            led.d["edge"]["c_star"] = c_star
            led.save()
        print(f"  {st:<10} done     {time.time() - t0:.0f}s")

    print(f"\n  ledger  {os.path.join(D, 'ledger.json')}")
    print(f"  log     {log}")
    print(f"\n  --status to review, --from <stage> to resume, "
          f"--force <stage> to redo one.")


if __name__ == "__main__":
    main()
