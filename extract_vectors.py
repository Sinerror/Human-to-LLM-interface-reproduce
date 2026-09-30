"""
Emotion LAT vector extractor — reads a prompts.jsonl, writes a SELF-DESCRIBING npz.

  python extract_vectors.py --prompts prmpts.jsonl --model google/gemma-3-1b-pt
  python extract_vectors.py --prompts prmpts.jsonl --model microsoft/phi-2 --suffix " ."
  python extract_vectors.py --selftest          # no model needed, checks the plumbing
  python extract_vectors.py --show FILE.npz     # print an existing file's provenance

SECURITY: put your HF token in the environment, never in the file:
  export HF_TOKEN=hf_xxx        (Windows: setx HF_TOKEN hf_xxx)

WHAT CHANGED, AND WHY
  1. --suffix replaces the hard-edited `r["text"] += " ."` line. Delimiter
     pooling is now a recorded experimental condition, not an invisible edit.
  2. Every npz carries a `meta` field: model, precision, pooling, suffix,
     prompt-file path AND sha256, counts, date. A file can now say what it is.
     Nothing depends on the filename any more.
  3. Output filenames include a tag, and the script REFUSES to overwrite an
     existing npz unless --force is given. Previously a second run with
     different prompts silently replaced the first — the direct cause of
     several mixed-up comparisons in this project.
  4. --prompts is REQUIRED. It used to default to a neutral-baseline file,
     so a bare run produced a neutral set under a corpus-looking name.
"""
import os, json, argparse, hashlib, datetime
from pathlib import Path
from collections import defaultdict
import numpy as np

# ── Defaults (all overridable on the command line) ────────────────────────────
OUT_DIR       = Path("./lat_outputs")
BATCH_SIZE    = 16
MAX_LEN       = 64
POOLING       = "last"     # "last" = last-token | "span" = mean over the word's tokens
SAVE_LAYERS   = None       # None = all layers
MIN_PROMPTS   = 1          # drop a word::sense with fewer than this many prompts
DROP_FALLBACK = False      # True = ignore rows whose "via" == "fallback"
# ──────────────────────────────────────────────────────────────────────────────


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_prompts(path, suffix="", min_prompts=MIN_PROMPTS,
                 drop_fallback=DROP_FALLBACK):
    """jsonl -> {"word::sense": [record, ...]}.  Sense is part of the key.

    `suffix` is appended to every prompt. Use " ." to pool a trailing delimiter
    instead of the final content word. It is recorded in the output metadata.
    """
    groups = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if drop_fallback and r.get("via") == "fallback":
                continue
            if suffix:
                r["text"] = r["text"] + suffix
            groups[f'{r["word"]}::{r.get("sense", "default")}'].append(r)
    groups = {k: v for k, v in groups.items() if len(v) >= min_prompts}
    n_words = len({k.split("::")[0] for k in groups})
    n_p = sum(len(v) for v in groups.values())
    print(f"{n_p} prompts | {len(groups)} word::sense keys | {n_words} distinct words"
          + (f" | suffix {suffix!r}" if suffix else ""))
    if len(groups) <= 120:
        print(f"  NOTE only {len(groups)} keys — is this a neutral/baseline set "
              f"rather than the corpus?")
    return groups


def load_model(model_id, use_4bit=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,
                                        token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    kw = dict(device_map="auto", trust_remote_code=True,
              token=os.environ.get("HF_TOKEN"))
    if use_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    return tok, model


def hidden_states(model, tok, texts, spans=None, pooling=POOLING):
    """Returns (n_texts, n_layers+1, hidden)."""
    import torch
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=MAX_LEN, return_offsets_mapping=(pooling == "span"))
    offsets = enc.pop("offset_mapping", None)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states
    mask = enc["attention_mask"]

    if tok.padding_side == "left":
        last_idx = torch.full((mask.shape[0],), mask.shape[1] - 1, device=mask.device)
    else:
        last_idx = mask.sum(1) - 1

    res = []
    for b in range(len(texts)):
        idx = None
        if pooling == "span" and spans and spans[b] and spans[b][0] >= 0:
            s, e = spans[b]
            idx = [i for i, (a, z) in enumerate(offsets[b].tolist())
                   if z > a and a < e and z > s]
        if idx:
            vec = torch.stack([h[b, idx, :].mean(0) for h in hs])
        else:
            vec = torch.stack([h[b, last_idx[b], :] for h in hs])
        res.append(vec.float().cpu().numpy())
    return np.stack(res)


def extract(model, tok, groups, pooling=POOLING):
    """Running mean per word::sense, so RAM stays flat regardless of prompt count."""
    items = [(k, r) for k, rs in groups.items() for r in rs]
    items.sort(key=lambda x: len(x[1]["text"]))
    sums, counts = {}, defaultdict(int)

    for i in range(0, len(items), BATCH_SIZE):
        chunk = items[i:i + BATCH_SIZE]
        texts = [r["text"] for _, r in chunk]
        spans = [(r.get("word_at", -1), r.get("word_at", -1) + len(r["word"]))
                 for _, r in chunk]
        hs = hidden_states(model, tok, texts, spans, pooling)
        for b, (key, _) in enumerate(chunk):
            sums[key] = hs[b] if key not in sums else sums[key] + hs[b]
            counts[key] += 1
        if (i // BATCH_SIZE) % 20 == 0:
            print(f"  {min(i + BATCH_SIZE, len(items))}/{len(items)} prompts")

    return {k: sums[k] / counts[k] for k in sums}, dict(counts)


def save(vecs, counts, meta, out_path, save_layers=SAVE_LAYERS, force=False):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        raise SystemExit(
            f"\nREFUSING TO OVERWRITE {out_path}\n"
            f"  An extraction already exists there. Overwriting it silently is how\n"
            f"  runs get confused. Either pass --tag to name this run differently,\n"
            f"  or pass --force if you really mean to replace it.\n")

    keys = list(vecs)
    arr = np.stack([vecs[k] for k in keys])
    if save_layers:
        arr = arr[:, save_layers, :]
    meta = dict(meta)
    meta.update(n_keys=len(keys), n_layers=int(arr.shape[1]),
                hidden=int(arr.shape[2]),
                n_words=len({k.split("::")[0] for k in keys}))

    np.savez_compressed(
        out_path,
        vecs=arr.astype(np.float16),
        keys=np.array(keys),
        words=np.array([k.split("::")[0] for k in keys]),
        senses=np.array([k.split("::", 1)[1] for k in keys]),
        n_prompts=np.array([counts[k] for k in keys]),
        layers=np.array(save_layers if save_layers else range(arr.shape[1])),
        meta=np.array([json.dumps(meta, indent=1)]),   # <- self-description
    )
    print(f"\nsaved -> {out_path}")
    print("  embedded provenance:")
    for k, v in meta.items():
        print(f"    {k:<16} {v}")


def show(path):
    """Print an npz's embedded provenance, or say plainly that it has none."""
    d = np.load(path, allow_pickle=False)
    print(f"\n{path}")
    print(f"  shape  {d['vecs'].shape}  {d['vecs'].dtype}")
    if "meta" not in d:
        print("  meta   ABSENT — this file predates provenance stamping.")
        print("         Its model / prompts / suffix cannot be recovered from the file.")
        print("         Use artifact.py to fingerprint and characterise it instead.")
        return
    meta = json.loads(str(d["meta"][0]))
    for k, v in meta.items():
        print(f"    {k:<16} {v}")


def selftest():
    import types, torch
    print("selftest: fake model, checking keys / pooling / aggregation / suffix\n")
    with open("_t.jsonl", "w") as f:
        for w, s, t in [("zeal", "fervor", "I feel zeal for the cause"),
                        ("zeal", "fervor", "Nothing but zeal"),
                        ("zeal", "fervor", "Pure zeal"),
                        ("zeal", "enthusiasm", "I am full of zeal for this"),
                        ("zeal", "enthusiasm", "So much zeal here"),
                        ("zeal", "enthusiasm", "Endless zeal"),
                        ("joy", "default", "I feel joy"),
                        ("joy", "default", "Such joy"),
                        ("joy", "default", "Pure joy")]:
            f.write(json.dumps({"word": w, "sense": s, "text": t,
                                "word_at": t.lower().rfind(w), "via": "llm"}) + "\n")
    g = load_prompts("_t.jsonl")
    assert set(g) == {"zeal::fervor", "zeal::enthusiasm", "joy::default"}, g.keys()
    print("OK  senses kept apart:", sorted(g))

    gs = load_prompts("_t.jsonl", suffix=" .")
    assert all(r["text"].endswith(" .") for rs in gs.values() for r in rs)
    assert not any(r["text"].endswith(" .") for rs in g.values() for r in rs), \
        "suffix leaked into the un-suffixed load"
    print("OK  --suffix applied only where asked")

    class Tok:
        pad_token = "<p>"; eos_token = "<p>"; padding_side = "right"
        def __call__(self, texts, **kw):
            ids = [[ord(c) % 50 for c in t[:MAX_LEN]] for t in texts]
            T = max(len(i) for i in ids)
            out = {"input_ids": torch.tensor([i + [0] * (T - len(i)) for i in ids]),
                   "attention_mask": torch.tensor([[1] * len(i) + [0] * (T - len(i)) for i in ids])}
            if kw.get("return_offsets_mapping"):
                out["offset_mapping"] = torch.tensor(
                    [[[j, j + 1] for j in range(len(i))] + [[0, 0]] * (T - len(i)) for i in ids])
            return types.SimpleNamespace(pop=lambda k, d=None: out.pop(k, d),
                                         __getitem__=out.__getitem__, items=out.items,
                                         keys=out.keys)
    class M:
        device = "cpu"
        def __call__(self, input_ids=None, attention_mask=None, **kw):
            B, T = input_ids.shape
            hs = tuple(torch.randn(B, T, 8) for _ in range(4))
            return types.SimpleNamespace(hidden_states=hs)
    tok, model = Tok(), M()
    vecs, counts = extract(model, tok, g)
    assert set(vecs) == set(g) and all(counts[k] == 3 for k in counts), counts
    print("OK  aggregation:", {k: counts[k] for k in counts},
          "| shape", vecs["joy::default"].shape)

    meta = dict(model_id="fake/model", precision="none", pooling="last",
                suffix="", prompts_path="_t.jsonl", prompts_sha256="0" * 64)
    save(vecs, counts, meta, "_t_out.npz", force=True)
    show("_t_out.npz")
    d = np.load("_t_out.npz", allow_pickle=False)
    assert "meta" in d and json.loads(str(d["meta"][0]))["model_id"] == "fake/model"
    print("OK  provenance survives a save/load round trip")

    try:
        save(vecs, counts, meta, "_t_out.npz", force=False)
        raise AssertionError("overwrite guard did not fire")
    except SystemExit:
        print("OK  refuses to overwrite without --force")

    os.remove("_t.jsonl"); os.remove("_t_out.npz")
    print("\nselftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", default=None, metavar="FILE.npz",
                    help="print an existing npz's embedded provenance and exit")
    ap.add_argument("--prompts", default=None,
                    help="REQUIRED for extraction. jsonl with word/sense/text.")
    ap.add_argument("--model", default=None, help="REQUIRED. HF model id.")
    ap.add_argument("--suffix", default="",
                    help='appended to every prompt. Use " ." for delimiter pooling.')
    ap.add_argument("--pooling", default=POOLING, choices=["last", "span"])
    ap.add_argument("--4bit", dest="four_bit", action="store_true",
                    help="load in 4-bit NF4 (NOT comparable with bf16 runs)")
    ap.add_argument("--tag", default=None,
                    help="distinguishes runs of the same model, e.g. 'delim' or "
                         "'templated'. Goes in the filename and the metadata.")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true", help="allow overwrite")
    ap.add_argument("--min-prompts", type=int, default=MIN_PROMPTS)
    a = ap.parse_args()

    if a.selftest:
        selftest(); raise SystemExit
    if a.show:
        show(a.show); raise SystemExit
    if not a.prompts or not a.model:
        raise SystemExit("--prompts and --model are both required "
                         "(no defaults: a wrong default is how neutral sets got "
                         "extracted under corpus filenames)")

    groups = load_prompts(a.prompts, suffix=a.suffix, min_prompts=a.min_prompts)
    meta = dict(
        model_id=a.model,
        precision="4bit-nf4" if a.four_bit else "bf16",
        pooling=a.pooling,
        suffix=a.suffix,
        tag=a.tag or "",
        prompts_path=str(a.prompts),
        prompts_sha256=sha256_file(a.prompts),
        min_prompts=a.min_prompts,
        extracted_utc=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    tag = ("_" + a.tag) if a.tag else ""
    out = Path(a.out_dir) / f"{a.model.replace('/', '_')}{tag}_vecs.npz"

    tok, model = load_model(a.model, use_4bit=a.four_bit)
    vecs, counts = extract(model, tok, groups, pooling=a.pooling)
    save(vecs, counts, meta, out, force=a.force)
