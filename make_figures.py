#!/usr/bin/env python3
"""
make_figures.py — every figure and table in the paper, from the result files.

  python make_figures.py --embed embed_align.json --embed2 embed_align_bge.json \
      --transfer transfer.json --validation probe_validation.json \
      --vecs google_gemma-3-1b-it_vecs.npz --labels frozen_emotionality_labels.csv \
      --probes canonical_probes.json --layer 17 --outdir figures/

Produces
  fig1_cross_alignment.{png,pdf}   7x7 construct specificity  (section 5.2)
  fig2_basis_geometry.{png,pdf}    Gram matrix + iso-cost     (section 3.2)
  fig3_steering_effects.{png,pdf}  deltas with CIs, 2 embedders (section 5.2)
  fig4_simultaneous.{png,pdf}      transfer-matrix summary    (section 6.2)
  table1_gram.md                   full Gram matrix           (section 3.2)
  table2_validation.md             per-axis A/B/C             (section 4.3)
  table3_entanglement.md           knob-pair guidance         (section 6.4)

Every panel is generated from a released result file, so a reviewer can
regenerate the figures from the same artifacts that produced the numbers.
"""
import argparse, csv, json, os
import sys
from collections import defaultdict
import numpy as np
import matplotlib

# A Windows console is often cp1251 or cp437 and cannot encode much of what
# either the models or these scripts print. Replace unencodable characters
# rather than raising, which would otherwise kill a run after its expensive
# stage had completed. JSON output uses ensure_ascii and is unaffected.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ORDER = ["valence", "heat", "arousal", "intensity", "antagonism_peace",
         "body_mind", "social_inner_outer"]
SHORT = {"valence": "valence", "heat": "heat", "arousal": "arousal",
         "intensity": "intensity", "antagonism_peace": "antagonism",
         "body_mind": "body–mind", "social_inner_outer": "social i/o",
         "security_alarm": "security", "desire": "desire", "drive": "drive"}


def lab(k):
    return SHORT.get(k, k)


def load_labels(p):
    out = {}
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        out[r["key"]] = (r.get("final", "").strip().upper() or r["heuristic"])
    return out


def probe_dirs(vecs, layer, labels, probes, names):
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
    out = {}
    for nm in names:
        s = probes.get(nm, {})
        pos = [w.lower() for w in s.get("pos", []) if w.lower() in wv]
        neg = [w.lower() for w in s.get("neg", []) if w.lower() in wv]
        if len(pos) >= 4 and len(neg) >= 4:
            d = np.mean([wv[w] for w in pos], 0) - np.mean([wv[w] for w in neg], 0)
            out[nm] = d / np.linalg.norm(d)
    return out, Xc


def heat(ax, M, rows, cols, vmin, vmax, fmt="{:+.2f}", cmap="RdBu_r", cbar=True):
    # TwoSlopeNorm needs vmin < 0 < vmax; a magnitude-only matrix uses a plain scale
    norm = (TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 < vmax
            else matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    im = ax.imshow(M, cmap=cmap, norm=norm)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([lab(c) for c in cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab(r) for r in rows], fontsize=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 * max(abs(vmin), abs(vmax)) else "black",
                    fontweight="bold" if i == j else "normal")
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def fig1(embed, outdir, boot=None):
    """7x7 cross-alignment: does each axis steer toward its OWN construct?

    Prefers the bootstrap file, which uses every generation. embed_align caps
    at 400 per group, so its matrix differs slightly from the one the paper
    quotes, and a figure should match the number printed beside it."""
    if boot:
        ks = boot["axes"]
        M = np.array(boot["matrix"])
    else:
        od = embed["conditions"]["dedup"]["POOLED"]["offdiag"]
        ks = [k for k in ORDER if k in od] + [k for k in od if k not in ORDER]
        M = np.array([[od[i].get(j, np.nan) for j in ks] for i in ks])
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    heat(ax, M, ks, ks, -0.35, 0.65)
    dm = sum(int(np.nanargmax(M[i]) == i) for i in range(len(ks)))
    ax.set_xlabel("probe contrast measured", fontsize=9)
    ax.set_ylabel("axis steered", fontsize=9)
    ax.set_title(f"Steering is construct-specific\n"
                 f"diagonal is the maximum for {dm}/{len(ks)} axes "
                 f"(permutation p < 0.001)", fontsize=10)
    for i in range(len(ks)):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                   edgecolor="black", lw=1.8))
    plt.tight_layout()
    for e in ("png", "pdf"):
        plt.savefig(f"{outdir}/fig1_cross_alignment.{e}", dpi=200, bbox_inches="tight")
    plt.close()
    return dm, len(ks)


def fig2(dirs, outdir):
    """Gram matrix and iso-cost: the basis is oblique, and by how much."""
    ks = [k for k in ORDER if k in dirs] + [k for k in dirs if k not in ORDER]
    P = np.array([dirs[k] for k in ks])
    G = P @ P.T
    iso = np.sqrt(np.diag(np.linalg.inv(G)))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    heat(axes[0], np.abs(G), ks, ks, 0, 1, fmt="{:.2f}", cmap="viridis")
    iu = np.triu_indices(len(ks), 1)
    axes[0].set_title(f"Basis Gram matrix  |cos|\nmean off-diagonal "
                      f"{np.abs(G[iu]).mean():.3f}, max {np.abs(G[iu]).max():.3f}, "
                      f"cond {np.linalg.cond(G):.1f}", fontsize=10)
    y = np.arange(len(ks))
    axes[1].barh(y, iso, color="#4a6fa5")
    axes[1].axvline(1.0, color="crimson", ls="--", lw=1.4)
    axes[1].text(1.02, len(ks) - 0.4, "orthogonal\nbasis = 1.0", color="crimson",
                 fontsize=8, va="top")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([lab(k) for k in ks], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("iso-cost  √diag(G⁻¹)", fontsize=9)
    axes[1].set_title("Cost of moving one coordinate\nwhile holding the others fixed",
                      fontsize=10)
    axes[1].set_xlim(0, max(iso) * 1.25)
    axes[1].grid(axis="x", alpha=.3)
    plt.tight_layout()
    for e in ("png", "pdf"):
        plt.savefig(f"{outdir}/fig2_basis_geometry.{e}", dpi=200, bbox_inches="tight")
    plt.close()
    return G, iso, ks


def fig3(embed, embed2, outdir, verified=None):
    """Steering deltas with bootstrap CIs, both embedders."""
    def per_model(E):
        """Per-model mean gain per axis, which is what the paper's table reports.
        An earlier version plotted the POOLED figures, which run higher and do
        not match the table beside the figure."""
        C = E["conditions"]["dedup"]
        mods = [k for k in C if k != "POOLED"]
        out = {}
        for k in C["POOLED"]["knobs"]:
            ds = [C[m]["knobs"][k]["delta"] for m in mods
                  if "delta" in C[m]["knobs"].get(k, {})]
            if ds:
                out[k] = {"delta": float(np.mean(ds)),
                          "sd": float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0}
        return out
    if verified:
        # verify.py's own recomputation: every generation, no per-group cap.
        # This is the source the paper's per-axis table quotes.
        gm = verified["gain_per_model"]
        A = {}
        for k in {x for d in gm.values() for x in d}:
            ds = [gm[m][k] for m in gm if k in gm[m]]
            A[k] = {"delta": float(np.mean(ds)),
                    "sd": float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0}
    else:
        A = per_model(embed)
    B = per_model(embed2) if embed2 else {}
    ks = [k for k in ORDER if k in A] + [k for k in A if k not in ORDER]
    ks = sorted(ks, key=lambda k: A[k].get("delta", 0))
    y = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    da = [A[k].get("delta", np.nan) for k in ks]
    err = [A[k]["sd"] for k in ks]
    ax.barh(y - 0.19, da, height=0.36, color="#2b6a4d", label="all-mpnet-base-v2")
    ax.errorbar(da, y - 0.19, xerr=err, fmt="none", ecolor="black",
                elinewidth=1, capsize=2)
    if B:
        db = [B.get(k, {}).get("delta", np.nan) for k in ks]
        ax.barh(y + 0.19, db, height=0.36, color="#8fb996", label="bge-large-en-v1.5")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([lab(k) for k in ks], fontsize=9)
    ax.set_xlabel("alignment gain over matched random-direction control", fontsize=9)
    ax.set_title("Every axis steers toward its own construct\n"
                 "mean over nine models at each model's c*, bars = SD across models",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=.3)
    plt.tight_layout()
    for e in ("png", "pdf"):
        plt.savefig(f"{outdir}/fig3_steering_effects.{e}", dpi=200, bbox_inches="tight")
    plt.close()



def model_label(m):
    """Readable model name from any of the key forms in use: hub slugs
    (google_gemma-3-1b-it), generation names with an arm suffix
    (gemma-3-1b-it_raw), or both. Taking the last '_' piece returns the arm."""
    for suf in ("_raw", "_dual", "_control"):
        if m.endswith(suf):
            m = m[: -len(suf)]
    for pre in ("google_", "Qwen_", "microsoft_", "state-spaces_", "EleutherAI_",
                "mistralai_", "meta-llama_", "unsloth_"):
        if m.startswith(pre):
            m = m[len(pre):]
    return m[:22]


def from_independent(path):
    """Build figure 4's input from transfer_independent output.

    Two implementations of the transfer matrix exist and, on the same c*
    generations, they disagree: transfer_matrix.py gives 91.5% diagonal
    dominance on average, transfer_independent.py 85.7%. The paper and
    verify.py use the latter, so figure 4 must too, or it contradicts the table
    beside it. Normalisation is verify's: each row scaled by its largest
    absolute entry, the ratio averaged over triples rather than formed from
    averaged numerators and denominators.
    """
    raw = json.load(open(path, encoding="utf-8"))
    out = {"models": {}}
    for m, v in raw.items():
        rows = hit = 0
        ratios, triples = [], {}
        iu, il = np.triu_indices(3, 1), np.tril_indices(3, -1)
        for p in v["per_triple"]:
            M = np.array(p["M"], float)
            Mn = M / (np.abs(M).max(axis=1, keepdims=True) + 1e-12)
            off = np.concatenate([np.abs(Mn[iu]), np.abs(Mn[il])]).mean()
            ratios.append(np.abs(np.diag(Mn)).mean() / max(off, 1e-9))
            hit += sum(int(np.argmax(np.abs(Mn[i])) == i) for i in range(3))
            rows += 3
            triples["|".join(p["triple"])] = {"M": Mn.tolist(),
                                              "mean_abs_offdiag": float(off)}
        out["models"][m] = {"diag_is_max": hit, "diag_rows": rows,
                            "ratio": float(np.mean(ratios)),
                            "mean_diag": float(np.mean(ratios)),
                            "mean_offdiag": 1.0, "triples": triples,
                            "n_triples": len(v["per_triple"])}
    return out


def fig4(tr, outdir):
    """Simultaneous control: diagonal dominance per model, and an example matrix."""
    ms = sorted(tr["models"], key=lambda m: -tr["models"][m]["mean_diag"] /
                max(tr["models"][m]["mean_offdiag"], 1e-9))
    ratio = [tr["models"][m]["mean_diag"] / max(tr["models"][m]["mean_offdiag"], 1e-9)
             for m in ms]
    frac = [tr["models"][m]["diag_is_max"] / tr["models"][m]["diag_rows"] for m in ms]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(len(ms))
    axes[0].barh(y, ratio, color="#3d5a80")
    axes[0].axvline(1.0, color="crimson", ls="--", lw=1.4)
    axes[0].text(1.04, len(ms) - .5, "entangled\n(ratio = 1)", color="crimson",
                 fontsize=8, va="top")
    for i, f in enumerate(frac):
        axes[0].text(ratio[i] + .06, i, f"{f:.0%} rows", va="center", fontsize=7.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([model_label(m) for m in ms], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("mean |diagonal| ÷ mean |off-diagonal|", fontsize=9)
    axes[0].set_xlim(0, max(ratio) * 1.28)
    tot_rows = sum(tr["models"][m]["diag_rows"] for m in ms)
    tot_max = sum(tr["models"][m]["diag_is_max"] for m in ms)
    axes[0].set_title("Three knobs at once: each moves its own construct\n"
                      f"{tot_max}/{tot_rows} knob-rows diagonal-dominant "
                      f"({tot_max/tot_rows:.1%}; chance 33%)", fontsize=10)
    axes[0].grid(axis="x", alpha=.3)

    best = ms[0]
    t = tr["models"][best]["triples"]
    pick = min(t.items(), key=lambda kv: kv[1]["mean_abs_offdiag"])
    ks = pick[0].split("|")
    M = np.array(pick[1]["M"])
    heat(axes[1], M, ks, ks, -1, 1, cbar=False)
    for i in range(3):
        axes[1].add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                        edgecolor="black", lw=1.8))
    axes[1].set_xlabel("construct moved", fontsize=9)
    axes[1].set_ylabel("knob turned", fontsize=9)
    axes[1].set_title(f"Example transfer matrix\n{model_label(best)}",
                      fontsize=10)
    plt.tight_layout()
    for e in ("png", "pdf"):
        plt.savefig(f"{outdir}/fig4_simultaneous.{e}", dpi=200, bbox_inches="tight")
    plt.close()


def fig5(tr, outdir):
    """Magnitude: how far does one knob move the text, in interpretable units?"""
    agg = defaultdict(lambda: defaultdict(list))
    for m, v in tr["models"].items():
        for k, mm in v.get("magnitude", {}).items():
            agg[k]["pole"].append(mm["pole_fraction"])
            agg[k]["d"].append(mm["cohens_d"])
    if not agg:
        return
    ks = sorted(agg, key=lambda x: np.nanmean(agg[x]["pole"]))
    y = np.arange(len(ks))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=True)
    pf = [np.nanmean(agg[k]["pole"]) for k in ks]
    pe = [np.nanstd(agg[k]["pole"]) for k in ks]
    axes[0].barh(y, pf, xerr=pe, color="#8c5a3c", ecolor="black", capsize=2)
    axes[0].set_yticks(y); axes[0].set_yticklabels([lab(k) for k in ks], fontsize=9)
    axes[0].set_xlabel("fraction of the probe's pole-to-pole distance", fontsize=9)
    axes[0].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].set_title("How far steering moves the text", fontsize=10)
    axes[0].grid(axis="x", alpha=.3)
    dd = [np.nanmean(agg[k]["d"]) for k in ks]
    de = [np.nanstd(agg[k]["d"]) for k in ks]
    axes[1].barh(y, dd, xerr=de, color="#3c6e8c", ecolor="black", capsize=2)
    for x, t in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        axes[1].axvline(x, color="grey", ls=":", lw=1)
        axes[1].text(x, len(ks) - .3, t, fontsize=7, color="grey",
                     rotation=90, va="top", ha="right")
    axes[1].set_xlabel("Cohen's d  (shift ÷ spread across generations)", fontsize=9)
    axes[1].set_title("The same shift as an effect size", fontsize=10)
    axes[1].grid(axis="x", alpha=.3)
    plt.suptitle("Steering magnitude per axis, mean over nine models",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    for e in ("png", "pdf"):
        plt.savefig(f"{outdir}/fig5_magnitude.{e}", dpi=200, bbox_inches="tight")
    plt.close()


def fig6(tr, outdir):
    """Honest spread: every triple, including the ones that barely steer."""
    pf, ratios, labels_ = [], [], []
    for m, v in tr["models"].items():
        for tname, t in v.get("triples", {}).items():
            for k, mm in t.get("magnitude", {}).items():
                if np.isfinite(mm.get("pole_fraction", np.nan)):
                    pf.append(mm["pole_fraction"])
                    ratios.append(t["mean_diag"] / max(t["mean_abs_offdiag"], 1e-9))
                    labels_.append(k)
    if not pf:
        return
    pf = np.array(pf); ratios = np.array(ratios)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    uk = [k for k in ORDER if k in set(labels_)]
    data = [pf[np.array(labels_) == k] for k in uk]
    # matplotlib 3.9 renamed boxplot's `labels` to `tick_labels`; newer
    # versions drop the old name silently and number the boxes 1..n instead.
    # Setting the ticks directly works on every version.
    bp = ax.boxplot(data, vert=False, showfliers=True, patch_artist=True,
                    widths=.62)
    ax.set_yticks(range(1, len(uk) + 1))
    ax.set_yticklabels([lab(k) for k in uk])
    for b in bp["boxes"]:
        b.set_facecolor("#c8ab7f"); b.set_alpha(.85)
    for i, d in enumerate(data):
        ax.scatter(d, np.full(len(d), i + 1) + np.random.default_rng(0)
                   .normal(0, .06, len(d)), s=7, color="#4a3b23", alpha=.55, zorder=3)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(0.05, color="crimson", ls=":", lw=1.2)
    ax.text(0.052, .6, "5%", color="crimson", fontsize=7.5)
    ax.set_xlabel("pole fraction per individual triple", fontsize=9)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Steering strength varies widely between triples\n"
                 "individual runs, not the model average", fontsize=10)
    ax.tick_params(labelsize=8); ax.grid(axis="x", alpha=.3)

    ax = axes[1]
    ax.scatter(pf, ratios, s=14, alpha=.5, color="#3c6e8c")
    ax.axvline(0.05, color="crimson", ls=":", lw=1.2)
    ax.axhline(1, color="crimson", ls="--", lw=1.2)
    weak = (pf < 0.05).mean()
    ax.set_xlabel("pole fraction  (how far it moves the text)", fontsize=9)
    ax.set_ylabel("diag ÷ off-diag  (how cleanly)", fontsize=9)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(f"Separable but weak is common\n"
                 f"{weak:.0%} of triple-axis cases move < 5% of pole distance",
                 fontsize=10)
    ax.grid(alpha=.3)
    plt.tight_layout()
    for e in ("png", "pdf"):
        plt.savefig(f"{outdir}/fig6_spread.{e}", dpi=200, bbox_inches="tight")
    plt.close()
    return float(weak)



def fig7(lb, outdir):
    """Section 7.3: what a linear model would show through this exact pipeline,
    against what was observed, per model. Drawn only from linear_baseline.json."""
    ms = list(lb["models"])
    lab = [model_label(m) for m in ms]
    y = np.arange(len(ms))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, key, title, fmt in (
            (axes[0], "sep", "Separability after the dual-basis correction\n(ratio to raw; >1 = improves)", "x"),
            (axes[1], "raw_anti_share", "Antisymmetric share of cross-talk\n(raw steering)", "%")):
        pred = [lb["models"][m]["+harness +readout"][key] for m in ms]
        obs = [lb["models"][m]["OBSERVED"][key] for m in ms]
        for yi, p_, o_ in zip(y, pred, obs):
            ax.plot([p_, o_], [yi, yi], color="#bbbbbb", lw=1.2, zorder=1)
        ax.scatter(pred, y, s=42, facecolors="white", edgecolors="#555555",
                   label="linear model, this pipeline", zorder=2)
        ax.scatter(obs, y, s=42, color="#b2182b", label="observed", zorder=3)
        if key == "sep":
            ax.axvline(1, color="#888888", ls=":", lw=1)
        else:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=.3)
    axes[0].set_yticks(y); axes[0].set_yticklabels(lab, fontsize=8)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Where the response departs from a linear model", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{outdir}/fig7_linear_baseline.{ext}", dpi=200)
    plt.close(fig)


def tables(G, iso, ks, val, tr, outdir, pn=None):
    with open(f"{outdir}/table1_gram.md", "w", encoding="utf-8") as f:
        f.write("**Table 1.** Gram matrix of the emotion basis, |cos| between axes "
                "(Gemma-3-1B-it, layer 17). The final column is the iso-cost "
                "√diag(G⁻¹): displacement needed to move one coordinate by one unit "
                "while holding the others fixed. An orthogonal basis would show 1.00.\n\n")
        f.write("| axis | " + " | ".join(lab(k) for k in ks) + " | iso-cost |\n")
        f.write("|" + "---|" * (len(ks) + 2) + "\n")
        for i, k in enumerate(ks):
            cells = " | ".join(("**1.00**" if i == j else f"{abs(G[i, j]):.2f}")
                               for j in range(len(ks)))
            f.write(f"| {lab(k)} | {cells} | {iso[i]:.2f} |\n")
        iu = np.triu_indices(len(ks), 1)
        f.write(f"\nMean off-diagonal {np.abs(G[iu]).mean():.3f}; "
                f"maximum {np.abs(G[iu]).max():.3f}; "
                f"condition number {np.linalg.cond(G):.2f}.\n")

    if pn:
        lw = pn["lowo"]
        with open(f"{outdir}/table2_validation.md", "w", encoding="utf-8") as f:
            f.write("**Table 2.** Leave-one-word-out validation, corrected labels. "
                    "The null permutes pole assignment among the same words. "
                    "Every axis also clears its pre-registered unseen-word null "
                    "in all nine models (minimum AUC 0.891).\n\n")
            f.write("| axis | LOWO accuracy | permuted null p95 | margin |\n"
                    "|---|---|---|---|\n")
            for k in [x for x in ORDER if x in lw]:
                acc, nul = lw[k]["accuracy"], lw[k]["null_p95"]
                f.write(f"| {lab(k)} | {acc:.3f} | {nul:.3f} | {acc-nul:+.3f} |\n")
    elif val:
        with open(f"{outdir}/table2_validation.md", "w", encoding="utf-8") as f:
            f.write("**Table 2.** Axis validation. All nulls are built from word "
                    "contrasts of the same pole sizes, never from random vectors. "
                    "Split-half reliability (SB) is reported and never used as a "
                    "gate.\n\n")
            f.write("| axis | LOWO acc | null p95 | unseen-word AUC | clears | "
                    "cross-model ρ | null p95 | SB |\n")
            f.write("|" + "---|" * 8 + "\n")
            for k in [x for x in ORDER if x in val["probes"]] + \
                     [x for x in val["probes"] if x not in ORDER]:
                e = val["probes"][k]
                pm = [v for v in e["per_model"].values() if v.get("status") == "ok"]
                if not pm:
                    continue
                bs = [v["B"] for v in pm if v.get("B")]
                cs = [v["C"] for v in pm if v.get("C") and v["C"].get("status") == "ok"]
                A_ = e.get("A") or {}
                sb = np.mean([v["sb"] for v in pm])
                f.write(f"| {lab(k)} "
                        f"| {np.mean([b['accuracy'] for b in bs]):.3f} " if bs else "| – ")
                f.write(f"| {np.mean([b['null_p95'] for b in bs]):.3f} " if bs else "| – ")
                f.write(f"| {np.mean([c['auc'] for c in cs]):.3f} " if cs else "| – ")
                f.write(f"| {sum(c['clears_null'] for c in cs)}/{len(cs)} " if cs else "| – ")
                f.write(f"| {A_.get('mean_rho', float('nan')):+.3f} "
                        f"| {A_.get('null_p95', float('nan')):.3f} "
                        f"| {sb:.3f} |\n")

    if tr:
        pair = defaultdict(list)
        for m, v in tr["models"].items():
            for tname, t in v["triples"].items():
                kk = tname.split("|")
                M = np.array(t["M"])
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            pair[tuple(sorted((kk[i], kk[j])))].append(abs(M[i, j]))
        rows = sorted(((np.mean(v), a, b, len(v)) for (a, b), v in pair.items()))
        with open(f"{outdir}/table3_entanglement.md", "w", encoding="utf-8") as f:
            f.write("**Table 3.** Mean |off-diagonal| by axis pair, pooled over all "
                    "triples and models, under raw-direction steering. Low values "
                    "combine freely; high values need the dual-basis correction to "
                    "be controlled independently.\n\n")
            f.write("| pair | mean cross-talk | n |\n|---|---|---|\n")
            for mu, a_, b_, n in rows[:6]:
                f.write(f"| {lab(a_)} – {lab(b_)} | **{mu:.3f}** | {n} |\n")
            f.write("| … | | |\n")
            for mu, a_, b_, n in rows[-6:]:
                f.write(f"| {lab(a_)} – {lab(b_)} | {mu:.3f} | {n} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", required=True)
    ap.add_argument("--embed2", default=None)
    ap.add_argument("--transfer", default=None)
    ap.add_argument("--validation", default=None)
    ap.add_argument("--vecs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--linear-baseline", default=None,
                    help="linear_baseline.json; draws figure 7 (section 7.3)")
    ap.add_argument("--transfer-indep", default=None,
                    help="t2_raw.json from transfer_independent.py; figure 4 "
                         "is drawn from it so it matches section 6.2 and verify")
    ap.add_argument("--verified", default=None,
                    help="verify_values.json from verify.py --dump; figure 3 "
                         "then shows exactly the per-axis gains the paper quotes")
    ap.add_argument("--crossalign", default=None,
                    help="crossalign_boot.json; figure 1 then matches the text")
    ap.add_argument("--paper-numbers", default=None,
                    help="paper_numbers.json; Table 2 is built from it instead "
                         "of the superseded probe_validation.json")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    embed = json.load(open(a.embed, encoding="utf-8"))
    # Each of these has a canonical source the paper quotes, and a fallback
    # that runs but draws something slightly different. A silent fallback
    # produced four figures that disagreed with the text; say it out loud.
    missing = [(flag, what) for flag, what, val in (
        ("--crossalign", "Fig 1 from capped ea2 (0.474), text says 0.478", a.crossalign),
        ("--verified", "Fig 3 from capped ea2, text quotes verify", a.verified),
        ("--transfer-indep", "Fig 4 from transfer_matrix.py (91.5%), text says 85.7%", a.transfer_indep),
        ("--paper-numbers", "Table 2 from superseded probe_validation.json", a.paper_numbers),
    ) if not val]
    if missing:
        print("\n" + "!" * 70)
        print("  FIGURES WILL NOT MATCH THE PAPER. Missing canonical sources:")
        for flag, what in missing:
            print(f"    {flag:<18} {what}")
        print("  See FIGURE_COMMANDS.md. Run verify.py --dump first for --verified.")
        print("!" * 70 + "\n")
    stale = []
    if "real" in (embed.get("arms") or {}):
        stale.append(f"{a.embed}: arm 'real' is the old harness")
    if a.transfer:
        _t = json.load(open(a.transfer, encoding="utf-8"))
        _n = [v.get("n_triples", 0) for v in _t.get("models", {}).values()]
        if _n and max(_n) < 35:
            stale.append(f"{a.transfer}: at most {max(_n)} triples per model; the "
                         f"c* run is a complete 35-triple design")
    if stale and not a.allow_stale:
        sys.exit("STALE INPUTS — these predate the coherence-edge protocol:\n  "
                 + "\n  ".join(stale) + "\nUse the c* files, or --allow-stale.")
    embed2 = json.load(open(a.embed2, encoding="utf-8")) if a.embed2 else None
    tr = json.load(open(a.transfer, encoding="utf-8")) if a.transfer else None
    val = json.load(open(a.validation, encoding="utf-8")) if a.validation else None
    probes = json.load(open(a.probes, encoding="utf-8"))
    labels = load_labels(a.labels)

    names = list(embed["conditions"]["dedup"]["POOLED"]["knobs"])
    dirs, _ = probe_dirs(a.vecs, a.layer, labels, probes, names)

    boot = json.load(open(a.crossalign, encoding="utf-8")) if a.crossalign else None
    dm, n = fig1(embed, a.outdir, boot)
    print(f"fig1  cross-alignment, diagonal max {dm}/{n}")
    G, iso, ks = fig2(dirs, a.outdir)
    print(f"fig2  basis geometry, cond {np.linalg.cond(G):.2f}")
    verified = (json.load(open(a.verified, encoding="utf-8"))
                if a.verified else None)
    fig3(embed, embed2, a.outdir, verified)
    print("fig3  steering effects with CIs")
    if tr:
        fig4(from_independent(a.transfer_indep) if a.transfer_indep else tr,
             a.outdir)
        print("fig4  simultaneous control")
        fig5(tr, a.outdir)
        print("fig5  steering magnitude")
        w = fig6(tr, a.outdir)
        print(f"fig6  honest spread ({w:.0%} of cases below 5% pole fraction)")
    if a.linear_baseline:
        fig7(json.load(open(a.linear_baseline, encoding="utf-8")), a.outdir)
        print("fig7  linear baseline against observation")
    pn = json.load(open(a.paper_numbers, encoding="utf-8")) if a.paper_numbers else None
    tables(G, iso, ks, val, tr, a.outdir, pn)
    print(f"tables -> {a.outdir}/table1_gram.md, table2_validation.md, "
          f"table3_entanglement.md")


if __name__ == "__main__":
    main()
