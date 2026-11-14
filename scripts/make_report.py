"""Build docs/RESULTS.md and docs/figures/*.png from the CSV/JSON files in results/.

Every number in RESULTS.md is computed here from saved experiment outputs.
Run after exp_image.py, exp_text.py and exp_selection.py.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

R, D = "results", "docs"
FIG = os.path.join(D, "figures")
# Reference categorical palette, slots 1-3 in fixed order; recessive ink for text/grid.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
METHOD_COLOR = {"lsbr": BLUE, "lsbm": ORANGE, "hill": AQUA}
METHOD_NAME = {"lsbr": "LSB replacement", "lsbm": "LSB matching", "hill": "HILL + STC"}


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def auc_ci(neg, pos, n=1000, seed=0):
    neg, pos = np.asarray(neg), np.asarray(pos)
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    a = roc_auc_score(y, np.r_[neg, pos])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        nn, pp = rng.choice(neg, len(neg)), rng.choice(pos, len(pos))
        bs.append(roc_auc_score(y, np.r_[nn, pp]))
    return a, np.quantile(bs, 0.025), np.quantile(bs, 0.975)


def fmt_ci(a, lo, hi):
    return f"{a:.2f} [{lo:.2f}, {hi:.2f}]"


def operating_points(neg, pos):
    """P_E and detection rate at a 5% false-alarm rate (higher score = more suspicious).

    P_E = min over thresholds of (P_FA + P_MD) / 2, the usual steganalysis error:
    0.5 = the detector is guessing, 0 = perfect. TPR@5%FPR = share of stego caught
    when the threshold lets through at most 5% of clean items as false alarms.
    """
    neg, pos = np.sort(np.asarray(neg, float)), np.asarray(pos, float)
    thr = np.unique(np.r_[neg, pos, np.inf])
    p_fa = np.array([(neg >= t).mean() for t in thr])
    p_md = np.array([(pos < t).mean() for t in thr])
    pe = float(np.min((p_fa + p_md) / 2))
    t5 = np.quantile(neg, 0.95, method="higher")
    tpr5 = float((pos > t5).mean())
    return pe, tpr5


def image_section(lines):
    img = pd.read_csv(os.path.join(R, "image_measurements.csv"))
    ref = pd.read_csv(os.path.join(R, "image_measurements_refused.csv"))
    st = img[img["method"] != "none"]
    lines += ["## 1. Image channel", ""]
    n_cov = (img["method"] == "none").sum()
    lines.append(f"{n_cov} cover crops (256×256 RGB): "
                 f"{(img[img.method=='none'].source=='A_kodak').sum()} from Kodak (source A), "
                 f"{(img[img.method=='none'].source=='B_skimage').sum()} from scikit-image photos (source B). "
                 f"{len(st)} stego images were embedded and decoded; "
                 f"{len(ref)} attempts were refused because the payload exceeded that cover's capacity "
                 f"(all HILL + STC on saturated or flat crops).")
    lines.append("")
    lines.append(f"**Exact recovery: {int(st['recovered'].sum())} / {len(st)} "
                 f"({100*st['recovered'].mean():.1f}%).**")
    lines.append("")
    q = st.groupby(["method", "rate"]).agg(psnr=("psnr_db", "mean"), ssim=("ssim", "mean"),
                                           changed=("change_rate", "mean"), n=("psnr_db", "size")).reset_index()
    lines += ["| Method | Rate (bits/value) | n | PSNR (dB) | SSIM | Values changed |", "| --- | --- | --- | --- | --- | --- |"]
    for _, r in q.iterrows():
        lines.append(f"| {METHOD_NAME[r.method]} | {r.rate:g} | {r.n} | {r.psnr:.1f} | {r.ssim:.4f} | {100*r.changed:.2f}% |")
    lines.append("")

    clean = img[img["method"] == "none"]
    rows = []
    for det, col in [("SPA (no training)", "spa"), ("EfficientNet-LSBR (pretrained)", "effnet_lsbr"),
                     ("EfficientNet-LSBM (pretrained)", "effnet_lsbm"), ("EfficientNet-HILL (pretrained)", "effnet_hill")]:
        for src in ("A_kodak", "B_skimage"):
            for m in ("lsbr", "lsbm", "hill"):
                for rate in (0.05, 0.1, 0.2, 0.4):
                    s = st[(st.method == m) & np.isclose(st.rate, rate) & (st.source == src)][col]
                    c = clean[clean.source == src][col]
                    if len(s) >= 5:
                        a, lo, hi = auc_ci(c, s)
                        rows.append({"detector": det, "col": col, "source": src, "method": m, "rate": rate,
                                     "auc": a, "lo": lo, "hi": hi, "n": len(s)})
    auc = pd.DataFrame(rows)
    auc.to_csv(os.path.join(R, "image_auc.csv"), index=False)

    lines += ["### Detectability (ROC-AUC, clean vs stego; 0.5 = chance, 1.0 = perfect)", "",
              "95% bootstrap intervals in brackets. Source A = Kodak.", ""]
    lines += ["| Detector | Method | 0.05 | 0.1 | 0.2 | 0.4 |", "| --- | --- | --- | --- | --- | --- |"]
    for det in auc.detector.unique():
        for m in ("lsbr", "lsbm", "hill"):
            g = auc[(auc.detector == det) & (auc.method == m) & (auc.source == "A_kodak")].set_index("rate")
            cells = [fmt_ci(g.loc[r, "auc"], g.loc[r, "lo"], g.loc[r, "hi"]) if r in g.index else "—" for r in (0.05, 0.1, 0.2, 0.4)]
            lines.append(f"| {det} | {METHOD_NAME[m]} | " + " | ".join(cells) + " |")
    lines.append("")

    # Figure: AUC vs rate per method, for SPA and EfficientNet-HILL, source A.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True, facecolor=SURF)
    for ax, (det, col) in zip(axes, [("SPA (training-free)", "spa"), ("EfficientNet-HILL (pretrained, mismatched source)", "effnet_hill")]):
        for m in ("lsbr", "lsbm", "hill"):
            g = auc[(auc.col == col) & (auc.method == m) & (auc.source == "A_kodak")].sort_values("rate")
            ax.plot(g.rate, g.auc, color=METHOD_COLOR[m], lw=2, marker="o", ms=5, label=METHOD_NAME[m])
            ax.fill_between(g.rate, g.lo, g.hi, color=METHOD_COLOR[m], alpha=0.12, lw=0)
        ax.axhline(0.5, color=INK2, lw=1, ls="--")
        ax.text(0.41, 0.51, "chance", color=INK2, fontsize=8, ha="right", va="bottom")
        ax.set_ylim(0.3, 1.02)
        style(ax, det, "payload (bits per colour value)", "ROC-AUC" if col == "spa" else None)
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "image_auc.png"), dpi=160)
    plt.close(fig)
    lines += ["![Image detectability](figures/image_auc.png)", ""]
    lines += ["**Same detectors on source B (scikit-image photos):**", "",
              "| Detector | Method | 0.05 | 0.1 | 0.2 | 0.4 |", "| --- | --- | --- | --- | --- | --- |"]
    for det in auc.detector.unique():
        for m in ("lsbr", "lsbm", "hill"):
            g = auc[(auc.detector == det) & (auc.method == m) & (auc.source == "B_skimage")].set_index("rate")
            cells = [fmt_ci(g.loc[r, "auc"], g.loc[r, "lo"], g.loc[r, "hi"]) if r in g.index else "—" for r in (0.05, 0.1, 0.2, 0.4)]
            lines.append(f"| {det} | {METHOD_NAME[m]} | " + " | ".join(cells) + " |")
    lines.append("")
    op_rows = []
    for det, col in [("SPA", "spa"), ("EfficientNet-HILL", "effnet_hill")]:
        for src in ("A_kodak", "B_skimage"):
            for m in ("lsbr", "hill"):
                for rate in (0.05, 0.1, 0.2):
                    sv = st[(st.method == m) & np.isclose(st.rate, rate) & (st.source == src)][col]
                    cv = clean[clean.source == src][col]
                    if len(sv) >= 5:
                        pe, tpr5 = operating_points(cv, sv)
                        op_rows.append({"detector": det, "source": src, "method": m, "rate": rate, "pe": pe, "tpr5": tpr5})
    op = pd.DataFrame(op_rows)
    op.to_csv(os.path.join(R, "image_operating_points.csv"), index=False)
    lines += ["**Operating points.** P_E = lowest achievable average error (0.5 = guessing). "
              "Caught @5% FA = share of stego images caught when at most 5% of clean images are falsely flagged.", "",
              "| Detector | Source | Method | P_E (0.05 / 0.1 / 0.2) | Caught @5% FA (0.05 / 0.1 / 0.2) |",
              "| --- | --- | --- | --- | --- |"]
    for (det, src, m), g in op.groupby(["detector", "source", "method"], sort=False):
        g = g.set_index("rate")
        pes = " / ".join(f"{g.loc[r, 'pe']:.2f}" if r in g.index else "—" for r in (0.05, 0.1, 0.2))
        tprs = " / ".join(f"{100 * g.loc[r, 'tpr5']:.0f}%" if r in g.index else "—" for r in (0.05, 0.1, 0.2))
        lines.append(f"| {det} | {'Kodak' if src == 'A_kodak' else 'scikit-image'} | {METHOD_NAME[m]} | {pes} | {tprs} |")
    lines.append("")
    # Warden A's image eye: SPAM + logistic regression, out-of-fold by photo (same as selection/carrier.py).
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, "src")
    from cmstego.selection.carrier import photo_group
    spam = np.load(os.path.join(R, "image_measurements_spam.npy"))
    img["group"] = [photo_group(a, b) for a, b in zip(img["source"], img["cover"])]
    keep = img["method"].isin(["none", "hill"]).values
    sub = img[keep].reset_index(drop=True)
    yy = (sub["method"] == "hill").astype(int).values
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))
    sub["spam"] = cross_val_predict(clf, spam[keep], yy, groups=sub["group"], cv=GroupKFold(5), method="predict_proba")[:, 1]
    cells = []
    for rate in (0.05, 0.1, 0.2):
        a, lo, hi = auc_ci(sub[sub.method == "none"]["spam"], sub[(sub.method == "hill") & np.isclose(sub.rate, rate)]["spam"])
        cells.append(f"{rate:g}: {fmt_ci(a, lo, hi)}")
    lines += ["**Warden A's image detector** (SPAM features + logistic regression trained on our covers, scored out-of-fold "
              "so no photo is scored by a model that saw it) vs HILL + STC, both sources pooled — " + "; ".join(cells) + ".", ""]

    # Cover-source effect on the pretrained HILL detector's raw scores.
    sh = clean.groupby("source")["effnet_hill"].agg(["mean", "std", "size"])
    lines.append("**Pretrained detectors are miscalibrated on our covers.** Mean EfficientNet-HILL score on *clean* "
                 f"covers: Kodak {sh.loc['A_kodak','mean']:.2f} (sd {sh.loc['A_kodak','std']:.2f}), "
                 f"scikit-image {sh.loc['B_skimage','mean']:.2f} (sd {sh.loc['B_skimage','std']:.2f}). "
                 "A score of 0.5 would be the decision boundary, so taken at face value the detector calls most "
                 "innocent photos stego. The warden therefore uses p-values against clean covers, not raw scores.")
    lines.append("")
    return auc


VARIANTS = [("clear", "subtext-codec framing as-is (length field in clear), filter guard"),
            ("filter", "+ nonce and whitening, filter guard (final system)"),
            ("retry", "+ nonce and whitening, retry guard (OD-Stega style)"),
            ("nlfix", "filter + line-break fix (blank-line tokens banned, line breaks shown doubled)")]


def text_section(lines, rdir=R, title="## 2. Text channel", fig_name="text_detectability.png", timing_note=True):
    present = [v for v, _ in VARIANTS if os.path.exists(os.path.join(rdir, f"text_{v}_measurements.csv"))]
    lines += [title, "",
              f"{len(present)} variant(s) of the text coder were run on the same 12 prompts with 24-byte random payloads. "
              "Covers are ordinary top-64 samples from the same model, cut to the same token count.", ""]
    meas_rows, auc_rows = [], []
    for v, desc in VARIANTS:
        mp = os.path.join(rdir, f"text_{v}_measurements.csv")
        if not os.path.exists(mp):
            continue
        m = pd.read_csv(mp)
        held = pd.read_csv(os.path.join(rdir, f"text_{v}_heldout_scores.csv"))
        sp = os.path.join(rdir, f"text_{v}_heldout_scores_style.csv")
        if os.path.exists(sp):
            held = pd.concat([held, pd.read_csv(sp)], ignore_index=True)
        for C, g in m.groupby("C"):
            row = {"variant": v, "desc": desc, "C": C, "n": len(g), "rec": g.recovered.sum(),
                   "bpt": g.bits_per_token.mean(), "kl": g.kl_bits_per_token.mean(),
                   "klc": g["kl_vs_topk_bits_per_token"].mean() if "kl_vs_topk_bits_per_token" in g else np.nan,
                   "att": g["attempts"].mean() if "attempts" in g else 1.0,
                   "enc": g.encode_s.median(), "dec": g.decode_s.median()}
            for det in ("ngram", "likelihood", "style"):
                h = held[(held.C == C) & (held.detector == det)]
                if h.empty:
                    continue
                a, lo, hi = auc_ci(h.cover_score, h.stego_score)
                pe, tpr5 = operating_points(h.cover_score, h.stego_score)
                row[det] = (a, lo, hi)
                row[det + "_op"] = (pe, tpr5)
                row["pairs"] = len(h)
                auc_rows.append({"variant": v, "C": C, "detector": det, "auc": a, "lo": lo, "hi": hi,
                                 "pe": pe, "tpr5": tpr5})
            meas_rows.append(row)
    if not meas_rows:
        return
    tot = sum(r["n"] for r in meas_rows)
    rec = sum(r["rec"] for r in meas_rows)
    fails = 0
    for v, _ in VARIANTS:
        fp = os.path.join(rdir, f"text_{v}_encode_failures.csv")
        if os.path.exists(fp):
            fails += len(pd.read_csv(fp))
    lines.append(f"**Exact recovery: {rec} / {tot} stego-texts ({100*rec/tot:.1f}%) across all variants.** "
                 + (f"{fails} further messages could not be encoded at all; they were reported as failures, never sent."
                    if fails else "No message failed to encode."))
    lines.append("")
    if timing_note:
        lines.append("*Timings were measured while several experiments shared a 2-core machine; the filter C = 0.1 and "
                     "retry runs overlapped with others and are not comparable speed measurements.*")
    else:
        lines.append("*Timings: one experiment at a time on a laptop CPU (no GPU); encode includes the per-step "
                     "tokenisation filter, decode is a single pass.*")
    lines.append("")
    lines += ["| Variant | C | n | Payload bits / token | KL vs full model (bits/token) | KL vs top-k sampling (bits/token) | Encodes per message | Encode s (median) | Decode s (median) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in meas_rows:
        klc = "—" if np.isnan(r["klc"]) else f"{r['klc']:.4f}"
        lines.append(f"| {r['variant']} | {r['C']:g} | {r['n']} | {r['bpt']:.2f} | {r['kl']:.4f} | {klc} | {r['att']:.1f} | {r['enc']:.2f} | {r['dec']:.2f} |")
    lines.append("")
    has_style = any("style" in r for r in meas_rows)
    dets = [("ngram", "N-gram detector (black box)"), ("likelihood", "Likelihood detector (white box)")]
    if has_style:
        dets.append(("style", "Style detector (black box)"))
    lines += ["### Detectability (held-out ROC-AUC; 0.5 = chance)", "",
              "| Variant | C | " + " | ".join(d[1] for d in dets) + " | Test pairs |",
              "| --- | --- | " + " | ".join("---" for _ in dets) + " | --- |"]
    for r in meas_rows:
        cells = [fmt_ci(*r[d]) if d in r else "—" for d, _ in dets]
        lines.append(f"| {r['variant']} | {r['C']:g} | " + " | ".join(cells) + f" | {r['pairs']} |")
    lines.append("")
    lines += ["| Variant | C | " + " | ".join(f"{d[1].split(' (')[0]}: P_E / caught @5% FA" for d in dets) + " |",
              "| --- | --- | " + " | ".join("---" for _ in dets) + " |"]
    for r in meas_rows:
        cells = [f"{r[d + '_op'][0]:.2f} / {100 * r[d + '_op'][1]:.0f}%" if d + "_op" in r else "—" for d, _ in dets]
        lines.append(f"| {r['variant']} | {r['C']:g} | " + " | ".join(cells) + " |")
    lines.append("")
    if has_style:
        lines += ["The style detector (word and sentence lengths, vocabulary richness, punctuation, line breaks; "
                  "no language model) was added later and scored on the same saved texts with the same "
                  "train/test split (`scripts/add_style_detector.py`).", ""]
    # Blank lines: the visible form of the tokenisation fingerprint (computed from the saved texts).
    bl_rows = []
    for v, _ in VARIANTS:
        sp = os.path.join(rdir, f"text_{v}_samples.jsonl")
        if not os.path.exists(sp):
            continue
        with open(sp, encoding="utf-8") as fh:
            smp = [json.loads(line) for line in fh]
        for C in sorted({x["C"] for x in smp}):
            cell = [x for x in smp if x["C"] == C]
            bl_rows.append((v, C, len(cell), sum("\n\n" in x["cover"] for x in cell), sum("\n\n" in x["stego"] for x in cell)))
    if bl_rows:
        lines += ["**Blank lines.** Share of texts containing at least one blank line (two line breaks in a row):", "",
                  "| Variant | C | Ordinary texts | Stego-texts |", "| --- | --- | --- | --- |"]
        for v, C, n, bc, bs in bl_rows:
            lines.append(f"| {v} | {C:g} | {bc}/{n} ({100*bc/n:.0f}%) | {bs}/{n} ({100*bs/n:.0f}%) |")
        lines.append("")
    pd.DataFrame(auc_rows).to_csv(os.path.join(rdir, "text_metrics.csv"), index=False)
    lines += ["Variants: " + "; ".join(f"**{v}** = {d}" for v, d in VARIANTS) + ".", ""]

    a = pd.DataFrame(auc_rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True, facecolor=SURF)
    colors = {"clear": ORANGE, "filter": AQUA, "retry": BLUE, "nlfix": "#8a5cd1"}
    for ax, det, title in ((axes[0], "ngram", "N-gram detector (black box)"), (axes[1], "likelihood", "Likelihood detector (white box)")):
        vs = [v for v, _ in VARIANTS if v in set(a.variant)]
        Cs = sorted(a.C.unique())
        width = 0.8 / len(vs)
        for i, v in enumerate(vs):
            g = a[(a.variant == v) & (a.detector == det)].set_index("C").reindex(Cs)
            x = np.arange(len(Cs)) + (i - (len(vs) - 1) / 2) * width
            ax.bar(x, g.auc, width=width * 0.9, color=colors[v], label=v)
            ax.errorbar(x, g.auc, yerr=[g.auc - g.lo, g.hi - g.auc], fmt="none", ecolor=INK2, lw=1, capsize=2)
        ax.axhline(0.5, color=INK2, lw=1, ls="--")
        ax.set_xticks(range(len(Cs)), [f"C = {c:g}" for c in Cs])
        ax.set_ylim(0, 1.05)
        style(ax, title, None, "held-out ROC-AUC" if det == "ngram" else None)
    handles, labels = axes[0].get_legend_handles_labels()
    short = {"clear": "clear (length field unscrambled)", "filter": "filter (+ nonce, filter guard)",
             "retry": "retry (+ nonce, retry guard)", "nlfix": "nlfix (+ line-break fix)"}
    fig.legend(handles, [short[l] for l in labels],
               loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(FIG, fig_name), dpi=160)
    plt.close(fig)
    lines += [f"![Text detectability by variant](figures/{fig_name})", ""]
    fp = os.path.join(rdir, "text_fingerprint.csv")
    if os.path.exists(fp):
        f = pd.read_csv(fp)
        counts = f[f.feature.str.startswith("texts with")].set_index("feature")
        top = f[~f.feature.str.startswith("texts with")].sort_values("single_feature_auc", ascending=False).iloc[0]
        cnt = counts.loc["texts with >=1 token outside top-k"]
        lines += ["### What the white-box detector sees at C = 0 (filter variant)", "",
                  f"The single most informative feature is *{top.feature}* (single-feature AUC {top.single_feature_auc:.2f}, "
                  "close to the full detector). When the text is re-read by the tokenizer, "
                  f"{int(cnt.cover_mean)} of {int(cnt.single_feature_auc)} ordinary samples contain at least one token outside the model's top 64, "
                  f"versus {int(cnt.stego_mean)} of {int(cnt.single_feature_auc)} stego-texts. Ordinary sampling sometimes picks token pairs "
                  "that merge into a different tokenisation; exact-decoding stego-text never can"
                  + (", under either guard. This is why the retry guard did not lower detectability."
                     if os.path.exists(os.path.join(rdir, "text_retry_measurements.csv")) else "."), ""]
        if "texts with a blank line" in counts.index:
            bl = counts.loc["texts with a blank line"]
            lines += [f"Texts with a blank line in this run: {int(bl.cover_mean)} of {int(bl.single_feature_auc)} ordinary, "
                      f"{int(bl.stego_mean)} of {int(bl.single_feature_auc)} stego.", ""]
    fn = os.path.join(rdir, "text_fingerprint_nlfix.csv")
    if os.path.exists(fn):
        f2 = pd.read_csv(fn).set_index("feature")
        c2 = f2.loc["texts with >=1 token outside top-k"]
        lines += ["**With the line-break fix (nlfix, C = 0):** texts with at least one token outside the top 64 when "
                  f"re-read: {int(c2.cover_mean)} of {int(c2.single_feature_auc)} ordinary vs {int(c2.stego_mean)} of "
                  f"{int(c2.single_feature_auc)} stego. Single-feature AUC of the share of such tokens: "
                  f"{f2.loc['share of tokens outside top-k', 'single_feature_auc']:.2f}.", ""]


NAMES = {"none": "No selection", "random": "Random", "heuristic": "Heuristic", "grid-A": "Grid search (vs A)",
         "ppo": "PPO v1 (slot-index policy)", "ppo-score": "PPO v2 (scoring policy)",
         "grid-ens": "Grid search (vs ensemble)", "ppo-ens": "PPO (scoring policy, vs ensemble)"}


def selection_section(lines, rdir=R, title="## 3. Carrier selection with a pooled warden", fig_name="selection.png"):
    s = json.load(open(os.path.join(rdir, "selection_summary.json"), encoding="utf-8"))
    df = pd.DataFrame(s["summary"])
    lines += [title, ""]
    lines.append(f"Text measurements from the `{s.get('text_variant','text_filter_').strip('_').replace('text_','')}` variant (C ∈ {s['C_values']}). "
                 f"{s['episodes_per_split']} held-out messages per test set; each has K = {s['K_covers']} candidate "
                 f"photos and M = {s['M_prompts']} candidate prompts, message sizes {s['message_bytes']} bytes. "
                 f"PPO trained for {s['ppo_steps']:,} steps ({s['ppo_train_seconds']:.0f} s on CPU) against warden A only. "
                 "Lower is better for the sender. Flag rate = share of pairs the warden flags at a 5% false-alarm rate; "
                 "a failed send (photo too small for the payload) counts as flagged.")
    lines.append("")
    names = NAMES
    order = [o for o in NAMES if o in set(df.strategy)]
    for split, label in (("test", "Unseen Kodak photos + unseen prompts"), ("test_B", "Unseen image source (scikit-image) + unseen prompts")):
        lines += [f"### {label}", "",
                  "| Strategy | Carriers evaluated / message | Suspicion, warden A | Flag rate, warden A | Suspicion, warden B (never trained on) | Flag rate, warden B |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for st in order:
            a = df[(df.split == split) & (df.strategy == st) & (df.warden == "A")].iloc[0]
            b = df[(df.split == split) & (df.strategy == st) & (df.warden == "B")].iloc[0]
            lines.append(f"| {names[st]} | {a.evaluations_per_message:.0f} | {fmt_ci(a.mean_suspicion, *a.susp_ci)} | "
                         f"{fmt_ci(a.flag_rate, *a.flag_ci)} | {fmt_ci(b.mean_suspicion, *b.susp_ci)} | {fmt_ci(b.flag_rate, *b.flag_ci)} |")
        lines.append("")

    fig, axes = plt.subplots(1, 2, figsize=(9, 0.5 * len(order) + 0.6), sharey=True, facecolor=SURF)
    for ax, w in zip(axes, ("A", "B")):
        g = df[(df.split == "test") & (df.warden == w)].set_index("strategy").loc[order]
        y = np.arange(len(order))[::-1]  # first strategy at the top
        err = np.array([[r[0] for r in g.flag_ci], [r[1] for r in g.flag_ci]])
        ax.barh(y, g.flag_rate, color=BLUE, height=0.55)
        ax.errorbar(g.flag_rate, y, xerr=np.abs(err - g.flag_rate.values), fmt="none", ecolor=INK2, lw=1, capsize=3)
        for yi, v, hi in zip(y, g.flag_rate, err[1]):
            ax.text(max(v, hi) + 0.03, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, [names[o] for o in order])
        ax.set_xlim(0, 1.05)
        style(ax, f"Warden {w}" + (" (training reward)" if w == "A" else " (never seen in training)"), "share of pairs flagged")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fig_name), dpi=160)
    plt.close(fig)
    lines += [f"![Carrier selection](figures/{fig_name})", ""]


DET_NAMES = {"spam": "SPAM+LR", "effnet_hill": "EfficientNet-HILL", "effnet_lsbm": "EfficientNet-LSBM",
             "effnet_lsbr": "EfficientNet-LSBR", "spa": "SPA", "ngram": "n-gram", "likelihood": "likelihood",
             "style": "style"}


def ensemble_section(lines, rdir):
    s = json.load(open(os.path.join(rdir, "selection_ensemble_summary.json"), encoding="utf-8"))
    df = pd.DataFrame(s["summary"])
    w = s["wardens"]
    lines += ["## 4. Training against several wardens (GPT-2 text measurements)", "",
              "Overfitting to one warden was the main weakness found in section 3, so the same selection problem was "
              "repeated with an **ensemble** of wardens during training: "
              + "; ".join(f"{k} = {DET_NAMES[a]} (image) + {DET_NAMES[b]} (text)" for k, (a, b) in w.items()) +
              f". The ensemble (ENS) is the worst case over {', '.join(s['ensemble'])}: the highest suspicion, flagged if "
              "any one flags. It shares no detector with warden B, which is never used in training. PPO here is the "
              f"scoring policy, {s['ppo_steps']:,} steps. Baselines are recomputed in this run; PPO policies are retrained.", ""]
    order = [o for o in ("none", "heuristic", "random", "grid-A", "grid-ens", "ppo-score", "ppo-ens", "ppo") if o in set(df.strategy)]
    names = dict(NAMES)
    names["ppo-score"] = "PPO (scoring policy, vs A)"
    for split, label in (("test", "Unseen Kodak photos + unseen prompts"), ("test_B", "Unseen image source + unseen prompts")):
        lines += [f"### {label}", "", "| Strategy | Flag rate, A | Flag rate, ENS (training ensemble) | Flag rate, B (never trained on) | Suspicion, B |",
                  "| --- | --- | --- | --- | --- |"]
        for st in order:
            r = {ww: df[(df.split == split) & (df.strategy == st) & (df.warden == ww)].iloc[0] for ww in ("A", "ENS", "B")}
            lines.append(f"| {names.get(st, st)} | {fmt_ci(r['A'].flag_rate, *r['A'].flag_ci)} | "
                         f"{fmt_ci(r['ENS'].flag_rate, *r['ENS'].flag_ci)} | {fmt_ci(r['B'].flag_rate, *r['B'].flag_ci)} | "
                         f"{fmt_ci(r['B'].mean_suspicion, *r['B'].susp_ci)} |")
        lines.append("")


def split_section(lines, rdir):
    s = json.load(open(os.path.join(rdir, "split_summary.json"), encoding="utf-8"))
    df = pd.DataFrame(s["summary"])
    toks = ", ".join(f"{k} bytes ≈ {v:.0f} tokens" for k, v in s["mean_tokens"].items())
    lines += ["## 5. Choosing the split (lock-box package between text and image)", "",
              f"Text fragment sizes measured on GPT-2 at C = 0: {toks}. Secrets of {s['message_bytes']} bytes, "
              f"{s['episodes_per_split']} held-out messages per test set. Each strategy picks a cover, a prompt and a text size; "
              "the image carries the rest of the package.", ""]
    # How fine-grained can a text p-value be at each size?
    res = []
    for size, folder in s["size_dirs"].items():
        h = pd.read_csv(os.path.join(folder.replace("\\", "/"), "text_filter_heldout_scores.csv"))
        n = int((h[np.isclose(h.C, 0)].detector == "likelihood").sum())
        res.append((int(size), n, 1 / (n + 1)))
    lines += ["**Read the flag rates with care.** A text p-value is computed against the ordinary texts of the same size: "
              + "; ".join(f"{sz} bytes: {n} texts, smallest possible p = {pmin:.3f}" for sz, n, pmin in res)
              + ". The text eye flags only below 0.025, so wherever the smallest possible p is above that, the text can "
              "never be flagged and the flag rate measures the image alone. Mean suspicion (1 − smallest p) is "
              "compared instead; its ceiling is slightly *higher* for the 24-byte size, which biases it against the "
              "small text share, not for it.", ""]
    names = {"min-text": "Smallest text share", "max-text": "Largest text share", "random": "Random",
             "grid-A": "Grid search (vs A)", "grid-ens": "Grid search (vs ensemble)", "ppo-A": "PPO (vs A)",
             "ppo-ens": "PPO (vs ensemble)"}
    ws = [w for w in ("A", "ENS", "B") if w in s["wardens"]]
    for split, label in (("test", "Unseen Kodak photos + unseen prompts"), ("test_B", "Unseen image source + unseen prompts")):
        lines += [f"### {label}", "", "| Strategy | " + " | ".join(f"Suspicion, {w}" for w in ws)
                  + " | Flag rate, B | Text sizes chosen |", "| --- | " + " | ".join("---" for _ in ws) + " | --- | --- |"]
        for st in [n for n in names if n in set(df.strategy)]:
            g = df[(df.split == split) & (df.strategy == st)].set_index("warden")
            share = g.iloc[0]["size_share"]
            share_txt = ", ".join(f"{k} B: {100*v:.0f}%" for k, v in sorted(share.items(), key=lambda kv: int(kv[0])))
            lines.append(f"| {names[st]} | " + " | ".join(fmt_ci(g.loc[w, 'mean_suspicion'], *g.loc[w, 'susp_ci']) for w in ws)
                         + f" | {g.loc['B', 'flag_rate']:.2f} | {share_txt} |")
        lines.append("")
    # Direct evidence: detector AUC by text size (same detectors, C = 0).
    rows = []
    for size, folder in s["size_dirs"].items():
        f = folder.replace("\\", "/")
        h = pd.read_csv(os.path.join(f, "text_filter_heldout_scores.csv"))
        sp = os.path.join(f, "text_filter_heldout_scores_style.csv")
        if os.path.exists(sp):
            h = pd.concat([h, pd.read_csv(sp)], ignore_index=True)
        h = h[np.isclose(h.C, 0)]
        cells = []
        for det in ("ngram", "likelihood", "style"):
            g = h[h.detector == det]
            cells.append(fmt_ci(*auc_ci(g.cover_score, g.stego_score)) if len(g) else "—")
        rows.append(f"| {size} | {s['mean_tokens'][size]:.0f} | " + " | ".join(cells) + f" | {len(h[h.detector == 'ngram'])} |")
    lines += ["**Text detectability by length** (held-out ROC-AUC, C = 0, without the line-break fix):", "",
              "| Text fragment (bytes) | Tokens | N-gram | Likelihood | Style | Test pairs |", "| --- | --- | --- | --- | --- | --- |"]
    lines += rows + [""]


G = "results_gpt2"


def main():
    os.makedirs(FIG, exist_ok=True)
    has_gpt2 = os.path.exists(os.path.join(G, "text_filter_measurements.csv"))
    lines = ["# First-stage results", "",
             "Generated by `scripts/make_report.py` from the files in `results/` and `results_gpt2/`. Every number "
             "below is measured by this repository's code; nothing is quoted from papers. Sample sizes are small.", ""]
    if has_gpt2:
        lines += ["**Language models.** Text results come from two runs: **GPT-2** (124M, the model the system uses; "
                  "`results_gpt2/`, run on a laptop CPU) and an earlier run with a 4-layer offline **toy model** "
                  "(`data/toy-lm`; `results/`), which covers more variants and budgets. Where they overlap they agree "
                  "in direction; the GPT-2 numbers are the ones to quote.", ""]
    image_section(lines)
    if has_gpt2:
        text_section(lines, G, "## 2. Text channel (GPT-2)", "text_detectability_gpt2.png", timing_note=False)
        text_section(lines, R, "## 2b. Text channel (toy model, more variants)", "text_detectability.png")
    else:
        text_section(lines)
    if has_gpt2 and os.path.exists(os.path.join(G, "selection_summary.json")):
        selection_section(lines, G, "## 3. Carrier selection with a pooled warden (GPT-2 text measurements)",
                          "selection_gpt2.png")
        if os.path.exists(os.path.join(R, "selection_summary.json")):
            selection_section(lines, R, "## 3b. Carrier selection (toy-model text measurements, earlier run)",
                              "selection.png")
    elif os.path.exists(os.path.join(R, "selection_summary.json")):
        selection_section(lines)
    if os.path.exists(os.path.join(G, "selection_ensemble_summary.json")):
        ensemble_section(lines, G)
    if os.path.exists(os.path.join(G, "split_summary.json")):
        split_section(lines, G)
    open(os.path.join(D, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote docs/RESULTS.md")


if __name__ == "__main__":
    main()
