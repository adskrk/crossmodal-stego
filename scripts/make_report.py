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
    # Warden A's image eye: SPAM + logistic regression, out-of-fold by photo (same as selection/pilot.py).
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, "src")
    from cmstego.selection.pilot import photo_group
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
            ("retry", "+ nonce and whitening, retry guard (OD-Stega style)")]


def text_section(lines):
    lines += ["## 2. Text channel", "",
              "Three variants of the text coder were run on the same 12 prompts with 24-byte random payloads. "
              "Covers are ordinary top-64 samples from the same model, cut to the same token count.", ""]
    meas_rows, auc_rows = [], []
    for v, desc in VARIANTS:
        mp = os.path.join(R, f"text_{v}_measurements.csv")
        if not os.path.exists(mp):
            continue
        m = pd.read_csv(mp)
        held = pd.read_csv(os.path.join(R, f"text_{v}_heldout_scores.csv"))
        for C, g in m.groupby("C"):
            row = {"variant": v, "desc": desc, "C": C, "n": len(g), "rec": g.recovered.sum(),
                   "bpt": g.bits_per_token.mean(), "kl": g.kl_bits_per_token.mean(),
                   "klc": g["kl_vs_topk_bits_per_token"].mean() if "kl_vs_topk_bits_per_token" in g else np.nan,
                   "att": g["attempts"].mean() if "attempts" in g else 1.0,
                   "enc": g.encode_s.median(), "dec": g.decode_s.median()}
            for det in ("ngram", "likelihood"):
                h = held[(held.C == C) & (held.detector == det)]
                a, lo, hi = auc_ci(h.cover_score, h.stego_score)
                row[det] = (a, lo, hi)
                row["pairs"] = len(h)
                auc_rows.append({"variant": v, "C": C, "detector": det, "auc": a, "lo": lo, "hi": hi})
            meas_rows.append(row)
    if not meas_rows:
        return
    tot = sum(r["n"] for r in meas_rows)
    rec = sum(r["rec"] for r in meas_rows)
    fails = 0
    for v, _ in VARIANTS:
        fp = os.path.join(R, f"text_{v}_encode_failures.csv")
        if os.path.exists(fp):
            fails += len(pd.read_csv(fp))
    lines.append(f"**Exact recovery: {rec} / {tot} stego-texts ({100*rec/tot:.1f}%) across all variants.** "
                 f"{fails} further messages could not be encoded at all (retry guard: no clean re-tokenisation in 64 attempts); "
                 "they were reported as failures, never sent.")
    lines.append("")
    lines.append("*Timings were measured while several experiments shared a 2-core machine; the filter C = 0.1 and "
                 "retry runs overlapped with others and are not comparable speed measurements. Re-measure on your laptop.*")
    lines.append("")
    lines += ["| Variant | C | n | Payload bits / token | KL vs full model (bits/token) | KL vs top-k sampling (bits/token) | Encodes per message | Encode s (median) | Decode s (median) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in meas_rows:
        klc = "—" if np.isnan(r["klc"]) else f"{r['klc']:.4f}"
        lines.append(f"| {r['variant']} | {r['C']:g} | {r['n']} | {r['bpt']:.2f} | {r['kl']:.4f} | {klc} | {r['att']:.1f} | {r['enc']:.2f} | {r['dec']:.2f} |")
    lines.append("")
    lines += ["### Detectability (held-out ROC-AUC; 0.5 = chance)", "",
              "| Variant | C | N-gram detector (black box) | Likelihood detector (white box) | Test pairs |",
              "| --- | --- | --- | --- | --- |"]
    for r in meas_rows:
        lines.append(f"| {r['variant']} | {r['C']:g} | {fmt_ci(*r['ngram'])} | {fmt_ci(*r['likelihood'])} | {r['pairs']} |")
    lines.append("")
    lines += ["Variants: " + "; ".join(f"**{v}** = {d}" for v, d in VARIANTS) + ".", ""]

    a = pd.DataFrame(auc_rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True, facecolor=SURF)
    colors = {"clear": ORANGE, "filter": AQUA, "retry": BLUE}
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
             "retry": "retry (+ nonce, retry guard)"}
    fig.legend(handles, [short[l] for l in labels],
               loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(FIG, "text_detectability.png"), dpi=160)
    plt.close(fig)
    lines += ["![Text detectability by variant](figures/text_detectability.png)", ""]
    fp = os.path.join(R, "text_fingerprint.csv")
    if os.path.exists(fp):
        f = pd.read_csv(fp)
        top = f.iloc[:-1].sort_values("single_feature_auc", ascending=False).iloc[0]
        cnt = f.iloc[-1]
        lines += ["### What the white-box detector sees at C = 0 (filter variant)", "",
                  f"The single most informative feature is *{top.feature}* (single-feature AUC {top.single_feature_auc:.2f}, "
                  "close to the full detector). When the text is re-read by the tokenizer, "
                  f"{int(cnt.cover_mean)} of {int(cnt.single_feature_auc)} ordinary samples contain at least one token outside the model's top 64, "
                  f"versus {int(cnt.stego_mean)} of {int(cnt.single_feature_auc)} stego-texts. Ordinary sampling sometimes picks token pairs "
                  "that merge into a different tokenisation; exact-decoding stego-text never can, under either guard. "
                  "This is why the retry guard did not lower detectability.", ""]


def selection_section(lines):
    s = json.load(open(os.path.join(R, "selection_summary.json")))
    df = pd.DataFrame(s["summary"])
    lines += ["## 3. Carrier selection with a pooled warden", ""]
    lines.append(f"Text measurements from the `{s.get('text_variant','text_filter_').strip('_').replace('text_','')}` variant (C ∈ {s['C_values']}). "
                 f"{s['episodes_per_split']} held-out messages per test set; each has K = {s['K_covers']} candidate "
                 f"photos and M = {s['M_prompts']} candidate prompts, message sizes {s['message_bytes']} bytes. "
                 f"PPO trained for {s['ppo_steps']:,} steps ({s['ppo_train_seconds']:.0f} s on CPU) against warden A only. "
                 "Lower is better for the sender. Flag rate = share of pairs the warden flags at a 5% false-alarm rate; "
                 "a failed send (photo too small for the payload) counts as flagged.")
    lines.append("")
    names = {"none": "No selection", "random": "Random", "heuristic": "Heuristic", "grid-A": "Grid search (vs A)",
             "ppo": "PPO v1 (slot-index policy)", "ppo-score": "PPO v2 (scoring policy)"}
    for split, label in (("test", "Unseen Kodak photos + unseen prompts"), ("test_B", "Unseen image source (scikit-image) + unseen prompts")):
        lines += [f"### {label}", "",
                  "| Strategy | Carriers evaluated / message | Suspicion, warden A | Flag rate, warden A | Suspicion, warden B (never trained on) | Flag rate, warden B |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for st in ("none", "random", "heuristic", "grid-A", "ppo", "ppo-score"):
            a = df[(df.split == split) & (df.strategy == st) & (df.warden == "A")].iloc[0]
            b = df[(df.split == split) & (df.strategy == st) & (df.warden == "B")].iloc[0]
            lines.append(f"| {names[st]} | {a.evaluations_per_message:.0f} | {fmt_ci(a.mean_suspicion, *a.susp_ci)} | "
                         f"{fmt_ci(a.flag_rate, *a.flag_ci)} | {fmt_ci(b.mean_suspicion, *b.susp_ci)} | {fmt_ci(b.flag_rate, *b.flag_ci)} |")
        lines.append("")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, facecolor=SURF)
    order = ["none", "random", "heuristic", "grid-A", "ppo", "ppo-score"]
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
    fig.savefig(os.path.join(FIG, "selection.png"), dpi=160)
    plt.close(fig)
    lines += ["![Carrier selection](figures/selection.png)", ""]


def main():
    os.makedirs(FIG, exist_ok=True)
    lines = ["# Pilot results", "",
             "Generated by `scripts/make_report.py` from the files in `results/`. Every number below is measured "
             "by this repository's code; nothing is quoted from papers. Sample sizes are pilot-scale.", "",
             "**Language model used for these numbers:** a 4-layer offline toy GPT-2 (`data/toy-lm`, built by "
             "`scripts/make_toy_lm.py`), because the build machine could not reach Hugging Face. Text-channel "
             "numbers will differ with GPT-2; re-run `exp_text.py --model gpt2` to replace them.", ""]
    image_section(lines)
    text_section(lines)
    if os.path.exists(os.path.join(R, "selection_summary.json")):
        selection_section(lines)
    open(os.path.join(D, "RESULTS.md"), "w").write("\n".join(lines) + "\n")
    print("wrote docs/RESULTS.md")


if __name__ == "__main__":
    main()
