"""Figures (PDF and PNG) and LaTeX tables for the paper, from the same saved outputs as
analysis/final_analysis.py. Re-run after any new result; figures that need the re-evaluation of saved
models (analysis/eval/*.json) are drawn when those files exist.

    python analysis/make_paper_assets.py   ->  paper_assets/figures/*.pdf|png, paper_assets/tables/*.tex
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_analysis import HEADS, LANG_NAME, LANGS, MB, NAME, ROOT, by_seed, load_runs, paired, pick, val_curve  # noqa: E402

OUT = ROOT / "paper_assets"
FIG, TEX = OUT / "figures", OUT / "tables"
# Categorical palette (dataviz reference, light mode), validated for these six slots in this order; three fall
# below 3:1 contrast, so every head also has its own marker and is labelled directly or in a legend.
COLOR = dict(zip(HEADS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]))
MARK = dict(zip(HEADS, ["o", "s", "^", "D", "v", "P"]))
SHORT = {"linear": "Linear", "cnn": "CNN", "bilstm": "BiLSTM", "cnn_bilstm": "CNN-BiLSTM", "mlp": "MLP", "attention": "Attention"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6, "savefig.dpi": 300,
    "pdf.fonttype": 42,
})
ONE, TWO = 3.4, 7.0   # column widths in inches


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


def mean_curve(rs):
    curves = [val_curve(r["dir"]) for r in rs]
    rounds = sorted(set.intersection(*(set(c) for c in curves))) if curves else []
    return rounds, [st.mean(c[k] for c in curves) for k in rounds]


# ---------------------------------------------------------------- figures

def fig_cost(runs):
    """Q4: test mean against MB each client sends per round (E2 FedAvg)."""
    fig, ax = plt.subplots(figsize=(ONE, 2.4))
    for h in HEADS:
        v = [r["test_best_mean_f1"] for r in pick(runs, exp="E2", method="fedavg", head=h)]
        ax.errorbar(MB[h], st.mean(v), yerr=st.stdev(v), fmt=MARK[h], color=COLOR[h], ms=6, capsize=2,
                    mec="white", mew=0.8, elinewidth=1, label=SHORT[h])
    lin = st.mean(r["test_best_mean_f1"] for r in pick(runs, exp="E2", method="fedavg", head="linear"))
    ax.axhspan(lin - 0.01, lin + 0.01, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.text(1.05, lin + 0.0085, "linear ± 0.01 (run-to-run spread)", color=MUTED, fontsize=6, va="top")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10])
    ax.set_xticklabels(["1", "2", "5", "10"])
    ax.set_xlabel("MB sent per client per round (log scale)")
    ax.set_ylabel("Test macro-F1 (mean of 4 languages)")
    ax.set_ylim(lin - 0.02, lin + 0.02)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=0.8)
    save(fig, "f1_score_vs_upload")


def fig_curves(runs):
    """Validation macro-F1 over rounds, E2 FedAvg, mean over seeds."""
    fig, ax = plt.subplots(figsize=(ONE, 2.4))
    for h in HEADS:
        x, y = mean_curve(pick(runs, exp="E2", method="fedavg", head=h))
        ax.plot(x, y, color=COLOR[h], marker=MARK[h], markevery=8, ms=4, mec="white", mew=0.5, label=SHORT[h])
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation macro-F1 (mean of 4)")
    ax.set_ylim(0.60, 0.72)
    ax.legend(ncol=3, frameon=False, loc="lower right", handletextpad=0.3, columnspacing=0.8)
    save(fig, "f2_validation_curves_e2")


SETTINGS = [("Even labels\nFedAvg", dict(exp="E2", method="fedavg")),
            ("Marathi by domain\nFedAvg", dict(exp="E4", method="fedavg")),
            ("Label skew α 0.5\nFedAvg", dict(exp="E3", setting="dir05", method="fedavg")),
            ("Label skew α 0.1\nFedAvg", dict(exp="E3", setting="dir01", method="fedavg")),
            ("Label skew α 0.1\nFedProx", dict(exp="E3", setting="dir01", method="fedprox")),
            ("Label skew α 0.1\nFedAdam", dict(exp="E3", setting="dir01", method="fedadam"))]


def fig_settings(runs):
    """Each head in each setting: per-seed scores (small) and their mean (large)."""
    fig, ax = plt.subplots(figsize=(TWO, 2.8))
    off = {h: (i - 2.5) * 0.11 for i, h in enumerate(HEADS)}
    for j, (label, kw) in enumerate(SETTINGS):
        for h in HEADS:
            v = [r["test_best_mean_f1"] for r in pick(runs, head=h, **kw)]
            ax.scatter([j + off[h]] * len(v), v, color=COLOR[h], marker=MARK[h], s=10, alpha=0.45, lw=0)
            ax.scatter(j + off[h], st.mean(v), color=COLOR[h], marker=MARK[h], s=34, edgecolor="white", lw=0.8,
                       label=SHORT[h] if j == 0 else None, zorder=3)
    ax.set_xticks(range(len(SETTINGS)))
    ax.set_xticklabels([s[0] for s in SETTINGS])
    ax.set_ylabel("Test macro-F1 (mean of 4 languages)")
    ax.set_ylim(0.50, 0.72)
    ax.text(5, 0.575, "linear and CNN means pulled\ndown by two collapsed seed-2\nruns (0.23, 0.25; off the axis)",
            ha="center", fontsize=6, color=MUTED)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.3)
    save(fig, "f3_heads_across_settings")


def fig_clients(runs):
    """Experiment 5: number of clients and partial participation (3 heads)."""
    fig, ax = plt.subplots(figsize=(ONE, 2.3))
    xs = ["6", "12", "24", "24, half\nper round"]
    for h in ("linear", "mlp", "cnn_bilstm"):
        groups = [pick(runs, exp="E5", setting="iid6", head=h), pick(runs, exp="E2", method="fedavg", head=h),
                  pick(runs, exp="E5", setting="iid24", head=h), pick(runs, exp="E5", setting="iid24p", head=h)]
        m = [st.mean(r["test_best_mean_f1"] for r in g) for g in groups]
        s = [st.stdev(r["test_best_mean_f1"] for r in g) for g in groups]
        ax.errorbar(range(4), m, yerr=s, color=COLOR[h], marker=MARK[h], ms=5, mec="white", mew=0.6, capsize=2,
                    lw=1.4, label=SHORT[h])
    ax.set_xticks(range(4))
    ax.set_xticklabels(xs)
    ax.set_xlabel("Clients")
    ax.set_ylabel("Test macro-F1 (mean of 4)")
    ax.set_ylim(0.69, 0.71)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "f4_number_of_clients")


def fig_methods(runs):
    """Validation curves on the harshest split (α 0.1, seed 2), linear head, three methods."""
    fig, axes = plt.subplots(1, 2, figsize=(TWO, 2.3), sharey=True)
    styles = {"fedavg": ("FedAvg", "#2a78d6", "o"), "fedprox": ("FedProx", "#1baf7a", "^"), "fedadam": ("FedAdam", "#eb6834", "s")}
    for ax, h in zip(axes, ("linear", "mlp")):
        for m, (lab, col, mk) in styles.items():
            rs = pick(runs, exp="E3", setting="dir01", method=m, head=h, seed=2)
            if rs:
                c = val_curve(rs[0]["dir"])
                ax.plot(list(c), list(c.values()), color=col, marker=mk, markevery=6, ms=4, mec="white", mew=0.5, label=lab)
        ax.set_title(f"{SHORT[h]} head, α 0.1, seed 2", loc="left")
        ax.set_xlabel("Round")
    axes[0].set_ylabel("Validation macro-F1 (mean of 4)")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "f5_method_stability")


def fig_languages(runs):
    """Per-language test macro-F1, E2 FedAvg."""
    fig, ax = plt.subplots(figsize=(ONE, 2.3))
    off = {h: (i - 2.5) * 0.12 for i, h in enumerate(HEADS)}
    for i, l in enumerate(LANGS):
        for h in HEADS:
            v = st.mean(r["test_best_by_language"][l] for r in pick(runs, exp="E2", method="fedavg", head=h))
            ax.scatter(i + off[h], v, color=COLOR[h], marker=MARK[h], s=22, edgecolor="white", lw=0.6,
                       label=SHORT[h] if i == 0 else None)
    ax.set_xticks(range(4))
    ax.set_xticklabels([LANG_NAME[l] for l in LANGS])
    ax.set_ylabel("Test macro-F1")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=0.8)
    save(fig, "f6_per_language")


def load_eval():
    out = {}
    for p in sorted((ROOT / "analysis" / "eval").glob("eval_*.json")):
        out.update(json.loads(p.read_text()))
    return out


def fig_domains(runs, ev):
    """Marathi test macro-F1 per domain, E4 (by-domain clients) and, when evaluated, E2 (mixed clients)."""
    sets = [("E4", "Marathi clients by domain")]
    if any(k.startswith("E2-iid12-fedavg") for k in ev):
        sets.insert(0, ("E2", "Mixed Marathi clients"))
    doms = ["movie", "political", "subtitle", "tweet"]
    fig, axes = plt.subplots(1, len(sets), figsize=(ONE * len(sets), 2.3), sharey=True, squeeze=False)
    for ax, (exp, title) in zip(axes[0], sets):
        off = {h: (i - 2.5) * 0.12 for i, h in enumerate(HEADS)}
        for i, d in enumerate(doms):
            for h in HEADS:
                v = [ev[r["run_id"]]["best"]["mr"]["by_domain"][d]["macro_f1"]
                     for r in pick(runs, exp=exp, method="fedavg", head=h) if r["run_id"] in ev]
                if v:
                    ax.scatter(i + off[h], st.mean(v), color=COLOR[h], marker=MARK[h], s=22, edgecolor="white", lw=0.6,
                               label=SHORT[h] if i == 0 else None)
        ax.set_xticks(range(4))
        ax.set_xticklabels([d.capitalize() for d in doms])
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Marathi test macro-F1")
    fig.legend(*axes[0][0].get_legend_handles_labels(), ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=0.8)
    save(fig, "f7_marathi_domains")


def fig_local_clients(runs, ev):
    """Per client: local-only (own model) against the FedAvg model on the client's language (needs Ada eval)."""
    pts = defaultdict(list)
    for r in pick(runs, exp="E2", method="local_only"):
        alt = r["run_id"]
        if alt not in ev or "clients" not in ev[alt]:
            continue
        fa = pick(runs, exp="E2", method="fedavg", head=r["head"], seed=r["seed"])
        if not fa or fa[0]["run_id"] not in ev:
            continue
        fev = ev[fa[0]["run_id"]]
        for cid, c in ev[alt]["clients"].items():
            lang = cid.split("_")[0]
            pts[lang].append((c["final"]["macro_f1"], fev["final"][lang]["macro_f1"]))
    if not pts:
        print("figure f8 skipped: per-client evaluation not available yet")
        return
    fig, ax = plt.subplots(figsize=(ONE, 3.0))
    cols = dict(zip(LANGS, ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]))
    mks = dict(zip(LANGS, ["o", "s", "^", "D"]))
    allv, off = [], []
    floor = 0.55
    for l in LANGS:
        keep = [(x, y) for x, y in pts[l] if x >= floor]
        off += [(l, x, y) for x, y in pts[l] if x < floor]
        x, y = zip(*keep)
        allv += list(x) + list(y)
        ax.scatter(x, y, color=cols[l], marker=mks[l], s=16, edgecolor="white", lw=0.5, label=LANG_NAME[l])
    lo, hi = min(allv) - 0.01, max(allv) + 0.01
    if off:
        ax.text(lo + 0.003, hi - 0.06, f"{len(off)} clients off scale: own model\ncollapsed by round 200 "
                f"({', '.join(f'{x:.2f}' for _, x, _ in off)})", fontsize=6, color=MUTED, va="top")
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Client's own model (local-only), round 200")
    ax.set_ylabel("Shared FedAvg model, round 200")
    ax.text(hi - 0.005, lo + 0.005, "below the line: client\nbetter off alone", ha="right", va="bottom", fontsize=6, color=MUTED)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "f8_local_vs_fedavg_per_client")


# ---------------------------------------------------------------- tables

WIDE = {"t1_main", "t3_settings", "t4_clients", "t5_per_client"}   # span both columns


def tex_table(name, caption, label, cols, header, rows, note=None):
    TEX.mkdir(parents=True, exist_ok=True)
    env = "table*" if name in WIDE else "table"
    width = r"\textwidth" if name in WIDE else r"\columnwidth"   # full width of the span, font size unchanged
    lines = [rf"\begin{{{env}}}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular*}}{{{width}}}{{@{{\extracolsep{{\fill}}}}{cols}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular*}"]
    if note:
        lines.append(rf"\par\smallskip\footnotesize {note}")
    lines.append(rf"\end{{{env}}}")
    (TEX / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("table", name)


def ms(v):
    v = list(v)
    return f"{st.mean(v):.3f} $\\pm$ {st.stdev(v):.3f}"


def tables(runs):
    rows = []
    for h in HEADS:
        fa = pick(runs, exp="E2", method="fedavg", head=h)
        rows.append([SHORT[h], f"{MB[h]:.2f}", ms(r["test_best_mean_f1"] for r in fa), ms(r["test_final_mean_f1"] for r in fa),
                     ms(r["test_best_worst_f1"] for r in fa),
                     ms(r["test_best_mean_f1"] for r in pick(runs, exp="E2", method="centralized", head=h)),
                     ms(r["test_best_mean_f1"] for r in pick(runs, exp="E2", method="local_only", head=h))])
    tex_table("t1_main", "Test macro-F1 (mean of four languages) with 12 clients and even labels; mean $\\pm$ s.d. over three seeds.",
              "tab:main", "lrrrrrr", ["Head", "MB/round", "FedAvg", "FedAvg (r.~200)", "Worst lang.", "Centralized", "Local-only"], rows,
              "MB/round: what each client sends per round (LoRA plus head, fp32). Worst lang.: the lowest of the four languages (Bengali in every run).")
    fa2 = {h: by_seed(pick(runs, exp="E2", method="fedavg", head=h)) for h in HEADS}
    rows = []
    for h in HEADS[1:]:
        _, d = paired(fa2[h], fa2["linear"])
        rows.append([f"{SHORT[h]} $-$ Linear", ", ".join(f"{x:+.3f}" for x in d), f"{st.mean(d):+.3f}"])
    _, d = paired(fa2["cnn_bilstm"], fa2["mlp"])
    rows.append(["CNN-BiLSTM $-$ MLP", ", ".join(f"{x:+.3f}" for x in d), f"{st.mean(d):+.3f}"])
    tex_table("t2_paired", "Paired differences per seed in test macro-F1 (FedAvg, 12 clients, even labels).", "tab:paired",
              "lrr", ["Comparison", "Seeds 0, 1, 2", "Mean"], rows,
              "Re-running on another GPU moved single runs by up to 0.011, so differences below about 0.01 are within run-to-run noise.")
    rows = []
    for label, kw in SETTINGS:
        rows.append([label.replace("\n", ", ").replace("α", r"$\alpha$")] +[f"{st.mean(r['test_best_mean_f1'] for r in pick(runs, head=h, **kw)):.3f}" for h in HEADS])
    tex_table("t3_settings", "Test macro-F1 per head across settings (mean over three seeds).", "tab:settings",
              "l" + "r" * len(HEADS), ["Setting"] + [SHORT[h] for h in HEADS], rows,
              "FedAdam at $\\alpha$ 0.1 includes two seed-2 runs that collapsed to one class (0.23 and 0.25); see the text.")
    rows = []
    for h in ("linear", "mlp", "cnn_bilstm"):
        rows.append([SHORT[h], ms(r["test_best_mean_f1"] for r in pick(runs, exp="E5", setting="iid6", head=h)),
                     ms(r["test_best_mean_f1"] for r in pick(runs, exp="E2", method="fedavg", head=h)),
                     ms(r["test_best_mean_f1"] for r in pick(runs, exp="E5", setting="iid24", head=h)),
                     ms(r["test_best_mean_f1"] for r in pick(runs, exp="E5", setting="iid24p", head=h))])
    tex_table("t4_clients", "Number of clients and partial participation: test macro-F1, FedAvg.", "tab:clients",
              "lrrrr", ["Head", "6", "12", "24", "24 (12 per round)"], rows)


def tables_eval(runs, side, ev):
    """Tables that need the re-evaluated saved models, and the FedAdam machine check."""
    gaps = defaultdict(lambda: {"best": [], "final": []})
    for r in pick(runs, exp="E2", method="local_only"):
        if r["run_id"] not in ev or "clients" not in ev[r["run_id"]]:
            continue
        fa = pick(runs, exp="E2", method="fedavg", head=r["head"], seed=r["seed"])
        if not fa or fa[0]["run_id"] not in ev:
            continue
        for cid, c in ev[r["run_id"]]["clients"].items():
            lang = cid.split("_")[0]
            for s in ("best", "final"):
                gaps[lang][s].append(ev[fa[0]["run_id"]][s][lang]["macro_f1"] - c[s]["macro_f1"])
    if gaps:
        n_clients = {"mr": 5, "en": 4, "bn": 2, "hi": 1}
        train = {"mr": 47648, "en": 45583, "bn": 10415, "hi": 6649}
        rows = []
        for lang in LANGS:
            g = gaps[lang]
            rows.append([LANG_NAME[lang], n_clients[lang], f"{train[lang] // n_clients[lang]:,}", f"{st.mean(g['best']):+.3f}",
                         f"{sum(x > 0 for x in g['best']) / len(g['best']):.0%}".replace("%", r"\%"),
                         f"{st.mean(g['final']):+.3f}", f"{sum(x > 0 for x in g['final']) / len(g['final']):.0%}".replace("%", r"\%")])
        tex_table("t5_per_client", "FedAvg against each client's own model (local-only) on the client's language, test macro-F1.",
                  "tab:perclient", "lrrrrrr", ["Language", "Clients", "Rows/client", "Gap, best", "FedAvg better", "Gap, r.~200",
                                               "FedAvg better"], rows,
                  "Gap: FedAvg minus the client's own model, averaged over the six heads and three seeds; positive means the "
                  "client gains from federation. ``FedAvg better'': share of client-run pairs with a positive gap.")
        pcs = defaultdict(list)
        for r in pick(runs, exp="E2", method="fedavg"):
            if r["run_id"] in ev:
                for lang in LANGS:
                    pcs[lang].append(ev[r["run_id"]]["best"][lang]["per_class_f1"])
        rows = [[LANG_NAME[l]] + [f"{st.mean(p[i] for p in pcs[l]):.3f}" for i in range(3)] for l in LANGS]
        tex_table("t6_per_class", "Per-class test F1 under FedAvg (mean over heads and seeds).", "tab:perclass", "lrrr",
                  ["Language", "Negative", "Neutral", "Positive"], rows)
    chk = [r for (mod, _), r in side.items() if mod == "E3A_CHK"]
    if chk:
        rows = []
        for r in sorted(chk, key=lambda r: r["head"]):
            k = runs[r["run_id"]]
            rows.append([SHORT[r["head"]], f"{k['test_best_mean_f1']:.3f}", f"{k['test_final_mean_f1']:.3f}",
                         f"{r['test_best_mean_f1']:.3f}", f"{r['test_final_mean_f1']:.3f}"])
        tex_table("t7_fedadam_check", "The same two FedAdam runs ($\\alpha$ 0.1, seed 2) on two GPUs: test macro-F1.", "tab:fedadam",
                  "lrrrr", ["Head", "T4, best", "T4, final", "2080 Ti, best", "2080 Ti, final"], rows,
                  "Best: round with the best validation score. Final: round 200.")


def main():
    runs, side = load_runs()
    ev = load_eval()
    tables_eval(runs, side, ev)
    fig_cost(runs)
    fig_curves(runs)
    fig_settings(runs)
    fig_clients(runs)
    fig_methods(runs)
    fig_languages(runs)
    fig_domains(runs, ev)
    fig_local_clients(runs, ev)
    tables(runs)


if __name__ == "__main__":
    main()
