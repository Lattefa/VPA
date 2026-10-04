"""MedCalc: figures and LaTeX tables from the voting CSVs.

Usage:
    python medcalc_vote.py     # first, writes results/vpa_results_*_ties.csv
    python medcalc_figs.py

Writes to figures/:
    {Abstain,Random}_vpa_vs_base.pdf   ours vs base models (error, error on answered)
    vpa_breakdown.pdf                  correct / incorrect / abstained per plurality
    table_medcalc_{abstain,random}.tex ours vs base models on the same questions
                                       (needs \\usepackage{booktabs})
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "savefig.dpi": 300,
})

HERE     = os.path.dirname(os.path.abspath(__file__))
SRC      = os.path.join(HERE, "data", "filtered_medCalc.csv")
RES      = os.path.join(HERE, "results")
FIGDIR   = os.path.join(HERE, "figures")
N_ROWS   = 70
META     = ["question_id", "calculator_name", "output_type",
            "correct_answer", "lower_limit", "upper_limit"]
KS       = range(1, 9)        # figures
TABLE_KS = range(5, 9)        # table columns
VARIANTS = {"abstain": ("vpa_results_abstain_ties.csv", "Abstain on ties"),
            "random":  ("vpa_results_random_ties.csv",  "Random tie-break")}
os.makedirs(FIGDIR, exist_ok=True)

# data
df   = pd.read_csv(SRC).head(N_ROWS)
mods = [c for c in df.columns if c not in META]
resp = df[mods].apply(pd.to_numeric, errors="coerce")
LO, HI = df["lower_limit"].values, df["upper_limit"].values
N    = len(df)

base_ok = pd.DataFrame({m: (resp[m].values >= LO) & (resp[m].values <= HI)
                        for m in mods}, index=df.index)


# ==========================================
#  FIGURES
# ==========================================
def curves(out):
    """Per plurality: coverage plus both risk denominators.
       risk_all = wrong / all questions   (abstentions are NOT errors)
       risk_ans = wrong / answered"""
    rows = []
    for k in KS:
        v   = out[f"mv_{k}"].values[:N].astype(float)
        ans = ~np.isnan(v)
        ok  = (v >= LO) & (v <= HI)
        n   = int(ans.sum())
        err = int(n - ok.sum())
        rows.append(dict(k=k, mask=ans, answered=n,
                         coverage=100 * n / N,
                         risk_all=100 * err / N,
                         risk_ans=100 * err / n if n else np.nan,
                         pct_correct=100 * ok.sum() / N,
                         pct_incorrect=100 * err / N,
                         pct_abstained=100 * (N - n) / N))
    return rows


def _vs_base(ax, cur, base_curves, mv_key, ylabel):
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    ks = [c["k"] for c in cur]
    for i, m in enumerate(mods):
        ax.plot(ks, base_curves[m], color=cmap[i % 20], lw=1.3, alpha=.85,
                linestyle="--", marker="o", ms=3, label=m.split("/")[-1])
    ax.plot(ks, [c[mv_key] for c in cur], color="0.20", lw=1.3,
            marker="^", ms=3, label="Ours", zorder=10)
    ax.set_xlabel(r"Plurality $\tau$"); ax.set_ylabel(ylabel)
    ax.set_xticks(ks); ax.grid(alpha=.3)


def plot_vs_base(cur, dst):
    flat = {m: [100 * (1 - base_ok[m].mean())] * len(KS) for m in mods}
    skip = {m: [100 * (1 - base_ok.loc[c["mask"], m].mean())
                if c["mask"].sum() else np.nan for c in cur] for m in mods}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _vs_base(axes[0], cur, flat, "risk_all", "Error (%)")
    _vs_base(axes[1], cur, skip, "risk_ans", "Error on answered (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="upper center", frameon=True,
               bbox_to_anchor=(.5, 1.07), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def plot_breakdown(curs, dst):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (lab, cur) in zip(axes, curs.items()):
        ks = [c["k"] for c in cur]
        bottom = np.zeros(len(ks))
        for key, name, col in [("pct_correct", "Correct", "#8FBC8F"),
                               ("pct_incorrect", "Incorrect", "#E8938C"),
                               ("pct_abstained", "Abstained", "#B8BFC7")]:
            vals = [c[key] for c in cur]
            ax.bar(ks, vals, bottom=bottom, color=col, edgecolor="black",
                   width=.6, label=name)
            for x, v, b in zip(ks, vals, bottom):
                if v > 2.5:
                    ax.text(x, b + v / 2, f"{v:.1f}", ha="center", va="center",
                            fontsize=9)
            bottom += np.array(vals)
        ax.set_title(lab, fontsize=13); ax.set_xlabel(r"Plurality $\tau$")
        ax.set_xticks(ks); ax.set_ylim(0, 100)
    axes[0].set_ylabel("Questions (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="upper center", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


# ==========================================
#  LATEX TABLE
# ==========================================
def tex(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def fmt(v, bold=False):
    if v is None or np.isnan(v):
        return "--"
    s = f"{v:.1f}"
    return rf"\textbf{{{s}}}" if bold else s


def make_table(out, tie_label):
    ks = [k for k in TABLE_KS if f"mv_{k}" in out.columns]
    full = 100 * (1 - base_ok.mean())                     # base model error, all questions
    same, ours, cov, n_ans = {}, {}, {}, {}
    for k in ks:
        v    = out[f"mv_{k}"].values[:N].astype(float)
        mask = ~np.isnan(v)
        ok   = (v >= LO) & (v <= HI)
        n    = int(mask.sum())
        n_ans[k] = n
        cov[k]   = 100 * n / N
        ours[k]  = 100 * (n - ok[mask].sum()) / n if n else np.nan
        same[k]  = (100 * (1 - base_ok.loc[mask, mods].mean()) if n
                    else pd.Series(np.nan, index=mods))

    order = full.sort_values().index                      # best base model first
    best  = {"all": full.min(), **{k: same[k].min() for k in ks}}

    n_cols = 1 + len(ks)
    lines = [
        r"\begin{tabular}{l" + "r" * n_cols + "}",
        r"\toprule",
        rf" & \multicolumn{{1}}{{c}}{{No abstention}} & \multicolumn{{{len(ks)}}}{{c}}"
        r"{Abstain where ours abstains ($k$)} \\",
        rf"\cmidrule(lr){{2-2}}\cmidrule(lr){{3-{n_cols + 1}}}",
        r"Error (\%) & All & " + " & ".join(str(k) for k in ks) + r" \\",
        r"\midrule",
    ]
    for m in order:
        cells = [fmt(full[m], np.isclose(full[m], best["all"]))]
        cells += [fmt(same[k][m], bool(np.isclose(same[k][m], best[k]))) for k in ks]
        lines.append(f"{tex(m.split('/')[-1])} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        "Best base model & " + " & ".join([fmt(best["all"])] + [fmt(best[k]) for k in ks]) + r" \\",
        f"Ours ({tie_label}) & -- & " + " & ".join(fmt(ours[k]) for k in ks) + r" \\",
        r"\midrule",
        r"Coverage (\%) & 100.0 & " + " & ".join(f"{cov[k]:.1f}" for k in ks) + r" \\",
        f"Questions answered & {N} & " + " & ".join(str(n_ans[k]) for k in ks) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    curs = {}
    for key, (path, label) in VARIANTS.items():
        out = pd.read_csv(os.path.join(RES, path))
        if len(out) < N:
            raise ValueError(f"{path}: {len(out)} rows, expected at least {N}")

        curs[label] = cur = curves(out)
        plot_vs_base(cur, os.path.join(FIGDIR, f"{label.split()[0]}_vpa_vs_base.pdf"))

        fn = os.path.join(FIGDIR, f"table_medcalc_{key}.tex")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(make_table(out, label[0].lower() + label[1:]))

    plot_breakdown(curs, os.path.join(FIGDIR, "vpa_breakdown.pdf"))
    print(f"saved figures and tables to {FIGDIR}")
