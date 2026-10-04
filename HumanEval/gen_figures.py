"""All HumanEval paper figures and the table (vector PDF, 300 dpi, fonts embedded).

Usage:
    python gen_figures.py
    python gen_figures.py --csv data/humaneval_plus_results.csv --pt 1 --eval plus --table-ties 1

Reads
    results/mv_out/max_completions/*_eval_results.json   EvalPlus result per selection run
    results/mv_out/max_rec_stats/{stem}_records.json      which tasks were abstained
                                                          (fallback: solution == "ABS")
    results/adv_out/...                                   same layout, copy-adversary runs (_adv{f})
                                                          and CodeT runs (_codet_adv{f})
    data/adj/{metric}_sim.npz                             similarity matrices (to rebuild groups)
    --csv                                                 per-model HumanEval+ pass/fail

Writes figures/
    results_all_runs.csv                 one row per eval file
    results_table.csv                    averaged over seeds, filtered by --tag / --pt
    HumanEval_coverage_accuracy.pdf      accuracy on answered tasks vs coverage (one point per k)
    HumanEval_error_vs_plurality.pdf     error on answered vs tau | vs coverage
    HumanEval_correct_fragmentation.pdf  largest correct component vs size of the correct class
    HumanEval_adversary_error_vs_tau.pdf error on answered vs tau under the copy adversary,
                                         colour = similarity, line style = f (0 = clean, 1, 2 copies)
                                         (--adv-panels: one panel per similarity)
    HumanEval_rtests_vs_codet.pdf        RTests (ours, curve over tau) vs CodeT (never abstains,
                                         horizontal line), one panel per f
    + a .csv next to each figure with the numbers
Writes figures/ablation/  (runs of --seed only)
    ablation_error_vs_tau.pdf
        error on answered vs tau, one panel per metric,
        tests passed then group size (pt1, solid) vs group size only (pt0, dashed),
        random tie-break (ties0), so both answer the same tasks.
    table_tests_vs_models.tex
        RTests metric, pt1, ties = --table-ties.
        Each base model: error on all tasks, then error on the SAME tasks our method
        answered at each tau (fair comparison). Best base model per column in bold.
        Last rows: ours (error on answered) and coverage.
"""
import os, re, sys, json, glob, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

BASE = os.path.dirname(os.path.abspath(__file__))
MV   = os.path.join(BASE, "results", "mv_out")
ADV  = os.path.join(BASE, "results", "adv_out")
ADJ  = os.path.join(BASE, "data", "adj")
OUT  = os.path.join(BASE, "figures")
OUT_ABL = os.path.join(OUT, "ablation")

K_MAX   = 7                                   # ignore runs with k above this
METRICS = ["crosshair", "codebleu", "tests", "init_tests"]
NAMES   = {"crosshair": "CrossHair", "codebleu": "CodeBLEU",
           "tests": "RTests", "init_tests": "Tests"}
COLORS  = {"crosshair": "#0072B2", "codebleu": "#E69F00",       # Okabe-Ito,
           "tests": "#009E73", "init_tests": "#CC79A7"}          # colour-blind safe
MARKERS = {"crosshair": "o", "codebleu": "s", "tests": "^", "init_tests": "D"}

PT_NAME = {1: r"$\tau$ and $V$", 0: r"$\tau$ only"}

# must match the selection script
TAU  = {"crosshair": 1.0, "codebleu": 0.9, "tests": 1.0, "init_tests": 1.0}

REF_GREY = "#555555"


# adversary figures (paper settings)
ADV_METRICS = ["tests", "crosshair"]          # similarities under attack
ADV_COPIES  = [1, 2]                          # f = number of copies; f = 0 is the clean run
ADV_KS      = [2, 3, 6, 7]                    # tau values
ADV_TIES    = 1                               # abstain on ties
F_STYLES    = {1: ":", 2: "-", 3: "-."}       # line style per f (single-axis mode)
F_COLORS    = ["#9ECAE1", "#4292C6", "#08519C", "#08306B"]   # --adv-panels: light -> dark = more copies
F_MARKS     = ["o", "s", "^", "D"]

CODET_KS        = [1, 2, 3, 6, 7]
CODET_COPIES    = [0, 1, 2]
CODET_TIES      = 1                           # RTests tie rule for tau >= 2 (abstain on ties)
CODET_TIES_TAU1 = 0                           # RTests tie rule at tau = 1 (random, no abstention)
RT_COL, CT_COL  = "#009E73", "#E69F00"        # RTests green, CodeT orange (as elsewhere)

# vector PDF, 300 dpi, fonts embedded
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          12,
    "axes.titlesize":     12,
    "axes.labelsize":     12,
    "axes.linewidth":     0.4,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "xtick.major.width":  0.4,
    "ytick.major.width":  0.4,
    "legend.fontsize":    10,
    "legend.frameon":     False,
    "lines.linewidth":    1.2,
    "lines.markersize":   4,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "savefig.dpi":        300,
    "savefig.format":     "pdf",
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
})



STEM_RE = re.compile(
    r"^(?P<metric>crosshair|codebleu|init_tests|tests)_k(?P<k>\d+)_tau(?P<tau>[\d.]+)"
    r"_pt(?P<pt>\d)_ties(?P<ties>\d)_s(?P<seed>\d+)(?P<tag>.*?)_eval_results?\.json$")

COLS = ["n_tasks", "n_answered", "coverage",
        "sel_acc_base", "sel_acc_plus", "acc_base", "acc_plus"]


# helpers
def pct():
    """Tick formatter: 0.85 -> 85%."""
    return PercentFormatter(xmax=1, decimals=0, symbol="")


def save(fig, name, out=OUT):
    fig.savefig(os.path.join(out, name))
    plt.close(fig)
    print(f"saved {name}")


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["id"] = df["id"].astype(str).str.strip()
    return df.set_index("id").astype(int)


def metric_handles(present):
    return [Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=NAMES[m])
            for m in METRICS if m in present]


# eval results
def answered_tasks(path, status):
    """(n_tasks, {answered task: passed?}) from an EvalPlus eval file + its records file.

    path   {root}/max_completions/{stem}_eval_results.json
    status "plus_status" or "base_status"
    Abstentions come from {root}/max_rec_stats/{stem}_records.json
    (fallback: missing from the eval file or solution == "ABS").
    A task answered but missing from the eval file counts as wrong.
    """
    stem = os.path.basename(path).rsplit("_eval_result", 1)[0]
    root = os.path.dirname(os.path.dirname(path))               # adv_out/ or mv_out/
    ev   = json.load(open(path, encoding="utf-8"))["eval"]
    rp   = os.path.join(root, "max_rec_stats", f"{stem}_records.json")
    recs = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else None

    tasks, answered = (list(recs) if recs else list(ev)), {}
    for tid in tasks:
        r = (ev.get(tid) or [None])[0]
        if recs is not None:
            abstained = recs[tid]["abstained"]
        else:
            abstained = r is None or (r.get("solution") or "").strip() == "ABS"
        if not abstained:
            answered[tid] = r is not None and r.get(status) == "pass"
    return len(tasks), answered


def score(path, status):
    """(n_tasks, n_answered, n_wrong) from an eval file + its records file."""
    n, answered = answered_tasks(path, status)
    return n, len(answered), sum(not ok for ok in answered.values())


def load_runs():
    """One row per eval file (k <= K_MAX), with the answered tasks and their pass status."""
    rows = []
    pattern = os.path.join(MV, "max_completions", "*_eval_result*.json")
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        m = STEM_RE.match(fname)
        if not m:
            print(f"skip (name not recognised): {fname}")
            continue
        if int(m["k"]) > K_MAX:
            continue
        answered = {}                               # eval -> {answered task: passed?}
        for ev_name in ("base", "plus"):
            n, answered[ev_name] = answered_tasks(path, f"{ev_name}_status")
        stem = fname.rsplit("_eval_result", 1)[0]
        ans = len(answered["plus"])
        pb, pp = sum(answered["base"].values()), sum(answered["plus"].values())
        rows.append(dict(
            stem=stem, metric=m["metric"], k=int(m["k"]), tau=float(m["tau"]),
            pt=int(m["pt"]), ties=int(m["ties"]), seed=int(m["seed"]), tag=m["tag"],
            n_tasks=n, n_answered=ans,
            coverage=ans / n if n else np.nan,
            sel_acc_base=pb / ans if ans else np.nan,     # accuracy on answered tasks
            sel_acc_plus=pp / ans if ans else np.nan,
            acc_base=pb / n if n else np.nan,             # abstain counts as wrong
            acc_plus=pp / n if n else np.nan,
            answered=answered))
    return pd.DataFrame(rows)



def to_adjacency(sim, known, valid, tau):
    e   = (sim >= tau) & known
    adj = e & e.T                                   # require_both=True
    adj &= valid[:, None] & valid[None, :]
    np.fill_diagonal(adj, valid)
    return adj


def components(adj):
    seen, out = set(), []
    for i in range(len(adj)):
        if i in seen or not adj[i, i]:
            continue
        comp, stack = [], [i]; seen.add(i)
        while stack:
            c = stack.pop(); comp.append(c)
            for j in np.flatnonzero(adj[c]):
                if int(j) not in seen:
                    seen.add(int(j)); stack.append(int(j))
        out.append(sorted(comp))
    return out


def load_groups(keep_models=None):
    """{metric: {task: (components, labels)}}, labels = group id per model (-1 = invalid)."""
    npz = {m: np.load(os.path.join(ADJ, f"{m}_sim.npz"))
           for m in METRICS if os.path.exists(os.path.join(ADJ, f"{m}_sim.npz"))}
    common = None
    for d in npz.values():
        s = {str(x).strip() for x in d["models"]}
        common = s if common is None else common & s
    common = sorted(common or [])
    if keep_models is not None:
        common = [m for m in common if m in keep_models]

    groups = {}
    for metric, d in npz.items():
        ms  = [str(x).strip() for x in d["models"]]
        idx = [ms.index(x) for x in common]
        r, c = np.ix_(idx, idx)
        S, K, V = d["sim"][:, r, c], d["known"][:, r, c], d["valid"][:, idx]
        per = {}
        for t, tid in enumerate(d["tasks"]):
            comps = components(to_adjacency(S[t], K[t], V[t], TAU[metric]))
            lab = np.full(len(common), -1)
            for g, comp in enumerate(comps):
                lab[comp] = g
            per[str(tid).strip()] = (comps, lab)
        groups[metric] = per
    return groups, common


# Figures
def fig_coverage_accuracy(agg, ev):
    y = f"sel_acc_{ev}"
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4), sharex=True, sharey=True)
    for ax, ties, title in zip(axes, (0, 1), ("Ties broken at random", "Abstain on ties")):
        sub = agg[agg.ties == ties]
        for m in METRICS:
            d = sub[sub.metric == m].sort_values("k")
            if not d.empty:
                ax.plot(d.coverage, d[y], color=COLORS[m], marker=MARKERS[m], clip_on=False)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(pct())
        ax.yaxis.set_major_formatter(pct())
        ax.set_xlabel("Coverage (%)",fontsize = 15)
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("Accuracy on answered", fontsize = 15)
    fig.legend(handles=metric_handles(set(agg.metric)), loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, 1.0))
    agg[["metric", "ties", "k", "coverage", y]].to_csv(
        os.path.join(OUT, "fig_coverage_accuracy.csv"), index=False)
    save(fig, "HumanEval_coverage_accuracy.pdf")


def fig_error_vs_k(agg, ev):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4))         
    acc = f"sel_acc_{ev}"
    ks  = sorted(agg.k.unique())
    pos = {k: i for i, k in enumerate(ks)}          
    axL, axR = axes

    for m in METRICS:
        for ties, ls, hollow in ((0, "-", False), (1, "--", True)):
            d = agg[(agg.metric == m) & (agg.ties == ties)].sort_values("k")
            if d.empty:
                continue
            kw = dict(color=COLORS[m], marker=MARKERS[m], ls=ls,
                      mfc="white" if hollow else COLORS[m])
            axL.plot(d.k.map(pos), 1 - d[acc], **kw)                  # error vs tau
            axR.plot(d.coverage,   1 - d[acc], clip_on=False, **kw)   # error vs coverage

    # left: error vs tau
    axL.set_xticks(range(len(ks)), ks)
    axL.set_xlabel(r"$\tau$", fontsize=15)
    axL.set_ylabel("Error (%)")

    # right: error vs coverage
    axR.set_xlim(0, 1)
    axR.xaxis.set_major_formatter(pct())
    axR.set_xlabel("Coverage (%)", fontsize=15)
    axR.set_ylabel("Error (%)")

    for ax in axes:
        ax.yaxis.set_major_formatter(pct())
        ax.grid(alpha=0.3, lw=0.4)

    style = [Line2D([], [], color="black", ls="-",  marker="o", label="Random tie-break"),
             Line2D([], [], color="black", ls="--", marker="o", mfc="white", label="Abstain on ties")]
    fig.legend(handles=metric_handles(set(agg.metric)) + style, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, 0.85))
    fig.subplots_adjust(wspace=0.35)
    save(fig, "HumanEval_error_vs_plurality.pdf")


def fig_correct_fragmentation(groups, models, df):
    """Per task: true correct class = #models passing (CSV).
       Found = largest number of correct models inside ONE cluster of the similarity.
       Fragmented if found < true. Plot mean found size vs true size (diagonal = perfect)."""
    ms = [m for m in METRICS if m in groups]
    rows = []
    for m in ms:
        for tid, (comps, _) in groups[m].items():
            if tid not in df.index:
                continue
            p = df.loc[tid, models].to_numpy()
            true = int(p.sum())
            if true < 2:                                  # nothing to fragment
                continue
            found = max((int(p[c].sum()) for c in comps), default=0)
            rows.append(dict(metric=m, task=tid, true=true, found=found,
                             fragmented=found < true))
    if not rows:
        print("skip fig_correct_fragmentation: no tasks with >= 2 correct models"); return
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "fig_correct_fragmentation_tasks.csv"), index=False)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    sizes = sorted(t.true.unique())
    ax.plot(sizes, sizes, color=REF_GREY, ls=":", lw=0.8, zorder=0)       # perfect
    for m in ms:
        d = t[t.metric == m].groupby("true").found.mean()
        ax.plot(d.index, d.values, color=COLORS[m], marker=MARKERS[m],
                label=NAMES[m], clip_on=False)
        print(f"{NAMES[m]:>10}: fragmented on {t[t.metric == m].fragmented.mean():.1%} "
              f"of tasks with >= 2 correct models")
    ax.set_xticks(sizes)
    ax.set_yticks(sizes)
    ax.set_xlabel("Size of correct class (HumanEval+)", fontsize = 12)
    ax.set_ylabel("Formed correct component", fontsize = 12)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "HumanEval_correct_fragmentation.pdf")


def fig_ablation_error_vs_k(runs):
    rows = [dict(metric=m, k=k, pt=pt,
                 err=1 - np.mean(list(a.values())) if a else np.nan)
            for (m, k, pt, ties), (_, a) in runs.items() if ties == 0]
    if not rows:
        print("skip ablation figure: no ties0 runs"); return
    d  = pd.DataFrame(rows)
    ms = [m for m in METRICS if m in set(d.metric)]

    fig, axes = plt.subplots(1, len(ms), figsize=(7, 2.0), sharex=True, sharey=True, squeeze=False)
    ks  = sorted(d.k.unique())
    pos = {k: i for i, k in enumerate(ks)}      # k -> 0, 1, 2, ...
    for ax, m in zip(axes[0], ms):
        for pt, ls, hollow in ((1, "-", False), (0, "--", True)):
            s = d[(d.metric == m) & (d.pt == pt)].sort_values("k")
            if not s.empty:
                ax.plot(s.k.map(pos), s.err, color=COLORS[m], marker=MARKERS[m], ls=ls,
                    mfc="white" if hollow else COLORS[m], clip_on=False)
        ax.set_title(NAMES[m])
        ax.set_xticks(range(len(ks)), ks)
        ax.set_xlabel(r"$\tau$", fontsize =15)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(pct())
        ax.grid(alpha=0.3, lw=0.4)
    axes[0][0].set_ylabel("Error on answered (%)",fontsize=15)
    handles = [Line2D([], [], color="black", ls="-",  marker="o", label=PT_NAME[1]),
               Line2D([], [], color="black", ls="--", marker="o", mfc="white", label=PT_NAME[0])]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "ablation_error_vs_tau.pdf", OUT_ABL)


def tex(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def fmt(v, bold=False):
    s = f"{100 * v:.1f}"
    return rf"\textbf{{{s}}}" if bold else s


def table_tests_vs_models(runs, df, ties):
    ks = sorted(k for (m, k, pt, t) in runs if m == "tests" and pt == 1 and t == ties)
    if not ks:
        print(f"skip table: no tests / pt1 / ties{ties} runs"); return
    models = list(df.columns)

    # columns: all tasks, then each k (tasks our method answered)
    full = 1 - df.mean()                                   # base model error, all tasks
    same, ours, cov, n_ans = {}, {}, {}, {}
    for k in ks:
        n_tasks, answered = runs[("tests", k, 1, ties)]
        A = [t for t in answered if t in df.index]
        if len(A) < len(answered):
            print(f"k={k}: {len(answered) - len(A)} answered tasks missing from CSV, ignored")
        same[k]  = 1 - df.loc[A, models].mean() if A else pd.Series(np.nan, index=models)
        ours[k]  = 1 - np.mean([answered[t] for t in A]) if A else np.nan
        cov[k]   = len(answered) / n_tasks if n_tasks else np.nan
        n_ans[k] = len(answered)

    order = full.sort_values().index                       
    best_col = {"all": full.min(), **{k: same[k].min() for k in ks}}

    n_cols = 1 + len(ks)
    lines = [
        r"\begin{tabular}{l" + "r" * n_cols + "}",
        r"\toprule",
        rf" & \multicolumn{{1}}{{c}}{{No abstention}} & \multicolumn{{{len(ks)}}}{{c}}"
        r"{Abstain where ours abstains ($k$)} \\",
        rf"\cmidrule(lr){{2-2}}\cmidrule(lr){{3-{n_cols + 1}}}",
        "Error (\\%) & All tasks & " + " & ".join(str(k) for k in ks) + r" \\",
        r"\midrule",
    ]
    for mdl in order:
        cells = [fmt(full[mdl], np.isclose(full[mdl], best_col["all"]))]
        cells += [fmt(same[k][mdl], np.isclose(same[k][mdl], best_col[k])) for k in ks]
        lines.append(f"{tex(mdl)} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        "Best base model & " + " & ".join(
            [fmt(best_col["all"])] + [fmt(best_col[k]) for k in ks]) + r" \\",
        rf"Ours (Tests, pt1{', abstain on ties' if ties else ''}) & -- & "
        + " & ".join(fmt(ours[k]) for k in ks) + r" \\",
        r"\midrule",
        "Coverage (\\%) & 100.0 & " + " & ".join(f"{100 * cov[k]:.1f}" for k in ks) + r" \\",
        "Tasks answered & " + str(len(df)) + " & " + " & ".join(str(n_ans[k]) for k in ks) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(os.path.join(OUT_ABL, "table_tests_vs_models.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved table_tests_vs_models.tex")


# copy adversary
def collect(root, metric, pt, ties, seed, tag, f, status):
    """{k: (n_tasks, n_answered, n_wrong)}. f=None -> clean files (no _adv suffix)."""
    suffix = rf"_adv{f}" if f is not None else ""
    rx = re.compile(rf"^{re.escape(metric)}_k(\d+)_tau[\d.]+_pt{pt}_ties{ties}_s{seed}"
                    rf"{re.escape(tag)}{suffix}_eval_results?\.json$")
    out = {}
    for path in glob.glob(os.path.join(root, "max_completions", "*_eval_result*.json")):
        m = rx.match(os.path.basename(path))
        if m:
            out[int(m.group(1))] = score(path, status)
    return out


def err_series(c, ks):
    kc = [k for k in ks if k in c and c[k][1]]
    return kc, [c[k][2] / c[k][1] for k in kc]


def fig_adversary_error_vs_k(pt, seed, tag, status, single=True):
    metrics, copies = ADV_METRICS, ADV_COPIES

    # data[metric][f] = {k: (n_tasks, n_answered, n_wrong)};  f = 0 is the clean baseline
    data = {}
    for m in metrics:
        data[m] = {0: collect(MV, m, pt, ADV_TIES, seed, tag, None, status)}
        for f in copies:
            data[m][f] = collect(ADV, m, pt, ADV_TIES, seed, tag, f, status)
        missing = [f for f, c in data[m].items() if not c]
        if missing:
            print(f"warn {m}: no eval results for f = {missing}")

    ks = sorted({k for m in data for f, c in data[m].items() if f > 0 for k in c} & set(ADV_KS))
    if not ks:
        print("skip adversary figure: no matching eval results found"); return
    pos = {k: i for i, k in enumerate(ks)}                          # evenly spaced tau

    if single:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        axes = {m: ax for m in metrics}
    else:
        fig, axs = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 2.4),
                                sharey=True, squeeze=False)
        axes = dict(zip(metrics, axs[0]))

    for m in metrics:
        ax = axes[m]
        for f, c in sorted(data[m].items()):
            kc, y = err_series(c, ks)
            if not kc:
                continue
            x = [pos[k] for k in kc]
            if single:
                if f == 0:
                    ax.plot(x, y, color=COLORS[m], ls="--", marker=MARKERS[m], mfc="white")
                else:
                    ax.plot(x, y, color=COLORS[m], ls=F_STYLES.get(f, "-"), marker=MARKERS[m])
            else:
                if f == 0:
                    ax.plot(x, y, color="black", ls="--", marker="o", mfc="white",
                            label=r"$f=0$")
                else:
                    i = copies.index(f)
                    ax.plot(x, y, color=F_COLORS[i % len(F_COLORS)],
                            marker=F_MARKS[i % len(F_MARKS)], label=rf"$f={f}$")

    for m, ax in axes.items():
        ax.set_xticks(range(len(ks)), ks)
        ax.set_xlabel(r"$\tau$", fontsize=15)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(pct())
        ax.grid(alpha=0.3, lw=0.4)
        if not single:
            ax.set_title(NAMES.get(m, m))

    first = axes[metrics[0]]
    first.set_ylabel("Error on answered (%)")

    if single:
        handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=NAMES.get(m, m))
                   for m in metrics]
        handles += [Line2D([], [], color="black", ls="--", marker="o", mfc="white", label=r"$f=0$")]
        handles += [Line2D([], [], color="black", ls=F_STYLES.get(f, "-"), label=rf"$f={f}$")
                    for f in copies]
        fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    else:
        h, l = first.get_legend_handles_labels()
        fig.legend(h, l, ncol=len(h), loc="lower center", bbox_to_anchor=(0.5, 1.0))
        fig.subplots_adjust(wspace=0.15)

    fig.savefig(os.path.join(OUT, "HumanEval_adversary_error_vs_tau.pdf"))
    plt.close(fig)

    rows = [dict(metric=m, f=f, k=k, n_tasks=v[0], n_answered=v[1], n_wrong=v[2])
            for m in metrics for f, c in sorted(data[m].items()) for k, v in sorted(c.items())
            if k in pos]
    t = pd.DataFrame(rows)
    t["coverage"]     = t.n_answered / t.n_tasks
    t["err_answered"] = t.n_wrong / t.n_answered.where(t.n_answered > 0)
    t.to_csv(os.path.join(OUT, "HumanEval_adversary_error_vs_tau.csv"), index=False)
    print(t.to_string(index=False, formatters={"coverage": "{:.1%}".format,
                                               "err_answered": "{:.1%}".format}))
    print("saved figures/HumanEval_adversary_error_vs_tau.pdf / .csv")


# VPA(RTests) vs CodeT
def find(root, k, ties, seed, tag, f):
    adv = rf"_adv{f}" if f is not None else ""
    rx = re.compile(rf"^tests_k{k}_tau[\d.]+_pt1_ties{ties}_s{seed}"
                    rf"{re.escape(tag)}{adv}_eval_results?\.json$")
    for p in glob.glob(os.path.join(root, "max_completions", "*_eval_result*.json")):
        if rx.match(os.path.basename(p)):
            return p
    return None


def get(k, ties, seed, tag, f, status, fallback_clean=False):
    p = find(ADV, k, ties, seed, tag, f)
    if p is None and fallback_clean and f == 0:
        p = find(MV, k, ties, seed, tag, None)
    if p is None:
        return None
    n, ans, wrong = score(p, status)
    return dict(n_tasks=n, n_answered=ans, n_wrong=wrong,
                coverage=ans / n if n else np.nan,
                err=wrong / ans if ans else np.nan)


def fig_rtests_vs_codet(seed, tag, status):
    ks, copies = CODET_KS, CODET_COPIES
    pos = {k: i for i, k in enumerate(ks)}                  

    rows = []
    fig, axes = plt.subplots(1, len(copies), figsize=(2.6 * len(copies), 2.4),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, f in zip(axes, copies):
        # VPA vs tau
        xs, ys = [], []
        for k in ks:
            ties = CODET_TIES_TAU1 if k == 1 else CODET_TIES
            c = get(k, ties, seed, tag, f, status, fallback_clean=True)
            if c and not np.isnan(c["err"]):
                xs.append(pos[k]); ys.append(c["err"])
                rows.append(dict(method="RTests", f=f, k=k, ties=ties, **c))
        if xs:
            ax.plot(xs, ys, color=RT_COL, marker="^", ls="-", zorder=3)
        else:
            print(f"warn: no RTests results for f={f}")

        # CodeT
        c = get(1, 0, seed, tag + "_codet", f, status)
        if c and not np.isnan(c["err"]):
            ax.plot([0, len(ks) - 1], [c["err"], c["err"]], color=CT_COL, ls="--",
                    lw=1.4, zorder=2)
            rows.append(dict(method="CodeT", f=f, k=1, ties=0, **c))
        else:
            print(f"warn: no CodeT results for f={f}")

        ax.set_title(rf"$f={f}$")
        ax.set_xticks(range(len(ks)), ks)
        ax.set_xlabel(r"$\tau$", fontsize=15)
        ax.yaxis.set_major_formatter(pct())
        ax.grid(alpha=0.3, lw=0.4)


    ymax = max((r["err"] for r in rows if not np.isnan(r["err"])), default=0.05)
    axes[0].set_ylim(0, ymax * 1.12)
    axes[0].set_ylabel("Error on answered (%)")
    handles = [Line2D([], [], color=RT_COL, ls="-",  marker="^", label="VPA (ours)"),
               Line2D([], [], color=CT_COL, ls="--", lw=1.4,   label="CodeT")]
    fig.legend(handles=handles, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(wspace=0.12)

    fig.savefig(os.path.join(OUT, "HumanEval_rtests_vs_codet.pdf"))
    plt.close(fig)

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "HumanEval_rtests_vs_codet.csv"), index=False)
    if not t.empty:
        print(t[["method", "f", "k", "ties", "n_answered", "n_wrong", "coverage", "err"]]
              .to_string(index=False, formatters={"coverage": "{:.1%}".format,
                                                  "err": "{:.1%}".format}))
    print("saved figures/HumanEval_rtests_vs_codet.pdf / .csv")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(BASE, "data", "humaneval_plus_results.csv"),
                    help="per-model HumanEval+ pass/fail CSV")
    ap.add_argument("--tag", default="", help="run tag to plot")
    ap.add_argument("--pt", type=int, default=1, help="passed_tests setting for the main figures")
    ap.add_argument("--seed", type=int, default=0, help="seed for the ablation figure and table")
    ap.add_argument("--eval", choices=["plus", "base"], default="plus",
                    help="HumanEval+ (plus) or HumanEval (base) status")
    ap.add_argument("--table-ties", type=int, default=1,
                    help="tie setting for the table: 1 = abstain on ties, 0 = random")
    ap.add_argument("--adv-panels", action="store_true",
                    help="adversary figure: one panel per similarity instead of one axis")
    args = ap.parse_args()
    os.makedirs(OUT_ABL, exist_ok=True)

    runs = load_runs()
    if runs.empty:
        sys.exit("no *_eval_results.json found in results/mv_out/max_completions/")
    runs.drop(columns="answered").to_csv(os.path.join(OUT, "results_all_runs.csv"), index=False)
    df = load_csv(args.csv)

    sel = runs[(runs.tag == args.tag) & (runs.pt == args.pt)]
    if sel.empty:
        sys.exit(f"no runs with tag={args.tag!r} and pt={args.pt}")
    agg = sel.groupby(["metric", "ties", "k"], as_index=False)[COLS].mean()
    agg.to_csv(os.path.join(OUT, "results_table.csv"), index=False)

    fig_coverage_accuracy(agg, args.eval)
    fig_error_vs_k(agg, args.eval)
    groups, models = load_groups(set(df.columns))
    fig_correct_fragmentation(groups, models, df)

    # ablation: pt1 vs pt0, one seed
    abl = runs[(runs.tag == args.tag) & (runs.seed == args.seed)]
    abl = {(r.metric, r.k, r.pt, r.ties): (r.n_tasks, r.answered[args.eval])
           for r in abl.itertuples()}
    fig_ablation_error_vs_k(abl)
    table_tests_vs_models(abl, df, args.table_ties)
    print(f"done -> {OUT}")


    status = f"{args.eval}_status"
    fig_adversary_error_vs_k(args.pt, args.seed, args.tag, status, single=not args.adv_panels)
    fig_rtests_vs_codet(args.seed, args.tag, status)


if __name__ == "__main__":
    main()
