import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from aggregations import *
import seaborn as sns
from matplotlib.lines import Line2D
import os
import re


def plot_acc_err_abs_mine(result_df, aggr_cols, task, outdir="figures", tag="VPA_varying_tau"):

    total = len(result_df)
    rows = []

    for aggr in aggr_cols:
        is_nan = result_df[aggr].isna()
        is_correct = result_df[aggr] == result_df["correct_label"]
        is_incorrect = (~is_nan) & (~is_correct)

        rows.append({
            "Aggr": aggr,
            "Correct":   is_correct.sum()   / total * 100,
            "Incorrect": is_incorrect.sum() / total * 100,
            "Abstained": is_nan.sum()       / total * 100,
        })

    df_plot = pd.DataFrame(rows).set_index("Aggr")

    # style is scoped to this figure so it neither depends on nor leaks into
    # whatever other plots set globally
    style = {**sns.axes_style("white"), **sns.plotting_context("talk"),
             "font.family": "DejaVu Sans", "axes.linewidth": 1.2}
    with plt.rc_context(style):
        colors = ["#8FAF8B", "#D99C93", "#B4BAC2"]   # muted sage / salmon / slate

        fig, ax = plt.subplots(figsize=(1.9 * len(aggr_cols), 6))

        df_plot.plot(
            kind="bar", stacked=True, ax=ax,
            color=colors, edgecolor="black", linewidth=1.1, width=0.35, zorder=3,
        )

        # ---- labels inside segments ----
        for container in ax.containers:
            labels = [f"{v:.1f}" if v > 3 else "" for v in container.datavalues]
            ax.bar_label(container, labels=labels, label_type="center",
                         color="black", fontsize=13, fontweight="medium",)

        # ---- axes ----
        ax.set_ylim(0, 100)
        ax.set_ylabel("Questions (%)", fontsize=16)
        ax.set_xlabel("")
        ax.set_xticklabels([re.sub(r"\D", "", a) for a in df_plot.index])
        ax.set_xlabel(r"Plurality $\tau$", fontsize=16)
        ax.tick_params(axis="x", rotation=0, labelsize=15, length=0)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="0.9", linewidth=1)
        sns.despine(ax=ax, left=False, bottom=False)

        # ---- legend as a single row on top ----
        handles, labels = ax.get_legend_handles_labels()
        ax.legend( handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 1.01),
            ncol=len(labels), frameon=False,
            fontsize=14, handlelength=1.4, handleheight=1.0,
            columnspacing=2.0, handletextpad=0.7,
        )


        ax.set_title(f"Aggregator Performance on {task}", fontsize=17, pad=58)

        os.makedirs(outdir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{outdir}/{task}_{tag}_abs_error_acc.pdf", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def get_hq_ids(file_path,model_names,treshold):
    ID_COL, LABEL_COL = "question_id", "correct_label"
    df =load_subset(file_path,model_names)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    models = [c for c in df.columns if c not in (ID_COL, LABEL_COL)]
    response_columns = models
    label_column = df.columns[-1]      # The last column

    incorrect_mask = df[response_columns].ne(df[label_column], axis=0)


    df['incorrect_count'] = incorrect_mask.sum(axis=1)

    hard_questions_df = df[df['incorrect_count'] >= treshold]


    target_question_ids = hard_questions_df[ID_COL].tolist()

    #print(f"Found {len(target_question_ids)} questions where {treshold} or more models were incorrect.")
    #print("Question IDs:", target_question_ids)
    return target_question_ids


def load_subset(path, new_names, id_col="question_id", label_col="correct_label"):
    """Read the results CSV, keep only the given model columns, and rename them."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    # strip stray whitespace from headers and cells
    df.columns = df.columns.str.strip()
    df = df.apply(lambda s: s.str.strip())

    missing = [m for m in new_names if m not in df.columns]
    if missing:
        raise KeyError(f"columns not found in {path}: {missing}")

    cols = [id_col] + list(new_names) + [label_col]
    out = df[cols].rename(columns=new_names)

    short = list(new_names.values())
    out = out.replace({"": pd.NA, "N/A": pd.NA, "NA": pd.NA})
    #out[short] = out[short].replace({"": pd.NA, "N/A": pd.NA, "NA": pd.NA})

    return out


AGGR_COL = re.compile(r"MV|VPA\d+")   # columns written by create_aggr_df


def create_aggr_df(df, req_votes, ID_COL, LABEL_COL, CHOICES):
    # work on a copy and never count earlier MV/VPA{k} outputs as voters,
    # so re-running a cell gives the same result
    df = df.copy()
    models = [c for c in df.columns
              if c not in (ID_COL, LABEL_COL) and not AGGR_COL.fullmatch(c)]
    rows_mv = [aggr_baseline(r, CHOICES) for r in df[models].to_numpy()]
    df["MV"] = [r[0] for r in rows_mv]
    for rv in req_votes:
        vpa = [aggregate(r, CHOICES, rv) for r in df[models].to_numpy()]
        df[f"VPA{rv}"]  = [r[0] for r in vpa]

    return df


def calculate_relative_accuracies(df,method_name,base_models):
    """
    Calculates the accuracy on the questions answered by the VPA compared to base model accuracy on same questions.
    """
    models = base_models + [method_name]
    if method_name not in df.columns:
        raise ValueError(f"Model '{method_name}' not found in the DataFrame.")
        
    total_questions = len(df)
    
    # Filter out rows where the reference model did not answer (is NaN)
    df_subset = df[df[method_name].notna()]
    answered_count = len(df_subset)
    
    #  Compute Coverage (# questions answered by model / total questions)
    coverage = answered_count / total_questions if total_questions > 0 else 0.0
    
    # Initialize the results dictionary
    results = {'plurality': method_name[3],'coverage': coverage}
    
    # Identify model columns by excluding the ID and correct_label columns
    #models = base_models #[col for col in df.columns if col not in ['question_id', 'correct_label']]
    
    # If the reference model answered 0 questions, return 0 for all accuracies
    if answered_count == 0:
        for model in models:
            results[model] = 0.0
        return results

    #  Compute accuracy for all models on the reference model's subset
    for model in models:

        correct_matches = (df_subset[model] == df_subset['correct_label'])

        accuracy = correct_matches.mean() *100
        model_key = "Ours" if model.startswith("VPA") else model
        results[model_key] = accuracy
        
    return results


def plot_accuracies_vs_coverage(results_list, task, ylim=None):
    """
    Takes a list of result dicts (each with 'plurality', 'coverage', and
    model accuracies in %) and plots each model's accuracy vs. plurality τ.
    """
    if not results_list:
        print("The results list is empty.")
        return

    df_plot = pd.DataFrame(results_list).sort_values(by='plurality')
    models = [c for c in df_plot.columns if c not in ['coverage', 'plurality']]

    sns.set_style("whitegrid")
    tab20 = sns.color_palette("tab20")
    colors = (tab20[::2] + tab20[1::2])[:len(models)]
    markers = ['o', 's', '>', 'D', 'v', 'p', '<', '^', '*', 'P']
    linestyles = ['-', '--', '-.', ':']

    # base models: thin coloured lines; ours: thick black line on top
    styles = {}
    for i, model in enumerate(m for m in models if m != "Ours"):
        styles[model] = dict(marker=markers[i % len(markers)], markersize=6,
                             linestyle=linestyles[i % len(linestyles)], linewidth=1.5,
                             color=colors[i], alpha=0.9, zorder=2)
    if "Ours" in models:
        styles["Ours"] = dict(marker="*", markersize=14, linestyle="-", linewidth=3,
                              color="black", alpha=1.0, zorder=10)

    fig = plt.figure(figsize=(6, 4), dpi=120)

    for model, st in styles.items():
        plt.plot(df_plot['plurality'], df_plot[model], markeredgecolor='black',
                 label=model.removesuffix(":free"), **st)

    plt.xlabel(r'Plurality $\tau$', fontsize=16, labelpad=10)
    plt.ylabel('Accuracy (%) on Answered', fontsize=16, labelpad=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if ylim is None:   # pad the data range out to multiples of 5
        vals = df_plot[models].to_numpy(dtype=float)
        ylim = (5 * np.floor(np.nanmin(vals) / 5) - 2,
                min(5 * np.ceil(np.nanmax(vals) / 5) + 2, 101))   # accuracy never exceeds 100
    plt.ylim(*ylim)
    plt.grid(True, linestyle='--', alpha=0.5)

    legend_elements = [
        Line2D([0], [0], markeredgecolor='black', label=model.removesuffix(":free"),
               **{**st, "markersize": st["markersize"] + 4})
        for model, st in styles.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        fontsize=13,
        frameon=True,
        columnspacing=1.5,
    )

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/Accuracy_vs_plurality_on_{task}.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


META = {"question_id", "correct_label"}


def add_mimic(df, seed=0, random_pick=False):
    """Add a Byzantine voter "mimic" that copies the most common wrong answer
    (ties broken at random). With random_pick=True also add "random_pick",
    a voter that copies a random model's answer."""
    rng = np.random.default_rng(seed)
    model_cols = [c for c in df.columns if c not in META]
    out = df.copy()

    mimic, rand = [], []
    for _, row in df.iterrows():
        votes = row[model_cols].dropna()
        wrong = votes[votes != row["correct_label"]]

        if len(wrong):
            counts = wrong.value_counts()
            top = counts[counts == counts.max()].index.tolist()
            mimic.append(rng.choice(top))          # tie -> random among tied
        else:
            mimic.append(np.nan)

        rand.append(rng.choice(votes.values) if len(votes) else np.nan)

    out["mimic"] = mimic
    if random_pick:
        out["random_pick"] = rand
    return out


def accuracy_vs_tau(result_df, taus, base_models):
    """Accuracy on the questions our aggregator answers at each plurality tau,
    for our aggregator and for every base model on the same questions."""
    return [calculate_relative_accuracies(result_df, f"VPA{k}", base_models) for k in taus]
