#!/usr/bin/env bash
# HumanEval paper figures and table.
#
#   ./make_figures.sh            figures + table from the evaluated runs in results/
#   ./make_figures.sh --select   first redo the selection runs they are built from
#
# Writes
#   figures/HumanEval_error_vs_plurality.pdf
#   figures/HumanEval_coverage_accuracy.pdf
#   figures/HumanEval_correct_fragmentation.pdf
#   figures/HumanEval_adversary_error_vs_tau.pdf     f = 0, 1, 2
#   figures/HumanEval_rtests_vs_codet.pdf            f = 0, 1, 2
#   figures/ablation/ablation_error_vs_tau.pdf
#   figures/ablation/table_tests_vs_models.tex
set -euo pipefail
cd "$(dirname "$0")"
CSV=data/humaneval_plus_results.csv

if [[ "${1:-}" == "--select" ]]; then
    # clean selection, every similarity: results/mv_out/
    python selection/vpa.py
    # copy adversary (f copies of a minority member): results/adv_out/
    python selection/byzantine_he.py --metrics tests,crosshair --copies 1,2 --ks 2,3,6,7 --pt 1 --ties 1
    python selection/byzantine_he.py --metrics tests --copies 0,1,2 --ks 1 --pt 1 --ties 0
    python selection/codeT.py          --metrics tests --copies 0,1,2 --ks 1 --ties 0
    echo
    echo "Selection done. Score every new .jsonl with EvalPlus (README, section 4),"
    echo "then run ./make_figures.sh without --select."
    exit 0
fi

python gen_figures.py --csv "$CSV"   # every figure + the table
