#!/bin/bash
# Re-evaluate saved models for the final analysis (scripts/eval_saved.py); no training. Submit with
#     sbatch ada/eval.sbatch
# The run list is built here from ~/fedheads/out: experiment 2 FedAvg and centralized runs, and the
# local-only runs that saved per-client models (seeds 1 and 2, and the seed-0 re-runs in E2L_01).
# Finished runs are skipped, so resubmitting continues.
#SBATCH --job-name=fh_eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=u22
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=$HOME/fedheads/logs/slurm_%x_%j.out

set -euo pipefail
source $HOME/fedheads/code/slurm/common.sh
[ -f "$HOME/envs/fedheads/.ready" ] || { echo "~/envs/fedheads not ready"; exit 1; }
mkdir -p "$PROJ/eval"
{
    ls -d "$PROJ"/out/E2_[01][0-9]/runs/*-fedavg-* "$PROJ"/out/E2_[01][0-9]/runs/*-centralized-*
    for d in "$PROJ"/out/E2_[01][0-9]/runs/*-local_only-* "$PROJ"/out/E2L_01/runs/*; do
        [ -d "$d/client_states" ] && echo "$d"
    done
} | sort > "$PROJ/eval/runs_ada.txt"
echo "=== runs to evaluate: $(wc -l < "$PROJ/eval/runs_ada.txt")"
python scripts/eval_saved.py --tok-dir "$PROJ/tokens" --runs "$PROJ/eval/runs_ada.txt" --out "$PROJ/eval/eval_ada.json" \
    --local-files-only
