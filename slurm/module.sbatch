#!/bin/bash
# One experiment module on Ada. Submit with
#     sbatch --export=ALL,MODULE=<name> ada/module.sbatch
# Finished runs are skipped and unfinished ones resume, so resubmitting continues the module.
#SBATCH --job-name=fh_module
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=u22
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti        # ~/envs/qm's torch 2.11 (cu128) needs sm_75+; GTX 1080 Ti (sm_61) fails
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --output=$HOME/fedheads/logs/slurm_%x_%j.out

set -euo pipefail
: "${MODULE:?set MODULE with --export=ALL,MODULE=<name>}"
source $HOME/fedheads/code/slurm/common.sh
[ -f "$HOME/envs/fedheads/.ready" ] || { echo "~/envs/fedheads not ready: run ada/env.sbatch first"; exit 1; }
echo "=== module $MODULE"
python -c "import torch, transformers, peft; print('versions: torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"

python scripts/run_module.py --module "$MODULE" --tok-dir "$PROJ/tokens" --out-dir "$PROJ/out/$MODULE" \
    --session-start "$JOB_START"
python -c "import json; r=json.load(open('$PROJ/out/$MODULE/module_report.json')); print('summary', r['summary'], r.get('hours_total'), 'h')"
