#!/bin/bash
# Second Ada job, after ada/env.sbatch: data check, tokenizing,
# ADA_TEST module with the speed benchmark. Expected about 30 min; the limit asked is 2 h.
#SBATCH --job-name=fh_setup
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=u22
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti        # ~/envs/qm's torch 2.11 (cu128) needs sm_75+; GTX 1080 Ti (sm_61) fails
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=$HOME/fedheads/logs/slurm_%x_%j.out

set -euo pipefail
source $HOME/fedheads/code/slurm/common.sh

# The env must have passed ada/env.sbatch.
[ -f "$HOME/envs/fedheads/.ready" ] || { echo "~/envs/fedheads not ready: run ada/env.sbatch first"; exit 1; }
python -c "import torch, transformers, peft, sklearn; print('versions: torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| scikit-learn', sklearn.__version__)"

# 2. Clean tables to scratch, MD5 checked against the laptop's data_report.json, then tokenized.
cp "$PROJ"/data_clean/*.parquet "$PROJ"/data_clean/data_report.json "$S"/
python scripts/tokenize_data.py --clean "$S" --out "$S/tokens" --md5-from "$S/data_report.json"
mkdir -p "$PROJ/tokens"
cp -r "$S/tokens/." "$PROJ/tokens/"

# 4. ADA_TEST: speed benchmark, then FedAvg, local-only and centralized, 2 rounds each.
python scripts/run_module.py --module ADA_TEST --tok-dir "$PROJ/tokens" --out-dir "$PROJ/out/ADA_TEST" \
    --session-start "$JOB_START"
python -c "import json; r=json.load(open('$PROJ/out/ADA_TEST/module_report.json')); print('summary', r['summary'], r.get('hours_total'), 'h')"
