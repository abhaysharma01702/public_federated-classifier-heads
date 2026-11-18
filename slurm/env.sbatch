#!/bin/bash
# Environment job: build and check ~/envs/fedheads, nothing else.
# No data, no training. Expected about 10 min; the limit asked is 45 min.
#SBATCH --job-name=fh_env
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=u22
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti        # ~/envs/qm's torch 2.11 (cu128) needs sm_75+; GTX 1080 Ti (sm_61) fails
#SBATCH --mem=28G
#SBATCH --time=00:45:00
#SBATCH --output=$HOME/fedheads/logs/slurm_%x_%j.out

set -euo pipefail
source $HOME/fedheads/code/slurm/common.sh

# 0. Internet from the compute node (pip and the XLM-R download need it).
code=$(curl -sS -o /dev/null --max-time 30 -w "%{http_code}" https://huggingface.co) || code=none
echo "=== internet check: huggingface.co HTTP $code"
[ "$code" != none ] || { echo "NO INTERNET on $(hostname): XLM-R and packages must be uploaded instead"; exit 1; }

# 1. The venv, made once. ~/envs/qm is itself a venv, so its PyTorch is linked in with a .pth file
#    (a --system-site-packages venv would only see the base Python). A half-made env stops the job;
#    nothing is deleted without the user's yes.
if [ ! -f "$HOME/envs/fedheads/.venv_made" ] && [ ! -f "$HOME/envs/fedheads/.complete" ]; then
    if [ -e "$HOME/envs/fedheads" ]; then
        echo "~/envs/fedheads exists but is half made; ask the user before removing it"; exit 1
    fi
    echo "=== making ~/envs/fedheads on top of ~/envs/qm"
    QM_SITE=$("$HOME/envs/qm/bin/python" -c "import site; print(site.getsitepackages()[0])")
    "$HOME/envs/qm/bin/python" -m venv "$HOME/envs/fedheads"
    echo "$QM_SITE" > "$("$HOME/envs/fedheads/bin/python" -c "import site; print(site.getsitepackages()[0])")/qm_packages.pth"
    touch "$HOME/envs/fedheads/.venv_made"
fi
set +u; source "$HOME/envs/fedheads/bin/activate"; set -u
echo "=== python: $(command -v python)"
python -c "import torch; print('qm torch visible:', torch.__version__, torch.__file__)"

# 2. Packages from ada/requirements-ada.txt. Every package qm has is pinned, so pip only adds what
#    is missing and never changes torch, numpy or pandas (and cannot pull a second multi-GB torch).
"$HOME/envs/qm/bin/python" -m pip list --format=freeze | grep -v -i -E '^(pip|setuptools|wheel)==' > "$S/qm_constraints.txt"
pip install --quiet -c "$S/qm_constraints.txt" -r ada/requirements-ada.txt
pip check || echo "(pip check reported the conflicts above)"

# 3. Every module of the code imports, on this Python.
python - <<'EOF'
import importlib, pkgutil, sys
sys.path[:0] = ["src", "scripts"]
import fedheads
mods = ["fedheads"] + [m.name for m in pkgutil.walk_packages(fedheads.__path__, "fedheads.")]
mods += ["prepare_data", "tokenize_data", "run_one", "run_module", "calibrate", "make_modules"]
for m in mods:
    importlib.import_module(m)
import torch, transformers, peft, pandas, pyarrow, sklearn, numpy
print(f"imports ok: {len(mods)} modules")
print("versions: python", sys.version.split()[0], "| torch", torch.__version__, "cuda", torch.version.cuda,
      "| transformers", transformers.__version__, "| peft", peft.__version__, "| numpy", numpy.__version__,
      "| pandas", pandas.__version__, "| pyarrow", pyarrow.__version__, "| scikit-learn", sklearn.__version__)
x = torch.randn(256, 256, device="cuda")
with torch.autocast("cuda", dtype=torch.float16):
    print("fp16 matmul on", torch.cuda.get_device_name(0), "ok:", bool(torch.isfinite(x @ x).all()))
EOF

# 4. Full unit tests (CPU; about 3 min).
(cd tests && python -m unittest 2>&1)

touch "$HOME/envs/fedheads/.ready"
echo "=== env ready: ~/envs/fedheads/.ready written"
