"""Kaggle side of the final re-evaluation: models trained on Kaggle (experiment 4 and the seed-0
local-only re-runs E2L_K2, E2L_K3), read from those notebooks' outputs attached as inputs.

    import glob; exec(open(glob.glob("/kaggle/input/**/eval_on_kaggle.py", recursive=True)[0]).read())
"""

import glob
import os
import shutil
import subprocess
import sys

hits = glob.glob("/kaggle/input/**/BUNDLE.txt", recursive=True)
assert hits, "Attach the fedheads-bundle dataset"
BUNDLE = os.path.dirname(hits[0])
print(open(hits[0]).read())
if not os.path.exists("/tmp/repo"):
    shutil.copytree(f"{BUNDLE}/code", "/tmp/repo")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "peft==0.21.2"], check=True)
runs = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.resolved.yaml", recursive=True)
               if os.path.exists(os.path.join(os.path.dirname(p), "best_state.pt"))
               or os.path.isdir(os.path.join(os.path.dirname(p), "client_states"))})
print(len(runs), "runs found")
assert len(runs) == 21, "expected 18 experiment-4 runs and 3 local-only re-runs"
os.makedirs("/kaggle/working/eval", exist_ok=True)
open("/kaggle/working/eval/runs_kaggle.txt", "w").write("\n".join(runs) + "\n")
r = subprocess.run([sys.executable, "/tmp/repo/scripts/eval_saved.py", "--tok-dir", f"{BUNDLE}/tokens",
                    "--runs", "/kaggle/working/eval/runs_kaggle.txt", "--out", "/kaggle/working/eval/eval_kaggle.json"],
                   env=dict(os.environ, USE_TF="0"), capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:])
assert r.returncode == 0
