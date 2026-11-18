"""Everything one Kaggle session needs, for a notebook with a single cell. The cell sets SESSION_START and MODULE, then runs this file from the attached
fedheads-bundle dataset:

    import time, glob; SESSION_START = time.time(); MODULE = "E4_K3"
    exec(open(glob.glob("/kaggle/input/**/run_on_kaggle.py", recursive=True)[0]).read())

Same steps as kaggle/run_module_bundle.ipynb: bundle check, Ada's library versions, unit tests
(stop before training if any fails), the module with the 10.5 h guard, a 30-minute no-progress
watchdog, and a summary. Output: /kaggle/working/<MODULE>.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

SESSION_START = globals().get("SESSION_START", time.time())
MODULE = globals()["MODULE"]
STALL_MIN = 30
OUT_DIR = f"/kaggle/working/{MODULE}"

# 1. Bundle
hits = glob.glob("/kaggle/input/**/BUNDLE.txt", recursive=True)
assert hits, "Attach the fedheads-bundle dataset (Add Input)"
BUNDLE = os.path.dirname(hits[0])
print(open(hits[0]).read())
if not os.path.isdir(f"{BUNDLE}/code"):
    for z in ("code", "tokens"):
        shutil.unpack_archive(f"{BUNDLE}/{z}.zip", f"/tmp/bundle/{z}")
    BUNDLE = "/tmp/bundle"
TOK_DIR = f"{BUNDLE}/tokens"
assert os.path.exists(f"{TOK_DIR}/tokens_report.json"), "token tables missing from the bundle"
assert os.path.exists(f"{BUNDLE}/code/configs/modules/{MODULE}.yaml"), f"module {MODULE} not in this bundle version"
if not os.path.exists("/tmp/repo"):
    shutil.copytree(f"{BUNDLE}/code", "/tmp/repo")

# 2. Libraries (Ada's versions) and GPUs
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "peft==0.21.2"], check=True)
out = subprocess.run([sys.executable, "-c", "import torch, transformers, peft; print('torch', torch.__version__, "
                      "'| transformers', transformers.__version__, '| peft', peft.__version__, '| GPUs', torch.cuda.device_count())"],
                     capture_output=True, text=True)
print(out.stdout, out.stderr[-500:])
assert "GPUs 2" in out.stdout, "Set Accelerator to GPU T4 x2 (Settings, Accelerator)"

# 3. Unit tests
r = subprocess.run("cd /tmp/repo/tests && USE_TF=0 python -m unittest 2>&1 | tail -4", shell=True, capture_output=True, text=True)
print(r.stdout)
assert "OK" in r.stdout, "unit tests failed: stopping before any training"
print(f"setup took {(time.time() - SESSION_START) / 60:.1f} min", flush=True)

# 4. The module, with a watchdog
os.makedirs(OUT_DIR, exist_ok=True)
env = dict(os.environ, USE_TF="0", PYTHONUNBUFFERED="1")
cmd = [sys.executable, "/tmp/repo/scripts/run_module.py", "--module", MODULE, "--tok-dir", TOK_DIR,
       "--out-dir", OUT_DIR, "--session-start", str(SESSION_START)]
proc = subprocess.Popen(cmd, stdout=open(f"{OUT_DIR}/runner.log", "a"), stderr=subprocess.STDOUT, env=env)
seen = {}


def newest_change():
    t = 0.0
    for root, _, files in os.walk(OUT_DIR):
        for f in files:
            try:
                t = max(t, os.path.getmtime(os.path.join(root, f)))
            except OSError:
                pass
    return t


def echo_new_lines():
    for path in sorted(glob.glob(f"{OUT_DIR}/*.log")):
        with open(path, errors="replace") as fh:
            fh.seek(seen.get(path, 0))
            for line in fh:
                if any(k in line for k in ("round", "===", "done in", "Error")) or "nan" in line.lower():
                    print(os.path.basename(path), line.rstrip()[:300], flush=True)
            seen[path] = fh.tell()


while proc.poll() is None:
    time.sleep(60)
    echo_new_lines()
    idle = (time.time() - newest_change()) / 60
    if idle > STALL_MIN:
        print(f"=== WATCHDOG: no progress for {idle:.0f} min; stopping the module", flush=True)
        proc.terminate(); time.sleep(30); proc.kill()
        break
echo_new_lines()
print("runner exit code", proc.returncode, f"| session {(time.time() - SESSION_START) / 3600:.2f} h")

# 5. Summary
rep = json.load(open(f"{OUT_DIR}/module_report.json"))
print(json.dumps({k: rep[k] for k in ("module", "gpus", "summary", "hours_total") if k in rep}, indent=1))
for row in rep["runs"]:
    print(row)
for p in sorted(glob.glob(f"{OUT_DIR}/runs/*/summary.json")):
    s = json.load(open(p))
    print(s["run_id"], {k: v for k, v in s.items() if k.startswith("test_best_")})
