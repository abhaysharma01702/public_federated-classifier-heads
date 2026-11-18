"""Write kaggle/run_module_bundle.ipynb: code and token tables come from one private
Kaggle dataset (the bundle), not a git clone, so no push is needed. A watchdog inside the session
stops it if no progress is written for 30 minutes, so a stalled session cannot burn quota even
when nobody is watching from the laptop.

    python kaggle/make_notebook.py
"""

import json
from pathlib import Path

CELLS = [
("markdown", """# Experiment module runner

Code and token tables come from the private dataset `fedheads-bundle`; the bundle's
`BUNDLE.txt` names the commit. GPU T4 x2, internet on (model download and pip), private.
Output: `/kaggle/working/<MODULE>`. To continue a module, attach the previous version's
output as input: finished runs are skipped, unfinished ones resume from checkpoints."""),
("code", """import time
SESSION_START = time.time()   # the session guard counts from here
MODULE = "E5P_K1"
STALL_MIN = 30                # watchdog: stop if nothing in the output changes for this long
import glob, os, shutil, subprocess, sys, json
OUT_DIR = f"/kaggle/working/{MODULE}"
hits = glob.glob("/kaggle/input/**/BUNDLE.txt", recursive=True)
assert hits, "Attach the fedheads-bundle dataset"
BUNDLE = os.path.dirname(hits[0])
print(open(hits[0]).read())
if not os.path.isdir(f"{BUNDLE}/code"):   # if Kaggle kept the folders as zip files
    for z in ("code", "tokens"):
        shutil.unpack_archive(f"{BUNDLE}/{z}.zip", f"/tmp/bundle/{z}")
    BUNDLE = "/tmp/bundle"
TOK_DIR = f"{BUNDLE}/tokens"
assert os.path.exists(f"{TOK_DIR}/tokens_report.json"), "token tables missing from the bundle"
if not os.path.exists("/tmp/repo"):
    shutil.copytree(f"{BUNDLE}/code", "/tmp/repo")
prev = [p for p in glob.glob(f"/kaggle/input/**/{MODULE}/module_report.json", recursive=True)]
if prev and not os.path.exists(OUT_DIR):
    shutil.copytree(os.path.dirname(prev[0]), OUT_DIR)
    print("continuing from", os.path.dirname(prev[0]))"""),
("code", """# Same library versions as Ada (env.json of job 82849)
!pip install -q "transformers==4.57.6" "peft==0.21.2" 2>&1 | tail -2
import torch, transformers, peft
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__, "| GPUs", torch.cuda.device_count())
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
("code", """r = subprocess.run("cd /tmp/repo/tests && USE_TF=0 python -m unittest 2>&1 | tail -4", shell=True, capture_output=True, text=True)
print(r.stdout)
assert "OK" in r.stdout, "unit tests failed: stopping before any training"
print(f"setup took {(time.time() - SESSION_START) / 60:.1f} min")"""),
("code", """# Run the module with a watchdog. Progress is echoed here, so `kaggle kernels logs` shows it.
os.makedirs(OUT_DIR, exist_ok=True)
env = dict(os.environ, USE_TF="0", PYTHONUNBUFFERED="1")
cmd = [sys.executable, "/tmp/repo/scripts/run_module.py", "--module", MODULE, "--tok-dir", TOK_DIR,
       "--out-dir", OUT_DIR, "--session-start", str(SESSION_START)]
runner_log = open(f"{OUT_DIR}/runner.log", "a")
proc = subprocess.Popen(cmd, stdout=runner_log, stderr=subprocess.STDOUT, env=env)
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
                if "round" in line or "===" in line or "done in" in line or "Error" in line or "nan" in line.lower():
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
print("runner exit code", proc.returncode, f"| session {(time.time() - SESSION_START) / 3600:.2f} h")"""),
("code", """r = json.load(open(f"{OUT_DIR}/module_report.json"))
print(json.dumps({k: r[k] for k in ("module", "gpus", "summary", "hours_total") if k in r}, indent=1))
for row in r["runs"]:
    print(row)
for p in sorted(glob.glob(f"{OUT_DIR}/runs/*/summary.json")):
    s = json.load(open(p))
    print(s["run_id"], {k: v for k, v in s.items() if k.startswith("test_best_")})"""),
]


def main():
    cells = []
    for kind, src in CELLS:
        c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = Path(__file__).with_name("run_module_bundle.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
