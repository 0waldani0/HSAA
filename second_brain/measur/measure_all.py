# The measurement pass, run in PARALLEL: each configuration is built in its
# own scratch directory and runs on ONE core, and the host has 16 cores, so
# six configurations at once cost the same wall time as one.
import os, re, subprocess, shutil, json, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MEASUR = os.path.join(os.environ["TEMP"], "opencode", "hssa_measur")
TMP = os.path.join(os.environ["TEMP"], "opencode")
HERE = os.path.dirname(os.path.abspath(__file__))
TRN = os.path.join(HERE, "data", "bpe_train.txt")
HOLD = os.path.join(HERE, "data", "bpe_holdout.txt")
REPORT = os.path.join(TMP, "measure_all_report.txt")
JSONOUT = os.path.join(TMP, "measure_all_report.json")

VSWHERE = os.path.join(os.environ["ProgramFiles(x86)"], "Microsoft Visual Studio",
                       "Installer", "vswhere.exe")
r = subprocess.run([VSWHERE, "-latest", "-products", "*", "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property", "installationPath"], capture_output=True, text=True)
inst = r.stdout.strip().splitlines()[0]
vcvars = os.path.join(inst, "VC", "Auxiliary", "Build", "vcvars64.bat")
FLAGS = "/nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /wd4068 /utf-8"

results = []
lock_results = __import__("threading").Lock()


def write_report():
    with open(REPORT, "w", encoding="utf-8", errors="replace") as fh:
        fh.write("=== one measurement pass over the shipped engine ===\n")
        fh.write("corpus: %s / %s, 6 epochs, seed 424242\n" % (TRN, HOLD))
        fh.write("%-26s %10s %10s %12s %s\n"
                 % ("config", "acc%", "cov%", "cov_n", "E[acc(1..3)]"))
        for r_ in sorted(results, key=lambda x: x["label"]):
            fh.write("%-26s %10s %10s %12s %s\n" % (
                r_["label"], r_["holdout_accuracy_percent"], r_["coverage_percent"],
                r_["coverage_contexts"],
                ",".join("%.4f" % x for x in r_["expected_accepted"][:3])))
    with open(JSONOUT, "w", encoding="utf-8", errors="replace") as fh:
        json.dump(sorted(results, key=lambda x: x["label"]), fh, indent=1)


def build(tag, defines):
    d = os.path.join(MEASUR, "ma_" + tag)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(os.path.join(ROOT, "second_brain")):
        if f.endswith(".h") or f == "popcnt_brain.cpp":
            shutil.copy2(os.path.join(ROOT, "second_brain", f), d)
    shutil.copy2(os.path.join(HERE, "acceptance_harness.cpp"), d)
    bat = os.path.join(d, "b.bat")
    with open(bat, "w", newline="\r\n") as fh:
        fh.write('@echo off\r\ncall "%s" >nul\r\n' % vcvars)
        fh.write('cl.exe %s %s /I"%s" /I"%s" "%s" /Fe:"%s\\a.exe" /Fo:"%s\\a.obj" '
                 '/link /STACK:33554432\r\nexit /b %%ERRORLEVEL%%\r\n'
                 % (FLAGS, " ".join(defines), d, d,
                    os.path.join(d, "acceptance_harness.cpp"), d, d))
    b = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True, timeout=3600)
    if b.returncode != 0:
        print("  BUILD FAILED:", tag, flush=True)
        print(b.stdout[-800:], flush=True)
        return None
    return os.path.join(d, "a.exe")


def one(tag, defines, label):
    t0 = time.time()
    exe = build(tag, defines)
    if not exe:
        return None
    g = subprocess.run([exe, TRN, HOLD, "6", "424242"], cwd=os.path.dirname(exe),
                       capture_output=True, text=True, timeout=7200)
    so = g.stdout
    acc = re.search(r'"holdout_accuracy_percent":\s*([0-9.]+)', so)
    cov = re.search(r'"heldout_shortlist_coverage_percent":\s*([0-9.]+)', so)
    covn = re.search(r'"heldout_shortlist_coverage_total":\s*(\d+)', so)
    eacc = re.search(r'"expected_accepted_by_draft_len":\s*\[([^\]]*)\]', so)
    e = []
    if eacc:
        for v in eacc.group(1).split(","):
            mm = re.search(r'"expected_accepted_tokens":\s*([0-9.]+)', v)
            if mm:
                e.append(float(mm.group(1)))
    rec = {"label": label,
           "holdout_accuracy_percent": float(acc.group(1)) if acc else None,
           "coverage_percent": float(cov.group(1)) if cov else None,
           "coverage_contexts": int(covn.group(1)) if covn else None,
           "expected_accepted": e[:6],
           "wall_s": round(time.time() - t0, 1)}
    with lock_results:
        results.append(rec)
        write_report()
    print("  %-26s acc=%s  cov=%s (n=%s)  E[acc(1..3)]=%s  wall=%ss"
          % (label, rec["holdout_accuracy_percent"], rec["coverage_percent"],
             rec["coverage_contexts"],
             ",".join("%.4f" % x for x in e[:3]), rec["wall_s"]), flush=True)
    return rec


JOBS = []
for w in (64, 256, 1024, 4096):
    JOBS.append(("w%d" % w, ["/DHSSA_BACKOFF_COUNT_WEIGHT=%d" % w], "weight=%d k=32" % w))
for k in (4, 32, 64, 256, 1024):
    JOBS.append(("k%d" % k, ["/DHSSA_BACKOFF_COUNT_WEIGHT=1024",
                             "/DHSSA_BACKOFF_SHORTLIST_K=%d" % k], "weight=1024 k=%d" % k))
JOBS.append(("plain", ["/DHSSA_DISABLE_BACKOFF"], "plain (no backoff)"))

print("%d configurations, six at a time" % len(JOBS), flush=True)
t0 = time.time()
with ThreadPoolExecutor(max_workers=6) as pool:
    futs = [pool.submit(one, tag, defines, label) for tag, defines, label in JOBS]
    for f in futs:
        f.result()
write_report()
print("WROTE %s" % REPORT, flush=True)
print("WROTE %s" % JSONOUT, flush=True)
print("total wall: %.0fs" % (time.time() - t0), flush=True)
