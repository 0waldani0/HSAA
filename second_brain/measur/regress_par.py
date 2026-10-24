# The regression, run with the steps that do not share build outputs in
# PARALLEL. The structure is dictated by what collides:
#
#   phase A (serial): the gates and the engine library -- they are the
#     targets every other step links against.
#   phase B (parallel): the four suites, the self-test, the C-API probe, the
#     e2e (it rebuilds the gates inside its own target) and the corpus group
#     (the reference corpus, the state round trip and the version
#     diagnostic, which must stay in order among themselves: they share
#     build/weight_bank_corpus_train.state).
#
# What makes this safe: every step writes its own obj/exe paths (the suite
# targets use build\obj_<target>\, the e2e uses build\e2e_test_runner.*), the
# builds carry no /Zi so there is no PDB server to serialise on, and the only
# shared FILE is the trainer state, which the corpus group owns alone.
# Two builds of the SAME target must never overlap -- that is what produced
# the "exit 91" of an earlier run.
import subprocess, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MEAS = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(MEAS, "regress_par_logs")
os.makedirs(LOGS, exist_ok=True)

results = []
lock = __import__("threading").Lock()


def run(name, cmd, timeout=3600):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=timeout)
        rc = r.returncode
        both = ((r.stdout or b"").decode("utf-8", errors="replace") + "\n" +
                (r.stderr or b"").decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
        both = "TIMEOUT after %ss" % timeout
    with open(os.path.join(LOGS, name.replace(" ", "_").replace("/", "_") + ".log"),
              "w", encoding="utf-8", errors="replace") as fh:
        fh.write(both)
    ok = (rc == 0) and ("error C" not in both) and ("GATE FAILED" not in both) \
        and ("[FAIL]" not in both) and ("RESULT] FAILED" not in both)
    with lock:
        results.append((name, rc, ok))
        print("  %-24s exit=%-8s %s (%.0fs)" % (name, rc, "PASS" if ok else "FAIL",
                                                 time.time() - t0), flush=True)
    return ok


def bat(path, timeout=3600):
    run(os.path.basename(path), ["cmd", "/c", path], timeout)


def bat_run(path, timeout=3600):
    """Build and execute a test script (its `run` mode propagates the exit code)."""
    run(os.path.basename(path) + " run", ["cmd", "/c", path, "run"], timeout)


def target(name, timeout=3600):
    run("build.bat " + name, ["cmd", "/c", os.path.join(ROOT, "build.bat"), name], timeout)


print("=== phase A: the gates and the engine library (serial) ===", flush=True)
t0 = time.time()
bat(os.path.join(ROOT, "tests", "build_popcnt_backoff_tests.bat"), 1800)
bat(os.path.join(ROOT, "tests", "build_popcnt_selector_tests.bat"), 1800)
target("brain", 2400)
# The calibration includes the same assoc_sketch.h the engine includes, so the
# recall/false-positive table is generated from shipped math, not copied.
bat(os.path.join(ROOT, "second_brain", "measur", "build_assoc_calib.bat"), 1800)
# The CUDA microbenchmark runs here, alone: its Test 3 asserts a 400 us step
# ceiling, and overlapping it with the parallel phase pushed it to 402-455 us.
bat_run(os.path.join(ROOT, "tests", "build_hssa_m2_tests.bat"), 1800)
print("phase A: %.0fs" % (time.time() - t0), flush=True)

print("=== phase B: the independent steps (parallel) ===", flush=True)
t0 = time.time()


def suite_group():
    target("bridge", 1800)
    run("test_bridge", [os.path.join(ROOT, "build", "test_bridge.exe")], 1800)
    target("vsa_db", 1800)
    run("test_vsa_db", [os.path.join(ROOT, "build", "test_vsa_db.exe")], 1800)
    target("model_mod", 1800)
    run("test_model_mod", [os.path.join(ROOT, "build", "test_model_mod.exe")], 1800)
    target("hssa", 2400)
    run("test_hssa", [os.path.join(ROOT, "build", "test_hssa.exe")], 1800)


def self_test_group():
    run("self-test", [os.path.join(ROOT, "build", "popcnt_brain.exe")], 1800)


def capi_group():
    bat(os.path.join(MEAS, "build_capi_probe.bat"), 1800)
    run("capi_probe", [os.path.join(MEAS, "capi_probe.exe")], 1800)


def e2e_group():
    target("test", 3600)


def corpus_group():
    # ONE training for the three facts (the reference corpus, the state round
    # trip and the format diagnostic): three trainings of the same corpus was
    # three times the compute for three facts.
    run("corpus group", [sys.executable, os.path.join(MEAS, "corpus_group.py")], 3600)


with ThreadPoolExecutor(max_workers=5) as pool:
    futs = [pool.submit(g) for g in (suite_group, self_test_group, capi_group,
                                      e2e_group, corpus_group)]
    for f in futs:
        f.result()
print("phase B: %.0fs" % (time.time() - t0), flush=True)

print()
print("=== phase C: adversarial and benchmark suites (serial) ===", flush=True)
t0 = time.time()
for script in ("build_stress.bat", "build_stress_bench.bat",
               "build_stress_tier5.bat", "build_stress_tier5_adversarial.bat",
               "build_challenger2.bat", "build_hssa_m1_tests.bat",
               "build_hssa_m3_tests.bat", "build_bench_model_mod.bat",
               "build_bench_hot_swap.bat", "build_ptq_validation.bat"):
    bat_run(os.path.join(ROOT, "tests", script), 1800)
print("phase C: %.0fs" % (time.time() - t0), flush=True)

print()
print("=== THE VERDICT ===")
for name, rc, ok in sorted(results):
    print("  %-28s exit=%-8s %s" % (name, rc, "PASS" if ok else "FAIL"))
bad = [n for n, rc, ok in results if not ok]
print()
print("REGRESSION:", "ALL GREEN" if not bad else "RED: " + ", ".join(bad))
sys.exit(0 if not bad else 1)
