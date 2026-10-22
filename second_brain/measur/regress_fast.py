# The FAST regression lane: the mechanical checks in about a minute.
#
# What it costs, and why it is not the release gate:
#   - the compile is the expensive part of a rebuild, so a target is only
#     rebuilt when its exe is older than the sources it is built from;
#   - everything that does not share build outputs runs in parallel;
#   - the corpus check trains ONCE for 3 epochs (the release lane uses the
#     full 12/12/6 epochs at three seeds).
#
# What it does NOT cover: the full native e2e suite (184 cases, each
# training its own small model -- that is the slow lane's job, run with
# --with-e2e), and the full-epoch corpus accuracy.
#
#   python regress_fast.py            the fast lane
#   python regress_fast.py --with-e2e the fast lane plus the e2e suite
import subprocess, os, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SB = os.path.join(ROOT, "second_brain")
TESTS = os.path.join(ROOT, "tests")
MEAS = os.path.dirname(os.path.abspath(__file__))
WITH_E2E = "--with-e2e" in sys.argv
results = []
lock = __import__("threading").Lock()


def newest(paths):
    t = 0.0
    for p in paths:
        if os.path.isfile(p):
            t = max(t, os.path.getmtime(p))
    return t


def run(name, cmd, timeout=1800, skip_if_fresh=None):
    if skip_if_fresh:
        exe, sources = skip_if_fresh
        if os.path.isfile(exe) and os.path.getmtime(exe) >= newest(sources):
            with lock:
                results.append((name, "up-to-date", True))
                print("  %-26s up to date (no rebuild)" % name, flush=True)
            return True
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=timeout)
        rc = r.returncode
        both = ((r.stdout or b"").decode("utf-8", errors="replace") + "\n" +
                (r.stderr or b"").decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired:
        rc, both = "TIMEOUT", "TIMEOUT"
    ok = (rc == 0) and ("error C" not in both) and ("GATE FAILED" not in both) \
        and ("[FAIL]" not in both) and ("RESULT] FAILED" not in both)
    with lock:
        results.append((name, rc, ok))
        print("  %-26s exit=%-8s %s (%.0fs)" % (name, rc, "PASS" if ok else "FAIL",
                                                 time.time() - t0), flush=True)
    return ok


def engine_sources():
    s = [os.path.join(SB, f) for f in os.listdir(SB)
         if f.endswith(".h") or f.endswith(".cpp")]
    return s


def gate(name, bat, exe, src):
    run(name, ["cmd", "/c", bat], 1800,
        (exe, [os.path.join(TESTS, src)] + engine_sources()))


def suite(name, target, exe, extra=()):
    run("build " + name, ["cmd", "/c", os.path.join(ROOT, "build.bat"), target], 1800,
        (exe, [os.path.join(ROOT, "tests", "test_%s.cpp" % name)] + list(extra)))
    run("run " + name, [exe], 1800)


print("=== the fast lane ===", flush=True)
t0 = time.time()

# group 1: the two gates (they compile the engine into their own obj files)
gates = [
    ("backoff gate", os.path.join(TESTS, "build_popcnt_backoff_tests.bat"),
     os.path.join(ROOT, "build", "test_popcnt_backoff.exe"), "test_popcnt_backoff.cpp"),
    ("selector gate", os.path.join(TESTS, "build_popcnt_selector_tests.bat"),
     os.path.join(ROOT, "build", "test_popcnt_selector.exe"), "test_popcnt_selector.cpp"),
]

# group 2: the engine library and everything that links it
def engine_and_suites():
    run("build brain", ["cmd", "/c", os.path.join(ROOT, "build.bat"), "brain"], 2400,
        (os.path.join(ROOT, "build", "popcnt_brain.exe"), engine_sources()))
    run("self-test", [os.path.join(ROOT, "build", "popcnt_brain.exe")], 1800)
    suite("bridge", "bridge", os.path.join(ROOT, "build", "test_bridge.exe"))
    suite("vsa_db", "vsa_db", os.path.join(ROOT, "build", "test_vsa_db.exe"))
    suite("model_mod", "model_mod", os.path.join(ROOT, "build", "test_model_mod.exe"))
    suite("hssa", "hssa", os.path.join(ROOT, "build", "test_hssa.exe"))


# group 3: the C-API probe
def capi():
    run("build capi", ["cmd", "/c", os.path.join(MEAS, "build_capi_probe.bat")], 1800)
    run("capi_probe", [os.path.join(MEAS, "capi_probe.exe")], 1800)


# group 4: the corpus check (ONE training for 3 epochs, then the round trip
# and the format diagnostic on the state it wrote)
def corpus():
    run("corpus group", [sys.executable, os.path.join(MEAS, "corpus_group.py"), "--fast"],
        3600)


# group 5: the native e2e suite (the slow lane's tail)
def e2e():
    run("build.bat test", ["cmd", "/c", os.path.join(ROOT, "build.bat"), "test"], 3600)


jobs = [lambda: [gate(*g) for g in gates],
        engine_and_suites, capi, corpus]
if WITH_E2E:
    jobs.append(e2e)

with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
    for f in [pool.submit(j) for j in jobs]:
        f.result()

print()
print("=== THE VERDICT ===")
for name, rc, ok in sorted(results):
    print("  %-28s exit=%-10s %s" % (name, rc, "PASS" if ok else "FAIL"))
bad = [n for n, rc, ok in results if not ok]
print()
print("FAST LANE:", "ALL GREEN" if not bad else "RED: " + ", ".join(bad))
print("wall: %.0fs" % (time.time() - t0))
sys.exit(0 if not bad else 2)
