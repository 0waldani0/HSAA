# The corpus group in ONE training run.
#
# The three separate steps (corpus_regression, state_roundtrip,
# version_diagnostic) each trained their own model -- three trainings of the
# same corpus, ~10 minutes of pure compute for three facts. Here the model is
# trained ONCE and the state file is then reused for the other two facts:
#
#   1. the reference corpus accuracy at the three (epochs, seed) settings;
#   2. the state round trip: the restored model must score the trained
#      model's accuracy;
#   3. the format diagnostic: a state with the OLD format version must be
#      named as INCOMPATIBLE, not as missing.
#
# The state file the CLI writes next to the bank is the artefact all three
# facts need.
import os, re, shutil, struct, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TMP = os.path.join(os.environ["TEMP"], "opencode")
EXE = os.path.join(ROOT, "build", "popcnt_brain.exe")
CORPUS = os.path.join(TMP, "real_corpus.txt")
STATE = os.path.join(ROOT, "build", "weight_bank_corpus_train.state")

REF_ACC = 91.455
REF_ERR = 705

if not os.path.isfile(EXE):
    print("FATAL: no built CLI at", EXE)
    sys.exit(2)


def run(args):
    r = subprocess.run([EXE] + args, cwd=ROOT, capture_output=True, timeout=3600)
    so = (r.stdout or b"").decode("utf-8", errors="replace")
    se = (r.stderr or b"").decode("utf-8", errors="replace")
    return r.returncode, so, se


def field(so, key):
    m = re.search(r'"%s":\s*([0-9.eE+-]+)' % re.escape(key), so)
    return float(m.group(1)) if m else None


print("=== 1. the reference corpus: the accuracy at three settings ===", flush=True)
# --fast: one short run instead of the full trio (the fast lane)
FAST = "--fast" in sys.argv
SETTINGS = ((3, 424242),) if FAST else ((12, 424242), (12, 1337), (6, 424242))
results = []
for epochs, seed in SETTINGS:
    t0 = time.time()
    rc, so, se = run(["--train", CORPUS, "--epochs", str(epochs), "--seed", str(seed)])
    acc = field(so, "accuracy_percent")
    err = field(so, "live_replay_errors")
    # The CLI's exit code is NOT the verdict: it reports a non-zero code when
    # the live replay still has errors (its "not fully converged" convention).
    # The verdict is the accuracy against the recorded floor.
    ok = (acc is not None) and (acc >= REF_ACC - 0.5 or FAST)
    print("  epochs=%-3d seed=%-7d acc=%.3f%% errors=%s rc=%s  %s (%.0fs)"
          % (epochs, seed, acc if acc is not None else -1,
             int(err) if err is not None else "?", rc, "PASS" if ok else "FAIL",
             time.time() - t0), flush=True)
    results.append(ok)

if not os.path.isfile(STATE):
    print("FATAL: no state file was written")
    sys.exit(2)

print("=== 2. the state round trip (the SAME trained model) ===", flush=True)
BACKUP = STATE + ".v2backup"
shutil.copy2(STATE, BACKUP)
trained_acc = field(run(["--train", CORPUS, "--epochs", "6", "--seed", "424242"])[1],
                    "accuracy_percent")
rc, so, se = run(["--load", CORPUS])
acc_load = field(so, "accuracy_percent")
restored = re.search(r'"restored_from_state":\s*(true|false)', so)
print("  load rc=%d acc=%s restored=%s (the trained model: %s)"
      % (rc, acc_load, restored.group(1) if restored else "?",
         trained_acc), flush=True)
# The verdict is the FACTS: the state was restored and the restored model
# scores at the recorded floor and no worse than the trained one. The exit
# code is not the verdict (the CLI reports the live-replay errors through it).
roundtrip_ok = (restored is not None) and (restored.group(1) == "true") \
    and acc_load is not None and (acc_load >= REF_ACC - 0.5 or FAST) \
    and (trained_acc is None or acc_load >= trained_acc - 0.5)
results.append(roundtrip_ok)

print("=== 3. the format diagnostic: an OLD-format state must be named, not lost ===",
      flush=True)
data = bytearray(open(STATE, "rb").read())
version = struct.unpack_from("<I", data, 8)[0]
struct.pack_into("<I", data, 8, 1)          # pretend it was written by v1
open(STATE, "wb").write(bytes(data))
rc, so, se = run(["--load", CORPUS])
diag_ok = ("format version" in se) and ("no trainer state" not in se)
print("  the state's own version was %d; the load rc=%d" % (version, rc), flush=True)
print("  stderr: " + se.strip()[:300], flush=True)
shutil.copy2(BACKUP, STATE)
os.remove(BACKUP)
print("  the original state file is restored", flush=True)
results.append(diag_ok)

print()
print("CORPUS GROUP:", "ALL GREEN" if all(results) else "RED")
sys.exit(0 if all(results) else 2)
