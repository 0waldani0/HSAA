import subprocess, os, re

root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
tmp = os.environ["TEMP"]
exe = os.path.join(root, "build", "popcnt_brain.exe")
corpus = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real_corpus.txt")
out = os.path.join(tmp, "opencode", "corpus_regression.txt")

REF_ACC = 91.455        # fresh floor of the shipped build (F21: mixed weight 1024;
                        # the trio measured 91.600 / 91.455 / 91.648), +0.55 over the
                        # pre-mixture floor 90.909
REF_ERR = 705

print("corpus:", corpus, os.path.isfile(corpus))
with open(out, "w", encoding="utf-8", errors="replace") as fh:
    fh.write("=== corpus-train regression (801-line reference corpus) ===\n")
    fh.write("reference before fixes: accuracy %.3f%%  live_replay_errors %d\n\n"
             % (REF_ACC, REF_ERR))
    for epochs, seed in ((12, 424242), (12, 1337), (6, 424242)):
        r = subprocess.run([exe, "--train", corpus, "--epochs", str(epochs), "--seed", str(seed)],
                           cwd=root, capture_output=True, timeout=3600)
        so = (r.stdout or b"").decode("utf-8", errors="replace")
        def g(k):
            m = re.search(r'"%s":\s*([0-9.eE+-]+)' % re.escape(k), so)
            return m.group(1) if m else "-"
        acc = g("accuracy_percent"); err = g("live_replay_errors")
        trans = g("train_transitions"); ms = g("train_time_ms")
        flag = ""
        try:
            if float(acc) < REF_ACC - 0.5:
                flag = "  <-- REGRESSION vs reference"
        except ValueError:
            pass
        fh.write("epochs=%-3d seed=%-7s transitions=%-9s errors=%-7s acc=%-9s ms=%-11s exit=%d%s\n"
                 % (epochs, seed, trans, err, acc, ms, r.returncode, flag))
        fh.flush()
        print("epochs=%d seed=%s acc=%s err=%s%s" % (epochs, seed, acc, err, flag))

print("WROTE " + out)
