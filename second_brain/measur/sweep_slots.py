# Two measurements that need the engine's internals:
#
#   1. THE ENGINE ON A GIVEN CORPUS. The word-level comparison was invalid
#      because the chain was measured on the trigram positions while the
#      engine scores the positions from the second token on, with the WHOLE
#      prefix as its context. This runs the engine on both corpora so the
#      chain and the engine are compared on ONE population.
#
#   2. THE CO-LOCATION LIMIT. A micro-expert page holds D_BOTTLENECK
#      bottleneck neurons (16 by default) and the STE writes into them
#      without any eviction accounting: on the subword split telemetry
#      (576,806 write events, 248,656 distinct pages) a page takes 9.28
#      touches on average but 9,953 at the worst, and 21,946 pages exceed
#      16, 6,583 exceed 32, 2,298 exceed 64, 943 exceed 128. This sweeps the
#      slot count and measures what the limit costs. The sweep runs on a
#      SCRATCH COPY of the engine: the shipped source is not modified, and
#      the copy's bank layout is patched to match the wider page. A slot
#      count is only shipped if this measures a gain.
#
# Every build is a fresh copy in its own scratch directory, one core each.
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRATCH = os.path.join(os.environ["TEMP"], "opencode", "hssa_slots")
TMP = os.path.join(os.environ["TEMP"], "opencode")
DATA = os.path.join(HERE, "data")

VSWHERE = os.path.join(os.environ["ProgramFiles(x86)"], "Microsoft Visual Studio",
                       "Installer", "vswhere.exe")
_v = subprocess.run([VSWHERE, "-latest", "-products", "*", "-requires",
                     "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                     "-property", "installationPath"], capture_output=True, text=True)
VCVARS = os.path.join(_v.stdout.strip().splitlines()[0], "VC", "Auxiliary", "Build", "vcvars64.bat")
FLAGS = "/nologo /std:c++20 /O2 /Oi /Ot /arch:AVX512 /openmp /EHsc /wd4068 /utf-8"

CORPORA = {
    "bpe": ("bpe_train.txt", "bpe_holdout.txt"),
    "word": ("samedomain_train.txt", "samedomain_holdout.txt"),
}


def patch_slots(src, dst, slots):
    """Copy the engine and widen its expert page to `slots` bottleneck neurons.

    Every anchor must be found EXACTLY once, or the copy is rejected: a
    silently unpatched copy would measure the default engine and the whole
    sweep would be a lie.
    """
    shutil.copy2(src, dst)
    txt = io_read(dst)
    edits = [
        # the slot count (and the page size it implies) must be declared BEFORE
        # the bank's model size, which is derived from it
        ("static constexpr size_t D_BOTTLENECK = 16;            // 16 ternary bottleneck neurons per 4 KB page",
         "// (the bottleneck slot count is declared with the vector constants below)"),
        ("static constexpr size_t WORDS_PER_VEC = D_MODEL / 64; // 8 x uint64_t = 64 bytes",
         "static constexpr size_t WORDS_PER_VEC = D_MODEL / 64; // 8 x uint64_t = 64 bytes\n"
         "static constexpr size_t D_BOTTLENECK = %d;            // ternary bottleneck neurons per expert page\n"
         "static constexpr size_t EXPERT_PAGE_BYTES = D_BOTTLENECK * 4 * WORDS_PER_VEC * sizeof(uint64_t);" % slots),
                # The page KEEPS the 4 KB alignment: only its SIZE grows (N*256
        # bytes, always a multiple of 4 KB; alignas above 8192 is rejected
        # by MSVC with error C2345).
        ('static_assert(sizeof(TernaryMLPExpertPage) == PAGE_BYTES, "TernaryMLPExpertPage must be 4096 bytes");',
         'static_assert(sizeof(TernaryMLPExpertPage) == EXPERT_PAGE_BYTES, "the expert page size must match the slot count");'),
        ("static constexpr size_t BANK_MODEL_BYTES = TOTAL_EXPERTS * PAGE_BYTES;",
         "static constexpr size_t BANK_MODEL_BYTES = TOTAL_EXPERTS * EXPERT_PAGE_BYTES;"),
        ("static constexpr uint32_t CAUSAL_GRAPH_PAGE_BASE   = 262144;",
         "static constexpr uint32_t MODEL_BLOCKS = (uint32_t)(TOTAL_EXPERTS * EXPERT_PAGE_BYTES / PAGE_BYTES);\n"
         "static constexpr uint32_t CAUSAL_GRAPH_PAGE_BASE   = MODEL_BLOCKS;"),
        ("static constexpr uint32_t PERSISTENT_FACT_PAGE_BASE = 282144;",
         "static constexpr uint32_t PERSISTENT_FACT_PAGE_BASE = MODEL_BLOCKS + CAUSAL_GRAPH_PAGES;"),
        ("static constexpr uint32_t BANK_TOTAL_BLOCKS        = 340144;",
         "static constexpr uint32_t BANK_TOTAL_BLOCKS        = MODEL_BLOCKS + 78000;"),
        ("blob.raw(page_base + static_cast<size_t>(pid) * PAGE_BYTES, PAGE_BYTES);",
         "blob.raw(page_base + static_cast<size_t>(pid) * EXPERT_PAGE_BYTES, EXPERT_PAGE_BYTES);"),
        ("if (!r.raw(page_base + static_cast<size_t>(pid) * PAGE_BYTES, PAGE_BYTES))",
         "if (!r.raw(page_base + static_cast<size_t>(pid) * EXPERT_PAGE_BYTES, EXPERT_PAGE_BYTES))"),
        # The POSIX fault handler divides the tracked-region offset by 4 KiB
        # to pick a page id, but its telemetry bitmap is indexed only over the
        # SHIPPED 4 KB expert pages: with a wider page one expert spans four
        # 4 KiB slots and page_id can reach 4*TOTAL_EXPERTS, past the bitmap's
        # end. Shipped (16-slot) layouts are always inside the bitmap; the
        # guard only changes measurement builds, not the shipped engine.
        ("        if (mprotect(page_base, PAGE_BYTES, PROT_READ | PROT_WRITE) == 0) {\n"
         "            if (g_page_touched_bitmap[page_id] == 0) {",
         "        if (mprotect(page_base, PAGE_BYTES, PROT_READ | PROT_WRITE) == 0) {\n"
         "            if (page_id < TOTAL_EXPERTS && g_page_touched_bitmap[page_id] == 0) {"),
    ]
    for old, new in edits:
        n = txt.count(old)
        if n != 1:
            raise SystemExit("FATAL: the anchor appears %d times, not once:\n%s" % (n, old))
        txt = txt.replace(old, new, 1)
    io_write(dst, txt)


def io_read(path):
    with open(path, "r", encoding="utf-8", errors="surrogateescape") as fh:
        return fh.read()


def io_write(path, txt):
    with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="\n") as fh:
        fh.write(txt)


def build(tag, slots, corpus, defines=()):
    d = os.path.join(SCRATCH, tag)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(os.path.join(ROOT, "second_brain")):
        if f.endswith(".h"):
            shutil.copy2(os.path.join(ROOT, "second_brain", f), d)
    engine = os.path.join(ROOT, "second_brain", "popcnt_brain.cpp")
    dst = os.path.join(d, "popcnt_brain.cpp")
    if slots == 16:
        shutil.copy2(engine, dst)          # the shipped source, byte for byte
    else:
        patch_slots(engine, dst, slots)
    shutil.copy2(os.path.join(HERE, "acceptance_harness.cpp"), d)
    bat = os.path.join(d, "b.bat")
    with open(bat, "w", newline="\r\n") as fh:
        fh.write('@echo off\r\ncall "%s" >nul\r\n' % VCVARS)
        fh.write('cl.exe %s %s /I"%s" /I"%s" "%s" /Fe:"%s\\a.exe" /Fo:"%s\\a.obj" '
                 '/link /STACK:33554432\r\nexit /b %%ERRORLEVEL%%\r\n'
                 % (FLAGS, " ".join(defines), d, d,
                    os.path.join(d, "acceptance_harness.cpp"), d, d))
    b = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True, timeout=3600)
    if b.returncode != 0:
        print("  BUILD FAILED: %s\n%s\n%s" % (tag, b.stdout[-1500:], b.stderr[-800:]), flush=True)
        return None
    return os.path.join(d, "a.exe")


def run(tag, slots, corpus, epochs="6", seed="424242", defines=()):
    t0 = time.time()
    exe = build(tag, slots, corpus, defines)
    if not exe:
        return None
    tr = os.path.join(DATA, CORPORA[corpus][0])
    ho = os.path.join(DATA, CORPORA[corpus][1])
    g = subprocess.run([exe, tr, ho, epochs, seed], cwd=os.path.dirname(exe),
                       capture_output=True, text=True, timeout=7200)
    so = g.stdout

    def num(pattern, cast=float):
        m = re.search(pattern, so)
        return cast(m.group(1)) if m else None

    occ = re.search(r"\[occupancy\].*", g.stderr)
    rec = {
        "tag": tag, "slots": slots, "corpus": corpus,
        "acc": num(r'"holdout_accuracy_percent":\s*([0-9.]+)'),
        "cov": num(r'"heldout_shortlist_coverage_percent":\s*([0-9.]+)'),
        "cov_n": num(r'"heldout_shortlist_coverage_total":\s*(\d+)', int),
        "positions": num(r'"holdout_transitions_scored":\s*(\d+)', int),
        "train_ms": num(r'"train_time_ms":\s*([0-9.]+)'),
        "ns_per_tok": num(r'"measured_draft_ns_per_token":\s*([0-9.]+)'),
        "vocab": num(r'"vocab_size":\s*(\d+)', int),
        "occupancy": occ.group(0) if occ else "",
        "wall_s": round(time.time() - t0, 1),
    }
    print("  %-22s acc=%-9s cov=%-9s n=%-7s train_ms=%-9s ns/tok=%-8s wall=%ss"
          % (tag, rec["acc"], rec["cov"], rec["positions"], rec["train_ms"],
             rec["ns_per_tok"], rec["wall_s"]), flush=True)
    if rec["occupancy"]:
        print("      %s" % rec["occupancy"], flush=True)
    return rec


JOBS = []
if "--word" in sys.argv:
    JOBS.append(("shipped_word", 16, "word", ()))
if "--word2" in sys.argv:
    # the OLD word-level configuration (K=256, the shortlist before F21) and
    # the plain engine, so the word-level delta of the count mixture is
    # separated from the change of K.
    JOBS.append(("plain_word", 16, "word", ("/DHSSA_DISABLE_BACKOFF",)))
    JOBS.append(("k256_word", 16, "word", ("/DHSSA_BACKOFF_COUNT_WEIGHT=1024",
                                           "/DHSSA_BACKOFF_SHORTLIST_K=256")))
if "--slots" in sys.argv:
    for s in (16, 32, 64, 128):
        JOBS.append(("slots%d_bpe" % s, s, "bpe", ()))
if "--bpe" in sys.argv:
    JOBS.append(("shipped_bpe", 16, "bpe", ()))
if not JOBS:
    JOBS = [("shipped_bpe", 16, "bpe", ()), ("shipped_word", 16, "word", ())]

print("%d configurations" % len(JOBS), flush=True)
t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=4) as pool:
    futs = [pool.submit(run, tag, slots, corpus, "6", "424242", defines)
            for tag, slots, corpus, defines in JOBS]
    for f in futs:
        results.append(f.result())

out = os.path.join(TMP, "slots_report.txt")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("=== the engine and the co-location limit ===\n")
    fh.write("%-16s %-6s %-6s %10s %10s %10s %12s %12s\n"
             % ("config", "slots", "corpus", "acc%", "cov%", "positions", "train_ms", "ns/tok"))
    for r in results:
        if not r:
            continue
        fh.write("%-16s %-6d %-6s %10s %10s %10s %12s %12s\n"
                 % (r["tag"], r["slots"], r["corpus"], r["acc"], r["cov"], r["positions"],
                    r["train_ms"], r["ns_per_tok"]))
        if r["occupancy"]:
            fh.write("    %s\n" % r["occupancy"])
print("WROTE %s" % out, flush=True)
print("total wall: %.0fs" % (time.time() - t0), flush=True)
