# Decomposition of the 46.899% "count backoff" reference: is it causal?
# The skip-gram level there answers "which token most often sits BETWEEN
# (prev, NEXT)" -- it uses the gold next token as context, so it cannot run
# online. Measure the causal part (trigram -> bigram -> unigram) separately.
import os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
tr = os.path.join(HERE, "data", "bpe_train.txt")
ho = os.path.join(HERE, "data", "bpe_holdout.txt")


def toks(line):
    return ["<bos>"] + line.split() + ["<eos>"]


train_lines = [l for l in open(tr, encoding="utf-8", errors="replace").read().splitlines() if l.strip()]
hold_lines = [l for l in open(ho, encoding="utf-8", errors="replace").read().splitlines() if l.strip()]

T = []
for l in train_lines:
    T.extend(toks(l))

tri = []
for l in hold_lines:
    t = toks(l)
    for i in range(2, len(t) - 1):
        tri.append((t[i - 1], t[i], t[i + 1]))

# The SAME population the engine scores: every position from the second token
# on (the engine's context at position t is (seq[t-1], seq[t]) and the target
# is seq[t+1]; the loop above starts one position later, so the two chains
# would be measured on different sets of positions).
tri_engine = []
for l in hold_lines:
    t = toks(l)
    for i in range(1, len(t) - 1):
        tri_engine.append((t[i - 1], t[i], t[i + 1]))

tri_c = collections.defaultdict(collections.Counter)
bi = collections.defaultdict(collections.Counter)
bi_skip = collections.defaultdict(collections.Counter)
uni = collections.Counter()
for a, b in zip(T, T[1:]):
    bi[a][b] += 1
    uni[b] += 1
for i in range(1, len(T) - 1):
    tri_c[(T[i - 1], T[i])][T[i + 1]] += 1
    bi_skip[(T[i - 1], T[i + 1])][T[i]] += 1

n = len(tri)
n_tri_hit = n_bi_hit = n_skip_hit = n_uni_hit = n_no_hit = 0
# the causal chain: trigram -> bigram -> unigram (top-1 of the first level seen >=2 times)
def causal_chain(pairs):
    hit_tri = hit_bi = hit_uni = miss = 0
    for a, b, c3 in pairs:
        d = tri_c.get((a, b))
        if d and sum(d.values()) >= 2 and d.most_common(1)[0][0] == c3:
            hit_tri += 1; continue
        e = bi.get(b)
        if e and sum(e.values()) >= 2 and e.most_common(1)[0][0] == c3:
            hit_bi += 1; continue
        if uni and uni.most_common(1)[0][0] == c3:
            hit_uni += 1; continue
        miss += 1
    return hit_tri, hit_bi, hit_uni, miss


n_tri_hit, n_bi_hit, n_uni_hit, n_no_hit = causal_chain(tri)
e_tri, e_bi, e_uni, e_no = causal_chain(tri_engine)

# the reference's own chain, with its non-causal skip-gram level last
for a, b, c3 in tri:
    d = tri_c.get((a, b))
    if d and sum(d.values()) >= 2 and d.most_common(1)[0][0] == c3:
        continue
    e = bi.get(b)
    if e and sum(e.values()) >= 2 and e.most_common(1)[0][0] == c3:
        continue
    f = bi_skip.get((a, c3))
    if f and f.most_common(1)[0][0] == b:
        n_skip_hit += 1

causal = n_tri_hit + n_bi_hit + n_uni_hit
print("heldout trigram decisions      : %d" % n)
print("[tri]                            : %.3f%% (%d)" % (100.0 * n_tri_hit / n, n_tri_hit))
print("[bi fallback]                    : %.3f%% (%d)" % (100.0 * n_bi_hit / n, n_bi_hit))
print("[uni fallback]                   : %.3f%% (%d)" % (100.0 * n_uni_hit / n, n_uni_hit))
print("CAUSAL tri->bi->uni top-1      : %.3f%%" % (100.0 * causal / n))
print("[non-causal skip level, ref]     : %.3f%% (%d)" % (100.0 * n_skip_hit / n, n_skip_hit))
print("ref chain total (tri+bi+skip)  : %.3f%%" % (100.0 * (n_tri_hit + n_bi_hit + n_skip_hit) / n))
print("the engine's population        : %d positions (from the second token on)" % len(tri_engine))
print("CAUSAL on the engine's positions: %.3f%%  (tri %.3f%%, bi %.3f%%, uni %.3f%%)"
      % (100.0 * (e_tri + e_bi + e_uni) / len(tri_engine),
         100.0 * e_tri / len(tri_engine),
         100.0 * e_bi / len(tri_engine),
         100.0 * e_uni / len(tri_engine)))
# The engine's own numbers on these files are NOT printed here: they are
# measured by measure_all.py (the count-weight sweep), never hardcoded.
print("the engine on the same files: measure_all.py (the count-weight sweep)")
