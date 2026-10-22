# The WORD-LEVEL (same-domain) split, in the ENGINE's population.
#
# Why this file exists in this shape: the first version of it measured the
# count chain on the TRIGRAM positions (the contexts (t[i-1], t[i]) that a
# trigram chain can see), while the engine scores the positions from the
# second token on. The two numbers (38.462% for the chain, 25.509% for the
# engine) were therefore NOT one population and could not be compared -- the
# engine's own context is the WHOLE prefix, not a trigram window.
#
# Three measurements live here, all on the SAME positions (the engine's):
#   1. the trigram-window chain (trigram -> bigram -> unigram), decomposed;
#   2. the FULL-PREFIX chain (the longest matching suffix of the whole
#      context, backoff down to the unigram): the strongest count model the
#      same training data supports, with the SAME context availability as the
#      engine's decision;
#   3. the decomposition of (2) by the level that answered.
#
# The engine's own numbers for these files come from measure_all.py (the
# engine is measured, never hardcoded here).
import os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
tr = os.path.join(HERE, "data", "samedomain_train.txt")
ho = os.path.join(HERE, "data", "samedomain_holdout.txt")

MAX_ORDER = 8  # the longest context the full-prefix chain looks at


def toks(line):
    return ["<bos>"] + line.split() + ["<eos>"]


train_lines = [l for l in open(tr, encoding="utf-8", errors="replace").read().splitlines() if l.strip()]
hold_lines = [l for l in open(ho, encoding="utf-8", errors="replace").read().splitlines() if l.strip()]

T = []
for l in train_lines:
    T.extend(toks(l))

# the trigram positions: the contexts (t[i-1], t[i]) the chain can see
tri = []
# the ENGINE's population: every position from the second token on. The
# engine's context at position t is the whole prefix t[0..t] and the target
# is t[t+1]; the loop below starts one position later, so the two chains
# would be measured on different sets of positions.
tri_engine = []
# the engine's positions WITH the full prefix as the context
ctx_engine = []
for l in hold_lines:
    t = toks(l)
    for i in range(2, len(t) - 1):
        tri.append((t[i - 1], t[i], t[i + 1]))
    for i in range(1, len(t) - 1):
        tri_engine.append((t[i - 1], t[i], t[i + 1]))
        ctx_engine.append((tuple(t[:i + 1]), t[i + 1]))

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

# the full-prefix tables: for every context length 1..MAX_ORDER, the successors
# seen in the training data
full = {k: collections.defaultdict(collections.Counter) for k in range(1, MAX_ORDER + 1)}
for i in range(len(T)):
    for k in range(1, MAX_ORDER + 1):
        if i + 1 >= len(T):
            break
        full[k][tuple(T[i - k + 1:i + 1])][T[i + 1]] += 1


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


def full_prefix_chain(contexts):
    """The longest matching suffix wins; the count must be >= 2 to answer."""
    by_len = {0: 0}
    for ctx, target in contexts:
        answered = False
        for k in range(min(MAX_ORDER, len(ctx)), 0, -1):
            d = full[k].get(ctx[-k:])
            if d and sum(d.values()) >= 2:
                by_len[k] = by_len.get(k, 0) + (1 if d.most_common(1)[0][0] == target else 0)
                answered = True
                break
        if not answered:
            by_len[0] = by_len.get(0, 0) + (1 if uni and uni.most_common(1)[0][0] == target else 0)
    return by_len


n = len(tri)
n_tri_hit, n_bi_hit, n_uni_hit, n_no_hit = causal_chain(tri)
e_tri, e_bi, e_uni, e_no = causal_chain(tri_engine)

n_skip_hit = 0
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

print("=== the WORD-LEVEL (same-domain) split ===")
print("train tokens=%d  holdout lines=%d" % (len(T), len(hold_lines)))
print()
print("--- 1. the trigram-window chain on the trigram positions (the old number)")
print("heldout trigram decisions   : %d" % n)
print("[tri]                         : %.3f%%" % (100.0 * n_tri_hit / n))
print("[bi fallback]                 : %.3f%%" % (100.0 * n_bi_hit / n))
print("[uni fallback]                : %.3f%%" % (100.0 * n_uni_hit / n))
print("CAUSAL tri->bi->uni top-1   : %.3f%%" % (100.0 * (n_tri_hit + n_bi_hit + n_uni_hit) / n))
print("[non-causal skip level]       : %.3f%%" % (100.0 * n_skip_hit / n))
print("ref chain total (tri+bi+sk) : %.3f%%" % (100.0 * (n_tri_hit + n_bi_hit + n_skip_hit) / n))
print()
print("--- 2. the SAME chain on the ENGINE's positions (one population)")
print("the engine's positions       : %d" % len(tri_engine))
print("trigram window                : %.3f%%  (tri %.3f%%, bi %.3f%%, uni %.3f%%)"
      % (100.0 * (e_tri + e_bi + e_uni) / len(tri_engine),
         100.0 * e_tri / len(tri_engine),
         100.0 * e_bi / len(tri_engine),
         100.0 * e_uni / len(tri_engine)))
print()
print("--- 3. the FULL-PREFIX chain on the engine's positions (context-matched)")
by_len = full_prefix_chain(ctx_engine)
m = len(ctx_engine)
tot = sum(by_len.values())
print("positions                     : %d" % m)
for k in sorted(by_len, reverse=True):
    print("  answered at order %-2d       : %.3f%%  (%d)" % (k, 100.0 * by_len[k] / m, by_len[k]))
print("FULL-PREFIX chain top-1       : %.3f%%  (%d of %d)" % (100.0 * tot / m, tot, m))
print()
print("the engine on the same files: measure_all.py --word (the engine is measured, never hardcoded)")
