# The same decomposition for the WORD-LEVEL (same-domain) split: is 30.022%
# causal? Same chain as baselines_bpe.py [5].
import os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
tr = os.path.join(HERE, "data", "samedomain_train.txt")
ho = os.path.join(HERE, "data", "samedomain_holdout.txt")


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
n_tri = n_bi = n_uni = n_skip = n_no = 0
for a, b, c3 in tri:
    d = tri_c.get((a, b))
    if d and sum(d.values()) >= 2 and d.most_common(1)[0][0] == c3:
        n_tri += 1; continue
    e = bi.get(b)
    if e and sum(e.values()) >= 2 and e.most_common(1)[0][0] == c3:
        n_bi += 1; continue
    if uni and uni.most_common(1)[0][0] == c3:
        n_uni += 1; continue
    n_no += 1
for a, b, c3 in tri:
    d = tri_c.get((a, b))
    if d and sum(d.values()) >= 2 and d.most_common(1)[0][0] == c3:
        continue
    e = bi.get(b)
    if e and sum(e.values()) >= 2 and e.most_common(1)[0][0] == c3:
        continue
    f = bi_skip.get((a, c3))
    if f and f.most_common(1)[0][0] == b:
        n_skip += 1

print("heldout trigram decisions   : %d" % n)
print("[tri]                         : %.3f%%" % (100.0 * n_tri / n))
print("[bi fallback]                 : %.3f%%" % (100.0 * n_bi / n))
print("[uni fallback]                : %.3f%%" % (100.0 * n_uni / n))
print("CAUSAL tri->bi->uni top-1   : %.3f%%" % (100.0 * (n_tri + n_bi + n_uni) / n))
print("[non-causal skip level]       : %.3f%%" % (100.0 * n_skip / n))
print("ref chain total (tri+bi+sk) : %.3f%%" % (100.0 * (n_tri + n_bi + n_skip) / n))
