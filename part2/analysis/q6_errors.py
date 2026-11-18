"""Q6 error analysis on dev: compare a model's dev SQL/records with ground truth."""
import pickle
import re
import sys
from collections import Counter

name = sys.argv[1]
# Optional second argument: "ft" (default, fine-tuned runs) or "scr" (from-scratch runs).
kind_prefix = sys.argv[2] if len(sys.argv) > 2 else "ft"
nl = [l.strip() for l in open("data/dev.nl")]
gt_sql = [l.strip() for l in open("data/dev.sql")]
pr_sql = [l.strip() for l in open(f"results/t5_{kind_prefix}_{name}_dev.sql")]
gt_rec, _ = pickle.load(open("records/ground_truth_dev.pkl", "rb"))
pr_rec, pr_err = pickle.load(open(f"records/t5_{kind_prefix}_{name}_dev.pkl", "rb"))
n = len(nl)
assert len(gt_sql) == len(pr_sql) == len(gt_rec) == len(pr_rec) == n

norm = lambda s: " ".join(s.split())
print("SQL EM raw:", sum(a == b for a, b in zip(gt_sql, pr_sql)) / n,
      " whitespace-normalised:", sum(norm(a) == norm(b) for a, b in zip(gt_sql, pr_sql)) / n)
# Show the first differing character region for a record-correct pair, to explain low SQL EM.
for i in range(n):
    if set(gt_rec[i]) == set(pr_rec[i]) and gt_sql[i] != pr_sql[i]:
        a, b = gt_sql[i], pr_sql[i]
        k = next(j for j in range(min(len(a), len(b))) if a[j] != b[j]) if a[:min(len(a), len(b))] != b[:min(len(a), len(b))] else min(len(a), len(b))
        print(f"\n[record-correct but SQL differs] #{i}\n  GT : ...{a[max(0,k-60):k+80]}\n  PRD: ...{b[max(0,k-60):k+80]}")
        break

err_idx = [i for i in range(n) if pr_err[i]]
wrong_idx = [i for i in range(n) if not pr_err[i] and set(gt_rec[i]) != set(pr_rec[i])]
right = n - len(err_idx) - len(wrong_idx)
print(f"\nrecords exactly right: {right}/{n}; SQL error: {len(err_idx)}; runs but wrong records: {len(wrong_idx)}")

def kind(msg):
    m = msg.lower()
    for k in ["syntax error", "no such column", "no such table", "ambiguous", "incomplete input",
              "unrecognized token", "timeout", "misuse"]:
        if k in m:
            return k
    return m[:60]
print("\nSQL error kinds:", Counter(kind(pr_err[i]) for i in err_idx).most_common())

# Sub-classify executable-but-wrong queries.
def literals(s):
    return set(re.findall(r"'[^']*'|\b\d+\b", s))
def conds(s):
    return len(re.findall(r"\b(?:AND|OR)\b", s))
cats = Counter()
ex = {}
for i in wrong_idx:
    g, p = gt_sql[i], pr_sql[i]
    if literals(g) != literals(p):
        c = "wrong constant/value"
    elif conds(g) != conds(p):
        c = "missing/extra condition"
    elif not pr_rec[i]:
        c = "empty result"
    else:
        c = "other structural (same literals & #conds)"
    cats[c] += 1
    ex.setdefault(c, []).append(i)
print("executable-but-wrong breakdown:", cats.most_common())

# Error-type examples
print("\n================ EXAMPLES")
for i in err_idx[:4]:
    print(f"\n[SQL error] #{i}: {nl[i]}\n  err: {pr_err[i][:150]}\n  PRD tail: ...{pr_sql[i][-160:]}")
for c, idxs in ex.items():
    for i in idxs[:3]:
        g, p = gt_sql[i], pr_sql[i]
        print(f"\n[{c}] #{i}: {nl[i]}\n  GT literals:  {sorted(literals(g))}\n  PRD literals: {sorted(literals(p))}"
              f"\n  #conds GT={conds(g)} PRD={conds(p)}  |GT rec|={len(set(gt_rec[i]))} |PRD rec|={len(set(pr_rec[i]))}")

# Truncation / length: do errors correlate with long targets?
avg = lambda xs: sum(xs) / max(len(xs), 1)
print("\nmean GT SQL chars: correct", avg([len(gt_sql[i]) for i in range(n) if i not in err_idx and i not in wrong_idx]),
      " sql-error", avg([len(gt_sql[i]) for i in err_idx]), " wrong", avg([len(gt_sql[i]) for i in wrong_idx]))
print("pred ends without closing paren balance:", sum(p.count("(") != p.count(")") for p in (pr_sql[i] for i in err_idx)), "of", len(err_idx))
