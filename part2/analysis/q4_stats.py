"""Q4 data statistics with the T5 tokenizer, before and after the preprocessing in load_data.py."""
import os
import sys

from transformers import T5TokenizerFast

DATA = sys.argv[1]
PREFIX = "translate English to SQL: "
tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")


def lines(split, ext):
    with open(os.path.join(DATA, f"{split}.{ext}")) as f:
        return f.readlines()


def stats(texts, special):
    ids = tok(texts, add_special_tokens=special).input_ids
    lens = [len(x) for x in ids]
    vocab = {t for x in ids for t in x}
    return len(texts), sum(lens) / len(lens), max(lens), len(vocab)


rows = {}
for split in ["train", "dev"]:
    nl_raw, sql_raw = lines(split, "nl"), lines(split, "sql")
    # "Before": raw file lines as-is (no strip, no prefix, no </s>).
    rows[("before", split)] = (stats(nl_raw, False), stats(sql_raw, False))
    # "After": exactly what T5Dataset feeds the model: strip, prefix on NL, "<" -> LESSTHAN in SQL, </s> appended,
    # truncation at 512 (checked separately below).
    nl = [PREFIX + l.strip() for l in nl_raw]
    sql = [l.strip().replace("<", "LESSTHAN") for l in sql_raw]  # same as load_data.sql_to_target
    rows[("after", split)] = (stats(nl, True), stats(sql, True))

for stage in ["before", "after"]:
    print(f"\n== {stage} preprocessing")
    print(f"{'stat':40s} {'train':>10s} {'dev':>10s}")
    tr_nl, tr_sql = rows[(stage, "train")]
    dv_nl, dv_sql = rows[(stage, "dev")]
    print(f"{'Number of examples':40s} {tr_nl[0]:>10d} {dv_nl[0]:>10d}")
    print(f"{'Mean sentence length (tokens)':40s} {tr_nl[1]:>10.2f} {dv_nl[1]:>10.2f}")
    print(f"{'Max sentence length (tokens)':40s} {tr_nl[2]:>10d} {dv_nl[2]:>10d}")
    print(f"{'Mean SQL query length (tokens)':40s} {tr_sql[1]:>10.2f} {dv_sql[1]:>10.2f}")
    print(f"{'Max SQL query length (tokens)':40s} {tr_sql[2]:>10d} {dv_sql[2]:>10d}")
    print(f"{'Vocabulary size, NL (distinct token ids)':40s} {tr_nl[3]:>10d} {dv_nl[3]:>10d}")
    print(f"{'Vocabulary size, SQL (distinct token ids)':40s} {tr_sql[3]:>10d} {dv_sql[3]:>10d}")

test = lines("test", "nl")
print(f"\ntest examples: {len(test)}")
print("any sequence > 512 tokens after preprocessing:",
      any(r[0][2] > 512 or r[1][2] > 512 for k, r in rows.items() if k[0] == "after"))
