"""Re-decode a saved from-scratch checkpoint with a chosen beam size.

Usage (from part2/):
    .venv/bin/python analysis/ec_decode.py <experiment> <num_beams> dev
    .venv/bin/python analysis/ec_decode.py <experiment> <num_beams> test

Dev writes results/t5_scr_<experiment>_b<beams>_dev.{sql} and prints metrics.
Test writes results/t5_scr_<experiment>_b<beams>_test.sql and the matching records pkl.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())

from load_data import load_t5_data
from t5_utils import load_model_from_checkpoint
from train_t5 import eval_epoch, test_inference

experiment, beams, split = sys.argv[1], int(sys.argv[2]), sys.argv[3]
args = SimpleNamespace(finetune=False, lora=False, experiment_name=experiment,
                       num_beams=beams, max_decode_length=512)

_, dev_loader, test_loader = load_t5_data(8, 8)
model = load_model_from_checkpoint(args, best=True)
model.eval()

tag = f"t5_scr_{experiment}_b{beams}"
if split == "dev":
    loss, f1, em, sql_em, err = eval_epoch(
        args, model, dev_loader, "data/dev.sql", f"results/{tag}_dev.sql",
        "records/ground_truth_dev.pkl", f"records/{tag}_dev.pkl")
    print(f"DEV beams={beams}: loss {loss:.4f} Record F1 {f1:.4f} Record EM {em:.4f} "
          f"SQL EM {sql_em:.4f} SQL errors {err*100:.2f}%")
else:
    test_inference(args, model, test_loader, f"results/{tag}_test.sql", f"records/{tag}_test.pkl")
    print(f"TEST decoded with beams={beams} -> results/{tag}_test.sql")
