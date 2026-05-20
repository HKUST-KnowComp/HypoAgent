"""
Evaluate unconditional log results.

Usage:
    python -m akgr.agent.judge_uncondition --dataname DBpedia50 --modelname gpt-5.4-mini
    python -m akgr.agent.judge_uncondition --log log/DBpedia50/uncondition_gpt-5.4-mini.jsonl
"""

import json
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None)
    parser.add_argument("--dataname", default=None)
    parser.add_argument("--modelname", default=None)
    args = parser.parse_args()

    log_path = args.log or os.path.join("log", args.dataname, f"uncondition_{args.modelname}.jsonl")
    assert os.path.exists(log_path), f"Log not found: {log_path}"

    jaccard, dice, overlap, n = 0.0, 0.0, 0.0, 0
    with open(log_path, encoding="utf-8") as f:
        content = f.read()
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break
        obj, pos = decoder.raw_decode(content, pos)
        u = obj.get("best")
        if u is None:
            continue
        if u.get("jaccard") is not None:
            jaccard += u["jaccard"]
            dice += u.get("dice", 0.0)
            overlap += u.get("overlap", 0.0)
            n += 1

    if n == 0:
        print("No valid records found.")
        return

    print(f"N={n}")
    print(f"Jaccard : {jaccard/n:.4f}")
    print(f"Dice    : {dice/n:.4f}")
    print(f"Overlap : {overlap/n:.4f}")


if __name__ == "__main__":
    main()
