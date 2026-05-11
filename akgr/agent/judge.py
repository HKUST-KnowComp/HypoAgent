"""
Evaluate singleturn log results.

Usage:
    python -m akgr.agent.judge --dataname PharmKG8k --modelname DeepSeek-V4-Flash
"""

import json
import re
import argparse
import os
import numpy as np


def _parse_conditions(raw) -> list[dict]:
    """Normalise parsed_conditions to list[{type, value}], handling both formats."""
    if not raw:
        return []
    if isinstance(raw[0], dict):
        return [c for c in raw if isinstance(c, dict) and "type" in c and "value" in c]
    # flat list: ["relation", "GG", "entitynumber", "2"]
    conds, it = [], iter(raw)
    for k in it:
        try:
            conds.append({"type": str(k), "value": str(next(it))})
        except StopIteration:
            break
    return conds


def _raw_to_pattern(raw: str) -> str:
    tokens = []
    for t in raw.strip().split():
        try:
            tokens.append("e" if int(t) > 0 else "p")
        except ValueError:
            tokens.append(t)
    return " ".join(tokens)


def _check(ctype, value, nl, raw, cond_ids: dict = None) -> bool:
    if ctype == "relation":
        # Use ID from source data if available: relation_id is stored as negative int in raw
        if cond_ids and "relation_id" in cond_ids:
            rid = str(cond_ids["relation_id"])  # e.g. "-8"
            return bool(raw) and rid in raw.split()
        return bool(nl and re.search(r'\bp\s*\(\s*' + re.escape(value) + r'\s*[,)]', nl))
    if ctype == "entity":
        # Use ID from source data if available: entity_id is stored as positive int in raw
        if cond_ids and "entity_id" in cond_ids:
            eid = str(cond_ids["entity_id"])
            return bool(raw) and eid in raw.split()
        return bool(nl and re.search(r'\be\s*\(\s*' + re.escape(value) + r'\s*\)', nl))
    if ctype == "relationnumber":
        return bool(raw) and sum(1 for t in raw.split() if re.fullmatch(r'-\d+', t)) == int(value)
    if ctype == "entitynumber":
        return bool(raw) and sum(1 for t in raw.split() if re.fullmatch(r'\d+', t)) == int(value)
    if ctype == "pattern":
        return bool(raw) and _raw_to_pattern(raw) == value.strip()
    return False


def _load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    decoder = json.JSONDecoder()
    records, pos = [], 0
    while pos < len(content):
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break
        obj, pos = decoder.raw_decode(content, pos)
        records.append(obj)
    return records


def evaluate(log_path: str, data_root: str = "data"):
    records = _load_records(log_path)

    # Load source data for ground-truth condition IDs (relation_id, entity_id)
    dataname = log_path.split(os.sep)[-2]
    src_path = os.path.join(data_root, dataname, "singleturn.jsonl")
    src_records = _load_records(src_path) if os.path.exists(src_path) else []
    # Build lookup: user_condition -> followup_condition_values
    src_cond_ids: dict[str, dict] = {}
    for s in src_records:
        q = s.get("followup_question", "")
        if q and "followup_condition_values" in s:
            src_cond_ids[q] = s["followup_condition_values"]

    jaccards, dices, overlaps = [], [], []
    per_type: dict[str, list[int]] = {}
    joint: list[int] = []
    per_hyp_avg: list[float] = []
    per_hyp_joint: list[int] = []
    # baseline: round 1 metrics
    b_jaccards, b_dices, b_overlaps = [], [], []
    b_per_type: dict[str, list[int]] = {}
    b_per_hyp_avg: list[float] = []
    b_per_hyp_joint: list[int] = []
    skipped = 0

    for r in records:
        if "error" in r or "best" not in r:
            skipped += 1
            continue
        best = r["best"]
        jaccards.append(best["jaccard"])
        dices.append(best["dice"])
        overlaps.append(best["overlap"])

        round1 = r.get("rounds", [{}])[0]
        raw_conds = round1.get("parsed_conditions", [])
        conditions = _parse_conditions(raw_conds)
        cond_ids = src_cond_ids.get(r.get("user_condition", ""), {})

        # baseline from round 1
        b_jaccards.append(round1.get("jaccard", 0.0))
        b_dices.append(round1.get("dice", 0.0))
        b_overlaps.append(round1.get("overlap", 0.0))

        if not conditions:
            joint.append(1)
            per_hyp_avg.append(1.0)
            per_hyp_joint.append(1)
            b_per_hyp_avg.append(1.0)
            b_per_hyp_joint.append(1)
            continue

        nl, raw = best.get("hypothesis_nl"), best.get("hypothesis_raw")
        sat = {c["type"]: _check(c["type"], c["value"], nl, raw, cond_ids) for c in conditions}
        for ctype, ok in sat.items():
            per_type.setdefault(ctype, []).append(int(ok))
        all_ok = list(sat.values())
        joint.append(int(all(all_ok)))
        per_hyp_avg.append(sum(all_ok) / len(all_ok))
        per_hyp_joint.append(int(all(all_ok)))

        b_nl = round1.get("hypothesis_nl")
        b_raw = round1.get("hypothesis_raw")
        b_sat = {c["type"]: _check(c["type"], c["value"], b_nl, b_raw, cond_ids) for c in conditions}
        for ctype, ok in b_sat.items():
            b_per_type.setdefault(ctype, []).append(int(ok))
        b_all_ok = list(b_sat.values())
        b_per_hyp_avg.append(sum(b_all_ok) / len(b_all_ok))
        b_per_hyp_joint.append(int(all(b_all_ok)))

    n = len(jaccards)
    print(f"Records evaluated: {n}  (skipped {skipped})\n")

    def _row(name, arr):
        a = np.array(arr)
        print(f"  {name:<22s}  mean={a.mean():.4f}  std={a.std():.4f}")

    print("=== Baseline (Round 1) ===")
    _row("Jaccard", b_jaccards)
    _row("Dice", b_dices)
    _row("Overlap", b_overlaps)
    print("\n=== Baseline Condition Accuracy (per type) ===")
    for ctype, hits in sorted(b_per_type.items()):
        _row(ctype, hits)
    print("\n=== Baseline Per-Hypothesis Condition Accuracy ===")
    _row("Avg cond acc / hyp", b_per_hyp_avg)
    _row("Joint acc / hyp", b_per_hyp_joint)

    print("\n=== Best Retrieval Metrics ===")
    _row("Jaccard", jaccards)
    _row("Dice", dices)
    _row("Overlap", overlaps)

    print("\n=== Best Condition Accuracy (per type) ===")
    for ctype, hits in sorted(per_type.items()):
        _row(ctype, hits)
    _row("Joint (all conds)", joint)

    print("\n=== Best Per-Hypothesis Condition Accuracy ===")
    _row("Avg cond acc / hyp", per_hyp_avg)
    _row("Joint acc / hyp", per_hyp_joint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--modelname", required=True)
    parser.add_argument("--log_root", default="log")
    parser.add_argument("--data_root", default="data")
    args = parser.parse_args()

    log_path = os.path.join(args.log_root, args.dataname, f"singleturn_{args.modelname}.jsonl")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log not found: {log_path}")
    print(f"Log: {log_path}\n")
    evaluate(log_path, args.data_root)


if __name__ == "__main__":
    main()
