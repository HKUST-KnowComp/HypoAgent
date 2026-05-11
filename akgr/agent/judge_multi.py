"""
Evaluate multiturn log results with LLM-based condition accuracy.

Usage:
    python akgr/agent/judge_multi.py --dataname PharmKG8k --modelname DeepSeek-V4-Flash
"""

import json
import os
import argparse
import numpy as np
from openai import OpenAI
from tqdm import tqdm


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


def _load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    return {}


def _save_cache(cache_path: str, cache: dict):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _parse_hypothesis(raw: str, rel_id2name: dict, ent_id2name: dict) -> dict:
    """Extract structured info from flat action string."""
    tokens = raw.strip().split()
    relations, entities, ops = [], [], []
    for t in tokens:
        try:
            n = int(t)
            if n < 0:
                name = rel_id2name.get(-n, str(-n))
                relations.append(name)
            else:
                name = ent_id2name.get(n, str(n))
                entities.append(name)
        except ValueError:
            if t in ("i", "u", "n"):
                ops.append(t)
    pattern = " ".join("p" if t.lstrip("-").isdigit() and int(t) < 0
                       else ("e" if t.lstrip("-").isdigit() else t)
                       for t in tokens)
    return {
        "relations": relations,
        "entities": entities,
        "relationnumber": len(relations),
        "entitynumber": len(entities),
        "pattern": pattern,
    }


def _llm_judge(client: OpenAI, model_id: str,
               history_conditions: list[str], current_condition: str,
               hypothesis_raw: str, hypothesis_nl: str,
               hyp_info: dict,
               rec_cache: list) -> bool:
    """
    Check rec_cache (list of turn dicts) for an existing result, else call LLM.
    Returns bool. Does NOT save cache — caller saves after appending.
    """
    for entry in rec_cache:
        if entry["condition"] == current_condition and entry["hypothesis_raw"] == hypothesis_raw:
            return entry["result"]

    hyp_desc = hypothesis_nl
    hist_text = ("\n".join(f"  - {c}" for c in history_conditions)) if history_conditions else "  (none)"

    prompt = (
        f"You are evaluating whether a knowledge graph hypothesis satisfies a user condition.\n\n"
        f"Hypothesis: {hyp_desc}\n"
        f"Raw action string: {hypothesis_raw}\n\n"
        f"Parsed hypothesis properties:\n"
        f"  - Logic pattern: {hyp_info['pattern']}\n"
        f"  - Relations ({hyp_info['relationnumber']}): {hyp_info['relations']}\n"
        f"  - Entities ({hyp_info['entitynumber']}): {hyp_info['entities']}\n\n"
        f"Conversation history (previous conditions, for context only):\n"
        f"{hist_text}\n\n"
        f"Current condition to judge: {current_condition}\n\n"
        f"Does the hypothesis satisfy the current condition? "
        f"Use the history only as context to interpret what the current condition means "
        f"(e.g. 'make it more complex' means relative to the previous condition). "
        f"Reply with ONLY a JSON object: {{\"result\": true}} or {{\"result\": false}}, nothing else."
    )

    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return bool(json.loads(text)["result"])


def evaluate(log_path: str, client: OpenAI, model_id: str, cache_path: str, data_root: str = "data"):
    records = _load_records(log_path)
    cache: dict = _load_cache(cache_path)  # {answers_key: {"answers": [...], "turns": [...]}}

    # Load KG id->name mappings
    dataname = log_path.split(os.sep)[-2]
    import pickle
    with open(os.path.join(data_root, dataname, f"{dataname}.pkl"), "rb") as f:
        kg = pickle.load(f)
    rel_id2name, ent_id2name = kg.rel_id2name, kg.ent_id2name

    jaccards, dices, overlaps = [], [], []
    per_hyp_avg: list[float] = []
    per_hyp_joint: list[int] = []
    skipped = 0

    for r in tqdm(records, desc="Evaluating"):
        if "error" in r or "turns" not in r:
            skipped += 1
            continue

        answers_key = json.dumps(r.get("answers", []), ensure_ascii=False)
        rec_entry = cache.get(answers_key, [])
        rec_cache: list = rec_entry
        rec_dirty = False
        accumulated_conditions: list[str] = []

        for turn in r["turns"]:
            rb = turn.get("round_best")
            if not rb or rb.get("hypothesis_raw") is None or rb.get("hypothesis_nl") is None:
                continue

            condition = turn.get("condition", "")
            jaccards.append(rb["jaccard"])
            dices.append(rb["dice"])
            overlaps.append(rb["overlap"])

            if not condition:
                per_hyp_avg.append(1.0)
                per_hyp_joint.append(1)
                accumulated_conditions.append(condition)
                continue

            try:
                hyp_info = _parse_hypothesis(rb["hypothesis_raw"], rel_id2name, ent_id2name)
                ok = _llm_judge(
                    client, model_id,
                    list(accumulated_conditions),
                    condition,
                    rb["hypothesis_raw"],
                    rb["hypothesis_nl"],
                    hyp_info,
                    rec_cache,
                )
                existing = any(
                    e["condition"] == condition and e["hypothesis_raw"] == rb["hypothesis_raw"]
                    for e in rec_cache
                )
                if not existing:
                    rec_cache.append({
                        "condition": condition,
                        "hypothesis_raw": rb["hypothesis_raw"],
                        "hypothesis_nl": rb["hypothesis_nl"],
                        "result": ok,
                    })
                    rec_dirty = True
                per_hyp_avg.append(float(ok))
                per_hyp_joint.append(int(ok))
            except Exception as e:
                print(f"  [WARN] answers={answers_key[:40]} LLM judge failed: {e}")
                per_hyp_avg.append(0.0)
                per_hyp_joint.append(0)

            accumulated_conditions.append(condition)

        if rec_dirty:
            cache[answers_key] = rec_cache
            _save_cache(cache_path, cache)

    n = len(jaccards)
    print(f"Turns evaluated: {n}  (skipped {skipped} records)\n")

    def _row(name, arr):
        a = np.array(arr)
        print(f"  {name:<24s}  mean={a.mean():.4f}  std={a.std():.4f}")

    print("=== Retrieval Metrics (per turn) ===")
    _row("Jaccard", jaccards)
    _row("Dice", dices)
    _row("Overlap", overlaps)

    print("\n=== Condition Accuracy (per turn, LLM) ===")
    _row("Avg cond acc / turn", per_hyp_avg)
    _row("Joint acc / turn", per_hyp_joint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--modelname", required=True)
    parser.add_argument("--log_root", default="log")
    parser.add_argument("--analysis", action="store_true", help="Use _analysis log variant")
    parser.add_argument("--api_config", default="akgr/configs/api_keys.yml")
    parser.add_argument("--data_root", default="data")
    args = parser.parse_args()

    suffix = "_analysis" if args.analysis else ""
    log_path = os.path.join(args.log_root, args.dataname, f"multiturn_{args.modelname}{suffix}.jsonl")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log not found: {log_path}")

    cache_path = log_path.replace(".jsonl", f"_judge_cache.json")

    import yaml
    with open(args.api_config) as f:
        api_cfg = yaml.safe_load(f)["deepinfra"]
    client = OpenAI(api_key=api_cfg["api_key"], base_url=api_cfg["api_base"])
    model_id = "deepseek-ai/DeepSeek-V4-Flash"

    print(f"Log:   {log_path}")
    print(f"Cache: {cache_path}")
    print(f"Judge: {model_id}\n")
    evaluate(log_path, client, model_id, cache_path, args.data_root)


if __name__ == "__main__":
    main()
