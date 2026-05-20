#!/usr/bin/env python3
from akgr.agent.single import build_adapter
from akgr.agent.case import case_pni, case_3i, case_2p, case_1p, case_2i, case_pi, case_inp, case_ip, case_up, case_2u, case_2in, case_3in, case_pin
from akgr.agent.tools import _enumerate_subquery_combinations, graph_validation_tool
import json

hypothesis_model_path = 'checkpoints/PharmKG8k-full-32-160-multi.pth'
data_root = './data/'
dataname = 'PharmKG8k'

adapter = build_adapter(hypothesis_model_path, data_root, dataname)
kg = adapter.kg
graph_sampler = kg.graph_samplers["test"]

all_cases = [
    ("1p",  case_1p),
    ("2p",  case_2p),
    ("2i",  case_2i),
    ("3i",  case_3i),
    ("ip",  case_ip),
    ("pi",  case_pi),
    ("2u",  case_2u),
    ("up",  case_up),
    ("inp", case_inp),
    ("pni", case_pni),
    ("2in", case_2in),
    ("3in", case_3in),
    ("pin", case_pin),
]

for name, case in all_cases:
    tokens = case["query"]
    label_set = set(case["answers"])
    subs = _enumerate_subquery_combinations(tokens)
    print(f"\n{'='*50}")
    print(f"Pattern: {name}  ({case['pattern_str']})")
    print(f"Sub-queries: {len(subs)}")
    for sq in subs:
        try:
            ans = set(graph_sampler.search_answers_to_query(sq))
            overlap = len(ans & label_set)
            jaccard = overlap / len(ans | label_set) if (ans | label_set) else 0.0
            print(f"  {sq}")
            print(f"    count={len(ans)}, overlap={overlap}, jaccard={jaccard:.4f}")
        except Exception as e:
            print(f"  {sq}")
            print(f"    ERROR: {e}")
