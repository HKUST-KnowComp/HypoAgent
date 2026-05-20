import json
from akgr.kgdata import load_kg

data_path = "./sampled_data/DBpedia50/False/DBpedia50-full-32-test-a2q.jsonl"

kg = load_kg(
    dataroot="./sampled_data",
    dataname="DBpedia50",
    reverse_edges_flag=False,
)

ent_id2name = kg.ent_id2name
rel_id2name = kg.rel_id2name


def decode_entity_token(x: int):
    # query/action 里的正整数通常是 1-based entity token
    raw_eid = x - 1
    return ent_id2name.get(raw_eid, f"UNK_ENT_{raw_eid}")


def decode_relation_token(x: int):
    # query/action 里的负整数通常是 -(rid + 1)
    raw_rid = abs(x) - 1
    return rel_id2name.get(raw_rid, f"UNK_REL_{raw_rid}")


def pretty_query(tokens):
    out = []
    for tok in tokens:
        if isinstance(tok, str):
            out.append(tok)
        elif isinstance(tok, int):
            if tok < 0:
                out.append(f"{tok}[{decode_relation_token(tok)}]")
            else:
                out.append(f"{tok}[{decode_entity_token(tok)}]")
        else:
            out.append(str(tok))
    return " ".join(out)


with open(data_path, "r") as f:
    for idx, line in enumerate(f):
        ex = json.loads(line)

        print(f"\n===== SAMPLE {idx} =====")
        print("pattern_str:", ex["pattern_str"])
        print("query raw:", ex["query"])
        print("query pretty:", pretty_query(ex["query"]))

        print("answers raw:", ex["answers"][:10])
        print(
            "answers pretty:",
            [ent_id2name.get(a, f"UNK_ENT_{a}") for a in ex["answers"][:10]]
        )

        if idx >= 4:
            break