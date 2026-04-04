import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from akgr.kgdata import load_kg


def decode_entity_name(ent_id: int, ent_id2name: Dict[int, str]) -> str:
    return str(ent_id2name.get(ent_id, f"entity_{ent_id}"))


def decode_relation_name(rel_token: int, rel_id2name: Dict[int, str]) -> str:
    # query 里的 relation token 是负数，通常是 -(rid + 1)
    rel_id = abs(int(rel_token)) - 1
    return str(rel_id2name.get(rel_id, f"relation_{rel_id}"))


def parse_query(tokens: List[Any], i: int = 0) -> Tuple[Dict[str, Any], int]:
    if i >= len(tokens) or tokens[i] != "(":
        raise ValueError(f"Expected '(' at index {i}, got: {tokens[i] if i < len(tokens) else 'EOF'}")

    i += 1
    op = tokens[i]
    i += 1

    if op == "e":
        if tokens[i] != "(":
            raise ValueError(f"Expected '(' after e at index {i}")
        ent_id = int(tokens[i + 1])
        if tokens[i + 2] != ")" or tokens[i + 3] != ")":
            raise ValueError(f"Malformed entity expression around index {i}")
        return {"type": "entity", "id": ent_id}, i + 4

    if op == "p":
        if tokens[i] != "(":
            raise ValueError(f"Expected '(' after p at index {i}")
        rel_token = int(tokens[i + 1])
        if tokens[i + 2] != ")":
            raise ValueError(f"Malformed relation expression around index {i}")
        i = i + 3
        child, i = parse_query(tokens, i)
        if tokens[i] != ")":
            raise ValueError(f"Expected ')' to close projection at index {i}")
        return {"type": "path", "rel": rel_token, "child": child}, i + 1

    if op in {"i", "u"}:
        children = []
        while i < len(tokens) and tokens[i] != ")":
            child, i = parse_query(tokens, i)
            children.append(child)
        if i >= len(tokens) or tokens[i] != ")":
            raise ValueError(f"Expected ')' to close {op} at index {i}")
        node_type = "intersection" if op == "i" else "union"
        return {"type": node_type, "children": children}, i + 1

    if op == "n":
        child, i = parse_query(tokens, i)
        if i >= len(tokens) or tokens[i] != ")":
            raise ValueError(f"Expected ')' to close negation at index {i}")
        return {"type": "negation", "child": child}, i + 1

    raise ValueError(f"Unknown operator '{op}'")


def query_to_natural_language(query_tokens: List[Any], ent_id2name: Dict[int, str], rel_id2name: Dict[int, str]) -> str:
    tree, idx = parse_query(query_tokens, 0)
    if idx != len(query_tokens):
        raise ValueError(f"Unconsumed query tokens: {query_tokens[idx:]}")
    return tree_to_natural_language(tree, ent_id2name, rel_id2name)


def tree_to_natural_language(node: Dict[str, Any], ent_id2name: Dict[int, str], rel_id2name: Dict[int, str]) -> str:
    node_type = node["type"]

    if node_type == "entity":
        return decode_entity_name(int(node["id"]), ent_id2name)

    if node_type == "path":
        rel = decode_relation_name(int(node["rel"]), rel_id2name)
        child_text = tree_to_natural_language(node["child"], ent_id2name, rel_id2name)
        return f"entities connected via relation '{rel}' to {child_text}"

    if node_type == "intersection":
        parts = [tree_to_natural_language(c, ent_id2name, rel_id2name) for c in node["children"]]
        return "entities that satisfy all of: " + " ; ".join(parts)

    if node_type == "union":
        parts = [tree_to_natural_language(c, ent_id2name, rel_id2name) for c in node["children"]]
        return "entities that satisfy at least one of: " + " ; ".join(parts)

    if node_type == "negation":
        child_text = tree_to_natural_language(node["child"], ent_id2name, rel_id2name)
        return f"entities that do NOT satisfy: {child_text}"

    return str(node)


def convert_one_record(record: Dict[str, Any], ent_id2name: Dict[int, str], rel_id2name: Dict[int, str]) -> Dict[str, Any]:
    query_tokens = record.get("query", [])
    query_nl = query_to_natural_language(query_tokens, ent_id2name, rel_id2name)

    answers = record.get("answers")
    if answers is None and "answer" in record:
        answers = record["answer"]

    if isinstance(answers, list):
        answers_nl = [decode_entity_name(int(a), ent_id2name) for a in answers]
    elif answers is None:
        answers_nl = []
    else:
        answers_nl = [decode_entity_name(int(answers), ent_id2name)]

    out = dict(record)
    out["query_nl"] = query_nl
    out["answers_nl"] = answers_nl
    return out


def convert_jsonl(input_file: str, output_file: str, ent_id2name: Dict[int, str], rel_id2name: Dict[int, str], max_samples: int = -1) -> None:
    n_total = 0
    n_ok = 0
    n_err = 0

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if max_samples >= 0 and n_total >= max_samples:
                break
            n_total += 1

            try:
                record = json.loads(line)
                converted = convert_one_record(record, ent_id2name, rel_id2name)
                fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:
                n_err += 1
                fout.write(
                    json.dumps(
                        {
                            "raw_line": line.strip(),
                            "convert_error": str(e),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"[DONE] total={n_total}, success={n_ok}, failed={n_err}")
    print(f"[OUTPUT] {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert jsonl query/answers to natural language.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/home/ycaicr/CtrlHGen/sampled_data/DBpedia50/False/DBpedia50-full-32-test-a2q.jsonl",
    )
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--data_root", type=str, default="/home/ycaicr/CtrlHGen/sampled_data")
    parser.add_argument("--dataname", type=str, default="DBpedia50")
    parser.add_argument("--reverse_edges_flag", action="store_true")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Only process first N lines; -1 means all",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_file = args.output_file
    if not output_file:
        p = Path(args.input_file)
        output_file = str(p.with_name(f"{p.stem}-nl.jsonl"))

    kg = load_kg(
        dataroot=args.data_root,
        dataname=args.dataname,
        reverse_edges_flag=args.reverse_edges_flag,
    )

    convert_jsonl(
        input_file=args.input_file,
        output_file=output_file,
        ent_id2name=kg.ent_id2name,
        rel_id2name=kg.rel_id2name,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()