import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from akgr.kgdata import load_kg

_NL_PROMPT = """You are a helpful assistant that converts a first-order logic query to natural language.
Operators: i=intersection, u=union, p=relation/projection, e=entity, n=negation.

Examples:
  p(live_in, e(stanford))  ->  "entities that have a 'live_in' link to stanford"
  i(p(live_in, e(stanford)), p(born_in, e(usa)))  ->  "entities that have a 'live_in' link to stanford AND a 'born_in' link to usa"
  u(p(live_in, e(stanford)), p(born_in, e(usa)))  ->  "entities that have a 'live_in' link to stanford OR a 'born_in' link to usa"
  i(n(p(live_in, e(stanford))), p(born_in, e(usa)))  ->  "entities that do NOT have a 'live_in' link to stanford, AND have a 'born_in' link to usa"

Now convert this query to natural language:
{query}

Return only the natural language description, no other text."""

prompt = _NL_PROMPT  # keep backward-compat name
def decode_entity_name(ent_id: int, ent_id2name: Dict[int, str]) -> str:
    return str(ent_id2name.get(ent_id, f"entity_{ent_id}"))


def decode_relation_name(rel_token: int, rel_id2name: Dict[int, str]) -> str:
    # query 里的 relation token 是负数，即 -rid（rid 是 0-indexed 的原始关系 ID）
    rel_id = abs(int(rel_token))
    return str(rel_id2name.get(rel_id, f"relation_{rel_id}"))


def _unshift_token(tok):
    """No-op for tokens already in KG format. Kept for backward compat."""
    try:
        return int(tok)
    except (ValueError, TypeError):
        return tok


def _action_str_to_expr(tokens: List[Any], ent_id2name: Dict[int, str], rel_id2name: Dict[int, str], i: int = 0) -> Tuple[str, int]:
    """Parse flat action string tokens (no parentheses): i/u/n/p/e + numeric ids."""
    op = str(tokens[i])
    i += 1

    if op == "e":
        name = decode_entity_name(int(tokens[i]), ent_id2name)
        return f"e({name})", i + 1

    if op == "p":
        rel_name = decode_relation_name(int(tokens[i]), rel_id2name)
        i += 1
        child_expr, i = _action_str_to_expr(tokens, ent_id2name, rel_id2name, i)
        return f"p({rel_name}, {child_expr})", i

    if op in {"i", "u"}:
        # consume exactly 2 sub-expressions
        children = []
        for _ in range(2):
            child_expr, i = _action_str_to_expr(tokens, ent_id2name, rel_id2name, i)
            children.append(child_expr)
        return f"{op}({', '.join(children)})", i

    if op == "n":
        child_expr, i = _action_str_to_expr(tokens, ent_id2name, rel_id2name, i)
        return f"n({child_expr})", i

    # numeric token: treat as entity
    try:
        raw_id = _unshift_token(op)
        val = int(raw_id)
        if val >= 0:
            name = decode_entity_name(val, ent_id2name)
            return f"e({name})", i
        else:
            rel_name = decode_relation_name(int(op), rel_id2name)
            return f"rel({rel_name})", i
    except Exception:
        return op, i


def _tokens_to_expr(tokens: List[Any], ent_id2name: Dict[int, str], rel_id2name: Dict[int, str], i: int = 0) -> Tuple[str, int]:
    """Recursively convert token list to a structured expression string like i(p(rel, e(ent)), ...)."""
    if i >= len(tokens) or tokens[i] != "(":
        raise ValueError(f"Expected '(' at {i}")
    i += 1
    op = tokens[i]
    i += 1

    if op == "e":
        # ( e ( id ) )
        name = decode_entity_name(int(tokens[i + 1]), ent_id2name)
        return f"e({name})", i + 4  # skip ( id ) )

    if op == "p":
        # ( p ( rel_token ) child )
        rel_name = decode_relation_name(int(tokens[i + 1]), rel_id2name)
        i = i + 3  # skip ( rel )
        child_expr, i = _tokens_to_expr(tokens, ent_id2name, rel_id2name, i)
        i += 1  # skip closing )
        return f"p({rel_name}, {child_expr})", i

    if op in {"i", "u"}:
        children = []
        while i < len(tokens) and tokens[i] != ")":
            child_expr, i = _tokens_to_expr(tokens, ent_id2name, rel_id2name, i)
            children.append(child_expr)
        i += 1  # skip closing )
        return f"{op}({', '.join(children)})", i

    if op == "n":
        child_expr, i = _tokens_to_expr(tokens, ent_id2name, rel_id2name, i)
        i += 1  # skip closing )
        return f"n({child_expr})", i

    raise ValueError(f"Unknown operator '{op}'")


def query_to_natural_language(
    query_tokens: List[Any],
    ent_id2name: Dict[int, str],
    rel_id2name: Dict[int, str],
    llm_call=None,
) -> str:
    """
    Convert query token list to natural language via LLM.

    Builds a structured expression string (e.g. i(p(rel, e(ent)), n(p(...))))
    and passes it to the LLM with _NL_PROMPT.

    Args:
        llm_call: callable(prompt_str) -> str. If None, falls back to tree-based NL.
    """
    # Support string input (flat action string from model output)
    if isinstance(query_tokens, str):
        from akgr.utils.parsing_util import qry_actionstr_2_wordlist
        query_tokens = qry_actionstr_2_wordlist(query_tokens)

    is_flat = len(query_tokens) > 0 and query_tokens[0] != "("
    try:
        if is_flat:
            expr, _ = _action_str_to_expr(query_tokens, ent_id2name, rel_id2name, 0)
        else:
            expr, _ = _tokens_to_expr(query_tokens, ent_id2name, rel_id2name, 0)
    except Exception:
        return str(query_tokens)

    if llm_call is None:
        # No LLM available: return the structured expression as-is
        return expr

    prompt_text = _NL_PROMPT.format(query=expr)
    return llm_call(prompt_text).strip()


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