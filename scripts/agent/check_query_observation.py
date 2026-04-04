import argparse
import ast
import json
import re
from typing import Any, List

QUERY_SYMBOLS = {"(", ")", "p", "e", "i", "u", "n"}


def _to_token_list(raw: str) -> List[Any]:
    """
    Parse query tokens from:
    - JSON list string, e.g. ["(", "p", "(", -119, ")", ...]
    - Python list literal string
    - Python list literal for union/intersection queries, e.g.
      ['(', 'u', '(', 'p', '(', -12, ')', '(', 'e', '(', 5838, ')', ')', ')', ...]
    - whitespace-separated tokens, e.g. ( p ( -119 ) ( e ( 2916 ) ) )
    """
    text = raw.strip()
    if not text:
        raise ValueError("Empty --query string")

    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
            if isinstance(obj, list):
                return _validate_query_tokens([_normalize_token(x) for x in obj], raw)
        except Exception:
            pass

    tokens = re.findall(
        r"-?\d+|[()]|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|[A-Za-z_][A-Za-z0-9_]*",
        text,
    )
    if not tokens:
        raise ValueError(f"Could not parse --query string: {raw}")
    return _validate_query_tokens([_normalize_token(x) for x in tokens], raw)


def _auto_int(x: Any) -> Any:
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if s.startswith("-") and s[1:].isdigit():
        return int(s)
    if s.isdigit():
        return int(s)
    return s


def _normalize_token(x: Any) -> Any:
    if isinstance(x, (int, float)):
        return _auto_int(x)

    s = str(x).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {"'", '"'}:
        try:
            s = ast.literal_eval(s)
        except Exception:
            s = s[1:-1]
    return _auto_int(s)


def _validate_query_tokens(tokens: List[Any], raw: str) -> List[Any]:
    if not tokens:
        raise ValueError("Parsed query token list is empty")

    balance = 0
    for token in tokens:
        if isinstance(token, str):
            if token not in QUERY_SYMBOLS:
                raise ValueError(f"Unsupported query token {token!r} parsed from: {raw}")
            if token == "(":
                balance += 1
            elif token == ")":
                balance -= 1
                if balance < 0:
                    raise ValueError(f"Unbalanced query parentheses in: {raw}")

    if balance != 0:
        raise ValueError(f"Unbalanced query parentheses in: {raw}")

    return tokens


def _name(ent_id: int, ent_id2name: dict) -> str:
    return str(ent_id2name.get(ent_id, f"UNK_ENT_{ent_id}"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a KG query and check whether a target observation set is covered."
    )
    parser.add_argument("--data_root", type=str, default="/home/ycaicr/CtrlHGen/sampled_data")
    parser.add_argument("--dataname", type=str, default="DBpedia50")
    parser.add_argument("--reverse_edges_flag", action="store_true")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Which graph sampler split to search on.",
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query token list as JSON/Python-list string, or whitespace token string.",
    )
    parser.add_argument(
        "--query_shifted",
        action="store_true",
        help="Set this if your query uses shifted indices (entity +1, relation -(rid+1)).",
    )

    parser.add_argument(
        "--observation_names",
        nargs="*",
        default=[],
        help='Observation entities by name. Example: --observation_names "Hammerhead_shark" "Bonnethead"',
    )
    parser.add_argument(
        "--observation_ids",
        nargs="*",
        type=int,
        default=[],
        help="Observation entities by id.",
    )
    parser.add_argument(
        "--observation_ids_shifted",
        action="store_true",
        help="Set this if provided observation ids are shifted (+1).",
    )
    parser.add_argument(
        "--print_topk_answers",
        type=int,
        default=30,
        help="Print up to top-K answer entities for inspection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from akgr.kgdata import load_kg
    from akgr.agent.kg_mapper import KGNameMapper
    from akgr.utils.parsing_util import ans_unshift_indices, qry_unshift_indices

    kg = load_kg(
        dataroot=args.data_root,
        dataname=args.dataname,
        reverse_edges_flag=args.reverse_edges_flag,
    )
    mapper = KGNameMapper(kg)

    query_tokens = _to_token_list(args.query)
    query_unshifted = qry_unshift_indices(query_tokens) if args.query_shifted else query_tokens

    obs_ids = list(args.observation_ids)
    if args.observation_ids_shifted and obs_ids:
        obs_ids = ans_unshift_indices(obs_ids)

    for name in args.observation_names:
        obs_ids.append(mapper.get_entity_id(name))

    obs_ids = sorted(set(obs_ids))
    if not obs_ids:
        raise ValueError("Please provide at least one observation via --observation_names or --observation_ids.")

    answers = kg.graph_samplers[args.split].search_answers_to_query(query_unshifted)
    if answers is None:
        raise ValueError(
            "Query could not be evaluated by the graph sampler. "
            f"Parsed query tokens: {query_unshifted}"
        )
    answer_set = set(answers)
    obs_set = set(obs_ids)

    covered = sorted(obs_set & answer_set)
    missing = sorted(obs_set - answer_set)
    extra = sorted(answer_set - obs_set)

    print("=== Query Check Result ===")
    print(f"split: {args.split}")
    print(f"query(raw): {query_tokens}")
    print(f"query(unshifted_for_search): {query_unshifted}")
    print(f"answers_count: {len(answer_set)}")
    print(f"observation_count: {len(obs_set)}")
    print(f"covered_count: {len(covered)}")
    print(f"missing_count: {len(missing)}")
    print(f"is_observation_subset_of_answers: {len(missing) == 0}")

    print("\n[covered]")
    for eid in covered:
        print(f"- {eid}\t{_name(eid, kg.ent_id2name)}")

    print("\n[missing]")
    for eid in missing:
        print(f"- {eid}\t{_name(eid, kg.ent_id2name)}")

    print(f"\n[top_{args.print_topk_answers}_answers]")
    for eid in sorted(answer_set)[: args.print_topk_answers]:
        print(f"- {eid}\t{_name(eid, kg.ent_id2name)}")

    if extra:
        print(f"\n[extra_answers_not_in_observation] count={len(extra)}")


if __name__ == "__main__":
    main()
