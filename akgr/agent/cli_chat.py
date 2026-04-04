import os
import uuid
import argparse
import json
import ast
import re

from akgr.agent.chat_session import ChatSession, TurnRecord
from akgr.agent.ctrlhgen_adapter import CtrlHGenAdapter
from akgr.agent.llm_parser import LocalQwenParser
from akgr.agent.llm_verbalizer import LocalQwenVerbalizer
from akgr.utils.load_util import load_yaml
from akgr.kgdata import load_kg


def parse_args():
    parser = argparse.ArgumentParser(description="CtrlHGen interactive chat agent")


    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained CtrlHGen checkpoint"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of KG data"
    )
    parser.add_argument(
        "--dataname",
        type=str,
        required=True,
        help="Dataset name, e.g. PharmKG8k"
    )
    parser.add_argument(
        "--reverse_edges_flag",
        action="store_true",
        help="Whether to load KG with reverse edges"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="GPT2_6_act_nt",
        help="CtrlHGen model name"
    )
    parser.add_argument(
        "--config_dataloader",
        type=str,
        default="akgr/configs/config-dataloader.yml",
        help="Path to dataloader config yaml"
    )
    parser.add_argument(
        "--config_model",
        type=str,
        default="akgr/configs/config-model.yml",
        help="Path to model config yaml"
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Local parser/verbalizer LLM name"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for agent generation"
    )
    parser.add_argument(
        "--test_top_k",
        type=int,
        default=0,
        help="Top-k sampling used in generation (same as test_loop)"
    )
    parser.add_argument(
        "--constrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable constrained decoding for action generation. Defaults to True for act models."
    )
    parser.add_argument(
        "--regen_until_overlap_max_rounds",
        type=int,
        default=1,
        help="Extra generate rounds in chat when no observation-overlap triples are found."
    )
    parser.add_argument(
        "--fallback_query",
        type=str,
        default="",
        help="Fallback query token list (JSON/Python list/space tokens) used when generated query finds no overlap triples."
    )
    parser.add_argument('--vs', action='store_true', help='verbose flag for smatch result')

    return parser.parse_args()


def _split_csv(text: str):
    return [item.strip() for item in text.split(",") if item.strip()]


def _is_affirmative(text: str) -> bool:
    t = (text or "").strip().lower()
    t = re.sub(r"[^\w\s\u3400-\u9fff]+", " ", t)
    t = " ".join(t.split())
    if not t:
        return False
    positives = {"y", "yes", "yeah", "yep", "true", "ok", "sure", "是", "好", "对"}
    if t in positives:
        return True
    return any(t.startswith(p + " ") for p in positives)


def _is_negative(text: str) -> bool:
    t = (text or "").strip().lower()
    t = re.sub(r"[^\w\s\u3400-\u9fff]+", " ", t)
    t = " ".join(t.split())
    if not t:
        return False
    negatives = {"n", "no", "nope", "false", "不是", "不", "否"}
    if t in negatives:
        return True
    return any(t.startswith(p + " ") for p in negatives)


def _derive_condition_from_followup(question: str, answer: str):
    """
    Convert follow-up QA into a condition request text that parser can consume.
    Example:
      Q: 是否有与“锤头鲨”相关的三元组？  A: yes
      -> "related to 锤头鲨"
    """
    is_yes = _is_affirmative(answer)
    is_no = _is_negative(answer)
    if not is_yes and not is_no:
        return None

    q = (question or "").strip()
    if not q:
        return None

    quoted = re.findall(r"[\"“']([^\"”']+)[\"”']", q)
    if quoted:
        term = quoted[0].strip()
        if term:
            return f"related to {term}" if is_yes else f"not related to {term}"

    m_related = re.search(r"related to\s+(.+?)[\?\.!]*$", q, flags=re.IGNORECASE)
    if m_related:
        term = m_related.group(1).strip()
        if term:
            return f"related to {term}" if is_yes else f"not related to {term}"

    m_relation = re.search(r"through the\s+(.+?)\s+relation", q, flags=re.IGNORECASE)
    if not m_relation:
        m_relation = re.search(r"relation\s+(.+?)[\?\.!]*$", q, flags=re.IGNORECASE)
    if m_relation:
        rel = m_relation.group(1).strip()
        if rel:
            return f"focus on relation {rel}" if is_yes else f"exclude relation {rel}"

    return None


def _normalize_query_token(x):
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if s.startswith("-") and s[1:].isdigit():
        return int(s)
    if s.isdigit():
        return int(s)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {"'", '"'}:
        try:
            s = ast.literal_eval(s)
        except Exception:
            s = s[1:-1]
        s = str(s).strip()
        if s.startswith("-") and s[1:].isdigit():
            return int(s)
        if s.isdigit():
            return int(s)
    return s


def _parse_query_text(raw: str):
    text = (raw or "").strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
            if isinstance(obj, list):
                return [_normalize_query_token(v) for v in obj]
        except Exception:
            pass
    tokens = re.findall(
        r"-?\d+|[()]|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|[A-Za-z_][A-Za-z0-9_]*",
        text,
    )
    if not tokens:
        return None
    return [_normalize_query_token(v) for v in tokens]


def _build_parsed_control(observation_entities, condition_parsed: dict, condition_text: str):
    condition_parsed = condition_parsed or {}
    conditions = condition_parsed.get("conditions", [{"type": "unconditional", "value": ""}])

    # Case 1: empty user condition should always stay unconditional.
    if not condition_text.strip():
        conditions = [{"type": "unconditional", "value": ""}]
    elif not isinstance(conditions, list) or not conditions:
        conditions = [{"type": "unconditional", "value": ""}]
    else:
        normalized_conditions = []
        for item in conditions:
            if not isinstance(item, dict):
                continue
            ctype = str(item.get("type", "")).strip()
            cvalue = item.get("value", "")
            if not ctype:
                continue
            normalized_conditions.append({"type": ctype, "value": cvalue})
        conditions = normalized_conditions or [{"type": "unconditional", "value": ""}]
    return {
        "observation_entities": observation_entities,
        "conditions": conditions,
        # Keep backward-compatible single fields for session memory.
        # "condition_type": conditions[0]["type"],
        # "condition_value": conditions[0]["value"],
        # Extra fields for transparency/debug.
        # "condition_prompt_hints": condition_parsed.get("condition_prompt_hints", []),
        "condition_interpretation": condition_parsed.get("condition_interpretation", {}),
        "raw_condition_text": condition_text,
    }


def _collect_observation_intersection_triples(kg, query_tokens, observation_entity_ids, split: str = "test"):
    if not isinstance(query_tokens, list) or not query_tokens:
        return []
    sampler = kg.graph_samplers.get(split)
    if sampler is None:
        sampler = next(iter(kg.graph_samplers.values()))
    answers = sampler.search_answers_to_query(query_tokens) or []
    obs_set = set(int(x) for x in observation_entity_ids)
    overlap_nodes = sorted(obs_set & set(int(x) for x in answers))
    if not overlap_nodes:
        return []

    triples = []
    seen = set()
    for node in overlap_nodes:
        for h, t, r in sampler.out_edges(node):
            h, t, r = int(h), int(t), int(r)
            key = (h, r, t)
            if key in seen:
                continue
            seen.add(key)
            triples.append(
                {
                    "head_id": h,
                    "head_name": str(kg.ent_id2name.get(h, f"UNK_ENT_{h}")),
                    "relation_id": r,
                    "relation_name": str(kg.rel_id2name.get(r, f"UNK_REL_{r}")),
                    "tail_id": t,
                    "tail_name": str(kg.ent_id2name.get(t, f"UNK_ENT_{t}")),
                    "intersection_entity_id": node,
                }
            )
        for h, t, r in sampler.in_edges(node):
            h, t, r = int(h), int(t), int(r)
            key = (h, r, t)
            if key in seen:
                continue
            seen.add(key)
            triples.append(
                {
                    "head_id": h,
                    "head_name": str(kg.ent_id2name.get(h, f"UNK_ENT_{h}")),
                    "relation_id": r,
                    "relation_name": str(kg.rel_id2name.get(r, f"UNK_REL_{r}")),
                    "tail_id": t,
                    "tail_name": str(kg.ent_id2name.get(t, f"UNK_ENT_{t}")),
                    "intersection_entity_id": node,
                }
            )
    return triples


def main():
    args = parse_args()
    if args.constrained is None:
        args.constrained = ("act" in str(args.model_name))
    fallback_query_tokens = _parse_query_text(args.fallback_query)

    config_dataloader = load_yaml(args.config_dataloader)
    special_tokens = config_dataloader["special_tokens"]
    offset = config_dataloader["offset"]
    is_act=('act' in args.model_name)
    tgt_len = config_dataloader['act_len'] + 1 if is_act else config_dataloader['qry_len'] + 1
    src_len = config_dataloader['ans_len'] + 1

    config_model = load_yaml(args.config_model)

    session = ChatSession(session_id=str(uuid.uuid4()))
    parser = LocalQwenParser(args.llm_name)
    verbalizer = LocalQwenVerbalizer(args.llm_name)

    kg = load_kg(
        dataroot=args.data_root,
        dataname=args.dataname,
        reverse_edges_flag=args.reverse_edges_flag,
    )

    # Read nentity/nrelation from stats.txt (same source as main_reverse.py training)
    # instead of kg.num_ent/kg.num_rel which may differ (e.g. pykeen reports 351 relations
    # for DBpedia50 without inverse, but the sampled data uses 702 including inverse).
    _stats_path = os.path.join(args.data_root, args.dataname, str(args.reverse_edges_flag), 'stats.txt')
    with open(_stats_path, 'r') as _f:
        _lines = _f.readlines()
        nentity = int(_lines[0].split('\t')[-1])
        nrelation = int(_lines[1].split('\t')[-1])
    print(f'# stats.txt  nentity={nentity}  nrelation={nrelation}')
    print(f'# kg         num_ent={kg.num_ent}  num_rel={kg.num_rel}')

    adapter = CtrlHGenAdapter(
        checkpoint_path=args.checkpoint_path,
        special_tokens=special_tokens,
        offset=offset,
        nentity=nentity,
        nrelation=nrelation,
        is_gpt=("GPT2" in args.model_name),
        model_name=args.model_name,
        config_model=config_model,
        kg=kg,
        src_len=src_len,
        tgt_len=tgt_len,
    )

    print("CtrlHGen chat started. Type 'exit' to quit. Input '1' to reset memory for a new question.")

    current_entities = None
    current_entity_text = ""
    pending_condition_text = None
    asked_followup_questions = []

    while True:
        if current_entities is None:
            print('\nPlease enter entities, split them with ",", and press Enter to continue.')
            entity_text = input("entities: ").strip()

            if entity_text.lower() in {"exit", "quit"}:
                break
            if entity_text == "1":
                session.reset_context()
                current_entities = None
                current_entity_text = ""
                asked_followup_questions = []
                print("Context reset. You can ask a new question now.")
                continue

            observation_entities = _split_csv(entity_text)
            if not observation_entities:
                print("No valid entities were provided. Please try again.")
                continue

            current_entities = observation_entities
            current_entity_text = entity_text
        else:
            observation_entities = current_entities
            entity_text = current_entity_text
            print(f'\n[Using current entities] {", ".join(observation_entities)}')

        if pending_condition_text is not None:
            condition_text = pending_condition_text
            pending_condition_text = None
            print("\n[Auto condition from last follow-up answer]")
            print(f"conditions: {condition_text}")
        else:
            print('Please enter conditions (split with ","), or press Enter if there is no special request.')
            condition_text = input("conditions: ").strip()

        if condition_text.lower() in {"exit", "quit"}:
            break
        if condition_text == "1":
            session.reset_context()
            current_entities = None
            current_entity_text = ""
            asked_followup_questions = []
            print("Context reset. You can ask a new question now.")
            continue

        if not condition_text:
            condition_parsed = {
                "conditions": [{"type": "unconditional", "value": ""}],
                "condition_interpretation": {"note": "empty condition -> unconditional"},
            }
        else:
            session_memory = {
                "previous_parsed_control": session.last_parsed_control,
                "all_user_inputs": session.user_inputs,
            }
            condition_parsed = parser.parse_condition_text(
                condition_text=condition_text,
                session_memory=session_memory,
            )

        user_text = f"entities={entity_text}; conditions={condition_text}"
        turn = TurnRecord(user_text=user_text)

        parsed = _build_parsed_control(
            observation_entities=observation_entities,
            condition_parsed=condition_parsed,
            condition_text=condition_text,
        )

        if args.vs:
            print(f'condition_parsed: {condition_parsed}')
            print(f'parsed: {parsed}')
            print(f'condition_text: {condition_text}')

        print("\n[PARSED CONTROL]")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))

        turn.parsed_control = parsed

        repair_attempts = 0
        while True:
            try:
                model_input = adapter.build_model_input(parsed)
                break
            except (KeyError, ValueError, NotImplementedError) as exc:
                if not condition_text or repair_attempts >= 2:
                    raise

                repair_attempts += 1
                if args.vs:
                    print(f"\n[PARSE REPAIR] attempt={repair_attempts} error={exc}")

                condition_parsed = parser.repair_condition_text(
                    condition_text=condition_text,
                    session_memory=session_memory,
                    failed_parse=parsed,
                    error_message=str(exc),
                )
                parsed = _build_parsed_control(
                    observation_entities=observation_entities,
                    condition_parsed=condition_parsed,
                    condition_text=condition_text,
                )
                turn.parsed_control = parsed

                print("\n[PARSED CONTROL REPAIRED]")
                print(json.dumps(parsed, ensure_ascii=False, indent=2))

        # print("\n[MODEL INPUT]")
        # print(f'model_input: {model_input}')

        turn.model_input = model_input

        model_output = adapter.generate(
            model_input,
            temperature=args.temperature,
            top_k=args.test_top_k,
            constrained=args.constrained,
        )
        print("\n[MODEL INPUT]")
        print(f'model_input: {model_input}')
        print("\n[MODEL OUTPUT]")
        print(f'model_output: {model_output}')
        judge_result = verbalizer.judge_condition_match(
            condition_text=condition_text,
            parsed_conditions=parsed.get("conditions", []),
            query_tokens=model_output.get("query"),
            query_nl=model_output.get("query_nl") or "",
        )
        condition_match = bool(judge_result.get("match", False))
        print("\n[CONDITION JUDGE]")
        print(json.dumps(judge_result, ensure_ascii=False))
        overlap_triples = _collect_observation_intersection_triples(
            kg=kg,
            query_tokens=model_output.get("query"),
            observation_entity_ids=model_input.get("observation_entity_ids", []),
            split="test",
        )
        regen_round = 0
        while (not overlap_triples or not condition_match) and regen_round < max(0, int(args.regen_until_overlap_max_rounds)):
            regen_round += 1
            if not overlap_triples and not condition_match:
                print(f"\n[REGENERATE] no overlap + condition mismatch, retry round={regen_round}")
            elif not overlap_triples:
                print(f"\n[REGENERATE] no overlap triples, retry round={regen_round}")
            else:
                print(f"\n[REGENERATE] condition mismatch, retry round={regen_round}")
            model_output = adapter.generate(
                model_input,
                temperature=args.temperature,
                top_k=(args.test_top_k if int(args.test_top_k) > 0 else 50),
                constrained=args.constrained,
            )
            judge_result = verbalizer.judge_condition_match(
                condition_text=condition_text,
                parsed_conditions=parsed.get("conditions", []),
                query_tokens=model_output.get("query"),
                query_nl=model_output.get("query_nl") or "",
            )
            condition_match = bool(judge_result.get("match", False))
            print("\n[CONDITION JUDGE]")
            print(json.dumps(judge_result, ensure_ascii=False))
            overlap_triples = _collect_observation_intersection_triples(
                kg=kg,
                query_tokens=model_output.get("query"),
                observation_entity_ids=model_input.get("observation_entity_ids", []),
                split="test",
            )
        if (not overlap_triples or not condition_match) and fallback_query_tokens:
            print("\n[FALLBACK QUERY] generated query not usable, trying script-provided query.")
            overlap_triples = _collect_observation_intersection_triples(
                kg=kg,
                query_tokens=fallback_query_tokens,
                observation_entity_ids=model_input.get("observation_entity_ids", []),
                split="test",
            )
            if overlap_triples:
                model_output["query"] = fallback_query_tokens
                model_output["query_nl"] = f"[fallback_query] {fallback_query_tokens}"
                judge_result = verbalizer.judge_condition_match(
                    condition_text=condition_text,
                    parsed_conditions=parsed.get("conditions", []),
                    query_tokens=model_output.get("query"),
                    query_nl=model_output.get("query_nl") or "",
                )
                condition_match = bool(judge_result.get("match", False))
                print("\n[CONDITION JUDGE]")
                print(json.dumps(judge_result, ensure_ascii=False))
        # print("\n[MODEL OUTPUT]")
        # print(f'model_output: {model_output}')
        if (not overlap_triples) or (not condition_match):
            print(
                "\n[REGEN FAILED] Query failed validation after retries "
                "(needs overlap triples and condition match). "
                "Please refine condition and try again."
            )
            continue

        print("\n[OBSERVATION-INTERSECTION TRIPLES]")
        for triple in overlap_triples:
            print(
                f"- ({triple['head_name']}, {triple['relation_name']}, {triple['tail_name']}) "
                f"[intersection_entity_id={triple['intersection_entity_id']}]"
            )

        followup_question = verbalizer.propose_followup_question(
            observation_entities=observation_entities,
            query_nl=model_output.get("query_nl") or "",
            triples=overlap_triples,
            previous_questions=asked_followup_questions,
        )
        asked_followup_questions.append(followup_question)
        print("\n[FOLLOW-UP QUESTION]")
        print(followup_question)
        followup_answer = input("your answer (used as next condition, Enter to skip): ").strip()
        if not followup_answer:
            followup_answer_retry = input(
                "Empty answer detected. Press Enter again to skip, or type your answer: "
            ).strip()
            if followup_answer_retry:
                followup_answer = followup_answer_retry
        if args.vs:
            print(f"[FOLLOW-UP RAW ANSWER] {repr(followup_answer)}")
        if followup_answer.lower() in {"exit", "quit"}:
            break
        if followup_answer == "1":
            session.reset_context()
            current_entities = None
            current_entity_text = ""
            asked_followup_questions = []
            print("Context reset. You can ask a new question now.")
            continue
        if followup_answer:
            derived_condition = _derive_condition_from_followup(
                question=followup_question,
                answer=followup_answer,
            )
            pending_condition_text = derived_condition or followup_answer
            if derived_condition:
                print(f"[AUTO-DERIVED CONDITION] {pending_condition_text}")
        else:
            print("[FOLLOW-UP SKIPPED] Empty answer, condition remains unchanged.")

        turn.model_output = model_output

        session.add_turn(turn)
        session.update_memory(parsed, model_output)

        if args.vs:
            print(f'turn: {turn}')
            print(f'session: {session}')

        print("If your current question has been fully answered and you want to ask a new one, please enter 1.")


if __name__ == "__main__":
    main()