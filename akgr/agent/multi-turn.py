import json
from smolagents import OpenAIServerModel
from akgr.agent.tools import format_conversion_tool, generate_hypothesis_tool, compute_metrics
from akgr.agent.loop import best_round_by_conditions, parse_conditions_from_question, run_loop
from akgr.agent.single import build_adapter


_PATTERN_HINT = (
    "  1p: p e | 2p: p p e | 2i: i p e p e | 3i: i i p e p e p e | "
    "ip: p i p e p e | pi: i p e p p e | 2u: u p e p e | "
    "up: p u p e p e | 2in: i n p e p e | 3in: i i n p e p e p e | "
    "inp: p i n p e p e | pni: i n p p e p e | pin: i n p e p p e"
)


def _generate_conditions_from_history(
    llm_model, history: list[dict], user_question: str, hypothesis_history: bool = False
) -> list[dict]:
    """Generate a condition list based on user question and history."""
    from akgr.agent.judge import _parse_conditions

    if hypothesis_history:
        # Also expose each previous turn's kept hypothesis, so the model can
        # reuse a structure that already explained the observations well.
        lines = []
        for h in history:
            rb = h.get("round_best") or h
            j = rb.get("jaccard")
            lines.append(
                f"  Turn {h['turn_id']}: question='{h['user_question']}', "
                f"condition='{h['condition']}', "
                f"best_hypothesis='{rb.get('hypothesis_nl')}', "
                f"jaccard={0.0 if j is None else j:.4f}"
            )
        history_text = "\n".join(lines) or "No previous turns."
        history_note = (
            "\n## How to use the history\n"
            "The best hypothesis kept in each previous turn is shown above for reference only. "
            "The user's requirements change from turn to turn, so the current request overrides "
            "the history whenever they conflict. Reuse a previous hypothesis' structure only "
            "where it is still consistent with the current request.\n"
        )
    else:
        history_text = "\n".join(
            f"  Turn {h['turn_id']}: question='{h['user_question']}', "
            f"condition='{h['condition']}', jaccard={h['jaccard']:.4f}"
            for h in history
        ) or "No previous turns."
        history_note = ""

    prompt = (
        f"You help generate structured conditions for a KG hypothesis generation task.\n\n"
        f"## Previous turns\n{history_text}\n"
        f"{history_note}\n"
        f"## User's current request\n{user_question}\n\n"
        f"## Available patterns\n{_PATTERN_HINT}\n\n"
        f"## Condition types\n"
        f"  relation: one relation name | entity: one entity name\n"
        f"  relationnumber: int (1-3) | entitynumber: int (1-3)\n"
        f"  pattern: a specific query pattern from the available patterns \n"
        f"## Constraints\n"
        f"- relationnumber and entitynumber values must be integers between 1 and 3 (inclusive).\n"
        f"- At most ONE 'relation' condition and at most ONE 'entity' condition per condition list.\n"
        f"  Do NOT include multiple relation or multiple entity conditions in the same list.\n"
        f"- Every condition's 'value' must be non-empty. Do NOT produce conditions with empty string values.\n"
        f"  If the request is vague, pick a reasonable concrete value (e.g. a pattern or a number).\n\n"
        f"Task: Convert the user's current request into a single condition list.\n"
        f"Use the history to understand context, but focus on the current request.\n\n"
        f"## Output format (STRICT)\n"
        f"You MUST return ONLY a raw JSON array of condition dicts. No explanation, no markdown, no extra text.\n"
        f"- Exactly 1 array of condition dicts with keys 'type' and 'value'\n"
        f"- Valid types: relation, entity, relationnumber, entitynumber, pattern\n"
        f"- Do NOT emit one array per condition; do NOT output several arrays.\n"
        f"- Do NOT use flat [\"type\", \"value\"] pairs.\n\n"
        f"CORRECT example:\n"
        f'[{{"type":"relation","value":"treats"}},{{"type":"entitynumber","value":"2"}}]\n\n'
        f"INCORRECT (do NOT do this):\n"
        f'["relation", "treats"]\\n["entitynumber", "2"]\n\n'
        f"WRONG (do NOT do this): any prose, markdown code fences, nested arrays."
    )
    response = llm_model([{"role": "user", "content": prompt}], stop_sequences=None)
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Despite the instructions above, the model sometimes emits one array per
        # condition on consecutive lines. Collect every array it produced instead
        # of keeping only the first, which would silently drop conditions. Each
        # array is normalised on its own, since a flat ["type", "value"] pair and
        # a dict list must not be concatenated into one ambiguous sequence.
        decoder, collected, pos = json.JSONDecoder(), [], 0
        while pos < len(text):
            start = text.find("[", pos)
            if start == -1:
                break
            try:
                obj, pos = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                break
            if isinstance(obj, list):
                collected.extend(_parse_conditions(obj))
            elif isinstance(obj, dict):
                collected.append(obj)
        if not collected:
            raise
        result = collected
    return result if isinstance(result, list) else [result]
def run_multiturn(
    adapter,
    llm_model,
    case: dict,
    analysis: bool = False,
    jaccard_threshold: float = 0.95,
    verbose: bool = True,
    hypothesis_history: bool = False,
):
    kg = adapter.kg
    answer_nl = case["answers_nl"]
    label_answers = case["answers"]
    turns = case["turns"]
    history: list[dict] = []

    for turn in turns:
        turn_id = turn["turn_id"]

        followup = turn["followup_question"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"  TURN {turn_id} / {len(turns)}  |  {followup}")
            print(f"{'='*60}\n")

        # Generate 5 candidate condition-lists
        conditions = _generate_conditions_from_history(
            llm_model, history, followup, hypothesis_history=hypothesis_history
        )
        if verbose:
            print(f"  [Conditions] {conditions}")

        try:
            fmt = format_conversion_tool(adapter=adapter, answer_nl=answer_nl, conditions=conditions)
            gen = generate_hypothesis_tool(adapter, fmt["model_input"]["source_text"])
            m = compute_metrics(raw_output=gen["raw_output"], label_answers=label_answers, graph_samplers=kg.graph_samplers)
            raw_output, hypothesis_nl, metrics, jaccard = gen["raw_output"], gen.get("query_nl", "N/A"), m, m["jaccard"]
            candidates = [{"conditions": conditions, "hypothesis_raw": raw_output, "hypothesis_nl": hypothesis_nl, "jaccard": jaccard}]
        except Exception as e:
            if verbose:
                print(f"  [Error] {e}")
            raw_output = hypothesis_nl = metrics = None
            jaccard = 0.0
            candidates = []

        if verbose:
            print(f"[Result] {raw_output}  Jaccard={jaccard:.4f}")

        turn_result = {
            "turn_id": turn_id,
            "user_question": followup,
            "condition": followup,
            "parsed_conditions": conditions,
            "hypothesis_raw": raw_output,
            "hypothesis_nl": hypothesis_nl,
            "jaccard": jaccard,
            "dice": metrics["dice"] if metrics else None,
            "overlap": metrics["overlap"] if metrics else None,
            "pred_answer_count": len(metrics["pred_answers"]) if metrics else 0,
            "label_answer_count": len(label_answers),
            "candidates": candidates,
            "round_best": {
                "hypothesis_raw": raw_output,
                "hypothesis_nl": hypothesis_nl,
                "jaccard": jaccard,
                "dice": metrics["dice"] if metrics else None,
                "overlap": metrics["overlap"] if metrics else None,
            },
        }

        if analysis and jaccard < jaccard_threshold:
            try:
                loop_case = {
                    "followup_question": followup,
                    "answers_nl": answer_nl,
                    "answers": label_answers,
                }
                loop_history = run_loop(
                    adapter=adapter, llm_model=llm_model, case=loop_case,
                    max_rounds=2, jaccard_threshold=jaccard_threshold, verbose=verbose,
                    initial_conditions=conditions,
                )
                conds = next(
                    (h.get("target_conditions") for h in loop_history if h.get("target_conditions")),
                    conditions,
                )
                best_round = best_round_by_conditions(loop_history, conds)
                best = best_round.get("round_best") or best_round
                turn_result["round_best"] = {
                    "hypothesis_raw": best.get("hypothesis_raw"),
                    "hypothesis_nl": best.get("hypothesis_nl"),
                    "jaccard": best.get("jaccard"),
                    "dice": best.get("dice"),
                    "overlap": best.get("overlap"),
                }
            except Exception as e:
                if verbose:
                    print(f"  [Analysis error, using direct result] {e}")

        history.append(turn_result)

    return history



def _save_result(log_path, case, history):
    record = {
        "answers": case["answers"],
        "answers_nl": case.get("answers_nl", []),
        "turn_count": case.get("turn_count"),
        "turns": [
            {
                "turn_id": h["turn_id"],
                "condition": h["condition"],
                "parsed_conditions": h.get("parsed_conditions", []),
                "round_best": h.get("round_best", {}),
            }
            for h in history
        ],
    }
    compact_keys = ("answers", "answers_nl")
    placeholders = {}
    for k in compact_keys:
        if record.get(k) is not None:
            ph = f'"__PH_{k}__"'
            placeholders[ph] = json.dumps(record[k], ensure_ascii=False)
            record[k] = f"__PH_{k}__"
    text = json.dumps(record, ensure_ascii=False, indent=2)
    for ph, val in placeholders.items():
        text = text.replace(ph, val)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    import argparse, os
    from akgr.agent.case import case_3turn,case_complex

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["case", "run"], default="case")
    parser.add_argument("--dataname", default="BioKG")
    parser.add_argument("--checkpoint", default="checkpoints/BioKG-full-32-multi.pth")
    parser.add_argument("--data_root", default="./data/")
    parser.add_argument("--analysis", action="store_true")
    parser.add_argument(
        "--hypothesis_history",
        action="store_true",
        help="Show each previous turn's kept hypothesis when deriving this turn's conditions.",
    )

    parser.add_argument("--jaccard_threshold", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    from akgr.utils.load_util import load_yaml
    api_cfg = load_yaml("akgr/configs/api_keys.yml")["deepinfra"]
    llm_model = OpenAIServerModel(
        model_id=api_cfg["model_id"],
        api_base=api_cfg["api_base"],
        api_key=api_cfg["api_key"],
        timeout=60,
    )
    adapter = build_adapter(args.checkpoint, args.data_root, args.dataname)

    if args.mode == "case":
        log_dir = os.path.join("log", args.dataname)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "multiturn.jsonl")
        history = run_multiturn(
            adapter=adapter, llm_model=llm_model, case=case_complex,
            analysis=args.analysis,
            jaccard_threshold=args.jaccard_threshold,
            hypothesis_history=args.hypothesis_history,
        )
        _save_result(log_path, case_complex, history)

    else:
        from tqdm import tqdm
        data_file = os.path.join(args.data_root, args.dataname, "3-multiturn.jsonl")
        log_dir = os.path.join("log", args.dataname)
        os.makedirs(log_dir, exist_ok=True)
        model_tag = api_cfg["model_id"].split("/")[-1]
        suffix = "_analysis" if args.analysis else ""
        suffix += "_hyphist" if args.hypothesis_history else ""
        log_path = os.path.join(log_dir, f"multiturn_{model_tag}{suffix}.jsonl")

        with open(data_file, encoding="utf-8") as f:
            content = f.read()
        decoder = json.JSONDecoder()
        cases, pos = [], 0
        while pos < len(content):
            while pos < len(content) and content[pos].isspace():
                pos += 1
            if pos >= len(content):
                break
            obj, pos = decoder.raw_decode(content, pos)
            cases.append(obj)

        for case in tqdm(cases[:args.limit], desc=args.dataname):
            try:
                history = run_multiturn(
                    adapter=adapter, llm_model=llm_model, case=case,
                    analysis=args.analysis, jaccard_threshold=args.jaccard_threshold,
                    verbose=False, hypothesis_history=args.hypothesis_history,
                )
                _save_result(log_path, case, history)
            except Exception as e:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"error": str(e), "answers": case.get("answers")}, ensure_ascii=False) + "\n")
