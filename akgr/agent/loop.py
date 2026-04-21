import json
from smolagents import OpenAIServerModel
from akgr.agent.tools import (
    format_conversion_tool,
    generate_hypothesis_tool,
    compute_metrics,
    graph_validation_tool,
    incoming_edge_intersection_tool,
    _split_top_level,
)
from akgr.agent.getsomesampleFromDB import query_to_natural_language
from akgr.utils.parsing_util import qry_actionstr_2_wordlist

case1 = {
    "answers": [5828, 5001, 5066, 2941, 5679, 5456, 2578, 3891, 2937, 3546, 6077, 2463],
    "query": "i n -8 -21 1128 -8 4922",
    "pattern_str": "(i,(n,(p,(p,(e)))),(p,(e)))",
    "query_nl": "Entities that do not have a 'GG' link to an entity that has a 'Rg' link to cdh1, and have a 'GG' link to pask",
    "answers_nl": ["rpgrip1l", "pdx1", "pfkfb1", "gys1", "recql4", "prpf6", "fxn", "ltk", "gyg1", "kcnh2", "ski", "flt4"],
    "intention_mode": "two-condition",
    "followup_condition_values": {
        "pattern": "i n p p e p e",
        "entitynumber": "2e",
        "relationnumber": "3p",
        "entity": "cdh1",
        "entity_id": "1128",
        "relation": "E",
        "relation_id": "-8"
    },
    "followup_question": "I want a hypothesis that includes the relation \"GG\" and contains 2 entities."
}
case2 = {
    "answers": [5056, 5057, 5058, 5061, 5062, 5063, 5053, 5055],
    "query": "i i -8 33 -3 5059 -3 5059",
    "pattern_str": "(i,(i,(p,(e)),(p,(e))),(p,(e)))",
    "query_nl": "Entities that have a 'GG' link to abcd1, and have a 'B' link to pex19, and have a 'B' link to pex19",
    "answers_nl": ["pex13", "pex14", "pex16", "pex3", "pex5", "pex6", "pex10", "pex12"],
    "intention_mode": "two-condition",
    "followup_condition_values": {
        "pattern": "i i p e p e p e",
        "entitynumber": "3e",
        "relationnumber": "3p",
        "entity": "abcd1",
        "entity_id": "33",
        "relation": "E",
        "relation_id": "-8"
    },
    "followup_question": "I want a hypothesis that follows the pattern \"i i p e p e p e\" and has 3 relations."
}


def build_adapter(hypothesis_model_path: str, data_root: str, dataname: str):
    from akgr.utils.load_util import load_yaml
    from akgr.kgdata import load_kg
    from akgr.agent.ctrlhgen_adapter import CtrlHGenAdapter

    config_dataloader = load_yaml('akgr/configs/config-dataloader.yml')
    config_model = load_yaml('akgr/configs/config-model.yml')
    modelname = 'GPT2_6_act_nt'
    is_gpt = 'GPT2' in modelname
    is_act = 'act' in modelname
    tgt_len = config_dataloader['act_len'] + 1 if is_act else config_dataloader['qry_len'] + 1
    src_len = config_dataloader['ans_len'] + 1

    kg = load_kg(data_root, dataname, reverse_edges_flag=False)

    return CtrlHGenAdapter(
        checkpoint_path=hypothesis_model_path,
        special_tokens=config_dataloader['special_tokens'],
        offset=config_dataloader['offset'],
        nentity=len(kg.ent_id2name),
        nrelation=len(kg.rel_id2name),
        is_gpt=is_gpt,
        model_name=modelname,
        config_model=config_model,
        kg=kg,
        src_len=src_len,
        tgt_len=tgt_len,
    )


def parse_conditions_from_question(llm_model, followup_question: str) -> list[dict]:
    """Use LLM to parse a natural language followup question into structured conditions."""
    prompt = (
        f"Parse the following question into a JSON array of condition dicts.\n"
        f"Valid condition types: unconditional, pattern, relation, entity, relationnumber, entitynumber.\n"
        f"Examples:\n"
        f'  "I want a hypothesis with relation GG" -> [{{"type":"relation","value":"GG"}}]\n'
        f'  "I want pattern i p e p e with 2 entities" -> [{{"type":"pattern","value":"i p e p e"}},{{"type":"entitynumber","value":"2"}}]\n'
        f"\nQuestion: {followup_question}\n\n"
        f"Return ONLY the JSON array, nothing else."
    )
    response = llm_model(
        [{"role": "user", "content": prompt}],
        stop_sequences=None,
    )
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _set_relation_str(pred_set: set, label_set: set) -> str:
    if not pred_set and not label_set:
        return "both_empty"
    if not pred_set:
        return "result_empty"
    if not label_set:
        return "label_empty"
    if pred_set == label_set:
        return "exact_match"
    if pred_set >= label_set:
        return "result_contains_label"
    if pred_set <= label_set:
        return "label_contains_result"
    if pred_set & label_set:
        return "partial_overlap"
    return "disjoint"


def run_loop(
    adapter,
    llm_model,
    case: dict,
    max_rounds: int = 3,
    jaccard_threshold: float = 0.8,
):
    """
    Multi-round hypothesis generation loop.

    Flow per round:
      1. Parse followup question -> conditions
      2. Build model input (observation + conditions, shifted) -> generate hypothesis -> unshift
      3. Execute unshifted hypothesis on KG -> pred_ans (raw), compare with label_answers (raw)
      4. If jaccard >= threshold or round >= max_rounds -> stop
      5. Otherwise:
         a. Split hypothesis into sub-queries at top-level i/u, validate each
         b. Incoming edge intersection on observations -> hints
         c. Summary agent analyzes results, keeps original conditions, adds new ones
      6. Next round with new conditions
    """
    kg = adapter.kg
    answer_nl = case["answers_nl"]
    label_answers = case["answers"]  # raw 0-indexed
    original_followup = case["followup_question"]
    current_followup = original_followup

    # Available info for the analysis agent
    rel_names = list(adapter.mapper.rel_name2id.keys())
    ent_sample = list(adapter.mapper.ent_name2id.keys())[:50]

    history: list[dict] = []

    for round_idx in range(1, max_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_idx} / {max_rounds}")
        print(f"  Condition: {current_followup}")
        print(f"{'='*60}\n")

        # ----- Step 1: Parse conditions from question -----
        conditions = parse_conditions_from_question(llm_model, current_followup)
        print(f"[Step 1] Parsed conditions: {conditions}")

        # ----- Step 2: Format conversion (observation + conditions -> source_text) -----
        fmt_result = format_conversion_tool(
            adapter=adapter,
            answer_nl=answer_nl,
            conditions=conditions,
        )
        source_text = fmt_result["model_input"]["source_text"]
        print(f"[Step 2] Source text: {source_text}")

        # ----- Step 3: Generate hypothesis -----
        gen_result = generate_hypothesis_tool(adapter, source_text)
        raw_output = gen_result["raw_output"]          # unshifted action string
        hypothesis_nl = gen_result.get("query_nl", "N/A")
        print(f"[Step 3] Hypothesis (raw): {raw_output}")
        print(f"[Step 3] Hypothesis (NL):  {hypothesis_nl}")

        # ----- Step 4: Compute metrics (both pred_ans and label_answers are raw) -----
        metrics = compute_metrics(
            raw_output=raw_output,
            label_answers=label_answers,
            graph_samplers=kg.graph_samplers,
            searching_split="train",
        )
        jaccard = metrics["jaccard"]
        print(f"[Step 4] Jaccard: {jaccard:.4f}, Dice: {metrics['dice']:.4f}")
        print(f"         Pred count: {len(metrics['pred_answers'])}, Label count: {len(metrics['label_answers'])}")

        round_result = {
            "round": round_idx,
            "condition": current_followup,
            "parsed_conditions": conditions,
            "hypothesis_raw": raw_output,
            "hypothesis_nl": hypothesis_nl,
            "jaccard": jaccard,
            "dice": metrics["dice"],
            "pred_answer_count": len(metrics["pred_answers"]),
            "label_answer_count": len(metrics["label_answers"]),
            "metrics": metrics,
        }
        history.append(round_result)

        # ----- Check stopping criteria -----
        if jaccard >= jaccard_threshold:
            print(f"\n*** Jaccard {jaccard:.4f} >= {jaccard_threshold}. Stopping. ***")
            break

        if round_idx == max_rounds:
            print(f"\n*** Reached max rounds ({max_rounds}). Stopping. ***")
            break

        # ----- Step 5: Analysis for next round -----
        print(f"\n--- Analysis Phase ---")

        # 5a. Split hypothesis into sub-queries and validate each
        pred_qry = qry_actionstr_2_wordlist(raw_output)
        sub_query_info = []
        if pred_qry:
            # Validate the whole query
            whole_result = graph_validation_tool(pred_qry, kg.graph_samplers["train"], label_answers)
            print(f"[5a] Whole query: {whole_result['answer_count']} answers, "
                  f"relation={whole_result.get('relation_to_label', 'N/A')}")

            # Validate sub-queries (only top-level i/u split)
            for sq in _split_top_level(pred_qry):
                try:
                    sq_ans = kg.graph_samplers["train"].search_answers_to_query(sq)
                    sq_set = set(sq_ans)
                    label_set = set(label_answers)
                    sq_nl = query_to_natural_language(sq, kg.ent_id2name, kg.rel_id2name)
                    info = {
                        "sub_query_nl": sq_nl,
                        "answer_count": len(sq_set),
                        "relation_to_label": _set_relation_str(sq_set, label_set),
                        "overlap_count": len(sq_set & label_set),
                    }
                    sub_query_info.append(info)
                    print(f"[5a] Sub-query: {sq_nl}")
                    print(f"     {info['answer_count']} answers, relation={info['relation_to_label']}, overlap={info['overlap_count']}")
                except Exception as e:
                    sub_query_info.append({"error": str(e)})

        # 5b. Incoming edge intersection on observation entities
        incoming_hints = incoming_edge_intersection_tool(
            answer_entity_ids=label_answers,
            graph_sampler=kg.graph_samplers["train"],
            ent_id2name=kg.ent_id2name,
            rel_id2name=kg.rel_id2name,
            top_k=10,
        )
        print(f"[5b] Incoming edge intersection: {incoming_hints['intersection_count']} common heads")
        for hint in incoming_hints.get("hints", [])[:5]:
            print(f"     Head: {hint['head_entity']}, Relations: {[r['name'] for r in hint['relations']]}")

        # 5c. Build history summary
        history_lines = []
        for entry in history:
            history_lines.append(
                f"  Round {entry['round']}:\n"
                f"    Condition: {entry['condition']}\n"
                f"    Hypothesis (NL): {entry['hypothesis_nl']}\n"
                f"    Hypothesis (raw): {entry['hypothesis_raw']}\n"
                f"    Jaccard: {entry['jaccard']:.4f}\n"
                f"    Pred answers: {entry['pred_answer_count']}, Label answers: {entry['label_answer_count']}"
            )
        history_text = "\n".join(history_lines)

        # 5d. Summary analysis agent: propose new conditions
        sub_query_text = json.dumps(sub_query_info, indent=2) if sub_query_info else "No sub-queries (not an i/u query)."
        hints_text = json.dumps(incoming_hints["hints"][:5], indent=2) if incoming_hints.get("hints") else "No common heads found."

        analysis_prompt = (
            f"You are an expert in knowledge graph abductive reasoning.\n"
            f"Analyze the results and propose improved conditions for the next hypothesis generation round.\n\n"
            f"## Observation entities\n{', '.join(answer_nl)}\n\n"
            f"## User's original question (MUST keep these conditions)\n{original_followup}\n\n"
            f"## History\n{history_text}\n\n"
            f"## Sub-query analysis of latest hypothesis\n{sub_query_text}\n\n"
            f"## Incoming edge hints (common incoming neighbors of observation entities)\n{hints_text}\n\n"
            f"## Available relations in KG\n{', '.join(rel_names)}\n\n"
            f"## Sample entities in KG\n{', '.join(ent_sample)}\n\n"
            f"## Available condition types\n"
            f"- pattern: structural pattern like 'p e', 'i p e p e', 'i n p p e p e'\n"
            f"- relation: a specific relation name from the KG\n"
            f"- entity: a specific entity name from the KG\n"
            f"- relationnumber: number of relations, e.g. '2' or '3'\n"
            f"- entitynumber: number of entities, e.g. '2' or '3'\n\n"
            f"## Instructions\n"
            f"1. You MUST keep the user's original conditions from the original question.\n"
            f"2. ADD new conditions based on the sub-query analysis and incoming edge hints.\n"
            f"   - If pred count is too large, add more specific constraints (entity, relation).\n"
            f"   - If pred count is too small, try a different pattern.\n"
            f"   - Use the incoming edge hints to discover useful relations/entities.\n"
            f"3. Return ONLY a JSON object:\n"
            f'   {{"analysis": "brief analysis (1-3 sentences)", '
            f'"new_condition": "the full new followup question string"}}\n'
            f"Return nothing else."
        )

        response = llm_model(
            [{"role": "user", "content": analysis_prompt}],
            stop_sequences=[],
        )
        result_text = response.content.strip()
        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            reflection = json.loads(result_text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
            if match:
                reflection = json.loads(match.group())
            else:
                reflection = {"analysis": "Parse failed", "new_condition": original_followup}

        analysis = reflection.get("analysis", "")
        new_condition = reflection.get("new_condition", original_followup)
        print(f"\n[Analysis] {analysis}")
        print(f"[New condition] {new_condition}")

        current_followup = new_condition

    # ----- Final summary -----
    print(f"\n{'='*60}")
    print("  LOOP SUMMARY")
    print(f"{'='*60}")
    best = max(history, key=lambda h: h["jaccard"])
    for entry in history:
        marker = " <-- BEST" if entry is best else ""
        print(
            f"  Round {entry['round']}: "
            f"Jaccard={entry['jaccard']:.4f}, "
            f"Pred={entry['pred_answer_count']}, "
            f"Condition='{entry['condition'][:80]}...'{marker}"
        )
    print(f"\n  Best Jaccard: {best['jaccard']:.4f} (Round {best['round']})")
    print(f"  Best hypothesis (NL): {best['hypothesis_nl']}")
    print(f"  Best hypothesis (raw): {best['hypothesis_raw']}")

    return history


if __name__ == "__main__":
    hypothesis_model_path = '/home/gaoyisen/akgr-agent/checkpoints/PharmKG8k-full-32-130-multi.pth'
    data_root = '/home/gaoyisen/akgr-agent/data/'
    dataname = 'PharmKG8k'
    case = case1

    from akgr.utils.load_util import load_yaml
    api_cfg = load_yaml('akgr/configs/api_keys.yml')['deepinfra']
    llm_model = OpenAIServerModel(
        model_id=api_cfg['model_id'],
        api_base=api_cfg['api_base'],
        api_key=api_cfg['api_key'],
    )

    adapter = build_adapter(hypothesis_model_path, data_root, dataname)

    history = run_loop(
        adapter=adapter,
        llm_model=llm_model,
        case=case,
        max_rounds=3,
        jaccard_threshold=0.8,
    )
