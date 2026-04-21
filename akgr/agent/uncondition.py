"""
Unconditional hypothesis generation + analysis.

Flow:
  1. Generate hypothesis with NO conditions (unconditional), only observation entities.
  2. Execute on KG, compute metrics against label answers.
  3. Split hypothesis at top-level i/u, validate each sub-query.
  4. Incoming edge intersection on observations -> structural hints.
  5. LLM synthesizes all results and proposes `num_suggestions` different condition sets.
"""
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
    "answers_nl": ["rpgrip1l", "pdx1", "pfkfb1", "gys1", "recql4", "prpf6", "fxn", "ltk", "gyg1", "kcnh2", "ski", "flt4"],
}
case2 = {
    "answers": [5056, 5057, 5058, 5061, 5062, 5063, 5053, 5055],
    "answers_nl": ["pex13", "pex14", "pex16", "pex3", "pex5", "pex6", "pex10", "pex12"],
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


def run_uncondition(adapter, llm_model, case: dict, num_suggestions: int = 3) -> dict:
    kg = adapter.kg
    answer_nl = case["answers_nl"]
    label_answers = case["answers"]
    rel_names = list(adapter.mapper.rel_name2id.keys())
    ent_sample = list(adapter.mapper.ent_name2id.keys())[:50]

    # Step 1: Unconditional generation
    print("=" * 60)
    print("  UNCONDITIONAL GENERATION")
    print("=" * 60)
    fmt_result = format_conversion_tool(
        adapter=adapter,
        answer_nl=answer_nl,
        conditions=[{"type": "unconditional", "value": ""}],
    )
    source_text = fmt_result["model_input"]["source_text"]
    gen_result = generate_hypothesis_tool(adapter, source_text)
    raw_output = gen_result["raw_output"]
    hypothesis_nl = gen_result.get("query_nl", "N/A")
    print(f"Hypothesis (raw): {raw_output}")
    print(f"Hypothesis (NL):  {hypothesis_nl}")

    # Step 2: Metrics
    metrics = compute_metrics(
        raw_output=raw_output,
        label_answers=label_answers,
        graph_samplers=kg.graph_samplers,
        searching_split="train",
    )
    jaccard = metrics["jaccard"]
    print(f"Jaccard: {jaccard:.4f}, Pred: {len(metrics['pred_answers'])}, Label: {len(metrics['label_answers'])}")

    # Step 3: Sub-query analysis
    pred_qry = qry_actionstr_2_wordlist(raw_output)
    sub_query_info = []
    if pred_qry:
        graph_validation_tool(pred_qry, kg.graph_samplers["train"], label_answers)
        for sq in _split_top_level(pred_qry):
            try:
                sq_ans = kg.graph_samplers["train"].search_answers_to_query(sq)
                sq_set = set(sq_ans)
                label_set = set(label_answers)
                sq_nl = query_to_natural_language(sq, kg.ent_id2name, kg.rel_id2name)
                sub_query_info.append({
                    "sub_query_nl": sq_nl,
                    "answer_count": len(sq_set),
                    "relation_to_label": _set_relation_str(sq_set, label_set),
                    "overlap_count": len(sq_set & label_set),
                })
            except Exception as e:
                sub_query_info.append({"error": str(e)})

    # Step 4: Incoming edge intersection
    incoming_hints = incoming_edge_intersection_tool(
        answer_entity_ids=label_answers,
        graph_sampler=kg.graph_samplers["train"],
        ent_id2name=kg.ent_id2name,
        rel_id2name=kg.rel_id2name,
        top_k=10,
    )
    print(f"Incoming intersection: {incoming_hints['intersection_count']} common heads")

    # Step 5: LLM proposes conditions
    sub_query_text = json.dumps(sub_query_info, indent=2) if sub_query_info else "No sub-queries."
    hints_text = json.dumps(incoming_hints["hints"][:5], indent=2) if incoming_hints.get("hints") else "No hints."

    prompt = (
        f"You are an expert in knowledge graph abductive reasoning.\n"
        f"An unconditional hypothesis was generated. Based on the analysis, "
        f"propose {num_suggestions} DIFFERENT condition sets to guide better hypothesis generation.\n\n"
        f"## Observations\n{', '.join(answer_nl)}\n\n"
        f"## Unconditional hypothesis\n"
        f"- NL: {hypothesis_nl}\n- Raw: {raw_output}\n"
        f"- Jaccard: {jaccard:.4f}, Pred: {len(metrics['pred_answers'])}, Label: {len(metrics['label_answers'])}\n\n"
        f"## Sub-query analysis\n{sub_query_text}\n\n"
        f"## Incoming edge hints\n{hints_text}\n\n"
        f"## Available relations\n{', '.join(rel_names)}\n\n"
        f"## Sample entities\n{', '.join(ent_sample)}\n\n"
        f"## Condition types\n"
        f"- pattern: e.g. 'i p e p e', 'i n p p e p e'\n"
        f"- relation: specific relation name\n"
        f"- entity: specific entity name\n"
        f"- relationnumber: e.g. '2', '3'\n"
        f"- entitynumber: e.g. '2', '3'\n\n"
        f"Propose {num_suggestions} strategies with different focuses "
        f"(e.g. pattern-focused, relation+entity, relation+count).\n\n"
        f"Return ONLY a JSON array of {num_suggestions} objects:\n"
        f'  [{{"analysis": "...", "condition": "I want a hypothesis that..."}}]\n'
        f"Return nothing else."
    )

    response = llm_model([{"role": "user", "content": prompt}], stop_sequences=None)
    result_text = response.content.strip()
    if result_text.startswith("```"):
        result_text = result_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        suggestions = json.loads(result_text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\[.*\]', result_text, re.DOTALL)
        suggestions = json.loads(match.group()) if match else [{"analysis": "Parse failed", "condition": "unconditional"}]

    print(f"\n{'='*60}\n  SUGGESTED CONDITIONS ({len(suggestions)})\n{'='*60}")
    for i, s in enumerate(suggestions):
        print(f"\n  [{i+1}] {s.get('condition', 'N/A')}")
        print(f"      {s.get('analysis', 'N/A')}")

    return {
        "hypothesis_raw": raw_output,
        "hypothesis_nl": hypothesis_nl,
        "metrics": metrics,
        "sub_query_analysis": sub_query_info,
        "incoming_hints": incoming_hints,
        "suggested_conditions": suggestions,
    }


if __name__ == "__main__":
    hypothesis_model_path = '/home/gaoyisen/akgr-agent/checkpoints/PharmKG8k-full-32-130-multi.pth'
    data_root = '/home/gaoyisen/akgr-agent/data/'
    dataname = 'PharmKG8k'

    from akgr.utils.load_util import load_yaml
    api_cfg = load_yaml('akgr/configs/api_keys.yml')['deepinfra']
    llm_model = OpenAIServerModel(
        model_id=api_cfg['model_id'],
        api_base=api_cfg['api_base'],
        api_key=api_cfg['api_key'],
    )
    adapter = build_adapter(hypothesis_model_path, data_root, dataname)
    run_uncondition(adapter=adapter, llm_model=llm_model, case=case1, num_suggestions=3)
