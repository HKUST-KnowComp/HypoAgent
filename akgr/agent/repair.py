"""
KG-guided hypothesis repair module.

run_repair() can be called:
  - after uncondition.run_uncondition() to repair the initial unconditional hypothesis
  - after each round of loop.run_loop() to repair the best hypothesis found
  - standalone with any (hypothesis, observation) pair

Repair actions (from prompt.txt):
  ADD_CONSTRAINT, REMOVE_CONSTRAINT, ADD_UNION_BRANCH, EXTEND_PATH, SHORTEN_PATH,
  ADD_ENTITY, REMOVE_ENTITY, GENERALIZE_RELATION, SPECIFY_RELATION,
  REPLACE_RELATION, REPLACE_ENTITY, REPLACE_PREDICATE, REPLACE_PATH_SEGMENT
"""
import json
from smolagents import OpenAIServerModel

from akgr.agent.tools import (
    execute_and_diagnose_tool,
    neighborhood_candidates_tool,
    format_conversion_tool,
    generate_hypothesis_tool,
    compute_metrics,
    graph_validation_tool,
    incoming_edge_intersection_tool,
    _split_top_level,
)
from akgr.agent.getsomesampleFromDB import query_to_natural_language
from akgr.utils.parsing_util import qry_actionstr_2_wordlist


# ---------------------------------------------------------------------------
# Repair scoring
# ---------------------------------------------------------------------------

def _score_hypothesis(raw_output: str, observation_ids: list[int], graph_sampler, alpha: float = 0.05) -> float:
    """F1 - alpha * complexity (number of tokens as proxy)."""
    from akgr.utils.parsing_util import qry_actionstr_2_wordlist
    try:
        pred_qry = qry_actionstr_2_wordlist(raw_output)
        pred_ans = set(graph_sampler.search_answers_to_query(pred_qry))
    except Exception:
        return -1.0
    obs_set = set(observation_ids)
    tp = len(pred_ans & obs_set)
    precision = tp / len(pred_ans) if pred_ans else 0.0
    recall = tp / len(obs_set) if obs_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    complexity = len(raw_output.split())
    return f1 - alpha * complexity * 0.01


# ---------------------------------------------------------------------------
# LLM-guided repair candidate generation
# ---------------------------------------------------------------------------

def _propose_repairs(
    llm_model,
    hypothesis_nl: str,
    hypothesis_raw: str,
    diagnosis: dict,
    rel_candidates: list,
    ent_candidates: list,
    incoming_hints: list,
    answer_nl: list[str],
    num_repairs: int = 5,
) -> list[dict]:
    """
    Ask LLM to propose repair actions given diagnosis + KG evidence.
    Returns list of {action, description, condition} dicts.
    """
    diag = diagnosis["diagnosis"]
    fp_count = diagnosis["fp_count"]
    fn_count = diagnosis["fn_count"]
    f1 = diagnosis["f1"]

    rel_str = json.dumps(rel_candidates[:8], indent=2)
    ent_str = json.dumps(ent_candidates[:8], indent=2)
    hints_str = json.dumps(incoming_hints[:5], indent=2)

    prompt = (
        f"You are an expert in knowledge graph abductive reasoning.\n"
        f"A hypothesis needs repair. Propose {num_repairs} different repair strategies.\n\n"
        f"## Observations\n{', '.join(answer_nl)}\n\n"
        f"## Current hypothesis\n- NL: {hypothesis_nl}\n- Raw: {hypothesis_raw}\n"
        f"- F1: {f1:.4f}, FP: {fp_count}, FN: {fn_count}\n"
        f"- Diagnosis: {diag}\n\n"
        f"## KG neighborhood candidates (scored by coverage of O minus FP)\n"
        f"Top relations:\n{rel_str}\n\nTop anchor entities:\n{ent_str}\n\n"
        f"## Incoming edge hints (common sources of observations)\n{hints_str}\n\n"
        f"## Available repair actions\n"
        f"ADD_CONSTRAINT, REMOVE_CONSTRAINT, ADD_UNION_BRANCH, EXTEND_PATH, SHORTEN_PATH,\n"
        f"ADD_ENTITY, REMOVE_ENTITY, GENERALIZE_RELATION, SPECIFY_RELATION,\n"
        f"REPLACE_RELATION, REPLACE_ENTITY, REPLACE_PREDICATE, REPLACE_PATH_SEGMENT\n\n"
        f"## Guidance\n"
        f"- too_broad (many FP): prefer ADD_CONSTRAINT, SPECIFY_RELATION, REPLACE_ENTITY, ADD_ENTITY\n"
        f"- too_narrow (many FN): prefer REMOVE_CONSTRAINT, GENERALIZE_RELATION, ADD_UNION_BRANCH, SHORTEN_PATH\n"
        f"- wrong_predicates: prefer REPLACE_RELATION, REPLACE_ENTITY, REPLACE_PREDICATE, REPLACE_PATH_SEGMENT\n\n"
        f"For each repair, provide a condition string that can be parsed by the hypothesis generator.\n"
        f"Valid condition types: pattern, relation, entity, relationnumber, entitynumber, unconditional.\n\n"
        f"Return ONLY a JSON array of {num_repairs} objects:\n"
        f'[{{"action":"ACTION","description":"brief reason","condition":"I want a hypothesis that..."}}]\n'
        f"Return nothing else."
    )

    response = llm_model([{"role": "user", "content": prompt}], stop_sequences=None)
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\[.*\]', text, re.DOTALL)
        return json.loads(m.group()) if m else []


# ---------------------------------------------------------------------------
# Main repair loop
# ---------------------------------------------------------------------------

def run_repair(
    adapter,
    llm_model,
    hypothesis_raw: str,
    hypothesis_nl: str,
    observation_ids: list[int],
    answer_nl: list[str],
    max_rounds: int = 3,
    num_repairs: int = 5,
    f1_threshold: float = 0.8,
    split: str = "train",
) -> dict:
    """
    Repair a hypothesis using KG execution feedback.

    Args:
        adapter:          CtrlHGenAdapter
        llm_model:        OpenAIServerModel
        hypothesis_raw:   unshifted action string to repair
        hypothesis_nl:    natural language of initial hypothesis
        observation_ids:  raw entity IDs of observations O
        answer_nl:        entity names of observations (for model input)
        max_rounds:       max repair iterations
        num_repairs:      repair candidates per round
        f1_threshold:     stop if F1 >= this
        split:            KG graph split to execute on

    Returns:
        {
          "best_hypothesis_raw": str,
          "best_hypothesis_nl":  str,
          "best_f1":             float,
          "history":             list of round results,
        }
    """
    kg = adapter.kg
    graph_sampler = kg.graph_samplers[split]

    current_raw = hypothesis_raw
    current_nl = hypothesis_nl
    history = []

    for round_idx in range(1, max_rounds + 1):
        print(f"\n{'='*60}\n  REPAIR ROUND {round_idx}/{max_rounds}\n{'='*60}")
        print(f"  Hypothesis: {current_nl}")
        print(f"  Raw:        {current_raw}")

        # Step 1: Execute + diagnose
        diag = execute_and_diagnose_tool(current_raw, observation_ids, graph_sampler)
        print(f"  F1={diag['f1']:.4f}  TP={diag['tp_count']} FP={diag['fp_count']} FN={diag['fn_count']}  [{diag['diagnosis']}]")

        history.append({
            "round": round_idx,
            "hypothesis_raw": current_raw,
            "hypothesis_nl": current_nl,
            "f1": diag["f1"],
            "diagnosis": diag["diagnosis"],
            "fp_count": diag["fp_count"],
            "fn_count": diag["fn_count"],
        })

        if diag["f1"] >= f1_threshold:
            print(f"  F1 {diag['f1']:.4f} >= {f1_threshold}. Done.")
            break

        # Step 2: Neighborhood candidates (use FN to find missing coverage, FP to penalize)
        fn_ids = diag["fn"] if diag["fn"] else observation_ids
        nb = neighborhood_candidates_tool(
            fn_ids, graph_sampler,
            kg.ent_id2name, kg.rel_id2name,
            fp_ids=diag["fp"],
        )

        # Step 3: Incoming edge hints on observations
        incoming = incoming_edge_intersection_tool(
            observation_ids, graph_sampler,
            kg.ent_id2name, kg.rel_id2name, top_k=8,
        )

        # Step 4: LLM proposes repair candidates
        repairs = _propose_repairs(
            llm_model=llm_model,
            hypothesis_nl=current_nl,
            hypothesis_raw=current_raw,
            diagnosis=diag,
            rel_candidates=nb["relation_candidates"],
            ent_candidates=nb["entity_candidates"],
            incoming_hints=incoming.get("hints", []),
            answer_nl=answer_nl,
            num_repairs=num_repairs,
        )
        print(f"  {len(repairs)} repair candidates proposed.")

        # Step 5: Execute each repair candidate and rank by score
        best_score = _score_hypothesis(current_raw, observation_ids, graph_sampler)
        best_raw = current_raw
        best_nl = current_nl

        for i, repair in enumerate(repairs):
            condition_str = repair.get("condition", "unconditional")
            action = repair.get("action", "?")
            print(f"\n  [{i+1}] {action}: {repair.get('description','')}")
            print(f"       Condition: {condition_str}")

            # Parse condition string -> conditions list via LLM
            parse_prompt = (
                f"Parse into JSON array of condition dicts. "
                f"Valid types: unconditional, pattern, relation, entity, relationnumber, entitynumber.\n"
                f"Question: {condition_str}\n"
                f"Return ONLY the JSON array."
            )
            try:
                resp = llm_model([{"role": "user", "content": parse_prompt}], stop_sequences=None)
                ctext = resp.content.strip()
                if ctext.startswith("```"):
                    ctext = ctext.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                conditions = json.loads(ctext)
            except Exception:
                conditions = [{"type": "unconditional", "value": ""}]

            # Generate repaired hypothesis
            try:
                fmt = format_conversion_tool(adapter, answer_nl, conditions)
                gen = generate_hypothesis_tool(adapter, fmt["model_input"]["source_text"])
                rep_raw = gen["raw_output"]
                rep_nl = gen.get("query_nl", "N/A")
            except Exception as e:
                print(f"       Generation failed: {e}")
                continue

            score = _score_hypothesis(rep_raw, observation_ids, graph_sampler)
            print(f"       Score={score:.4f}  NL: {rep_nl}")

            if score > best_score:
                best_score = score
                best_raw = rep_raw
                best_nl = rep_nl

        current_raw = best_raw
        current_nl = best_nl

        if round_idx == max_rounds:
            print(f"\n  Reached max rounds ({max_rounds}).")

    # Final summary
    best_entry = max(history, key=lambda h: h["f1"])
    print(f"\n{'='*60}\n  REPAIR SUMMARY\n{'='*60}")
    for h in history:
        marker = " <-- BEST" if h is best_entry else ""
        print(f"  Round {h['round']}: F1={h['f1']:.4f} [{h['diagnosis']}]{marker}")
    print(f"  Best: {best_entry['hypothesis_nl']}")

    return {
        "best_hypothesis_raw": current_raw,
        "best_hypothesis_nl": current_nl,
        "best_f1": best_entry["f1"],
        "history": history,
    }


# ---------------------------------------------------------------------------
# Integration helpers: plug repair into uncondition / loop flows
# ---------------------------------------------------------------------------

def repair_after_uncondition(adapter, llm_model, uncondition_result: dict, case: dict, **kwargs) -> dict:
    """
    Run repair on the hypothesis produced by uncondition.run_uncondition().
    uncondition_result: return value of run_uncondition()
    case: same case dict passed to run_uncondition()
    """
    return run_repair(
        adapter=adapter,
        llm_model=llm_model,
        hypothesis_raw=uncondition_result["hypothesis_raw"],
        hypothesis_nl=uncondition_result["hypothesis_nl"],
        observation_ids=case["answers"],
        answer_nl=case["answers_nl"],
        **kwargs,
    )


def repair_after_loop(adapter, llm_model, loop_history: list, case: dict, **kwargs) -> dict:
    """
    Run repair on the best hypothesis from loop.run_loop().
    loop_history: return value of run_loop()
    case: same case dict passed to run_loop()
    """
    best = max(loop_history, key=lambda h: h["jaccard"])
    return run_repair(
        adapter=adapter,
        llm_model=llm_model,
        hypothesis_raw=best["hypothesis_raw"],
        hypothesis_nl=best["hypothesis_nl"],
        observation_ids=case["answers"],
        answer_nl=case["answers_nl"],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from akgr.utils.load_util import load_yaml
    from akgr.kgdata import load_kg
    from akgr.agent.ctrlhgen_adapter import CtrlHGenAdapter

    hypothesis_model_path = 'checkpoints/PharmKG8k-full-32-130-multi.pth'
    data_root = './data/'
    dataname = 'PharmKG8k'

    config_dataloader = load_yaml('akgr/configs/config-dataloader.yml')
    config_model = load_yaml('akgr/configs/config-model.yml')
    modelname = 'GPT2_6_act_nt'
    is_gpt = 'GPT2' in modelname
    is_act = 'act' in modelname
    tgt_len = config_dataloader['act_len'] + 1 if is_act else config_dataloader['qry_len'] + 1
    src_len = config_dataloader['ans_len'] + 1
    kg = load_kg(data_root, dataname, reverse_edges_flag=False)
    adapter = CtrlHGenAdapter(
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

    api_cfg = load_yaml('akgr/configs/api_keys.yml')['deepinfra']
    llm_model = OpenAIServerModel(
        model_id=api_cfg['model_id'],
        api_base=api_cfg['api_base'],
        api_key=api_cfg['api_key'],
    )

    case = {
        "answers": [5828, 5001, 5066, 2941, 5679, 5456, 2578, 3891, 2937, 3546, 6077, 2463],
        "answers_nl": ["rpgrip1l", "pdx1", "pfkfb1", "gys1", "recql4", "prpf6", "fxn", "ltk", "gyg1", "kcnh2", "ski", "flt4"],
    }

    # --- Option A: standalone repair from uncondition ---
    from akgr.agent.uncondition import run_uncondition
    unc_result = run_uncondition(adapter=adapter, llm_model=llm_model, case=case, num_suggestions=1)
    repair_result = repair_after_uncondition(adapter, llm_model, unc_result, case, max_rounds=3)

    # --- Option B: repair after loop ---
    # from akgr.agent.loop import run_loop
    # loop_history = run_loop(adapter=adapter, llm_model=llm_model, case=case, max_rounds=3)
    # repair_result = repair_after_loop(adapter, llm_model, loop_history, case, max_rounds=2)

    print("\nFinal best hypothesis:", repair_result["best_hypothesis_nl"])
    print("Final best F1:", repair_result["best_f1"])
