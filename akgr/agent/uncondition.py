"""
Unconditional hypothesis generation + 4 candidate strategies.

Workflow:
  1. Generate unconditional hypothesis
  2. Call graph_validation + incoming_edge_intersection once
  3. Generate 4 candidates:
     - unconditional (keep original)
     - structural (entitynumber + relationnumber + pattern)
     - semantic (one relation + one entity)
     - hybrid (structural + semantic)
  4. Evaluate all 4, save candidates + best
"""
import json
from smolagents import CodeAgent, OpenAIServerModel
from akgr.agent.tools import (
    format_conversion_tool,
    generate_hypothesis_tool,
    compute_metrics,
    GraphValidationTool,
    IncomingEdgeIntersectionTool,
    IntersectionCandidatesTool,
)
from akgr.agent.loop import parse_conditions_from_question, sub_query_prompt
from akgr.agent.single import build_adapter


def run_uncondition(
    adapter,
    llm_model,
    case: dict,
    jaccard_threshold: float = 0.95,
    verbose: bool = True,
):
    kg = adapter.kg
    answer_nl = case["answers_nl"]
    label_answers = case["answers"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  UNCONDITIONAL GENERATION")
        print(f"{'='*60}\n")

    # Step 1: Unconditional generation
    fmt_result = format_conversion_tool(
        adapter=adapter,
        answer_nl=answer_nl,
        conditions=[],
    )
    source_text = fmt_result["model_input"]["source_text"]
    gen_result = generate_hypothesis_tool(adapter, source_text)
    raw_output = gen_result["raw_output"]
    hypothesis_nl = gen_result.get("query_nl", "N/A")
    metrics = compute_metrics(
        raw_output=raw_output,
        label_answers=label_answers,
        graph_samplers=kg.graph_samplers,
    )
    jaccard = metrics["jaccard"]

    if verbose:
        print(f"[Unconditional] {raw_output}")
        print(f"  NL: {hypothesis_nl}")
        print(f"  Jaccard: {jaccard:.4f}\n")

    # Step 2: Analysis (call tools once)
    from akgr.utils.parsing_util import qry_actionstr_2_wordlist
    tokens_str = " ".join(str(t) for t in qry_actionstr_2_wordlist(raw_output)) if raw_output else ""
    label_str = ",".join(str(a) for a in label_answers)

    analysis_agent = CodeAgent(
        tools=[
            GraphValidationTool(kg=kg),
            IncomingEdgeIntersectionTool(kg=kg),
            IntersectionCandidatesTool(kg=kg),
        ],
        model=llm_model,
        additional_authorized_imports=["json"],
        max_steps=3,
        verbosity_level=1 if verbose else 0,
    )

    _is_gpt = "gpt" in getattr(llm_model, "model_id", "").lower()
    _gpt_prefix = (
        f"You are a CodeAgent. You must always respond in the following exact format:\n\n"
        f"Thoughts: Briefly explain your plan.\n\n"
        f"<code>\n# valid Python code only\n# call final_answer(...) when done\n</code>\n\n"
        f"Do not use Markdown code fences.\n"
        f"Do not omit the opening <code> tag.\n"
        f"Do not output text after </code>.\n\n"
        f"Example of a valid response:\n"
        f"Thoughts: I'll call graph_validation, then incoming_edge_intersection, then return the analysis.\n\n"
        f"<code>\n"
        f"import json\n"
        f"result = graph_validation(...)\n"
        f'final_answer({{"structural": {{"entitynumber": 2}}, "semantic": {{"relation": "GG"}}, "hybrid": {{"entitynumber": 2, "relation": "GG"}}}})\n'
        f"</code>\n\n"
    ) if _is_gpt else ""

    analysis_prompt = (
        f"{_gpt_prefix}"
        f"## Task\n"
        f"You are an analysis agent for knowledge graph (KG) abductive reasoning.\n"
        f"Given a set of observed entities $O$, the goal is to find a logical hypothesis $H$ (a KG query) "
        f"such that executing $H$ on the KG returns exactly $O$.\n"
        f"A hypothesis $H$ is a logical query in one of 13 patterns (1p/2p/2i/3i/ip/pi/2u/up/2in/3in/inp/pni/pin). "
        f"Conditions control what kind of hypothesis the generative model produces "
        f"(e.g. which relation/entity to include, the pattern shape, or counts).\n"
        f"An unconditional hypothesis was generated. Your job: generate the structural and semantic hints as the conditions to improve the hypothesis quality.\n"
        f"## Observations (entity names)\n{', '.join(answer_nl)}\n\n"
        f"## Observation IDs (raw 0-based)\n{label_str}\n\n"
        f"## Unconditional hypothesis\n"
        f"- Natural language: {hypothesis_nl}\n"
        f"- Raw action string: {raw_output}\n"
        f"- Jaccard vs observations: {jaccard:.4f}\n\n"
        f"## ID to Name Lookup\n"
        f"When you extract a relation_id R from a sub_query token (negative integer -R), look up its name: rel_id2name[R] (use positive R).\n"
        f"{sub_query_prompt()}\n\n"
        f"**Note on KG incompleteness**: The training graph may be incomplete. "
        f"If tool results seem sparse, use semantic understanding of entity/relation names to reason about plausible alternatives.\n\n"
        f"## Step 1 — Sub-logic decomposition (graph_validation)\n"
        f"IMPORTANT: Always start your code with `import json`.\n"
        f"ALL tools return JSON strings — always parse with `json.loads()` before indexing.\n"
        f"TOOL BUDGET: Call graph_validation at most 1 time. Then call incoming_edge_intersection once. Then output final_answer.\n"
        f"Call graph_validation(query_tokens='{tokens_str}', label_answers='{label_str}', split='train').\n"
        f"It returns sub_query_results: each entry has 'sub_query', 'answer_count', 'overlap_count', 'relation_to_label'.\n"
        f"- Find the sub-query with HIGHEST overlap_count — best building block for the semantic condition.\n"
        f"- Find the sub-query with LOWEST overlap_count — weakest branch, informs structural redesign.\n"
        f"- Count the number of sub-queries to infer entitynumber and relationnumber.\n\n"
        f"## Step 2 — Neighborhood search (incoming_edge_intersection)\n"
        f"Call incoming_edge_intersection(answer_entity_ids='{label_str}', split='train', top_k=10).\n"
        f"The result contains:\n"
        f"- flat_candidates: 1-hop (entity, relation) pairs with jaccard vs O. Pick the top-1 as the best semantic (relation, entity) pair.\n"
        f"- two_hop_candidates: 2-hop paths with jaccard. Use to infer a good pattern (e.g. 2p if top two_hop has high jaccard).\n\n"
        f"## Step 3 — Return analysis result\n"
        f"Based on Steps 1–2, return a JSON object via final_answer() with three top-level keys (always include all three):\n"
        f"- 'structural': optional keys among 'entitynumber' (int 1-3), 'relationnumber' (int 1-3), 'pattern' (string using i/u/n/p/e tokens). "
        f"You do **not** need to fill every key—include **at least one** structural hint you want to condition on.\n"
        f"- 'semantic': optional keys among 'relation' (ONE relation NAME from flat_candidates), 'entity' (ONE entity NAME from flat_candidates). "
        f"Include **at least one** of relation or entity.\n"
        f"- 'hybrid': optional keys among ones in structural and semantic. "
        f"You do **not** need every eligible key—include **at least one** from structural (entitynumber/relationnumber/pattern) **and** **at least one** from semantic (relation/entity) so hybrid always mixes both.\n"
        f"Constraints:\n"
        f"- structural must be a dict with **at least one** of: entitynumber, relationnumber, pattern.\n"
        f"- semantic must be a dict with **at least one** of: relation, entity.\n"
        f"- When present, entitynumber and relationnumber must be integers between 1 and 3.\n"
        f"- When present, pattern must use only i/u/n/p/e tokens (e.g. 'i p e p e', 'p p e').\n"
        f"- When present, relation and entity must be NAME strings (not IDs).\n\n"
        f"Example:\n"
        f"```python\n"
        f"final_answer({{\n"
        f'  "structural": {{"entitynumber": 2}},\n'
        f'  "semantic": {{"relation": "GG", "entity": "pask"}},\n'
        f'  "hybrid": {{"entitynumber": 2, "relation": "GG"}},\n'
        f"}})\n"
        f"```\n\n"
        f"Suggestion:\n"
        f"- Anchor conditions to the **unconditional hypothesis** already shown (natural language + raw action string): treat it as the primary reference—reuse relations, entities, and compositional structure that are already present rather than proposing unrelated constraints.\n"
        f"- **graph_validation** sub-logic (per-branch overlap vs observations) helps judge which parts of that hypothesis are well supported vs weak; favor conditioning on the stronger fragments and use the weakest branch to justify structural or semantic fixes.\n"
        f"- **incoming_edge_intersection** neighborhood search (1-hop / 2-hop candidates vs observations) further validates which hooks are plausible; use those signals to turn the most defensible pieces of the unconditional output into explicit structural or semantic conditions.\n"
        + (
            f"REMINDER: wrap ALL code in <code>...</code> tags. Never use ```python```. Never write 'Thought:' — use 'Thoughts:'.\n"
            if _is_gpt else ""
        )
    )

    try:
        analysis_result = analysis_agent.run(analysis_prompt)
        if isinstance(analysis_result, str):
            text = analysis_result.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            analysis_data = json.loads(text)
        else:
            analysis_data = analysis_result
    except Exception as e:
        if verbose:
            print(f"  [Analysis error] {e}")
        analysis_data = {
            "structural": {"entitynumber": 2, "relationnumber": 2, "pattern": "i p e p e"},
            "semantic": {"relation": "", "entity": ""},
            "analysis": "Analysis failed, using defaults",
        }

    structural = analysis_data.get("structural", {})
    semantic = analysis_data.get("semantic", {})
    hybrid = analysis_data.get("hybrid", {})

    def _dict_to_conditions(d):
        """Convert a condition dict to a list of {type, value} entries, filtering empty values."""
        return [{"type": k, "value": str(v)} for k, v in d.items() if v not in (None, "")]

    # Step 3: Generate 4 candidates
    candidate_conditions = [
        # 1. unconditional
        [],
        # 2. structural
        _dict_to_conditions(structural),
        # 3. semantic
        _dict_to_conditions(semantic),
        # 4. hybrid
        _dict_to_conditions(hybrid),
    ]

    candidate_names = ["unconditional", "structural", "semantic", "hybrid"]
    candidates = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  EVALUATING 4 CANDIDATES")
        print(f"{'='*60}\n")

    for name, cond_list in zip(candidate_names, candidate_conditions):
        try:
            fmt = format_conversion_tool(adapter=adapter, answer_nl=answer_nl, conditions=cond_list)
            gen = generate_hypothesis_tool(adapter, fmt["model_input"]["source_text"])
            m = compute_metrics(raw_output=gen["raw_output"], label_answers=label_answers, graph_samplers=kg.graph_samplers)
            candidates.append({
                "name": name,
                "conditions": cond_list,
                "hypothesis_raw": gen["raw_output"],
                "hypothesis_nl": gen.get("query_nl"),
                "jaccard": m["jaccard"],
                "dice": m["dice"],
                "overlap": m["overlap"],
            })
            if verbose:
                print(f"  [{name}] J={m['jaccard']:.4f} | {gen['raw_output']}")
        except Exception as e:
            if verbose:
                print(f"  [{name}] Error: {e}")
            candidates.append({
                "name": name,
                "conditions": cond_list,
                "hypothesis_raw": None,
                "hypothesis_nl": None,
                "jaccard": 0.0,
                "dice": 0.0,
                "overlap": 0.0,
            })

    # Pick best — include the initial unconditional generation
    all_results = [{
        "name": "unconditional_initial",
        "conditions": [],
        "hypothesis_raw": raw_output,
        "hypothesis_nl": hypothesis_nl,
        "jaccard": jaccard,
        "dice": metrics["dice"],
        "overlap": metrics["overlap"],
    }] + candidates
    best = max(all_results, key=lambda c: (c["jaccard"], c.get("dice") or 0))

    if verbose:
        print(f"\n[Best] {best['name']} | Jaccard={best['jaccard']:.4f}")

    return {
        "answers": label_answers,
        "answers_nl": answer_nl,
        "unconditional": {
            "hypothesis_raw": raw_output,
            "hypothesis_nl": hypothesis_nl,
            "jaccard": jaccard,
            "dice": metrics["dice"],
            "overlap": metrics["overlap"],
        },
        "analysis": analysis_data.get("analysis", ""),
        "generated_conditions": {
            "structural": structural,
            "semantic": semantic,
            "hybrid": hybrid,
        },
        "candidates": candidates,
        "best": best,
    }


def _save_result(log_path, result):
    compact_keys = ("answers", "answers_nl")
    placeholders = {}
    for k in compact_keys:
        if result.get(k) is not None:
            ph = f'"__PH_{k}__"'
            placeholders[ph] = json.dumps(result[k], ensure_ascii=False)
            result[k] = f"__PH_{k}__"
    text = json.dumps(result, ensure_ascii=False, indent=2)
    for ph, val in placeholders.items():
        text = text.replace(ph, val)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    import argparse, os
    from akgr.agent.case import case_1p,case_2in

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["case", "run"], default="case")
    parser.add_argument("--dataname", default="DBpedia50")
    parser.add_argument("--checkpoint", default="/home/gaoyisen/akgr-agent/checkpoints/DBpedia50-full-32-300-multi.pth")
    parser.add_argument("--data_root", default="/home/gaoyisen/akgr-agent/data/")
    parser.add_argument("--jaccard_threshold", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    from akgr.utils.load_util import load_yaml
    # api_cfg = load_yaml("akgr/configs/api_keys.yml")["deepinfra"]
    api_cfg = load_yaml("akgr/configs/api_keys.yml")["xlabapi"]
    _is_gpt_model = "gpt" in api_cfg["model_id"].lower()
    llm_model = OpenAIServerModel(
        model_id=api_cfg["model_id"],
        api_base=api_cfg["api_base"],
        api_key=api_cfg["api_key"],
        timeout=60,
        **({"extra_headers": {"User-Agent": "claude-cli/2.0.76 (external, cli)"}} if _is_gpt_model else {}),
    )
    adapter = build_adapter(args.checkpoint, args.data_root, args.dataname)
    case2 =case_2in
    if args.mode == "case":
        log_dir = os.path.join("log", args.dataname)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "uncondition.jsonl")
        result = run_uncondition(
            adapter=adapter, llm_model=llm_model, case=case2,
            jaccard_threshold=args.jaccard_threshold,
        )
        _save_result(log_path, result)

    else:
        from tqdm import tqdm
        data_file = os.path.join(args.data_root, args.dataname, "singleturn.jsonl")
        log_dir = os.path.join("log", args.dataname)
        os.makedirs(log_dir, exist_ok=True)
        model_tag = api_cfg["model_id"].split("/")[-1]
        log_path = os.path.join(log_dir, f"uncondition_{model_tag}.jsonl")

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
                result = run_uncondition(
                    adapter=adapter, llm_model=llm_model, case=case,
                    jaccard_threshold=args.jaccard_threshold, verbose=False,
                )
                _save_result(log_path, result)
            except Exception as e:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"error": str(e), "answers": case.get("answers")}, ensure_ascii=False) + "\n")
