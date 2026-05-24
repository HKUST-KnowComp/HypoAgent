import json
from smolagents import CodeAgent, OpenAIServerModel
from akgr.agent.tools import (
    format_conversion_tool,
    generate_hypothesis_tool,
    compute_metrics,
    GraphValidationTool,
    IncomingEdgeIntersectionTool,
    IntersectionCandidatesTool,
    GenerateHypothesisLLMTool,
)
from akgr.utils.parsing_util import qry_actionstr_2_wordlist
from akgr.agent.case import case_2i,case_2p,case_1p,case_2u,case_up,case_ip


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
        f"Valid condition types and their value formats:\n"
        f"  - 'relation':      value = ONE relation NAME (string), e.g. 'GG'\n"
        f"  - 'entity':        value = ONE entity NAME (string), e.g. 'chrnb3'\n"
        f"  - 'relationnumber': value = integer count of relations, e.g. '3'\n"
        f"  - 'entitynumber':  value = integer count of entities, e.g. '2'\n"
        f"  - 'pattern':       value = structural pattern using only i/u/n/p/e tokens, e.g. 'i p e p e'\n"
        f"Rules:\n"
        f"  - NEVER use 'unconditional' — always infer at least one concrete condition from the question.\n"
        f"  - AT MOST ONE 'relation' condition and AT MOST ONE 'entity' condition.\n"
        f"  - 'relationnumber' and 'entitynumber' take integer values, NOT relation/entity names.\n"
        f"  - 'relation' value must be a name string, NOT an integer ID.\n"
        f"  - 'pattern' value must use only i/u/n/p/e tokens, NOT NL expressions like 'p(GG, e(X))'.\n"
        f"  - Never output empty string as value for 'relation' or 'entity'.\n"
        f"Examples:\n"
        f'  "I want a hypothesis with relation GG"\n'
        f'  -> [{{"type":"relation","value":"GG"}}]\n\n'
        f'  "I want pattern i p e p e with 2 entities"\n'
        f'  -> [{{"type":"pattern","value":"i p e p e"}},{{"type":"entitynumber","value":"2"}}]\n\n'
        f'  "I want a hypothesis with 3 relations and 2 entities including relation GG and entity chrnb3"\n'
        f'  -> [{{"type":"relationnumber","value":"3"}},{{"type":"entitynumber","value":"2"}},{{"type":"relation","value":"GG"}},{{"type":"entity","value":"chrnb3"}}]\n\n'
        f"Question: {followup_question}\n\n"
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

def sub_query_prompt() -> str:
    return """
# How to Analyze a Hypothesis

## What is a Hypothesis?

A hypothesis H is a logical query over a knowledge graph (KG). When executed on the KG, H returns
a set of entities. The goal of abductive reasoning is to find H such that its result set matches
the observed entities O as closely as possible (measured by Jaccard similarity).

A hypothesis is built from:
- **p(relation, sub)** — path: follow `relation` from the result of `sub`
- **e(entity)** — anchor: start from a specific entity
- **i(A, B, ...)** — intersection: entities in ALL of A, B, ...
- **u(A, B)** — union: entities in EITHER A or B
- **n(A)** — negation: entities NOT in A (used only inside i to exclude)

### Examples

**Example 1 — 1p: `p(GG, e(pask))`**
"Entities that pask has a GG link to."
Execution: find all entities X such that (pask, GG, X) is a triple in the KG.


**Example 2 — 2i: `i(p(B, e(pex19)), p(GG, e(abcd1)))`**
"Entities that pex19 has a B link to, AND that abcd1 has a GG link to."
Execution: branch1 = {X | (pex19, B, X)}, branch2 = {X | (abcd1, GG, X)}, result = branch1 ∩ branch2.


**Example 3 — 2p: `p(Ra, p(GG, e(chrnb3)))`**
"Entities that some intermediate Y has a Ra link to, where chrnb3 has a GG link to Y."
Execution: step1 = {Y | (chrnb3, GG, Y)}, step2 = {X | (Y, Ra, X) for Y in step1}.


**Example 4 — 2in: `i(n(p(Q, e(arg1))), p(Ra, e(chrnb3)))`**
"Entities that chrnb3 has a Ra link to, but that arg1 does NOT have a Q link to."
Execution: positive = {X | (chrnb3, Ra, X)}, excluded = {X | (arg1, Q, X)}, result = positive − excluded.
The positive branch drives recall; the negation only trims false positives.

---

## 13 Patterns: Structure and Analysis Focus

| Pattern | Structure | Primary Analysis Focus |
|---------|-----------|----------------------|
| **1p** | `p(r, e(a))` | Neighborhood search: find the best (relation, anchor) pair via incoming_edge_intersection flat_candidates. |
| **2p** | `p(r1, p(r2, e(a)))` | Neighborhood search: use two_hop_candidates to find (r1, r2, anchor) with high jaccard. Check if inner p(r2,e(a)) is non-empty. |
| **2i** | `i(p(r1,e(a1)), p(r2,e(a2)))` | Sub-logic decomposition: validate each branch independently. Fix the branch with lowest overlap first. |
| **3i** | `i(p(r1,e(a1)), p(r2,e(a2)), p(r3,e(a3)))` | Sub-logic decomposition: validate all 3 branches. Use intersection_candidates(mode='3i') to find best triple. |
| **ip** | `p(r0, i(p(r1,e(a1)), p(r2,e(a2))))` | Sub-logic decomposition of inner 2i first; then check if inner_i_sublogic is non-empty. Use two_hop_candidates for outer r0. |
| **pi** | `i(p(r1,e(a1)), p(r2, p(r3,e(a2))))` | Sub-logic decomposition: fix the 1p branch and the 2p chain branch separately. Use two_hop_candidates for the chain branch. |
| **2u** | `u(p(r1,e(a1)), p(r2,e(a2)))` | Sub-logic decomposition: each branch contributes independently. Adding a well-chosen OR branch can increase coverage without breaking correctness. |
| **up** | `p(r0, u(p(r1,e(a1)), p(r2,e(a2))))` | Sub-logic decomposition of inner 2u; then use two_hop_candidates for outer r0. |
| **2in** | `i(n(p(r1,e(a1))), p(r2,e(a2)))` | Fix positive branch p(r2,e(a2)) first to maximize recall. Negation only reduces FP — do not over-constrain it. |
| **3in** | `i(i(n(p(r1,e(a1))), p(r2,e(a2))), p(r3,e(a3)))` | Fix the inner 2in and outer positive branch independently. Negation serves only to narrow scope. |
| **inp** | `p(r0, i(n(p(r1,e(a1))), p(r2,e(a2))))` | Fix inner 2in first; check inner_i_sublogic is non-empty. Use two_hop_candidates for outer r0. |
| **pni** | `i(n(p(r1, p(r2,e(a1)))), p(r3,e(a2)))` | Fix positive branch p(r3,e(a2)) first. The negated 2p chain excludes a broad set — verify it does not over-exclude. |
| **pin** | `i(n(p(r1,e(a1))), p(r2, p(r3,e(a2))))` | Fix positive 2p chain using two_hop_candidates. Negated 1p branch only trims FP. |

### Key Rules

- **Intersectionpatterns (2i, 3i, ip, pi or others contain i)**: Each branch is critical. Identify the weakest branch (lowest overlap_count in graph_validation) and fix it first.
- **Chain patterns (1p, 2p, and chain branches in ip/pi/up/inp/pni/pin)**: Use incoming_edge_intersection flat_candidates (1-hop) and two_hop_candidates (2-hop) to find better (relation, anchor) combinations.
- **Negation patterns (2in, 3in, inp, pni, pin)**: The non-negated sub-logic is the primary building block — maximize its recall first. The negated branch serves ONLY to narrow the result (reduce FP); never use it to fix missing answers.
- **inner_i_sublogic empty check**: For ip/inp/up patterns, if graph_validation reports inner_i_sublogic.empty=true, the inner intersection yields nothing — the entire i-branch must be replaced.
- **Answer set quality (best to worst)**: exact_match > label_contains_result > partial_overlap > disjoint
"""



def run_loop(
    adapter,
    llm_model,
    case: dict,
    max_rounds: int = 3,
    jaccard_threshold: float = 0.8,
    verbose: bool = True,
    initial_conditions: list[dict] = None,
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

    # Available info for the analysis agent
    rel_name2id = adapter.mapper.rel_name2id

    def _is_better(j1, d1, j2, d2):
        """Return True if (j1, d1) is better than (j2, d2): jaccard first, dice as tiebreak within 1e-5."""
        if j1 > j2 + 1e-5:
            return True
        if abs(j1 - j2) <= 1e-5 and d1 > d2:
            return True
        return False

    history: list[dict] = []

    for round_idx in range(1, max_rounds + 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  ROUND {round_idx} / {max_rounds}")
            print(f"  Condition: {original_followup}")
            print(f"{'='*60}\n")

        # ----- Step 1: Parse conditions from question -----
        if round_idx == 1 and initial_conditions is not None:
            conditions = initial_conditions
        else:
            conditions = parse_conditions_from_question(llm_model, original_followup)
        if verbose:
            print(f"[Step 1] Parsed conditions: {conditions}")

        # ----- Step 2: Format conversion -----
        fmt_result = format_conversion_tool(adapter=adapter, answer_nl=answer_nl, conditions=conditions)
        source_text = fmt_result["model_input"]["source_text"]
        if verbose:
            print(f"[Step 2] Source text: {source_text}")

        # ----- Step 3: Generate hypothesis -----
        gen_result = generate_hypothesis_tool(adapter, source_text)
        raw_output = gen_result["raw_output"]
        hypothesis_nl = gen_result.get("query_nl", "N/A")
        metrics = compute_metrics(raw_output=raw_output, label_answers=label_answers, graph_samplers=kg.graph_samplers)
        jaccard = metrics["jaccard"]
        if verbose:
            print(f"[Step 3] Hypothesis (raw): {raw_output}")
            print(f"[Step 4] Jaccard: {jaccard:.4f}, Dice: {metrics['dice']:.4f}, Overlap: {metrics['overlap']:.4f}")

        round_result = {
            "round": round_idx,
            "condition": original_followup,
            "parsed_conditions": conditions,
            "hypothesis_raw": raw_output,
            "hypothesis_nl": hypothesis_nl,
            "jaccard": jaccard,
            "dice": metrics["dice"],
            "overlap": metrics["overlap"],
            "pred_answer_count": len(metrics["pred_answers"]),
            "label_answer_count": len(metrics["label_answers"]),
            "metrics": metrics,
            "candidates": [],
        }
        history.append(round_result)

        # ----- Check stopping criteria -----
        if jaccard >= jaccard_threshold:
            if verbose:
                print(f"\n*** Jaccard {jaccard:.4f} >= {jaccard_threshold}. Stopping. ***")
            break

        if round_idx == max_rounds and verbose:
            print(f"\n*** Reached max rounds ({max_rounds}). Running final analysis. ***")

        # ----- Step 5: Analysis agent for next round -----
        if verbose:
            print(f"\n--- Analysis Phase (CodeAgent) ---")

        # Build history summary
        history_lines = []
        for entry in history:
            history_lines.append(
                f"  Round {entry['round']}: condition='{entry['condition']}', "
                f"hypothesis_nl='{entry['hypothesis_nl']}', jaccard={entry['jaccard']:.4f}, "
                f"pred={entry['pred_answer_count']}, label={entry['label_answer_count']}"
            )
        history_text = "\n".join(history_lines)

        tokens_str = " ".join(str(t) for t in qry_actionstr_2_wordlist(raw_output)) if raw_output else ""
        label_str = ",".join(str(a) for a in label_answers)
        answer_ids_str = ",".join(str(a) for a in label_answers)

        # Detect pattern of current hypothesis by matching pattern_str
        _PATTERN_MAP = {
            "(p,(e))": "1p", "(p,(p,(e)))": "2p", "(p,(p,(p,(e))))": "3p",
            "(i,(p,(e)),(p,(e)))": "2i", "(i,(i,(p,(e)),(p,(e))),(p,(e)))": "3i",
            "(p,(i,(p,(e)),(p,(e))))": "ip", "(i,(p,(e)),(p,(p,(e))))": "pi",
            "(u,(p,(e)),(p,(e)))": "2u", "(p,(u,(p,(e)),(p,(e))))": "up",
            "(i,(n,(p,(e))),(p,(e)))": "2in", "(i,(i,(n,(p,(e))),(p,(e))),(p,(e)))": "3in",
            "(p,(i,(n,(p,(e))),(p,(e))))": "inp", "(i,(n,(p,(p,(e)))),(p,(e)))": "pni",
            "(i,(n,(p,(e))),(p,(p,(e))))": "pin",
        }
        _TWO_HOP_PATTERNS = {"2p", "3p", "ip", "pi", "up", "inp", "pni", "pin"}

        def _tokens_to_pattern(toks: list) -> str:
            if not toks or toks[0] != '(':
                return ""
            op = toks[1]
            if op == 'e':
                return "(e)"
            if op in ('p', 'n', 'u', 'i'):
                depth, start, children = 0, None, []
                for idx, t in enumerate(toks):
                    if t == '(':
                        depth += 1
                        if depth == 2:
                            start = idx
                    elif t == ')':
                        depth -= 1
                        if depth == 1 and start is not None:
                            candidate = toks[start:idx+1]
                            # only recurse into sub-expressions (op is a string), skip scalar brackets like (-8) or (1312)
                            if len(candidate) > 1 and isinstance(candidate[1], str):
                                children.append(candidate)
                            start = None
                subs = ",".join(_tokens_to_pattern(c) for c in children)
                return f"({op},{subs})"
            return ""

        def _infer_pattern_from_flat(raw: str) -> str:
            """Fallback: infer pattern from flat action string tokens."""
            toks = raw.strip().split()
            if not toks:
                return ""
            ops = [t for t in toks if t in ('i', 'u', 'n')]
            neg_count = sum(1 for t in toks if t.lstrip('-').isdigit() and int(t) < 0)
            pos_count = sum(1 for t in toks if t.lstrip('-').isdigit() and int(t) > 0)
            first = toks[0]
            has_i = 'i' in ops
            has_n = 'n' in ops
            has_u = 'u' in ops
            is_neg = first.lstrip('-').isdigit() and int(first) < 0

            # p-headed patterns: start with negative number
            if is_neg:
                if not has_i and not has_n and not has_u:
                    if neg_count == 1 and pos_count == 1: return "(p,(e))"       # 1p
                    if neg_count == 2 and pos_count == 1: return "(p,(p,(e)))"   # 2p
                if has_i and not has_n and not has_u:
                    if neg_count == 3 and pos_count == 2: return "(p,(i,(p,(e)),(p,(e))))"  # ip
                if has_i and has_n and not has_u:
                    if neg_count == 3 and pos_count == 2: return "(p,(i,(n,(p,(e))),(p,(e))))"  # inp
                if has_u and not has_i and not has_n:
                    if neg_count == 3 and pos_count == 2: return "(p,(u,(p,(e)),(p,(e))))"  # up

            # i-headed patterns
            if first == 'i' and not has_n and not has_u:
                if neg_count == 2 and pos_count == 2: return "(i,(p,(e)),(p,(e)))"          # 2i
                if neg_count == 3 and pos_count == 3: return "(i,(p,(e)),(p,(e)),(p,(e)))"  # 3i
                if neg_count == 3 and pos_count == 2: return "(i,(p,(e)),(p,(p,(e))))"      # pi

            # u-headed patterns
            if first == 'u' and not has_n:
                if neg_count == 2 and pos_count == 2: return "(u,(p,(e)),(p,(e)))"          # 2u

            # i-headed with negation
            if first == 'i' and has_n and not has_u:
                if neg_count == 2 and pos_count == 2: return "(i,(n,(p,(e))),(p,(e)))"      # 2in
                if neg_count == 3 and pos_count == 3: return "(i,(i,(n,(p,(e))),(p,(e))),(p,(e)))"  # 3in
                if neg_count == 3 and pos_count == 2:
                    n_idx = toks.index('n')
                    consec = 0
                    for t in toks[n_idx+1:]:
                        if t.lstrip('-').isdigit() and int(t) < 0:
                            consec += 1
                        else:
                            break
                    if consec >= 2:
                        return "(i,(n,(p,(p,(e)))),(p,(e)))"   # pni
                    else:
                        return "(i,(n,(p,(e))),(p,(p,(e))))"   # pin

            return ""

        try:
            _qtoks = qry_actionstr_2_wordlist(raw_output)
            _pstr = _tokens_to_pattern(_qtoks) if _qtoks else ""
            if not _pstr:
                _pstr = _infer_pattern_from_flat(raw_output)
        except Exception:
            _pstr = _infer_pattern_from_flat(raw_output)
        current_pattern = _PATTERN_MAP.get(_pstr, "unknown") if _pstr else "unknown"

        analysis_agent = CodeAgent(
            tools=[
                GraphValidationTool(kg=kg),
                IncomingEdgeIntersectionTool(kg=kg),
                IntersectionCandidatesTool(kg=kg),
                GenerateHypothesisLLMTool(kg=kg, llm_model=llm_model),
            ],
            model=llm_model,
            additional_authorized_imports=["json"],
            max_steps=5,
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
            f"Thoughts: I'll inspect the current hypothesis using graph_validation, then use neighborhood search "
            f"to identify better candidates, and finally return exactly 3 candidate hypotheses.\n\n"
            f"<code>\n"
            f"import json\n"
            f"# TODO: replace with actual tool calls\n"
            f"candidates = [\n"
            f'  {{"analysis": "...", "new_condition": "I want a hypothesis that...", "hypothesis_raw": None}},\n'
            f'  {{"analysis": "...", "new_condition": "I want a hypothesis that...", "hypothesis_raw": None}},\n'
            f'  {{"analysis": "...", "new_condition": "I want a hypothesis that...", "hypothesis_raw": "i -8 1312 -20 1312"}},\n'
            f"]\n"
            f"final_answer(candidates)\n"
            f"</code>\n\n"
        ) if _is_gpt else ""

        agent_prompt = (
            f"{_gpt_prefix}"
            # --- a. Task setup ---
            f"## Task\n"
            f"You are an analysis agent for knowledge graph (KG) abductive reasoning.\n"
            f"Given a set of observed entities $O$, the goal is to find a logical hypothesis $H$ (a KG query) "
            f"such that executing $H$ on the KG returns exactly $O$.\n"
            f"A hypothesis $H$ is a logical query in one of 13 patterns (1p/2p/2i/3i/ip/pi/2u/up/2in/3in/inp/pni/pin). "
            f"Conditions control what kind of hypothesis the generative model produces "
            f"(e.g. which relation/entity to include, the pattern shape, or counts).\n"
            f"Your job: analyze why the current hypothesis is imperfect, then propose 3 candidates to improve it.\n\n"
            # --- b. Current context ---
            f"## Observations (entity names)\n{', '.join(answer_nl)}\n\n"
            f"## Observation IDs (raw 0-based)\n{answer_ids_str}\n\n"
            f"## User's original condition (MUST be respected in all candidates)\n{original_followup}\n\n"
            f"## Generation history\n{history_text}\n\n"
            f"## Current hypothesis\n"
            f"- Natural language: {hypothesis_nl}\n"
            f"- Raw action string: {raw_output}\n"
            f"- Pattern: {current_pattern}\n"
            f"- Jaccard vs observations: {jaccard:.4f}\n\n"
            f"## ID to Name Lookup\n"
            f"When you extract a relation_id R from a sub_query token (negative integer -R), look up its name: rel_id2name[R] (use positive R).\n"
            f"When you extract an entity_id E, look up its name: ent_id2name[E].\n"
            f"Relation id->name: {dict(list(kg.rel_id2name.items())[:40])}\n"
            f"Entity id->name (sample): {dict(list(kg.ent_id2name.items())[:20])}\n\n"
            # --- c. Sub-logic decomposition ---
            f"{sub_query_prompt()}\n\n"
            f"**Note on KG incompleteness**: The training graph may be incomplete — some true edges may be missing. "
            f"If tool results seem sparse or a sub-query returns unexpectedly few results, use your semantic understanding "
            f"of entity/relation names to reason about plausible alternatives, rather than relying solely on graph statistics.\n\n"
            f"## Step 1 — Sub-logic decomposition (graph_validation)\n"
            f"IMPORTANT: Always start your code with `import json`.\n"
            f"ALL tools return JSON strings — always parse with `json.loads()` before indexing.\n"
            f"TOOL BUDGET: Call graph_validation at most 3 times total. After that, output your final_answer immediately without further tool calls.\n"
            f"Call graph_validation(query_tokens='{tokens_str}', label_answers='{label_str}', split='train').\n"
            f"It returns sub_query_results: each entry has 'sub_query', 'answer_count', 'overlap_count', 'relation_to_label'.\n"
            f"- Find the sub-query with HIGHEST overlap_count — best building block.\n"
            f"- Find the sub-query with LOWEST overlap_count — weakest branch to fix.\n"
            f"- If a p-headed sub-query has 'inner_i_sublogic' with empty=true, the i-branch must be replaced.\n"
            f"- DO NOT try to parse sub_query token lists manually. Just use overlap_count and relation_to_label to judge quality.\n"
            f"  The entity/relation names and IDs you need will come from Step 2 (incoming_edge_intersection).\n"
            f"{'- This pattern contains a 2-hop chain — you MUST inspect two_hop_candidates (Step 2) for better (hop1, hop2, anchor) combinations.' if current_pattern in _TWO_HOP_PATTERNS else ''}\n\n"
            # --- d. Neighborhood search ---
            f"## Step 2 — Neighborhood search (incoming_edge_intersection)\n"
            f"Call incoming_edge_intersection(answer_entity_ids='{answer_ids_str}', split='train', top_k=10).\n"
            f"Store the result string in a variable, e.g. `incoming_str = incoming_edge_intersection(...)`.\n"
            f"The result contains:\n"
            f"- flat_candidates: 1-hop (entity, relation) pairs with jaccard vs O. Each represents p(relation, e(entity)).\n"
            f"- two_hop_candidates: 2-hop paths (hop1_relation, hop2_relation, anchor_entity) with jaccard. Each represents p(hop1, p(hop2, e(anchor))).\n"
            f"Use flat_candidates to find better 1-hop anchors/relations for weak branches.\n"
            f"Use two_hop_candidates when the pattern is 2p/ip/pi/up/inp/pni/pin.\n\n"
            f"If the pattern is 2i/3i/ip/pi, also call:\n"
            f"  intersection_candidates(flat_candidates_json=incoming_str, observation_ids='{answer_ids_str}', "
            f"mode='{'3i' if current_pattern == '3i' else '2i'}', split='train')\n"
            f"Pass incoming_str DIRECTLY as flat_candidates_json — do NOT json.loads it first.\n\n"
            # --- e. Generate 3 candidates ---
            f"## Step 3 — Produce 3 candidates\n"
            f"Based on Steps 1–2, produce EXACTLY 3 candidates as a JSON array.\n"
            f"Constraints for all candidates:\n"
            f"- AT MOST 3 relations and AT MOST 3 entities total in the hypothesis.\n"
            f"- Relation IDs passed to generate_hypothesis_llm must be POSITIVE integers (tool negates internally).\n\n"
            f"**new_condition format rules** (Candidate 1 and 2 only):\n"
            f"A new_condition is a natural language string that will be parsed into AT MOST ONE of each allowed condition type.\n"
            f"Allowed condition types (use AT MOST ONE entity and AT MOST ONE relation):\n"
            f"  - entitynumber: 'I want a hypothesis with 2 entities' (integer count)\n"
            f"  - relationnumber: 'I want a hypothesis with 3 relations' (integer count)\n"
            f"  - relation: 'I want a hypothesis that includes relation GG' (ONE relation NAME, not ID)\n"
            f"  - entity: 'I want a hypothesis that includes entity chrnb3' (ONE entity NAME, not ID)\n"
            f"  - pattern: 'I want a hypothesis with pattern i p e p e' (structural pattern using i/u/n/p/e tokens only)\n"
            f"FORBIDDEN in new_condition: multiple relations, multiple entities, NL query expressions like 'p(Ra, p(GG, e(X)))'.\n"
            f"MANDATORY: Every new_condition MUST incorporate all constraints from the user's original condition: '{original_followup}'. Do not drop any of them.\n"
            f"Good example: 'I want a hypothesis that includes relation Ra and entity chrnb3 with 2 entities'\n"
            f"Bad example: 'I want a hypothesis based on the chain p(Ra, p(GG, e(chrnb3)))' ← forbidden NL query\n\n"
            f"Candidate 1 (keep): Keep the original condition unchanged. Set hypothesis_raw=null.\n"
            f"Candidate 2 (update): Propose a new_condition that EXTENDS the original condition with additional constraints from your analysis. Set hypothesis_raw=null.\n"
            f"Candidate 3 (generate): Based on your analysis, directly write a flat action string as hypothesis_raw. "
            f"Do NOT call any tool for this — just compose the string yourself.\n"
            f"Flat action string format: space-separated tokens, NO parentheses, NO 'p'/'e' tokens.\n"
            f"  Operators: i (intersection), u (union), n (negation)\n"
            f"  Relations: negative integers (e.g. -8 for relation id 8)\n"
            f"  Entities: positive integers (e.g. 1312 for entity id 1312)\n"
            f"  Examples: 1p: '-8 1312', 2i: 'i -8 1312 -20 1303', pin: 'i n -8 1312 -20 -13 4527'\n"
            f"  The entity/relation IDs must come from the tool results above (flat_candidates, two_hop_candidates, sub_query_results).\n\n"
            f"Return your result by calling final_answer() with the candidates list. Example:\n"
            f"```python\n"
            f"candidates = [\n"
            f'  {{"analysis": "...", "new_condition": "I want a hypothesis that...", "hypothesis_raw": None}},\n'
            f'  {{"analysis": "...", "new_condition": "I want a hypothesis that...", "hypothesis_raw": None}},\n'
            f'  {{"analysis": "...", "new_condition": "I want a hypothesis that...", "hypothesis_raw": "i -8 1312 -20 1312"}},\n'
            f"]\n"
            f"final_answer(candidates)\n"
            f"```\n"
            + (
            f"REMINDER: wrap ALL code in <code>...</code> tags. Never use ```python```. Never write 'Thought:' — use 'Thoughts:'.\n"
            if _is_gpt else ""
            )
        )

        agent_result = analysis_agent.run(agent_prompt)
        # agent_result may already be a list/dict (from final_answer), or a JSON string
        if agent_result is None:
            candidates = []
        elif isinstance(agent_result, list):
            candidates = [c for c in agent_result if isinstance(c, dict)]
        elif isinstance(agent_result, dict):
            candidates = [agent_result]
        else:
            result_text = str(agent_result).strip()
            if result_text.startswith("```"):
                result_text = result_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            try:
                candidates = json.loads(result_text)
                if isinstance(candidates, dict):
                    candidates = [candidates]
            except (json.JSONDecodeError, ValueError):
                import re
                match = re.search(r'\[.*?\]', result_text, re.DOTALL)
                try:
                    candidates = json.loads(match.group()) if match else None
                except Exception:
                    candidates = None
                if not candidates:
                    candidates = [{"analysis": "Parse failed", "new_condition": original_followup}]

        # Run generation for each candidate, pick best Jaccard
        if verbose:
            print(f"\n--- Evaluating {len(candidates)} candidates ---")
        early_stop = False
        evaluated_candidates = []
        for ci, cand in enumerate(candidates):
            cand_condition = cand.get("new_condition", original_followup)
            if verbose:
                print(f"\n  Candidate {ci+1}: {cand_condition}")
            cand_entry = {
                "condition": cand_condition,
                "analysis": cand.get("analysis", ""),
                "hypothesis_raw": None,
                "jaccard": None, "dice": None, "overlap": None,
            }
            try:
                if cand.get("hypothesis_raw"):
                    cand_raw = cand["hypothesis_raw"]
                    cand_entry["hypothesis_nl"] = cand.get("hypothesis_nl") or adapter.raw_to_nl(cand_raw)
                else:
                    cand_conditions = parse_conditions_from_question(llm_model, cand_condition)
                    cand_fmt = format_conversion_tool(adapter=adapter, answer_nl=answer_nl, conditions=cand_conditions)
                    cand_gen = generate_hypothesis_tool(adapter, cand_fmt["model_input"]["source_text"])
                    cand_raw = cand_gen["raw_output"]
                    cand_entry["hypothesis_nl"] = cand_gen.get("query_nl")
                cand_metrics = compute_metrics(raw_output=cand_raw, label_answers=label_answers, graph_samplers=kg.graph_samplers)
                cand_jaccard = cand_metrics["jaccard"]
                if verbose:
                    print(f"  Jaccard: {cand_jaccard:.4f}")
                cand_entry.update({
                    "hypothesis_raw": cand_raw,
                    "jaccard": cand_jaccard,
                    "dice": cand_metrics["dice"],
                    "overlap": cand_metrics["overlap"],
                })
                if cand_jaccard >= jaccard_threshold:
                    early_stop = True
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
            evaluated_candidates.append(cand_entry)

        history[-1]["candidates"] = evaluated_candidates

        # ----- Round best: main hypothesis + candidates, jaccard first, dice as tiebreak -----
        all_this_round = [{"hypothesis_raw": raw_output, "hypothesis_nl": hypothesis_nl,
                           "jaccard": jaccard, "dice": metrics["dice"], "overlap": metrics["overlap"],
                           "condition": original_followup}]
        for ce in evaluated_candidates:
            if ce["jaccard"] is not None:
                all_this_round.append(ce)
        round_best = all_this_round[0]
        for entry in all_this_round[1:]:
            if _is_better(entry["jaccard"], entry["dice"] or 0, round_best["jaccard"], round_best["dice"] or 0):
                round_best = entry
        history[-1]["round_best"] = {
            "hypothesis_raw": round_best["hypothesis_raw"],
            "hypothesis_nl": round_best.get("hypothesis_nl"),
            "jaccard": round_best["jaccard"],
            "dice": round_best["dice"],
            "overlap": round_best.get("overlap"),
        }
        if verbose:
            print(f"\n[Round {round_idx} best] Jaccard={round_best['jaccard']:.4f}, raw={round_best['hypothesis_raw']}")

        if early_stop:
            if verbose:
                print(f"\n*** Candidate reached threshold {jaccard_threshold}. Stopping. ***")
            break

    # ----- Final summary -----
    def _best_key(h):
        rb = h.get("round_best") or h
        return (rb["jaccard"], rb.get("dice") or 0)

    best_round = max(history, key=_best_key)
    best = best_round.get("round_best") or best_round
    if verbose:
        print(f"\n{'='*60}")
        print("  LOOP SUMMARY")
        print(f"{'='*60}")
        for entry in history:
            rb = entry.get("round_best") or entry
            marker = " <-- BEST" if entry is best_round else ""
            print(f"  Round {entry['round']}: Jaccard={rb['jaccard']:.4f}{marker}")
        print(f"\n  Best Jaccard: {best['jaccard']:.4f} (Round {best_round['round']})")
        print(f"  Best hypothesis (raw): {best['hypothesis_raw']}")

    return history


if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["case", "run"], default="run")
    parser.add_argument("--dataname", default="BioKG")
    parser.add_argument("--checkpoint", default="checkpoints/BioKG-full-32-multi.pth")
    parser.add_argument("--data_root", default="./data/")
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--jaccard_threshold", type=float, default=0.95)
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

    def _save_result(log_path, case, history):
        def _rb(h):
            rb = h.get("round_best") or h
            return (rb["jaccard"], rb.get("dice") or 0)
        best_round = max(history, key=_rb)
        best = best_round.get("round_best") or best_round
        record = {
            "answers": case["answers"],
            "answers_nl": case.get("answers_nl", []),
            "query": case.get("query"),
            "user_condition": case["followup_question"],
            "rounds": [
                {
                    "round": h["round"],
                    "condition": h["condition"],
                    "parsed_conditions": h.get("parsed_conditions", []),
                    "hypothesis_raw": h.get("hypothesis_raw"),
                    "hypothesis_nl": h.get("hypothesis_nl"),
                    "jaccard": h.get("jaccard"),
                    "dice": h.get("dice"),
                    "overlap": h.get("metrics", {}).get("overlap"),
                    "candidates": h.get("candidates", []),
                }
                for h in history
            ],
            "best": {
                "hypothesis_raw": best.get("hypothesis_raw"),
                "hypothesis_nl": best.get("hypothesis_nl"),
                "jaccard": best.get("jaccard"),
                "dice": best.get("dice"),
                "overlap": best.get("overlap"),
            },
        }
        # Serialize compact fields inline, then re-embed via placeholder
        compact_keys = ("answers", "answers_nl", "query")
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

    if args.mode == "case":
        log_dir = os.path.join("log", args.dataname)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "singleturn.jsonl")
        history = run_loop(
            adapter=adapter,
            llm_model=llm_model,
            case=case_ip,
            max_rounds=args.max_rounds,
            jaccard_threshold=args.jaccard_threshold,
        )
        _save_result(log_path, case_ip, history)

    else:  # run
        from tqdm import tqdm
        data_file = os.path.join(args.data_root, args.dataname, "singleturn.jsonl")
        log_dir = os.path.join("log", args.dataname)
        os.makedirs(log_dir, exist_ok=True)
        model_tag = api_cfg["model_id"].split("/")[-1]
        log_path = os.path.join(log_dir, f"singleturn_{model_tag}.jsonl")

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

        # Build set of already-processed answers keys from existing log
        done_keys: set[str] = set()
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                _content = f.read()
            _dec = json.JSONDecoder()
            _pos = 0
            while _pos < len(_content):
                while _pos < len(_content) and _content[_pos].isspace():
                    _pos += 1
                if _pos >= len(_content):
                    break
                _obj, _pos = _dec.raw_decode(_content, _pos)
                if "answers" in _obj:
                    done_keys.add(json.dumps(_obj["answers"], ensure_ascii=False))

        for idx, case in enumerate(tqdm(cases[:args.limit], desc=args.dataname)):
            # Skip already processed cases
            case_key = json.dumps(case.get("answers", []), ensure_ascii=False)
            if case_key in done_keys:
                continue
            try:
                history = run_loop(
                    adapter=adapter,
                    llm_model=llm_model,
                    case=case,
                    max_rounds=args.max_rounds,
                    jaccard_threshold=args.jaccard_threshold,
                    verbose=False,
                )
                _save_result(log_path, case, history)
            except Exception as e:
                print(f"  [ERROR] sample {idx+1}: {e}")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"error": str(e), "answers": case.get("answers"), "user_condition": case.get("followup_question")}, ensure_ascii=False) + "\n")
