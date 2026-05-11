from __future__ import annotations
from typing import Any
from smolagents import Tool

# ---------------------------------------------------------------------------
# 1. load_kg tool
# ---------------------------------------------------------------------------

_kg_cache: dict[str, Any] = {}   # dataroot+dataname -> {"kg", "mapper"}


def load_kg_tool(data_root: str, dataname: str) -> dict:
    """
    Load a knowledge graph and build bidirectional name<->id mappings.

    Args:
        data_root: root directory of KG data
        dataname:  dataset name, e.g. "WN18RR", "DBpedia50", "FB15k-237"
        reverse_edges: whether to add inverse edges (doubles relation count)

    Returns:
        {
          "kg":           KG object,
          "mapper":       KGNameMapper (name<->id, fuzzy),
          "ent_id2name":  dict[int, str],
          "ent_name2id":  dict[str, int],
          "rel_id2name":  dict[int, str],
          "rel_name2id":  dict[str, int],
          "nentity":      int,
          "nrelation":    int,
        }
    """
    cache_key = f"{data_root}|{dataname}"
    if cache_key in _kg_cache:
        return _kg_cache[cache_key]

    from akgr.kgdata import load_kg
    from akgr.agent.kg_mapper import KGNameMapper

    kg = load_kg(data_root, dataname, reverse_edges_flag=False)
    mapper = KGNameMapper(kg)

    result = {
        "kg": kg,
        "mapper": mapper,
        "ent_id2name": kg.ent_id2name,
        "ent_name2id": mapper.ent_name2id,
        "rel_id2name": kg.rel_id2name,
        "rel_name2id": mapper.rel_name2id,
        "nentity": len(kg.ent_id2name),
        "nrelation": len(kg.rel_id2name),
    }
    _kg_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# 2. generate_hypothesis tool
# ---------------------------------------------------------------------------

def generate_hypothesis_tool(
    adapter,
    source_text: str,
    temperature: float = 1.0,
    top_k: int = 0,
) -> dict:
    return adapter.generate({"source_text": source_text}, temperature=temperature, top_k=top_k)


# ---------------------------------------------------------------------------
# 3. validate_hypothesis tool
# ---------------------------------------------------------------------------

def validate_hypothesis_tool(
    hypothesis_str: str,
    observation_entity_ids: list[int],
    condition_type: str,
    condition_value: Any,
    graph_samplers,                 # from kg.graph_samplers
    searching_split: str = "train",
    scoring_method: list[str] | None = None,
) -> dict:
    """
    Validate a hypothesis (unshifted id-form action string) against conditions
    and execute it on the KG graph.

    Args:
        hypothesis_str:         unshifted action string, e.g. "i -9 5530 -3 12"
        observation_entity_ids: list of raw (0-based) entity ids (the observations)
        condition_type:         condition type used during generation
        condition_value:        condition value used during generation
        graph_samplers:         kg.graph_samplers dict
        searching_split:        which graph split to execute on ("train"/"test"/...)
        scoring_method:         list of metrics to compute; defaults to
                                ["validity", "specific", "smatch"]

    Returns:
        {
          "valid":    bool,   True if hypothesis executes on the graph
          "scores":   dict,   metric name -> score
          "condition_type":  str,
          "condition_value": any,
        }
    """
    from akgr.evaluation import scoring_input_act_batch_condition

    if scoring_method is None:
        scoring_method = ["validity", "specific", "smatch"]

    # scoring function expects batch lists; wrap single sample
    ans_str = " ".join(str(e) for e in observation_entity_ids)
    condition_batch = [{"type": condition_type, "value": condition_value}]

    scores_list, _ = scoring_input_act_batch_condition(
        pred_word_batch=[hypothesis_str],
        label_word_batch=[hypothesis_str],   # no ground truth; self-compare for structure
        ans_word_batch=[ans_str],
        condition_batch=condition_batch,
        scoring_method=scoring_method,
        do_correction=False,
        graph_samplers=graph_samplers,
        searching_split=searching_split,
        return_failures=True,
        verbose=False,
    )

    scores = scores_list[0] if scores_list else {}
    valid = bool(scores.get("validity", 0) > 0)

    return {
        "valid": valid,
        "scores": scores,
        "condition_type": condition_type,
        "condition_value": condition_value,
    }


# ---------------------------------------------------------------------------
# smol-agent Tool wrappers
# ---------------------------------------------------------------------------

class LoadKGTool(Tool):
    name = "load_kg"
    description = (
        "Load a knowledge graph and return entity/relation name<->id mappings. "
        "Returns nentity, nrelation, and samples of ent_id2name / rel_id2name."
    )
    inputs = {
        "data_root":     {"type": "string",  "description": "Root directory of KG data"},
        "dataname":      {"type": "string",  "description": "Dataset name, e.g. WN18RR, DBpedia50, FB15k-237"},
    }
    output_type = "string"

    def forward(self, data_root: str, dataname: str) -> str:
        info = load_kg_tool(data_root, dataname)
        sample_ent = {k: info["ent_id2name"][k] for k in list(info["ent_id2name"])[:3]}
        sample_rel = {k: info["rel_id2name"][k] for k in list(info["rel_id2name"])[:3]}
        return (
            f"KG loaded: {dataname}\n"
            f"  nentity={info['nentity']}, nrelation={info['nrelation']}\n"
            f"  sample entities: {sample_ent}\n"
            f"  sample relations: {sample_rel}"
        )


class GenerateHypothesisTool(Tool):
    name = "generate_hypothesis"
    description = (
        "Generate a logical hypothesis by running the CtrlHGen model on a pre-built source_text. "
        "source_text is the tokenized model input returned by format_conversion "
        "(e.g. '2464 2579 SEP -8 2308'). Returns the action string and natural language description."
    )
    inputs = {
        "source_text": {"type": "string", "description": "Tokenized model input from format_conversion tool"},
    }
    output_type = "string"

    def __init__(self, adapter, **kwargs):
        super().__init__(**kwargs)
        self.adapter = adapter

    def forward(self, source_text: str) -> str:
        tokens = self.adapter.tokenizer.encode(source_text)
        print(f"[DEBUG] input tokens ({len(tokens)}): {tokens}")
        result = generate_hypothesis_tool(self.adapter, source_text)
        return (
            f"raw_output: {result['raw_output']}\n"
            f"query_nl: {result['query_nl']}\n"
            f"entitynumber: {result['entitynumber']}, relationnumber: {result['relationnumber']}"
        )


# ---------------------------------------------------------------------------
# 4. format_conversion tool
# ---------------------------------------------------------------------------

def format_conversion_tool(
    adapter,
    answer_nl: list[str],
    conditions: list[dict],
) -> dict:
    parsed_control = {
        "observation_entities": list(answer_nl),
        "conditions": conditions,
    }
    model_input = adapter.build_model_input(parsed_control)
    return {
        "parsed": parsed_control,
        "model_input": model_input,
    }


class FormatConversionTool(Tool):
    name = "format_conversion"
    description = (
        "Build tokenized model input from observation entities and structured conditions. "
        "conditions_json must be a JSON string of an array of {type, value} dicts. "
        "Valid types and value formats:\n"
        "  relation: specific relation name, e.g. {\"type\":\"relation\",\"value\":\"relation_name\"}\n"
        "  entity: specific entity name, e.g. {\"type\":\"entity\",\"value\":\"entity_name\"}\n"
        "  relationnumber: count of relations (integer string), e.g. {\"type\":\"relationnumber\",\"value\":\"3\"}\n"
        "  entitynumber: count of entities (integer string), e.g. {\"type\":\"entitynumber\",\"value\":\"2\"}\n"
        "  pattern: pattern string, e.g. {\"type\":\"pattern\",\"value\":\"i p p e p e\"}\n"
        "  unconditional: no condition, e.g. {\"type\":\"unconditional\",\"value\":\"\"}\n"
        "For multi-condition pass multiple dicts: "
        "'[{\"type\":\"relation\",\"value\":\"E\"},{\"type\":\"entitynumber\",\"value\":\"2\"}]'"
    )
    inputs = {
        "answer_nl":       {"type": "string", "description": "Comma-separated observation entity names"},
        "conditions_json": {"type": "string", "description": 'JSON array of condition dicts, e.g. [{"type":"relation","value":"E"}]'},
    }
    output_type = "string"

    def __init__(self, adapter, **kwargs):
        super().__init__(**kwargs)
        self.adapter = adapter

    def forward(self, answer_nl: str, conditions_json: str) -> str:
        import json
        entities = [e.strip() for e in answer_nl.split(",") if e.strip()]
        conditions = json.loads(conditions_json)
        result = format_conversion_tool(
            adapter=self.adapter,
            answer_nl=entities,
            conditions=conditions,
        )
        mi = result["model_input"]
        import json
        return json.dumps({
            "source_text": mi["source_text"],
            "observation_entity_ids": mi["observation_entity_ids"],
            "conditions": mi["conditions"],
        })


class ValidateHypothesisTool(Tool):
    name = "validate_hypothesis"
    description = (
        "Validate a hypothesis (unshifted action string) against conditions and execute it on the KG graph. "
        "Returns validity and scores."
    )
    inputs = {
        "hypothesis_str":         {"type": "string", "description": "Unshifted action string, e.g. 'i -9 5530 -3 12'"},
        "observation_entity_ids": {"type": "string", "description": "Comma-separated raw entity ids"},
        "condition_type":         {"type": "string", "description": "Condition type used during generation"},
        "condition_value":        {"type": "string", "description": "Condition value used during generation", "nullable": True},
    }
    output_type = "string"

    def __init__(self, graph_samplers, searching_split: str = "train", **kwargs):
        super().__init__(**kwargs)
        self.graph_samplers = graph_samplers
        self.searching_split = searching_split

    def forward(self, hypothesis_str: str, observation_entity_ids: str,
                condition_type: str, condition_value: str = None) -> str:
        ids = [int(x.strip()) for x in observation_entity_ids.split(",") if x.strip()]
        result = validate_hypothesis_tool(
            hypothesis_str, ids, condition_type, condition_value,
            self.graph_samplers, self.searching_split,
        )
        return f"valid={result['valid']}, scores={result['scores']}"


def compute_metrics(
    raw_output: str,
    label_answers: list[int],
    graph_samplers,
    searching_split: str = "test",
) -> dict:
    from akgr.utils.parsing_util import qry_actionstr_2_wordlist
    pred_qry = qry_actionstr_2_wordlist(raw_output)
    pred_ans = graph_samplers[searching_split].search_answers_to_query(pred_qry)
    label_ans = label_answers  # already raw 0-indexed

    pred_set = set(pred_ans)
    label_set = set(label_ans)
    I = len(pred_set & label_set)
    U = len(pred_set | label_set)
    A, B = len(label_set), len(pred_set)

    return {
        "jaccard":  I / U if U > 0 else 0.0,
        "dice":     2. * I / (A + B) if (A + B) > 0 else 0.0,
        "overlap":  I / (min(A, B) + 1e-5),
        "pred_answers": list(pred_set),
        "label_answers": list(label_set),
    }


class MetricTool(Tool):
    name = "compute_metrics"
    description = (
        "Execute the generated hypothesis on the KG and compute Jaccard, Dice, and Overlap "
        "against the ground-truth answer entities. "
        "raw_output is the unshifted action string from generate_hypothesis. "
        "label_answers is a comma-separated list of raw entity IDs (ground truth)."
    )
    inputs = {
        "raw_output":     {"type": "string", "description": "Unshifted action string, e.g. 'i -8 5530 -3 12'"},
        "label_answers":  {"type": "string", "description": "Comma-separated ground-truth entity IDs, e.g. '5828,5001,5066'"},
    }
    output_type = "string"

    def __init__(self, graph_samplers, searching_split: str = "train", **kwargs):
        super().__init__(**kwargs)
        self.graph_samplers = graph_samplers
        self.searching_split = searching_split

    def forward(self, raw_output: str, label_answers: str) -> str:
        import json
        ids = [int(x.strip()) for x in label_answers.split(",") if x.strip()]
        result = compute_metrics(raw_output, ids, self.graph_samplers, self.searching_split)
        return json.dumps(result)


# ---------------------------------------------------------------------------
# 5. query_translation tool
# ---------------------------------------------------------------------------

def _get_direct_child(tokens: list):
    """Return the first sub-expression child (depth-2) of a p/n node, or None."""
    depth = 0
    start = None
    for idx, t in enumerate(tokens):
        if t == '(':
            depth += 1
            if depth == 2:
                start = idx
        elif t == ')':
            depth -= 1
            if depth == 1 and start is not None:
                candidate = tokens[start:idx+1]
                if len(candidate) > 1 and isinstance(candidate[1], str):
                    return candidate
                start = None
    return None


def _split_top_level(tokens: list) -> list[list]:
    """
    If the top-level operator is i/u, return the 2 direct children.
    Otherwise return [] (no split needed for p/n/e).
    """
    if not tokens or tokens[0] != '(':
        return []
    op = tokens[1]
    if op not in ('i', 'u'):
        return []
    # collect direct children at depth==2
    depth = 0
    start = None
    children = []
    for idx, t in enumerate(tokens):
        if t == '(':
            depth += 1
            if depth == 2:
                start = idx
        elif t == ')':
            depth -= 1
            if depth == 1 and start is not None:
                children.append(tokens[start:idx+1])
                start = None
    return children


def _collect_leaf_subqueries(tokens: list) -> list[list]:
    """
    Recursively collect all leaf (p,(e)) sub-queries from an i/u/n tree.
    """
    if not tokens or tokens[0] != '(':
        return []
    op = tokens[1]
    if op == 'p':
        return [tokens]
    if op in ('i', 'u'):
        leaves = []
        for child in _split_top_level(tokens):
            leaves.extend(_collect_leaf_subqueries(child))
        return leaves
    if op == 'n':
        # recurse into the single child of n
        depth = 0
        start = None
        for idx, t in enumerate(tokens):
            if t == '(':
                depth += 1
                if depth == 2:
                    start = idx
            elif t == ')':
                depth -= 1
                if depth == 1 and start is not None:
                    return _collect_leaf_subqueries(tokens[start:idx+1])
    return []


def _decompose_chain(tokens: list) -> list[list]:
    """
    For a chain query (p,(p,...,(e)...)), return each prefix sub-chain as a sub-query.
    E.g. (p,(p,(e))) -> [(p,(e)), (p,(p,(e)))]
    Also handles mixed heads: (p,(i/u,...)) by returning the inner i/u as a sub-query.
    """
    if not tokens or tokens[0] != '(':
        return []
    op = tokens[1]
    if op != 'p':
        return []
    # find the sub-query child: the depth-2 child whose first token is '('
    # (skip relation/entity scalar children like (-20) or (1305))
    depth = 0
    start = None
    child = None
    for idx, t in enumerate(tokens):
        if t == '(':
            depth += 1
            if depth == 2:
                start = idx
        elif t == ')':
            depth -= 1
            if depth == 1 and start is not None:
                candidate = tokens[start:idx+1]
                # a sub-query child starts with '(' and has a string op token
                if len(candidate) > 1 and isinstance(candidate[1], str):
                    child = candidate
                    break
                start = None
    if child is None:
        return []
    child_op = child[1] if len(child) > 1 else None
    if child_op == 'e':
        # base case: (p,(e)) — no further decomposition
        return []
    if child_op == 'p':
        # chain: recurse into child, then add child itself
        inner = _decompose_chain(child)
        return inner + [child]
    if child_op in ('i', 'u', 'n'):
        # mixed (ip/up/inp): return outer-p applied to each leaf only
        leaves = _collect_leaf_subqueries(child)
        outer_rel = None
        depth = 0
        for t in tokens:
            if t == '(':
                depth += 1
            elif t == ')':
                depth -= 1
            elif depth == 2 and not isinstance(t, str):
                outer_rel = t
                break
        if outer_rel is None:
            return []
        return [['(', 'p', '(', outer_rel, ')'] + leaf + [')'] for leaf in leaves]
    return []


def _enumerate_subquery_combinations(tokens: list) -> list[list]:
    """
    Return all meaningful sub-queries for graph validation:
    - i/u queries: individual leaf (p,(e)) + all C(n,2) pairs as 2i
    - p-chain queries (2p, 3p): each intermediate hop
    - mixed queries (ip, pi, inp, pni, up, etc.): inner sub-queries + leaves
    """
    if not tokens or tokens[0] != '(':
        return []
    op = tokens[1]

    if op in ('i', 'u'):
        leaves = _collect_leaf_subqueries(tokens)
        result = list(leaves)
        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                combined = ['(', op] + leaves[i] + leaves[j] + [')']
                result.append(combined)
        # also decompose any non-leaf children (e.g. pi has a (p,(p,(e))) child)
        for child in _split_top_level(tokens):
            if child[1] == 'p':
                result.extend(_decompose_chain(child))
            elif child[1] == 'n':
                # negated child: add the n node itself + its inner positive sub-query
                result.append(child)
                inner_children = _split_top_level(child) if len(child) > 2 else []
                result.extend(inner_children)
        return result

    if op == 'p':
        # chain or mixed (ip, inp, etc.)
        return _decompose_chain(tokens)

    return []


def query_translation_tool(
    query: list,
    ent_id2name: dict,
    rel_id2name: dict,
) -> dict:
    """
    Translate a query token list to natural language.
    If top-level is i/u, split into 2 sub-query NL descriptions.
    """
    from akgr.agent.getsomesampleFromDB import query_to_natural_language

    nl = query_to_natural_language(query, ent_id2name, rel_id2name)
    sub_nls = []
    for sq in _split_top_level(query):
        try:
            snl = query_to_natural_language(sq, ent_id2name, rel_id2name)
            if snl:
                sub_nls.append(snl)
        except Exception:
            pass
    return {"nl": nl, "sub_queries_nl": sub_nls}


class QueryTranslationTool(Tool):
    name = "query_translation"
    description = (
        "Translate a query (token list or action string) into natural language "
        "and decompose it into sub-query NL descriptions."
    )
    inputs = {
        "query_tokens": {
            "type": "string",
            "description": (
                "Space-separated query token list, e.g. '( i ( p ( -8 ) ( e ( 1128 ) ) ) ( p ( -21 ) ( e ( 4922 ) ) ) )'"
            ),
        },
    }
    output_type = "string"

    def __init__(self, kg, **kwargs):
        super().__init__(**kwargs)
        self.kg = kg

    def forward(self, query_tokens: str) -> str:
        import json
        # parse space-separated tokens, converting numeric strings to int
        raw = query_tokens.strip().split()
        tokens = []
        for t in raw:
            try:
                tokens.append(int(t))
            except ValueError:
                tokens.append(t)
        result = query_translation_tool(tokens, self.kg.ent_id2name, self.kg.rel_id2name)
        return json.dumps(result)


# ---------------------------------------------------------------------------
# 6. graph_validation tool
# ---------------------------------------------------------------------------

def _set_relation(result_set: set, label_set: set) -> str:
    """Describe the set relationship between result and label answers."""
    if not result_set and not label_set:
        return "both_empty"
    if not result_set:
        return "result_empty"
    if not label_set:
        return "label_empty"
    if result_set == label_set:
        return "exact_match"
    if result_set >= label_set:
        return "result_contains_label"
    if result_set <= label_set:
        return "label_contains_result"
    if result_set & label_set:
        return "partial_overlap"
    return "disjoint"


def graph_validation_tool(
    query: list,
    graph_sampler,
    label_answers: list[int] | None = None,
) -> dict:
    """
    Execute a query on the KG and return the answer entities.
    If top-level is i/u, split into 2 sub-queries and validate each.
    Reports set relationship against label_answers for each.
    """
    answers = graph_sampler.search_answers_to_query(query)
    label_set = set(label_answers) if label_answers else set()

    def _build_result(ans_list, sub_query=None):
        s = set(ans_list)
        out = {
            "answer_count": len(s),
            "answers": list(s)[:20],
        }
        if sub_query is not None:
            out["sub_query"] = sub_query
        if label_answers is not None:
            out["relation_to_label"] = _set_relation(s, label_set)
            out["overlap_count"] = len(s & label_set)
        return out

    main_result = _build_result(answers)
    main_result["valid"] = len(answers) > 0

    sub_results = []
    for sq in _enumerate_subquery_combinations(query):
        try:
            sub_ans = graph_sampler.search_answers_to_query(sq)
            entry = _build_result(sub_ans, sub_query=sq)
            # For p-headed sub-queries whose direct child is an i-node (ip/inp/up patterns),
            # also execute the inner i-sublogic to check if its conclusion set is empty.
            # e.g. (p,(i,(p,(e)),(p,(e)))) -> check (i,(p,(e)),(p,(e)))
            if len(sq) > 1 and sq[1] == 'p':
                inner = _get_direct_child(sq)
                if inner is not None and len(inner) > 1 and inner[1] == 'i':
                    try:
                        i_ans = graph_sampler.search_answers_to_query(inner)
                        entry["inner_i_sublogic"] = {
                            "sub_query": inner,
                            "answer_count": len(i_ans),
                            "empty": len(i_ans) == 0,
                        }
                    except Exception:
                        entry["inner_i_sublogic"] = {"empty": None}
            sub_results.append(entry)
        except Exception as e:
            sub_results.append({"sub_query": sq, "error": str(e)})

    main_result["sub_query_results"] = sub_results
    return main_result


class GraphValidationTool(Tool):
    name = "graph_validation"
    description = (
        "Execute a query (token list) on the KG graph and validate it. "
        "For i/u queries, recursively collects all leaf (p,(e)) sub-queries and enumerates "
        "all individual leaves plus all C(n,2) pairs as 2i combinations. "
        "Reports answer count and set relationship against label answers for each."
    )
    inputs = {
        "query_tokens": {
            "type": "string",
            "description": "Space-separated query token list (same format as query_translation)",
        },
        "label_answers": {
            "type": "string",
            "description": "Comma-separated ground-truth entity IDs (raw 0-indexed). Optional.",
            "nullable": True,
        },
        "split": {
            "type": "string",
            "description": "Graph split to search on: train, valid, or test (default: train)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, kg, **kwargs):
        super().__init__(**kwargs)
        self.kg = kg

    def forward(self, query_tokens: str, label_answers: str = None, split: str = "train") -> str:
        import json
        from akgr.utils.parsing_util import qry_actionstr_2_wordlist
        raw = query_tokens.strip().split()
        # Try to detect if input is a flat action string (no parentheses) and convert
        if '(' not in query_tokens:
            tokens = qry_actionstr_2_wordlist(query_tokens.strip())
            if tokens is None:
                return json.dumps({"error": "Could not parse query_tokens as flat action string"})
        else:
            tokens = []
            for t in raw:
                try:
                    tokens.append(int(t))
                except ValueError:
                    tokens.append(t)
        label_ids = None
        if label_answers:
            label_ids = [int(x.strip()) for x in label_answers.split(",") if x.strip()]
        graph_sampler = self.kg.graph_samplers[split]
        result = graph_validation_tool(tokens, graph_sampler, label_ids)
        # Add NL description for each sub_query
        from akgr.agent.getsomesampleFromDB import query_to_natural_language
        for entry in result.get("sub_query_results", []):
            sq = entry.get("sub_query")
            if sq:
                try:
                    entry["sub_query_nl"] = query_to_natural_language(sq, self.kg.ent_id2name, self.kg.rel_id2name)
                except Exception:
                    pass
        return json.dumps(result)


# ---------------------------------------------------------------------------
# 7. incoming_edge_intersection tool
# ---------------------------------------------------------------------------

def incoming_edge_intersection_tool(
    answer_entity_ids: list[int],
    graph_sampler,
    ent_id2name: dict,
    rel_id2name: dict,
    top_k: int = 10,
) -> dict:
    """
    For each entity in answer_entity_ids, collect all incoming triples (head, rel, entity).
    Then intersect the head entities across all answer entities to find common sources.
    Returns the intersection with relation info as hints.
    """
    if not answer_entity_ids:
        return {"intersection": [], "hints": []}

    # For each answer entity, collect set of (head, rel) pairs
    per_entity_heads: list[set] = []
    per_entity_triples: list[list] = []

    for eid in answer_entity_ids:
        in_edges = list(graph_sampler.in_edges(eid))
        heads = set()
        triples = []
        for (head, _, rel) in in_edges:
            heads.add(head)
            triples.append((head, rel, eid))
        per_entity_heads.append(heads)
        per_entity_triples.append(triples)

    # Intersect head entity sets
    if not per_entity_heads:
        return {"intersection": [], "hints": []}

    common_heads = per_entity_heads[0]
    for s in per_entity_heads[1:]:
        common_heads = common_heads & s

    # Build hints: for each common head, collect relations + execute (p,(e(head))) per relation
    obs_set = set(answer_entity_ids)
    hints = []
    for head in list(common_heads)[:top_k]:
        rels = set()
        for triples in per_entity_triples:
            for (h, r, _) in triples:
                if h == head:
                    rels.add(r)
        head_name = ent_id2name.get(head, str(head))
        rel_info = []
        for r in rels:
            # query: entities reachable from head via relation r (outgoing from head)
            try:
                pred_set = set(graph_sampler.search_answers_to_query(["(", "p", "(", -r, ")", "(", "e", "(", head, ")", ")", ")"]))
            except Exception:
                pred_set = set()
            overlap = len(pred_set & obs_set)
            jaccard = overlap / len(pred_set | obs_set) if (pred_set | obs_set) else 0.0
            rel_info.append({
                "name": rel_id2name.get(r, str(r)),
                "id": r,
                "pred_count": len(pred_set),
                "jaccard": round(jaccard, 4),
            })
        # sort by jaccard descending
        rel_info.sort(key=lambda x: x["jaccard"], reverse=True)
        hints.append({"head_entity": head_name, "head_id": head, "relations": rel_info})

    # Flatten all (head, relation) pairs sorted by jaccard
    flat = []
    for h in hints:
        for rel in h["relations"]:
            flat.append({
                "entity": h["head_entity"],
                "entity_id": h["head_id"],
                "relation": rel["name"],
                "relation_id": rel["id"],
                "pred_count": rel["pred_count"],
                "jaccard": rel["jaccard"],
            })
    flat.sort(key=lambda x: x["jaccard"], reverse=True)

    # --- 2-hop detection ---
    # Step 1: find rel1 that exists as in-edge for EVERY observation entity
    from collections import defaultdict
    two_hop_candidates = []

    # per_entity_triples[i] = list of (head, rel, obs_i)
    # rel1 must appear in every observation's in-edges
    rel1_per_obs: list[set] = [set(r for (_, r, _) in triples) for triples in per_entity_triples]
    if not rel1_per_obs:
        pass
    else:
        valid_rel1s = rel1_per_obs[0]
        for s in rel1_per_obs[1:]:
            valid_rel1s = valid_rel1s & s

        for rel1_id in valid_rel1s:
            # Step 2: per-observation intermediates (rel1 heads for each obs)
            obs_intermediates: list[set] = []
            for triples in per_entity_triples:
                mids = set(h for (h, r, _) in triples if r == rel1_id)
                obs_intermediates.append(mids)

            all_intermediates: set = set()
            for s in obs_intermediates:
                all_intermediates |= s
            if len(all_intermediates) < 2:
                continue

            # Step 3: find rel2 reachable for EVERY observation
            # i.e. for each obs, at least one of its intermediates has rel2 as in-edge
            mid_to_in_edges: dict[int, list] = {}
            for mid in all_intermediates:
                try:
                    mid_to_in_edges[mid] = list(graph_sampler.in_edges(mid))
                except Exception:
                    mid_to_in_edges[mid] = []

            # rel2 sets per observation
            rel2_per_obs: list[set] = []
            for mids in obs_intermediates:
                rel2s = set()
                for mid in mids:
                    for (_, _, r) in mid_to_in_edges.get(mid, []):
                        rel2s.add(r)
                rel2_per_obs.append(rel2s)

            valid_rel2s = rel2_per_obs[0]
            for s in rel2_per_obs[1:]:
                valid_rel2s = valid_rel2s & s

            # Step 4: for each (rel1, rel2), enumerate all head2 candidates and score
            seen: set = set()
            for rel2_id in valid_rel2s:
                head2_candidates: set = set()
                for mid, edges in mid_to_in_edges.items():
                    for (h, _, r) in edges:
                        if r == rel2_id:
                            head2_candidates.add(h)

                for head2 in head2_candidates:
                    key = (rel1_id, rel2_id, head2)
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        pred_set = set(graph_sampler.search_answers_to_query([
                            "(", "p", "(", -rel1_id, ")",
                            "(", "p", "(", -rel2_id, ")",
                            "(", "e", "(", head2, ")", ")", ")", ")"
                        ]))
                    except Exception:
                        pred_set = set()
                    overlap = len(pred_set & obs_set)
                    jaccard = overlap / len(pred_set | obs_set) if (pred_set | obs_set) else 0.0
                    if jaccard > 0:
                        two_hop_candidates.append({
                            "hop1_relation": rel_id2name.get(rel1_id, str(rel1_id)),
                            "hop1_relation_id": rel1_id,
                            "hop2_relation": rel_id2name.get(rel2_id, str(rel2_id)),
                            "hop2_relation_id": rel2_id,
                            "anchor_entity": ent_id2name.get(head2, str(head2)),
                            "anchor_entity_id": head2,
                            "intermediate_count": len(all_intermediates),
                            "pred_count": len(pred_set),
                            "jaccard": round(jaccard, 4),
                        })

    two_hop_candidates.sort(key=lambda x: x["jaccard"], reverse=True)

    return {
        "intersection_count": len(common_heads),
        "hints": hints,
        "flat_candidates": flat,
        "two_hop_candidates": two_hop_candidates[:top_k],
    }


def compute_2i_candidates(
    flat_candidates: list[dict],
    obs_set: set,
    graph_sampler,
    top_k: int = 10,
) -> list[dict]:
    """
    Enumerate pairs from flat_candidates, execute 2i query, return top_k by jaccard.
    Only considers top-20 singles to avoid combinatorial explosion (C(20,2)=190).
    """
    pool = flat_candidates[:20]
    results = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            a, b = pool[i], pool[j]
            query = [
                "(", "i",
                "(", "p", "(", -a["relation_id"], ")", "(", "e", "(", a["entity_id"], ")", ")", ")",
                "(", "p", "(", -b["relation_id"], ")", "(", "e", "(", b["entity_id"], ")", ")", ")",
                ")"
            ]
            try:
                pred_set = set(graph_sampler.search_answers_to_query(query))
            except Exception:
                pred_set = set()
            union = pred_set | obs_set
            jaccard = len(pred_set & obs_set) / len(union) if union else 0.0
            results.append({
                "cond_a": {"entity": a["entity"], "relation": a["relation"]},
                "cond_b": {"entity": b["entity"], "relation": b["relation"]},
                "pred_count": len(pred_set),
                "jaccard": round(jaccard, 4),
            })
    results.sort(key=lambda x: x["jaccard"], reverse=True)
    return results[:top_k]


def compute_2u_candidates(
    flat_candidates: list[dict],
    obs_set: set,
    graph_sampler,
    top_k: int = 10,
) -> list[dict]:
    """
    Enumerate pairs from flat_candidates, execute 2u query, return top_k by jaccard.
    2u: union of two 1p queries.
    """
    pool = flat_candidates[:20]
    results = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            a, b = pool[i], pool[j]
            query = [
                "(", "u",
                "(", "p", "(", -a["relation_id"], ")", "(", "e", "(", a["entity_id"], ")", ")", ")",
                "(", "p", "(", -b["relation_id"], ")", "(", "e", "(", b["entity_id"], ")", ")", ")",
                ")"
            ]
            try:
                pred_set = set(graph_sampler.search_answers_to_query(query))
            except Exception:
                pred_set = set()
            union = pred_set | obs_set
            jaccard = len(pred_set & obs_set) / len(union) if union else 0.0
            results.append({
                "cond_a": {"entity": a["entity"], "relation": a["relation"]},
                "cond_b": {"entity": b["entity"], "relation": b["relation"]},
                "pred_count": len(pred_set),
                "jaccard": round(jaccard, 4),
            })
    results.sort(key=lambda x: x["jaccard"], reverse=True)
    return results[:top_k]


def compute_3i_candidates(
    flat_candidates: list[dict],
    obs_set: set,
    graph_sampler,
    top_k: int = 10,
) -> list[dict]:
    """
    Enumerate triples from flat_candidates, execute 3i query, return top_k by jaccard.
    Only considers top-10 singles to avoid explosion (C(10,3)=120).
    """
    pool = flat_candidates[:10]
    results = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            for k in range(j + 1, len(pool)):
                a, b, c = pool[i], pool[j], pool[k]
                query = [
                    "(", "i",
                    "(", "p", "(", -a["relation_id"], ")", "(", "e", "(", a["entity_id"], ")", ")", ")",
                    "(", "p", "(", -b["relation_id"], ")", "(", "e", "(", b["entity_id"], ")", ")", ")",
                    "(", "p", "(", -c["relation_id"], ")", "(", "e", "(", c["entity_id"], ")", ")", ")",
                    ")"
                ]
                try:
                    pred_set = set(graph_sampler.search_answers_to_query(query))
                except Exception:
                    pred_set = set()
                union = pred_set | obs_set
                jaccard = len(pred_set & obs_set) / len(union) if union else 0.0
                results.append({
                    "cond_a": {"entity": a["entity"], "relation": a["relation"]},
                    "cond_b": {"entity": b["entity"], "relation": b["relation"]},
                    "cond_c": {"entity": c["entity"], "relation": c["relation"]},
                    "pred_count": len(pred_set),
                    "jaccard": round(jaccard, 4),
                })
    results.sort(key=lambda x: x["jaccard"], reverse=True)
    return results[:top_k]


class IncomingEdgeIntersectionTool(Tool):
    name = "incoming_edge_intersection"
    description = (
        "For each observation (answer) entity, find all incoming triples in the KG "
        "(triples where the entity is the tail). Then intersect the head entities across "
        "all answer entities to find common sources. Returns intersection hints to guide "
        "hypothesis generation."
    )
    inputs = {
        "answer_entity_ids": {
            "type": "string",
            "description": "Comma-separated raw (0-based) entity IDs of the observations",
        },
        "split": {
            "type": "string",
            "description": "Graph split: train, valid, or test (default: train)",
            "nullable": True,
        },
        "top_k": {
            "type": "integer",
            "description": "Max number of intersection results to return (default: 10)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, kg, **kwargs):
        super().__init__(**kwargs)
        self.kg = kg

    def forward(self, answer_entity_ids: str, split: str = "test", top_k: int = 10) -> str:
        import json
        ids = [int(x.strip()) for x in answer_entity_ids.split(",") if x.strip()]
        graph_sampler = self.kg.graph_samplers[split]
        result = incoming_edge_intersection_tool(
            ids, graph_sampler, self.kg.ent_id2name, self.kg.rel_id2name, top_k=top_k
        )
        return json.dumps(result)


# ---------------------------------------------------------------------------
# 8. execute_and_diagnose tool
# ---------------------------------------------------------------------------

def execute_and_diagnose_tool(
    raw_output: str,
    observation_ids: list[int],
    graph_sampler,
) -> dict:
    """
    Execute hypothesis on KG, compute TP/FP/FN against observations O.
    Returns diagnosis: too_broad (many FP), too_narrow (many FN), wrong_predicates (both).
    """
    from akgr.utils.parsing_util import qry_actionstr_2_wordlist
    pred_qry = qry_actionstr_2_wordlist(raw_output)
    try:
        pred_ans = set(graph_sampler.search_answers_to_query(pred_qry))
    except Exception:
        pred_ans = set()
    obs_set = set(observation_ids)

    tp = pred_ans & obs_set
    fp = pred_ans - obs_set
    fn = obs_set - pred_ans

    if len(fp) > len(fn) * 2:
        diagnosis = "too_broad"
    elif len(fn) > len(fp) * 2:
        diagnosis = "too_narrow"
    elif fp and fn:
        diagnosis = "wrong_predicates"
    else:
        diagnosis = "good"

    precision = len(tp) / len(pred_ans) if pred_ans else 0.0
    recall = len(tp) / len(obs_set) if obs_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "pred_answers": list(pred_ans),
        "tp": list(tp), "fp": list(fp), "fn": list(fn),
        "tp_count": len(tp), "fp_count": len(fp), "fn_count": len(fn),
        "precision": precision, "recall": recall, "f1": f1,
        "diagnosis": diagnosis,
    }


class ExecuteAndDiagnoseTool(Tool):
    name = "execute_and_diagnose"
    description = (
        "Execute a hypothesis (unshifted action string) on the KG and diagnose errors "
        "by comparing predicted answers against observation entities O. "
        "Returns TP/FP/FN counts and diagnosis: too_broad, too_narrow, wrong_predicates, or good."
    )
    inputs = {
        "raw_output": {"type": "string", "description": "Unshifted action string, e.g. 'i -8 5530 -3 12'"},
        "observation_ids": {"type": "string", "description": "Comma-separated observation entity IDs (ground truth O)"},
        "split": {"type": "string", "description": "Graph split: train/test (default: train)", "nullable": True},
    }
    output_type = "string"

    def __init__(self, kg, **kwargs):
        super().__init__(**kwargs)
        self.kg = kg

    def forward(self, raw_output: str, observation_ids: str, split: str = "train") -> str:
        import json
        obs = [int(x.strip()) for x in observation_ids.split(",") if x.strip()]
        result = execute_and_diagnose_tool(raw_output, obs, self.kg.graph_samplers[split])
        return json.dumps(result)


# ---------------------------------------------------------------------------
# 9. neighborhood_candidates tool
# ---------------------------------------------------------------------------

def neighborhood_candidates_tool(
    entity_ids: list[int],
    graph_sampler,
    ent_id2name: dict,
    rel_id2name: dict,
    fp_ids: list[int] | None = None,
    top_k: int = 15,
) -> dict:
    """
    Collect 1-hop and 2-hop (p,p,e) neighbors of entity_ids, score candidates
    by coverage over entity_ids minus coverage over fp_ids.
    Returns top_k relation candidates, entity candidates, and two_hop_candidates.
    """
    from collections import defaultdict

    obs_set = set(entity_ids)
    fp_set = set(fp_ids) if fp_ids else set()

    rel_obs: dict[int, set] = defaultdict(set)
    rel_fp: dict[int, set] = defaultdict(set)
    ent_obs: dict[int, set] = defaultdict(set)
    ent_fp: dict[int, set] = defaultdict(set)

    # 1-hop: collect in-edges and out-edge relations
    for eid in entity_ids:
        for (head, _, rel) in graph_sampler.in_edges(eid):
            rel_obs[rel].add(eid)
            ent_obs[head].add(eid)
        for (_, _, rel) in graph_sampler.out_edges(eid):
            rel_obs[rel].add(eid)

    for eid in fp_set:
        for (head, _, rel) in graph_sampler.in_edges(eid):
            rel_fp[rel].add(eid)
            ent_fp[head].add(eid)
        for (_, _, rel) in graph_sampler.out_edges(eid):
            rel_fp[rel].add(eid)

    def score(cov_obs, cov_fp):
        return len(cov_obs) / max(len(obs_set), 1) - len(cov_fp) / max(len(fp_set), 1)

    rel_scores = {r: score(rel_obs[r], rel_fp.get(r, set())) for r in rel_obs}
    ent_scores = {e: score(ent_obs[e], ent_fp.get(e, set())) for e in ent_obs if e not in obs_set}

    top_rels = sorted(rel_scores, key=rel_scores.__getitem__, reverse=True)[:top_k]
    top_ents = sorted(ent_scores, key=ent_scores.__getitem__, reverse=True)[:top_k]

    # 2-hop: enumerate (rel1, rel2, anchor) paths via intermediates
    # For each obs entity, collect (intermediate, rel1) from in-edges
    # Then for each intermediate, collect (anchor, rel2) from its in-edges
    two_hop: dict[tuple, dict] = {}  # (rel1, rel2, anchor) -> {obs_covered, fp_covered}

    mid_rel1_per_obs: list[dict] = []  # per obs: {mid: set of rel1}
    for eid in entity_ids:
        d: dict[int, set] = defaultdict(set)
        for (mid, _, rel1) in graph_sampler.in_edges(eid):
            d[mid].add(rel1)
        mid_rel1_per_obs.append(d)

    # Collect all intermediates across all obs
    all_mids: set = set()
    for d in mid_rel1_per_obs:
        all_mids |= set(d.keys())

    # For each mid, get its in-edges (anchor, rel2)
    mid_in: dict[int, list] = {}
    for mid in all_mids:
        try:
            mid_in[mid] = list(graph_sampler.in_edges(mid))
        except Exception:
            mid_in[mid] = []

    # Score 2-hop paths over obs entities
    for i, eid in enumerate(entity_ids):
        for mid, rel1_set in mid_rel1_per_obs[i].items():
            for (anchor, _, rel2) in mid_in.get(mid, []):
                if anchor in obs_set:
                    continue
                for rel1 in rel1_set:
                    key = (rel1, rel2, anchor)
                    if key not in two_hop:
                        two_hop[key] = {"obs": set(), "fp": set()}
                    two_hop[key]["obs"].add(eid)

    # Score 2-hop paths over fp entities
    mid_rel1_per_fp: list[dict] = []
    for eid in fp_set:
        d: dict[int, set] = defaultdict(set)
        for (mid, _, rel1) in graph_sampler.in_edges(eid):
            d[mid].add(rel1)
        mid_rel1_per_fp.append((eid, d))

    for eid, d in mid_rel1_per_fp:
        for mid, rel1_set in d.items():
            for (anchor, _, rel2) in mid_in.get(mid, []):
                for rel1 in rel1_set:
                    key = (rel1, rel2, anchor)
                    if key in two_hop:
                        two_hop[key]["fp"].add(eid)

    def _jaccard(pred_set, obs_set):
        u = pred_set | obs_set
        return len(pred_set & obs_set) / len(u) if u else 0.0

    # Compute jaccard for top relation candidates: p(rel, e(obs_entity)) for each obs entity
    # Use the best single-entity anchor from ent_obs for each relation
    rel_candidate_list = []
    for r in top_rels:
        best_j = 0.0
        best_anchor_id = None
        for e in top_ents:
            try:
                pred = set(graph_sampler.search_answers_to_query(
                    ["(", "p", "(", -r, ")", "(", "e", "(", e, ")", ")", ")"]
                ))
                j = _jaccard(pred, obs_set)
                if j > best_j:
                    best_j = j
                    best_anchor_id = e
            except Exception:
                pass
        rel_candidate_list.append({
            "id": r, "name": rel_id2name.get(r, str(r)),
            "score": round(rel_scores[r], 4),
            "best_jaccard": round(best_j, 4),
            "best_anchor_id": best_anchor_id,
        })

    # Compute jaccard for top entity candidates: p(rel, e(anchor)) for each top relation
    ent_candidate_list = []
    for e in top_ents:
        best_j = 0.0
        best_rel_id = None
        for r in top_rels:
            try:
                pred = set(graph_sampler.search_answers_to_query(
                    ["(", "p", "(", -r, ")", "(", "e", "(", e, ")", ")", ")"]
                ))
                j = _jaccard(pred, obs_set)
                if j > best_j:
                    best_j = j
                    best_rel_id = r
            except Exception:
                pass
        ent_candidate_list.append({
            "id": e, "name": ent_id2name.get(e, str(e)),
            "score": round(ent_scores[e], 4),
            "best_jaccard": round(best_j, 4),
            "best_relation_id": best_rel_id,
        })

    two_hop_scored = []
    for (rel1, rel2, anchor), cov in two_hop.items():
        s = score(cov["obs"], cov["fp"])
        try:
            pred = set(graph_sampler.search_answers_to_query([
                "(", "p", "(", -rel1, ")",
                "(", "p", "(", -rel2, ")",
                "(", "e", "(", anchor, ")", ")", ")", ")"
            ]))
            j = _jaccard(pred, obs_set)
        except Exception:
            j = 0.0
        two_hop_scored.append({
            "rel1_id": rel1, "rel1_name": rel_id2name.get(rel1, str(rel1)),
            "rel2_id": rel2, "rel2_name": rel_id2name.get(rel2, str(rel2)),
            "anchor_id": anchor, "anchor_name": ent_id2name.get(anchor, str(anchor)),
            "obs_coverage": len(cov["obs"]),
            "score": round(s, 4),
            "jaccard": round(j, 4),
        })
    two_hop_scored.sort(key=lambda x: x["score"], reverse=True)

    return {
        "relation_candidates": rel_candidate_list,
        "entity_candidates": ent_candidate_list,
        "two_hop_candidates": two_hop_scored[:top_k],
    }


class NeighborhoodCandidatesTool(Tool):
    name = "neighborhood_candidates"
    description = (
        "Collect 1-hop and 2-hop (p,p,e) KG neighbors of given entities and score candidates "
        "by coverage over observation entities minus coverage over FP entities. "
        "Returns relation_candidates, entity_candidates, and two_hop_candidates."
    )
    inputs = {
        "entity_ids": {"type": "string", "description": "Comma-separated entity IDs to collect neighbors for (use FN or O)"},
        "fp_ids": {"type": "string", "description": "Comma-separated FP entity IDs to penalize (optional)", "nullable": True},
        "split": {"type": "string", "description": "Graph split: train/test (default: train)", "nullable": True},
        "top_k": {"type": "integer", "description": "Max candidates to return (default: 15)", "nullable": True},
    }
    output_type = "string"

    def __init__(self, kg, **kwargs):
        super().__init__(**kwargs)
        self.kg = kg

    def forward(self, entity_ids: str, fp_ids: str = None, split: str = "train", top_k: int = 15) -> str:
        import json
        eids = [int(x.strip()) for x in entity_ids.split(",") if x.strip()]
        fps = [int(x.strip()) for x in fp_ids.split(",") if x.strip()] if fp_ids else []
        result = neighborhood_candidates_tool(
            eids, self.kg.graph_samplers[split],
            self.kg.ent_id2name, self.kg.rel_id2name,
            fp_ids=fps, top_k=top_k,
        )
        return json.dumps(result)


class GenerateHypothesisLLMTool(Tool):
    name = "generate_hypothesis_llm"
    description = (
        "Construct or modify a hypothesis raw action string. "
        "Three modes: "
        "(1) 'raw': pass a raw_query string directly (any format: 1p/2p/2i/3i/2u/pi/ip/up/etc.); "
        "(2) 'conditions': build from entity/relation ID pairs (auto-infers 1p/2i/3i/2u); "
        "(3) 'modify': provide history_raw (previous hypothesis) and a modification_instruction "
        "    (natural language), the LLM will rewrite the hypothesis accordingly. "
        "Also returns the natural language description."
    )
    inputs = {
        "mode": {
            "type": "string",
            "description": "'raw', 'conditions', or 'modify'",
            "nullable": True,
        },
        "raw_query": {
            "type": "string",
            "description": "Used when mode='raw'. The raw action string, e.g. 'p -3 5059'.",
            "nullable": True,
        },
        "conditions_json": {
            "type": "string",
            "description": (
                "Used when mode='conditions'. JSON array of dicts with entity_id (int), relation_id (int), "
                "and optional 'op' ('i' or 'u', default 'i'). "
                "e.g. '[{\"entity_id\": 5059, \"relation_id\": 3}, {\"entity_id\": 4352, \"relation_id\": 8, \"op\": \"u\"}]'"
            ),
            "nullable": True,
        },
        "history_raw": {
            "type": "string",
            "description": "Used when mode='modify'. The previous hypothesis raw action string.",
            "nullable": True,
        },
        "modification_instruction": {
            "type": "string",
            "description": "Used when mode='modify'. Natural language instruction for how to modify the hypothesis.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, kg, llm_model=None, **kwargs):
        super().__init__(**kwargs)
        self.kg = kg
        self.llm_model = llm_model

    def forward(
        self,
        mode: str = "conditions",
        raw_query: str = None,
        conditions_json: str = None,
        history_raw: str = None,
        modification_instruction: str = None,
    ) -> str:
        import json
        from akgr.utils.parsing_util import qry_actionstr_2_wordlist
        from akgr.agent.getsomesampleFromDB import query_to_natural_language

        def verbalize(raw):
            try:
                tokens = qry_actionstr_2_wordlist(raw)
                return query_to_natural_language(
                    query_tokens=tokens,
                    ent_id2name=self.kg.ent_id2name,
                    rel_id2name=self.kg.rel_id2name,
                )
            except Exception as e:
                return f"(verbalization failed: {e})"

        if mode == "raw":
            if not raw_query:
                return json.dumps({"error": "raw_query required for mode='raw'"})
            raw = raw_query.strip()

        elif mode == "conditions":
            if not conditions_json:
                return json.dumps({"error": "conditions_json required for mode='conditions'"})
            conditions = json.loads(conditions_json)
            if not conditions:
                return json.dumps({"error": "empty conditions"})
            # Determine operator: if any condition has op='u', use union; else intersection
            ops = [c.get("op", "i") for c in conditions]
            use_union = any(o == "u" for o in ops)
            n = len(conditions)
            if n == 1:
                c = conditions[0]
                raw = f"p {-abs(c['relation_id'])} {c['entity_id']}"
            else:
                op_token = "u" if use_union else "i"
                branches = " ".join(
                    f"(p ({-abs(c['relation_id'])}) (e ({c['entity_id']})))"
                    for c in conditions
                )
                raw = f"({op_token} {branches})"
            # Convert parenthesized form to flat action string
            from akgr.utils.parsing_util import qry_str_2_actionstr
            raw = qry_str_2_actionstr(raw)

        elif mode == "modify":
            if not history_raw or not modification_instruction:
                return json.dumps({"error": "history_raw and modification_instruction required for mode='modify'"})
            if self.llm_model is None:
                return json.dumps({"error": "llm_model not provided; cannot use mode='modify'"})
            history_nl = verbalize(history_raw)
            prompt = (
                f"You are modifying a knowledge graph query hypothesis.\n"
                f"Current hypothesis (raw): {history_raw}\n"
                f"Current hypothesis (natural language): {history_nl}\n"
                f"Modification instruction: {modification_instruction}\n\n"
                f"Output ONLY the new raw action string (same token format as the input). "
                f"Do not explain."
            )
            raw = self.llm_model(prompt).strip()
        else:
            return json.dumps({"error": f"unknown mode '{mode}'"})

        return json.dumps({"raw_output": raw, "query_nl": verbalize(raw)})


class IntersectionCandidatesTool(Tool):
    name = "intersection_candidates"
    description = (
        "Enumerate 2i, 3i, or 2u combinations from flat_candidates (from incoming_edge_intersection) "
        "and return top-k by jaccard. "
        "mode='2i': intersection of 2 branches, top-20 singles → C(20,2)=190 pairs. "
        "mode='3i': intersection of 3 branches, top-10 singles → C(10,3)=120 triples. "
        "mode='2u': union of 2 branches, top-20 singles → C(20,2)=190 pairs."
    )
    inputs = {
        "flat_candidates_json": {"type": "string", "description": "JSON array of flat_candidates from incoming_edge_intersection"},
        "observation_ids": {"type": "string", "description": "Comma-separated observation entity IDs"},
        "mode": {"type": "string", "description": "'2i', '3i', or '2u'", "nullable": True},
        "split": {"type": "string", "description": "Graph split: train/test (default: train)", "nullable": True},
        "top_k": {"type": "integer", "description": "Max results to return (default: 10)", "nullable": True},
    }
    output_type = "string"

    def __init__(self, kg, **kwargs):
        super().__init__(**kwargs)
        self.kg = kg

    def forward(self, flat_candidates_json: str, observation_ids: str, mode: str = "2i", split: str = "train", top_k: int = 10) -> str:
        import json
        parsed = json.loads(flat_candidates_json)
        # Accept either the full incoming_edge_intersection output or just the flat_candidates list
        if isinstance(parsed, dict) and "flat_candidates" in parsed:
            flat = parsed["flat_candidates"]
        else:
            flat = parsed
        obs_set = set(int(x.strip()) for x in observation_ids.split(",") if x.strip())
        graph_sampler = self.kg.graph_samplers[split]
        if mode == "3i":
            results = compute_3i_candidates(flat, obs_set, graph_sampler, top_k=top_k)
        elif mode == "2u":
            results = compute_2u_candidates(flat, obs_set, graph_sampler, top_k=top_k)
        else:
            results = compute_2i_candidates(flat, obs_set, graph_sampler, top_k=top_k)
        return json.dumps(results)
