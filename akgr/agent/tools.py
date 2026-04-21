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
    for sq in _split_top_level(query):
        try:
            sub_ans = graph_sampler.search_answers_to_query(sq)
            sub_results.append(_build_result(sub_ans, sub_query=sq))
        except Exception as e:
            sub_results.append({"sub_query": sq, "error": str(e)})

    main_result["sub_query_results"] = sub_results
    return main_result


class GraphValidationTool(Tool):
    name = "graph_validation"
    description = (
        "Execute a query (token list) on the KG graph and validate it. "
        "If the top-level operator is i (intersection) or u (union), splits into 2 sub-queries. "
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
        raw = query_tokens.strip().split()
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

    # Build hints: for each common head, collect which relations connect it to answer entities
    hints = []
    for head in list(common_heads)[:top_k]:
        rels = set()
        for triples in per_entity_triples:
            for (h, r, _) in triples:
                if h == head:
                    rels.add(r)
        head_name = ent_id2name.get(head, str(head))
        rel_info = [{"name": rel_id2name.get(r, str(r)), "id": r} for r in rels]
        hints.append({"head_entity": head_name, "head_id": head, "relations": rel_info})

    return {
        "intersection_count": len(common_heads),
        "intersection_ids": list(common_heads)[:top_k],
        "hints": hints,
    }


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
