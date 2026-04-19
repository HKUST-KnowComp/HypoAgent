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
    from akgr.utils.parsing_util import qry_actionstr_2_wordlist, ans_unshift_indices
    pred_qry = qry_actionstr_2_wordlist(raw_output)
    pred_ans = graph_samplers[searching_split].search_answers_to_query(pred_qry)
    label_ans = ans_unshift_indices(label_answers)

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
