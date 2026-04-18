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
    adapter,                        # CtrlHGenAdapter (already initialised)
    observation_entity_names: list[str],
    condition_type: str = "unconditional",
    condition_value: Any = None,
    temperature: float = 1.0,
    top_k: int = 0,
    constrained: bool = True,
) -> dict:
    """
    Generate a logical hypothesis from observations + optional condition.

    Args:
        adapter:                  CtrlHGenAdapter instance
        observation_entity_names: list of entity name strings (source observations)
        condition_type:           one of unconditional / pattern / relation /
                                  entity / relationnumber / entitynumber
        condition_value:          condition value (name string or count int), or None
        temperature:              sampling temperature
        top_k:                    top-k sampling (0 = disabled)
        constrained:              whether to use constrained decoding

    Returns:
        {
          "source_text":    str,   tokenizer input
          "raw_output":     str,   unshifted action string  (e.g. "i -9 5530 -3 12")
          "query":          list,  structured query tokens
          "query_nl":       str,   natural language description
          "entitynumber":   int | None,
          "relationnumber": int | None,
          "conditions":     list[dict],
        }
    """
    parsed = {
        "observation_entities": observation_entity_names,
        "conditions": [{"type": condition_type, "value": condition_value}],
    }
    model_input = adapter.build_model_input(parsed)
    result = adapter.generate(model_input, temperature=temperature,
                              top_k=top_k, constrained=constrained)
    result["conditions"] = model_input["conditions"]
    return result


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
        "Generate a logical hypothesis from observation entities and an optional condition "
        "using the trained CtrlHGen model. Returns the action string and natural language description."
    )
    inputs = {
        "observation_entities": {"type": "string", "description": "Comma-separated entity names (observations)"},
        "condition_type":       {"type": "string", "description": "unconditional/pattern/relation/entity/relationnumber/entitynumber", "nullable": True},
        "condition_value":      {"type": "string", "description": "Condition value (name or count)", "nullable": True},
    }
    output_type = "string"

    def __init__(self, adapter, **kwargs):
        super().__init__(**kwargs)
        self.adapter = adapter

    def forward(self, observation_entities: str, condition_type: str = "unconditional", condition_value: str = None) -> str:
        obs = [e.strip() for e in observation_entities.split(",") if e.strip()]
        result = generate_hypothesis_tool(self.adapter, obs, condition_type, condition_value)
        return (
            f"raw_output: {result['raw_output']}\n"
            f"query_nl: {result['query_nl']}\n"
            f"entitynumber: {result['entitynumber']}, relationnumber: {result['relationnumber']}"
        )


# ---------------------------------------------------------------------------
# 4. format_conversion tool
# ---------------------------------------------------------------------------

def format_conversion_tool(
    adapter,                        # CtrlHGenAdapter
    answer_nl: list[str],           # natural-language entity names (observations)
    followup_questions: str,        # free-text condition request from user
    session_memory: dict | None = None,
    llm_parser=None,                # LocalQwenParser or compatible; None -> rule-based fallback
) -> dict:
    """
    Convert answer_nl + followup_questions into a tokenized model input dict
    ready for adapter.generate().

    Returns:
        {
          "parsed":       dict,   output of llm_parser (conditions, observation_entities, ...)
          "model_input":  dict,   output of adapter.build_model_input (source_text, input_ids, ...)
        }
    """
    # 1. Parse the followup question into structured conditions
    if llm_parser is not None:
        parsed_control = llm_parser.parse_condition_text(
            condition_text=followup_questions,
            session_memory=session_memory or {},
        )
    else:
        # Rule-based fallback: unconditional
        parsed_control = {
            "observation_entities": [],
            "conditions": [{"type": "unconditional", "value": ""}],
        }

    # 2. Inject observation entities from answer_nl
    parsed_control["observation_entities"] = list(answer_nl)

    # 3. Build tokenized model input via adapter
    model_input = adapter.build_model_input(parsed_control)

    return {
        "parsed": parsed_control,
        "model_input": model_input,
    }


class FormatConversionTool(Tool):
    name = "format_conversion"
    description = (
        "Convert natural-language observation entities (answer_nl) and a followup "
        "condition request into a tokenized model input for the hypothesis model. "
        "Returns source_text and structured conditions."
    )
    inputs = {
        "answer_nl":          {"type": "string", "description": "Comma-separated observation entity names"},
        "followup_questions": {"type": "string", "description": "Free-text condition request, e.g. 'more specific' or 'from spouse perspective'"},
    }
    output_type = "string"

    def __init__(self, adapter, llm_parser=None, **kwargs):
        super().__init__(**kwargs)
        self.adapter = adapter
        self.llm_parser = llm_parser

    def forward(self, answer_nl: str, followup_questions: str) -> str:
        entities = [e.strip() for e in answer_nl.split(",") if e.strip()]
        result = format_conversion_tool(
            adapter=self.adapter,
            answer_nl=entities,
            followup_questions=followup_questions,
            llm_parser=self.llm_parser,
        )
        mi = result["model_input"]
        return (
            f"source_text: {mi['source_text']}\n"
            f"conditions: {mi['conditions']}\n"
            f"observation_entity_ids: {mi['observation_entity_ids']}"
        )


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
