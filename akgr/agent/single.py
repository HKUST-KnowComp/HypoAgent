from smolagents import CodeAgent, OpenAIServerModel
from akgr.agent.tools import FormatConversionTool, GenerateHypothesisTool, MetricTool

case1 ={
  "answers": [5828,5001,5066,2941,5679,5456,2578,3891,2937,3546,6077,2463],
  "query": "i n -8 -21 1128 -8 4922",
#   "query": ["(","i","(","n","(","p","(",-8,")","(","p","(",-21,")","(","e","(",1128,")",")",")",")",")","(","p","(",-8,")","(","e","(",4922,")",")",")",")"],
  "pattern_str": "(i,(n,(p,(p,(e)))),(p,(e)))",
  "query_nl": "Entities that do not have a 'GG' link to an entity that has a 'Rg' link to cdh1, and have a 'GG' link to pask",
  "answers_nl": ["rpgrip1l","pdx1","pfkfb1","gys1","recql4","prpf6","fxn","ltk","gyg1","kcnh2","ski","flt4"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "i n p p e p e",
    "entitynumber": "2e",
    "relationnumber": "3p",
    "entity": "cdh1",
    "entity_id": "1129",
    "relation": "E",
    "relation_id": "-9"
  },
  "followup_question": "I want a hypothesis that includes the relation \"GG\" and contains 2 entities."
}
case2 ={
  "answers": [5056,5057,5058,5061,5062,5063,5053,5055],
  "query": ["(","i","(","i","(","p","(",-8,")","(","e","(",33,")",")",")","(","p","(",-3,")","(","e","(",5059,")",")",")",")","(","p","(",-3,")","(","e","(",5059,")",")",")",")"],
  "pattern_str": "(i,(i,(p,(e)),(p,(e))),(p,(e)))",
  "query_nl": "Entities that have a 'GG' link to abcd1, and have a 'B' link to pex19, and have a 'B' link to pex19",
  "answers_nl": ["pex13","pex14","pex16","pex3","pex5","pex6","pex10","pex12"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "i i p e p e p e",
    "entitynumber": "3e",
    "relationnumber": "3p",
    "entity": "abcd1",
    "entity_id": "34",
    "relation": "E",
    "relation_id": "-9"
  },
  "followup_question": "I want a hypothesis that follows the pattern \"i i p e p e p e\" and has 3 relations."
}
def build_adapter(hypothesis_model_path: str, data_root: str, dataname: str):
    import yaml
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

    # Agent 1: format conversion — parses answer_nl + followup into model input
    format_conversion_agent = CodeAgent(
        tools=[FormatConversionTool(adapter=adapter)],
        model=llm_model,
        additional_authorized_imports=["json"],
    )

    # Agent 2: hypothesis generation — calls the loaded model and returns hypothesis
    hypothesis_generate_agent = CodeAgent(
        tools=[GenerateHypothesisTool(adapter=adapter)],
        model=llm_model,
        additional_authorized_imports=["json"],
    )

    # Agent 3: metric computation — jaccard/dice/overlap against ground truth
    metric_agent = CodeAgent(
        tools=[MetricTool(graph_samplers=adapter.kg.graph_samplers)],
        model=llm_model,
        additional_authorized_imports=["json"],
    )

    answer_nl = case["answers_nl"]
    followup = case["followup_question"]

    # Step 1: format conversion
    fmt_result = format_conversion_agent.run(
        f"Parse the followup question '{followup}' into a conditions_json array. "
        f"Valid condition types: unconditional, pattern, relation, entity, relationnumber, entitynumber. "
        f"For multi-condition, include multiple dicts, e.g. "
        f'[{{"type":"relation","value":"relation_name"}},{{"type":"entity","value":"entity_name"}}]. '
        f"Then call format_conversion with answer_nl='{', '.join(answer_nl)}' "
        f"and the conditions_json you constructed."
    )
    print("=== Format Conversion Result ===")
    print(fmt_result)

    import json as _json
    source_text = _json.loads(fmt_result)["source_text"]

    # Step 2: hypothesis generation using source_text from Step 1
    hyp_result = hypothesis_generate_agent.run(
        f"Call generate_hypothesis with source_text='{source_text}'"
    )
    print("=== Hypothesis Generation Result ===")
    print(hyp_result)

    # Step 3: metric computation
    label_answers_str = ",".join(str(x) for x in case["answers"])
    metric_result = metric_agent.run(
        f"Call compute_metrics with raw_output='{hyp_result.split(chr(10))[0].replace('raw_output: ', '')}' "
        f"and label_answers='{label_answers_str}'"
    )
    print("=== Metric Result ===")
    print(metric_result)
