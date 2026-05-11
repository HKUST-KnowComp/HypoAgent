from smolagents import CodeAgent, OpenAIServerModel
from akgr.agent.tools import FormatConversionTool
from akgr.agent.case import case_2i,case_2p,case_1p,case_2u,case_up,case_ip


case1 = {
  "answers": [5828,5001,5066,2941,5679,5456,2578,3891,2937,3546,6077,2463],
  "query": ["(","i","(","n","(","p","(",-8,")","(","p","(",-21,")","(","e","(",1128,")",")",")",")",")","(","p","(",-8,")","(","e","(",4922,")",")",")",")"],
  "pattern_str": "(i,(n,(p,(p,(e)))),(p,(e)))",
  "query_nl": "Entities that do not have a 'GG' link to an entity that has a 'Rg' link to cdh1, and have a 'GG' link to pask",
  "answers_nl": ["rpgrip1l","pdx1","pfkfb1","gys1","recql4","prpf6","fxn","ltk","gyg1","kcnh2","ski","flt4"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "i n p p e p e",
    "entitynumber": "2e",
    "relationnumber": "3p",
    "entity": "cdh1",
    "entity_id": "1128",
    "relation": "GG",
    "relation_id": "-8"
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
    "entity_id": "33",
    "relation": "GG",
    "relation_id": "-8"
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
    hypothesis_model_path = '/home/gaoyisen/akgr-agent/checkpoints/PharmKG8k-full-32-160-multi.pth'
    data_root = '/home/gaoyisen/akgr-agent/data/'
    dataname = 'PharmKG8k'
    case = case_ip
    from akgr.utils.load_util import load_yaml
    api_cfg = load_yaml('akgr/configs/api_keys.yml')['deepinfra']
    llm_model = OpenAIServerModel(
        model_id=api_cfg['model_id'],
        api_base=api_cfg['api_base'],
        api_key=api_cfg['api_key'],
        timeout=60,
    )

    adapter = build_adapter(hypothesis_model_path, data_root, dataname)

    # Agent 1: format conversion — parses answer_nl + followup into model input
    format_conversion_agent = CodeAgent(
        tools=[FormatConversionTool(adapter=adapter)],
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
    from akgr.agent.tools import generate_hypothesis_tool, compute_metrics
    gen_result = generate_hypothesis_tool(adapter, source_text)
    raw_output = gen_result["raw_output"]
    print("=== Hypothesis Generation Result ===")
    print(gen_result.get("query_nl", raw_output))

    # Step 3: metric computation
    label_answers_str = ",".join(str(x) for x in case["answers"])
    metrics = compute_metrics(raw_output, case["answers"], adapter.kg.graph_samplers)
    print("=== Metric Result ===")
    print(f"Jaccard={metrics['jaccard']:.4f}  Dice={metrics['dice']:.4f}  Overlap={metrics['overlap']:.4f}")  

    # Step 4: TP/FP/FN diagnosis
    from akgr.agent.tools import execute_and_diagnose_tool
    diag = execute_and_diagnose_tool(raw_output, case["answers"], adapter.kg.graph_samplers["train"])
    print("=== Diagnosis ===")
    print(f"TP={diag['tp_count']}  FP={diag['fp_count']}  FN={diag['fn_count']}")
    print(f"Precision={diag['precision']:.4f}  Recall={diag['recall']:.4f}  F1={diag['f1']:.4f}")
    print(f"Diagnosis: {diag['diagnosis']}")
