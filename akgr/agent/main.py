from smolagents import CodeAgent, OpenAIServerModel
from akgr.agent.tools import LoadKGTool, FormatConversionTool, GenerateHypothesisTool

case = {
  "answers": [6313,5965],
  "query": ["(","p","(",-18,")","(","i","(","n","(","p","(",-18,")","(","e","(",1559,")",")",")",")","(","p","(",-19,")","(","e","(",13780,")",")",")",")",")"],
  "pattern_str": "(p,(i,(n,(p,(e))),(p,(e))))",
  "query_nl": "Entities that have a 'associatedAct' link to an entity that does not have a 'associatedAct' link to André_Vida, and has a 'associatedBand' link to Mark_Helias",
  "answers_nl": ["Ed_Blackwell","Don_Cherry_(trumpeter)"],
  "intention_mode": "two-condition",
  "followup_condition_values": {
    "pattern": "p p e p e",
    "entitynumber": "2e",
    "relationnumber": "3p",
    "entity": "André_Vida",
    "entity_id": "1560",
    "relation": "associatedAct",
    "relation_id": "-19"
  },
  "followup_question": "I want a hypothesis that follows the pattern \"p p e p e\" and has 3 relations."
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
    hypothesis_model_path = '/home/gaoyisen/akgr-agent/checkpoints/DBpedia50-full-32-430-multi.pth'
    data_root = '/home/gaoyisen/akgr-agent/data/'
    dataname = 'DBpedia50'

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
    )

    # Agent 2: hypothesis generation — calls the loaded model and returns hypothesis
    hypothesis_generate_agent = CodeAgent(
        tools=[GenerateHypothesisTool(adapter=adapter)],
        model=llm_model,
    )

    answer_nl = case["answers_nl"]
    followup = case["followup_question"]

    # Step 1: format conversion
    fmt_result = format_conversion_agent.run(
        f"Convert these observation entities {answer_nl} and followup question "
        f"'{followup}' into model input using the format_conversion tool. "
        f"Return the source_text and conditions."
    )
    print("=== Format Conversion Result ===")
    print(fmt_result)

    # Step 2: hypothesis generation
    obs_str = ", ".join(answer_nl)
    # Extract condition from case for direct generation
    conditions = case["followup_condition_values"]
    hyp_result = hypothesis_generate_agent.run(
        f"Generate a hypothesis using the generate_hypothesis tool. "
        f"observation_entities='{obs_str}', "
        f"condition_type='multi', condition_value='{followup}'"
    )
    print("=== Hypothesis Generation Result ===")
    print(hyp_result)
