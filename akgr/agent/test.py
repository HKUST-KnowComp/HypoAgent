#!/usr/bin/env python3
from akgr.agent.single import case1, build_adapter
from akgr.agent.getsomesampleFromDB import query_to_natural_language
from akgr.utils.load_util import load_yaml
from smolagents import OpenAIServerModel

hypothesis_model_path = '/home/gaoyisen/akgr-agent/checkpoints/PharmKG8k-full-32-130-multi.pth'
data_root = '/home/gaoyisen/akgr-agent/data/'
dataname = 'PharmKG8k'

adapter = build_adapter(hypothesis_model_path, data_root, dataname)
kg = adapter.kg
# print(kg.rel_id2name)
api_cfg = load_yaml('akgr/configs/api_keys.yml')['deepinfra']
llm = OpenAIServerModel(
    model_id=api_cfg['model_id'],
    api_base=api_cfg['api_base'],
    api_key=api_cfg['api_key'],
)

def llm_call(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_cfg['api_key'], base_url=api_cfg['api_base'])
    resp = client.chat.completions.create(
        model=api_cfg['model_id'],
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

result = query_to_natural_language(
    query_tokens=case1["query"],
    ent_id2name=kg.ent_id2name,
    rel_id2name=kg.rel_id2name,
    llm_call=llm_call,
)

print("Ground truth:", case1["query_nl"])
print("LLM result:  ", result)

# --- Test 1: Graph Validation ---
from akgr.agent.tools import GraphValidationTool
from akgr.utils.parsing_util import qry_actionstr_2_wordlist

graph_val_tool = GraphValidationTool(kg=kg)

# flat action string -> bracketed wordlist (already raw, no unshift needed)
wordlist = qry_actionstr_2_wordlist(case1["query"])
tokens_str = " ".join(str(t) for t in wordlist)
print("\n=== Graph Validation ===")
print("tokens:", tokens_str)
label_str = ",".join(str(a) for a in case1["answers"])
print(graph_val_tool.forward(tokens_str, label_answers=label_str, split="train"))

# --- Test 2: Incoming Edge Intersection ---
from akgr.agent.tools import IncomingEdgeIntersectionTool

incoming_tool = IncomingEdgeIntersectionTool(kg=kg)
answer_ids_str = ",".join(str(a) for a in case1["answers"])

print("\n=== Incoming Edge Intersection ===")
print(incoming_tool.forward(answer_ids_str, split="test", top_k=10))
