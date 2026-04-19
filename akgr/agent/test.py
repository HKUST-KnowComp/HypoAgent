#!/usr/bin/env python3
"""
测试脚本：加载 PharmKG8k 知识图谱并检查实体
"""

from akgr.kgdata import load_kg
from akgr.agent.kg_mapper import KGNameMapper

def load_pharmkg8k(data_root: str = '/home/gaoyisen/akgr-agent/data/'):
    """加载 PharmKG8k 知识图谱"""
    print(f"Loading PharmKG8k from {data_root}...")
    kg = load_kg(
        dataroot=data_root,
        dataname='DBpedia50',
        reverse_edges_flag=False
    )
    print(f"Loaded!")
    print(f"  - Number of entities: {len(kg.ent_id2name)}")
    print(f"  - Number of relations: {len(kg.rel_id2name)}")
    return kg

def check_entity_by_name(kg, entity_name: str):
    """通过实体名称检查实体是否在 KG 中"""
    mapper = KGNameMapper(kg)

    print(f"\n=== Checking entity: '{entity_name}' ===")

    # 方法1: 直接精确匹配
    if entity_name in mapper.ent_name2id:
        entity_id = mapper.ent_name2id[entity_name]
        print(f"✓ Found (exact match): ID = {entity_id}")
        return entity_id

    # 方法2: 规范化名称匹配
    from difflib import get_close_matches

    def normalize_name(name: str) -> str:
        import re
        name = name.lower().replace('_', ' ').replace('-', ' ')
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    norm_name = normalize_name(entity_name)
    if norm_name in mapper.norm_ent_name2id:
        entity_id = mapper.norm_ent_name2id[norm_name]
        original_name = kg.ent_id2name[entity_id]
        print(f"✓ Found (normalized match): '{original_name}' (ID = {entity_id})")
        return entity_id

    # 方法3: 模糊匹配，找相似的实体
    print(f"✗ Exact match not found. Searching for similar names...")
    candidates = get_close_matches(
        norm_name,
        list(mapper.norm_ent_name2id.keys()),
        n=5,
        cutoff=0.6
    )

    if candidates:
        print(f"  Similar entities found:")
        for i, cand in enumerate(candidates, 1):
            cand_id = mapper.norm_ent_name2id[cand]
            original_name = kg.ent_id2name[cand_id]
            print(f"    {i}. '{original_name}' (ID = {cand_id})")
    else:
        print(f"  No similar entities found.")

    return None

def check_entity_by_id(kg, entity_id: int):
    """通过实体 ID 检查实体是否在 KG 中"""
    print(f"\n=== Checking entity ID: {entity_id} ===")

    if entity_id in kg.ent_id2name:
        entity_name = kg.ent_id2name[entity_id]
        print(f"✓ Found: '{entity_name}' (ID = {entity_id})")
        return entity_name
    else:
        print(f"✗ Entity ID {entity_id} not found")
        return None

def list_sample_entities(kg, n: int = 10):
    """列出 KG 中的一些示例实体"""
    print(f"\n=== Sample entities (first {n}) ===")
    for i, (eid, name) in enumerate(kg.ent_id2name.items()):
        if i >= n:
            break
        print(f"  ID {eid}: {name}")

def search_entities(kg, keyword: str):
    """搜索包含关键字的实体"""
    print(f"\n=== Searching entities containing '{keyword}' ===")
    keyword_lower = keyword.lower()
    matches = []

    for eid, name in kg.ent_id2name.items():
        if keyword_lower in str(name).lower():
            matches.append((eid, name))

    if matches:
        print(f"  Found {len(matches)} matches:")
        for eid, name in matches[:20]:  # 最多显示20个
            print(f"    ID {eid}: {name}")
        if len(matches) > 20:
            print(f"    ... and {len(matches) - 20} more")
    else:
        print(f"  No matches found")

    return matches

if __name__ == "__main__":
    # 加载知识图谱
    kg = load_pharmkg8k()

    # 显示一些示例实体
    list_sample_entities(kg, n=10)

    # 示例1: 通过名称检查实体
    # 修改这里来测试你想检查的实体名称
    entity_name_to_check = "Dirty_Dozen_Brass_Band"  # 改为你想检查的实体名
    check_entity_by_name(kg, entity_name_to_check)

    # 示例2: 通过 ID 检查实体
    # entity_id_to_check = 100
    # check_entity_by_id(kg, entity_id_to_check)

    # 示例3: 搜索包含关键字的实体
    # search_entities(kg, "diabetes")
