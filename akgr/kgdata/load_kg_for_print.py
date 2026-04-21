import argparse

import pandas as pd
import csv
import os
import pickle
import pykeen.datasets as pk_datasets
import pykeen.utils as pk_utils
import torch
# import datasets as hg_datasets

import networkx as nx
from akgr.utils.nx_util import df_to_graph

# ===================== DEBUG CONFIG =====================
DEBUG_FILE = "debug_loadkg_all.txt"
MAX_STR_LEN = 12000
# ========================================================

def _safe_str(x, max_len=MAX_STR_LEN):
    """安全把对象转成字符串，防止太长。"""
    try:
        s = str(x)
    except Exception as e:
        s = f"<cannot stringify: {e}>"
    if len(s) > max_len:
        s = s[:max_len] + f"\n... <truncated, total_len={len(s)}>"
    return s

def debug_dump(name, value, section=None):
    """把变量名/类型/值写到同一个文件。"""
    with open(DEBUG_FILE, "a", encoding="utf-8") as f:
        if section is not None:
            f.write(f"\n########## SECTION: {section} ##########\n")
        f.write(f"\n===== {name} =====\n")
        f.write(f"type: {type(value)}\n")
        f.write("value:\n")
        f.write(_safe_str(value) + "\n")
        f.write("=" * 60 + "\n")

def debug_dump_many(section, **kwargs):
    with open(DEBUG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\n==================== {section} ====================\n")
    for k, v in kwargs.items():
        debug_dump(k, v)

def df_concat(df_list: list):
    return pd.concat(df_list, ignore_index=True)

def update_inverse_edges(rel_id2name: dict, raw_df: pd.DataFrame):
    """
    Input: the rel_id2name map and the single-direction data raw_df

    Process: For each split, create inverse edges for existing edges. New
        edges and original edges are copied into a separate dataframe. The
        relation id maps are updated accordingly.

    Output: new rel_id2name map, rel_id2inv map, and the new data new_df.
    """
    debug_dump_many(
        "ENTER_update_inverse_edges",
        rel_id2name_type=type(rel_id2name),
        rel_id2name_len=len(rel_id2name),
        raw_df_type=type(raw_df),
        raw_df_keys=list(raw_df.keys()) if isinstance(raw_df, dict) else None
    )

    new_id2name = {}
    rel_id2inv = {}
    for id, name in rel_id2name.items():
        new_id2name[id * 2] = f'+{name}'
        new_id2name[id * 2 + 1] = f'-{name}'
        rel_id2inv[id * 2] = id * 2 + 1
        rel_id2inv[id * 2 + 1] = id * 2

    debug_dump_many(
        "AFTER_relation_remap",
        new_id2name_len=len(new_id2name),
        rel_id2inv_len=len(rel_id2inv),
        new_id2name_preview=list(new_id2name.items())[:10],
        rel_id2inv_preview=list(rel_id2inv.items())[:10],
    )

    new_df = {}
    for split, df in raw_df.items():
        df_inv = pd.DataFrame(data=df, copy=True)
        # inverse edges
        df_inv.loc[:, ['head_id', 'tail_id']] = (df_inv.loc[:, ['tail_id', 'head_id']].values)
        # reindex rel id
        df['relation_id'] = df['relation_id'].apply(lambda x: x * 2)
        df_inv['relation_id'] = df_inv['relation_id'].apply(lambda x: x * 2 + 1)
        df_all = df_concat([df, df_inv])
        new_df[split] = df_all.sort_values(by=['relation_id'])

        debug_dump_many(
            f"update_inverse_edges_SPLIT_{split}",
            df_shape=getattr(df, "shape", None),
            df_inv_shape=getattr(df_inv, "shape", None),
            df_all_shape=getattr(df_all, "shape", None),
            new_df_split_shape=getattr(new_df[split], "shape", None),
            new_df_split_head=new_df[split].head(5),
        )

    return new_id2name, rel_id2inv, new_df

def dump_kg(kg, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        kg = pickle.dump(kg, f)
        print(f"# KG saved to {output_path}")
    debug_dump_many(
        "dump_kg",
        output_path=output_path,
        kg_dump_return_value=kg,  # pickle.dump returns None
    )
    return kg

def load_kg_from_disk(input_path):
    with open(input_path, 'rb') as f:
        kg = pickle.load(f)
        print(f"# KG loaded from {input_path}")
        debug_dump_many(
            "load_kg_from_disk",
            input_path=input_path,
            kg_type=type(kg),
            kg_str=_safe_str(kg, 4000),
        )
        return kg

def load_kg_common(dataname: str, reverse_edges_flag: bool, id_map_only: bool):
    # 用 PyKEEN 读数据集 → 转成 DataFrame → 重新划分 split → 可选地手工加逆边
    # → 转成 networkx 图 → 返回各种映射和图。
    """
    :param dataname:
    :return: a dict, see return
    """
    debug_dump_many(
        "ENTER_load_kg_common",
        dataname=dataname,
        reverse_edges_flag=reverse_edges_flag,
        id_map_only=id_map_only,
    )

    if dataname == 'YAGO310':
        ds = pk_datasets.YAGO310(create_inverse_triples=False)
    elif dataname == 'FB15k-237':
        ds = pk_datasets.FB15k237(create_inverse_triples=False)
    elif dataname == 'DBpedia50':
        ds = pk_datasets.DBpedia50(create_inverse_triples=False)
    elif dataname == 'BioKG':
        ds = pk_datasets.BioKG(create_inverse_triples=False)
    elif dataname == 'PharmKG8k':
        ds = pk_datasets.PharmKG8k(create_inverse_triples=False)
    elif dataname == 'WN18RR':
        ds = pk_datasets.WN18RR(create_inverse_triples=False)
    elif dataname == 'OGBWikiKG2':
        ds = pk_datasets.OGBWikiKG2(create_inverse_triples=False)
    else:
        print(f'# Dataset "{dataname}" not supported, return None')
        debug_dump("unsupported_dataname", dataname, section="load_kg_common")
        return None

    debug_dump_many(
        "DATASET_LOADED",
        ds_type=type(ds),
        ds_str=_safe_str(ds, 3000),
    )

    num_ent = ds.num_entities
    num_rel = ds.num_relations

    ent_id2name = pk_utils.invert_mapping(ds.entity_to_id)
    rel_id2name = pk_utils.invert_mapping(ds.relation_to_id)
    rel_id2inv = {}

    debug_dump_many(
        "ID_MAPPINGS",
        num_ent=num_ent,
        num_rel=num_rel,
        ent_id2name_len=len(ent_id2name),
        rel_id2name_len=len(rel_id2name),
        ent_id2name_preview=list(ent_id2name.items())[:10],
        rel_id2name_preview=list(rel_id2name.items())[:10],
    )

    print('# During loading raw kg:')
    raw_df = {}
    for split in ['training', 'validation', 'testing']:
        if id_map_only == True:
            continue

        factory = ds.factory_dict[split]
        mapped_triples = factory.mapped_triples
        triples_df = factory.tensor_to_df(mapped_triples)[['head_id', 'tail_id', 'relation_id']]
        raw_df[split] = triples_df

        debug_dump_many(
            f"RAW_SPLIT_{split}",
            factory_type=type(factory),
            mapped_triples_type=type(mapped_triples),
            mapped_triples_shape=getattr(mapped_triples, "shape", None),
            triples_df_type=type(triples_df),
            triples_df_shape=triples_df.shape,
            triples_df_head=triples_df.head(5),
        )

    raw_df_all = df_concat([raw_df['training'], raw_df['validation'], raw_df['testing']])

    raw_df['training'] = raw_df_all.sample(frac=0.8, replace=False)
    raw_df_remaining = raw_df_all.drop(raw_df['training'].index)
    raw_df['validation'] = raw_df_remaining.sample(frac=0.5, replace=False)
    raw_df['testing'] = raw_df_remaining.drop(raw_df['validation'].index)

    debug_dump_many(
        "SPLITS_REALLOCATED",
        raw_df_all_shape=raw_df_all.shape,
        training_shape=raw_df['training'].shape,
        validation_shape=raw_df['validation'].shape,
        testing_shape=raw_df['testing'].shape,
    )

    if reverse_edges_flag == True:
        rel_id2name, rel_id2inv, raw_df = update_inverse_edges(rel_id2name, raw_df)
        num_rel *= 2

        debug_dump_many(
            "AFTER_reverse_edges",
            num_rel=num_rel,
            rel_id2name_len=len(rel_id2name),
            rel_id2inv_len=len(rel_id2inv),
            rel_id2name_preview=list(rel_id2name.items())[:10],
            rel_id2inv_preview=list(rel_id2inv.items())[:10],
        )

    print('# Sizes after adding inverse edges')
    print(raw_df['training'].shape)
    print(raw_df['validation'].shape)
    print(raw_df['testing'].shape)

    if id_map_only == True:
        result = {
            'ent_id2name': ent_id2name,
            'rel_id2name': rel_id2name
        }
        debug_dump_many("RETURN_id_map_only", result=result)
        return result

    our_df = {
        'train': raw_df['training'],
        'valid': df_concat([raw_df['training'], raw_df['validation']]),
        'test': df_concat([raw_df['training'], raw_df['validation'], raw_df['testing']]),
        'test_only': raw_df['testing']
    }

    debug_dump_many(
        "OUR_DF_BUILT",
        train_shape=our_df['train'].shape,
        valid_shape=our_df['valid'].shape,
        test_shape=our_df['test'].shape,
        test_only_shape=our_df['test_only'].shape,
    )

    graphs = {}
    for split, df in our_df.items():
        graphs[split] = df_to_graph(df)
        # 只输出图的摘要，避免写爆文件
        g = graphs[split]
        debug_dump_many(
            f"GRAPH_BUILT_{split}",
            graph_type=type(g),
            num_nodes=getattr(g, "number_of_nodes", lambda: None)(),
            num_edges=getattr(g, "number_of_edges", lambda: None)(),
        )

    print('# Checking id ranges (in graphs)')
    print(f'ent id: {min(ent_id2name.keys()), max(ent_id2name.keys())}')
    print(f'rel id: {min(rel_id2name.keys()), max(rel_id2name.keys())}')

    result = {
        'num_ent': num_ent,
        'num_rel': num_rel,
        'ent_id2name': ent_id2name,
        'rel_id2name': rel_id2name,
        'rel_id2inv': rel_id2inv,
        'graphs': graphs
    }

    debug_dump_many(
        "RETURN_load_kg_common",
        result_keys=list(result.keys()),
        num_ent=num_ent,
        num_rel=num_rel,
        ent_id2name_len=len(ent_id2name),
        rel_id2name_len=len(rel_id2name),
        rel_id2inv_len=len(rel_id2inv),
        graphs_keys=list(graphs.keys()),
    )

    return result

def load_fb15k237_ent_2idname(ent_id2name):
    mid2name_path = 'akgr/metadata/FB15k_mid2name.txt'
    if os.path.exists(mid2name_path) == False:
        print(f'# Error: {mid2name_path} does not exist')
    mid2name = {}
    with open(mid2name_path, 'r', encoding='utf-8') as f:
        rows = csv.reader(f, delimiter='\t')
        for row in rows:
            mid, name = row
            mid2name[mid] = name
    for id, name in ent_id2name.items():
        ent_id2name[id] = mid2name[name]

    debug_dump_many(
        "load_fb15k237_ent_2idname",
        mid2name_len=len(mid2name),
        ent_id2name_len=len(ent_id2name),
        ent_id2name_preview=list(ent_id2name.items())[:10],
    )
    return ent_id2name

def load_wn18rr_ent_id2name(ent_id2name):
    import nltk
    nltk.download('wordnet')
    from nltk.corpus import wordnet
    for id, name in ent_id2name.items():
        ent_id2name[id] = wordnet.synset_from_pos_and_offset('n', int(name))

    debug_dump_many(
        "load_wn18rr_ent_id2name",
        ent_id2name_len=len(ent_id2name),
        ent_id2name_preview=list(ent_id2name.items())[:5],
    )
    return ent_id2name

from akgr.kgdata.kgclass import GraphSampler, KG
def load_kg(dataroot, dataname, reverse_edges_flag, id_map_only=False):
    print(f'# loading {dataname}')

    debug_dump_many(
        "ENTER_load_kg",
        dataroot=dataroot,
        dataname=dataname,
        reverse_edges_flag=reverse_edges_flag,
        id_map_only=id_map_only,
    )

    raw_kg_dict = load_kg_common(
        dataname,
        reverse_edges_flag,
        id_map_only=id_map_only
    )
    if raw_kg_dict is None:
        debug_dump("raw_kg_dict", raw_kg_dict, section="load_kg")
        return None

    if dataname == 'FB15k-237':
        raw_kg_dict['ent_id2name'] = load_fb15k237_ent_2idname(raw_kg_dict['ent_id2name'])
    elif dataname == 'WN18RR':
        raw_kg_dict['ent_id2name'] = load_wn18rr_ent_id2name(raw_kg_dict['ent_id2name'])

    path = f'{dataroot}/{dataname}/{dataname}.pkl'
    debug_dump("kg_cache_path", path, section="load_kg")

    if os.path.exists(path):
        kg = load_kg_from_disk(path)
        debug_dump_many(
            "KG_LOADED_FROM_DISK",
            kg_type=type(kg),
            kg_str=_safe_str(kg, 4000),
        )
    else:
        kg = KG(
            num_ent=raw_kg_dict['num_ent'],
            num_rel=raw_kg_dict['num_rel'],
            ent_id2name=raw_kg_dict['ent_id2name'],
            rel_id2name=raw_kg_dict['rel_id2name'],
            rel_id2inv=raw_kg_dict['rel_id2inv'],
            graphs=raw_kg_dict['graphs']
        )
        debug_dump_many(
            "KG_CREATED",
            kg_type=type(kg),
            kg_str=_safe_str(kg, 4000),
        )
        dump_kg(kg, path)

    return kg

def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataname', default='YAGO310')
    # 可选：加一个参数控制是否加逆边（不改默认行为，不给也行）
    parser.add_argument('-r', '--reverse_edges_flag', action='store_true', default=False)
    parser.add_argument('--data_root', default='./sampled_data/')
    args = parser.parse_args()
    return args

def debug():
    # 清空 debug 文件
    with open(DEBUG_FILE, "w", encoding="utf-8") as f:
        f.write("DEBUG OUTPUT (load_kg script)\n")

    args = my_parse_args()
    debug_dump_many("ARGS", args=args)

    kg = load_kg(args.data_root, args.dataname, reverse_edges_flag=args.reverse_edges_flag)
    debug_dump_many(
        "FINAL_KG",
        kg_type=type(kg),
        kg_str=_safe_str(kg, 4000),
    )

    print(f"[DEBUG] wrote debug output to: {DEBUG_FILE}")

if __name__ == '__main__':
    debug()