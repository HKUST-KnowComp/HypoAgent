import os, sys, argparse
from tqdm import tqdm

# data loading and saving
import yaml
import json
import pandas as pd
# import multiprocessing as mp

# sys.path.append('../utils/')
from akgr.utils.load_util import load_yaml, load_csv, load_and_filter_query_patterns
from akgr.kgdata import load_kg

# multiprocessing
import time, math, random
from functools import partial
from multiprocessing import Pool

# ===================== DEBUG CONFIG =====================
DEBUG_FILE = "debug_first2.txt"
MAX_DEBUG_PER_MODE = 2  # 只打印每个 mode 前两条数据
# ========================================================

def _safe_str(x, max_len=8000):
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
    """一次性 dump 多个变量。"""
    with open(DEBUG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\n==================== {section} ====================\n")
    for k, v in kwargs.items():
        debug_dump(k, v)

# Global variable
graph_samplers = None
def init_workers(init_value):
    global graph_samplers
    graph_samplers = init_value

def sample_good_query_given_pattern(mode, max_answers_size, pattern_str):
    """
    graphs['test'] = train + valid + test
    """
    answers_from = {}

    while True:
        sampled_query = graph_samplers[mode].sample_valid_query_given_pattern(pattern_str)

        answers_from['train'] = graph_samplers['train'].search_answers_to_query(sampled_query)
        if mode == 'train':
            if len(answers_from[mode]) > max_answers_size:
                continue
            if len(answers_from['train']) > 0:
                break

        if mode in ['valid', 'test']:
            answers_from['valid'] = graph_samplers['valid'].search_answers_to_query(sampled_query)

        if mode == 'valid':
            if len(answers_from[mode]) > max_answers_size:
                continue
            if len(answers_from['train']) > 0:
                if len(answers_from['valid']) > 0:
                    if len(answers_from['train']) != len(answers_from['valid']):
                        break

        if mode == 'test':
            answers_from['test'] = graph_samplers['test'].search_answers_to_query(sampled_query)
            if len(answers_from[mode]) > max_answers_size:
                continue
            if len(answers_from['train']) > 0:
                if len(answers_from['valid']) > 0:
                    if len(answers_from['test']) > 0:
                        if len(answers_from['test']) != len(answers_from['valid']):
                            break

    # 注意：这里只返回，不在这里写 debug（debug 写在 sample_mode 里，只写前两条）
    return sampled_query, answers_from, pattern_str

def judge(answers_from, mode):
    """
    If the answers are good or not. Return True if it is good, and False ow.
    """
    if mode == 'train':
        return len(answers_from['train']) > 0
    elif mode == 'valid':
        if len(answers_from['train']) > 0:
            if len(answers_from['valid']) > 0:
                return len(answers_from['train']) != len(answers_from['valid'])
            else:
                return False
        else:
            return False
    elif mode == 'test':
        if len(answers_from['train']) > 0:
            if len(answers_from['valid']) > 0:
                if len(answers_from['test']) > 0:
                    return len(answers_from['test']) != len(answers_from['valid'])
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False

def append_aq(answers_queries: list, mode: str, answers_from: dict, query: str, max_answers_size: int, query_type: str):
    def subsample():
        answers = set(random.sample(answers_from[mode], max_answers_size))
        sampled_answers_from = {}
        for split in ['train', 'valid', 'test']:
            sampled_answers_from[split] = list(answers.intersection(answers_from[split]))
            if split == mode:
                break
        return sampled_answers_from

    if len(answers_from[mode]) > max_answers_size:
        while True:
            sampled_answers_from = subsample()
            if judge(sampled_answers_from, mode):
                break
        answers = sampled_answers_from[mode]
    else:
        answers = answers_from[mode]

    answers_queries.append({'answers': answers, 'query': query, 'pattern_str': query_type})

def write_output(answers_queries, dataname, mode, args, id2ent=None):
    df = pd.DataFrame.from_records(answers_queries)
    # random shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    def func_str2int(ids):
        return [int(id) for id in ids]
    df['answers'] = df['answers'].apply(func_str2int)

    output_prefix = f'{dataname}-{args.scale}-{args.max_answer_size}-{mode}'
    path = os.path.join(args.data_root, f'{dataname}/{str(args.reverse_edges_flag)}/', f'{output_prefix}-a2q.jsonl')
    df.to_json(path, orient='records', lines=True)

    def func_id2ent(ids):
        return [id2ent[id] for id in ids]
    if id2ent is not None:
        df['answers'] = df['answers'].apply(func_id2ent)
        os.makedirs(dataname, exist_ok=True)
        df.to_json(f'{dataname}/{output_prefix}-a2q-ent.jsonl', orient='records', lines=True)

def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-sampling', default="akgr/configs/config-sampling.yml")
    parser.add_argument('-s', '--scale', help='n-queries setting, e.g., debug/tiny/same')
    parser.add_argument('-a', '--max-answer-size', type=int, default=32)
    parser.add_argument('-p', '--nproc', type=int, default=1, help='num proc')
    parser.add_argument('--data_root', default='./sampled_data/')
    parser.add_argument('-r', '--reverse_edges_flag', action='store_true', default=False, help='reverse_edges_flag')
    args = parser.parse_args()
    return args

def main():
    # 清空 debug 文件
    with open(DEBUG_FILE, "w", encoding="utf-8") as f:
        f.write("DEBUG OUTPUT (first 2 samples per mode)\n")

    args = my_parse_args()  # 传入的str变成namespace对象
    print(args)

    config_sampling = load_yaml(args.config_sampling)  # 把yaml变成dict
    print(yaml.dump(config_sampling[args.scale]))  # yaml.dump把dict变成yaml格式的str

    # 先把全局大变量 dump 一次
    debug_dump_many(
        "GLOBAL_INIT",
        args=args,
        config_sampling=config_sampling,
        config_sampling_scale_block=config_sampling.get(args.scale, None),
    )

    pattern_filtered = load_and_filter_query_patterns(
        file_name=config_sampling['pattern_table_file'],
        max_dep=2, exclu=None, column='original'
    )

    pattern_filtered.to_csv('akgr/metadata/pattern_filtered.csv', index=True)

    debug_dump_many(
        "PATTERN_FILTERED",
        pattern_filtered_type=type(pattern_filtered),
        pattern_filtered_head=pattern_filtered.head(10),
        pattern_filtered_shape=getattr(pattern_filtered, "shape", None),
        pattern_filtered_columns=getattr(pattern_filtered, "columns", None),
    )

    os.makedirs(args.data_root, exist_ok=True)
    scaling_factor = config_sampling[args.scale]['scale']

    for dataname in config_sampling[args.scale]['datasets']:
        kg = load_kg(args.data_root, dataname, reverse_edges_flag=args.reverse_edges_flag)
        num_ent = kg.num_ent
        num_rel = kg.num_rel
        global graph_samplers
        graph_samplers = kg.graph_samplers
        num_train_edges = kg.num_train_edges

        debug_dump_many(
            "DATASET_LOADED",
            dataname=dataname,
            kg_type=type(kg),
            num_ent=num_ent,
            num_rel=num_rel,
            num_train_edges=num_train_edges,
            graph_samplers_type=type(graph_samplers),
            graph_samplers_keys=getattr(graph_samplers, "keys", lambda: None)(),
        )

        print(f'# Sampling from {dataname} dataset, num_samples_perpattern:')
        num_samples_perpattern = {
            'train': num_train_edges // scaling_factor,
            'valid': (num_train_edges // scaling_factor) // 8,
            'test': (num_train_edges // scaling_factor) // 8
        }
        print(num_samples_perpattern)

        debug_dump_many(
            "SAMPLE_COUNTS",
            scaling_factor=scaling_factor,
            num_samples_perpattern=num_samples_perpattern
        )

        os.makedirs(os.path.join(args.data_root, dataname, str(args.reverse_edges_flag)), exist_ok=True)
        stats_path = os.path.join(args.data_root, f'{dataname}/{str(args.reverse_edges_flag)}/stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f'nentity\t{num_ent}\n')
            f.write(f'nrelation\t{num_rel}\n')

        from itertools import repeat, chain

        patterns_pool = {}
        patterns_total = {}

        pattern_list = pattern_filtered['pattern_str'].tolist()

        debug_dump_many(
            "PATTERN_LIST",
            pattern_list_type=type(pattern_list),
            pattern_list_len=len(pattern_list),
            pattern_list_preview=pattern_list[:20],
        )

        for split in ['train', 'valid', 'test']:
            iters = []
            total = 0
            for pattern_str in pattern_list:
                num_samples = num_samples_perpattern[split]
                iters.append(repeat(pattern_str, num_samples))
                total += num_samples

            patterns_pool[split] = chain.from_iterable(iters)  # 不再是 list
            patterns_total[split] = total  # 保存总长度

        debug_dump_many(
            "PATTERNS_POOL_BUILT",
            patterns_total=patterns_total,
            patterns_pool_train_type=type(patterns_pool['train']),
            patterns_pool_valid_type=type(patterns_pool['valid']),
            patterns_pool_test_type=type(patterns_pool['test']),
        )

        def sample_mode(mode):
            """
            Input: patterns list and mode name
            Output: Write to data files
            """
            print(f"Sampling {mode} queries")
            answers_queries = []
            accepted = 0  # 已经成功保存的条数（只要前两条）

            debug_dump_many(
                "SAMPLE_MODE_START",
                mode=mode,
                mode_type=type(mode),
                max_answer_size=args.max_answer_size,
                nproc=args.nproc,
                patterns_total_for_mode=patterns_total[mode],
            )

            if args.nproc == 1:
                init_workers(graph_samplers)
                func = partial(sample_good_query_given_pattern, mode, args.max_answer_size)

                # 只跑到收集到 2 条为止
                for pattern_str in tqdm(patterns_pool[mode], total=patterns_total[mode]):
                    sampled_query, answers_from, query_type = func(pattern_str)

                    # 先把本次采样涉及的所有关键变量 dump（只 dump 前两条“成功加入”的）
                    # 这里先不 dump，等 append 后确定成功计数，再 dump
                    append_aq(answers_queries, mode, answers_from, sampled_query, args.max_answer_size, query_type)

                    accepted += 1
                    # dump 第 accepted 条（只 dump 前两条）
                    if accepted <= MAX_DEBUG_PER_MODE:
                        debug_dump_many(
                            f"ACCEPTED_SAMPLE_{mode}_{accepted}",
                            mode=mode,
                            max_answers_size=args.max_answer_size,
                            pattern_str=pattern_str,
                            sampled_query=sampled_query,
                            answers_from=answers_from,
                            query_type=query_type,
                            appended_record=answers_queries[-1],
                            appended_record_type=type(answers_queries[-1]),
                            answers_queries_len=len(answers_queries),
                        )

                    if accepted >= MAX_DEBUG_PER_MODE:
                        break

            else:
                # 多进程：只收集前两条，然后 terminate
                with tqdm(total=patterns_total[mode]) as pbar:
                    with Pool(processes=args.nproc, initializer=init_workers, initargs=(graph_samplers,)) as pool:
                        func = partial(sample_good_query_given_pattern, mode, args.max_answer_size)
                        chunksize = 128
                        for sampled_query, answers_from, query_type in pool.imap_unordered(func, patterns_pool[mode], chunksize=chunksize):
                            append_aq(answers_queries, mode, answers_from, sampled_query, args.max_answer_size, query_type)
                            accepted += 1

                            if accepted <= MAX_DEBUG_PER_MODE:
                                debug_dump_many(
                                    f"ACCEPTED_SAMPLE_{mode}_{accepted}",
                                    mode=mode,
                                    max_answers_size=args.max_answer_size,
                                    sampled_query=sampled_query,
                                    answers_from=answers_from,
                                    query_type=query_type,
                                    appended_record=answers_queries[-1],
                                    appended_record_type=type(answers_queries[-1]),
                                    answers_queries_len=len(answers_queries),
                                )

                            pbar.update()

                            if accepted >= MAX_DEBUG_PER_MODE:
                                pool.terminate()
                                break

            debug_dump_many(
                "SAMPLE_MODE_END",
                mode=mode,
                answers_queries_len=len(answers_queries),
                answers_queries_type=type(answers_queries),
                answers_queries_preview=answers_queries[:2],
            )

            write_output(answers_queries, dataname, mode, args)

        sample_mode(mode='train')
        sample_mode(mode='valid')
        sample_mode(mode='test')

    print(f"[DEBUG] wrote debug output to: {DEBUG_FILE}")

if __name__ == '__main__':
    main()