import sys, json

# sys.path.append('./utils/')
from akgr.utils.load_util import load_yaml
# from utils.load_util import load_yaml


from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, T5TokenizerFast, GPT2TokenizerFast
import random


def _normalize_stage(stage: str, is_gen: bool) -> str:
    if stage:
        return str(stage).lower()
    return 'test' if is_gen else 'train'


def _is_generation_stage(stage: str) -> bool:
    return stage in {'test', 'optimize', 'agent'}


def _agent_query_stats(target):
    return [
        {
            "entitynumber": number_to_epnumber(tgt)[0],
            "relationnumber": number_to_epnumber(tgt)[1],
        }
        for tgt in target
    ]
def number_to_pattern(input_str):
    elements = input_str.split()

    result = []
    for elem in elements:
        if elem.lstrip('-').isdigit():  # 检查是否是数字（包括负数）
            num = int(elem)
            if num < 0:
                result.append('p')  # 负数变为 p
            else:
                result.append('e')  # 正数变为 e
        else:
            result.append(elem)  # 非数字保持不变
    
    output_str = ' '.join(result)
    return output_str

def number_to_epnumber(input_string):
    elements = input_string.split()  # 按空格分割字符串
    count1 = 0
    count2 = 0
    for elem in elements:
        try:
            num = int(elem)       # 尝试转为浮点数（兼容整数和小数）
            if num > 0:             # 判断是否为正数
                count1 += 1
            elif num < 0:
                count2 += 1
        
        except ValueError:          # 忽略非数字元素（如字母、符号等）
            pass
    count_e = f"{count1}e"
    count_p = f"{count2}p"
    return count_e, count_p

def number_to_epspecific(input_string):
    # 分割字符串并初始化正负集合
    elements = input_string.split()
    positive_numbers = []
    negative_numbers = []
    
    for elem in elements:
        try:
            num = int(elem)
            if num > 0:
                positive_numbers.append(num)
            elif num < 0:
                negative_numbers.append(num)
        except ValueError:
            continue  # 跳过非数字
    
    # # 随机选择（如果集合非空）
    # random_positive = random.choice(positive_numbers) if positive_numbers else None
    # random_negative = random.choice(negative_numbers) if negative_numbers else None
    random_positive = positive_numbers[0] if positive_numbers else None
    random_negative = negative_numbers[0] if negative_numbers else None
    # if 'n' in elements:
    #     random_negative = f"n {random_negative}" if random_negative is not None else None
    #     random_positive = f"n {random_positive}" if random_positive is not None else None
    return str(random_positive), str(random_negative)

def get_vocab(special_tokens, offset, nentity, nrelation):
    vocab = {}
    vocab.update(special_tokens)
    for i in range(1, nentity+1): # [offset, offset + nentity - 1]
        vocab[str(i)] = offset + i - 1
    for i in range(1, nrelation+1): # [offset + nentity, offset + nentity + nrelation - 1]
        vocab[str(-i)] = offset + nentity + i - 1
    # vocab["-1"] = offset
    return vocab, offset + nentity + nrelation

def create_tokenizer(
        special_tokens: dict, offset: int,
        nentity: int, nrelation: int,
        is_gpt: bool):
    pre_tokenizer = WhitespaceSplit()
    vocab, vocab_size = get_vocab(special_tokens, offset=offset, nentity=nentity, nrelation=nrelation)
    model = WordLevel(vocab, unk_token='UNK')
    if not is_gpt:
        post_processor = TemplateProcessing(
            single='$0 END',
            # pair='$A START $B END',
            special_tokens=[('END', special_tokens['END'])]
        )
    else:
        post_processor = TemplateProcessing(
            single='$0 SEP',
            pair='$A SEP $B END',
            special_tokens=[('SEP', special_tokens['SEP']), ('END', special_tokens['END'])]
        )
    tokenizer = Tokenizer(model=model)

    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.post_processor = post_processor
    # Just to let the tokenizer know about special tokens
    tokenizer.add_special_tokens(['START', 'END', 'PAD', 'UNK', 'SEP'])
    import io
    from contextlib import redirect_stdout
    trap = io.StringIO()
    with redirect_stdout(trap):
        TokenizerFast = GPT2TokenizerFast if is_gpt else T5TokenizerFast
        tokenizer = TokenizerFast(
            tokenizer_object=tokenizer,
            bos_token='START',
            eos_token='END',
            pad_token='PAD',
            unk_token='UNK',
            sep_token='SEP',
            ) # default padding side
        # tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, vocab_size

def search_one_hop(source, graph,src_len):
    G = graph
    new_source_list = []
    for src in source:
            tmp = 0 
            count_list = []
            for node in src.split():
                node = int(node)
                tmp = tmp + 1
                in_edges = G.in_edges(node)  # 获取所有指向 node 的入边
                for (u,v,k) in in_edges:
                    count_list.append(u)
                    
            str_list = set(count_list)
            str_list = list(str_list)
            str_list = map(str, str_list)
            
            new_source = src +' '+ ' '.join(str_list)
            # print(new_source)
            split_list = new_source.split()
            truncated_list = split_list[:src_len]
            truncated_list = ' '.join(truncated_list)
            new_source_list.append(truncated_list)
    return new_source_list
            
           


import torch
def new_extract_sample_to_device(device,
        sample, tokenizer, is_gpt:bool,
        src_len, tgt_len, is_gen:bool, stage: str = None):
    stage = _normalize_stage(stage, is_gen)
    source = sample['source']
    target = sample['target']
    target_pattern = [''] * len(target)
    pattern_id = sample['pattern_id']

    # print("\n===== DEBUG merged_source =====")
    # for i, text in enumerate(source[:5]):
    #     print(f"sample {i}:")
    #     print(text)
    #     print("-"*80)

    # breakpoint()

    if not is_gpt:
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # special treatment for T5: ignore end in source
        
        attention_mask[input_ids == tokenizer.eos_token_id] = 0
        

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        source_target_tokenized = tokenizer(
            source, target,
            padding='longest',
            # max_length=src_len+tgt_len,
            return_tensors="pt").to(device)
        # labels is the source SEP target END, ...
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # ... with the source part's loss ignored
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if not _is_generation_stage(stage): # (train/valid) input = source SEP target END, default padding side
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
            # print('gpt')
            # print(input_ids)
            # print(tokenizer.batch_decode(input_ids))
        else: # (test/optimize) input = source c, left padding (align the last tokens to the right)
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                source,
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

        # labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id

    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    if stage == 'agent':
        target_pattern = _agent_query_stats(target)

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask,target_pattern

def new_extract_sample_to_device_search(device,
        sample, tokenizer,graph, is_gpt:bool,
        src_len, tgt_len, is_gen:bool, stage: str = None):
    stage = _normalize_stage(stage, is_gen)
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    source = search_one_hop(source, graph, src_len)
    if not is_gpt:
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # special treatment for T5: ignore end in source
        
        attention_mask[input_ids == tokenizer.eos_token_id] = 0
        

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        source_target_tokenized = tokenizer(
            source, target,
            padding='longest',
            # max_length=src_len+tgt_len,
            return_tensors="pt").to(device)
        # labels is the source SEP target END, ...
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # ... with the source part's loss ignored
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if not _is_generation_stage(stage): # (train/valid) input = source SEP target END, default padding side
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
            # print('gpt')
            # print(input_ids)
            # print(tokenizer.batch_decode(input_ids))
        else: # (test/optimize) input = source c, left padding (align the last tokens to the right)
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                source,
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

        # labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id

    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask

def new_extract_sample_to_device_pattern(device, sample, tokenizer, is_gpt: bool, src_len, tgt_len, is_gen: bool, stage: str = None):
    stage = _normalize_stage(stage, is_gen)
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    target_pattern = [number_to_pattern(tgt) for tgt in target]
    # target_pattern = [number_to_epnumber(tgt)[0] for tgt in target]
    merged_source = [f"{s} [SEP] {t}" for s, t in zip(source, target_pattern)]
    # merged_source = f"{source} [SEP] {pattern_id}"  
    # print(merged_source)

    # print("\n===== DEBUG merged_source =====")
    # for i, text in enumerate(merged_source[:5]):
    #     print(f"sample {i}:")
    #     print(text)
    #     print("-"*80)
    
    if not is_gpt:
        # 非 GPT 情况：直接处理合并后的 source
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # 特殊处理 T5：忽略 source 中的 eos_token
        attention_mask[input_ids == tokenizer.eos_token_id] = 0

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        # GPT 情况：将合并后的 source 和 target 一起处理
        source_target_tokenized = tokenizer(
            merged_source, target,  # 使用合并后的 source
            padding='longest',
            return_tensors="pt").to(device)
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # 忽略 source 部分的 loss
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if not _is_generation_stage(stage):  # 训练/验证阶段
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
        else:  # 测试/生成阶段（左填充）
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                merged_source,  # 使用合并后的 source
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

    # DEBUG: print aligned text and tokenized outputs
    print("\n===== DEBUG text + tokenized source =====")
    debug_n = min(5, len(merged_source), source_tokenized.input_ids.size(0))
    for i in range(debug_n):
        text = merged_source[i]
        ids = source_tokenized.input_ids[i].tolist()
        mask = source_tokenized.attention_mask[i].tolist()
        valid_ids = source_tokenized.input_ids[i][source_tokenized.attention_mask[i] == 1].tolist()
        print(f"sample {i} text: {text}")
        print(f"sample {i} input_ids: {ids}")
        print(f"sample {i} attention_mask: {mask}")
        print(f"sample {i} valid_input_ids: {valid_ids}")
        print(f"sample {i} decoded_all: {tokenizer.decode(ids, skip_special_tokens=False)}")
        print(f"sample {i} decoded_valid: {tokenizer.decode(valid_ids, skip_special_tokens=False)}")
        print("-" * 80)

    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask


    breakpoint()

    if stage == 'agent':
        target_pattern = [
            {
                "pattern": pat,
                "entitynumber": number_to_epnumber(tgt)[0],
                "relationnumber": number_to_epnumber(tgt)[1],
            }
            for pat, tgt in zip(target_pattern, target)
        ]

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_pattern

def new_extract_sample_to_device_number_entity(device, sample, tokenizer, is_gpt: bool, src_len, tgt_len, is_gen: bool, stage: str = None):
    stage = _normalize_stage(stage, is_gen)
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    target_pattern = [number_to_epnumber(tgt)[0] for tgt in target]
    merged_source = [f"{s} [SEP] {t}" for s, t in zip(source, target_pattern)]
    # merged_source = f"{source} [SEP] {pattern_id}"  
    # print(merged_source)

    # print("\n===== DEBUG merged_source =====")
    # for i, text in enumerate(merged_source[:5]):
    #     print(f"sample {i}:")
    #     print(text)
    #     print("-"*80)
    
    if not is_gpt:
        # 非 GPT 情况：直接处理合并后的 source
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # 特殊处理 T5：忽略 source 中的 eos_token
        attention_mask[input_ids == tokenizer.eos_token_id] = 0

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        # GPT 情况：将合并后的 source 和 target 一起处理
        source_target_tokenized = tokenizer(
            merged_source, target,  # 使用合并后的 source
            padding='longest',
            return_tensors="pt").to(device)
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # 忽略 source 部分的 loss
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if not _is_generation_stage(stage):  # 训练/验证阶段
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
        else:  # 测试/生成阶段（左填充）
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                merged_source,  # 使用合并后的 source
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

    # DEBUG: print aligned text and tokenized outputs
    print("\n===== DEBUG text + tokenized source =====")
    debug_n = min(5, len(merged_source), source_tokenized.input_ids.size(0))
    for i in range(debug_n):
        text = merged_source[i]
        ids = source_tokenized.input_ids[i].tolist()
        mask = source_tokenized.attention_mask[i].tolist()
        valid_ids = source_tokenized.input_ids[i][source_tokenized.attention_mask[i] == 1].tolist()
        print(f"sample {i} text: {text}")
        print(f"sample {i} input_ids: {ids}")
        print(f"sample {i} attention_mask: {mask}")
        print(f"sample {i} valid_input_ids: {valid_ids}")
        print(f"sample {i} decoded_all: {tokenizer.decode(ids, skip_special_tokens=False)}")
        print(f"sample {i} decoded_valid: {tokenizer.decode(valid_ids, skip_special_tokens=False)}")
        print("-" * 80)

    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask


    breakpoint()

    if stage == 'agent':
        target_pattern = [
            {
                "entitynumber": number_to_epnumber(tgt)[0],
                "relationnumber": number_to_epnumber(tgt)[1],
            }
            for tgt in target
        ]

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_pattern

def new_extract_sample_to_device_number_relation(device, sample, tokenizer, is_gpt: bool, src_len, tgt_len, is_gen: bool, stage: str = None):
    stage = _normalize_stage(stage, is_gen)
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    target_pattern = [number_to_epnumber(tgt)[1] for tgt in target]
    merged_source = [f"{s} [SEP] {t}" for s, t in zip(source, target_pattern)]
    # merged_source = f"{source} [SEP] {pattern_id}"  
    # print(merged_source)

    # print("\n===== DEBUG merged_source =====")
    # for i, text in enumerate(merged_source[:5]):
    #     print(f"sample {i}:")
    #     print(text)
    #     print("-"*80)

    if not is_gpt:
        # 非 GPT 情况：直接处理合并后的 source
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # 特殊处理 T5：忽略 source 中的 eos_token
        attention_mask[input_ids == tokenizer.eos_token_id] = 0

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        # GPT 情况：将合并后的 source 和 target 一起处理
        source_target_tokenized = tokenizer(
            merged_source, target,  # 使用合并后的 source
            padding='longest',
            return_tensors="pt").to(device)
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # 忽略 source 部分的 loss
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if not _is_generation_stage(stage):  # 训练/验证阶段
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
        else:  # 测试/生成阶段（左填充）
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                merged_source,  # 使用合并后的 source
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

    # 统一处理 labels 的 padding
    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask
        # DEBUG: print aligned text and tokenized outputs
    print("\n===== DEBUG text + tokenized source =====")
    debug_n = min(5, len(merged_source), source_tokenized.input_ids.size(0))
    for i in range(debug_n):
        text = merged_source[i]
        ids = source_tokenized.input_ids[i].tolist()
        mask = source_tokenized.attention_mask[i].tolist()
        valid_ids = source_tokenized.input_ids[i][source_tokenized.attention_mask[i] == 1].tolist()
        print(f"sample {i} text: {text}")
        print(f"sample {i} input_ids: {ids}")
        print(f"sample {i} attention_mask: {mask}")
        print(f"sample {i} valid_input_ids: {valid_ids}")
        print(f"sample {i} decoded_all: {tokenizer.decode(ids, skip_special_tokens=False)}")
        print(f"sample {i} decoded_valid: {tokenizer.decode(valid_ids, skip_special_tokens=False)}")
        print("-" * 80)
    breakpoint()

    if stage == 'agent':
        target_pattern = [
            {
                "entitynumber": number_to_epnumber(tgt)[0],
                "relationnumber": number_to_epnumber(tgt)[1],
            }
            for tgt in target
        ]

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_pattern

def new_extract_sample_to_device_specific_relation(device, sample, tokenizer, is_gpt: bool, src_len, tgt_len, is_gen: bool, stage: str = None):
    stage = _normalize_stage(stage, is_gen)
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    target_pattern = [number_to_epspecific(tgt)[1] for tgt in target]
    merged_source = [f"{s} [SEP] {t}" for s, t in zip(source, target_pattern)]
    # merged_source = f"{source} [SEP] {pattern_id}"  
    # print(merged_source)

    # print("\n===== DEBUG merged_source =====")
    # for i, text in enumerate(merged_source[:5]):
    #     print(f"sample {i}:")
    #     print(text)
    #     print("-"*80)
    
    if not is_gpt:
        # 非 GPT 情况：直接处理合并后的 source
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # 特殊处理 T5：忽略 source 中的 eos_token
        attention_mask[input_ids == tokenizer.eos_token_id] = 0

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        # GPT 情况：将合并后的 source 和 target 一起处理
        source_target_tokenized = tokenizer(
            merged_source, target,  # 使用合并后的 source
            padding='longest',
            return_tensors="pt").to(device)
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # 忽略 source 部分的 loss
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if not _is_generation_stage(stage):  # 训练/验证阶段
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
        else:  # 测试/生成阶段（左填充）
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                merged_source,  # 使用合并后的 source
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

    # 统一处理 labels 的 padding
    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask
        # DEBUG: print aligned text and tokenized outputs
    print("\n===== DEBUG text + tokenized source =====")
    debug_n = min(5, len(merged_source), source_tokenized.input_ids.size(0))
    for i in range(debug_n):
        text = merged_source[i]
        ids = source_tokenized.input_ids[i].tolist()
        mask = source_tokenized.attention_mask[i].tolist()
        valid_ids = source_tokenized.input_ids[i][source_tokenized.attention_mask[i] == 1].tolist()
        print(f"sample {i} text: {text}")
        print(f"sample {i} input_ids: {ids}")
        print(f"sample {i} attention_mask: {mask}")
        print(f"sample {i} valid_input_ids: {valid_ids}")
        print(f"sample {i} decoded_all: {tokenizer.decode(ids, skip_special_tokens=False)}")
        print(f"sample {i} decoded_valid: {tokenizer.decode(valid_ids, skip_special_tokens=False)}")
        print("-" * 80)
    breakpoint()

    if stage == 'agent':
        target_pattern = [
            {
                "relation": rel,
                "entitynumber": number_to_epnumber(tgt)[0],
                "relationnumber": number_to_epnumber(tgt)[1],
            }
            for rel, tgt in zip(target_pattern, target)
        ]

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_pattern

def new_extract_sample_to_device_specific_entity(device, sample, tokenizer, is_gpt: bool, src_len, tgt_len, is_gen: bool, stage: str = None):
    stage = _normalize_stage(stage, is_gen)
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    target_pattern = [number_to_epspecific(tgt)[0] for tgt in target]
    merged_source = [f"{s} [SEP] {t}" for s, t in zip(source, target_pattern)]
    # merged_source = f"{source} [SEP] {pattern_id}"  
    # print(merged_source)
    if not is_gpt:
        # 非 GPT 情况：直接处理合并后的 source
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # 特殊处理 T5：忽略 source 中的 eos_token
        attention_mask[input_ids == tokenizer.eos_token_id] = 0

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        # GPT 情况：将合并后的 source 和 target 一起处理
        source_target_tokenized = tokenizer(
            merged_source, target,  # 使用合并后的 source
            padding='longest',
            return_tensors="pt").to(device)
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # 忽略 source 部分的 loss
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if not _is_generation_stage(stage):  # 训练/验证阶段
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
        else:  # 测试/生成阶段（左填充）
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                merged_source,  # 使用合并后的 source
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

    # # DEBUG: print aligned text and tokenized outputs
    # print("\n===== DEBUG text + tokenized source =====")
    # debug_n = min(5, len(merged_source), source_tokenized.input_ids.size(0))
    # for i in range(debug_n):
    #     text = merged_source[i]
    #     ids = source_tokenized.input_ids[i].tolist()
    #     mask = source_tokenized.attention_mask[i].tolist()
    #     valid_ids = source_tokenized.input_ids[i][source_tokenized.attention_mask[i] == 1].tolist()
    #     print(f"sample {i} text: {text}")
    #     print(f"sample {i} input_ids: {ids}")
    #     print(f"sample {i} attention_mask: {mask}")
    #     print(f"sample {i} valid_input_ids: {valid_ids}")
    #     print(f"sample {i} decoded_all: {tokenizer.decode(ids, skip_special_tokens=False)}")
    #     print(f"sample {i} decoded_valid: {tokenizer.decode(valid_ids, skip_special_tokens=False)}")
    #     print("-" * 80)
    # breakpoint()

    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask




    if stage == 'agent':
        target_pattern = [
            {
                "entity": ent,
                "entitynumber": number_to_epnumber(tgt)[0],
                "relationnumber": number_to_epnumber(tgt)[1],
            }
            for ent, tgt in zip(target_pattern, target)
        ]

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_pattern

def new_extract_sample_to_device_masked(device,
        sample, tokenizer, is_gpt:bool,
        src_len, tgt_len, is_gen:bool, stage: str = None):
    stage = _normalize_stage(stage, is_gen)
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    if not is_gpt:
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
            
        source_input_ids = source_tokenized.input_ids
        source_attention_mask = source_tokenized.attention_mask
    

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
        labels_input_ids = labels.input_ids
        labels_attention_mask = labels.attention_mask
    else:
        source_target_tokenized = tokenizer(
            source, target,
            padding='longest',
            # max_length=src_len+tgt_len,
            return_tensors="pt").to(device)
        # labels is the source SEP target END, ...
        labels = torch.clone(source_target_tokenized.input_ids)
        # ... with the source part's loss ignored
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id

        if not _is_generation_stage(stage): # (train/valid) input = source SEP target END, default padding side
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
        else: # (test/optimize) input = source SEP, left padding (align the last tokens to the right)
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                source,
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

        # labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
    
    
    input_ids = torch.cat([source_input_ids, labels_input_ids], dim=1)  # 在序列维度上拼接

    # 将 source 和 labels 的 attention_mask 拼接在一起
    combined_attention_mask = torch.cat([source_attention_mask, labels_attention_mask], dim=1)
    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = combined_attention_mask
    attention_mask = source_attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask
def debug():
    config_dataloader = load_yaml('akgr/configs/config-dataloader.yml')
    offset = config_dataloader['offset']
    special_tokens = config_dataloader['special_tokens']
    tokenizer, _ = create_tokenizer(special_tokens, offset, nentity=200000, nrelation=2000, is_gpt=True)
    sample1 = {'answers': [1, 2, 3, 4], "query": ["(","i","(","n","(","p","(",-1,")","(","p","(",0,")","(","e","(",0,")",")",")",")",")","(","p","(",-567,")","(","e","(",24623,")",")",")",")"], "pattern_str":"(i,(n,(p,(p,(e)))),(p,(e)))"}
    sample2 = {'answers': [1, 2], "query": ["(", "p", "(", -1, ")", "(", "e", "(", 0, ")", ")", ")"], "pattern_str": "(p,(e))"}
    from utils.parsing_util import qry_shift_indices, ans_shift_indices, qry_str_2_actionstr
    def list_to_str(l: list) -> str:
        # print('before', l)
        # print('after', ' '.join([str(x) if isinstance(x, int) else x for x in l]))
        return ' '.join([str(x) if isinstance(x, int) else x for x in l])
    sample = {}
    sample['source'] = [list_to_str(ans_shift_indices(sample1['answers'])), list_to_str(ans_shift_indices(sample2['answers']))]
    sample['target'] = [qry_str_2_actionstr(list_to_str(qry_shift_indices(sample1['query']))), qry_str_2_actionstr(list_to_str(qry_shift_indices(sample2['query'])))]
    sample['pattern_id'] = [1, 2]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask = \
        new_extract_sample_to_device(device, sample, tokenizer, is_gpt=True, src_len=33, tgt_len=66, is_gen=True)
    # input_ids = tokenizer(sample['source'], padding='max_length', max_length=33, return_tensors="pt")
    print('----')
    print('source')
    print(source)
    print('target')
    print(target)
    print('input_ids')
    print(input_ids)
    print('attention_mask')
    print(attention_mask)
    print('labels')
    print(labels)
    labels[labels == -100] = 0
    print(tokenizer.batch_decode(labels, skip_special_tokens=True))


    source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask = \
        new_extract_sample_to_device(device, sample, tokenizer, is_gpt=True, src_len=33, tgt_len=33, is_gen=False)
    # input_ids = tokenizer(sample['source'], padding='max_length', max_length=33, return_tensors="pt")
    print('source')
    print(source)
    print('target')
    print(target)
    print('input_ids')
    print(input_ids)
    print('attention_mask')
    print(attention_mask)
    print(source_attention_mask)
    print('labels')
    print(labels)
    labels[labels == -100] = 0
    print(tokenizer.batch_decode(labels, skip_special_tokens=True))

def source_to_prompt(sample,args):
    source = sample['source']              # 单个字符串，如 "19346"
    target = sample['target']              # 单个目标
    condition = args.condition

    if condition == 'unconditional':
        sample['prompt'] = source
    elif condition == 'pattern':
        target_pattern = number_to_pattern(target)
        sample['prompt'] = f"{source} [SEP] {target_pattern}"
    elif condition == 'relationnumber':
        sample['prompt'] = f"{source} [SEP] {number_to_epnumber(target)[1]}"
    elif condition == 'entitynumber':
        sample['prompt'] = f"{source} [SEP] {number_to_epnumber(target)[0]}"
    elif condition == 'relation':
        sample['prompt'] = f"{source} [SEP] {number_to_epspecific(target)[1]}"
    elif condition == 'entity':
        sample['prompt'] = f"{source} [SEP] {number_to_epspecific(target)[0]}"
    else:
        raise ValueError(f"Unsupported condition: {condition}")
    return sample
if __name__ == '__main__':
    debug()


def new_extract_sample_to_device_multi(
    device, sample, tokenizer, is_gpt: bool, src_len, tgt_len, is_gen: bool,
    multi_condition_list, random_multi: bool, seed: int,
    random_multi_keep_range=(0, 3), stage: str = None
):
    stage = _normalize_stage(stage, is_gen)
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']

    condition = []
    merged_source = []

    rng = random.Random(seed)

    for s, tgt in zip(source, target):
        cond_list = []
        cond_dict = {}
        cond_name_set = set(multi_condition_list)

        ordered_conditions = [
            'relationnumber',
            'entitynumber',
            'relation',
            'entity',
            'pattern',
        ]



        if random_multi and stage in {'train', 'optimize', 'test'}:
            # Randomly keep conditions from the configured range [min_keep, max_keep].
            min_keep, max_keep = random_multi_keep_range
            if min_keep == max_keep:
                keep_n = min_keep
            else:
                max_available = len(cond_name_set)
                min_keep = max(0, min(min_keep, max_available))
                max_keep = max(min_keep, min(max_keep, max_available))
                keep_n = rng.randint(min_keep, max_keep+1)


            # print(f"length of cond_name_set: {len(cond_name_set)}")
            active_cond_name_set = (rng.sample((cond_name_set), keep_n))
        else:
            active_cond_name_set = cond_name_set

        for cond_name in ordered_conditions:
            if cond_name == 'relationnumber':
                cond_value = number_to_epnumber(tgt)[1] if cond_name in active_cond_name_set else 'none'
            elif cond_name == 'entitynumber':
                cond_value = number_to_epnumber(tgt)[0] if cond_name in active_cond_name_set else 'none'
            elif cond_name == 'relation':
                cond_value = number_to_epspecific(tgt)[1] if cond_name in active_cond_name_set else 'none'
            elif cond_name == 'entity':
                cond_value = number_to_epspecific(tgt)[0] if cond_name in active_cond_name_set else 'none'
            elif cond_name == 'pattern':
                cond_value = number_to_pattern(tgt) if cond_name in active_cond_name_set else 'none'
            else:
                raise ValueError(f"Unsupported multi condition: {cond_name}")

            if cond_value is not None:
                cond_text = str(cond_value)
                cond_list.append(cond_text)
                cond_dict[cond_name] = cond_text

        merged_cond = '  '.join(cond_list) 
        merged_source.append(f"{s} [SEP] {merged_cond}")
        if stage == 'agent':
            cond_dict['entitynumber'] = number_to_epnumber(tgt)[0]
            cond_dict['relationnumber'] = number_to_epnumber(tgt)[1]
        condition.append(cond_dict)
        
        # Keep per-sample loop output quiet; use aligned batch debug below.

    if not is_gpt:
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt"
        ).to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask

        attention_mask[input_ids == tokenizer.eos_token_id] = 0

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt"
        ).input_ids.to(device)

    else:
        source_target_tokenized = tokenizer(
            merged_source, target,
            padding='longest',
            return_tensors="pt"
        ).to(device)

        labels = torch.clone(source_target_tokenized.input_ids)

        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt"
        ).to(device)

        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id

        if not _is_generation_stage(stage):
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
        else:
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                merged_source,
                padding='longest',
                max_length=src_len,
                return_tensors="pt"
            ).to(device)
            tokenizer.padding_side = original_padding_side

            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

    # DEBUG: print aligned text and tokenized outputs
    print("\n===== DEBUG text + tokenized source =====")
    debug_n = min(5, len(merged_source), source_tokenized.input_ids.size(0))
    for i in range(debug_n):
        text = merged_source[i]
        ids = source_tokenized.input_ids[i].tolist()
        mask = source_tokenized.attention_mask[i].tolist()
        valid_ids = source_tokenized.input_ids[i][source_tokenized.attention_mask[i] == 1].tolist()
        print(f"sample {i} text: {text}")
        print(f"sample {i} input_ids: {ids}")
        print(f"sample {i} attention_mask: {mask}")
        print(f"sample {i} valid_input_ids: {valid_ids}")
        print(f"sample {i} decoded_all: {tokenizer.decode(ids, skip_special_tokens=False)}")
        print(f"sample {i} decoded_valid: {tokenizer.decode(valid_ids, skip_special_tokens=False)}")
        print("-" * 80)

    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask


    # breakpoint()

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, condition