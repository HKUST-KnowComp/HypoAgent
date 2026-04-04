# akgr/agent/source_builder.py

def build_ctrlhgen_source(
    observation_entity_ids,
    condition_type="unconditional",
    condition_value=None,
):
    """
    observation_entity_ids: list[int]
    return: 训练时同分布的 source 字符串
    """
    if not observation_entity_ids:
        raise ValueError("observation_entity_ids is empty")

    source = " ".join(str(int(x)) for x in observation_entity_ids)

    if condition_type == "unconditional":
        return source

    elif condition_type == "entity":
        # condition_value 应该是 entity id
        return f"{source} SEP {int(condition_value)}"

    elif condition_type == "relation":
        # 训练里 relation token 是负数串
        rel_id = int(condition_value)
        return f"{source} SEP -{rel_id}"

    elif condition_type == "entitynumber":
        return f"{source} SEP {condition_value}"

    elif condition_type == "relationnumber":
        return f"{source} SEP {condition_value}"

    elif condition_type == "pattern":
        return f"{source} SEP {condition_value}"

    else:
        raise ValueError(f"Unsupported condition_type: {condition_type}")