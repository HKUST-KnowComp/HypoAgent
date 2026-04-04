def parse_path(tokens, i=0):
    """
    Parse a path-like suffix ending in an entity id.
    Example:
        ['-9', '5531'] -> path(-9, entity 5531)
        ['-9', '-22', '1137'] -> path(-9, path(-22, entity 1137))
    Returns: (node, next_idx)
    """
    rels = []
    n = len(tokens)

    while i < n and tokens[i].lstrip("-").isdigit():
        val = int(tokens[i])
        if val < 0:
            rels.append(val)
            i += 1
        else:
            # positive integer = entity id
            node = {"type": "entity", "id": val}
            for rel in reversed(rels):
                node = {"type": "path", "rel": rel, "child": node}
            return node, i + 1

    raise ValueError(f"Invalid path tokens near position {i}: {tokens[i:]}")


def parse_action(tokens, i=0):
    if i >= len(tokens):
        raise ValueError("Unexpected end of tokens")

    tok = tokens[i]

    if tok == "i":
        left, j = parse_action(tokens, i + 1)
        right, k = parse_action(tokens, j)
        return {"type": "intersection", "children": [left, right]}, k

    if tok == "u":
        left, j = parse_action(tokens, i + 1)
        right, k = parse_action(tokens, j)
        return {"type": "union", "children": [left, right]}, k

    if tok == "n":
        child, j = parse_action(tokens, i + 1)
        return {"type": "negation", "child": child}, j

    # otherwise assume it's a path expression starting with relation ids
    return parse_path(tokens, i)


def action_string_to_tree(action_str: str):
    """
    Convert a decoded action string into a tree.
    Example:
        'i -9 5531 -9 -22 1137'
    """
    action_str = action_str.strip()
    if not action_str:
        raise ValueError("Empty action string")

    tokens = action_str.split()
    tree, idx = parse_action(tokens, 0)

    if idx != len(tokens):
        raise ValueError(f"Unconsumed tokens: {tokens[idx:]}")

    return tree


def action_string_to_tree_prefix(action_str: str):
    """
    Like `action_string_to_tree`, but tolerates trailing tokens by only parsing
    the first valid action expression.

    Returns: (tree, remaining_tokens)
    """
    action_str = action_str.strip()
    if not action_str:
        raise ValueError("Empty action string")

    tokens = action_str.split()
    tree, idx = parse_action(tokens, 0)
    return tree, tokens[idx:]


def tree_to_structured_text(node):
    t = node["type"]

    if t == "entity":
        return f"entity({node['id']})"

    if t == "path":
        return f"path(rel={node['rel']}, child={tree_to_structured_text(node['child'])})"

    if t == "intersection":
        a, b = node["children"]
        return f"intersection({tree_to_structured_text(a)}, {tree_to_structured_text(b)})"

    if t == "union":
        a, b = node["children"]
        return f"union({tree_to_structured_text(a)}, {tree_to_structured_text(b)})"

    if t == "negation":
        return f"negation({tree_to_structured_text(node['child'])})"

    return str(node)


def tree_to_natural_language(node, id2ent=None, id2rel=None):
    t = node["type"]

    def ent_name(eid_token):
        # action string 里的正整数 token 是 1-based entity index
        raw_eid = eid_token - 1
        if id2ent is not None and raw_eid in id2ent:
            return id2ent[raw_eid]
        return f"entity_{raw_eid}"

    def rel_name(rid_token):
        # action string 里的负整数 token 是 -1-based relation index
        raw_rid = abs(rid_token) - 1
        if id2rel is not None and raw_rid in id2rel:
            return id2rel[raw_rid]
        return f"relation_{raw_rid}"

    if t == "entity":
        return ent_name(node["id"])

    if t == "path":
        return (
            f"entities connected via {rel_name(node['rel'])} to "
            f"{tree_to_natural_language(node['child'], id2ent, id2rel)}"
        )

    if t == "intersection":
        a, b = node["children"]
        return (
            f"entities that satisfy both "
            f"({tree_to_natural_language(a, id2ent, id2rel)}) "
            f"and "
            f"({tree_to_natural_language(b, id2ent, id2rel)})"
        )

    if t == "union":
        a, b = node["children"]
        return (
            f"entities that satisfy either "
            f"({tree_to_natural_language(a, id2ent, id2rel)}) "
            f"or "
            f"({tree_to_natural_language(b, id2ent, id2rel)})"
        )

    if t == "negation":
        return (
            f"entities that do not satisfy "
            f"({tree_to_natural_language(node['child'], id2ent, id2rel)})"
        )

    return str(node)