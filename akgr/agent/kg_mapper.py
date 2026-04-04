from difflib import get_close_matches

def normalize_name(s: str) -> str:
    return (
        str(s).lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("'", "")
        .replace('"', "")
        .strip()
    )

def canonical_token_set(s: str) -> str:
    """
    将名字拆词、排序，解决词序不同的问题：
    'Systemic Lupus Erythematosus'
    -> 'erythematosus lupus systemic'
    """
    s = normalize_name(s)
    toks = [x for x in s.split() if x]
    toks = sorted(toks)
    return " ".join(toks)

class KGNameMapper:
    def __init__(self, kg):
        self.ent_id2name = getattr(kg, "ent_id2name", None)
        self.rel_id2name = getattr(kg, "rel_id2name", None)

        if self.ent_id2name is None:
            raise ValueError("kg.ent_id2name not found")
        if self.rel_id2name is None:
            raise ValueError("kg.rel_id2name not found")

        self.ent_name2id = {str(name): int(idx) for idx, name in self.ent_id2name.items()}
        self.rel_name2id = {str(name): int(idx) for idx, name in self.rel_id2name.items()}

        self.norm_ent_name2id = {
            normalize_name(name): idx for name, idx in self.ent_name2id.items()
        }
        self.norm_rel_name2id = {
            normalize_name(name): idx for name, idx in self.rel_name2id.items()
        }

        # 新增：按词集合排序后的索引
        self.token_set_ent_name2id = {
            canonical_token_set(name): idx for name, idx in self.ent_name2id.items()
        }
        self.token_set_rel_name2id = {
            canonical_token_set(name): idx for name, idx in self.rel_name2id.items()
        }

    def get_entity_id(self, name: str) -> int:
        name = str(name)

        # 1) 精确匹配
        if name in self.ent_name2id:
            return self.ent_name2id[name]

        # 2) 规范化匹配
        norm = normalize_name(name)
        if norm in self.norm_ent_name2id:
            return self.norm_ent_name2id[norm]

        # 3) token-set 排序后匹配
        canon = canonical_token_set(name)
        if canon in self.token_set_ent_name2id:
            return self.token_set_ent_name2id[canon]

        # 4) fuzzy 匹配
        cands = get_close_matches(norm, list(self.norm_ent_name2id.keys()), n=5, cutoff=0.6)
        if cands:
            best = cands[0]
            # 如果最优候选非常接近，可以直接接受
            if len(cands) == 1 or best == norm or len(best.split()) == len(norm.split()):
                return self.norm_ent_name2id[best]

            # 如果候选之间有多个接近，可以列出它们给用户
            cand_text = [self.ent_id2name[self.norm_ent_name2id[c]] for c in cands]
            raise KeyError(f"Entity not found in KG: {name}. Close matches: {cand_text}")

        # 5) token-set fuzzy 匹配
        cands2 = get_close_matches(canon, list(self.token_set_ent_name2id.keys()), n=5, cutoff=0.6)
        if cands2:
            best = cands2[0]
            return self.token_set_ent_name2id[best]

        raise KeyError(f"Entity not found in KG: {name}")

    def get_relation_id(self, name: str) -> int:
        name = str(name)

        if name in self.rel_name2id:
            return self.rel_name2id[name]

        norm = normalize_name(name)
        if norm in self.norm_rel_name2id:
            return self.norm_rel_name2id[norm]

        canon = canonical_token_set(name)
        if canon in self.token_set_rel_name2id:
            return self.token_set_rel_name2id[canon]

        cands = get_close_matches(norm, list(self.norm_rel_name2id.keys()), n=5, cutoff=0.6)
        if cands:
            return self.norm_rel_name2id[cands[0]]

        cands2 = get_close_matches(canon, list(self.token_set_rel_name2id.keys()), n=5, cutoff=0.6)
        if cands2:
            return self.token_set_rel_name2id[cands2[0]]

        raise KeyError(f"Relation not found in KG: {name}")