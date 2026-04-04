from typing import Any, Dict, List
import os
import torch

from akgr.tokenizer import create_tokenizer, number_to_epnumber
from akgr.abduction_model.transformer import create_transformer
from akgr.utils.load_util import load_model  # noqa: F401 (kept for potential external use)

from akgr.agent.kg_mapper import KGNameMapper

from akgr.kgdata.kgclass import KG

from akgr.utils.parsing_util import ans_shift_indices, list_to_str
from akgr.agent.getsomesampleFromDB import query_to_natural_language
from akgr.abduction_model.generation import PrefixAllowedTokensFn, generate_with_constraints


class CtrlHGenAdapter:
    def __init__(
        self,
        checkpoint_path: str,
        special_tokens: dict,
        offset: int,
        nentity: int,
        nrelation: int,
        is_gpt: bool,
        model_name: str,
        config_model: dict,

        kg: KG,
        src_len: int,
        tgt_len: int,
        device: str = "cuda",
        max_gen_len: int = 0,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        # Keep generation-length behavior aligned with main_reverse test loop.
        # main_reverse uses:
        #   max_length = tgt_len + src_len * (is_gpt == True)
        # so we treat max_gen_len<=0 as "follow tgt_len exactly".
        self.max_gen_len = int(max_gen_len) if int(max_gen_len) > 0 else int(tgt_len)
        self.special_tokens = special_tokens
        self.offset = offset
        self.nentity = nentity
        self.nrelation = nrelation
        self.is_gpt = is_gpt
        self.model_name = model_name
        self.config_model = config_model
        
        self.kg = kg
        self.mapper = KGNameMapper(kg)   # 关键就在这里
        # For action->NL verbalization: map raw (0-based) ids to names.
        # `tree_to_natural_language` expects entity token E -> raw id (E-1),
        # relation token R -> raw id (abs(R)-1).
        self.id2ent = getattr(kg, "ent_id2name", None)
        self.id2rel = getattr(kg, "rel_id2name", None)
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.is_act = ('act' in str(model_name))

        print(f'nrelation: {nrelation}\n,nentity: {nentity}\n')

        print(f"[CtrlHGenAdapter] Using checkpoint: {self.checkpoint_path}")
        print(f"[CtrlHGenAdapter] Using device: {self.device}")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        print("[CtrlHGenAdapter] Loading tokenizer...")
        self.tokenizer, self.vocab_size = create_tokenizer(
            special_tokens=special_tokens,
            offset=offset,
            nentity=nentity,
            nrelation=nrelation,
            is_gpt=is_gpt,
        )
        print("\n===== TOKENIZER DEBUG =====")
        print("len(tokenizer) =", len(self.tokenizer))
        print("tokenizer.vocab_size =", self.tokenizer.vocab_size)

        for tok in ["PAD", "END", "START", "SEP", "4", "44", "i", "u", "n"]:
            try:
                print(tok, "->", self.tokenizer.convert_tokens_to_ids(tok))
            except Exception as e:
                print(tok, "-> ERROR:", e)
        print("===========================\n")


        print("[CtrlHGenAdapter] Building model...")
        self.model = create_transformer(
            ntoken=self.vocab_size,
            special_tokens=self.special_tokens,
            model_name=self.model_name,
            config_model=self.config_model,
        ).to(self.device)

        print("\n===== MODEL DEBUG =====")
        print("model.config.vocab_size =", self.model.config.vocab_size)
        print("embedding.shape =", self.model.get_input_embeddings().weight.shape)
        print("=======================\n")

        print("[CtrlHGenAdapter] Loading checkpoint weights...")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], torch.nn.Module):
            saved_state = ckpt["model"].state_dict()
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            saved_state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            saved_state = ckpt["model"]
        else:
            saved_state = ckpt
        self.model.load_state_dict(saved_state, strict=False)

        self.model.eval()
        print("[CtrlHGenAdapter] Model ready.")

    def _generation_max_length(self) -> int:
        """
        Align with main_reverse constrained_inference/test_loop.
        """
        return int(self.max_gen_len + self.src_len * (self.is_gpt is True))

    def _normalize_pattern(self, value: str) -> str:
        pattern_map = {
            "1p": "(p,(e))",
            "2i": "(i,(p,(e)),(p,(e)))",
            "3in": "(i,(i,(p,(e)),(p,(e))),(n,(p,(e))))",
        }
        return pattern_map.get(value, value)


    def _build_prompt_from_observation(self, observation_entities):
        return " ; ".join(observation_entities)

    def _tree_to_query_tokens(self, node: Dict[str, Any]) -> List[Any]:
        """
        Convert action tree into dataset-style query tokens used in *-a2q-nl.jsonl.
        Example:
          path(rel=-9, child=entity(id=5530)) ->
            ["(", "p", "(", -9, ")", "(", "e", "(", 5530, ")", ")", ")"]
        Note:
        - Action tree here is parsed from unshifted output, so entity id is raw 0-based.
        - Relation token is kept as-is (usually negative token like -(rid+1)).
        """
        t = node["type"]

        if t == "entity":
            raw_eid = int(node["id"])
            return ["(", "e", "(", raw_eid, ")", ")"]

        if t == "path":
            rel = int(node["rel"])
            child_tokens = self._tree_to_query_tokens(node["child"])
            return ["(", "p", "(", rel, ")", *child_tokens, ")"]

        if t == "intersection":
            a, b = node["children"]
            return ["(", "i", *self._tree_to_query_tokens(a), *self._tree_to_query_tokens(b), ")"]

        if t == "union":
            a, b = node["children"]
            return ["(", "u", *self._tree_to_query_tokens(a), *self._tree_to_query_tokens(b), ")"]

        if t == "negation":
            return ["(", "n", *self._tree_to_query_tokens(node["child"]), ")"]

        raise ValueError(f"Unsupported action tree node type: {t}")

    def _normalize_conditions(self, parsed: dict) -> list:
        """
        Normalize parsed control into:
          [{"type": <cond_type>, "value": <cond_value>}, ...]
        """
        alias_map = {
            "specific_entity": "entity",
            "entitynumber": "entity_number",
            "relationnumber": "relation_number",
        }
        allowed = {
            "unconditional",
            "pattern",
            "relation_number",
            "entity_number",
            "relation",
            "entity",
        }

        normalized = []
        raw_conditions = parsed.get("conditions")
        if isinstance(raw_conditions, list):
            for item in raw_conditions:
                if not isinstance(item, dict):
                    continue
                ctype = alias_map.get(str(item.get("type", "")).strip(), str(item.get("type", "")).strip())
                if ctype not in allowed:
                    continue
                normalized.append({"type": ctype, "value": item.get("value", "")})

        # Backward compatibility: single condition fields.
        if not normalized:
            ctype = alias_map.get(
                str(parsed.get("condition_type", "unconditional")).strip(),
                str(parsed.get("condition_type", "unconditional")).strip(),
            )
            cvalue = parsed.get("condition_value", "")
            if ctype not in allowed:
                ctype, cvalue = "unconditional", ""
            normalized = [{"type": ctype, "value": cvalue}]

        if not normalized:
            normalized = [{"type": "unconditional", "value": ""}]

        if len(normalized) > 1 and any(c["type"] != "unconditional" for c in normalized):
            normalized = [c for c in normalized if c["type"] != "unconditional"]
            if not normalized:
                normalized = [{"type": "unconditional", "value": ""}]

        return normalized

    def _condition_to_source_token(self, cond_type: str, cond_value, mapped_values: list):
        if cond_type == "unconditional":
            return None

        if cond_type == "entity":
            cond_entity_id = self.mapper.get_entity_id(str(cond_value))
            mapped_values.append({"type": "entity", "name": cond_value, "id": cond_entity_id})
            shifted_cond_entity_id = ans_shift_indices([cond_entity_id])[0]
            return str(int(shifted_cond_entity_id))

        if cond_type == "relation":
            cond_rel_id = self.mapper.get_relation_id(str(cond_value))
            mapped_values.append({"type": "relation", "name": cond_value, "id": cond_rel_id})
            shifted_rel = -(abs(cond_rel_id) + 1)
            return str(int(shifted_rel))

        if cond_type == "entity_number":
            return self._format_count_condition(cond_value, suffix="e")

        if cond_type == "relation_number":
            return self._format_count_condition(cond_value, suffix="p")

        if cond_type == "pattern":
            normalized_pattern = self._normalize_pattern(str(cond_value).strip())
            mapped_values.append({"type": "pattern", "raw": cond_value, "normalized": normalized_pattern})
            return normalized_pattern

        raise NotImplementedError(f"Unsupported condition_type: {cond_type}")

    def _format_count_condition(self, cond_value, suffix: str) -> str:
        """
        Match the condition style used by main_reverse prompt construction.
        Examples:
          3 / "3" / "3p" / "3 p" -> "3 p" (suffix='p')
          2 / "2" / "2e" / "2 e" -> "2 e" (suffix='e')
        """
        text = str(cond_value).strip().lower()
        if not text:
            raise ValueError("Empty count condition value")

        # Keep digits only for count; tolerate formats like "3p" / "3 p".
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            raise ValueError(f"Invalid count condition value: {cond_value}")
        # IMPORTANT: keep count condition tokenizable by the fixed word-level vocab.
        # "3p"/"4e" become UNK, while "3 p"/"4 e" are recognized tokens.
        return f"{int(digits)} {suffix}"

    def generate(
        self,
        model_input: dict,
        temperature: float = 1.0,
        top_k: int = 0,
        constrained: bool = True,
    ) -> dict:
        source_text = model_input["source_text"]

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        batch = self.tokenizer(
            [source_text],
            padding="longest",
            max_length=self.src_len,
            # Match main_reverse/main.py GPT generation behavior:
            # keep the textual "SEP condition" inside source_text, and still let
            # the tokenizer append the trailing SEP that marks target generation.
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.device)
        self.tokenizer.padding_side = original_padding_side

        print("\n===== GENERATION INPUT DEBUG =====")
        print("source_text =", source_text)
        print("input_ids =", batch.input_ids[0].tolist())
        print("attention_mask =", batch.attention_mask[0].tolist())
        print("decoded_input =", self.tokenizer.decode(batch.input_ids[0], skip_special_tokens=False))
        print("==================================\n")

        prefix_allowed_tokens_fn = None
        if self.is_act and constrained:
            prefix_allowed_tokens_fn = PrefixAllowedTokensFn(
                offset=self.offset,
                nentity=self.nentity,
                nrelation=self.nrelation,
                tokenizer=self.tokenizer,
                allow_entity_as_first_token=False,
            )

        # Keep test-time decoding consistent with main_reverse: always sample.
        generation_temperature = float(temperature) if float(temperature) > 0.0 else 1.0

        # T5 (encoder-decoder): mask out the EOS appended to source by the post-processor,
        # matching the attention_mask treatment in main_reverse.py test loop.
        input_attention_mask = batch.attention_mask
        if not self.is_gpt:
            input_attention_mask = input_attention_mask.clone()
            input_attention_mask[batch.input_ids == self.tokenizer.eos_token_id] = 0

        outputs = generate_with_constraints(
            model=self.model,
            input_ids=batch.input_ids,
            attention_mask=input_attention_mask,
            max_length=self._generation_max_length(),
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            top_p=1.0,
            top_k=int(top_k),
            do_sample=True,
            temperature=generation_temperature,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        # GPT outputs source+target in one sequence; slice off the source prefix.
        # T5 outputs only decoder tokens directly — no slicing needed.
        if self.is_gpt:
            new_ids = outputs[0][batch.input_ids.shape[1]:].detach().cpu().tolist()
        else:
            new_ids = outputs[0].detach().cpu().tolist()
        pred_text_shifted = self.decode_action_ids(new_ids)
        pred_text_unshifted = self._unshift_action_text(pred_text_shifted)

        if self.is_gpt:
            source_attention_mask = batch.attention_mask
            bsz = outputs.shape[0]
            diff = outputs.shape[-1] - source_attention_mask.shape[-1]
            prefix_mask = torch.cat(
                [
                    source_attention_mask,
                    torch.zeros((bsz, diff), dtype=torch.bool, device=self.device),
                ],
                dim=1,
            ).to(self.device)
            outputs[prefix_mask == 1] = self.tokenizer.pad_token_id
        pred_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


        print('pred_de')
        print(pred_decoded[:5])

        # 解析模型输出的 action string -> query/query_nl
        from akgr.agent.action_to_nl import (
            action_string_to_tree,
            action_string_to_tree_prefix,
        )
        try:
            # Some checkpoints occasionally emit multiple expressions concatenated.
            # Prefer parsing the first valid expression so we can still verbalize.
            try:
                tree = action_string_to_tree(pred_text_unshifted)  # strict
                trailing = []
            except Exception:
                tree, trailing = action_string_to_tree_prefix(pred_text_unshifted)  # tolerant
            query = self._tree_to_query_tokens(tree)
            query_nl = query_to_natural_language(
                query_tokens=query,
                ent_id2name=getattr(self, "id2ent", None),
                rel_id2name=getattr(self, "id2rel", None),
            )
            if trailing:
                print("[POSTPROCESS WARNING] Trailing tokens ignored:", trailing[:50], "..." if len(trailing) > 50 else "")
        except Exception as e:
            query = None
            query_nl = None
            print("[POSTPROCESS ERROR]", e)

        # 输出原始文本、结构化查询和自然语言解释
        print("[DEBUG] Predicted text (action string, shifted):", pred_text_shifted)
        print("[DEBUG] Predicted text (action string, unshifted):", pred_text_unshifted)
        print("[DEBUG] Query:", query)
        print("[DEBUG] Query NL:", query_nl)

        entity_number = None
        relation_number = None
        if isinstance(query, list):
            query_text = " ".join(str(tok) for tok in query)
            entity_number, relation_number = number_to_epnumber(query_text)

        return {
            "source_text": source_text,
            "raw_output": pred_text_unshifted,
            "raw_output_shifted": pred_text_shifted,
            "raw_output_unshifted": pred_text_unshifted,
            "query": query,
            "query_nl": query_nl,
            "entitynumber": entity_number,
            "relationnumber": relation_number,
        }
    def build_model_input(self, parsed: dict) -> dict:
        obs_names = parsed.get("observation_entities", [])
        conditions = self._normalize_conditions(parsed)

        if not obs_names:
            raise ValueError("parsed['observation_entities'] is empty")

        obs_ids = [self.mapper.get_entity_id(name) for name in obs_names]
        shifted_obs_ids = ans_shift_indices(obs_ids)
        mapped_cond_values = []
        cond_tokens = []
        for cond in conditions:
            token = self._condition_to_source_token(
                cond_type=cond["type"],
                cond_value=cond.get("value", ""),
                mapped_values=mapped_cond_values,
            )
            if token:
                cond_tokens.append(token)

        if cond_tokens:
            # Use "SEP" token directly (not "[SEP]"), consistent with fixed vocab.
            source_text = f"{list_to_str(shifted_obs_ids)} SEP {' '.join(cond_tokens)}"
        else:
            source_text = list_to_str(shifted_obs_ids)

        return {
            "observation_entities": obs_names,
            "observation_entity_ids": obs_ids,
            "shifted_observation_entity_ids": shifted_obs_ids,
            "conditions": conditions,
            "condition_type": conditions[0]["type"] if conditions else "unconditional",
            "condition_value": conditions[0]["value"] if conditions else "",
            "condition_mapped_value": mapped_cond_values[0] if mapped_cond_values else None,
            "condition_mapped_values": mapped_cond_values,
            "source_text": source_text,
        }
    def decode_action_ids(self, generated_ids):
        """
        将模型生成的 token ids 解码成 action string。
        例如:
            [15, 24910, 10545] -> "i -9 5531"
        """
        decoded = self.tokenizer.batch_decode([generated_ids], skip_special_tokens=False)[0]
        stop_tokens = {
            tok
            for tok in [self.tokenizer.eos_token, self.tokenizer.sep_token]
            if tok is not None
        }
        stop_tokens.update({"END", "<END>", "</s>"})
        skip_tokens = {
            tok
            for tok in [self.tokenizer.pad_token, self.tokenizer.bos_token]
            if tok is not None
        }

        tokens = []
        for tok in decoded.split():
            if tok in stop_tokens:
                break
            if tok in skip_tokens:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def _unshift_action_text(self, action_text: str) -> str:
        """
        Convert shifted action string to unshifted form for downstream parsing/NL.
        entity: e_shifted -> e_shifted - 1
        relation: r_shifted -> -(abs(r_shifted) - 1)
        """
        if not action_text:
            return action_text

        out_tokens = []
        for tok in action_text.split():
            if tok in {"i", "u", "n"} or tok.startswith("<UNK:"):
                out_tokens.append(tok)
                continue
            try:
                val = int(tok)
            except ValueError:
                out_tokens.append(tok)
                continue

            if val > 0:
                out_tokens.append(str(val - 1))
            elif val < 0:
                out_tokens.append(str(-(abs(val) - 1)))
            else:
                out_tokens.append("0")
        return " ".join(out_tokens)