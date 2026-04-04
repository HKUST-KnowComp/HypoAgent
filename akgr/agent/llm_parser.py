import json
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


SYSTEM_PROMPT = """
You are a condition parser for a controllable logical hypothesis generation system.

You must return JSON only. No prose, no markdown fences, no comments.

Required output schema:
{
  "observation_entities": [string, ...],
  "conditions": [
    {
      "type": "unconditional" | "pattern" | "relation_number" | "entity_number" | "relation" | "entity",
      "value": string
    }
  ]
}

Hard constraints:
- `conditions` must be a flat JSON array of objects. Never nest lists inside `conditions`.
- Every condition object must contain both `type` and `value`.
- `value` must always be a string.
- Allowed condition types are exactly: `unconditional`, `pattern`, `relation_number`, `entity_number`, `relation`, `entity`.
- Do not invent unsupported fields.
- Do not copy observation entity names into `relation` or `entity` conditions unless the user explicitly asked for them.

Intent rules:
1. "simpler"
   - Prefer count-based constraints only.
   - Decrease specificity by using `relation_number` and/or `entity_number`.
   - Do not output `relation` or `entity` for this intent unless the user explicitly mentioned a relation/entity name.

2. "more specific"
   - Prefer count-based constraints only.
   - Increase specificity by using `relation_number` and/or `entity_number`.
   - Do not output `relation` or `entity` for this intent unless the user explicitly mentioned a relation/entity name.

3. "from <something> perspective"
   - Extract <something> as a relation.
   - Add {"type": "relation", "value": "<something>"}.

4. "related to <something>"
   - Extract <something> as an entity.
   - Add {"type": "entity", "value": "<something>"}.

Few-shot examples:

Example 1
Input condition request: more specific
Output:
{"observation_entities":[],"conditions":[{"type":"relation_number","value":"3"},{"type":"entity_number","value":"4"}]}

Example 2
Input condition request: simpler
Output:
{"observation_entities":[],"conditions":[{"type":"relation_number","value":"1"},{"type":"entity_number","value":"2"}]}

Example 3
Input condition request: from spouse perspective
Output:
{"observation_entities":[],"conditions":[{"type":"relation","value":"spouse"}]}

Example 4
Input condition request: related to shark
Output:
{"observation_entities":[],"conditions":[{"type":"entity","value":"shark"}]}

Example 5
Input condition request: more specific and related to Triakis
Output:
{"observation_entities":[],"conditions":[{"type":"relation_number","value":"3"},{"type":"entity_number","value":"4"},{"type":"entity","value":"Triakis"}]}
"""

REPAIR_PROMPT = SYSTEM_PROMPT + """

You are repairing a previously generated JSON object after a downstream validation or execution error.

Repair rules:
- Keep the user's intent unchanged.
- Fix the JSON so it matches the schema and avoids the reported error.
- If the error says a relation/entity name was not found, replace that condition with a more faithful valid interpretation instead of repeating the same bad value.
- If the original request is only about "more specific" or "simpler", output only count-based constraints unless the user explicitly named a relation/entity.
"""


class LocalQwenParser:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Default to CPU to avoid fighting for GPU memory with CtrlHGen.
        # Set AKGR_LLM_DEVICE_MAP=auto to enable accelerator placement.
        device_map = os.environ.get("AKGR_LLM_DEVICE_MAP", "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map
        )

    def _generate_text(self, messages, max_new_tokens=256):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        return response.strip()

    def _extract_json(self, text: str):
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Parser output is not valid JSON: {text}")

    def _build_condition_user_prompt(
        self,
        condition_text: str,
        session_memory: dict = None,
        failed_parse: dict = None,
        error_message: str = None,
    ) -> str:
        lines = [
            f"Session memory: {json.dumps(session_memory or {}, ensure_ascii=False)}",
            f"Current condition request: {condition_text}",
        ]
        if failed_parse is not None:
            lines.append(f"Previous JSON to repair: {json.dumps(failed_parse, ensure_ascii=False)}")
        if error_message:
            lines.append(f"Downstream error to fix: {error_message}")
        return "\n".join(lines)

    def _parse_condition_messages(
        self,
        messages,
        condition_text: str,
        session_memory: dict = None,
    ):
        raw_text = self._generate_text(messages, max_new_tokens=256)
        parsed = self._extract_json(raw_text)
        parsed = self._normalize_conditions(parsed, session_memory=session_memory or {})
        parsed.setdefault("condition_prompt_hints", [])
        parsed.setdefault("condition_interpretation", {})
        parsed["raw_condition_text"] = condition_text
        return parsed

    def _normalize_conditions(self, parsed: dict, session_memory: dict = None) -> dict:
        """
        Normalize parser output to a stable schema:
          - conditions: list[{type, value}]
        Keep backward compatibility with old fields:
          - condition_type / condition_value
        """
        if not isinstance(parsed, dict):
            parsed = {}

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

        def normalize_value(value):
            if value is None:
                return ""
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)

        def normalize_single_condition(item):
            if not isinstance(item, dict):
                return None
            ctype = alias_map.get(str(item.get("type", "")).strip(), str(item.get("type", "")).strip())
            if ctype not in allowed:
                return None
            return {"type": ctype, "value": normalize_value(item.get("value", ""))}

        def normalize_condition_list(raw_conditions, fallback_type=None, fallback_value=None):
            out = []
            if isinstance(raw_conditions, dict):
                raw_conditions = [raw_conditions]
            if isinstance(raw_conditions, list):
                for item in raw_conditions:
                    if isinstance(item, list):
                        out.extend(normalize_condition_list(item))
                        continue
                    normalized_item = normalize_single_condition(item)
                    if normalized_item is not None:
                        out.append(normalized_item)
            if not out and fallback_type is not None:
                ctype = alias_map.get(str(fallback_type).strip(), str(fallback_type).strip())
                cvalue = normalize_value(fallback_value)
                if ctype in allowed:
                    out = [{"type": ctype, "value": cvalue}]
            return out

        # Current-turn conditions from LLM output.
        current_conditions = normalize_condition_list(
            parsed.get("conditions"),
            fallback_type=parsed.get("condition_type", "unconditional"),
            fallback_value=parsed.get("condition_value", ""),
        )

        # Previous-turn conditions from memory for additive/override merge.
        previous_conditions = []
        if isinstance(session_memory, dict):
            previous_parsed = session_memory.get("previous_parsed_control") or {}
            previous_conditions = normalize_condition_list(
                previous_parsed.get("conditions"),
                fallback_type=previous_parsed.get("condition_type"),
                fallback_value=previous_parsed.get("condition_value", ""),
            )
            if not previous_conditions:
                previous_conditions = normalize_condition_list(
                    session_memory.get("last_conditions"),
                    fallback_type=session_memory.get("last_condition_type"),
                    fallback_value=session_memory.get("last_condition_value", ""),
                )

        prev_concrete = [
            c for c in previous_conditions
            if isinstance(c, dict) and c.get("type") != "unconditional"
        ]
        curr_concrete = [
            c for c in current_conditions
            if isinstance(c, dict) and c.get("type") != "unconditional"
        ]

        # Merge policy:
        # - If current turn provides concrete constraints, replace same-type previous
        #   constraints and keep untouched previous types.
        # - If current turn does not provide concrete constraints, keep previous ones.
        if curr_concrete:
            curr_by_type = {c["type"]: c for c in curr_concrete}
            merged = []
            for p in prev_concrete:
                ptype = p["type"]
                if ptype in curr_by_type:
                    merged.append(curr_by_type.pop(ptype))
                else:
                    merged.append(p)
            merged.extend(curr_by_type.values())
            normalized = merged
        elif prev_concrete:
            normalized = prev_concrete
        else:
            normalized = [{"type": "unconditional", "value": ""}]

        # Safety: if still empty, keep unconditional.
        if not normalized:
            normalized = [{"type": "unconditional", "value": ""}]

        parsed["conditions"] = normalized
        # Keep old fields for downstream compatibility.
        parsed["condition_type"] = normalized[0]["type"]
        parsed["condition_value"] = normalized[0]["value"]
        return parsed


    def parse_user_text(self, user_text: str, session_memory: dict):
        user_prompt = (
            f"Session memory: {json.dumps(session_memory, ensure_ascii=False)}\n"
            f"Current user request: {user_text}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw_text = self._generate_text(messages, max_new_tokens=256)
        parsed = self._extract_json(raw_text)
        return self._normalize_conditions(parsed, session_memory=session_memory)

    def parse_condition_text(self, condition_text: str, session_memory: dict = None):
        user_prompt = self._build_condition_user_prompt(
            condition_text=condition_text,
            session_memory=session_memory,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self._parse_condition_messages(
            messages,
            condition_text=condition_text,
            session_memory=session_memory,
        )

    def repair_condition_text(
        self,
        condition_text: str,
        session_memory: dict = None,
        failed_parse: dict = None,
        error_message: str = "",
    ):
        user_prompt = self._build_condition_user_prompt(
            condition_text=condition_text,
            session_memory=session_memory,
            failed_parse=failed_parse,
            error_message=error_message,
        )
        messages = [
            {"role": "system", "content": REPAIR_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self._parse_condition_messages(
            messages,
            condition_text=condition_text,
            session_memory=session_memory,
        )