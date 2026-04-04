import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re
import json


VERBALIZER_PROMPT = """
You are a helpful assistant.

You will receive:
1) observation entities extracted from the user's input
2) a hypothesis sentence (query_nl)

Treat the observation entities as what the user observed.
Use the hypothesis to explain that observation in fluent natural language.
Keep the explanation faithful to the hypothesis and do not invent facts.
The final sentence must always be exactly:
If your current question has been fully answered and you want to ask a new one, please enter 1.
"""

RESET_REMINDER = "If your current question has been fully answered and you want to ask a new one, please enter 1."

FOLLOWUP_PROMPT = """
You are a KG assistant that asks one short clarification question to narrow down query conditions.

Inputs:
- observation entities
- generated hypothesis (query_nl)
- triples intersecting observation entities

Ask exactly ONE concise English question, focusing on:
1) whether a relation is relevant, OR
2) whether an entity is relevant.

Hard constraints:
- English only.
- One sentence only.
- No explanations.
- Output question text only.
"""

CONDITION_JUDGE_PROMPT = """
You are a strict query-condition consistency judge for a KG system.

Inputs:
- original condition text from user
- parsed conditions
- generated query (token form)
- generated query_nl (natural language explanation)

Task:
- Decide whether generated query satisfies the condition.
- If user condition is empty/unconditional, return true.
- Be strict: if uncertain, return false.

Output format (JSON only, one line):
{"match": true/false, "reason": "short English reason"}
"""


class LocalQwenVerbalizer:
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

    def verbalize(self, observation_entities: list, query_nl: str) -> str:
        prompt = (
            f"observation_entities:\n{observation_entities}\n\n"
            f"query_nl:\n{query_nl}"
        )

        messages = [
            {"role": "system", "content": VERBALIZER_PROMPT},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256,
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
        response = response.strip()
        if not response.endswith(RESET_REMINDER):
            if response:
                response = f"{response}\n{RESET_REMINDER}"
            else:
                response = RESET_REMINDER
        return response

    def judge_condition_match(
        self,
        condition_text: str,
        parsed_conditions: list,
        query_tokens: list,
        query_nl: str,
    ) -> dict:
        condition_text = (condition_text or "").strip()
        if not condition_text:
            return {"match": True, "reason": "Empty condition is treated as unconditional."}

        prompt = (
            f"condition_text:\n{condition_text}\n\n"
            f"parsed_conditions:\n{parsed_conditions}\n\n"
            f"query_tokens:\n{query_tokens}\n\n"
            f"query_nl:\n{query_nl}"
        )
        messages = [
            {"role": "system", "content": CONDITION_JUDGE_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=96,
            do_sample=False,
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()
        return self._parse_condition_judge_response(response)

    def propose_followup_question(
        self,
        observation_entities: list,
        query_nl: str,
        triples: list,
        previous_questions: list = None,
    ) -> str:
        triples_preview = triples[:8]
        previous_questions = previous_questions or []
        prompt = (
            f"observation_entities:\n{observation_entities}\n\n"
            f"query_nl:\n{query_nl}\n\n"
            f"triples:\n{triples_preview}\n\n"
            f"previous_questions_must_not_repeat:\n{previous_questions}"
        )

        messages = [
            {"role": "system", "content": FOLLOWUP_PROMPT},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=64,
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        response = self._sanitize_followup_question(response, triples, previous_questions)
        return response

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        if not text:
            return False
        return re.search(r"[\u3400-\u9fff]", text) is not None

    @staticmethod
    def _normalize_question(text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())

    def _fallback_candidates(self, triples: list) -> list:
        triples = triples or []
        cands = []
        if triples:
            t0 = triples[0]
            rel = str(t0.get("relation_name", "")).strip()
            ent = str(t0.get("intersection_entity_name", "") or t0.get("head_name", "")).strip()
            if rel:
                cands.extend(
                    [
                        f"Is the relation '{rel}' relevant to your question?",
                        f"Does the relation '{rel}' match what you want?",
                        f"Should we focus on the relation '{rel}'?",
                    ]
                )
            if ent:
                cands.extend(
                    [
                        f"Is '{ent}' the entity you want to focus on?",
                        f"Should we focus on '{ent}' as the key entity?",
                        f"Is '{ent}' central to your intent?",
                    ]
                )
        cands.extend(
            [
                "Is any of these candidate relations more relevant?",
                "Which candidate relation is the best fit for your intent?",
                "Do you want to prioritize a specific relation from these candidates?",
            ]
        )
        return cands

    def _build_english_fallback_question(self, triples: list, previous_questions: list = None) -> str:
        previous_questions = previous_questions or []
        seen = {self._normalize_question(q) for q in previous_questions if q}
        for cand in self._fallback_candidates(triples):
            if self._normalize_question(cand) not in seen:
                return cand
        return "Could you clarify which relation or entity should be prioritized?"

    def _sanitize_followup_question(self, question: str, triples: list, previous_questions: list = None) -> str:
        previous_questions = previous_questions or []
        seen = {self._normalize_question(q) for q in previous_questions if q}
        q = (question or "").strip()
        # Hard force English-only output.
        if (not q) or self._contains_cjk(q):
            return self._build_english_fallback_question(triples, previous_questions)
        # Keep one concise English sentence.
        q = q.replace("\n", " ").strip()
        if not q.endswith("?"):
            q = q.rstrip(".! ") + "?"
        if self._normalize_question(q) in seen:
            return self._build_english_fallback_question(triples, previous_questions)
        return q

    @staticmethod
    def _parse_condition_judge_response(text: str) -> dict:
        raw = (text or "").strip()
        if not raw:
            return {"match": False, "reason": "Empty judge response."}

        candidates = []
        try:
            candidates.append(json.loads(raw))
        except Exception:
            pass
        for chunk in re.findall(r"\{.*?\}", raw, flags=re.DOTALL):
            try:
                candidates.append(json.loads(chunk))
            except Exception:
                continue

        for obj in candidates:
            if not isinstance(obj, dict):
                continue
            match = obj.get("match")
            reason = str(obj.get("reason", "")).strip()
            if isinstance(match, bool):
                return {"match": match, "reason": reason or "Parsed from LLM JSON output."}

        lowered = raw.lower()
        if '"match": true' in lowered or "match=true" in lowered:
            return {"match": True, "reason": "Heuristic parse from non-JSON output."}
        if '"match": false' in lowered or "match=false" in lowered:
            return {"match": False, "reason": "Heuristic parse from non-JSON output."}
        return {"match": False, "reason": f"Unparseable judge response: {raw[:120]}".strip()}