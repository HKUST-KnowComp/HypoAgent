import copy
from typing import Callable, List, Optional

import torch

from akgr.utils.parsing_util import qry_actionprefix_get_branching


class PrefixAllowedTokensFn:
    def __init__(
        self,
        offset: int,
        nentity: int,
        nrelation: int,
        tokenizer,
        allow_entity_as_first_token: bool = True,
    ):
        self.offset = int(offset)
        self.nentity = int(nentity)
        self.nrelation = int(nrelation)
        self.tokenizer = tokenizer
        self.allow_entity_as_first_token = allow_entity_as_first_token
        self.iun_ids = tokenizer.convert_tokens_to_ids(["i", "u", "n"])

    def get_gathered_tokens(self) -> List[int]:
        return list(range(self.offset + self.nentity + self.nrelation))

    def get_non_special_tokens(self) -> List[int]:
        return self.iun_ids + list(
            range(self.offset, self.offset + self.nentity + self.nrelation)
        )

    def get_iun_allowed_tokens(self) -> List[int]:
        return self.iun_ids + list(
            range(
                self.offset + self.nentity,
                self.offset + self.nentity + self.nrelation,
            )
        )

    def __call__(self, batch_id: int, input_ids: torch.LongTensor) -> List[int]:
        if input_ids.shape[-1] <= 1:
            return self.get_gathered_tokens()

        is_gpt = not (input_ids[1] in self.get_iun_allowed_tokens())
        prefix_ids = list(input_ids)

        if is_gpt:
            if self.tokenizer.sep_token_id in prefix_ids:
                sep_pos = prefix_ids.index(self.tokenizer.sep_token_id)
                prefix_ids = prefix_ids[sep_pos:]
            else:
                return self.get_gathered_tokens()

        last_action = prefix_ids[-1]

        if last_action in [self.tokenizer.bos_token_id, self.tokenizer.sep_token_id]:
            if self.allow_entity_as_first_token:
                return self.get_non_special_tokens()
            return self.get_iun_allowed_tokens()

        if last_action in self.iun_ids:
            return self.get_iun_allowed_tokens()

        if self.offset <= last_action < self.offset + self.nentity:
            actionstr_prefix = self.tokenizer.decode(prefix_ids, skip_special_tokens=True)
            branching = qry_actionprefix_get_branching(action_prefix=actionstr_prefix)
            if branching == "EMPTY":
                return [self.tokenizer.eos_token_id]
            return self.get_iun_allowed_tokens()

        if self.offset + self.nentity <= last_action:
            return self.get_non_special_tokens()

        return [self.tokenizer.pad_token_id]


def generate_with_constraints(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    max_length: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    top_k: int = 0,
    top_p: float = 1.0,
    do_sample: bool = True,
    temperature: float = 1.0,
    prefix_allowed_tokens_fn: Optional[Callable] = None,
):
    # Recent Transformers versions reject generation kwargs that override
    # values in ``model.generation_config``.  This repository historically
    # passed all generation controls directly to ``model.generate``; that
    # works with older versions but raises a ValueError in newer runtimes.
    # Copy the config so requests remain isolated, update it explicitly, and
    # pass a single GenerationConfig object instead.
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        from transformers import GenerationConfig

        generation_config = GenerationConfig()
    else:
        generation_config = copy.deepcopy(generation_config)

    generation_config.max_length = int(max_length)
    generation_config.min_length = 0
    generation_config.pad_token_id = int(pad_token_id)
    generation_config.bos_token_id = int(bos_token_id)
    generation_config.eos_token_id = int(eos_token_id)
    generation_config.top_p = float(top_p)
    generation_config.top_k = int(top_k)
    generation_config.do_sample = bool(do_sample)
    generation_config.temperature = float(temperature)

    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )
