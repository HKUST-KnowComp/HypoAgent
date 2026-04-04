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
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        min_length=-1,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        temperature=float(temperature),
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )
