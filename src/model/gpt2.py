from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, GPT2LMHeadModel, PreTrainedTokenizerBase


class GPT2BasedDecryptModel(GPT2LMHeadModel):
    def __init__(
        self, config: AutoConfig, embedding_dim: int, tokenizer: PreTrainedTokenizerBase
    ):
        super().__init__(config)
        # * init
        self.embedding_dim = embedding_dim
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        # * projection layer
        self.proj_layer = nn.Sequential(
            nn.Linear(embedding_dim, self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        if labels is not None:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device

            # * input embeddings = sentence embedding + <eos> + label embeddings
            # special token
            special_token_ids = torch.full(
                [batch_size, 1], self.eos_token_id, device=device
            )
            special_token_embeds = self.transformer.wte(special_token_ids)
            # sentence embedding
            inputs_embeds = inputs_embeds.to(special_token_embeds.dtype)
            inputs_embeds = self.proj_layer(inputs_embeds)
            inputs_embeds = inputs_embeds.unsqueeze(1)
            # label embeddings
            _labels = labels.clone().to(device)
            _labels[labels == -100] = self.pad_token_id
            label_embeds = self.transformer.wte(_labels)
            # input emebeds
            inputs_embeds = torch.cat(
                [inputs_embeds, special_token_embeds, label_embeds], axis=1
            )
            # * labels = paddings + labels
            padding_size = inputs_embeds.shape[1] - labels.shape[1]
            paddings = torch.full([batch_size, padding_size], -100, device=device)
            labels = torch.cat([paddings, labels], axis=1)

        return super().forward(
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs,
        )

    def generate(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

        # * input embeddings = sentence embedding + <eos> + label embeddings
        # special token
        special_token_ids = torch.full(
            [batch_size, 1], self.eos_token_id, device=device
        )
        special_token_embeds = self.transformer.wte(special_token_ids)
        # sentence embedding
        inputs_embeds = inputs_embeds.to(special_token_embeds.dtype)
        inputs_embeds = self.proj_layer(inputs_embeds)
        inputs_embeds = inputs_embeds.unsqueeze(1)
        # input emebeds
        inputs_embeds = torch.cat([inputs_embeds, special_token_embeds], axis=1)

        return super().generate(
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
