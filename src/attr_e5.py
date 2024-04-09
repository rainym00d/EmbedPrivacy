import argparse
from dataclasses import dataclass
from collections import OrderedDict
import json
import datasets
import torch
import os
from transformers import (
    AutoTokenizer,
    BertModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from typing import List, Optional, Tuple, Union


@dataclass
class MyDataCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        text = [f["text"] for f in features]
        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        return inputs

def preprocess_logits_for_metrics(logits, labels):
    return F.normalize(logits[1], p=2, dim=1)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
    r"""
    encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
        the model is configured as a decoder.
    encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
        the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.
    past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
        Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

        If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
        don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
        `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    use_cache (`bool`, *optional*):
        If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
        `past_key_values`).
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if self.config.is_decoder:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
    else:
        use_cache = False

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    batch_size, seq_length = input_shape
    device = input_ids.device if input_ids is not None else inputs_embeds.device

    # past_key_values_length
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    if attention_mask is None:
        attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

    if token_type_ids is None:
        if hasattr(self.embeddings, "token_type_ids"):
            buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    embedding_output = self.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
        inputs_embeds=inputs_embeds,
        past_key_values_length=past_key_values_length,
    )
    encoder_outputs = self.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    sequence_output = encoder_outputs[0]
    # * different from original code
    pooled_output = average_pool(sequence_output, attention_mask)

    if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        past_key_values=encoder_outputs.past_key_values,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions,
    )


def main():
    # * parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_list", type=str, required=True)
    parser.add_argument("--dataset_save_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args, unknown = parser.parse_known_args()
    
    dataset_list = [dataset.strip() for dataset in args.dataset_list.split(",")]
    
    dataset_dict = OrderedDict()
    for dataset_name in dataset_list:
        path = os.path.join(args.dataset_save_dir, f"attr_{dataset_name}.json")
        dataset_dict[dataset_name] = datasets.Dataset.from_json(path, cache_dir="cache")
    
    # * load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
    BertModel.forward = forward
    model = BertModel.from_pretrained("intfloat/e5-large-v2")
    
    # * set trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=100,
            remove_unused_columns=False,
            dataloader_num_workers=32,
        ),
        data_collator=MyDataCollator(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    all_results = {}
    for dataset_name, test_dataset in dataset_dict.items():
        sample_size = len(test_dataset)
        # * sentence to embedding
        new_test_dataset = datasets.Dataset.from_dict({
            "text":sum(test_dataset["pred"], [])
        })
        test_result = trainer.predict(new_test_dataset)
        text_embeddings = test_result.predictions
        text_embeddings = text_embeddings.reshape(sample_size, 10, -1)

        # * test each attribute
        # * get all attribute in test dataset
        attribute_list = []
        for k in test_dataset.features.keys():
            if k not in ["text", "pred"] and "candidate_list" not in k:
                attribute_list.append(k)
        
        for attr in attribute_list:
            candidate_size = len(test_dataset[0][f"{attr}_candidate_list"])
            new_test_dataset = {"text": []}
            for example in test_dataset:
                for label in example[f"{attr}_candidate_list"]:
                    prompt = f"The {' '.join(attr.split('_'))} is {label}"
                    new_test_dataset["text"].append(prompt)
            new_test_dataset = datasets.Dataset.from_dict(new_test_dataset)
            
            # * attr to embedding
            test_result = trainer.predict(new_test_dataset)
            attr_embeddings = test_result.predictions
            attr_embeddings = attr_embeddings.reshape(sample_size, candidate_size, -1)
            
            if trainer.is_world_process_zero():
                metrics = [0] * 10
                for i in range(sample_size):
                    scores = text_embeddings[i] @ attr_embeddings[i].T
                    answers = scores.argmax(axis=-1)
                    for idx, answer in enumerate(answers):
                        if answer == 0:
                            metrics[idx] += 1
                metrics = [metric / sample_size for metric in metrics]
                all_results[f"{dataset_name}_{attr}"] = metrics

    if trainer.is_world_process_zero():
        output_file = os.path.join(args.output_dir, "attr_results.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=4)
            å

if __name__ == "__main__":
    main()
