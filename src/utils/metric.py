import re
import string
from collections import Counter
from typing import Dict, List

import evaluate
import numpy as np
import torch
from nltk.corpus import stopwords
from transformers import EvalPrediction


class Metric:
    rouge_evalutor = evaluate.load("./src/utils/rouge")
    bleu_evalutor = evaluate.load("./src/utils/bleu")

    @classmethod
    def compute(cls, preds: List[str], labels: List[str]) -> Dict[str, float]:
        result = {}
        result |= cls.f1(preds, labels)
        result |= cls.rouge(preds, labels)
        result |= cls.bleu(preds, labels)

        return result

    @classmethod
    def rouge(cls, preds: List[str], labels: List[str]) -> Dict[str, float]:
        result = cls.rouge_evalutor.compute(predictions=preds, references=labels)
        return result

    @classmethod
    def bleu(cls, preds: List[str], labels: List[str]) -> Dict[str, float]:
        result = {}
        for i in range(1, 5):
            bleu = cls.bleu_evalutor.compute(
                predictions=preds, references=labels, max_order=i
            )["bleu"]
            result[f"bleu{i}"] = bleu
        return result

    @staticmethod
    def f1(preds: List[str], labels: List[str]) -> Dict[str, float]:
        def get_pure_tokens(text: str):
            # * lower
            text = text.lower()
            # * remove punctuation
            punctuation_set = set(string.punctuation)
            text = "".join([ch for ch in text if ch not in punctuation_set])
            # * remove article
            regex = re.compile(
                rf"\b({'|'.join(stopwords.words('english'))})\b", re.UNICODE
            )
            text = re.sub(regex, " ", text)
            # * split text
            return text.split()

        num = len(preds)
        pred_tokens_list = [get_pure_tokens(pred) for pred in preds]
        label_tokens_list = [get_pure_tokens(label) for label in labels]

        common_tokens_num_list = [
            sum((Counter(pred_tokens_list[i]) & Counter(label_tokens_list[i])).values())
            for i in range(num)
        ]

        prec, recall = 0, 0
        for i in range(num):
            if len(pred_tokens_list[i]) == 0 or len(label_tokens_list[i]) == 0:
                is_same = len(pred_tokens_list[i]) == len(label_tokens_list[i])
                prec += is_same
                recall += is_same
            else:
                prec += common_tokens_num_list[i] / len(pred_tokens_list[i])
                recall += common_tokens_num_list[i] / len(label_tokens_list[i])

        prec /= num
        recall /= num

        f1 = (2 * prec * recall) / (prec + recall) if prec + recall != 0 else 0

        result = {
            "precision": prec,
            "recall": recall,
            "f1": f1,
        }
        return result


def get_compute_metrics(tokenizer):
    def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, float]:
        predictions, label_ids = eval_preds

        predictions[predictions == -100] = tokenizer.pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return Metric.compute(decoded_preds, decoded_labels)

    return compute_metrics


def get_choice_preprocess_logits_for_metrics(prompt_token_ids_length):
    def preprocess_logits_for_metrics(logits, labels):
        # * process dim
        labels = labels[:, prompt_token_ids_length:]
        seq_size = labels.shape[-1]
        logits = logits.reshape(-1, *logits.shape[-2:])
        logits = logits[:, -seq_size:, :]
        # * get corresponding logits value
        mask = labels != -100
        indices = labels[mask]
        logits = torch.gather(logits[mask], 1, indices.unsqueeze(1))
        final_logits = torch.zeros_like(labels, dtype=logits.dtype)
        final_logits[mask] = logits.flatten()
        # * calculate mean logits
        final_logits[final_logits == 0] = torch.nan
        final_logits = torch.nanmean(final_logits, dim=-1)

        return final_logits.reshape(-1, 1)

    return preprocess_logits_for_metrics


def get_choice_compute_metrics(candidate_size):
    def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, float]:
        predictions, _ = eval_preds
        predictions = predictions.reshape(-1, candidate_size)
        answer = predictions.argmax(axis=-1)
        result = {"exact_match": (answer == 0).mean()}

        return result

    return compute_metrics
