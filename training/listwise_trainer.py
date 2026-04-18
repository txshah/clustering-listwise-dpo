"""
ListwiseTrainer — LIPO-λ loss adapted for text (no images).

Loss source: lpoi_dpo_trainer_5img.py:1537-1542 (cascading softmax)
Scoring fn:  score_i = beta * (policy_logps_i - ref_logps_i)
Lambda:      each cascade level weighted by λ_i (higher rank = higher weight)

Cascade structure (1 chosen + 4 rejected, ranked best→worst):
  losses1 = -log[ exp(s_chosen) / (s_chosen + s_r1 + s_r2 + s_r3 + s_r4) ]
  losses2 = -log[ exp(s_r1)     / (s_r1 + s_r2 + s_r3 + s_r4) ]
  losses3 = -log[ exp(s_r2)     / (s_r2 + s_r3 + s_r4) ]
  losses4 = -log[ exp(s_r3)     / (s_r3 + s_r4) ]
  total   = λ1*losses1 + λ2*losses2 + λ3*losses3 + λ4*losses4
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorWithPadding
from dataclasses import dataclass
from typing import Any


# ── Log-prob computation ───────────────────────────────────────────────────────

def get_per_sample_logps(model, input_ids, attention_mask, labels):
    """
    Returns per-sample average log prob over response tokens only.
    labels has -100 for prompt tokens (masked out of loss).

    shape: input_ids [B, T] → returns [B]
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, T, V]

    shift_logits = logits[:, :-1, :].contiguous()          # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()              # [B, T-1]

    log_probs    = F.log_softmax(shift_logits, dim=-1)
    token_logps  = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    response_mask = (shift_labels != -100).float()         # [B, T-1]
    per_sample = (token_logps * response_mask).sum(-1) / response_mask.sum(-1).clamp(min=1)
    return per_sample                                      # [B]


# ── LIPO-λ loss ────────────────────────────────────────────────────────────────

def lipo_lambda_loss(
    chosen_score,
    r1_score, r2_score, r3_score, r4_score,
    lambdas: tuple[float, float, float, float] = (1.0, 0.75, 0.5, 0.25),
) -> torch.Tensor:
    """
    Cascading softmax loss over 5 rank levels (verbatim from LPOI, eq. in context doc).
    lambdas: importance weights per cascade level, index 0 = top level (chosen vs all).
    """
    def logsumexp(*scores):
        return torch.logsumexp(torch.stack(list(scores), dim=-1), dim=-1)

    losses1 = -(chosen_score - logsumexp(chosen_score, r1_score, r2_score, r3_score, r4_score))
    losses2 = -(r1_score     - logsumexp(r1_score, r2_score, r3_score, r4_score))
    losses3 = -(r2_score     - logsumexp(r2_score, r3_score, r4_score))
    losses4 = -(r3_score     - logsumexp(r3_score, r4_score))

    λ1, λ2, λ3, λ4 = lambdas
    loss = (λ1 * losses1 + λ2 * losses2 + λ3 * losses3 + λ4 * losses4).mean()
    return loss


# ── Dataset ────────────────────────────────────────────────────────────────────

class ListwiseDataset(Dataset):
    """
    Tokenises (prompt, chosen, rejected1-4) and returns label-masked tensors.
    labels = -100 for prompt tokens, token ids for response tokens.
    """

    KEYS = ["chosen", "rejected1", "rejected2", "rejected3", "rejected4"]

    def __init__(self, data: list[dict], tokenizer, max_length: int = 1024):
        self.data      = data
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def _encode(self, prompt: str, response: str) -> dict:
        prompt_ids   = self.tokenizer.encode(prompt,   add_special_tokens=True)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        response_ids = response_ids + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + response_ids)[: self.max_len]
        labels    = ([-100] * len(prompt_ids) + response_ids)[: self.max_len]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids":       torch.tensor(input_ids,       dtype=torch.long),
            "attention_mask":  torch.tensor(attention_mask,  dtype=torch.long),
            "labels":          torch.tensor(labels,          dtype=torch.long),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        prompt = item["prompt"]
        out    = {}
        for key in self.KEYS:
            enc = self._encode(prompt, item[key])
            for k, v in enc.items():
                out[f"{key}_{k}"] = v
        return out


# ── Data collator ──────────────────────────────────────────────────────────────

@dataclass
class ListwiseDataCollator:
    """Pads each of the 5 sequence groups independently."""

    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = {}
        keys  = ["chosen", "rejected1", "rejected2", "rejected3", "rejected4"]

        for key in keys:
            input_ids_list      = [f[f"{key}_input_ids"]      for f in features]
            attention_mask_list = [f[f"{key}_attention_mask"]  for f in features]
            labels_list         = [f[f"{key}_labels"]          for f in features]

            max_len = max(x.size(0) for x in input_ids_list)

            def pad(tensors, pad_val):
                return torch.stack([
                    F.pad(t, (0, max_len - t.size(0)), value=pad_val)
                    for t in tensors
                ])

            batch[f"{key}_input_ids"]      = pad(input_ids_list,      self.pad_token_id)
            batch[f"{key}_attention_mask"] = pad(attention_mask_list,  0)
            batch[f"{key}_labels"]         = pad(labels_list,         -100)

        return batch


# ── Trainer ────────────────────────────────────────────────────────────────────

class ListwiseTrainer(Trainer):
    """
    Extends HF Trainer with LIPO-λ loss.
    ref_model must be the frozen SFT checkpoint (same architecture).
    """

    def __init__(self, ref_model, beta: float, lambdas: tuple, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model.eval()
        self.beta      = beta
        self.lambdas   = lambdas
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def _get_logps(self, model, key: str, inputs: dict) -> torch.Tensor:
        return get_per_sample_logps(
            model,
            inputs[f"{key}_input_ids"].to(model.device),
            inputs[f"{key}_attention_mask"].to(model.device),
            inputs[f"{key}_labels"].to(model.device),
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        keys = ["chosen", "rejected1", "rejected2", "rejected3", "rejected4"]

        # Policy log probs (with grad)
        policy_logps = {k: self._get_logps(model, k, inputs) for k in keys}

        # Reference log probs (no grad)
        with torch.no_grad():
            ref_logps = {k: self._get_logps(self.ref_model, k, inputs) for k in keys}

        # Scores: β * (policy - ref)  — the scoring function from the LPOI doc
        scores = {k: self.beta * (policy_logps[k] - ref_logps[k]) for k in keys}

        loss = lipo_lambda_loss(
            scores["chosen"],
            scores["rejected1"], scores["rejected2"],
            scores["rejected3"], scores["rejected4"],
            lambdas=self.lambdas,
        )

        return (loss, None) if return_outputs else loss
