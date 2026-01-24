"""Main entry point for Geobias."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra

if TYPE_CHECKING:
    from omegaconf import DictConfig

import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_deterministic(seed: int = 0) -> None:
    """Applies deterministic settings for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path="configs", config_name="generation.yaml", version_base=None)  # type: ignore
def main(cfg: DictConfig) -> None:
    """Main entry point for text generation.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object.
    """
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, token=cfg.model.hf_token if "hf_token" in cfg.model else "", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name, token=cfg.model.hf_token if "hf_token" in cfg.model else "", use_fast=True
    )

    tokenizer.pad_token = tokenizer.eos_token

    messages = [
        {"role": "system", "content": cfg.prompt.system},
        {"role": "user", "content": cfg.prompt.user},
    ]

    make_deterministic(cfg.seed)

    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    enc = tokenizer(
        chat,
        return_tensors="pt",
        padding=True,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    outputs = model.generate(
        **enc,
        max_new_tokens=1000,
        do_sample=True,
        temperature=cfg.model.temperature,
        top_p=cfg.model.top_p,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    main()
