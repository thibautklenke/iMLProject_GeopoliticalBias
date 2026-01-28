"""Util functions."""

from __future__ import annotations

import logging
import re
from typing import Any

import nltk
import numpy as np
import torch
from rich.logging import RichHandler
from scipy.stats import ttest_ind
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

TEMPLATES: list[str] = ["this is ", "that is ", "there is ", "the person is ", "here is ", " is here", " is there"]


def setup_logging() -> None:
    """Setup logging module with rich handler."""
    FORMAT = "%(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])


def get_logger(logger_name: str) -> logging.Logger:
    """Get logger instance by name.

    Parameters
    ----------
    logger_name : str
        Name of the logger.

    Returns
    -------
    logging.Logger
        Configured logger object.
    """
    return logging.getLogger(logger_name)


def load_model_for_embedding_retrieval(
    model_name: str, device: str, hf_token: str = ""
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load tokenizer and embedding model from Hugging Face.

    Parameters
    ----------
    model_name : str
        Model identifier from Hugging Face hub.
    device : str
        Device to load model on (e.g., 'cuda', 'cpu').
    hf_token : str, optional
        Hugging Face API token for private models, by default "".

    Returns
    -------
    tuple[PreTrainedTokenizer, PreTrainedModel]
        Tuple of (tokenizer, model) ready for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)

    try:
        embedding_model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, device_map="auto", token=hf_token
        )
    except ValueError:
        # Fallback for models where device_map is not supported (typically smaller, older models)
        embedding_model = AutoModel.from_pretrained(model_name, output_hidden_states=True, token=hf_token).to(device)

    embedding_model.eval()
    return tokenizer, embedding_model


def get_number_of_hidden_states(tokenizer: PreTrainedTokenizer, embedding_model: PreTrainedModel) -> int:
    """Get number of hidden layers in the embedding model.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer for encoding test context.
    embedding_model : PreTrainedModel
        Model to inspect for hidden states.

    Returns
    -------
    int
        Number of hidden layers.
    """
    encoded = tokenizer.encode_plus("test", return_tensors="pt").to(embedding_model.device)
    with torch.no_grad():
        hidden_states = embedding_model(**encoded).hidden_states
    return len(hidden_states)


def get_word_idx(tokenizer: PreTrainedTokenizer, encoded_context: Any, word: str) -> list[int]:
    """Find token indices for a word in encoded context.

    Handles various tokenization styles (including subword tokenization) by
    iteratively trying longer token sequences to locate the word.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer used for encoding.
    encoded_context : Any
        Encoded context with word_ids() method.
    word : str
        Word to locate in context (case-insensitive).

    Returns
    -------
    list[int]
        Token indices corresponding to the word.

    Raises
    ------
    ValueError
        If word is not found in encoded context.
    """
    word_pattern = re.compile(r"^\W?" + word + r"\W?$", flags=re.IGNORECASE)
    unique_word_ids = list({wid for wid in encoded_context.word_ids() if wid is not None})

    # Try increasingly longer token sequences for subword handling
    for seq_len in range(1, 6):
        for word_id in unique_word_ids:
            word_id_sequence = list(range(word_id, word_id + seq_len))
            idx = np.where(np.isin(encoded_context.word_ids(), word_id_sequence))[0].tolist()
            decoded = tokenizer.decode(encoded_context["input_ids"][0][idx])
            alt_decoded = "".join(decoded.split(" "))

            if re.search(word_pattern, decoded) or re.search(word_pattern, alt_decoded):
                return idx

    context_text = tokenizer.decode(encoded_context["input_ids"][0][1:])
    raise ValueError(f'"{word}" not found in "{context_text}"')


def get_word_embedding_by_layer(
    tokenizer: PreTrainedTokenizer,
    embedding_model: PreTrainedModel,
    context: str,
    primer: str,
    word: str,
    layers: list[int],
) -> torch.Tensor:
    """Extract word embeddings across specified layers.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer for encoding context.
    embedding_model : PreTrainedModel
        Model to extract embeddings from.
    context : str
        Context sentence containing the word.
    primer : str
        Optional system prompt to prepend to context.
    word : str
        Word to extract embeddings for.
    layers : list[int]
        Indices of layers to extract embeddings from.

    Returns
    -------
    torch.Tensor
        Stacked embeddings (num_layers, embedding_dim), with subword embeddings averaged.
    """
    if primer:
        message = f"[SYSTEM]\n{primer}\n\n[CONTEXT]\n{context}"
        encoded_context = tokenizer.encode_plus(message, return_tensors="pt", truncation=True).to(
            embedding_model.device
        )
    else:
        encoded_context = tokenizer.encode_plus(context, return_tensors="pt", truncation=True).to(
            embedding_model.device
        )

    word_idx = get_word_idx(tokenizer, encoded_context, word)
    embedding_model.eval()

    with torch.no_grad():
        hidden_states = embedding_model(**encoded_context).hidden_states

    embeddings_by_layer = [hidden_states[layer][0][word_idx].mean(dim=0).to("cpu") for layer in layers]

    return torch.stack(embeddings_by_layer)


def fill_template(gendered_term: str, template: str, isNNP: bool = False) -> str:  # noqa: FBT001, FBT002, N803
    """Fill template with gendered term/name using POS tagging.

    Adjusts template placement based on part-of-speech and whether the term
    is a proper noun (detected or specified).

    Parameters
    ----------
    gendered_term : str
        Term or name to insert into template.
    template : str
        Template string with placeholder position indicated by structure.
    isNNP : bool, optional
        Whether term is a proper noun, by default False.

    Returns
    -------
    str
        Filled template with term appropriately positioned.
    """
    pos_tag = nltk.pos_tag([gendered_term])[0][1]
    is_noun_or_adj = pos_tag in ["NN", "JJ"]

    if template.startswith(" is"):
        prefix = "the " if is_noun_or_adj and not isNNP else ""
        return prefix + gendered_term + template
    suffix = "the " if is_noun_or_adj and not isNNP else ""
    return template + suffix + gendered_term


def get_stats(values: list[float] | np.ndarray) -> tuple[float, float]:
    """Calculate mean and standard deviation.

    Parameters
    ----------
    values : list[float] | np.ndarray
        Values to compute statistics for.

    Returns
    -------
    tuple[float, float]
        Tuple of (mean, standard deviation).
    """
    arr = np.asarray(values)
    return float(np.mean(arr)), float(np.std(arr))


def standardize(values: list[float] | np.ndarray) -> np.ndarray:
    """Standardize values to zero mean and unit variance.

    Parameters
    ----------
    values : list[float] | np.ndarray
        Values to standardize.

    Returns
    -------
    np.ndarray
        Standardized values (z-scores).
    """
    arr = np.asarray(values)
    mean, std = get_stats(arr)
    return (arr - mean) / std


def get_diff(
    arr1: list[float] | np.ndarray, arr2: list[float] | np.ndarray
) -> tuple[float, float, float, float, float, float, float]:
    """Compute difference statistics between two groups.

    Performs independent samples t-test and computes means, standard
    deviations, and absolute difference.

    Parameters
    ----------
    arr1 : list[float] | np.ndarray
        First group of values.
    arr2 : list[float] | np.ndarray
        Second group of values.

    Returns
    -------
    tuple[float, float, float, float, float, float, float]
        Tuple of (mean1, std1, mean2, std2, diff, p_value, abs_diff).
    """
    try:
        arr1 = np.array(arr1, dtype=float)
        arr2 = np.array(arr2, dtype=float)
    except ValueError as e:
        print(f"Error converting data to float. \nArr1: {arr1}\nArr2: {arr2}")
        raise e
    mean1, std1 = get_stats(arr1)
    mean2, std2 = get_stats(arr2)
    diff = mean1 - mean2
    _, p_val = ttest_ind(arr1, arr2)
    return mean1, std1, mean2, std2, diff, p_val, abs(diff)
