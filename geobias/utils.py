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
    tokenizer_kwargs: dict[str, Any] = {
        "use_fast": True,
    }
    if hf_token:
        tokenizer_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    # Set padding token if not already set (needed for batched processing)
    # Use eos_token or unk_token as pad_token to avoid needing to resize model embeddings
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # Last resort: use the last token in vocab (should exist)
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(len(tokenizer) - 1)

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "output_hidden_states": True,
    }
    if hf_token:
        model_kwargs["token"] = hf_token

    try:
        embedding_model = AutoModel.from_pretrained(model_name, **model_kwargs)
    except ValueError:
        # Fallback for models where device_map is not supported (typically smaller, older models)
        embedding_model = AutoModel.from_pretrained(model_name, **model_kwargs).to(device)

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
    encoded = tokenizer._encode_plus("test", return_tensors="pt").to(embedding_model.device)
    with torch.inference_mode():
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
        message = f"""
            <system_instructions>
            {primer}
            </system_instructions>

            <context>
            {context}
            </context>
        """
        encoded_context = tokenizer.encode_plus(message, return_tensors="pt", truncation=True).to(
            embedding_model.device
        )
    else:
        encoded_context = tokenizer.encode_plus(context, return_tensors="pt", truncation=True).to(
            embedding_model.device
        )

    word_idx = get_word_idx(tokenizer, encoded_context, word)
    embedding_model.eval()

    with torch.inference_mode():
        hidden_states = embedding_model(**encoded_context).hidden_states

    embeddings_by_layer = [hidden_states[layer][0][word_idx].mean(dim=0).to("cpu") for layer in layers]

    return torch.stack(embeddings_by_layer)


def get_word_embeddings_by_layer_batched(
    tokenizer: PreTrainedTokenizer,
    embedding_model: PreTrainedModel,
    contexts: list[str],
    primer: str,
    word: str,
    layers: list[int],
) -> torch.Tensor:
    """Extract word embeddings across specified layers for multiple contexts in a single batch.

    This is a batched version of get_word_embedding_by_layer that processes multiple
    contexts for the same word in a single forward pass, which is much faster on GPU.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer for encoding context.
    embedding_model : PreTrainedModel
        Model to extract embeddings from.
    contexts : list[str]
        List of context sentences, all containing the same word.
    primer : str
        Optional system prompt to prepend to each context.
    word : str
        Word to extract embeddings for (must appear in all contexts).
    layers : list[int]
        Indices of layers to extract embeddings from.

    Returns
    -------
    torch.Tensor
        Stacked embeddings (num_layers, embedding_dim), averaged across contexts.
        Same shape and values as calling get_word_embedding_by_layer on each context
        and averaging the results.

    Raises
    ------
    ValueError
        If word is not found in any context or contexts list is empty.
    """
    if len(contexts) == 0:
        raise ValueError("Contexts list cannot be empty")

    # Prepare messages with primer if provided
    if primer:
        messages = [
            f"""
            <system_instructions>
            {primer}
            </system_instructions>

            <context>
            {context}
            </context>
            """
            for context in contexts
        ]
    else:
        messages = contexts

    # Ensure padding token is set (safety check)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # Last resort: use the last token in vocab
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(len(tokenizer) - 1)

    # Encode all contexts in a batch with padding
    encoded_batch = tokenizer(
        messages,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    ).to(embedding_model.device)

    # Find word indices for each context in the batch
    # We process each context separately to find word positions, using the same
    # encoding settings as the batch to ensure positions match
    word_indices_per_context: list[list[int]] = []
    for context in contexts:
        # Create a single-item encoding with the same settings as batch encoding
        if primer:
            single_message = f"""
            <system_instructions>
            {primer}
            </system_instructions>

            <context>
            {context}
            </context>
            """
        else:
            single_message = context

        # Use encode_plus with same settings as batch to ensure tokenization matches
        single_encoded = tokenizer.encode_plus(single_message, return_tensors="pt", truncation=True, padding=False)
        word_idx = get_word_idx(tokenizer, single_encoded, word)
        word_indices_per_context.append(word_idx)

    embedding_model.eval()

    # Forward pass on the entire batch
    with torch.inference_mode():
        hidden_states = embedding_model(**encoded_batch).hidden_states

    # Extract embeddings for each context and average
    batch_size = len(contexts)
    embeddings_by_layer_per_context = []

    for layer in layers:
        layer_embeddings = []
        for batch_idx in range(batch_size):
            # Get word indices for this context
            word_idx = word_indices_per_context[batch_idx]
            # Extract embeddings for this word in this context
            # hidden_states[layer] shape: (batch_size, seq_len, hidden_dim)
            word_embeddings = hidden_states[layer][batch_idx][word_idx]
            # Average subword tokens if word was split
            word_embedding = word_embeddings.mean(dim=0)
            layer_embeddings.append(word_embedding)

        # Stack and average across contexts (keep on GPU for now)
        layer_embeddings_tensor = torch.stack(layer_embeddings)
        avg_embedding = layer_embeddings_tensor.mean(dim=0)
        embeddings_by_layer_per_context.append(avg_embedding)

    # Stack all layers, then move to CPU once
    return torch.stack(embeddings_by_layer_per_context).to("cpu")


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
