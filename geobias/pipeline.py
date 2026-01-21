"""Pipeline for computing and projecting word embeddings across stereotype dimensions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nltk
import numpy as np
import pandas as pd
import torch
from scipy import linalg
from tqdm import tqdm

from geobias.utils import (
    TEMPLATES,
    fill_template,
    get_diff,
    get_logger,
    get_number_of_hidden_states,
    get_word_embedding_by_layer,
    load_model_for_embedding_retrieval,
    setup_logging,
    standardize,
)

WARMTH_COMPETENCE_DIMENSIONS: dict[str, list[str]] = {
    "Warmth": ["Sociability", "Morality"],
    "Competence": ["Ability", "Agency"],
}

setup_logging()
logger = get_logger(__file__)


class GeobiasPipeline:
    """Pipeline for computing and projecting word embeddings across stereotype dimensions.

    Computes embeddings for terms across different layers, standardizes them using
    learned basis changes, and projects population terms into stereotype spaces.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        populations_path: str,
        examples_path: str,
        stereotypes_path: str,
        stereotype_dimensions: list[str],
        max_examples: int,
        standardization: bool,  # noqa: FBT001
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        hf_token: str = "",
        embeddings_dir: str = "embeddings",
        projections_dir: str = "projections",
        primer: str = "",
        primer_name: str = "default",
    ) -> None:
        """Initialize the GeobiasPipeline.

        Parameters
        ----------
        model_name : str
            Name or path of the transformer model.
        populations_path : str
            Path to populations JSON file relative to data/populations/.
        examples_path : str
            Path to examples JSONL file relative to data/examples/.
        stereotypes_path : str
            Path to stereotype dimensions CSV file relative to data/dictionaries/.
        stereotype_dimensions : list[str]
            List of stereotype dimension names.
        max_examples : int
            Maximum number of context examples per term.
        standardization : bool
            Whether to standardize projection results.
        device : str, optional
            Device for model inference, by default "cuda" if available else "cpu".
        hf_token : str, optional
            Hugging Face API token for private models, by default "".
        embeddings_dir : str, optional
            Output directory for embeddings relative to output/, by default "embeddings".
        projections_dir : str, optional
            Output directory for projections relative to output/, by default "projections".
        primer : str, optional
            Optional system prompt for context, by default "".
        primer_name : str, optional
            Name identifier for primer variant, by default "default".
        """
        self._model_name = model_name
        self._populations_path = populations_path
        self._examples_path = examples_path
        self._stereotypes_path = stereotypes_path
        self._stereotype_dimensions = stereotype_dimensions
        self._max_examples = max_examples
        self._standardization = standardization
        self._device = device
        self._hf_token = hf_token
        self._embeddings_dir = Path(f"output/{embeddings_dir}")
        self._projections_dir = Path(f"output/{projections_dir}")
        self._primer = primer
        self._primer_name = primer_name

        self._stereodim_dictionary = pd.read_csv(f"data/dictionaries/{self._stereotypes_path}", index_col=0)

        self._context_examples: dict[str, list[str]] = {}
        with open(f"data/examples/{self._examples_path}") as f:
            for entry in f.readlines():
                data = json.loads(entry)
                key = f'{data["term"]} - {data["synset"]}'
                self._context_examples[key] = data["examples"][:max_examples]

        with open(f"data/populations/{self._populations_path}") as f:
            self._populations = json.load(f)
        self._group1 = next(iter(self._populations.keys()))
        self._group2 = list(self._populations.keys())[1]

        self._tokenizer, self._embedding_model = load_model_for_embedding_retrieval(
            self._model_name, self._device, hf_token=self._hf_token
        )
        self._layers = list(range(get_number_of_hidden_states(self._tokenizer, self._embedding_model)))
        self._model_name = f"{self._model_name.split('/')[-1]}-{self._primer_name}"

        if not self._projections_dir.is_dir():
            self._projections_dir.mkdir(parents=True)

        if not self._embeddings_dir.is_dir():
            self._embeddings_dir.mkdir(parents=True)
        for layer in self._layers:
            if not (self._embeddings_dir / f"{self._model_name}-L{layer}").is_dir():
                (self._embeddings_dir / f"{self._model_name}-L{layer}").mkdir()
        self._embedding_dict: dict[int, dict[str, Any]] = {
            layer: {"sense_embeddings": [], "sense_embedding_labels": [], "pole_embedding_dict": {}}
            for layer in self._layers
        }

        all_dimensions = list(WARMTH_COMPETENCE_DIMENSIONS.keys()) + self._stereotype_dimensions
        self._result_dict: dict[str, dict[str, Any]] = {}
        for layer in self._layers:
            model_key = f"{self._model_name}-L{layer}"
            self._result_dict[model_key] = {
                "Model": len(all_dimensions) * [model_key],
                "Dimension": all_dimensions,
            }

    def compute_embeddings(self) -> None:
        """Compute and store embeddings for all stereotype dimension terms.

        Retrieves embeddings for each term-synset pair across all layers and
        contexts, averaging across contexts and organizing by dimension/direction.
        """
        for _, row in tqdm(
            self._stereodim_dictionary.iterrows(), desc="Retrieving embeddings for stereotype dimensions."
        ):
            term = row["term"]
            synset = row["synset"]
            dimension = row["dimension"]
            direction = row["dir"]

            contexts = self._context_examples[f"{term} - {synset}"]

            if len(contexts) == 0:
                logger.info(f"No examples for term '{term}'.")
                continue

            layerwise_sense_embeddings = [
                get_word_embedding_by_layer(
                    self._tokenizer, self._embedding_model, context, self._primer, term, self._layers
                )
                for context in contexts
            ]
            layerwise_sense_embeddings = torch.stack(layerwise_sense_embeddings).mean(dim=0)

            pole_key = f"{dimension}-{direction}"
            label = f"{term} - {synset} - {pole_key}"

            for layer, layer_dict in self._embedding_dict.items():
                sense_embedding = layerwise_sense_embeddings[layer]

                if pole_key not in layer_dict["pole_embedding_dict"]:
                    layer_dict["pole_embedding_dict"][pole_key] = []
                layer_dict["pole_embedding_dict"][pole_key].append(sense_embedding)
                layer_dict["sense_embeddings"].append(sense_embedding)
                layer_dict["sense_embedding_labels"].append(label)

    def save_embeddings(self) -> None:
        """Save computed embeddings to disk as numpy files and label text files.

        Saves pole embeddings, warmth/competence aggregates, and sense embeddings
        for each layer.
        """
        for layer, layer_dict in self._embedding_dict.items():
            layer_dir = self._embeddings_dir / f"{self._model_name}-L{layer}"

            for pole, embeddings in layer_dict["pole_embedding_dict"].items():
                embeddings_array = np.vstack(embeddings)
                layer_dir.mkdir(parents=True, exist_ok=True)
                with open(layer_dir / f"{pole}_embeddings.npy", "wb") as f:
                    np.save(f, embeddings_array)

            for dim, subdimensions in WARMTH_COMPETENCE_DIMENSIONS.items():
                for direction in ["low", "high"]:
                    embeddings_array = np.concatenate(
                        [
                            layer_dict["pole_embedding_dict"][f"{subdimensions[0]}-{direction}"],
                            layer_dict["pole_embedding_dict"][f"{subdimensions[1]}-{direction}"],
                        ],
                        axis=0,
                    )
                    with open(layer_dir / f"{dim}-{direction}_embeddings.npy", "wb") as f:
                        np.save(f, embeddings_array)

            sense_embeddings = np.vstack(layer_dict["sense_embeddings"])
            with open(layer_dir / "sense_embeddings.npy", "wb") as f:
                np.save(f, sense_embeddings)

            with open(layer_dir / "sense_embedding_labels.txt", "w") as f:
                f.writelines(f"{label}\n" for label in layer_dict["sense_embedding_labels"])

    def compute_base_change(self) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Compute base change matrices for projecting embeddings.

        Computes direction vectors for each stereotype dimension and warmth/competence
        aggregate dimensions, then calculates pseudo-inverse matrices for projection.

        Returns
        -------
        tuple[dict[int, np.ndarray], dict[int, np.ndarray]]
            Tuple of (warmth_competence_base_change_inv, stereodim_base_change_inv)
            dicts mapping layer indices to pseudo-inverse matrices.
        """
        warmth_competence_base_change_inv: dict[int, np.ndarray] = {}
        stereodim_base_change_inv: dict[int, np.ndarray] = {}

        for layer in tqdm(self._layers, desc="Preparing layerwise base change matrices."):
            stereodim_base_change: list[np.ndarray] = []

            for dim in self._stereotype_dimensions:
                low_embeddings = np.load(self._embeddings_dir / f"{self._model_name}-L{layer}/{dim}-low_embeddings.npy")
                high_embeddings = np.load(
                    self._embeddings_dir / f"{self._model_name}-L{layer}/{dim}-high_embeddings.npy"
                )

                low_mean = np.average(low_embeddings, axis=0)
                high_mean = np.average(high_embeddings, axis=0)
                stereodim_base_change.append(high_mean - low_mean)

            with open(self._embeddings_dir / f"{self._model_name}-L{layer}/stereodim_base_change.npy", "wb") as f:
                np.save(f, stereodim_base_change)

            stereodim_base_change_inv[layer] = linalg.pinv(np.transpose(np.vstack(stereodim_base_change)))

            warmth_competence_base_change: list[np.ndarray] = []

            for dim in WARMTH_COMPETENCE_DIMENSIONS:
                low_embeddings = np.load(self._embeddings_dir / f"{self._model_name}-L{layer}/{dim}-low_embeddings.npy")
                high_embeddings = np.load(
                    self._embeddings_dir / f"{self._model_name}-L{layer}/{dim}-high_embeddings.npy"
                )

                low_mean = np.average(low_embeddings, axis=0)
                high_mean = np.average(high_embeddings, axis=0)
                warmth_competence_base_change.append(high_mean - low_mean)

            with open(
                self._embeddings_dir / f"{self._model_name}-L{layer}/warmth_competence_base_change.npy", "wb"
            ) as f:
                np.save(f, warmth_competence_base_change)

            warmth_competence_base_change_inv[layer] = linalg.pinv(
                np.transpose(np.vstack(warmth_competence_base_change))
            )

        return warmth_competence_base_change_inv, stereodim_base_change_inv

    def project(
        self, warmth_competence_base_change_inv: dict[int, np.ndarray], stereodim_base_change_inv: dict[int, np.ndarray]
    ) -> None:
        """Project population terms into stereotype dimension spaces.

        Parameters
        ----------
        warmth_competence_base_change_inv : dict[int, np.ndarray]
            Pseudo-inverse matrices for warmth/competence projections by layer.
        stereodim_base_change_inv : dict[int, np.ndarray]
            Pseudo-inverse matrices for stereotype dimension projections by layer.
        """
        for group, terms in self._populations.items():
            for term in tqdm(terms, desc=f"Projecting {group} to stereotype dimensions"):
                is_proper_noun = nltk.pos_tag([term])[0][1] == "NNP" or "names" in group.lower()
                contexts = [fill_template(term, template, isNNP=is_proper_noun) for template in TEMPLATES]

                layerwise_sense_embeddings = [
                    get_word_embedding_by_layer(
                        self._tokenizer, self._embedding_model, context, self._primer, term, self._layers
                    )
                    for context in contexts
                ]
                layerwise_sense_embeddings = torch.stack(layerwise_sense_embeddings).mean(dim=0)

                for layer in self._layers:
                    sense_embedding = layerwise_sense_embeddings[layer].double()

                    warmth_comp = torch.matmul(
                        torch.from_numpy(warmth_competence_base_change_inv[layer]).double(), sense_embedding
                    ).numpy()
                    stereodim = torch.matmul(
                        torch.from_numpy(stereodim_base_change_inv[layer]).double(), sense_embedding
                    ).numpy()
                    self._result_dict[f"{self._model_name}-L{layer}"][term] = np.concatenate((warmth_comp, stereodim))

    def gather_results(self) -> None:
        """Aggregate layer-wise projections, compute statistics, and save results.

        Averages projections across layers per dimension, optionally standardizes,
        computes group differences with p-values, and saves to CSV.
        """
        results = pd.concat([pd.DataFrame.from_dict(self._result_dict[model_name]) for model_name in self._result_dict])

        for dimension in [*self._stereotype_dimensions, "Warmth", "Competence"]:
            layer_averages = (
                results.loc[results["Dimension"] == dimension].set_index(["Model", "Dimension"]).mean(axis=0).tolist()
            )
            new_row = pd.DataFrame([[self._model_name, dimension, *layer_averages]], columns=results.columns)
            results = pd.concat([results, new_row], ignore_index=True)

        if len(self._populations) > 2:  # noqa: PLR2004
            print(
                f"More than two populations in {self._populations}, "
                f"comparing first two: {self._group1} vs {self._group2}."
            )

        group_cols = self._populations[self._group1] + self._populations[self._group2]

        if self._standardization:
            results[group_cols] = results[group_cols].apply(
                lambda x: standardize(x), axis="columns", result_type="expand"
            )

        stats = pd.DataFrame(
            results.apply(
                lambda x: get_diff(x[self._populations[self._group1]], x[self._populations[self._group2]]),
                axis="columns",
                result_type="expand",
            )
        )
        stats.columns = [
            f"{self._group1}_mean",
            f"{self._group1}_std",
            f"{self._group2}_mean",
            f"{self._group2}_std",
            "diff",
            "diff_pvalue",
            "diff_abs",
        ]
        results = pd.concat([results, stats], axis=1)
        results.to_csv(self._projections_dir / f"{self._model_name}_projections.csv")

    def __call__(self) -> None:
        """Execute the complete pipeline: compute, save, project, and gather results."""
        self.compute_embeddings()
        self.save_embeddings()
        warmth_comp_inv, stereodim_inv = self.compute_base_change()
        self.project(warmth_comp_inv, stereodim_inv)
        self.gather_results()
