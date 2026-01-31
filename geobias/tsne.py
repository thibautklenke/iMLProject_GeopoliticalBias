"""TSNE Pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
import torch
from scipy import linalg
from sklearn.manifold import TSNE
from tqdm import tqdm

if TYPE_CHECKING:
    from omegaconf import DictConfig

from geobias.utils import (
    TEMPLATES,
    fill_template,
    get_number_of_hidden_states,
    get_word_embedding_by_layer,
    load_model_for_embedding_retrieval,
)


class TSNEPipeline:
    """Class."""

    def __init__(
        self,
        populations_path: str,
        model_name: str,
        examples_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        hf_token: str = "",
        embeddings_dir: str = "embeddings",
        dim: int = 2,
        primer_text: str = "",
        primer_name: str = "default",
    ):
        """Func."""
        self._dim = dim
        self._populations_path = populations_path
        self._model_name = model_name
        self._device = device
        self._hf_token = hf_token
        self._embeddings_dir = Path(f"output/{embeddings_dir}")
        self._primer_text = primer_text
        self._primer_name = primer_name
        self._examples_path = examples_path

        # Load names which we infer
        with open(f"data/populations/{self._populations_path}") as f:
            self._populations = json.load(f)

        # Load tokenizer and model
        self._tokenizer, self._embedding_model = load_model_for_embedding_retrieval(
            self._model_name, self._device, hf_token=self._hf_token
        )
        self._layers = list(range(get_number_of_hidden_states(self._tokenizer, self._embedding_model)))
        self._model_name = (
            f"{self._model_name.split('/')[-1]}"
            f"-{self._primer_name}"
            f"-{self._examples_path[0]}"
            f"-{self._populations_path[0]}"
        )

        # Load base change matrix and compute inverse
        self._base_change = np.load(self._embeddings_dir / f"{self._model_name}-L{len(self._layers) - 1}/stereodim_base_change.npy")
        self._inv_base_change = linalg.pinv(np.transpose(self._base_change))

        # Embedding result
        self._result_embedding = {}
        self._result = {}

    def compute_embeddings(self):
        """Doc."""
        for group, terms in self._populations.items():
            group_embeddings = None
            for term in tqdm(terms, desc=f"Projection {group} to stereotype dimensions", position=0, leave=True):
                is_proper_noun = nltk.pos_tag([term])[0][1] == "NNP" or "names" in group.lower()
                contexts = [fill_template(term, template, isNNP=is_proper_noun) for template in TEMPLATES]

                layerwise_sense_embeddings = [
                    get_word_embedding_by_layer(
                        self._tokenizer, self._embedding_model, context, self._primer_text, term, self._layers
                    )
                    for context in contexts
                ]
                layerwise_sense_embeddings = torch.stack(layerwise_sense_embeddings).mean(dim=0)

                # Only care about last layer
                sense_embeddings = layerwise_sense_embeddings[-1].double()
                if group_embeddings is None:
                    group_embeddings = sense_embeddings.unsqueeze(dim=0)
                else:
                    group_embeddings = np.concat([group_embeddings, sense_embeddings.unsqueeze(dim=0)], axis=0)

            self._result_embedding[group] = group_embeddings

    def stereodim_metric(self, a, b):
        """."""
        a_stereo = self._inv_base_change @ a
        b_stereo = self._inv_base_change @ b

        return linalg.norm(a_stereo - b_stereo)


    def compute_tsne(self, metric):
        """Doc."""
        # TSNE
        if metric == "euclidean":
            tsne = TSNE(n_components = self._dim, metric="euclidean", random_state=0)
        elif metric == "stereodim":
            tsne = TSNE(n_components = self._dim, metric=self.stereodim_metric, random_state=0)
        else:
            raise ValueError("Wrong metric")

        group_embeddings = list(self._result_embedding.values())
        total_embeddings = np.concat(group_embeddings, axis=0)
        result = tsne.fit_transform(total_embeddings)

        idx = 0
        result_embeddings = []
        for group_emb in group_embeddings:
            size = group_emb.shape[0]
            result_embeddings.append(result[range(idx, idx + size), :])
            idx += size

        self._result[metric] = result_embeddings

    def plot_tsne(self):
        """."""
        plots = len(self._result.keys())
        fig, ax = plt.subplots(1, plots)

        for idx, (metric, result_emb) in enumerate(self._result.items()):
            ax[idx].set_title(metric)
            for group_emb in result_emb:
                sns.scatterplot(x=group_emb[:, 0], y=group_emb[:, 1], ax=ax[idx])

        fig.savefig(f"figures/tsne/{self._model_name}.pdf")

    def __call__(self) -> None:
        """Doc."""
        self.compute_embeddings()
        self.compute_tsne("stereodim")
        self.compute_tsne("euclidean")
        self.plot_tsne()


@hydra.main(config_path="configs", config_name="pipeline.yaml", version_base=None)  # type: ignore
def main(cfg: DictConfig) -> None:
    """Main entry point for TSNEPipeline.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object.
    """
    pipeline = TSNEPipeline(
        model_name=cfg.model.name,
        populations_path=cfg.data.populations,
        examples_path=cfg.data.examples,
        embeddings_dir=cfg.embeddings_output_dir,
        primer_text=cfg.primer.text,
        primer_name=cfg.primer.name,
        hf_token=cfg.model.hf_token if "hf_token" in cfg.model else "",
    )

    pipeline()


if __name__ == "__main__":
    main()
