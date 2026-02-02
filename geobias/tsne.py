"""TSNE Pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    """t-SNE pipeline for visualizing stereotype embeddings and group comparisons.

    This pipeline loads populations and a transformer model/tokenizer, computes
    and saves group-wise word embeddings, and produces t-SNE visualizations
    using either the Euclidean distance or a stereodimension-aware metric.
    """

    def __init__(  # noqa: PLR0913
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
        plot: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the TSNEPipeline.

        Parameters
        ----------
        populations_path : str
            Path to populations JSON file relative to data/populations/.
        model_name : str
            Name or path of the transformer model.
        examples_path : str
            Path to examples JSONL file relative to data/examples/.
        device : str, optional
            Device for model inference, by default "cuda" if available else "cpu".
        hf_token : str, optional
            Hugging Face API token for private models, by default "".
        embeddings_dir : str, optional
            Output directory for embeddings relative to output/, by default "embeddings".
        dim : int, optional
            Number of dimensions for t-SNE (e.g., 2 for 2D plots), by default 2.
        primer_text : str, optional
            Optional primer/system prompt to prepend to contexts, by default "".
        primer_name : str, optional
            Name identifier for primer variant, by default "default".
        plot : bool, optional
            If True, load precomputed embeddings and produce t-SNE plots; if False,
            compute and save embeddings instead, by default False.
        """
        self._dim = dim
        self._populations_path = populations_path
        self._model_name = model_name
        self._device = device
        self._hf_token = hf_token
        self._embeddings_dir = Path(f"output/{embeddings_dir}")
        self._primer_text = primer_text
        self._primer_name = primer_name
        self._examples_path = examples_path
        self._plot = plot

        self._model_name_no_primer = (
            f"{self._model_name.split('/')[-1]}" f"-{self._examples_path[0]}" f"-{self._populations_path[0]}"
        )
        self._model_name = (
            f"{self._model_name.split('/')[-1]}"
            f"-{self._primer_name}"
            f"-{self._examples_path[0]}"
            f"-{self._populations_path[0]}"
        )

        # Load names which we infer
        with open(f"data/populations/{self._populations_path}") as f:
            self._populations = json.load(f)

        if not self._plot:
            # Load tokenizer and model
            self._tokenizer, self._embedding_model = load_model_for_embedding_retrieval(
                model_name, self._device, hf_token=self._hf_token
            )

            self._layers = list(range(get_number_of_hidden_states(self._tokenizer, self._embedding_model)))
        else:
            # Figure out layer count from folder structure
            self._layers = [
                int(entry.name.removeprefix(f"{self._model_name_no_primer}-L"))
                for entry in self._embeddings_dir.iterdir()
                if entry.name.startswith(f"{self._model_name_no_primer}-L")
            ]

        # Load base change matrix and compute inverse
        self._base_change = np.load(
            self._embeddings_dir / f"{self._model_name_no_primer}-L{len(self._layers) - 1}/stereodim_base_change.npy"
        )
        self._inv_base_change = linalg.pinv(np.transpose(self._base_change))

        # Embedding result
        self._result_embedding_default: dict[str, np.ndarray] = {}
        self._result_embedding_primer: dict[str, np.ndarray] = {}
        self._result_default: list[Any] = []
        self._result_primer: list[Any] = []
        self._result_embedding: dict[str, Any] = {}

    def compute_embeddings(self) -> None:
        """Compute group-wise word embeddings and store results.

        For each population group, iterates through terms, generates context
        sentences from templates, retrieves layer-wise token embeddings from
        the model, averages across templates and contexts, and stores the
        resulting per-term embeddings (last layer) for the group in
        ``self._result_embedding``.
        """
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

    def save_embeddings(self) -> None:
        """Save computed embeddings to disk.

        Ensures the directory ``output/tsne_embeddings`` exists and saves each
        group's embeddings as a NumPy ``.npy`` file named
        ``{model_name}_{group}.npy`` under that directory.
        """
        embeddings_dir = Path("output/tsne_embeddings")
        if not embeddings_dir.is_dir():
            embeddings_dir.mkdir(parents=True)

        for group in self._populations:
            np.save(embeddings_dir / f"{self._model_name}_{group}.npy", self._result_embedding[group])

    def load_embeddings(self) -> None:
        """Load precomputed embeddings from disk into ``self._result_embedding``.

        Loads per-group NumPy arrays from ``output/tsne_embeddings``. Raises an
        Exception with guidance if the directory is missing, instructing to
        generate embeddings with ``plot=False``.
        """
        embeddings_dir = Path("output/tsne_embeddings")
        if not embeddings_dir.is_dir():
            raise Exception("Generate data first with plot=False")

        model_name_default = self._model_name.replace(self._primer_name, "default")

        for group in self._populations:
            self._result_embedding_default[group] = np.load(embeddings_dir / f"{model_name_default}_{group}.npy")
            self._result_embedding_primer[group] = np.load(embeddings_dir / f"{self._model_name}_{group}.npy")

    def stereodim_metric(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute stereodimension-aware distance between two vectors.

        Projects the input vectors into the stereotype-dimension basis using the
        precomputed inverse base change matrix and returns the Euclidean norm
        of their difference.

        Parameters
        ----------
        a : np.ndarray
            First embedding vector.
        b : np.ndarray
            Second embedding vector.

        Returns
        -------
        float
            Euclidean distance between the two projected vectors.
        """
        a_stereo = self._inv_base_change @ a
        b_stereo = self._inv_base_change @ b

        return linalg.norm(a_stereo - b_stereo)

    def compute_tsne(self, embeddings: dict) -> list[Any]:
        """Compute t-SNE projections for the stored embeddings."""
        # TSNE
        tsne = TSNE(n_components=self._dim, metric=self.stereodim_metric, random_state=0)

        group_embeddings = list(embeddings.values())
        total_embeddings = np.concat(group_embeddings, axis=0)
        result = tsne.fit_transform(total_embeddings)

        idx = 0
        result_embeddings = []
        for group_emb in group_embeddings:
            size = group_emb.shape[0]
            result_embeddings.append(result[range(idx, idx + size), :])
            idx += size

        return result_embeddings

    def plot_tsne(self) -> None:
        """Plot stored t-SNE projections and save the figure.

        Creates one subplot per metric stored in ``self._result`` and produces a
        scatter plot for each group's projection. The resulting figure is saved
        to ``figures/tsne/{self._model_name}.pdf``.
        """
        poplist = list(self._populations.keys())

        fig, ax = plt.subplots(1, 2)
        fig.suptitle(f"t-SNE: {'-'.join(self._model_name.split('-')[:-3])}\nStereotype Metric", wrap=True)

        ax[0].set_title("No Primer", y=-0.15)
        for group_idx, group_emb in enumerate(self._result_default):
            sns.scatterplot(x=group_emb[:, 0], y=group_emb[:, 1], ax=ax[0], label=poplist[group_idx], legend=False)

        ax[1].set_title(f"{self._primer_name.capitalize()} Primer", y=-0.15)
        for group_idx, group_emb in enumerate(self._result_primer):
            sns.scatterplot(x=group_emb[:, 0], y=group_emb[:, 1], ax=ax[1], label=poplist[group_idx], legend=False)

        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels)

        figures_dir = Path("figures/tsne")
        if not figures_dir.is_dir():
            figures_dir.mkdir(parents=True)

        fig.savefig(f"figures/tsne/{self._model_name}.pdf", bbox_inches="tight")
        fig.savefig(f"figures/tsne/{self._model_name}.png", bbox_inches="tight", dpi=600)

    def __call__(self) -> None:
        """Run the TSNE pipeline.

        If ``self._plot`` is False, computes and saves embeddings. If True,
        loads precomputed embeddings, computes t-SNE projections using both
        stereodimension and Euclidean metrics, and produces the plots.
        """
        if not self._plot:
            self.compute_embeddings()
            self.save_embeddings()

        if self._plot:
            self.load_embeddings()
            self._result_default = self.compute_tsne(self._result_embedding_default)
            self._result_primer = self.compute_tsne(self._result_embedding_primer)
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
        plot=cfg.plot,
    )

    pipeline()


if __name__ == "__main__":
    main()
