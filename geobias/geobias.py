"""Main entry point for Geobias."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra

if TYPE_CHECKING:
    from omegaconf import DictConfig

from geobias.pipeline import GeobiasPipeline


@hydra.main(config_path="configs", config_name="pipeline.yaml", version_base=None)  # type: ignore
def main(cfg: DictConfig) -> None:
    """Main entry point for Geobias.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object.
    """
    pipeline = GeobiasPipeline(
        model_name=cfg.model.name,
        populations_path=cfg.data.populations,
        examples_path=cfg.data.examples,
        stereotypes_path=cfg.data.stereotypes,
        stereotype_dimensions=cfg.dimensions.stereotypes,
        max_examples=cfg.processing.max_examples,
        standardization=cfg.processing.standardization,
        embeddings_dir=cfg.embeddings_output_dir,
        projections_dir=cfg.projections_output_dir,
        primer_text=cfg.primer.text,
        primer_name=cfg.primer.name,
        hf_token=cfg.model.hf_token if "hf_token" in cfg.model else "",
    )

    pipeline()


if __name__ == "__main__":
    main()
