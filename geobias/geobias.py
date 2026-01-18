"""Main entry point for Geobias."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="base.yaml", version_base=None)  # type: ignore
def main(cfg: DictConfig) -> None:
    """Main entry point for Geobias.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object.
    """
    data = cfg.data
    print(data)


if __name__ == "__main__":
    main()
