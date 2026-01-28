"""Plotting functions for bias profiles."""

from typing import Any
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import matplotlib.axes
import pandas as pd
import numpy as np
import json
import hydra
from pathlib import Path


def _get_style_config() -> dict[str, dict[int, str]]:
    """Returns default style configuration for plotting.

    Returns
    -------
    dict[str, dict[str, str]]
        Dictionary containing line and color styles.
    """
    return {
        "line": {0: "solid", 1: "dashdot"},
        "color": {0: "green", 1: "lightgreen"},
    }


def _load_populations(path: str) -> dict[str, list[str]]:
    """Loads population groups from JSON file.

    Parameters
    ----------
    path : str
        Path to the populations JSON file.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping group names to population lists.
    """
    with open(path) as f:
        return json.load(f)


def _setup_output_directory(output_dir: str) -> None:
    """Creates output directory if it doesn't exist.

    Parameters
    ----------
    output_dir : str
        Path to the output directory.
    """
    path = Path(output_dir)
    if not path.is_dir():
        path.mkdir(parents=True)


def _extract_plot_data(
    results: pd.DataFrame,
    dimensions: list[str],
    model_name: str,
    groups: tuple[str, str],
    polar_labels: dict[str, tuple[str, str]],
) -> tuple[dict[str, list[float]], dict[str, list[str]], list[str]]:
    """Extracts plotting data from results dataframe.

    Parameters
    ----------
    results : pd.DataFrame
        Dataframe containing projection results.
    dimensions : list[str]
        List of stereotype dimensions to plot.
    model_name : str
        Name of the model.
    groups : tuple[str, str]
        Tuple of (group1, group2) names.
    polar_labels : dict[str, tuple[str, str]]
        Dictionary mapping dimensions to (low, high) labels.

    Returns
    -------
    tuple[dict[str, list[float]], dict[str, list[str]], list[str]]
        Tuple of (plot_values, plot_labels, bold_labels).
    """
    group1, group2 = groups
    plot_values: dict[str, list[float]] = {group1: [], group2: []}
    plot_labels: dict[str, list[str]] = {"low labels": [], "high labels": []}
    bold_labels: list[str] = []

    for dimension in dimensions:
        row = results.loc[
            (results["Dimension"] == dimension) & (results["Model"] == model_name)
        ]
        for group in [group1, group2]:
            plot_values[group].append(row[f"{group}_mean"].values[0])

        low_label = polar_labels[dimension][0]
        high_label = polar_labels[dimension][1]

        if row["diff_pvalue"].values[0] < 0.05:
            high_label += "$^*$"
            bold_labels.extend([low_label, high_label])

        plot_labels["low labels"].append(low_label)
        plot_labels["high labels"].append(high_label)

    return plot_values, plot_labels, bold_labels


def _plot_values(
    ax1: matplotlib.axes.Axes,
    plot_values: dict[str, list[float]],
) -> None:
    """Plots projection values for both groups.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        Matplotlib axes object to plot on.
    plot_values : dict[str, list[float]]
        Dictionary mapping groups to their projection values.
    groups : tuple[str, str]
        Tuple of (group1, group2) names.
    """
    styles = _get_style_config()

    for idx, (group, values) in enumerate(plot_values.items()):
        color = styles["color"].get(idx, "gray")
        linestyle = styles["line"].get(idx, "solid")
        ax1.plot(
            values,
            np.arange(len(values)),
            label=str(group),
            color=color,
            marker="o",
            linestyle=linestyle,
        )


def _setup_left_axis(
    ax1: matplotlib.axes.Axes,
    labels: list[str],
    bold_labels: list[str],
) -> None:
    """Configures left y-axis with dimension labels.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        Matplotlib axes object to configure.
    labels : list[str]
        List of low-end dimension labels.
    bold_labels : list[str]
        List of labels to display in bold.
    """
    ylim = ax1.get_ylim()
    ax1.set_ylim(ylim[0] - 0.3, ylim[1] + 0.3)
    ax1.set_yticks(
        np.arange(len(labels)),
        labels=labels,
        verticalalignment="center",
        fontsize=9,
    )
    for label in ax1.get_yticklabels():
        if label.get_text() in bold_labels:
            label.set_fontweight("bold")


def _setup_right_axis(
    ax1: matplotlib.axes.Axes,
    ax2: matplotlib.axes.Axes,
    labels: list[str],
    bold_labels: list[str],
) -> None:
    """Configures right y-axis with dimension labels.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        Left axes object (for synchronizing limits).
    ax2 : matplotlib.axes.Axes
        Right axes object to configure.
    labels : list[str]
        List of high-end dimension labels.
    bold_labels : list[str]
        List of labels to display in bold.
    """
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(
        np.arange(len(labels)),
        labels=labels,
        verticalalignment="center",
        fontsize=9,
    )
    for label in ax2.get_yticklabels():
        if label.get_text() in bold_labels:
            label.set_fontweight("bold")


def _setup_x_axis(ax1: matplotlib.axes.Axes) -> None:
    """Configures x-axis with symmetric limits and labels.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        Matplotlib axes object to configure.
    """
    max_x = max(abs(ax1.get_xlim()[0]), abs(ax1.get_xlim()[1]), 1)
    ax1.set_xlim(-max_x, max_x)
    ax1.tick_params(axis="x", which="major", labelsize=8)
    ax1.set_xlabel("projected values", fontsize=8)


def _save_figure(
    fig: matplotlib.figure.Figure,
    ax1: matplotlib.axes.Axes,
    model_name: str,
    n_dimensions: int,
    output_path: str,
) -> None:
    """Configures and saves the figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.
    ax1 : matplotlib.axes.Axes
        Left axes object.
    model_name : str
        Name of the model for the title.
    n_dimensions : int
        Number of dimensions (for sizing).
    output_path : str
        Path where the figure will be saved.
    """
    ax1.set_title(model_name, fontsize=9)
    bbox_val = -0.12 if n_dimensions > 2 else -0.25
    ax1.legend(bbox_to_anchor=(1, bbox_val), ncol=2, fontsize=9)
    fig.set_figwidth(3.5)
    fig.set_figheight(n_dimensions * 0.5 + 1)
    plt.savefig(output_path, bbox_inches="tight")


@hydra.main(config_path="configs", config_name="base.yaml", version_base=None)  # type: ignore
def main(cfg: DictConfig) -> None:
    """Creates and saves bias profile plots from projection results.

    Reads stereotype dimensions and projection results, then generates
    a comparative visualization showing bias profiles for two groups
    with statistical significance indicators.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing:
        - dimensions.stereotypes: List of dimension names
        - dimensions.polar_labels: Dict mapping dimensions to label pairs
        - model.name: Model identifier (path format)
        - primer.name: Primer configuration name
        - data.populations: Path to populations JSON file
        - projections_output_dir: Output directory for results

    Returns
    -------
    None
    """
    
    # Load configuration and data
    dimensions: list[str] = cfg.dimensions.stereotypes
    polar_labels: dict[str, tuple[str, str]] = cfg.dimensions.polar_labels
    model_name: str = f"{cfg.model.name.split('/')[-1]}-{cfg.primer.name}"

    populations: dict[str, list[str]] = _load_populations(
        f"data/populations/{cfg.data.populations}"
    )
    groups: tuple[str, str] = (
        list(populations.keys())[0],
        list(populations.keys())[1],
    )

    results: pd.DataFrame = pd.read_csv(
        f"output/{cfg.projections_output_dir}/{model_name}_projections.csv"
    )

    # Setup output directory and figure
    _setup_output_directory("output/figures")
    fig, ax1 = plt.subplots(1, 1)

    # Extract and organize plot data
    plot_values, plot_labels, bold_labels = _extract_plot_data(
        results, dimensions, model_name, groups, polar_labels
    )

    # Create plots
    _plot_values(ax1, plot_values)
    _setup_left_axis(ax1, plot_labels["low labels"], bold_labels)
    _setup_x_axis(ax1)

    # Configure right axis and finalize
    ax2 = ax1.twinx()
    _setup_right_axis(ax1, ax2, plot_labels["high labels"], bold_labels)

    output_path: str = f"output/figures/{model_name}_bias_profile.pdf"
    _save_figure(fig, ax1, model_name, len(dimensions), output_path)

    print(f"Bias profile saved in file: {output_path}")

if __name__ == "__main__":
    main()
