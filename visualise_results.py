import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise batched needle experiment results")
    parser.add_argument(
        "-f", "--file_path",
        type=str,
        help="Path to the JSONL results file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for the plot (default: results_file stem + .png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot",
    )
    return parser.parse_args()


def load_results(results_file):
    results = []
    with open(results_file, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return pd.DataFrame(results)


def plot_heatmap(df, output_path, title=None):
    pivot = df.pivot(
        index="num_token",
        columns="avg_nll_change_perc",
        values="avg_accuracy_modified_cache"
    )
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )

    ax.set_xlabel("Compression %")
    ax.set_ylabel("Sequence Length")

    if title:
        ax.set_title(title)
    else:
        num_epochs = df.iloc[0].get("num_epoch", "?")
        ax.set_title(f"{num_epochs} Test-Time Training Epochs")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

    print("\nSummary Table:")
    print(pivot)


def main():
    args = parse_args()

    df = load_results(args.file_path)

    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.file_path).stem + ".png"

    plot_heatmap(df, output_path, args.title)


if __name__ == "__main__":
    main()
