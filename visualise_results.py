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


SEQ_LENGTH_BUCKETS = [500, 1000, 2000, 4000, 8000, 10000]
SEQ_LENGTH_LABELS = {500: "500", 1000: "1k", 2000: "2k", 4000: "4k", 8000: "8k", 10000: "10k"}


def snap_to_bucket(n):
    return min(SEQ_LENGTH_BUCKETS, key=lambda b: abs(b - n))


def plot_heatmap(df, output_path, title=None):
    df = df.copy()
    df["seq_length"] = df["num_token"].apply(snap_to_bucket)

    pivot = df.pivot(
        index="avg_nll_change_perc",
        columns="seq_length",
        values="avg_accuracy_modified_cache",
    )
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot[sorted(pivot.columns)]

    pivot.columns = [SEQ_LENGTH_LABELS[c] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGn",
        vmin=pivot.min().min(),
        vmax=pivot.max().max(),
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Percentage of KV Changed")

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
