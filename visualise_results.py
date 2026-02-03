import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results = []
with open("results_needle_mse_long.jsonl", "r") as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))

df = pd.DataFrame(results)

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
ax.set_title("Passkey Retrieval Accuracy vs KV Cache Compression")

plt.tight_layout()
plt.savefig("results_visualisation.png", dpi=150)
print("Plot saved to results_visualisation.png")

print("\nSummary Table:")
print(pivot)
