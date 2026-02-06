import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    path = "./results_language_modeling_mse.jsonl"
    #path = './results_needle_mse_100.jsonl'
    path = './results_needle_mse_long (1).jsonl'
    #save_path = "nll_heatmap_lm_mse.png"
    save_path = "nll_heatmap_needle_mse.png"
    colums = "num_token_per_training"
    #values = "avg_nll_change_perc"
    values = "avg_accuracy_modified_cache"
    index = "percentage_changed_kv"
    ds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            new_l = json.loads(line)
            new_l["percentage_changed_kv"] = round(new_l["percentage_changed_kv"])
            new_l["avg_accuracy_modified_cache"] = new_l["avg_accuracy_modified_cache"] * 100
            ds.append(new_l)
    if values == "avg_accuracy_modified_cache": vmin, vmax = 70, 98
    else: vmin, vmax = 2.70, 3
    df = pd.DataFrame(ds)
    heatmap_df = df.pivot(
        index=index,
        columns=colums,
        values=values,
    )
    plt.figure(figsize=(10, 6))

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=vmin, vmax=vmax,
    )

    plt.title("NLL with updated KV cache", fontsize=16)
    plt.xlabel("Sequence length")
    plt.ylabel("Percentage KV changed")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
