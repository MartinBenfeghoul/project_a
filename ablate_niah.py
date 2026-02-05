import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import list_of_strings, list_of_floats

from niah import (
    main as benchmark_niah,
    get_parser,
)


def get_dataset_paths(seq_lens, data_dir):
    dataset_paths = []
    for seq_len in seq_lens:
        dataset_path = os.path.join(data_dir, seq_len.lower())
        if os.path.exists(dataset_path):
            dataset_paths.append(dataset_path)
        else:
            raise ValueError(
                f"Requested dataset path {dataset_path} for seq_len {seq_len} does not exist."
            )
    return dataset_paths

def get_unique_save_path(save_path):
    if not os.path.exists(save_path.format('')):
        return save_path.format('')
    for i in range(100):
        new_path = save_path.format(f'_{i}')
        if not os.path.exists(new_path):
            return new_path
    raise ValueError(
        f"There appears to be at least 100 numbered variations of {save_path}!"
    )

def plot_results(
    success_matrix,
    seq_lens,
    key,
    values,
    cache_type,
    save_path='NIAH_ablations{}.png',
):
    plt.imshow(success_matrix)
    plt.colorbar()
    for i in range(success_matrix.shape[0]):
        for j in range(success_matrix.shape[1]):
            plt.text(
                j, i, f"{success_matrix[i, j]:.2f}",
                ha="center", va="center",
                color="white"
            )
    plt.xticks(values)
    plt.xticks(
        ticks=range(len(values)),
        labels=values
    )
    plt.yticks(
        ticks=range(len(seq_lens)),
        labels=seq_lens
    )
    plt.xlabel(key)
    plt.ylabel("Sequence Length")
    plt.title(f"Ablating cache type {cache_type}")

    save_path = get_unique_save_path(save_path)
    plt.savefig(save_path, dpi=300)

def main(
    data_dir: str,
    seq_lens: list[str],
    comp_ratios: list[float],
    energy_thresholds: list[float],
    **kwargs,
):
    """
    The point of this function is to benchmark a key comrpession method on
     a matrix of sequence lengths and compression ratios/energy thresholds.
    """

    datasets = get_dataset_paths(seq_lens, data_dir)


    comp_ratios = comp_ratios
    energy_thresholds = energy_thresholds
    assert comp_ratios is None or energy_thresholds is None, "You can choose to either ablate over compression ratios or energy_thresholds"

    if comp_ratios is not None:
        key = 'comp_ratio'
        values = comp_ratios
    else:
        key = 'energy_threshold'
        values = energy_thresholds
        raise NotImplementedError("Energy-based rank selection is not currently implemented.")

    # remove key from kwargs to overwrite defaults from the other script
    kwargs.pop(key)

    n_datasets = len(datasets)
    n_values = len(values)
    pbar = tqdm(total=n_datasets * n_values)
    success_matrix = np.zeros(
        (len(datasets), len(values)), dtype=np.float32
    )
    for i, dataset in tqdm(enumerate(datasets)):
        for j, value in enumerate(values):
            print(f"Testing {key}={value} on dataset {dataset}")
            try:
                success_matrix[i, j] = benchmark_niah(**{
                    'dataset': dataset,
                    key: value,
                    **kwargs,
                })
            except Exception as e:
                print(f"Error processing this ablation: \n {e}")
                print("Full traceback: \n")
                import traceback; traceback.print_exc()
                # TODO: test entering a different value (eg. inf) in matrix to denote error
            pbar.update(1)
    
    print(success_matrix)
    plot_results(
        success_matrix,
        seq_lens,
        key,
        values,
        kwargs['cache_type'],
        save_path='NIAH_ablations{}.png',
    )

def add_to_parser(parser):
    parser.add_argument(
        "--data_dir", type=str, default='data/NIAH/multi-keys'
    )
    parser.add_argument(
        "-s", "--seq_lens", type=list_of_strings, default=['1k', '2k', '4k', '8k']
    )
    parser.add_argument(
        "--comp_ratios", type=list_of_floats, 
        default=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    )
    parser.add_argument(
        "--energy_thresholds", type=list_of_floats, 
        default=None,  # [0.95, 0.96, 0.97, 0.99]
    )
    return parser


if __name__ == "__main__":
    parser = add_to_parser(
        get_parser()
    )
    args, unknown = parser.parse_known_args()
    main(**vars(args))