import os
import numpy as np
from tqdm import tqdm

from utils import (
    list_of_strings, 
    list_of_floats,
    plot_success_matrix,
)

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
        rank_selection = 'comp_ratio'
    else:
        key = 'energy_threshold'
        values = energy_thresholds
        rank_selection = 'energy'

    # remove key from kwargs to overwrite defaults from the other script
    kwargs.pop(key)
    kwargs.pop('rank_selection')

    n_datasets = len(datasets)
    n_values = len(values)
    pbar = tqdm(total=n_datasets * n_values)
    success_matrix = np.zeros(
        (len(datasets), len(values)), dtype=np.float32
    )
    crs = np.ones_like(success_matrix)
    for i, dataset in tqdm(enumerate(datasets)):
        for j, value in enumerate(values):
            print(f"Testing {key}={value} on dataset {dataset}")
            try:
                success_matrix[i, j], (cr_avg, _) = benchmark_niah(**{
                    'dataset': dataset,
                    'rank_selection': rank_selection,
                    key: value,
                    **kwargs,
                })
                crs[i, j] = cr_avg
            except Exception as e:
                print(f"Error processing this ablation: \n {e}")
                print("Full traceback: \n")
                import traceback; traceback.print_exc()
                success_matrix[i, j] = None
                crs[i, j] = None
            pbar.update(1)
    
    print("Success matrix: ", success_matrix)
    print("Compression ratios: ", crs)
    plot_success_matrix(
        success_matrix,
        seq_lens,
        key,
        values,
        kwargs['cache_type'],
        save_path='results/NIAH/NIAH_ablations{}.png',
        crs=crs if key =='energy_threshold' else None,
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
        default=None,  # [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    )
    parser.add_argument(
        "--energy_thresholds", type=list_of_floats, 
        default=None,  # [0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99]
    )
    return parser


if __name__ == "__main__":
    parser = add_to_parser(
        get_parser()
    )
    args, unknown = parser.parse_known_args()
    main(**vars(args))