import os
import argparse

from process_all_results_from_logs import main as process_single_log


def main(
    log_dir,
    benchmark="lolcats",
):
    log_files = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if os.path.isfile(os.path.join(log_dir, f)) and f.endswith(".out")
    ]
    print(f"Found {len(log_files)} log files in directory {log_dir}.")

    for log_file in log_files:
        print(f"\n\n=== Processing log file: {log_file} ===")
        process_single_log(log_file, benchmark=benchmark)
        cont = input("Continue to next part? (y/n): ")
        if cont.lower() != "y":
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Print LM evaluation results.")
    parser.add_argument(
        "-d",
        "--log_dir",
        type=str,
        required=True,
        help="Path to the directory containing log files to be processed.",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        required=False,
        default="lolcats",
        help="Name of the benchmark to use when printing results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        log_dir=args.log_dir,
        benchmark=args.benchmark,
    )
