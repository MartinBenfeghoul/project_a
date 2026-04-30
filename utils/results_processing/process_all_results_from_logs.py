import os

from print_lm_eval import main as print_results
from extract_results_from_logs import parse_args, main as extract_results


def main(
    log_file,
    file_path="./results/extracted_results_{}.json",
    add_key_configs=True,
    start_from_line=0,
    stop_at_line=-1,
    print_each_part=True,
    benchmark="ruler",  # TODO: handle this more dynamically
):
    with open(log_file, "r") as f:
        lines = f.readlines()
        if stop_at_line == -1:
            stop_at_line = len(lines)
        print(
            f"File contains {len(lines)} lines. ",
            f"Reading lines from {start_from_line} to {stop_at_line}...",
        )
        lines = lines[start_from_line:stop_at_line]
        parts = []
        st = start_from_line
        for i, line in enumerate(lines):
            if "Results saved to" in line:
                results_file = (
                    line.strip().split("Results saved to")[-1].strip()
                )
                ed = start_from_line + i + 1
                parts.append((st, ed, results_file))
                st = ed

    for i, (st, end, results_file) in enumerate(parts):
        print(f"\n\n=== Extracting part {i} (lines {st} to {end}) ===")
        if (
            results_file
            and results_file.endswith(".json")
            and os.path.exists(results_file)
        ):
            print(
                f"Results file {results_file} already exists, skipping extraction."
            )
        else:
            try:
                results_file = extract_results(
                    log_file,
                    file_path,
                    add_key_configs=add_key_configs,
                    start_from_line=st,
                    stop_at_line=end,
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Error extracting part {i}: {e}")
                results_file = None

        if results_file and print_each_part:
            print(
                f"\n--- Printing results for part {i}, lines {st} to {end} ---"
            )
            print_results(
                benchmark, results_file, show_key_configs=add_key_configs
            )

        # ask user whether to continue
        if i < len(parts) - 1:
            cont = input("Continue to next part? (y/n): ")
            if cont.lower() != "y":
                break
        else:
            print("No more parts to extract.")


if __name__ == "__main__":
    args = parse_args()
    log_file = args.log_path
    add_key_configs = not args.no_key_configs
    start_from_line = args.start_from_line
    stop_at_line = args.stop_at_line

    main(
        log_file,
        add_key_configs=add_key_configs,
        start_from_line=start_from_line,
        stop_at_line=stop_at_line,
    )
