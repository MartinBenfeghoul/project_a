import os
import json
import argparse

from print_lm_eval import (
    ALL_TASKS,
    KEY_CONFIGS,
)


def convert_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        return v


def get_output_path(output_path):
    for i in range(100):
        if not os.path.exists(output_path.format(i)):
            return output_path.format(i)
    raise RuntimeError(
        f"No free output path found using template: {output_path!r}"
    )


def print_dict(d, prefix=""):
    """Prints a dictionary in a readable format."""
    for key, value in d.items():
        if isinstance(value, dict) or str(value).startswith("{"):
            print(f"{prefix}{key}:")
            print_dict(value, prefix=prefix + "   ")
        else:
            print(f"{prefix}{key}: {value}")


def check_for_key_configs(line):
    for key in KEY_CONFIGS:
        k = key.split(".")[-1]
        if f"{k}:" in line:
            return True, key
    return False, None


def extract_config_from_line(line, key, dict_obj):
    parts = key.split(".")
    key_name = parts[-1]
    value = line.split(f"{key_name}:")[-1].strip()
    if len(value) == 0:
        raise ValueError(f"No value found for key {key} in line: {line}")
    if len(parts) > 1:
        for part in parts[:-1]:
            if part not in dict_obj:
                dict_obj[part] = {}
            dict_obj = dict_obj[part]
    dict_obj[key_name] = convert_to_bool(value)


def check_for_results(line):
    count = line.count("|")
    if count >= 3:
        return True
    return False


def extract_results_from_line(line, dict_obj, prev_task_name=None):
    parts = line.split("|")
    parts = [p.strip() for p in parts]
    _, task_name, version, filter_name, n_shot, metric, _, value, _, std, _ = (
        parts
    )

    full_task_name = task_name
    return_task_name = task_name
    if task_name.startswith("- "):
        full_task_name = f"{prev_task_name}_{task_name[2:].strip()}"
        return_task_name = prev_task_name
    elif task_name == "":
        full_task_name = prev_task_name
        return_task_name = prev_task_name
    elif task_name not in ALL_TASKS:
        return

    full_metric = f"{metric},{filter_name}"
    full_std = f"{metric}_stderr,{filter_name}"

    if full_task_name not in dict_obj:
        dict_obj[full_task_name] = {}
    dict_obj[full_task_name][full_metric] = float(value)
    if std == "N/A":
        dict_obj[full_task_name][full_std] = std
    else:
        dict_obj[full_task_name][full_std] = float(std)

    return return_task_name


def main(
    log_file,
    file_path,
    add_key_configs=True,
    start_from_line=0,
    stop_at_line=-1,
):
    # Stream log text file line by line, looking for configs or printed results
    res = {
        "config": {},
    }
    prev_task_name = None
    with open(log_file, "r") as f:
        lines = f.readlines()
        if stop_at_line == -1:
            stop_at_line = len(lines)
        print(
            f"File contains {len(lines)} lines. ",
            f"Reading lines from {start_from_line} to {stop_at_line}...",
        )
        lines = lines[start_from_line:stop_at_line]
        for i, line in enumerate(lines):
            if "Traceback" in line or "Error" in line:
                raise RuntimeError(
                    f"Error found in log file line {start_from_line + i}: {line}"
                )
            if add_key_configs:
                has_key_config, key = check_for_key_configs(line)
                if has_key_config:
                    extract_config_from_line(line, key, res["config"])
                    continue
            has_results = check_for_results(line)
            if has_results:
                prev_task_name = extract_results_from_line(
                    line, res, prev_task_name=prev_task_name
                )
                continue
    # Save extracted results to JSON file
    if res and len(res) > 1:
        print_dict(res)
        assert file_path.endswith(
            "json"
        ), "Extracted file path does not end with 'json'"
        file_path = get_output_path(file_path)
        print(f"\nSaving extracted results to {file_path} ...")
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
            raise FileExistsError(
                f"File {file_path} already exists."
                " Please remove it before proceeding."
            )
        else:
            with open(file_path, "w") as f:
                json.dump(res, f, indent=4)
            print(f"Extracted results saved to: {file_path}")
    else:
        print("No results extracted.")
    return file_path


def parse_args():
    parser = argparse.ArgumentParser(description="Print LM evaluation results.")
    parser.add_argument(
        "-l",
        "--log_path",
        type=str,
        required=True,
        help="Path to the input text (or .out) file with evaluation results.",
    )
    parser.add_argument(
        "--no_key_configs",
        action="store_true",
        help="Whether to skip extracting key configs.",
    )
    parser.add_argument(
        "--start_from_line",
        type=int,
        default=0,
        help="Line number to start reading from.",
    )
    parser.add_argument(
        "--stop_at_line",
        type=int,
        default=-1,
        help="Line number to stop reading at. -1 means read till end.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_file = args.log_path
    add_key_configs = not args.no_key_configs
    start_from_line = args.start_from_line
    stop_at_line = args.stop_at_line

    file_path = "./results/extracted_results_{}.json"
    main(
        log_file,
        file_path,
        add_key_configs=add_key_configs,
        start_from_line=start_from_line,
        stop_at_line=stop_at_line,
    )
