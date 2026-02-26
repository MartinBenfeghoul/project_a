import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Submit a sequence of sbatch scripts.")
    parser.add_argument("-n", "--n", type=int, default=None, help="Number of runs to submit (e.g. 20 will run.sh, run1.sh, ... run20.sh)")
    parser.add_argument("-s", "--start", type=int, default=0, help="Starting index (default: 0)")
    parser.add_argument("--run_dir", type=str, default="scripts/lazy_runs", help="Directory containing the run scripts")
    args = parser.parse_args()

    if args.n is None:
        files = os.listdir(args.run_dir)
        run_scripts = [f for f in files if f.startswith("run") and f.endswith(".sh")]
        args.n = len(run_scripts) - 1  # Adjust for zero-based index
    for i in range(args.start, args.n + 1):  # include run.sh (0) through runN.sh
        script = f"{args.run_dir}/run{i}.sh" if i > 0 else f"{args.run_dir}/run.sh"
        print(f"Submitting {script}...")
        subprocess.run(["sbatch", script], check=True)

if __name__ == "__main__":
    main()