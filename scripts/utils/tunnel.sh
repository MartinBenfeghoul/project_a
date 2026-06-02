#!/bin/bash
#SBATCH --job-name=martin_dev
#SBATCH --partition=agent-long,agent-long-15
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

set -e

CLI_PATH="/tmp/${USER}-vscode_cli"
echo "CLI_PATH: ${CLI_PATH}"

# Install the VS Code CLI command if it doesn't exist
if [[ ! -e ${CLI_PATH}/code ]]; then
  echo "Downloading and installing the VS Code CLI command"
  mkdir -p "${CLI_PATH}"
  pushd "${CLI_PATH}"
  # Process from: https://code.visualstudio.com/docs/remote/tunnels#_using-the-code-cli
  curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
  # unpack the code binary file
  tar -xf vscode_cli.tar.gz
  # clean-up
  rm vscode_cli.tar.gz
  popd
fi

# run the code tunnel command and accept the licence
srun ${CLI_PATH}/code tunnel --accept-server-license-terms
