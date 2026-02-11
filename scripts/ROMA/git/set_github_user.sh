#!/usr/bin/bash
set -euo pipefail

source .env

git config --local user.name $GIT_USER
git config --local user.email $GIT_EMAIL
