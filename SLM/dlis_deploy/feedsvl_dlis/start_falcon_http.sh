#!/usr/bin/env bash

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-falcon}"
PORT="${1:-${_ListeningPort_:-18080}}"
PARALLELISM="${_Parallelism_:-1}"

export CURRENT_PATH="$DIR"
export _ListeningPort_="$PORT"
export _Parallelism_="$PARALLELISM"

echo "Starting HTTP service from: $DIR"
echo "Using conda env: $ENV_NAME"
echo "Listening port: $PORT"
echo "Parallelism: $PARALLELISM"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
    exec python "$DIR/model/main.py" http
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH" >&2
    exit 1
fi

exec conda run -n "$ENV_NAME" python "$DIR/model/main.py" http