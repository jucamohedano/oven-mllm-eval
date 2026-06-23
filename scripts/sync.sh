#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail
[[ "${TRACE-0}" == "1" ]] && set -o xtrace

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    cat <<'EOF'
usage: sync.sh [-h] [--push LOGS_SUBDIR]

Sync the local workspace to remote and the remote logs/results to local
(ignoring files that are newer on the receiver).

Options:
  --push LOGS_SUBDIR   Push a specific local logs/ subdirectory back to the
                       remote (only files missing on the remote).  Use this
                       to rescue data from local snapshots.
EOF
    exit 0
fi

PUSH_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --push) PUSH_DIR="$2"; shift 2 ;;
        *) echo "Error: unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate repo root
# ---------------------------------------------------------------------------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$script_dir"
while [[ ! -f "$repo_root/pyproject.toml" ]]; do
    repo_root="$(dirname "$repo_root")"
    if [[ "$repo_root" == "/" ]]; then
        echo "[error] Could not find pyproject.toml. Are you running this from the oven-mllm-eval repository?"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Exclude patterns
# ---------------------------------------------------------------------------
workspace_exclude_opts=(
    --exclude=".cache/"
    --exclude=".venv/"
    --exclude=".pytest_cache/"
    --exclude=".vscode/"
    --exclude="__pycache__/"
    --exclude="*.egg-info/"
    --exclude="/data/images/"       # huge — download separately on cluster
    --exclude="/data/raw/"          # download separately on cluster
    --exclude="/data/processed/"    # download separately on cluster
    --exclude="/results/"           # synced back separately below
    --exclude="/logs/"              # generated on cluster; synced back separately below
    --exclude="uv.lock"
)

raw_data_files=(
    "oven_wikidata_chains_cleaned_descs.jsonl"
)

results_exclude_opts=(
    --exclude="/debug/"
    --exclude="/slurm/"
    --include="*/"
    --include="runs.tsv"
    --include="*_metadata.json"
    --include="generations_results.json"
    --include="*_results*.json"
    --exclude="*"
)

# ---------------------------------------------------------------------------
# Read remotes from configuration file
# ---------------------------------------------------------------------------
config_file="$repo_root/configs/sync.conf"
if [[ ! -f "$config_file" ]]; then
    echo "[error] Configuration file not found at $config_file"
    echo "        Create it with one remote path per line, e.g.:"
    echo "        user@cluster:/path/to/oven-mllm-eval/"
    exit 1
fi

mapfile -t remotes < <(grep -v '^\s*$\|^\s*#' "$config_file" | sed 's|/*$||')

if [[ ${#remotes[@]} -eq 0 ]]; then
    echo "[error] No remotes found in $config_file"
    exit 1
fi

# ---------------------------------------------------------------------------
# Sync local workspace → each remote (with --delete to remove stale files)
# ---------------------------------------------------------------------------
for remote in "${remotes[@]}"; do
    echo "[info] Syncing $repo_root/ → $remote/ ..."
    rsync -azhv "${workspace_exclude_opts[@]}" "$repo_root/" "$remote/"

    ssh "${remote%%:*}" \
        "mkdir -p ${remote#*:}/{results,logs,data/raw,data/processed,data/images}" \
        2>/dev/null \
        || echo "[warn] Could not create remote directories via ssh"

    for data_file in "${raw_data_files[@]}"; do
        local_data_path="$repo_root/data/raw/$data_file"
        if [[ -f "$local_data_path" ]]; then
            echo "[info] Syncing data/raw/$data_file → $remote/data/raw/ ..."
            rsync -azhv "$local_data_path" "$remote/data/raw/" \
                || echo "[warn] Could not sync data/raw/$data_file"
        else
            echo "[warn] Missing local data/raw/$data_file; skipping"
        fi
    done
done

# ---------------------------------------------------------------------------
# Sync remote results/logs/data back → local (only newer files; non-fatal)
# ---------------------------------------------------------------------------
declare -A back_sync=(
    [logs]="logs"
)

for remote in "${remotes[@]}"; do
    for dir in "${!back_sync[@]}"; do
        local_dir="$repo_root/${back_sync[$dir]}"
        mkdir -p "$local_dir"
        echo "[info] Syncing $remote/$dir/ → $local_dir/ ..."
        rsync --update -azhv "${results_exclude_opts[@]}" \
            "$remote/$dir/" "$local_dir/" \
            || echo "[warn] Could not sync $dir/ (may not exist on remote yet)"
    done
done

# ---------------------------------------------------------------------------
# Targeted push: rescue local data back to remote (only missing files)
# ---------------------------------------------------------------------------
if [[ -n "$PUSH_DIR" ]]; then
    local_path="$repo_root/logs/$PUSH_DIR"
    if [[ ! -d "$local_path" ]]; then
        echo "[error] Local directory not found: $local_path" >&2
        exit 1
    fi
    for remote in "${remotes[@]}"; do
        echo "[info] Pushing missing files: $local_path/ → $remote/logs/$PUSH_DIR/ ..."
        rsync --ignore-existing -azhv "$local_path/" "$remote/logs/$PUSH_DIR/" \
            || echo "[warn] Push failed (may not be writable)"
    done
fi
