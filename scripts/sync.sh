#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail
[[ "${TRACE-0}" == "1" ]] && set -o xtrace

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    cat <<'EOF'
usage: sync.sh [-h] [--pull-judge-scored] [--pull-images QID[,QID]...]
               [--remote-image-root PATH] [--image-dest DIR] [--push LOGS_SUBDIR]

Sync the local workspace to remote and the remote logs/results to local
(ignoring files that are newer on the receiver).

Options:
  --pull-judge-scored
                       Also pull the large merged standard-sampling
                       judge-specific scored JSONLs needed for judge audits:
                       *_samples_scored_qwen_qwen3-4b_with_desc_rich.jsonl
                       *_samples_scored_google_gemma-4-e4b-it_with_desc_rich.jsonl
  --pull-images QIDS   Pull specific OVEN images by Wikidata QID. Accepts
                       comma-separated and/or space-separated QIDs, e.g.:
                       --pull-images Q517545,Q49136 Q577270
  --remote-image-root PATH
                       Remote image directory used by --pull-images.
                       Default: /leonardo_work/EUHPC_D33_243/oven/data/images
  --image-dest DIR     Local destination for --pull-images (relative to the
                       repo root unless absolute). Default: data/images
  --push LOGS_SUBDIR   Push a specific local logs/ subdirectory back to the
                       remote (only files missing on the remote).  Use this
                       to rescue data from local snapshots.
EOF
    exit 0
fi

PULL_JUDGE_SCORED=0
PULL_IMAGE_QIDS=()
REMOTE_IMAGE_ROOT="/leonardo_work/EUHPC_D33_243/oven/data/images"
IMAGE_DEST=""
PUSH_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --pull-judge-scored) PULL_JUDGE_SCORED=1; shift ;;
        --pull-images)
            shift
            if [[ $# -eq 0 || "${1-}" == --* ]]; then
                echo "Error: --pull-images requires at least one QID" >&2
                exit 1
            fi
            while [[ $# -gt 0 && "${1-}" != --* ]]; do
                IFS=',' read -ra qid_parts <<< "$1"
                for qid in "${qid_parts[@]}"; do
                    if [[ -n "$qid" ]]; then
                        PULL_IMAGE_QIDS+=("$qid")
                    fi
                done
                shift
            done
            ;;
        --remote-image-root)
            if [[ $# -lt 2 || "${2-}" == --* ]]; then
                echo "Error: --remote-image-root requires a path" >&2
                exit 1
            fi
            REMOTE_IMAGE_ROOT="$2"
            shift 2
            ;;
        --image-dest)
            if [[ $# -lt 2 || "${2-}" == --* ]]; then
                echo "Error: --image-dest requires a path" >&2
                exit 1
            fi
            IMAGE_DEST="$2"
            shift 2
            ;;
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
    --exclude="/docs/"             # thesis writing only; not needed on cluster
    --exclude="uv.lock"
    --exclude="CLAUDE.md"
    --exclude="/tests/"
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

judge_scored_include_opts=(
    --include="*/"
    --include="*_samples_scored_qwen_qwen3-4b_with_desc_rich.jsonl"
    --include="*_samples_scored_google_gemma-4-e4b-it_with_desc_rich.jsonl"
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

if [[ "$PULL_JUDGE_SCORED" -eq 1 ]]; then
    for remote in "${remotes[@]}"; do
        local_dir="$repo_root/logs"
        mkdir -p "$local_dir"
        echo "[info] Pulling large judge-specific scored JSONLs for audit: $remote/logs/ → $local_dir/ ..."
        rsync --update -azhv "${judge_scored_include_opts[@]}" \
            "$remote/logs/" "$local_dir/" \
            || echo "[warn] Could not pull judge-specific scored JSONLs"
    done
fi

if [[ ${#PULL_IMAGE_QIDS[@]} -gt 0 ]]; then
    mapfile -t image_requests < <(
        python - "$repo_root" "${PULL_IMAGE_QIDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
requested = []
seen = set()
for raw in sys.argv[2:]:
    qid = raw.strip()
    if qid and qid not in seen:
        requested.append(qid)
        seen.add(qid)

sources = [
    repo / "data/processed/vlm_compatible_val_aligned.jsonl",
    repo / "data/processed/vlm_compatible_val.jsonl",
    repo / "data/raw/oven_entity_val.jsonl",
]
records = {qid: None for qid in requested}
for source in sources:
    if not source.exists():
        continue
    with source.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("entity_id")
            if qid in records and records[qid] is None and row.get("image_id"):
                records[qid] = row
    if all(records[qid] is not None for qid in requested):
        break

missing = [qid for qid in requested if records[qid] is None]
for qid in missing:
    print(f"[warn] No local validation row found for QID {qid}", file=sys.stderr)

if len(missing) == len(requested):
    raise SystemExit(2)

for qid in requested:
    row = records[qid]
    if row is None:
        continue
    label = row.get("answer") or row.get("entity_text") or ""
    print("\t".join([qid, row["image_id"], row.get("data_id", ""), label]))
PY
    )

    if [[ ${#image_requests[@]} -eq 0 ]]; then
        echo "[error] Could not resolve any requested QIDs to OVEN image IDs" >&2
        exit 1
    fi

    if [[ -n "$IMAGE_DEST" ]]; then
        case "$IMAGE_DEST" in
            /*) local_image_dir="$IMAGE_DEST" ;;
            *)  local_image_dir="$repo_root/$IMAGE_DEST" ;;
        esac
    else
        local_image_dir="$repo_root/data/images"
    fi
    mkdir -p "$local_image_dir"
    for remote in "${remotes[@]}"; do
        remote_host="${remote%%:*}"
        echo "[info] Pulling ${#image_requests[@]} requested OVEN images from $remote_host:$REMOTE_IMAGE_ROOT/ → $local_image_dir/ ..."
        for request in "${image_requests[@]}"; do
            IFS=$'\t' read -r qid image_id data_id label <<< "$request"
            echo "[info] $qid ${label:+($label) }→ $image_id.*${data_id:+ [$data_id]}"
            # Match any extension (.jpg/.jpeg/.JPEG/.JPG/.png); the remote store
            # is not consistently lowercase .jpg. The glob expands remotely and
            # image_ids are unique, so this pulls the single matching file.
            rsync --update -azhv \
                "$remote_host:$REMOTE_IMAGE_ROOT/$image_id.*" \
                "$local_image_dir/" \
                || echo "[warn] Could not pull image for $qid ($image_id.*)"
        done
    done
fi

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
