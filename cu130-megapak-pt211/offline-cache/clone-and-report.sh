#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
REPOS_DIR="$SCRIPT_DIR/repos"
REPORT_FILE="$SCRIPT_DIR/README.md"
PRELOAD_SCRIPT="$BASE_DIR/builder-scripts/preload-nodes.sh"

mkdir -p "$REPOS_DIR"

gcs() {
    local url="$1"
    local max_attempts=3
    local attempt=1
    local wait_time=2

    while [ $attempt -le $max_attempts ]; do
        echo "--> [Attempt $attempt/$max_attempts] Cloning $url..."

        if git clone --depth=1 --no-tags --recurse-submodules --shallow-submodules "$url"; then
            return 0
        fi

        echo "--> [Warning] Clone failed for $url. Retrying in ${wait_time}s..."
        sleep $wait_time
        attempt=$((attempt + 1))
        wait_time=$((wait_time * 2))
    done

    echo "--> [Error] Failed to clone $url after $max_attempts attempts."
    return 1
}

extract_urls() {
    grep -oP '^\s*gcs\s+\Khttps://github\.com/[^\s]+\.git' "$PRELOAD_SCRIPT" | sed 's/\.git$//'
}

get_repo_name() {
    local url="$1"
    basename "$url"
}

get_repo_desc() {
    local url="$1"
    local api_url
    api_url=$(echo "$url" | sed 's|https://github.com/|https://api.github.com/repos/|')
    local desc
    desc=$(curl -sf --max-time 10 "$api_url" 2>/dev/null \
        | grep -oP '"description":\s*"\K[^"]*' \
        | head -1) || true
    if [ -z "$desc" ]; then
        desc="-"
    fi
    echo "$desc"
}

get_repo_info() {
    local repo_dir="$1"
    local url="$2"

    local branch commit commit_short commit_msg commit_time desc

    branch=$(git -C "$repo_dir" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    commit=$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || echo "unknown")
    commit_short=$(git -C "$repo_dir" rev-parse --short=7 HEAD 2>/dev/null || echo "unknown")
    commit_msg=$(git -C "$repo_dir" log -1 --format='%s' 2>/dev/null || echo "unknown")
    commit_time=$(git -C "$repo_dir" log -1 --format='%ci' 2>/dev/null || echo "unknown")
    desc=$(get_repo_desc "$url")

    commit_msg=$(echo "$commit_msg" | sed 's/|/\&#124;/g' | sed 's/\*/\\\*/g')
    desc=$(echo "$desc" | sed 's/|/\&#124;/g' | sed 's/\*/\\\*/g')

    echo "$branch|$commit|$commit_short|$commit_msg|$commit_time|$url|$desc"
}

echo "########################################"
echo "[INFO] Reading repo list from: $PRELOAD_SCRIPT"
echo "########################################"

mapfile -t URLS < <(extract_urls)

if [ ${#URLS[@]} -eq 0 ]; then
    echo "[ERROR] No GitHub URLs found in $PRELOAD_SCRIPT"
    exit 1
fi

echo "[INFO] Found ${#URLS[@]} remote repositories"

declare -A remote_names
for url in "${URLS[@]}"; do
    remote_names[$(get_repo_name "$url")]=1
done

local_names=()
if [ -d "$REPOS_DIR" ]; then
    for dir in "$REPOS_DIR"/*/; do
        [ -d "$dir" ] || continue
        name=$(basename "$dir")
        local_names+=("$name")
    done
fi

declare -A local_names_map
for name in "${local_names[@]}"; do
    local_names_map["$name"]=1
done

new_nodes=()
existing_nodes=()
removed_nodes=()

for name in "${!remote_names[@]}"; do
    if [[ -v local_names_map["$name"] ]]; then
        existing_nodes+=("$name")
    else
        new_nodes+=("$name")
    fi
done

for name in "${local_names[@]}"; do
    if [[ ! -v remote_names["$name"] ]]; then
        removed_nodes+=("$name")
    fi
done

echo ""
echo "########################################"
echo "[INFO] Node Status Detection"
echo "########################################"
echo ""
if [ ${#new_nodes[@]} -gt 0 ]; then
    echo "  🆕 New nodes (${#new_nodes[@]}):"
    for name in "${new_nodes[@]}"; do echo "     - $name"; done
fi
if [ ${#existing_nodes[@]} -gt 0 ]; then
    echo "  ✅ Existing nodes (${#existing_nodes[@]}):"
    for name in "${existing_nodes[@]}"; do echo "     - $name"; done
fi
if [ ${#removed_nodes[@]} -gt 0 ]; then
    echo "  ⚠️  Removed nodes (${#removed_nodes[@]}):"
    for name in "${removed_nodes[@]}"; do echo "     - $name"; done
fi
echo ""

if [ ${#removed_nodes[@]} -gt 0 ]; then
    if [ "${FORCE_DEPRECATE_NODE_DELETE:-}" = "true" ]; then
        echo "[INFO] FORCE_DEPRECATE_NODE_DELETE=true, auto-deleting removed nodes..."
        for name in "${removed_nodes[@]}"; do
            echo "  🗑️  Deleting: $name"
            rm -rf "$REPOS_DIR/$name"
        done
    else
        echo "[INFO] The following local repos are no longer in preload-nodes.sh:"
        for name in "${removed_nodes[@]}"; do
            echo "  - $name"
        done
        echo ""
        read -r -p "Delete these removed local repos? [y/N] " answer
        case "$answer" in
            [yY]|[yY][eE][sS])
                for name in "${removed_nodes[@]}"; do
                    echo "  🗑️  Deleting: $name"
                    rm -rf "$REPOS_DIR/$name"
                done
                ;;
            *)
                echo "[INFO] Skipped deletion of removed nodes."
                ;;
        esac
    fi
fi

echo ""
echo "########################################"
echo "[INFO] Cloning / Updating repositories..."
echo "########################################"

RESULTS=()

for url in "${URLS[@]}"; do
    repo_name=$(get_repo_name "$url")
    repo_dir="$REPOS_DIR/$repo_name"

    if [ -d "$repo_dir" ]; then
        echo ""
        echo "--> [UPDATE] $repo_name already exists, pulling latest..."
        if (cd "$repo_dir" && git fetch --all --prune && git reset --hard origin/$(git rev-parse --abbrev-ref HEAD) 2>/dev/null); then
            echo "--> [OK] $repo_name updated"
        else
            echo "--> [WARN] Failed to update $repo_name, re-cloning..."
            rm -rf "$repo_dir"
            (cd "$REPOS_DIR" && gcs "$url.git") || true
        fi
    else
        echo ""
        (cd "$REPOS_DIR" && gcs "$url.git") || true
    fi

    if [ -d "$repo_dir" ]; then
        RESULTS+=("$(get_repo_info "$repo_dir" "$url")")
    else
        RESULTS+=("unknown|unknown|unknown|CLONE_FAILED|unknown|$url|-")
    fi
done

echo ""
echo "########################################"
echo "[INFO] Generating report: $REPORT_FILE"
echo "########################################"

{
    echo "# ComfyUI Custom Nodes - Repository Report"
    echo ""
    echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "| # | Repository | Description | Branch | Commit | Commit Message | Commit Time |"
    echo "|---|-----------|-------------|--------|--------|---------------|-------------|"

    idx=1
    for entry in "${RESULTS[@]}"; do
        IFS='|' read -r branch commit commit_short commit_msg commit_time url desc <<< "$entry"
        repo_name=$(get_repo_name "$url")
        repo_link="[${repo_name}](${url})"
        commit_link="[${commit_short}](${url}/commit/${commit})"
        echo "| ${idx} | ${repo_link} | ${desc} | ${branch} | ${commit_link} | ${commit_msg} | ${commit_time} |"
        idx=$((idx + 1))
    done

    echo ""
    echo "---"
    echo ""
    echo "## Summary"
    echo ""
    echo "- Total repositories: ${#RESULTS[@]}"
    echo "- Source: \`builder-scripts/preload-nodes.sh\`"
    echo ""

} > "$REPORT_FILE"

echo "[INFO] Report generated: $REPORT_FILE"
echo "[INFO] Total: ${#RESULTS[@]} repositories"