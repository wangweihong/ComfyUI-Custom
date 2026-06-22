#!/bin/bash

set -euo pipefail

function git_force_sync () {
    local repo_dir="$1"
    local max_attempts=3
    local attempt=1
    local wait_time=2
    local git_remote_url

    git_remote_url=$(git -C "$repo_dir" remote get-url origin 2>/dev/null) || return 0

    if [[ ! $git_remote_url =~ ^(https:\/\/github\.com\/)(.*)(\.git)$ ]]; then
        return 0
    fi

    while [ $attempt -le $max_attempts ]; do
        if git -C "$repo_dir" fetch --depth=1 --no-tags 2>/dev/null; then
            local _local_head _remote_head
            _local_head=$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || echo "")
            _remote_head=$(git -C "$repo_dir" rev-parse '@{upstream}' 2>/dev/null || echo "")

            if [ -n "$_local_head" ] && [ -n "$_remote_head" ] \
               && [ "$_local_head" != "$_remote_head" ]; then
                echo "[INFO] Updating: $repo_dir"
                if git -C "$repo_dir" reset --hard '@{upstream}' 2>/dev/null \
                   && git -C "$repo_dir" submodule update --init --recursive --depth=1 2>/dev/null; then
                    echo "[INFO] Done Updating: $repo_dir"
                    return 0
                fi
            else
                return 0
            fi
        fi

        echo "[WARN] Attempt $attempt/$max_attempts failed for $repo_dir, retrying in ${wait_time}s..."
        sleep $wait_time
        attempt=$((attempt + 1))
        wait_time=$((wait_time * 2))
    done

    echo "[ERROR] Failed to sync $repo_dir after $max_attempts attempts."
    return 1
}

function git_clone_retry () {
    local max_attempts=3
    local attempt=1
    local wait_time=2
 
    while [ $attempt -le $max_attempts ]; do
        echo "[INFO] Cloning (attempt $attempt/$max_attempts)..."
        if git clone --depth=1 --no-tags "$@"; then
            return 0
        fi
 
        echo "[WARN] Clone failed, retrying in ${wait_time}s..."
        sleep $wait_time
        attempt=$((attempt + 1))
        wait_time=$((wait_time * 2))
    done
 
    echo "[ERROR] Failed to clone after $max_attempts attempts."
    return 1
}


echo "########################################"
echo "[INFO] Updating ComfyUI..."

cd /default-comfyui-bundle/ComfyUI

git fetch --all --tags --prune --prune-tags
git reset --hard '@{upstream}'

# Using stable version (has a release tag)
## 注意：在 Git 的逻辑里，Tag（标签）是一个静态的“快照指针”。它一旦打在某个具体的 Commit（提交）上，除非你手动删除并重新打标签，否则它永远指向那个时间点的代码状态。
## 之后提交的commit, 不会被包含在标签的分支中。即使这些commit早于这个边
git reset --hard "$(git tag -l 'v*' | sort -V | tail -1)"

echo "########################################"
echo "[INFO] Updating Custom Nodes..."

cd /default-comfyui-bundle/ComfyUI/custom_nodes

for D in *; do
    if [ -d "${D}" ]; then
        git_force_sync "${D}" &
    fi
done

# Do not quote (jobs -p), word splitting is intended here.
wait $(jobs -p)

echo "########################################"
echo "[INFO] Installing additional Custom Nodes..."

cd /default-comfyui-bundle/ComfyUI/custom_nodes

# FastVideo
git_clone_retry https://github.com/hao-ai-lab/FastVideo.git \
    /tmp/FastVideo

mkdir -p /default-comfyui-bundle/ComfyUI/custom_nodes/FastVideo
cp --archive --update=none "/tmp/FastVideo/comfyui/." "/default-comfyui-bundle/ComfyUI/custom_nodes/FastVideo/"
rm -rf /tmp/FastVideo

# ComfyUI-SageAttention3
# A simple connector node for adapting SA3
cd /default-comfyui-bundle/ComfyUI/custom_nodes
git_clone_retry https://github.com/wallen0322/ComfyUI-SageAttention3.git

echo "########################################"
echo "[INFO] Configuring ComfyUI & Manager..."

mkdir -p /default-comfyui-bundle/ComfyUI/user/default

# Enable TAESD preview by default
cat <<EOF > /default-comfyui-bundle/ComfyUI/user/default/comfy.settings.json
{
    "Comfy.Execution.PreviewMethod": "taesd"
}
EOF

# Configure Manager
mkdir -p /default-comfyui-bundle/ComfyUI/user/__manager

cat <<EOF > /default-comfyui-bundle/ComfyUI/user/__manager/config.ini
[default]
use_uv = False
security_level = weak
downgrade_blacklist = torch, torchvision, torchaudio
db_mode = local
network_mode = personal_cloud
EOF

echo "########################################"