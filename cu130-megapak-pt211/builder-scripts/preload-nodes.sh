#!/bin/bash

set -euo pipefail

gcs() {
    git clone --depth=1 --no-tags --recurse-submodules --shallow-submodules "$@"
}

echo "########################################"
echo "[INFO] Downloading Additional Custom Nodes..."
echo "########################################"

cd /default-comfyui-bundle/ComfyUI/custom_nodes

# 3d trellies.2
gcs https://github.com/visualbruno/ComfyUI-Trellis2.git
# gcs https://github.com/example/node1.git

echo "[INFO] Additional custom nodes downloaded successfully"