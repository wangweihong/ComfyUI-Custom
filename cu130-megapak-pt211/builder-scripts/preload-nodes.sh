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

# toolbox
gcs https://github.com/wangweihong/ComfyUI-ToolBox.git
# audio (不支持cu130)
#gcs https://github.com/wangweihong/ComfyUI-faster-whisper.git

# Qwen3-VL-Instruct
gcs https://github.com/wangweihong/ComfyUI_Qwen3-VL-Instruct.git

# audio separation nodes
gcs https://github.com/christian-byrne/audio-separation-nodes-comfyui.git

# qwen3 tts
gcs https://github.com/flybirdxx/ComfyUI-Qwen-TTS.git

# whisper tts
gcs https://github.com/1038lab/ComfyUI-EdgeTTS.git

# index tts 2
gcs https://github.com/yolain/ComfyUI-Easy-IndexTTS2.git

# ltx/wan vace prep (视频外展)
gcs https://github.com/stuttlepress/ComfyUI-Wan-VACE-Prep.git

# gaussian blur
gcs https://github.com/HallettVisual/comfyui-GaussianViewer.git
gcs https://github.com/PozzettiAndrea/ComfyUI-Sharp.git

# qwenmultiangle
gcs https://github.com/jtydhr88/ComfyUI-qwenmultiangle.git

# vram/mem cleanup
gcs https://github.com/LAOGOU-666/Comfyui-Memory_Cleanup.git

# zit seed variance enhancer
gcs https://github.com/ChangeTheConstants/SeedVarianceEnhancer.git

# sam3
gcs https://github.com/yolain/ComfyUI-Easy-Sam3.git

# seedvr2 8k
gcs https://github.com/TTPlanetPig/Comfyui_TTP_Toolset.git

## 图像掩码
gcs https://github.com/BadCafeCode/masquerade-nodes-comfyui.git 

# 视频水印
gcs https://github.com/Artificial-Sweetener/comfyui-WhiteRabbit.git

# flux2klein 
gcs https://github.com/princepainter/Comfyui-PainterFluxImageEdit.git

# 3d pose ood (效果比dwPose好)
gcs https://github.com/judian17/ComfyUI-SDPose-OOD.git

# flux2klein enhancer
## https://www.reddit.com/r/StableDiffusion/comments/1somo2r/coming_up_tomorrow_flux2klein_identity_transfer/
gcs https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer.git

# convas
## from https://www.reddit.com/r/comfyui/comments/1soqoz6/comfy_canvas_v10/
gcs https://github.com/Zlata-Salyukova/Comfy-Canvas.git

# klein enhancer
## 扩散模型的通用负面引导
gcs https://github.com/BigStationW/ComfyUI-NAG.git
## 图像缩放到总像素级别
gcs https://github.com/BigStationW/ComfyUi-Scale-Image-to-Total-Pixels-Advanced.git

echo "[INFO] Additional custom nodes downloaded successfully"