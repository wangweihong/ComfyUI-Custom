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


# klein edit composite
## https://www.reddit.com/r/StableDiffusion/comments/1rstals/how_do_you_handle_klein_edits_colour_drift/
## 可以在某些程度上解决klein编辑的色彩漂移问题。见klein人物换装-v3
gcs https://github.com/supermansundies/comfyui-klein-edit-composite.git
gcs https://github.com/BigStationW/ComfyUi-TextEncodeEditAdvanced.git


# yedp action director
## https://www.linkedin.com/posts/sedpid_hello-everyone-i-just-pushed-a-big-update-activity-7432741280683642882-3eio    
gcs https://github.com/yedp123/ComfyUI-Yedp-Action-Director.git

# Qwen3.5
## 依赖transformers > 5.0
gcs https://github.com/WingeD123/ComfyUI_QwenVL_PromptCaption.git

# ltx2.3 VR 360外扩
gcs https://github.com/Burgstall-labs/ComfyUI-EquirectProjector.git

# Sapiens2
## 一系列高分辨率 Transformer 模型，在 10 亿张人体图像上进行预训练，在各种以人为中心的任务（姿态估计、身体部位分割、表面法线和点图）中取得了最先进的性能。
gcs https://github.com/kijai/ComfyUI-Sapiens2.git

# SKBundle
## 一系列自定义节点，包括：
## 1. PaintPro: 使用压感画笔、橡皮擦和形状工具直接在节点上绘制和遮罩
## 2. Lens Flare: 为图像添加逼真的镜头光晕效果。您可以自定义光晕类型、大小、旋转和强度等设置
## 3. TitlePlus: 为视频添加标题，支持自定义字体、颜色、大小等设置
## 4. SeamlessTexture: 为图像添加无缝纹理效果，使图像看起来更真实
## 5. AspectRatioPlus: 高级宽高比调整节点，支持自定义宽高比
gcs https://github.com/SKBv0/ComfyUI_SKBundle.git

# ZML Lora Power
gcs https://github.com/zml-w/ComfyUI-ZML-Image.git

# ltx prompt relay
gcs https://github.com/kijai/ComfyUI-PromptRelay.git

# 音效
gcs https://github.com/Saganaki22/ComfyUI-Woosh.git


# Reference video
gcs https://github.com/alisson-anjos/ComfyUI-BFSNodes.git
gcs https://github.com/lucafoscili/lf-nodes.git

# 很好用的加载视频,支持加载时裁减
gcs https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git

echo "[INFO] Additional custom nodes downloaded successfully"