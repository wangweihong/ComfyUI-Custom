#!/bin/bash

set -euo pipefail

# gcs() {
#     git clone --depth=1 --no-tags --recurse-submodules --shallow-submodules "$@"
# }
# 带有重试机制的克隆函数
gcs() {
    local url="$1"
    local max_attempts=3
    local attempt=1
    local wait_time=2

    while [ $attempt -le $max_attempts ]; do
        echo "--> [Attempt $attempt/$max_attempts] Cloning $url..."
        
        # 尝试执行 git clone，如果成功则直接返回
        if git clone --depth=1 --no-tags --recurse-submodules --shallow-submodules "$url"; then
            return 0
        fi

        echo "--> [Warning] Clone failed for $url. Retrying in ${wait_time}s..."
        sleep $wait_time
        attempt=$((attempt + 1))
        wait_time=$((wait_time * 2)) # 指数退避，第一次等2秒，第二次等4秒
    done

    # 如果达到最大尝试次数仍失败，则触发 set -e 导致脚本退出
    echo "--> [Error] Failed to clone $url after $max_attempts attempts."
    return 1
}

gcsBranchRename() {
    local url="$1"
    local max_attempts=3
    local attempt=1
    local wait_time=2

    local branch="$2"
     local rename="$3"

    while [ $attempt -le $max_attempts ]; do
        echo "--> [Attempt $attempt/$max_attempts] Cloning $url..."
        
        # 尝试执行 git clone，如果成功则直接返回
        if git clone --depth=1 --branch $branch --no-tags --recurse-submodules --shallow-submodules "$url" "$rename"; then
            return 0
        fi

        echo "--> [Warning] Clone failed for $url. Retrying in ${wait_time}s..."
        sleep $wait_time
        attempt=$((attempt + 1))
        wait_time=$((wait_time * 2)) # 指数退避，第一次等2秒，第二次等4秒
    done

    # 如果达到最大尝试次数仍失败，则触发 set -e 导致脚本退出
    echo "--> [Error] Failed to clone $url after $max_attempts attempts."
    return 1
}

echo "########################################"
echo "[INFO] Downloading Additional Custom Nodes..."
echo "########################################"

cd /default-comfyui-bundle/ComfyUI/custom_nodes

# 分类规则：模型相关节点及配套工具集中归组；跨模型共用工具按功能归组。
# 原有仓库、链接注释、固定提交和注释禁用状态均保留；每个安装条目只列一次。

# =============================================================================
# A 通用功能节点
# =============================================================================

# -----------------------------------------------------------------------------
# A01 综合工具与工作流
# -----------------------------------------------------------------------------

# toolbox
gcs https://github.com/wangweihong/ComfyUI-ToolBox.git
# Pin the reviewed 3DGS source so both dependency variants have the same nodes/resources.
git -C ComfyUI-ToolBox fetch --depth=1 origin 1a131682cfe467f2211626007c63c7678f6c3db9
git -C ComfyUI-ToolBox checkout --detach 1a131682cfe467f2211626007c63c7678f6c3db9

gcs https://github.com/lucafoscili/lf-nodes.git

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

# 10Eros
gcs https://github.com/TenStrip/10S-Comfy-nodes.git

# 更好的高斯
gcs https://github.com/jamesWalker55/comfyui-various.git

# easy media
# https://www.bilibili.com/video/BV14NL26nE93/?spm_id_from=333.1387.homepage.video_card.click
gcs https://github.com/yolain/ComfyUI-Easy-Media.git

# vnccs Has bug
# gcs https://github.com/AHEKOT/ComfyUI_VNCCS.git
# gcs https://github.com/AHEKOT/ComfyUI_VNCCS_Utils.git

# -----------------------------------------------------------------------------
# A02 界面交互、图库与布局
# -----------------------------------------------------------------------------

# convas
## from https://www.reddit.com/r/comfyui/comments/1soqoz6/comfy_canvas_v10/
gcs https://github.com/Zlata-Salyukova/Comfy-Canvas.git

# 不改变loadImage节点的前提下，提供load image gallery能力
gcs https://github.com/laurigates/comfyui-gallery-loader

# 支持image browser以及查看资产元数据
gcs https://github.com/laurigates/comfyui-image-browser.git

# PCB版连线
gcs https://github.com/vtokic/comfyui-cable-management.git

# 连线捆绑
gcs https://github.com/safzanpirani/comfyui-zip-ties

# moss studio
gcs https://github.com/wangweihong/ComfyUI-MossLayoutStudio.git

# ShowMe
#gcs https://github.com/SKBv0/ComfyUI_ShowMe.git

# -----------------------------------------------------------------------------
# A03 通用提示词与 LLM 接口
# -----------------------------------------------------------------------------

# llm api
gcs https://github.com/wangweihong/comfyui_LLM_party.git

# illustion 随机提示词生成
gcs https://github.com/rainlizard/ComfyUI-Raffle.git

# -----------------------------------------------------------------------------
# A04 图像编辑、尺寸、色彩与像素修复
# -----------------------------------------------------------------------------

## 图像缩放到总像素级别
gcs https://github.com/BigStationW/ComfyUi-Scale-Image-to-Total-Pixels-Advanced.git

gcs https://github.com/o-l-l-i/ComfyUI-Olm-DragCrop.git

# 图片颜色参考 channel_wise强度设置为1,接近于原图
gcs https://github.com/DenRakEiw/Latent_Nodes.git

# image resolution
gcs https://github.com/gseth/ControlAltAI-Nodes.git

# 像素修复
gcs https://github.com/Mozer/ComfyUI-PixelDriftFix

# -----------------------------------------------------------------------------
# A05 遮罩、人物比例与人像修饰
# -----------------------------------------------------------------------------

## 图像掩码
gcs https://github.com/BadCafeCode/masquerade-nodes-comfyui.git

gcs https://github.com/grmchn/ComfyUI-ProportionChanger.git

gcs https://github.com/wuwukaka/ComfyUI-BodyRatioMapper.git

# sunxAI facetools
gcs https://github.com/upseem/comfyui_sunxAI_facetools.git

# 人物皮肤美颜
gcs https://github.com/chen-zhang-ai/ComfyUI-SkinBeauty-CN.git

# 自动打马赛克
gcs https://github.com/sugarkwork/comfyui-auto-mosaic.git

#gcs https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git

# -----------------------------------------------------------------------------
# A06 通用采样与条件控制
# -----------------------------------------------------------------------------

# klein enhancer
## 扩散模型的通用负面引导
gcs https://github.com/BigStationW/ComfyUI-NAG.git

# -----------------------------------------------------------------------------
# A07 通用视频处理与参考素材
# -----------------------------------------------------------------------------

# The base image normally includes VHS; download it only when missing.
if [ ! -d ComfyUI-VideoHelperSuite ]; then
    gcs https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    git -C ComfyUI-VideoHelperSuite fetch --depth=1 origin 4d907bee61e92c2e65af3bd6383a4e4d356126d1
    git -C ComfyUI-VideoHelperSuite checkout --detach 4d907bee61e92c2e65af3bd6383a4e4d356126d1
fi

# yedp action director
## https://www.linkedin.com/posts/sedpid_hello-everyone-i-just-pushed-a-big-update-activity-7432741280683642882-3eio
gcs https://github.com/yedp123/ComfyUI-Yedp-Action-Director.git

# Reference video
gcs https://github.com/alisson-anjos/ComfyUI-BFSNodes.git

# 视频水印
gcs https://github.com/Artificial-Sweetener/comfyui-WhiteRabbit.git

# -----------------------------------------------------------------------------
# A08 3D 可视化、高斯、网格与动作工具
# -----------------------------------------------------------------------------

# gaussian blur
gcs https://github.com/HallettVisual/comfyui-GaussianViewer.git

gcs https://github.com/PozzettiAndrea/ComfyUI-GaussianPack.git

gcs https://github.com/supart/comfyui_gaussian_splat.git

# mesh2motion
gcs https://github.com/jtydhr88/ComfyUI-mesh2motion.git

# 3d pixal(可以用virsual bruno的?),已移除
#gcs https://github.com/Saganaki22/Pixal3D-ComfyUI.git

# 3d节点
# https://www.reddit.com/r/comfyui/comments/1v868dt/nkd_vfx_tools_just_got_an_update_based_on_all/
#gcs https://github.com/Nekodificador/ComfyUI-NKD-VFX-Tools.git

# -----------------------------------------------------------------------------
# A09 音频处理与音效
# -----------------------------------------------------------------------------

# audio separation nodes
gcs https://github.com/christian-byrne/audio-separation-nodes-comfyui.git

# 音效, 仓库已经移除
#gcs https://github.com/Saganaki22/ComfyUI-Woosh.git

# -----------------------------------------------------------------------------
# A10 显存与内存管理
# -----------------------------------------------------------------------------

# vram/mem cleanup
gcs https://github.com/LAOGOU-666/Comfyui-Memory_Cleanup.git

gcs https://github.com/Windecay/ComfyUI-ReservedVRAM.git

# =============================================================================
# B 模型专用节点与配套工具
# =============================================================================

# -----------------------------------------------------------------------------
# B01 MiniMax H3：生成、编辑、提示词与加速
# -----------------------------------------------------------------------------

# pixaroma minimax h3
gcs https://github.com/pixaroma/ComfyUI-Pixaroma.git

# minimax脸部修复 https://www.reddit.com/r/StableDiffusion/comments/1vn4uho/ok_this_is_cool_minimax_h3_face_detailer/
gcs https://github.com/Carasibana/ComfyUI-H3-FaceRefine.git

# minimax h3 R2F 增强
gcs https://github.com/scottmudge/ComfyUI_MinimaxH3HybridLoader.git

# minimax 3d镜头控制
gcs https://github.com/NyckM/3d-Camera-control-H3-Minimax.git

# minimax h3 提示词生成
gcs https://github.com/1038lab/ComfyUI-MiniMax-H3-Promptor.git

# minimax h3 提示词生成
gcs https://github.com/ethanfel/ComfyUI-H3-Prompt-IDE.git

# minimax face swap 节点
gcs https://github.com/PGCRT/CRT-Nodes.git

# minimax h3 loopsampler
gcs https://github.com/silveroxides/ComfyUI-UtilsCollection.git

# minimax单图八表情节点
## https://www.reddit.com/r/StableDiffusion/comments/1x1mp6o/minimax_h3_ref_fizgig_nodes_as_blitz_visual_novel/
gcs https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still.git

# minimaxh3
# https://www.reddit.com/r/StableDiffusion/comments/1wtu82e/improved_h3_refmods_better_accuracy_and_low/
# gcs https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder
gcs https://github.com/wangweihong/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder

# minimax 加速方案
gcs https://github.com/zechengtang/Spark-H3.git
gcs https://github.com/veda-sparse/Veda-on-ComfyUI.git

# ref mode
gcs https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod

# -----------------------------------------------------------------------------
# B02 LTX：提示词、参考、外扩与镜头控制
# -----------------------------------------------------------------------------

# ltx2.3 VR 360外扩
gcs https://github.com/Burgstall-labs/ComfyUI-EquirectProjector.git

# ltx prompt relay
gcs https://github.com/kijai/ComfyUI-PromptRelay.git

# 非常强大的视频处理节点
# 包含Ltx Director(Prompt Relay Encoded增强版), 多图管理等功能
gcs https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git

# ltx multi-subject-reference
gcs https://github.com/liconstudio/ComfyUI-Licon-MSR.git

# LTX CrossView-Warp IC-LoRA - 更精确地改变现有视频的摄像机角度和轨道路径
# https://www.reddit.com/r/StableDiffusion/comments/1v89kih/ltx_crossviewwarp_iclora_change_the_camera_angle/
gcs https://github.com/cseti007/ComfyUI-CrossViewWarp.git

# ltx 视频深度图
gcs https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git

# -----------------------------------------------------------------------------
# B03 Wan：视频生成与动画预处理
# -----------------------------------------------------------------------------

# wan animate
gcs https://github.com/kijai/ComfyUI-WanVideoWrapper.git
gcs https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git

# -----------------------------------------------------------------------------
# B04 LTX / Wan 共用：VACE 预处理
# -----------------------------------------------------------------------------

# ltx/wan vace prep (视频外展)
gcs https://github.com/stuttlepress/ComfyUI-Wan-VACE-Prep.git

# -----------------------------------------------------------------------------
# B05 Krea 2：生成、编辑、提示词与画质工具
# -----------------------------------------------------------------------------

# krea2去除安全审查 https://www.reddit.com/r/StableDiffusion/comments/1udhaio/this_custom_node_removes_the_builtin_krea_2/
gcs https://github.com/nova452/ComfyUI-ConditioningKrea2Rebalance.git

# krea2 controlnet depth
# https://www.reddit.com/r/StableDiffusion/comments/1ummd2h/krea_2_depth_controlnet_lora/
gcs https://github.com/facok/comfyui-krea2-controlnet.git

# 多人物lora控制,krea2可以同时使用在一次生图使用多个lora
gcsBranchRename https://github.com/yaoliliu/FreeFuse.git comfyui freefuse_comfyui

# krea2 ostris edit, 风格参考
gcs https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit.git

# krea2 人物编辑
gcs https://github.com/lbouaraba/comfyui-krea2edit.git

# krea2 prompt styler, 指定krea2生成各种风格的提示词
# https://www.reddit.com/r/StableDiffusion/comments/1v4u26q/krea_2_styles_wildcards_update/
gcs https://github.com/Gothdir/ComfyUI-Krea2-PromptStyler

# krea2 图片颜色参考
gcs https://github.com/SparknightLLC/ComfyUI-ImageAutotone.git

# krea2像素修复
# https://www.reddit.com/r/StableDiffusion/comments/1wi0mpd/krea_2_pixel_art_loras_32x32_64x64_128x128/
gcs https://github.com/envy-ai/ComfyUI-Krea2-Pixel-Art-Refiner

# -----------------------------------------------------------------------------
# B06 FLUX.2 Klein：编辑、条件控制与采样
# -----------------------------------------------------------------------------

# flux2klein
gcs https://github.com/princepainter/Comfyui-PainterFluxImageEdit.git

# flux2klein enhancer
## https://www.reddit.com/r/StableDiffusion/comments/1somo2r/coming_up_tomorrow_flux2klein_identity_transfer/
# 作者已经移除仓库，迁移到ToolBox
#gcs https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer.git

# klein edit composite
## https://www.reddit.com/r/StableDiffusion/comments/1rstals/how_do_you_handle_klein_edits_colour_drift/
## 可以在某些程度上解决klein编辑的色彩漂移问题。见klein人物换装-v3
gcs https://github.com/supermansundies/comfyui-klein-edit-composite.git
gcs https://github.com/BigStationW/ComfyUi-TextEncodeEditAdvanced.git

# f2k 合一节点 https://www.youtube.com/watch?v=L4ItbBWXqCo
gcs https://github.com/yanokusnir-ai/one-node-flux-2-klein.git

# klein控制光角度
# https://www.reddit.com/r/StableDiffusion/comments/1ulomm5/precise_control_of_the_sun_direction_with_this/
gcs https://github.com/eric-venti-seeds/Sphere-Light-Render-ComfyUI.git

# klein 人物眼睛方向控制
# https://www.reddit.com/r/StableDiffusion/comments/1wbfptw/precise_control_of_the_eyes_direction_with_this/
gcs https://github.com/eric-venti-seeds/Eyes_Direction_Lora_Control.git

# klein 转3d
gcs https://github.com/BlackSnowSkill/ComfyUI-BSS_FLSampler.git

# -----------------------------------------------------------------------------
# B07 Qwen Image：编辑、多视角与对齐
# -----------------------------------------------------------------------------

# qwenmultiangle
gcs https://github.com/jtydhr88/ComfyUI-qwenmultiangle.git

gcs https://github.com/lrzjason/Comfyui-QwenEditUtils.git

# qwen image 2.1对齐
gcs https://github.com/zuanfilm/Qalign

# -----------------------------------------------------------------------------
# B08 Z-Image / Z-Image Turbo：种子与风格参考
# -----------------------------------------------------------------------------

# zit seed variance enhancer
gcs https://github.com/ChangeTheConstants/SeedVarianceEnhancer.git

# zit 图像参考
gcs https://github.com/WuMIn259/ComfyUI-ZTurbo-Style-Transfer.git

# -----------------------------------------------------------------------------
# B09 Anima / PixAI：增强与反推配套
# -----------------------------------------------------------------------------

# anima
gcs https://github.com/AdamNizol/ComfyUI-Anima-Enhancer.git

# anima pixAI反推（速度块)
gcs https://github.com/adbrasi/booru-helper-mini.git

# -----------------------------------------------------------------------------
# B10 Ideogram 4：提示词构建与导演工具
# -----------------------------------------------------------------------------

# ideogram4 prompt builder实时
gcs https://github.com/Work-Fisher/ComfyUI-Ideogram4-DirectJSON-Modified.git

# ideogram4 director
gcs https://github.com/Deno2026/comfyui-deno-custom-nodes.git

# -----------------------------------------------------------------------------
# B11 Qwen-VL：多模态理解与提示词生成
# -----------------------------------------------------------------------------

# Qwen3-VL-Instruct
gcs https://github.com/wangweihong/ComfyUI_Qwen3-VL-Instruct.git

# Qwen3.5
## 依赖transformers > 5.0
gcs https://github.com/WingeD123/ComfyUI_QwenVL_PromptCaption.git

# -----------------------------------------------------------------------------
# B12 Gemma 4：图像反推配套
# -----------------------------------------------------------------------------

# gemma4反推
gcs https://github.com/tl2012tl/comfyUI-llama-TE.git

# -----------------------------------------------------------------------------
# B13 Bernini：视频生成与编辑
# -----------------------------------------------------------------------------

# bernini studio 视频生成编辑
gcs https://github.com/CCpt5/ComfyUI-BerniniStudio.git

# -----------------------------------------------------------------------------
# B14 超分模型与引擎：FlashVSR / SeedVR2 / NVIDIA RTX
# -----------------------------------------------------------------------------

# FlashVSR+ Turbo (Triton); flashvsr-attn is installed from Release in Dockerfile.
if [ ! -d ComfyUI_FlashVSR ]; then
    gcs https://github.com/smthemex/ComfyUI_FlashVSR.git
fi
git -C ComfyUI_FlashVSR fetch --depth=1 origin 153cf03343a116c85aee40d291125921cc33935a
git -C ComfyUI_FlashVSR checkout --detach 153cf03343a116c85aee40d291125921cc33935a

# seedvr2 8k
gcs https://github.com/TTPlanetPig/Comfyui_TTP_Toolset.git

# nvidia专用放大
gcs https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.git

# -----------------------------------------------------------------------------
# B15 3D 模型：Trellis.2 / SHARP / LiTo
# -----------------------------------------------------------------------------

# 3d trellies.2
gcs https://github.com/visualbruno/ComfyUI-Trellis2.git

gcs https://github.com/PozzettiAndrea/ComfyUI-Sharp.git

# lita 3d
gcs https://github.com/PozzettiAndrea/ComfyUI-LiTo.git

# -----------------------------------------------------------------------------
# B16 视觉分析模型：SAM3 / SDPose / Sapiens2 / YOLO
# -----------------------------------------------------------------------------

# sam3
gcs https://github.com/yolain/ComfyUI-Easy-Sam3.git

# 3d pose ood (效果比dwPose好)
gcs https://github.com/judian17/ComfyUI-SDPose-OOD.git

# Sapiens2
## 一系列高分辨率 Transformer 模型，在 10 亿张人体图像上进行预训练，在各种以人为中心的任务（姿态估计、身体部位分割、表面法线和点图）中取得了最先进的性能。
gcs https://github.com/kijai/ComfyUI-Sapiens2.git

gcs https://github.com/judian17/ComfyUI_YOLO_For_Multi_SDPose_Detection.git

# -----------------------------------------------------------------------------
# B17 语音模型与 TTS 服务：Qwen / IndexTTS2 / Edge / Breeze / Whisper / Fish
# -----------------------------------------------------------------------------

# qwen3 tts
gcs https://github.com/flybirdxx/ComfyUI-Qwen-TTS.git

# whisper tts
gcs https://github.com/1038lab/ComfyUI-EdgeTTS.git

# index tts 2
gcs https://github.com/yolain/ComfyUI-Easy-IndexTTS2.git

# breeze
gcs https://github.com/Saganaki22/ComfyUI-Breeze-TTS-2

# audio (不支持cu130)
#gcs https://github.com/wangweihong/ComfyUI-faster-whisper.git

# fishs2
#gcs https://github.com/Saganaki22/ComfyUI-FishAudioS2.git

echo "[INFO] Additional custom nodes downloaded successfully"
