################################################################################
# Dockerfile that builds 'yanwk/comfyui-boot:cu130-megapak-pt211'
# A runtime environment for https://github.com/Comfy-Org/ComfyUI
# Using 'root' user inside the container (easy for rootless deploy).
#
# Image hierarchy:
# 1. base-cu130-devel: 
#   - glibc 2.39 (from Rocky Linux 10)
#   - GCC 14 (from Rocky Linux 10)
#   - CUDA Toolkit devel 13.0.2
# 2. base-cu130-pt211:
#   - Python 3.13 (with GIL)
#   - PyTorch 2.11
# 3. base-cu130-pt211-cache
#   - Python dependencies for ComfyUI and Custom Nodes
#       - Without version-sensitive packages like comfyui-frontend
#   - Preloaded ComfyUI bundle
################################################################################

ARG REGISTRY=registry.cn-hangzhou.aliyuncs.com
ARG REPO_NAME=eazycloud
FROM ${REGISTRY}/${REPO_NAME}/comfyui-boot:base-cu130-pt211-cache

LABEL maintainer="wangweihong <wangweihong2018@qq.com>"

ARG USING_GITHUB_ACTIONS=false
#ARG PIP_INDEX_URL="https://mirrors.cernet.edu.cn/pypi/web/simple"
ARG PIP_ROOT_USER_ACTION='ignore'
ARG PIP_NO_BUILD_ISOLATION=1

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

################################################################################
# Update the ComfyUI bundle in the image

WORKDIR /default-comfyui-bundle

COPY builder-scripts/.  /builder-scripts/

RUN bash /builder-scripts/update-cache.sh

# Install deps (comfyui-frontend-package, etc) pair to the preloaded ComfyUI version
# Install ComfyUI-Manager (Python package)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        -c /builder-scripts/constraints.txt \
        -r '/default-comfyui-bundle/ComfyUI/requirements.txt' \
        -r '/default-comfyui-bundle/ComfyUI/manager_requirements.txt' \
    && pip list

# Disable ComfyUI-Manager's cache update on startup ("FETCH ComfyRegistry Data")
RUN cd /usr/local/lib/python3.13/site-packages/comfyui_manager/legacy/ \
    && echo "Deleting the line in $(pwd)/manager_server.py:" \
    && grep -n "run(default_cache_update())" ./manager_server.py \
    && sed -i '/run(default_cache_update())/d' ./manager_server.py

################################################################################
# add additional custom nodes and its dependencies

# add additional-pak.txt to constraints.txt
# Generate a version constraints file to pin PyTorch (prevent later installs from changing its version)
RUN pip list --format=freeze \
        | awk -F== 'BEGIN{want["torch"]=1;want["torchvision"]=1;want["torchaudio"]=1} $1 in want{print $1 "==" $2}' \
        > /builder-scripts/constraints.txt \
    && echo "Pinned PyTorch version:" \
    && cat /builder-scripts/constraints.txt
    
# Additional custom nodes and dependencies
RUN bash /builder-scripts/generate-nodes.sh
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        -c /builder-scripts/constraints.txt \
        -r /builder-scripts/additional-pak.txt
RUN bash /builder-scripts/preload-nodes.sh

# 3d trellis2 wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    # 修复了 o_voxel 漏掉 pip install 的问题
    # 使用 ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} 实现动态代理切换
    pip install ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} \
        https://github.com/wangweihong/Wheels-Containerfiles/releases/download/20260413/cumesh-1.0-cp313-cp313-linux_x86_64.whl \
    && pip install ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} \
        https://github.com/wangweihong/Wheels-Containerfiles/releases/download/20260413/flex_gemm-1.0.0-cp313-cp313-linux_x86_64.whl \
    && pip install ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} \
        https://github.com/wangweihong/Wheels-Containerfiles/releases/download/20260413/o_voxel-0.0.1-cp313-cp313-linux_x86_64.whl \
    && pip install ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} \
        https://github.com/wangweihong/Wheels-Containerfiles/releases/download/20260413/nvdiffrast-0.4.0-cp313-cp313-linux_x86_64.whl \
    && pip install ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} \
        https://github.com/wangweihong/Wheels-Containerfiles/releases/download/20260413/nvdiffrec_render-0.0.0-cp313-cp313-linux_x86_64.whl 


################################################################################

RUN df -h \
    && du -ah /root \
    && fd -H . /root/ --min-depth 1 -x rm -rfv

COPY runner-scripts/.  /runner-scripts/

USER root
WORKDIR /root
EXPOSE 8188
ENV CLI_ARGS=""
CMD ["bash","/runner-scripts/entrypoint.sh"]
