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
ARG INSTALL_3DGS=false
FROM ${REGISTRY}/${REPO_NAME}/comfyui-boot:base-cu130-pt211-cache AS common

LABEL maintainer="wangweihong <wangweihong2018@qq.com>"

ARG USING_GITHUB_ACTIONS=false
#ARG PIP_INDEX_URL="https://mirrors.cernet.edu.cn/pypi/web/simple"
ARG PIP_ROOT_USER_ACTION='ignore'
ARG PIP_NO_BUILD_ISOLATION=1

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# [新增] 声明是否开启离线拷贝模式，默认不开启
ARG OFFLINE_MODE=false
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


# # copy presets to default-comfyui-bundle/
# COPY _presets/assets/. /default-comfyui-bundle/assets/ 
# COPY _presets/subgraphs/. /default-comfyui-bundle/subgraphs/
# COPY _presets/workflows/. /default-comfyui-bundle/workflows/

################################################################################
# add additional custom nodes and its dependencies

# 3d trellis2 wheels
RUN --mount=type=cache,target=/root/.cache/pip \
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

# index tts 2 wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    # 使用 ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} 实现动态代理切换
    pip install ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} \
        https://github.com/wangweihong/Wheels-Containerfiles/releases/download/20260419/descript_audiotools-0.7.4-py2.py3-none-any.whl

# SD POSE
RUN --mount=type=cache,target=/root/.cache/pip \
    # 使用 ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} 实现动态代理切换
    pip install ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} \
        https://github.com/wangweihong/Wheels-Containerfiles/releases/download/20260420/mmcv-2.1.0-py2.py3-none-any.whl


# lito 3d wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    # 使用 ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} 实现动态代理切换
    pip install ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} \
        https://github.com/wangweihong/Wheels-Containerfiles/releases/download/20260523/pytorch3d-0.7.9-cp313-cp313-linux_x86_64.whl

# fix sageattention head dim 256 bug in ideogram4 
RUN --mount=type=cache,target=/root/.cache/pip \
    # 使用 ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} 实现动态代理切换
    pip uninstall -y sageattention \
    && pip install ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} \
        https://github.com/wangweihong/Wheels-Containerfiles/releases/download/sageattention/sageattention-2.2.0+git.5bcf8c3.cu130.pt211-cp313-cp313-linux_x86_64.whl

# 使用新版llama-cpp-python wheels才能支持gemm4
RUN --mount=type=cache,target=/root/.cache/pip \
    # 使用 ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} 实现动态代理切换
    pip uninstall -y llama-cpp-python \
    && pip install ${HTTPS_PROXY:+--proxy $HTTPS_PROXY} \
        https://github.com/wangweihong/Wheels-Containerfiles/releases/download/llama-cpp-python/llama_cpp_python-0.3.40+git.6f21d4d.cu130.pt211-cp313-cp313-linux_x86_64.whl
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
        -r /builder-scripts/additional-pak.txt \
        -r /builder-scripts/pak1.txt
#RUN bash /builder-scripts/preload-nodes.sh

COPY ./offline-cache/repos/. /tmp/local-preload-nodes/

# 3. 用一条 RUN 指令配合 Shell 的 if 逻辑处理分流
RUN if [ "$OFFLINE_MODE" = "true" ] ; then \
        echo "====================================================" && \
        echo "检测到 OFFLINE_MODE=true，正在跳过网络下载，改由本地离线包拷贝..." && \
        echo "====================================================" && \
        # 假设脚本下载的目标路径是 /default-comfyui-bundle/ComfyUI/custom_nodes/
        # 将刚刚 COPY 进来的临时文件移动到最终节点目录
        cp -r /tmp/local-preload-nodes/* /default-comfyui-bundle/ComfyUI/custom_nodes/ ; \
    else \
        echo "====================================================" && \
        echo "检测到 OFFLINE_MODE=false，正在通过网络下载节点..." && \
        echo "====================================================" && \
        bash /builder-scripts/preload-nodes.sh ; \
    fi && \
    # 无论走哪条分支，最后都要删掉临时目录，确保不会有重复和垃圾文件留在镜像中
    rm -rf /tmp/local-preload-nodes

# fix Qwen3-tts /bin/sh: line 1: sox: command not found
#RUN dnf install -y sox
## fix seedvr2 video upscaler报错
RUN dnf install -y openssl


RUN ( [ -z "$HTTPS_PROXY" ] || export https_proxy="$HTTPS_PROXY" ) \
    && mkdir -p /default-comfyui-bundle \
    && wget -O /default-comfyui-bundle/prompts.sqlite3 \
        https://github.com/wangweihong/ComfyUI-ToolBox/releases/download/0.0.1/prompts.sqlite3

# fix以下报错
## Warning:
## Unable to load the following plugins:
## 	libfilter_ao.so: libfilter_ao.so does not seem to be a Qt Plugin.
## Cannot load library /usr/local/lib/python3.13/site-packages/pymeshlab/lib/plugins/libfilter_ao.so: (libOpenGL.so.0: cannot open shared object file: No such file or directory)
## 	libfilter_meshing.so: libfilter_meshing.so does not seem to be a Qt Plugin.
RUN dnf install -y mesa-libGL mesa-libGL-devel

################################################################################
# Remove unused node
# This node will cause ltx error when running,report "RuntimeError: The size of tensor a (28160) must match the size of tensor b (3619968) at non-singleton dimension 2"
RUN rm -rf /default-comfyui-bundle/ComfyUI/custom_nodes/ComfyUI_smZNodes
RUN rm -rf /default-comfyui-bundle/ComfyUI/custom_nodes/VideoX-Fun

# https://github.com/kijai/ComfyUI-DepthAnythingV2/issues/27
# 解决更好的高斯流 DeepAnythingV2报错的问题
RUN pip3 uninstall xformers -y
###############################################################################
# offline nodes
COPY _nodes/. /default-comfyui-bundle/ComfyUI/custom_nodes/
################################################################################
RUN df -h \
    && du -ah /root \
    && fd -H . /root/ --min-depth 1 -x rm -rfv

COPY default/comfy.settings.json /default-comfyui-bundle/ComfyUI/user/default/comfy.settings.json
COPY runner-scripts/.  /runner-scripts/

USER root
WORKDIR /root
EXPOSE 8188
ENV CLI_ARGS=""
CMD ["bash","/runner-scripts/entrypoint.sh"]

# ToolBox owns nodes, examples and docs in both dependency variants.
RUN set -eu; \
    toolbox=/default-comfyui-bundle/ComfyUI/custom_nodes/ComfyUI-ToolBox; \
    workflows=/default-comfyui-bundle/ComfyUI/user/default/workflows/006视频生成/minimaxh3/单图转3DGS_COLMAP_Brush; \
    test -f "$toolbox/nodes/3dgs/sfm_worker.py"; \
    mkdir -p "$workflows/docs" /default-comfyui-bundle/ComfyUI/input; \
    cp -a "$toolbox/examples/3dgs/workflows/." "$workflows/"; \
    cp -a "$toolbox/docs/3dgs/." "$workflows/docs/"; \
    cp -a "$toolbox/examples/3dgs/assets/." /default-comfyui-bundle/ComfyUI/input/

# BuildKit only evaluates the selected branch. The default image never downloads
# or installs optional native dependencies. ToolBox nodes/resources remain present.
FROM common AS feature-3dgs-false

FROM common AS feature-3dgs-true
ARG OFFLINE_MODE=false
ARG HTTP_PROXY
ARG HTTPS_PROXY
RUN --mount=type=bind,source=3dgs.sha256,target=/3dgs.sha256,ro \
    --mount=type=bind,source=offline-cache/3dgs,target=/feature-artifacts,ro \
    --mount=type=cache,target=/root/.cache/pip <<'SH'
set -eu
python3.13 - <<'PY'
import platform, sys
assert platform.system() == 'Linux' and platform.machine() in ('x86_64', 'amd64')
assert sys.version_info[:2] == (3, 13), '3DGS requires Linux amd64 and Python 3.13'
PY
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
release=https://github.com/wangweihong/Wheels-Containerfiles/releases/download/3dgs-v1
wheel=pycolmap-4.2.1-cp313-cp313-manylinux_2_28_x86_64.whl
brush=brush-app-x86_64-unknown-linux-gnu.tar.xz
loader=vulkan-loader-1.4.328.1-1.el10.x86_64.rpm
case "$OFFLINE_MODE" in
    true)
        for file in "$wheel" "$brush" "$loader"; do
            if [ ! -f "/feature-artifacts/$file" ]; then
                echo "Missing offline 3DGS artifact: $file" >&2; exit 1
            fi
            cp "/feature-artifacts/$file" "$work/$file"
        done
        ;;
    false)
        export https_proxy="${HTTPS_PROXY:-${https_proxy:-}}"
        export http_proxy="${HTTP_PROXY:-${http_proxy:-}}"
        for file in "$wheel" "$brush" "$loader"; do
            wget --no-verbose --https-only --tries=3 --timeout=120 -O "$work/$file" "$release/$file"
        done
        ;;
    *) echo 'OFFLINE_MODE must be true or false for 3DGS dependencies' >&2; exit 1 ;;
esac
cd "$work"
sha256sum -c /3dgs.sha256
dnf -y --disablerepo='*' --setopt=install_weak_deps=False install "$work/$loader"
dnf clean all
PIP_USER=false python3.13 -m pip install --no-index --no-deps "$work/$wheel"
python3.13 - <<'PY'
import tarfile
with tarfile.open('brush-app-x86_64-unknown-linux-gnu.tar.xz') as archive:
    members = [archive.getmember('brush-app-x86_64-unknown-linux-gnu/' + name)
               for name in ('brush_app', 'LICENSE')]
    assert all(member.isfile() for member in members), 'Invalid Brush archive members'
    archive.extractall(members=members, filter='data')
PY
install -D -m 755 brush-app-x86_64-unknown-linux-gnu/brush_app /opt/3dgs/bin/brush_app
install -D -m 644 brush-app-x86_64-unknown-linux-gnu/LICENSE /opt/3dgs/licenses/Brush-LICENSE
test "$(/opt/3dgs/bin/brush_app --version)" = 'brush-cli 0.3.0'
/opt/3dgs/bin/brush_app --help >/dev/null
linked=$(ldd /opt/3dgs/bin/brush_app)
case "$linked" in *'not found'*) echo "$linked" >&2; exit 1 ;; esac
cp /3dgs.sha256 /opt/3dgs/SHA256SUMS
python3.13 - <<'PY'
import ctypes, json
from pathlib import Path
import pycolmap, numpy, PIL, cv2, torch
assert pycolmap.__version__ == '4.2.1' and not pycolmap.has_cuda
assert callable(pycolmap.global_mapping) and callable(pycolmap.incremental_mapping)
ctypes.CDLL('libvulkan.so.1')
Path('/opt/3dgs/manifest.json').write_text(json.dumps(dict(
    release='https://github.com/wangweihong/Wheels-Containerfiles/releases/tag/3dgs-v1',
    pycolmap_version=pycolmap.__version__, brush_version='0.3.0',
    base_python_packages={k: m.__version__ for k,m in [('numpy',numpy),('Pillow',PIL),('opencv',cv2),('torch',torch)]},
    driver_libraries_bundled=False, models_bundled=False), indent=2)+'\n')
PY
cd /
SH

FROM feature-3dgs-${INSTALL_3DGS} AS final
ARG INSTALL_3DGS
RUN case "$INSTALL_3DGS" in true|false) ;; \
    *) echo 'INSTALL_3DGS must be true or false' >&2; exit 1 ;; esac
