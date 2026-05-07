import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

function chainCallback(object, property, callback) {
    if (!object) return;
    const original = object[property];
    object[property] = function (...args) {
        let result;
        if (original) {
            result = original.apply(this, args);
        }
        const chained = callback.apply(this, args);
        return chained !== undefined ? chained : result;
    };
}

function fitHeight(node) {
    if (!node?.widgets) return;
    const widget = node.widgets.find((w) => w.name === "videopreview");
    if (!widget) return;
    requestAnimationFrame(() => {
        node.onResize?.(node.size);
        app.graph.setDirtyCanvas(true, true);
    });
}

async function requestRealPreviewRebuild(node) {
    const previewWidget = node.widgets?.find((w) => w.name === "videopreview");
    const formatWidget = node.widgets?.find((w) => w.name === "format");
    const crfWidget = node.widgets?.find((w) => w.name === "crf");
    const params = previewWidget?.value?.params || {};

    if (!previewWidget || !params.nf_preview_token || !params.filename) {
        node.applyPreviewStyle?.();
        return;
    }

    const formatValue = formatWidget?.value;
    const crfValue = Number(crfWidget?.value ?? 19);
    const seq = (node.__nfPreviewRequestSeq = (node.__nfPreviewRequestSeq || 0) + 1);

    previewWidget.hintEl.hidden = false;
    previewWidget.hintEl.textContent = "预览重建中";

    try {
        const resp = await api.fetchApi("/nf_video_preview/rebuild", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                token: params.nf_preview_token,
                format: formatValue,
                crf: crfValue,
            }),
        });

        const data = await resp.json();
        if (!resp.ok || !data?.ok || !data?.preview) {
            throw new Error(data?.error || `HTTP ${resp.status}`);
        }
        if (seq !== node.__nfPreviewRequestSeq) {
            return;
        }
        node.updateParameters(data.preview, true);
    } catch (err) {
        console.error("[NF_VideoPreview] 真实预览重建失败:", err);
        previewWidget.hintEl.hidden = false;
        previewWidget.hintEl.textContent = "预览失败";
        setTimeout(() => {
            if (!previewWidget?.hintEl) return;
            const hasAudio = !!previewWidget.value?.params?.nf_has_audio;
            previewWidget.hintEl.hidden = !hasAudio;
            previewWidget.hintEl.textContent = hasAudio ? "悬停开声" : "";
        }, 1200);
    } finally {
        node.applyPreviewStyle?.();
    }
}

function buildPreviewWidget(node) {
    const element = document.createElement("div");
    const widget = node.addDOMWidget("videopreview", "preview", element, {
        serialize: false,
        hideOnZoom: false,
        getValue() {
            return element.value;
        },
        setValue(v) {
            element.value = v;
        },
    });

    widget.value = { hidden: false, paused: false, params: {}, muted: true };
    widget.parentEl = document.createElement("div");
    widget.parentEl.className = "nf_video_preview_wrap";
    widget.parentEl.style.width = "100%";
    widget.parentEl.style.background = "rgba(12,16,24,0.65)";
    widget.parentEl.style.borderRadius = "10px";
    widget.parentEl.style.overflow = "hidden";
    widget.parentEl.style.border = "1px solid rgba(110,150,255,0.28)";
    widget.parentEl.style.boxShadow = "0 0 0 1px rgba(110,150,255,0.05) inset";
    widget.parentEl.style.transition = "all 120ms ease";
    widget.parentEl.hidden = true;
    element.appendChild(widget.parentEl);

    widget.videoEl = document.createElement("video");
    widget.videoEl.controls = false;
    widget.videoEl.loop = true;
    widget.videoEl.muted = true;
    widget.videoEl.autoplay = true;
    widget.videoEl.playsInline = true;
    widget.videoEl.preload = "auto";
    widget.videoEl.style.width = "100%";
    widget.videoEl.style.display = "block";
    widget.videoEl.style.background = "transparent";
    widget.videoEl.style.filter = "none";

    widget.imgEl = document.createElement("img");
    widget.imgEl.hidden = true;
    widget.imgEl.style.width = "100%";
    widget.imgEl.style.display = "block";
    widget.imgEl.style.background = "transparent";
    widget.imgEl.style.filter = "none";

    widget.hintEl = document.createElement("div");
    widget.hintEl.textContent = "悬停开声";
    widget.hintEl.style.position = "absolute";
    widget.hintEl.style.right = "8px";
    widget.hintEl.style.top = "8px";
    widget.hintEl.style.padding = "3px 8px";
    widget.hintEl.style.borderRadius = "999px";
    widget.hintEl.style.fontSize = "11px";
    widget.hintEl.style.lineHeight = "1";
    widget.hintEl.style.color = "rgba(255,255,255,0.92)";
    widget.hintEl.style.background = "rgba(0,0,0,0.42)";
    widget.hintEl.style.pointerEvents = "none";
    widget.hintEl.style.zIndex = "2";
    widget.hintEl.hidden = true;

    widget.parentEl.style.position = "relative";

    widget.videoEl.addEventListener("loadedmetadata", () => {
        widget.aspectRatio = widget.videoEl.videoWidth / Math.max(widget.videoEl.videoHeight, 1);
        widget.parentEl.hidden = false;
        node.applyPreviewStyle?.();
        fitHeight(node);
        widget.videoEl.play().catch(() => {});
    });

    widget.videoEl.addEventListener("canplay", () => {
        widget.videoEl.play().catch(() => {});
    });

    widget.imgEl.addEventListener("load", () => {
        widget.aspectRatio = widget.imgEl.naturalWidth / Math.max(widget.imgEl.naturalHeight, 1);
        widget.parentEl.hidden = false;
        node.applyPreviewStyle?.();
        fitHeight(node);
    });

    widget.videoEl.addEventListener("error", () => {
        widget.parentEl.hidden = true;
        fitHeight(node);
    });

    const hoverPlayWithAudio = async () => {
        const hasAudio = !!widget.value?.params?.nf_has_audio;
        if (!hasAudio) return;
        widget.videoEl.muted = false;
        widget.value.muted = false;
        try {
            await widget.videoEl.play();
            widget.hintEl.textContent = "有声预览";
        } catch (_err) {
            widget.videoEl.muted = true;
            widget.value.muted = true;
            widget.hintEl.textContent = "点击开声";
        }
    };

    const leaveMute = () => {
        widget.videoEl.muted = true;
        widget.value.muted = true;
        if (widget.value?.params?.nf_has_audio) {
            widget.hintEl.textContent = "悬停开声";
        }
    };

    widget.parentEl.addEventListener("mouseenter", () => {
        if (!widget.videoEl.hidden) {
            hoverPlayWithAudio();
        }
    });

    widget.parentEl.addEventListener("mouseleave", () => {
        if (!widget.videoEl.hidden) {
            leaveMute();
        }
    });

    widget.parentEl.addEventListener("click", async (e) => {
        if (widget.videoEl.hidden) return;
        e.preventDefault();
        const hasAudio = !!widget.value?.params?.nf_has_audio;
        if (!hasAudio) {
            widget.videoEl.play().catch(() => {});
            return;
        }
        widget.videoEl.muted = false;
        widget.value.muted = false;
        try {
            await widget.videoEl.play();
            widget.hintEl.textContent = "有声预览";
        } catch (_err) {
            widget.videoEl.muted = true;
            widget.value.muted = true;
            widget.hintEl.textContent = "点击开声";
        }
    });

    widget.parentEl.appendChild(widget.videoEl);
    widget.parentEl.appendChild(widget.imgEl);
    widget.parentEl.appendChild(widget.hintEl);

    widget.computeSize = function (width) {
        if (this.aspectRatio && !this.parentEl.hidden) {
            const height = Math.max(((node.size[0] - 20) / this.aspectRatio) + 10, 0);
            this.computedHeight = height + 10;
            return [width, height];
        }
        return [width, -4];
    };

    widget.updateSource = function () {
        const params = { ...(this.value?.params || {}) };
        if (!params.filename) {
            this.parentEl.hidden = true;
            fitHeight(node);
            return;
        }
        params.timestamp = Date.now();
        params.nfv = params.nf_cache_bust || params.nfv || String(Date.now());

        const url = api.apiURL("/view?" + new URLSearchParams({
            filename: params.filename,
            subfolder: params.subfolder || "",
            type: params.type || "output",
            nfv: params.nfv,
            timestamp: params.timestamp,
        }));

        this.parentEl.hidden = false;

        if ((params.format || "").startsWith("video/")) {
            this.videoEl.src = url;
            this.videoEl.hidden = false;
            this.imgEl.hidden = true;
            this.videoEl.muted = true;
            this.value.muted = true;
            this.hintEl.hidden = !params.nf_has_audio;
            this.hintEl.textContent = params.nf_has_audio ? "悬停开声" : "";
            this.videoEl.load();
            this.videoEl.play().catch(() => {});
        } else {
            this.imgEl.src = url;
            this.videoEl.pause?.();
            this.videoEl.removeAttribute("src");
            this.videoEl.load?.();
            this.videoEl.hidden = true;
            this.imgEl.hidden = false;
            this.hintEl.hidden = true;
        }

        node.applyPreviewStyle?.();
        fitHeight(node);
    };

    node.updateParameters = (params, force = false) => {
        widget.value = widget.value || { hidden: false, paused: false, params: {}, muted: true };
        widget.value.params = widget.value.params || {};
        Object.assign(widget.value.params, params || {});
        if (force) {
            widget.updateSource();
            return;
        }
        clearTimeout(node.__nfPreviewTimer);
        node.__nfPreviewTimer = setTimeout(() => widget.updateSource(), 80);
    };

    return widget;
}

app.registerExtension({
    name: "nfypnode.video_preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "NF_VideoPreview") return;

        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            const previewWidget = buildPreviewWidget(this);
            const styles = nodeData?.input?.required?.format?.[1]?.nf_preview_styles || {};
            this.__nfPreviewStyles = styles;

            this.applyPreviewStyle = () => {
                const formatWidget = this.widgets?.find((w) => w.name === "format");
                const current = formatWidget?.value;
                const style = (current && this.__nfPreviewStyles?.[current]) || {};
                const wrap = previewWidget.parentEl;

                wrap.style.background = style.background || "rgba(12,16,24,0.65)";
                wrap.style.border = style.border || "1px solid rgba(110,150,255,0.28)";
                wrap.style.boxShadow = style.box_shadow || "0 0 0 1px rgba(110,150,255,0.05) inset";
                previewWidget.videoEl.style.filter = "none";
                previewWidget.imgEl.style.filter = "none";
            };

            this.scheduleRealPreviewRebuild = () => {
                clearTimeout(this.__nfRealPreviewDebounce);
                this.__nfRealPreviewDebounce = setTimeout(() => {
                    requestRealPreviewRebuild(this);
                }, 180);
            };

            const attachLiveCallback = (name) => {
                const targetWidget = this.widgets?.find((w) => w.name === name);
                if (!targetWidget) return;
                chainCallback(targetWidget, "callback", () => {
                    requestAnimationFrame(() => {
                        previewWidget.parentEl.hidden = !previewWidget.value?.params?.filename;
                        this.applyPreviewStyle?.();
                        this.scheduleRealPreviewRebuild?.();
                    });
                });
            };

            attachLiveCallback("format");
            attachLiveCallback("crf");

            this.applyPreviewStyle();
        });

        chainCallback(nodeType.prototype, "onExecuted", function (message) {
            if (message?.gifs?.length) {
                this.updateParameters(message.gifs[0], true);
            }
            this.applyPreviewStyle?.();
        });
    },
});
