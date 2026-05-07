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

    widget.value = { hidden: false, paused: false, params: {} };
    widget.parentEl = document.createElement("div");
    widget.parentEl.className = "nf_video_preview_wrap";
    widget.parentEl.style.width = "100%";
    widget.parentEl.style.background = "rgba(12,16,24,0.65)";
    widget.parentEl.style.borderRadius = "10px";
    widget.parentEl.style.overflow = "hidden";
    widget.parentEl.style.border = "1px solid rgba(110,150,255,0.28)";
    widget.parentEl.style.boxShadow = "0 0 0 1px rgba(110,150,255,0.05) inset";
    widget.parentEl.style.transition = "all 120ms ease, filter 120ms ease";
    element.appendChild(widget.parentEl);

    widget.videoEl = document.createElement("video");
    widget.videoEl.controls = false;
    widget.videoEl.loop = true;
    widget.videoEl.muted = true;
    widget.videoEl.autoplay = true;
    widget.videoEl.playsInline = true;
    widget.videoEl.style.width = "100%";
    widget.videoEl.style.display = "block";
    widget.videoEl.style.transition = "filter 120ms ease";

    widget.imgEl = document.createElement("img");
    widget.imgEl.hidden = true;
    widget.imgEl.style.width = "100%";
    widget.imgEl.style.display = "block";
    widget.imgEl.style.transition = "filter 120ms ease";

    widget.videoEl.addEventListener("loadedmetadata", () => {
        widget.aspectRatio = widget.videoEl.videoWidth / Math.max(widget.videoEl.videoHeight, 1);
        fitHeight(node);
    });
    widget.imgEl.addEventListener("load", () => {
        widget.aspectRatio = widget.imgEl.naturalWidth / Math.max(widget.imgEl.naturalHeight, 1);
        fitHeight(node);
    });
    widget.videoEl.addEventListener("error", () => {
        widget.parentEl.hidden = true;
        fitHeight(node);
    });

    widget.parentEl.appendChild(widget.videoEl);
    widget.parentEl.appendChild(widget.imgEl);

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
        const url = api.apiURL("/view?" + new URLSearchParams(params));
        this.parentEl.hidden = false;
        this.videoEl.src = url;
        this.videoEl.hidden = false;
        this.imgEl.hidden = true;
        node.applyPreviewStyle?.();
        fitHeight(node);
    };

    node.updateParameters = (params, force = false) => {
        widget.value = widget.value || { hidden: false, paused: false, params: {} };
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
                const filter = style.filter || "none";
                wrap.style.background = style.background || "rgba(12,16,24,0.65)";
                wrap.style.border = style.border || "1px solid rgba(110,150,255,0.28)";
                wrap.style.boxShadow = style.box_shadow || "0 0 0 1px rgba(110,150,255,0.05) inset";
                previewWidget.videoEl.style.filter = filter;
                previewWidget.imgEl.style.filter = filter;
            };

            const formatWidget = this.widgets?.find((w) => w.name === "format");
            if (formatWidget) {
                chainCallback(formatWidget, "callback", function () {
                    requestAnimationFrame(() => {
                        previewWidget.parentEl.hidden = !previewWidget.value?.params?.filename;
                        previewWidget.parentEl.style.opacity = "1";
                        previewWidget.parentEl.style.transform = "translateZ(0)";
                        previewWidget.parentEl.style.willChange = "filter, border, box-shadow";
                        previewWidget.parentEl.style.transition = "all 120ms ease, filter 120ms ease";
                        previewWidget.videoEl.style.transition = "filter 120ms ease";
                        previewWidget.imgEl.style.transition = "filter 120ms ease";
                        this.parent?.applyPreviewStyle?.();
                    });
                });
                formatWidget.parent = this;
            }

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
