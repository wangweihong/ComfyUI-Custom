import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EXT_NAME = "nfypnode.NF_VideoPreview.VHSFix";
const NODE_NAME = "NF_VideoPreview";
const DISPLAY_NAME = "南风视频预览";

function isTargetNode(nodeData) {
  return nodeData?.name === NODE_NAME || nodeData?.display_name === DISPLAY_NAME;
}

function chainCallback(object, property, callback) {
  if (!object) return;
  if (property in object && object[property]) {
    const original = object[property];
    object[property] = function () {
      const r = original.apply(this, arguments);
      return callback.apply(this, arguments) ?? r;
    };
  } else {
    object[property] = callback;
  }
}

function fitHeight(node) {
  try {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true, true);
  } catch {
    // ignore
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function getWidget(node, name) {
  return node?.widgets?.find((w) => w?.name === name) || null;
}

function getPreviewStyles(node) {
  const formatWidget = getWidget(node, "format");
  return formatWidget?.options?.nf_preview_styles || {};
}

function getCurrentFormat(node) {
  return String(getWidget(node, "format")?.value || "");
}

function getCurrentCrf(node) {
  const value = Number(getWidget(node, "crf")?.value ?? 19);
  return Number.isFinite(value) ? value : 19;
}

function buildCrfApproxFilter(crf) {
  const normalized = clamp((Number(crf) - 18) / 33, 0, 1);
  const blurPx = normalized * 1.2;
  const saturate = 1 - normalized * 0.18;
  const contrast = 1 - normalized * 0.05;
  const brightness = 1 - normalized * 0.02;
  return `blur(${blurPx.toFixed(2)}px) saturate(${saturate.toFixed(3)}) contrast(${contrast.toFixed(3)}) brightness(${brightness.toFixed(3)})`;
}

function combineFilters(baseFilter, crfFilter) {
  const a = (baseFilter || "").trim();
  const b = (crfFilter || "").trim();
  if (!a || a === "none") return b || "none";
  if (!b || b === "none") return a || "none";
  return `${a} ${b}`.trim();
}

function ensureState(node) {
  if (!node.__nfvp) {
    node.__nfvp = {
      previewWidget: null,
      lastPreview: null,
    };
  }
  return node.__nfvp;
}

function applyLiveStyle(node) {
  const state = ensureState(node);
  const previewWidget = state.previewWidget;
  if (!previewWidget) return;
  const parentEl = previewWidget.parentEl;
  const videoEl = previewWidget.videoEl;
  const imgEl = previewWidget.imgEl;
  const styleMap = getPreviewStyles(node);
  const style = styleMap[getCurrentFormat(node)] || state.lastPreview?.nf_preview_style || {};
  const crf = getCurrentCrf(node);
  const finalFilter = combineFilters(style.filter || "none", buildCrfApproxFilter(crf));

  for (const el of [videoEl, imgEl]) {
    if (!el) continue;
    el.style.width = "100%";
    el.style.display = "block";
    el.style.filter = finalFilter;
    el.style.transition = "filter 140ms ease, box-shadow 140ms ease, border-color 140ms ease, background-color 140ms ease";
    el.style.background = "transparent";
    el.style.borderRadius = "10px";
  }

  if (parentEl) {
    parentEl.style.width = "100%";
    parentEl.style.overflow = "hidden";
    parentEl.style.borderRadius = "12px";
    parentEl.style.background = style.background || "rgba(12,16,24,0.65)";
    parentEl.style.border = style.border || "1px solid rgba(110,150,255,0.28)";
    parentEl.style.boxShadow = style.box_shadow || "0 0 0 1px rgba(110,150,255,0.05) inset";
  }
}

function refreshSource(node, preview) {
  const state = ensureState(node);
  const previewWidget = state.previewWidget;
  if (!previewWidget || !preview) return;

  const params = {
    filename: preview.filename,
    subfolder: preview.subfolder || "",
    type: preview.type || "output",
    format: preview.format,
    nfv: preview.nf_cache_bust || Date.now().toString(),
    timestamp: Date.now(),
  };

  previewWidget.value.params = params;
  previewWidget.value.hidden = false;
  previewWidget.value.paused = false;
  previewWidget.value.muted = false; // 鼠标移入时直接开声，行为贴近 VHS

  if (params.format?.split("/")[0] === "video") {
    previewWidget.videoEl.autoplay = true;
    previewWidget.videoEl.loop = true;
    previewWidget.videoEl.muted = true; // 初始静音，mouseenter 再开声
    previewWidget.videoEl.src = api.apiURL("/view?" + new URLSearchParams(params));
    previewWidget.videoEl.hidden = false;
    previewWidget.imgEl.hidden = true;
    previewWidget.parentEl.hidden = false;
  } else if (params.format?.split("/")[0] === "image") {
    previewWidget.imgEl.src = api.apiURL("/view?" + new URLSearchParams(params));
    previewWidget.videoEl.hidden = true;
    previewWidget.imgEl.hidden = false;
    previewWidget.parentEl.hidden = false;
  }

  applyLiveStyle(node);
  fitHeight(node);
}

function addVideoPreview(nodeType) {
  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    const node = this;
    const state = ensureState(node);

    const element = document.createElement("div");
    const previewWidget = this.addDOMWidget("videopreview", "preview", element, {
      serialize: false,
      hideOnZoom: false,
      getValue() {
        return element.value;
      },
      setValue(v) {
        element.value = v;
      },
    });

    state.previewWidget = previewWidget;

    previewWidget.computeSize = function (width) {
      if (this.aspectRatio && !this.parentEl.hidden) {
        let height = (node.size[0] - 20) / this.aspectRatio + 10;
        if (!(height > 0)) height = 0;
        this.computedHeight = height + 10;
        return [width, height];
      }
      return [width, -4];
    };

    const passthrough = (handlerName) => (e) => {
      e.preventDefault();
      return app.canvas?.[handlerName]?.(e);
    };
    element.addEventListener("contextmenu", passthrough("_mousedown_callback"), true);
    element.addEventListener("pointerdown", passthrough("_mousedown_callback"), true);
    element.addEventListener("mousewheel", passthrough("_mousewheel_callback"), true);
    element.addEventListener("pointermove", passthrough("_mousemove_callback"), true);
    element.addEventListener("pointerup", passthrough("_mouseup_callback"), true);

    previewWidget.value = { hidden: false, paused: false, params: {}, muted: false };
    previewWidget.parentEl = document.createElement("div");
    previewWidget.parentEl.className = "nfvp_preview";
    previewWidget.parentEl.style.width = "100%";
    element.appendChild(previewWidget.parentEl);

    previewWidget.videoEl = document.createElement("video");
    previewWidget.videoEl.controls = false;
    previewWidget.videoEl.loop = true;
    previewWidget.videoEl.muted = true;
    previewWidget.videoEl.playsInline = true;
    previewWidget.videoEl.style.width = "100%";
    previewWidget.videoEl.addEventListener("loadedmetadata", () => {
      previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
      previewWidget.parentEl.hidden = false;
      applyLiveStyle(node);
      fitHeight(node);
      previewWidget.videoEl.play().catch(() => {});
    });
    previewWidget.videoEl.addEventListener("error", () => {
      previewWidget.parentEl.hidden = true;
      fitHeight(node);
    });
    previewWidget.videoEl.onmouseenter = () => {
      if (state.lastPreview?.nf_has_audio) {
        previewWidget.videoEl.muted = previewWidget.value.muted;
      }
    };
    previewWidget.videoEl.onmouseleave = () => {
      previewWidget.videoEl.muted = true;
    };

    previewWidget.imgEl = document.createElement("img");
    previewWidget.imgEl.style.width = "100%";
    previewWidget.imgEl.hidden = true;
    previewWidget.imgEl.onload = () => {
      previewWidget.aspectRatio = previewWidget.imgEl.naturalWidth / previewWidget.imgEl.naturalHeight;
      previewWidget.parentEl.hidden = false;
      applyLiveStyle(node);
      fitHeight(node);
    };

    previewWidget.parentEl.appendChild(previewWidget.videoEl);
    previewWidget.parentEl.appendChild(previewWidget.imgEl);

    const wrapWidgetCallback = (name) => {
      const widget = getWidget(node, name);
      if (!widget || widget.__nfvpWrapped) return;
      widget.__nfvpWrapped = true;
      const original = widget.callback;
      widget.callback = function () {
        const r = original ? original.apply(this, arguments) : undefined;
        applyLiveStyle(node);
        return r;
      };
    };
    wrapWidgetCallback("format");
    wrapWidgetCallback("crf");

    fitHeight(node);
  });

  chainCallback(nodeType.prototype, "onExecuted", function (message) {
    const preview = message?.ui?.gifs?.[0];
    if (!preview) return;
    const state = ensureState(this);
    state.lastPreview = preview;
    refreshSource(this, preview);
  });

  chainCallback(nodeType.prototype, "onConfigure", function () {
    const state = ensureState(this);
    if (state.lastPreview) {
      applyLiveStyle(this);
      fitHeight(this);
    }
  });
}

app.registerExtension({
  name: EXT_NAME,
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!isTargetNode(nodeData)) return;
    addVideoPreview(nodeType);
  },
});
