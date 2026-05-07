import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_CLASS = "NanFengAudioWaveformEditor";
const DEFAULT_WIDTH = 860;
const DEFAULT_HEIGHT = 470;
const SNAP_STEP_SECONDS = 1.0;
const EDIT_SNAP_STEP_SECONDS = 0.1;
const MANUAL_AUDIO_PREFIX = "__nf_manual_audio__:";
const LINKED_AUDIO_WIDGET_NAMES = {
    LoadAudio: ["audio", "audio_file"],
    RecordAudio: ["audio", "audio_file"],
    LoadAudioUpload: ["audio", "audio_file"],
    VHS_LoadAudio: ["audio_file", "audio"],
    VHS_LoadAudioUpload: ["audio", "audio_file"],
};

let sharedAudioContext = null;
let modalStylesReady = false;

function getWidget(node, name) {
    return (node.widgets || []).find((widget) => widget?.name === name);
}

function getInput(node, name) {
    return (node.inputs || []).find((input) => input?.name === name);
}

function setWidgetValue(node, widgetName, value, invokeCallback = true) {
    const widget = getWidget(node, widgetName);
    if (!widget) return;
    widget.value = value;
    if (invokeCallback) {
        widget.callback?.(value);
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

function unwrapManualAudioFile(value) {
    const text = String(value || "").trim();
    if (text.startsWith(MANUAL_AUDIO_PREFIX)) {
        return {
            audioFile: text.slice(MANUAL_AUDIO_PREFIX.length).trim(),
            isManual: true,
        };
    }
    return {
        audioFile: text,
        isManual: false,
    };
}

function encodeManualAudioFile(audioFile) {
    const text = String(audioFile || "").trim();
    return text ? `${MANUAL_AUDIO_PREFIX}${text}` : "";
}

function getEditedAudioOverrideValue(node) {
    return String(getWidget(node, "edited_audio_file")?.value || "").trim();
}

function isManualAudioOverrideEnabled(node) {
    return Boolean(getEditedAudioOverrideValue(node)) || unwrapManualAudioFile(getWidget(node, "audio_file")?.value).isManual;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function snapTime(seconds, step = SNAP_STEP_SECONDS) {
    const value = Math.max(0, Number(seconds) || 0);
    return Math.round(value / step) * step;
}

function formatTime(seconds) {
    const clamped = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(clamped / 60);
    const remainder = clamped - minutes * 60;
    return `${minutes}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function normalizeSegmentIndexWidget(node) {
    const widget = getWidget(node, "segment_index");
    if (!widget) return;
    const raw = widget.value;
    if (raw === "" || raw === null || raw === undefined || Number.isNaN(Number(raw))) {
        widget.value = 0;
    } else {
        widget.value = Math.max(0, Math.floor(Number(raw)));
    }
}

function resizeNode(node) {
    const computedSize = node.computeSize?.();
    const width = Math.max(DEFAULT_WIDTH, computedSize?.[0] ?? 0, Array.isArray(node.size) ? node.size[0] : 0);
    const height = Math.max(DEFAULT_HEIGHT, computedSize?.[1] ?? 0, Array.isArray(node.size) ? node.size[1] : 0);
    node.size = [width, height];
    app.graph.setDirtyCanvas(true, true);
}

function hideWidget(widget) {
    if (!widget || widget.__nfAudioHidden) return;
    widget.__nfAudioOriginalType = widget.type;
    widget.__nfAudioOriginalComputeSize = widget.computeSize;
    widget.__nfAudioOriginalSerializeValue = widget.serializeValue;
    const element = widget.inputEl || widget.element || widget.el;
    const targets = [element, element?.parentElement, element?.parentElement?.parentElement].filter(Boolean);
    if (targets.length) {
        widget.__nfAudioElements = targets.map((target) => ({ target, cssText: target.style.cssText }));
        for (const { target } of widget.__nfAudioElements) {
            target.style.display = "none";
            target.style.visibility = "hidden";
            target.style.height = "0";
            target.style.minHeight = "0";
            target.style.maxHeight = "0";
            target.style.margin = "0";
            target.style.padding = "0";
            target.style.border = "0";
            target.style.overflow = "hidden";
            target.style.pointerEvents = "none";
        }
    }
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.serializeValue = () => widget.value;
    widget.__nfAudioHidden = true;
}

function parseKeyframes(value) {
    try {
        const parsed = JSON.parse(String(value || "[]"));
        if (!Array.isArray(parsed)) return [];
        return parsed
            .map((item) => Number.parseFloat(item))
            .filter((item) => Number.isFinite(item) && item >= 0)
            .sort((left, right) => left - right);
    } catch {
        return [];
    }
}

function normalizeKeyframes(keyframes, duration) {
    const unique = [];
    const seen = new Set();
    for (const value of keyframes || []) {
        let seconds = snapTime(value, SNAP_STEP_SECONDS);
        if (Number.isFinite(duration) && duration > 0) {
            seconds = Math.min(seconds, Math.max(0, duration - 0.001));
        }
        const bucket = Math.round(seconds / SNAP_STEP_SECONDS);
        if (seen.has(bucket)) continue;
        seen.add(bucket);
        unique.push(seconds);
    }
    unique.sort((left, right) => left - right);
    return unique;
}

function styleButton(button, variant = "default") {
    const styles = {
        default: {
            background: "#2b2f36",
            color: "#f3f4f6",
            border: "1px solid rgba(255,255,255,0.12)",
        },
        accent: {
            background: "linear-gradient(135deg, #ff7a18, #c73a18)",
            color: "#fff7ed",
            border: "1px solid rgba(255,185,120,0.55)",
        },
        primary: {
            background: "linear-gradient(135deg, #ea580c, #b91c1c)",
            color: "#fffaf5",
            border: "1px solid rgba(255,166,123,0.65)",
        },
        subtle: {
            background: "#1f2937",
            color: "#e5e7eb",
            border: "1px solid rgba(255,255,255,0.10)",
        },
    };
    const selected = styles[variant] || styles.default;
    button.style.background = selected.background;
    button.style.color = selected.color;
    button.style.border = selected.border;
    button.style.borderRadius = "9px";
    button.style.padding = "7px 12px";
    button.style.fontSize = "12px";
    button.style.fontWeight = variant === "primary" ? "700" : "600";
    button.style.cursor = "pointer";
    button.style.boxShadow = variant === "primary"
        ? "0 10px 24px rgba(185,28,28,0.18)"
        : "0 4px 12px rgba(0,0,0,0.12)";
}

async function uploadAudioFile(file) {
    const body = new FormData();
    const uploadFile = new File([file], file.name, {
        type: file.type,
        lastModified: file.lastModified,
    });
    body.append("image", uploadFile);

    const response = await api.fetchApi("/upload/image", {
        method: "POST",
        body,
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok || !payload?.name) {
        throw new Error(payload?.error || `Upload failed (HTTP ${response.status})`);
    }
    return payload.name;
}

function normalizePreviewPayload(payload) {
    if (!payload) return null;
    const rawAudioFile = String(payload.audioFile ?? payload.audio_file ?? "").trim();
    const { audioFile, isManual } = unwrapManualAudioFile(rawAudioFile);
    if (!audioFile) return null;
    const audioIdentity = String(payload.audioIdentity ?? payload.audio_identity ?? "").trim();
    let label = String(payload.label || payload.sourceLabel || payload.source || "").trim();
    if (isManual) label = "audio_file";
    if (label.startsWith("linked:")) {
        const sourceType = label.slice("linked:".length).trim();
        label = sourceType ? `上游 ${sourceType}` : "上游音频";
    }
    if (label === "input_audio") label = "运行时音频";
    if (label === "audio_file") label = "手动路径";
    if (label === "widget") label = "手动路径";
    if (label === "node_cached_audio") label = "节点缓存";
    return {
        audioFile,
        audioIdentity,
        label,
        sourceNode: payload.sourceNode || null,
        sourceWidget: payload.sourceWidget || null,
        kind: String(payload.kind || "").trim(),
    };
}

function resolveAudioOrigin(node) {
    const input = getInput(node, "audio");
    const graph = node.graph || app.graph;
    if (!input?.link || !graph?.links) return null;

    let link = graph.links[input.link];
    let originNode = link ? graph.getNodeById?.(link.origin_id) : null;
    const visited = new Set();

    while (originNode && (originNode.type === "Reroute" || originNode.comfyClass === "Reroute")) {
        if (visited.has(originNode.id)) return null;
        visited.add(originNode.id);
        const rerouteInput = originNode.inputs?.[0];
        if (!rerouteInput?.link) return null;
        link = graph.links[rerouteInput.link];
        originNode = link ? graph.getNodeById?.(link.origin_id) : null;
    }

    if (!originNode) return null;
    return { link, originNode };
}

function resolveLinkedAudioSource(node) {
    const origin = resolveAudioOrigin(node);
    const originNode = origin?.originNode;
    if (!originNode) return null;

    const sourceType = originNode.comfyClass || originNode.type || "";
    const widgetNames = LINKED_AUDIO_WIDGET_NAMES[sourceType];
    if (!widgetNames?.length) {
        return null;
    }

    const sourceWidget = widgetNames
        .map((widgetName) => getWidget(originNode, widgetName))
        .find(Boolean);
    const audioFile = String(sourceWidget?.value || "").trim();
    if (!audioFile) return null;

    return normalizePreviewPayload({
        audioFile,
        label: `linked:${sourceType}`,
        sourceNode: originNode,
        sourceWidget,
        kind: "linked",
    });
}

function getPreferredAudioSource(node) {
    const editedOverrideSource = normalizePreviewPayload({
        audioFile: getEditedAudioOverrideValue(node),
        label: "edited_audio_override",
        kind: "edited_override",
    });
    const rawWidgetAudioFile = String(getWidget(node, "audio_file")?.value || "").trim();
    const widgetSource = normalizePreviewPayload({
        audioFile: rawWidgetAudioFile,
        label: "widget",
        kind: "widget",
    });
    const linkedSource = resolveLinkedAudioSource(node);
    const runtimeSource = normalizePreviewPayload(node.__nfAudioRuntimePreview);

    if (editedOverrideSource) return editedOverrideSource;
    if (unwrapManualAudioFile(rawWidgetAudioFile).isManual && widgetSource) return widgetSource;
    if (linkedSource) return linkedSource;
    if (runtimeSource) return runtimeSource;
    return widgetSource;
}

function hookExternalSourceWidget(node, source, reloadCallback) {
    const sourceWidget = source?.sourceWidget;
    if (!sourceWidget) return;
    if (!sourceWidget.__nfAudioTargets) {
        sourceWidget.__nfAudioTargets = new Set();
    }
    const targetKey = String(node.id);
    if (sourceWidget.__nfAudioTargets.has(targetKey)) return;
    const originalCallback = sourceWidget.callback;
    sourceWidget.callback = function () {
        const result = originalCallback?.apply(this, arguments);
        reloadCallback?.();
        return result;
    };
    sourceWidget.__nfAudioTargets.add(targetKey);
}

function getAudioContext() {
    if (!sharedAudioContext) {
        const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
        sharedAudioContext = new AudioContextCtor();
    }
    return sharedAudioContext;
}

function ensureModalStyles() {
    if (modalStylesReady) return;
    const style = document.createElement("style");
    style.textContent = `
        .nf-audio-edit-overlay {
            position: fixed;
            inset: 0;
            background: rgba(5, 8, 12, 0.72);
            backdrop-filter: blur(4px);
            z-index: 99999;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .nf-audio-edit-dialog {
            width: min(1200px, calc(100vw - 40px));
            height: min(760px, calc(100vh - 40px));
            background: linear-gradient(180deg, #131821 0%, #0f141b 100%);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 16px;
            box-shadow: 0 30px 80px rgba(0,0,0,0.40);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            color: #f8fafc;
            font-family: sans-serif;
        }
        .nf-audio-edit-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 18px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.02);
        }
        .nf-audio-edit-title {
            font-size: 15px;
            font-weight: 700;
            color: #fff7ed;
        }
        .nf-audio-edit-body {
            display: flex;
            flex-direction: column;
            gap: 10px;
            padding: 14px 18px 18px;
            min-height: 0;
            flex: 1;
        }
        .nf-audio-edit-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }
        .nf-audio-edit-pill {
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 12px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            color: #e5e7eb;
        }
        .nf-audio-edit-number {
            width: 86px;
            background: rgba(255,255,255,0.06);
            color: #f8fafc;
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 8px;
            padding: 7px 9px;
            outline: none;
        }
        .nf-audio-edit-help {
            font-size: 12px;
            color: rgba(255,255,255,0.72);
        }
        .nf-audio-edit-canvas {
            width: 100%;
            flex: 1;
            min-height: 360px;
            background: #0b1016;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            cursor: crosshair;
        }
        .nf-audio-edit-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }
    `;
    document.head.appendChild(style);
    modalStylesReady = true;
}

function computePeaksFromChannelData(channelData, bins = 2200) {
    if (!Array.isArray(channelData) || !channelData.length) return [];
    const frameCount = channelData[0]?.length || 0;
    if (!frameCount) return [];
    const safeBins = clamp(Math.floor(bins), 120, 4096);
    const peaks = new Array(safeBins).fill(0);
    for (let index = 0; index < safeBins; index += 1) {
        const start = Math.floor(index * frameCount / safeBins);
        const end = Math.floor((index + 1) * frameCount / safeBins);
        let peak = 0;
        for (let frame = start; frame < end; frame += 1) {
            let value = 0;
            for (let channel = 0; channel < channelData.length; channel += 1) {
                value += Math.abs(channelData[channel][frame] || 0);
            }
            value /= Math.max(1, channelData.length);
            if (value > peak) peak = value;
        }
        peaks[index] = peak;
    }
    return peaks;
}

function cloneWorkingData(data) {
    return {
        sampleRate: data.sampleRate,
        numberOfChannels: data.numberOfChannels,
        length: data.length,
        channelData: data.channelData.map((channel) => new Float32Array(channel)),
    };
}

function audioBufferToWorkingData(audioBuffer) {
    const channelData = [];
    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
        channelData.push(new Float32Array(audioBuffer.getChannelData(channel)));
    }
    return {
        sampleRate: audioBuffer.sampleRate,
        numberOfChannels: audioBuffer.numberOfChannels,
        length: audioBuffer.length,
        channelData,
    };
}

function workingDataDuration(data) {
    return data?.sampleRate > 0 ? data.length / data.sampleRate : 0;
}

function createSilentFrames(frameCount) {
    return new Float32Array(Math.max(0, frameCount));
}

function insertSilenceIntoWorkingData(data, insertSeconds, atSeconds) {
    const seconds = Math.max(0, Number(insertSeconds) || 0);
    if (!(seconds > 0)) return cloneWorkingData(data);
    const insertFrames = Math.max(1, Math.round(seconds * data.sampleRate));
    const insertIndex = clamp(Math.round((Number(atSeconds) || 0) * data.sampleRate), 0, data.length);
    const channelData = data.channelData.map((channel) => {
        const next = new Float32Array(data.length + insertFrames);
        next.set(channel.slice(0, insertIndex), 0);
        next.set(createSilentFrames(insertFrames), insertIndex);
        next.set(channel.slice(insertIndex), insertIndex + insertFrames);
        return next;
    });
    return {
        sampleRate: data.sampleRate,
        numberOfChannels: data.numberOfChannels,
        length: data.length + insertFrames,
        channelData,
    };
}

function silenceSelectionInWorkingData(data, startSeconds, endSeconds) {
    const startFrame = clamp(Math.round(startSeconds * data.sampleRate), 0, data.length);
    const endFrame = clamp(Math.round(endSeconds * data.sampleRate), startFrame, data.length);
    if (endFrame <= startFrame) return cloneWorkingData(data);
    const next = cloneWorkingData(data);
    for (let channel = 0; channel < next.channelData.length; channel += 1) {
        next.channelData[channel].fill(0, startFrame, endFrame);
    }
    return next;
}

function deleteSelectionFromWorkingData(data, startSeconds, endSeconds) {
    const startFrame = clamp(Math.round(startSeconds * data.sampleRate), 0, data.length);
    const endFrame = clamp(Math.round(endSeconds * data.sampleRate), startFrame, data.length);
    if (endFrame <= startFrame) return cloneWorkingData(data);
    const removedFrames = endFrame - startFrame;
    const channelData = data.channelData.map((channel) => {
        const next = new Float32Array(data.length - removedFrames);
        next.set(channel.slice(0, startFrame), 0);
        next.set(channel.slice(endFrame), startFrame);
        return next;
    });
    return {
        sampleRate: data.sampleRate,
        numberOfChannels: data.numberOfChannels,
        length: data.length - removedFrames,
        channelData,
    };
}

function buildWavBlobFromWorkingData(data) {
    const numberOfChannels = Math.max(1, data.numberOfChannels || data.channelData.length || 1);
    const sampleRate = Math.max(1, data.sampleRate || 44100);
    const length = Math.max(0, data.length || 0);
    const bytesPerSample = 2;
    const blockAlign = numberOfChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = length * blockAlign;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    const writeString = (offset, text) => {
        for (let index = 0; index < text.length; index += 1) {
            view.setUint8(offset + index, text.charCodeAt(index));
        }
    };

    writeString(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bytesPerSample * 8, true);
    writeString(36, "data");
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let frame = 0; frame < length; frame += 1) {
        for (let channel = 0; channel < numberOfChannels; channel += 1) {
            const sample = clamp(data.channelData[channel]?.[frame] || 0, -1, 1);
            const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
            view.setInt16(offset, Math.round(int16), true);
            offset += 2;
        }
    }

    return new Blob([buffer], { type: "audio/wav" });
}

async function decodeAudioFileToWorkingData(audioFile) {
    const response = await fetch(`/nfypnode/audio-file?audio_file=${encodeURIComponent(audioFile)}`);
    if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.error || `音频读取失败 (${response.status})`);
    }
    const arrayBuffer = await response.arrayBuffer();
    const ctx = getAudioContext();
    const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
    return audioBufferToWorkingData(decoded);
}

function createObjectUrlFromWorkingData(data) {
    return URL.createObjectURL(buildWavBlobFromWorkingData(data));
}

function ensureEditorModal(node, parentState) {
    if (node.__nfAudioEditModal) return node.__nfAudioEditModal;
    ensureModalStyles();

    const overlay = document.createElement("div");
    overlay.className = "nf-audio-edit-overlay";
    overlay.style.display = "none";

    const dialog = document.createElement("div");
    dialog.className = "nf-audio-edit-dialog";
    overlay.appendChild(dialog);

    const header = document.createElement("div");
    header.className = "nf-audio-edit-header";
    const title = document.createElement("div");
    title.className = "nf-audio-edit-title";
    title.textContent = "编辑波形";
    const headerButtons = document.createElement("div");
    headerButtons.className = "nf-audio-edit-row";
    const closeButton = document.createElement("button");
    closeButton.textContent = "关闭";
    styleButton(closeButton, "subtle");
    headerButtons.appendChild(closeButton);
    header.append(title, headerButtons);

    const body = document.createElement("div");
    body.className = "nf-audio-edit-body";

    const row1 = document.createElement("div");
    row1.className = "nf-audio-edit-row";
    const playButton = document.createElement("button");
    playButton.textContent = "播放";
    styleButton(playButton, "default");
    const undoButton = document.createElement("button");
    undoButton.textContent = "撤销";
    styleButton(undoButton, "default");
    const clearSelectionButton = document.createElement("button");
    clearSelectionButton.textContent = "清空选区";
    styleButton(clearSelectionButton, "default");
    const timePill = document.createElement("div");
    timePill.className = "nf-audio-edit-pill";
    const selectionPill = document.createElement("div");
    selectionPill.className = "nf-audio-edit-pill";
    const sourcePill = document.createElement("div");
    sourcePill.className = "nf-audio-edit-pill";
    row1.append(playButton, undoButton, clearSelectionButton, timePill, selectionPill, sourcePill);

    const row2 = document.createElement("div");
    row2.className = "nf-audio-edit-row";
    const silenceInput = document.createElement("input");
    silenceInput.className = "nf-audio-edit-number";
    silenceInput.type = "number";
    silenceInput.min = "0";
    silenceInput.step = "0.1";
    silenceInput.value = "1.0";
    const insertSilenceButton = document.createElement("button");
    insertSilenceButton.textContent = "在播放头插入静音";
    styleButton(insertSilenceButton, "accent");
    const silenceSelectionButton = document.createElement("button");
    silenceSelectionButton.textContent = "选区变静音";
    styleButton(silenceSelectionButton, "accent");
    const deleteSelectionButton = document.createElement("button");
    deleteSelectionButton.textContent = "删除选区";
    styleButton(deleteSelectionButton, "accent");
    const saveButton = document.createElement("button");
    saveButton.textContent = "保存覆盖当前节点音频";
    styleButton(saveButton, "primary");
    row2.append(silenceInput, insertSilenceButton, silenceSelectionButton, deleteSelectionButton, saveButton);

    const help = document.createElement("div");
    help.className = "nf-audio-edit-help";
    help.textContent = "拖动波形可框选区间；单击波形可移动播放头。支持插入静音、把选区置静音、删除选区、撤销。保存后会让当前节点改用编辑后的音频。";

    const canvas = document.createElement("canvas");
    canvas.className = "nf-audio-edit-canvas";

    const footer = document.createElement("div");
    footer.className = "nf-audio-edit-footer";
    const footerLeft = document.createElement("div");
    footerLeft.className = "nf-audio-edit-help";
    const footerRight = document.createElement("div");
    footerRight.className = "nf-audio-edit-row";
    footerRight.append(closeButton.cloneNode(true));
    const footerClose = footerRight.querySelector("button");
    styleButton(footerClose, "subtle");
    footer.append(footerLeft, footerRight);

    const previewAudio = document.createElement("audio");
    previewAudio.preload = "metadata";
    previewAudio.style.display = "none";

    body.append(row1, row2, help, canvas, footer, previewAudio);
    dialog.append(header, body);
    document.body.appendChild(overlay);

    const state = {
        node,
        parentState,
        overlay,
        dialog,
        canvas,
        previewAudio,
        title,
        timePill,
        selectionPill,
        sourcePill,
        footerLeft,
        playButton,
        undoButton,
        clearSelectionButton,
        silenceInput,
        insertSilenceButton,
        silenceSelectionButton,
        deleteSelectionButton,
        saveButton,
        closeButton,
        footerClose,
        sourceAudioFile: "",
        sourceLabel: "",
        workingData: null,
        history: [],
        peaks: [],
        objectUrl: "",
        selection: null,
        dragState: null,
        rafId: 0,
        isOpen: false,
        isLoading: false,
        isSaving: false,
        error: "",
    };

    const updateUiState = () => {
        const duration = workingDataDuration(state.workingData);
        const playhead = clamp(state.previewAudio.currentTime || 0, 0, duration || 0);
        state.timePill.textContent = `播放头 ${formatTime(playhead)} / ${formatTime(duration)}`;
        if (state.selection && state.selection.end > state.selection.start) {
            const length = state.selection.end - state.selection.start;
            state.selectionPill.textContent = `选区 ${formatTime(state.selection.start)} → ${formatTime(state.selection.end)}（${length.toFixed(2)}s）`;
        } else {
            state.selectionPill.textContent = "选区 无";
        }
        state.sourcePill.textContent = `来源 ${state.sourceLabel || "未加载"}`;
        state.footerLeft.textContent = state.error || (state.isSaving ? "正在保存编辑后的音频..." : (state.isLoading ? "正在加载音频..." : "保存后会把当前节点切换到编辑后的音频版本。"));
        state.playButton.textContent = state.previewAudio.paused ? "播放" : "暂停";
        state.undoButton.disabled = state.history.length <= 0;
        const hasSelection = Boolean(state.selection && state.selection.end > state.selection.start);
        state.silenceSelectionButton.disabled = !hasSelection || !state.workingData;
        state.deleteSelectionButton.disabled = !hasSelection || !state.workingData;
        state.clearSelectionButton.disabled = !hasSelection;
        state.insertSilenceButton.disabled = !state.workingData;
        state.saveButton.disabled = !state.workingData || state.isSaving || state.isLoading;
    };

    const stopAnimationLoop = () => {
        if (state.rafId) {
            cancelAnimationFrame(state.rafId);
            state.rafId = 0;
        }
    };

    const startAnimationLoop = () => {
        stopAnimationLoop();
        const tick = () => {
            renderCanvas();
            if (state.isOpen && !state.previewAudio.paused) {
                state.rafId = requestAnimationFrame(tick);
            } else {
                state.rafId = 0;
            }
        };
        state.rafId = requestAnimationFrame(tick);
    };

    const releaseObjectUrl = () => {
        if (state.objectUrl) {
            try {
                URL.revokeObjectURL(state.objectUrl);
            } catch {
                // ignore
            }
            state.objectUrl = "";
        }
    };

    const refreshPreviewAudio = () => {
        releaseObjectUrl();
        if (!state.workingData) {
            state.previewAudio.removeAttribute("src");
            state.previewAudio.load();
            state.peaks = [];
            updateUiState();
            renderCanvas();
            return;
        }
        state.objectUrl = createObjectUrlFromWorkingData(state.workingData);
        state.previewAudio.src = state.objectUrl;
        state.previewAudio.load();
        state.peaks = computePeaksFromChannelData(state.workingData.channelData, 2200);
        updateUiState();
        renderCanvas();
    };

    const pushHistory = () => {
        if (!state.workingData) return;
        state.history.push(cloneWorkingData(state.workingData));
        if (state.history.length > 24) {
            state.history.shift();
        }
        updateUiState();
    };

    const setWorkingData = (nextData, options = {}) => {
        state.workingData = nextData ? cloneWorkingData(nextData) : null;
        if (!options.keepSelection) {
            state.selection = null;
        }
        if (!options.keepTime) {
            state.previewAudio.currentTime = 0;
        }
        refreshPreviewAudio();
    };

    function renderCanvas() {
        const width = Math.max(960, Math.floor(state.canvas.clientWidth || 960));
        const height = Math.max(360, Math.floor(state.canvas.clientHeight || 420));
        state.canvas.width = width;
        state.canvas.height = height;
        const context = state.canvas.getContext("2d");
        if (!context) return;

        context.clearRect(0, 0, width, height);
        context.fillStyle = "#0b1016";
        context.fillRect(0, 0, width, height);

        const duration = workingDataDuration(state.workingData);
        const midY = Math.floor(height / 2);

        context.strokeStyle = "rgba(255,255,255,0.08)";
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(0, midY);
        context.lineTo(width, midY);
        context.stroke();

        if (duration > 0) {
            const step = duration <= 30 ? 1 : duration <= 120 ? 5 : 10;
            for (let second = 0; second <= duration; second += step) {
                const x = (second / duration) * width;
                context.strokeStyle = "rgba(255,255,255,0.06)";
                context.beginPath();
                context.moveTo(x, 0);
                context.lineTo(x, height);
                context.stroke();
                context.fillStyle = "rgba(255,255,255,0.45)";
                context.font = "11px sans-serif";
                context.fillText(`${second}s`, Math.min(width - 30, x + 4), height - 8);
            }
        }

        if (state.peaks.length) {
            const barWidth = width / state.peaks.length;
            context.fillStyle = "#C93A1E";
            for (let index = 0; index < state.peaks.length; index += 1) {
                const peak = clamp(state.peaks[index] || 0, 0, 1);
                const barHeight = Math.max(1, peak * (height * 0.42));
                const x = index * barWidth;
                context.fillRect(x, midY - barHeight, Math.max(1, barWidth - 1), barHeight * 2);
            }
        }

        if (state.selection && state.selection.end > state.selection.start && duration > 0) {
            const startX = (state.selection.start / duration) * width;
            const endX = (state.selection.end / duration) * width;
            context.fillStyle = "rgba(255, 209, 102, 0.18)";
            context.fillRect(startX, 0, Math.max(1, endX - startX), height);
            context.strokeStyle = "#ffd166";
            context.lineWidth = 2;
            context.beginPath();
            context.moveTo(startX, 0);
            context.lineTo(startX, height);
            context.moveTo(endX, 0);
            context.lineTo(endX, height);
            context.stroke();
        }

        if (duration > 0) {
            const playhead = clamp(state.previewAudio.currentTime || 0, 0, duration);
            const x = (playhead / duration) * width;
            context.strokeStyle = "#ffffff";
            context.lineWidth = 2;
            context.beginPath();
            context.moveTo(x, 0);
            context.lineTo(x, height);
            context.stroke();
        }

        if (state.error && !state.workingData) {
            context.fillStyle = "#ffb4b4";
            context.font = "14px sans-serif";
            context.fillText(state.error, 16, 28);
        }

        updateUiState();
    }

    const eventToTime = (event) => {
        const rect = state.canvas.getBoundingClientRect();
        const ratio = clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
        const duration = workingDataDuration(state.workingData);
        return clamp(snapTime(ratio * duration, EDIT_SNAP_STEP_SECONDS), 0, Math.max(0, duration - 0.001));
    };

    const clearSelection = () => {
        state.selection = null;
        renderCanvas();
    };

    const closeModal = () => {
        state.previewAudio.pause();
        stopAnimationLoop();
        state.overlay.style.display = "none";
        state.isOpen = false;
    };

    const openModal = async (source) => {
        const normalized = normalizePreviewPayload(source) || getPreferredAudioSource(node);
        const audioFile = String(normalized?.audioFile || "").trim();
        if (!audioFile) {
            parentState.error = "请先给这个节点接入音频，或先选一个音频文件。";
            parentState.render?.();
            return;
        }

        state.overlay.style.display = "flex";
        state.isOpen = true;
        state.isLoading = true;
        state.isSaving = false;
        state.error = "";
        state.sourceAudioFile = audioFile;
        state.sourceLabel = String(normalized?.label || "当前节点音频");
        state.title.textContent = `编辑波形 · ${state.sourceLabel}`;
        state.selection = null;
        state.history = [];
        renderCanvas();

        try {
            const workingData = await decodeAudioFileToWorkingData(audioFile);
            setWorkingData(workingData, { keepSelection: false, keepTime: false });
        } catch (error) {
            state.error = error?.message || String(error);
            setWorkingData(null, { keepSelection: false, keepTime: false });
        } finally {
            state.isLoading = false;
            renderCanvas();
        }
    };

    const saveEdits = async () => {
        if (!state.workingData || state.isSaving) return;
        state.isSaving = true;
        state.error = "";
        renderCanvas();

        try {
            const blob = buildWavBlobFromWorkingData(state.workingData);
            const formData = new FormData();
            formData.append("audio", new File([blob], "edited.wav", { type: "audio/wav" }));
            formData.append("unique_id", String(node.id ?? ""));

            const response = await api.fetchApi("/nfypnode/audio-save-edits", {
                method: "POST",
                body: formData,
            });
            const payload = await response.json().catch(() => null);
            if (!response.ok || !payload?.audio_file) {
                throw new Error(payload?.error || `保存失败 (HTTP ${response.status})`);
            }

            const manualPath = encodeManualAudioFile(payload.audio_file);
            setWidgetValue(node, "edited_audio_file", manualPath);
            setWidgetValue(node, "audio_file", manualPath);
            setWidgetValue(node, "render_id", String(Date.now()));
            node.__nfAudioRuntimePreview = normalizePreviewPayload({
                audioFile: manualPath,
                label: "edited_audio_override",
                kind: "edited_override",
            });
            parentState.loadWaveform?.({
                audioFile: manualPath,
                label: "edited_audio_override",
                kind: "edited_override",
            });
            closeModal();
        } catch (error) {
            state.error = error?.message || String(error);
            renderCanvas();
        } finally {
            state.isSaving = false;
            renderCanvas();
        }
    };

    playButton.addEventListener("click", async () => {
        if (!state.workingData) return;
        if (state.previewAudio.paused) {
            await state.previewAudio.play();
            startAnimationLoop();
        } else {
            state.previewAudio.pause();
            renderCanvas();
        }
    });

    undoButton.addEventListener("click", () => {
        const previous = state.history.pop();
        if (!previous) return;
        setWorkingData(previous, { keepSelection: false, keepTime: true });
    });

    clearSelectionButton.addEventListener("click", clearSelection);

    insertSilenceButton.addEventListener("click", () => {
        if (!state.workingData) return;
        const seconds = Math.max(0, Number(state.silenceInput.value) || 0);
        if (!(seconds > 0)) {
            state.error = "静音秒数要大于 0。";
            renderCanvas();
            return;
        }
        pushHistory();
        const currentTime = clamp(state.previewAudio.currentTime || 0, 0, workingDataDuration(state.workingData));
        const next = insertSilenceIntoWorkingData(state.workingData, seconds, currentTime);
        setWorkingData(next, { keepSelection: false, keepTime: true });
        state.previewAudio.currentTime = clamp(currentTime + seconds, 0, workingDataDuration(state.workingData));
        state.error = "";
        renderCanvas();
    });

    silenceSelectionButton.addEventListener("click", () => {
        if (!state.workingData || !state.selection || state.selection.end <= state.selection.start) return;
        pushHistory();
        const next = silenceSelectionInWorkingData(state.workingData, state.selection.start, state.selection.end);
        const currentTime = clamp(state.selection.start, 0, workingDataDuration(next));
        setWorkingData(next, { keepSelection: true, keepTime: true });
        state.previewAudio.currentTime = currentTime;
        state.error = "";
        renderCanvas();
    });

    deleteSelectionButton.addEventListener("click", () => {
        if (!state.workingData || !state.selection || state.selection.end <= state.selection.start) return;
        pushHistory();
        const next = deleteSelectionFromWorkingData(state.workingData, state.selection.start, state.selection.end);
        const currentTime = clamp(state.selection.start, 0, workingDataDuration(next));
        setWorkingData(next, { keepSelection: false, keepTime: true });
        state.previewAudio.currentTime = currentTime;
        state.error = "";
        renderCanvas();
    });

    saveButton.addEventListener("click", saveEdits);
    closeButton.addEventListener("click", closeModal);
    footerClose.addEventListener("click", closeModal);
    overlay.addEventListener("click", (event) => {
        if (event.target === overlay) closeModal();
    });

    state.previewAudio.addEventListener("play", () => {
        startAnimationLoop();
        renderCanvas();
    });
    state.previewAudio.addEventListener("pause", () => {
        stopAnimationLoop();
        renderCanvas();
    });
    state.previewAudio.addEventListener("ended", () => {
        stopAnimationLoop();
        renderCanvas();
    });
    state.previewAudio.addEventListener("timeupdate", renderCanvas);
    state.previewAudio.addEventListener("loadedmetadata", renderCanvas);

    state.canvas.addEventListener("pointerdown", (event) => {
        if (!state.workingData) return;
        state.dragState = {
            anchor: eventToTime(event),
            moved: false,
        };
        state.canvas.setPointerCapture?.(event.pointerId);
    });

    state.canvas.addEventListener("pointermove", (event) => {
        if (!state.dragState || !state.workingData) return;
        const current = eventToTime(event);
        if (Math.abs(current - state.dragState.anchor) >= EDIT_SNAP_STEP_SECONDS) {
            state.dragState.moved = true;
            state.selection = {
                start: Math.min(state.dragState.anchor, current),
                end: Math.max(state.dragState.anchor, current),
            };
            renderCanvas();
        }
    });

    state.canvas.addEventListener("pointerup", (event) => {
        if (!state.dragState || !state.workingData) return;
        const current = eventToTime(event);
        if (state.dragState.moved) {
            state.selection = {
                start: Math.min(state.dragState.anchor, current),
                end: Math.max(state.dragState.anchor, current),
            };
            state.previewAudio.currentTime = state.selection.start;
        } else {
            state.selection = null;
            state.previewAudio.currentTime = current;
        }
        state.dragState = null;
        renderCanvas();
    });

    state.canvas.addEventListener("dblclick", () => {
        clearSelection();
    });

    node.__nfAudioEditModal = {
        open: openModal,
        close: closeModal,
        state,
    };
    return node.__nfAudioEditModal;
}

function buildEditor(node) {
    if (node.__nfAudioWaveformEditor) return node.__nfAudioWaveformEditor;

    const container = document.createElement("div");
    container.className = "nf-audio-waveform-editor";
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.gap = "8px";
    container.style.padding = "8px 0 4px";
    container.style.minWidth = "790px";

    const toolbar = document.createElement("div");
    toolbar.style.display = "flex";
    toolbar.style.flexWrap = "wrap";
    toolbar.style.gap = "8px";
    toolbar.style.alignItems = "center";

    const playButton = document.createElement("button");
    playButton.textContent = "播放";
    styleButton(playButton, "default");

    const chooseAudioButton = document.createElement("button");
    chooseAudioButton.textContent = "选择音频";
    styleButton(chooseAudioButton, "default");

    const editButton = document.createElement("button");
    editButton.textContent = "编辑波形";
    styleButton(editButton, "primary");

    const clearOverrideButton = document.createElement("button");
    clearOverrideButton.textContent = "取消覆盖";
    styleButton(clearOverrideButton, "subtle");

    const addButton = document.createElement("button");
    addButton.textContent = "添加标记";
    styleButton(addButton, "default");

    const removeButton = document.createElement("button");
    removeButton.textContent = "删除选中";
    styleButton(removeButton, "default");

    const status = document.createElement("div");
    status.style.fontSize = "12px";
    status.style.opacity = "0.85";

    toolbar.append(playButton, chooseAudioButton, editButton, clearOverrideButton, addButton, removeButton, status);

    const canvas = document.createElement("canvas");
    canvas.style.width = "100%";
    canvas.style.height = "190px";
    canvas.style.border = "1px solid rgba(255,255,255,0.18)";
    canvas.style.borderRadius = "10px";
    canvas.style.background = "#111";
    canvas.style.cursor = "pointer";

    const help = document.createElement("div");
    help.style.fontSize = "12px";
    help.style.opacity = "0.78";
    help.textContent = "点击波形可定位播放头（自动吸附到 1 秒网格）。点“编辑波形”会弹出独立编辑窗口，可以像剪音频那样框选、插入静音、把选区变静音、删除片段，再保存覆盖到当前节点。";

    const audio = document.createElement("audio");
    audio.preload = "metadata";
    audio.style.display = "none";

    const uploadInput = document.createElement("input");
    uploadInput.type = "file";
    uploadInput.accept = "audio/*,.wav,.mp3,.flac,.ogg,.m4a,.aac";
    uploadInput.style.display = "none";

    container.append(toolbar, canvas, help, audio, uploadInput);

    const domWidget = typeof node.addDOMWidget === "function"
        ? node.addDOMWidget("waveform_editor", "waveform_editor", container, {
            serialize: false,
            hideOnZoom: false,
            getValue: () => "",
            setValue: () => {},
        })
        : null;
    if (domWidget) {
        domWidget.computeSize = () => [DEFAULT_WIDTH - 30, 255];
    }

    const state = {
        node,
        container,
        canvas,
        audio,
        status,
        playButton,
        chooseAudioButton,
        editButton,
        clearOverrideButton,
        addButton,
        removeButton,
        uploadInput,
        peaks: [],
        duration: 0,
        selectedIndex: 0,
        pointerDown: false,
        error: "",
        sourceLabel: "",
        lastAudioIdentity: "",
        lastAudioSourceKey: "",
        lastAudioFile: "",
    };

    node.__nfAudioWaveformEditor = state;
    ensureEditorModal(node, state);

    const updateSourceControls = () => {
        const manualOverride = isManualAudioOverrideEnabled(node);
        clearOverrideButton.disabled = !manualOverride;
        chooseAudioButton.textContent = manualOverride ? "替换覆盖音频" : "选择音频";
    };

    const getStatusText = () => {
        const keyframesWidget = getWidget(node, "keyframes_json");
        const keyframes = normalizeKeyframes(parseKeyframes(keyframesWidget?.value), state.duration);
        const skipWidget = getWidget(node, "skip_initial_segment");
        const tailWidget = getWidget(node, "include_tail_segment");
        const sourceText = state.sourceLabel ? ` | 来源 ${state.sourceLabel}` : "";
        return `时间 ${formatTime(audio.currentTime || 0)} / ${formatTime(state.duration)} | 标记 ${keyframes.length} | 选中 ${state.selectedIndex + 1} | 跳过首段 ${Boolean(skipWidget?.value)} | 包含尾段 ${Boolean(tailWidget?.value)}${sourceText}`;
    };

    const render = () => {
        const width = Math.max(780, Math.floor(container.clientWidth || DEFAULT_WIDTH));
        const height = 190;
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d");
        if (!context) return;

        context.clearRect(0, 0, width, height);
        context.fillStyle = "#101317";
        context.fillRect(0, 0, width, height);

        if (state.error) {
            context.fillStyle = "#ff8f8f";
            context.font = "14px sans-serif";
            context.fillText(state.error, 12, 24);
            status.textContent = state.error;
            updateSourceControls();
            return;
        }

        const midY = Math.floor(height / 2);
        context.strokeStyle = "rgba(255,255,255,0.12)";
        context.beginPath();
        context.moveTo(0, midY);
        context.lineTo(width, midY);
        context.stroke();

        if (state.peaks.length) {
            const barWidth = width / state.peaks.length;
            context.fillStyle = "#C93A1E";
            for (let index = 0; index < state.peaks.length; index += 1) {
                const peak = Math.max(0, Math.min(1, state.peaks[index] || 0));
                const barHeight = Math.max(1, peak * (height * 0.42));
                const x = index * barWidth;
                context.fillRect(x, midY - barHeight, Math.max(1, barWidth - 1), barHeight * 2);
            }
        }

        const keyframesWidget = getWidget(node, "keyframes_json");
        const keyframes = normalizeKeyframes(parseKeyframes(keyframesWidget?.value), state.duration);
        keyframes.forEach((time, index) => {
            const ratio = state.duration > 0 ? time / state.duration : 0;
            const x = Math.max(0, Math.min(width, ratio * width));
            context.strokeStyle = index === state.selectedIndex ? "#ffd166" : "#ff7f50";
            context.lineWidth = index === state.selectedIndex ? 3 : 2;
            context.beginPath();
            context.moveTo(x, 0);
            context.lineTo(x, height);
            context.stroke();
            context.fillStyle = context.strokeStyle;
            context.font = "12px sans-serif";
            context.fillText(String(index + 1), Math.min(width - 14, x + 4), 14 + (index % 2) * 14);
        });

        if (state.duration > 0) {
            const ratio = Math.max(0, Math.min(1, (audio.currentTime || 0) / state.duration));
            const x = ratio * width;
            context.strokeStyle = "#ffffff";
            context.lineWidth = 2;
            context.beginPath();
            context.moveTo(x, 0);
            context.lineTo(x, height);
            context.stroke();
        }

        status.textContent = getStatusText();
        updateSourceControls();
    };

    const syncKeyframes = (nextKeyframes, preferredSelectedIndex = null) => {
        const keyframesWidget = getWidget(node, "keyframes_json");
        const normalized = normalizeKeyframes(nextKeyframes, state.duration);
        const serialized = JSON.stringify(normalized.map((value) => Number(value.toFixed(3))));
        if (keyframesWidget) keyframesWidget.value = serialized;
        state.selectedIndex = Math.max(0, Math.min(preferredSelectedIndex ?? state.selectedIndex, Math.max(0, normalized.length - 1)));
        render();
    };

    const loadWaveform = async (preferredSource = null) => {
        const source = normalizePreviewPayload(preferredSource) || getPreferredAudioSource(node);
        const audioFile = String(source?.audioFile || "").trim();

        state.error = "";
        state.sourceLabel = String(source?.label || "").trim();
        updateSourceControls();
        if (source?.kind === "linked") {
            hookExternalSourceWidget(node, source, () => loadWaveform());
        }

        if (!audioFile) {
            state.peaks = [];
            state.duration = 0;
            state.sourceLabel = "";
            state.lastAudioIdentity = "";
            state.lastAudioSourceKey = "";
            state.lastAudioFile = "";
            state.selectedIndex = 0;

            syncKeyframes([], 0);
            setWidgetValue(node, "segment_index", 0, false);

            audio.removeAttribute("src");
            audio.load();
            render();
            return;
        }

        try {
            const response = await fetch(`/nfypnode/audio-waveform?audio_file=${encodeURIComponent(audioFile)}&bins=1400`);
            const payload = await response.json();
            if (!response.ok) throw new Error(payload?.error || "加载波形失败");

            state.peaks = Array.isArray(payload.peaks) ? payload.peaks : [];
            state.duration = Number(payload.duration) || 0;
            audio.src = payload.audio_url || `/nfypnode/audio-file?audio_file=${encodeURIComponent(audioFile)}`;
            audio.load();

            const nextAudioIdentity = String(payload.audioIdentity ?? payload.audio_identity ?? source?.audioIdentity ?? audioFile).trim();
            const audioChanged = Boolean(state.lastAudioIdentity) && state.lastAudioIdentity !== nextAudioIdentity;

            if (audioChanged) {
                state.selectedIndex = 0;
                syncKeyframes([], 0);
                setWidgetValue(node, "segment_index", 0, false);
            } else {
                const currentKeyframes = parseKeyframes(getWidget(node, "keyframes_json")?.value);
                syncKeyframes(currentKeyframes, state.selectedIndex);
            }

            state.lastAudioIdentity = nextAudioIdentity;
            state.lastAudioSourceKey = `${String(source?.kind || "").trim()}::${audioFile}`;
            state.lastAudioFile = audioFile;
        } catch (error) {
            state.error = error?.message || String(error);
            state.peaks = [];
            state.duration = 0;
            render();
        }
    };

    state.loadWaveform = loadWaveform;
    state.render = render;

    const seekToPosition = (event) => {
        if (!(state.duration > 0)) return;
        const rect = canvas.getBoundingClientRect();
        const offsetX = event.clientX - rect.left;
        const ratio = Math.max(0, Math.min(1, offsetX / rect.width));
        let snapped = snapTime(ratio * state.duration, SNAP_STEP_SECONDS);
        if (state.duration > 0) {
            snapped = Math.min(snapped, Math.max(0, state.duration - 0.001));
        }
        audio.currentTime = snapped;

        const keyframes = normalizeKeyframes(parseKeyframes(getWidget(node, "keyframes_json")?.value), state.duration);
        if (!keyframes.length) {
            render();
            return;
        }
        let nearestIndex = 0;
        let nearestDistance = Number.POSITIVE_INFINITY;
        for (let index = 0; index < keyframes.length; index += 1) {
            const distance = Math.abs(keyframes[index] - audio.currentTime);
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestIndex = index;
            }
        }
        if (nearestDistance <= Math.max(0.2, state.duration / 100)) {
            state.selectedIndex = nearestIndex;
        }
        render();
    };

    playButton.addEventListener("click", async () => {
        if (audio.paused) {
            await audio.play();
        } else {
            audio.pause();
        }
    });

    chooseAudioButton.addEventListener("click", () => {
        uploadInput.value = "";
        uploadInput.click();
    });

    clearOverrideButton.addEventListener("click", () => {
        node.__nfAudioRuntimePreview = null;
        setWidgetValue(node, "edited_audio_file", "");
        setWidgetValue(node, "audio_file", "");
        setWidgetValue(node, "render_id", String(Date.now()));
    });

    editButton.addEventListener("click", () => {
        const modal = ensureEditorModal(node, state);
        modal.open(getPreferredAudioSource(node));
    });

    addButton.addEventListener("click", () => {
        let current = snapTime(audio.currentTime || 0, SNAP_STEP_SECONDS);
        if (state.duration > 0) {
            current = Math.min(current, Math.max(0, state.duration - 0.001));
        }
        audio.currentTime = current;
        const keyframes = parseKeyframes(getWidget(node, "keyframes_json")?.value);
        keyframes.push(current);
        const normalized = normalizeKeyframes(keyframes, state.duration);
        const bucket = Math.round(current / SNAP_STEP_SECONDS);
        const preferredIndex = normalized.findIndex((value) => Math.round(value / SNAP_STEP_SECONDS) === bucket);
        syncKeyframes(normalized, preferredIndex >= 0 ? preferredIndex : normalized.length - 1);
    });

    removeButton.addEventListener("click", () => {
        const keyframes = normalizeKeyframes(parseKeyframes(getWidget(node, "keyframes_json")?.value), state.duration);
        if (!keyframes.length) return;
        keyframes.splice(state.selectedIndex, 1);
        syncKeyframes(keyframes, Math.max(0, state.selectedIndex - 1));
    });

    audio.addEventListener("play", () => { playButton.textContent = "暂停"; render(); });
    audio.addEventListener("pause", () => { playButton.textContent = "播放"; render(); });
    audio.addEventListener("timeupdate", render);
    audio.addEventListener("loadedmetadata", render);

    canvas.addEventListener("pointerdown", (event) => { state.pointerDown = true; seekToPosition(event); });
    canvas.addEventListener("pointermove", (event) => { if (state.pointerDown) seekToPosition(event); });
    window.addEventListener("pointerup", () => { state.pointerDown = false; });

    uploadInput.addEventListener("change", async () => {
        const selectedFile = uploadInput.files?.[0];
        if (!selectedFile) return;

        const originalLabel = chooseAudioButton.textContent;
        chooseAudioButton.disabled = true;
        chooseAudioButton.textContent = "上传中...";
        state.error = "";
        render();

        try {
            const uploadedName = await uploadAudioFile(selectedFile);
            const audioFileWidget = getWidget(node, "audio_file");
            if (Array.isArray(audioFileWidget?.options?.values) && !audioFileWidget.options.values.includes(uploadedName)) {
                audioFileWidget.options.values.push(uploadedName);
            }
            node.__nfAudioRuntimePreview = null;
            setWidgetValue(node, "edited_audio_file", "");
            setWidgetValue(node, "audio_file", encodeManualAudioFile(uploadedName));
            setWidgetValue(node, "render_id", String(Date.now()));
        } catch (error) {
            state.error = error?.message || String(error);
            render();
        } finally {
            chooseAudioButton.disabled = false;
            chooseAudioButton.textContent = originalLabel;
            updateSourceControls();
            uploadInput.value = "";
        }
    });

    const audioFileWidget = getWidget(node, "audio_file");
    if (audioFileWidget && !audioFileWidget.__nfAudioHooked) {
        const originalCallback = audioFileWidget.callback;
        audioFileWidget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            loadWaveform();
            return result;
        };
        audioFileWidget.__nfAudioHooked = true;
    }

    const editedAudioFileWidget = getWidget(node, "edited_audio_file");
    if (editedAudioFileWidget && !editedAudioFileWidget.__nfAudioHooked) {
        const originalCallback = editedAudioFileWidget.callback;
        editedAudioFileWidget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            loadWaveform();
            return result;
        };
        editedAudioFileWidget.__nfAudioHooked = true;
    }

    const keyframesWidget = getWidget(node, "keyframes_json");
    const renderIdWidget = getWidget(node, "render_id");
    hideWidget(audioFileWidget);
    hideWidget(editedAudioFileWidget);
    hideWidget(keyframesWidget);
    hideWidget(renderIdWidget);

    requestAnimationFrame(() => {
        render();
        loadWaveform();
        resizeNode(node);
    });

    return state;
}

function syncFromStoredState(node) {
    normalizeSegmentIndexWidget(node);
    const audioFileWidget = getWidget(node, "audio_file");
    const editedAudioFileWidget = getWidget(node, "edited_audio_file");
    const keyframesWidget = getWidget(node, "keyframes_json");
    const renderIdWidget = getWidget(node, "render_id");
    hideWidget(audioFileWidget);
    hideWidget(editedAudioFileWidget);
    hideWidget(keyframesWidget);
    hideWidget(renderIdWidget);

    const editor = buildEditor(node);
    const keyframes = normalizeKeyframes(parseKeyframes(keyframesWidget?.value), editor.duration);
    editor.selectedIndex = Math.max(0, Math.min(editor.selectedIndex, Math.max(0, keyframes.length - 1)));
    if (editor.container.isConnected) {
        editor.render?.();
        editor.loadWaveform?.();
    }
}

app.registerExtension({
    name: "NfypNode.AudioWaveformEditor",
    async beforeRegisterNodeDef(nodeType) {
        if (nodeType.comfyClass !== TARGET_CLASS) return;

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            syncFromStoredState(this);
            return result;
        };

        const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = originalOnConnectionsChange?.apply(this, arguments);
            syncFromStoredState(this);
            return result;
        };

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = originalOnExecuted?.apply(this, arguments);
            const payload = message?.ui || message || {};
            const preview = Array.isArray(payload.nf_audio_preview) ? payload.nf_audio_preview[0] : payload.nf_audio_preview;
            const normalizedPreview = normalizePreviewPayload(preview);
            if (normalizedPreview?.audioFile) {
                this.__nfAudioRuntimePreview = normalizedPreview;
                const editor = buildEditor(this);
                // Keep the current audio selection when execution only refreshes preview data.
                // This avoids clearing markers unless the user actually switches to another audio.
                editor.loadWaveform?.();
            }
            return result;
        };
    },
    async nodeCreated(node) {
        if (node.comfyClass !== TARGET_CLASS) return;
        buildEditor(node);
        syncFromStoredState(node);
    },
});
