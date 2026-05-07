import { app } from "../../scripts/app.js";

const NF_IGNORE_GROUP_TYPE = "南风阳平/工具/南风忽略";
const POLL_MS = 180;

function toBool(value, fallback = false) {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  if (typeof value === "string") {
    const v = value.trim().toLowerCase();
    if (["true", "yes", "on", "1", "enable", "enabled"].includes(v)) return true;
    if (["false", "no", "off", "0", "disable", "disabled", "none", "null", ""].includes(v)) return false;
  }
  return fallback;
}

function normalizeText(v) {
  return String(v ?? "").trim();
}

function getAllGroups(graph) {
  return Array.isArray(graph?._groups) ? graph._groups : [];
}

function refreshGroupNodes(group) {
  try {
    if (typeof group?.recomputeInsideNodes === "function") {
      group.recomputeInsideNodes();
    }
  } catch (e) {
    console.warn("[南风忽略] recomputeInsideNodes 失败", e);
  }
  return Array.isArray(group?._nodes) ? group._nodes : [];
}

function setNodeMode(node, mode) {
  if (!node || node.mode === mode) return false;
  node.mode = mode;
  try {
    if (typeof node.onModeChange === "function") node.onModeChange(mode);
  } catch (e) {
    console.warn("[南风忽略] onModeChange 调用失败", e);
  }
  return true;
}

function isBooleanWidget(widget) {
  if (!widget) return false;
  if (typeof widget.value === "boolean") return true;
  const type = String(widget.type || "").toLowerCase();
  return type.includes("toggle") || type.includes("boolean") || type === "combo";
}

function readBooleanFromNode(node, visited = new Set()) {
  if (!node || visited.has(node.id)) return null;
  visited.add(node.id);

  // 1) 先直接看 widgets
  if (Array.isArray(node.widgets)) {
    for (const widget of node.widgets) {
      if (!widget) continue;
      if (typeof widget.value === "boolean") return widget.value;
      if (isBooleanWidget(widget)) {
        if (typeof widget.value === "string" || typeof widget.value === "number") {
          return toBool(widget.value, false);
        }
      }
    }
  }

  // 2) 再看 widgets_values
  if (Array.isArray(node.widgets_values)) {
    for (const value of node.widgets_values) {
      if (typeof value === "boolean") return value;
      if (typeof value === "string" || typeof value === "number") {
        const normalized = String(value).trim().toLowerCase();
        if (["true", "false", "yes", "no", "on", "off", "0", "1"].includes(normalized)) {
          return toBool(value, false);
        }
      }
    }
  }

  // 3) 如果是 reroute / 中转节点，继续向上追
  if (Array.isArray(node.inputs) && node.inputs.length) {
    const linkId = node.inputs[0]?.link;
    if (linkId != null && node.graph?.links?.[linkId]) {
      const link = node.graph.links[linkId];
      const prev = node.graph.getNodeById?.(link.origin_id);
      const fromPrev = readBooleanFromNode(prev, visited);
      if (fromPrev != null) return fromPrev;
    }
  }

  // 4) 最后兜底看看 properties 里有没有明显布尔字段
  if (node.properties && typeof node.properties === "object") {
    for (const [key, value] of Object.entries(node.properties)) {
      const lk = key.toLowerCase();
      if (typeof value === "boolean") return value;
      if (["enabled", "enable", "toggle", "switch", "value", "state", "bool", "boolean"].includes(lk)) {
        if (typeof value === "string" || typeof value === "number") return toBool(value, false);
      }
    }
  }

  return null;
}

function readInputBool(controllerNode) {
  const graph = controllerNode?.graph;
  const input = controllerNode?.inputs?.[0];
  if (!graph || !input || input.link == null || !graph.links?.[input.link]) return null;
  const link = graph.links[input.link];
  const originNode = graph.getNodeById?.(link.origin_id);
  return readBooleanFromNode(originNode);
}

function getMatchedGroups(graph, mode, pattern) {
  const groups = getAllGroups(graph);
  const text = normalizeText(pattern);
  if (!text) return [];

  const matchMode = String(mode || "exact").toLowerCase();
  if (matchMode === "regex") {
    let reg = null;
    try {
      reg = new RegExp(text, "i");
    } catch (e) {
      console.warn("[南风忽略] 组名正则无效", text, e);
      return [];
    }
    return groups.filter((g) => reg.test(String(g?.title || "")));
  }

  if (matchMode === "contains") {
    const lowered = text.toLowerCase();
    return groups.filter((g) => String(g?.title || "").toLowerCase().includes(lowered));
  }

  return groups.filter((g) => normalizeText(g?.title) === text);
}

function applyGroupsState(controllerNode, inputBool) {
  const graph = controllerNode?.graph;
  if (!graph) return;

  const groupPattern = controllerNode.properties.groupName;
  const matchMode = controllerNode.properties.matchMode;
  const trueMeansIgnore = toBool(controllerNode.properties.trueMeansIgnore, true);
  const fallbackBool = toBool(controllerNode.properties.fallbackValue, false);
  const effectiveBool = inputBool == null ? fallbackBool : toBool(inputBool, fallbackBool);
  const shouldIgnore = trueMeansIgnore ? effectiveBool : !effectiveBool;
  const targetMode = shouldIgnore ? LiteGraph.NEVER : LiteGraph.ALWAYS;
  const matchedGroups = getMatchedGroups(graph, matchMode, groupPattern);

  let changed = false;
  for (const group of matchedGroups) {
    const nodes = refreshGroupNodes(group);
    for (const node of nodes) {
      if (!node) continue;
      if (node.id === controllerNode.id) continue; // 别把控制器自己关掉
      changed = setNodeMode(node, targetMode) || changed;
    }
    group.rgthree_hasAnyActiveNode = !shouldIgnore;
  }

  controllerNode._nfLastSummary = `${matchedGroups.length}组 / ${shouldIgnore ? "已忽略" : "已启用"}`;
  controllerNode._nfLastResolvedBool = effectiveBool;
  controllerNode._nfLastAppliedMode = targetMode;

  if (changed) {
    graph.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
  }
}

class NanFengIgnoreGroup extends LiteGraph.LGraphNode {
  constructor() {
    super();
    this.title = NanFengIgnoreGroup.title;
    this.isVirtualNode = true;
    this.serialize_widgets = true;
    this.size = [320, 170];

    this.addInput("布尔开关", "BOOLEAN");
    this.addOutput("OPT_CONNECTION", "*");

    this.properties = this.properties || {};
    this.properties.groupName = this.properties.groupName ?? "语音推理";
    this.properties.matchMode = this.properties.matchMode ?? "exact";
    this.properties.trueMeansIgnore = this.properties.trueMeansIgnore ?? true;
    this.properties.fallbackValue = this.properties.fallbackValue ?? false;
    this.properties.note = this.properties.note ?? "前端节点；建议接可见布尔/开关节点";

    this.addWidget("text", "目标组名", this.properties.groupName, (v) => {
      this.properties.groupName = String(v ?? "");
      this._nfForceApply = true;
    });

    this.addWidget("combo", "匹配方式", this.properties.matchMode, (v) => {
      this.properties.matchMode = v;
      this._nfForceApply = true;
    }, { values: ["exact", "contains", "regex"] });

    this.addWidget("toggle", "TRUE时忽略目标组", this.properties.trueMeansIgnore, (v) => {
      this.properties.trueMeansIgnore = !!v;
      this._nfForceApply = true;
    }, { on: "yes", off: "no" });

    this.addWidget("toggle", "本地兜底布尔", this.properties.fallbackValue, (v) => {
      this.properties.fallbackValue = !!v;
      this._nfForceApply = true;
    }, { on: "true", off: "false" });

    this.addWidget("button", "立即同步一次", null, () => {
      this._nfForceApply = true;
      this._nfTick();
    });

    this._nfTimer = null;
    this._nfForceApply = true;
    this._nfLastSummary = "未同步";
    this._nfLastResolvedBool = null;
    this._nfLastAppliedMode = null;
  }

  onAdded(graph) {
    this._nfStartWatcher();
    this._nfForceApply = true;
  }

  onRemoved() {
    this._nfStopWatcher();
  }

  onConfigure(info) {
    this._nfForceApply = true;
  }

  onConnectionsChange() {
    this._nfForceApply = true;
  }

  onPropertyChanged() {
    this._nfForceApply = true;
  }

  onDrawForeground(ctx) {
    if (this.flags?.collapsed) return;
    const x = 10;
    let y = this.size[1] - 28;
    ctx.save();
    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#BBB";
    const sourceText = this._nfLastResolvedBool == null ? "输入: 未识别，使用本地兜底" : `输入: ${this._nfLastResolvedBool}`;
    ctx.fillText(sourceText, x, y);
    y += 14;
    ctx.fillStyle = "#89A";
    ctx.fillText(this._nfLastSummary || "未同步", x, y);
    ctx.restore();
  }

  _nfStartWatcher() {
    this._nfStopWatcher();
    this._nfTimer = setInterval(() => this._nfTick(), POLL_MS);
  }

  _nfStopWatcher() {
    if (this._nfTimer) {
      clearInterval(this._nfTimer);
      this._nfTimer = null;
    }
  }

  _nfSignatureFromInput(inputBool) {
    return JSON.stringify({
      inputBool: inputBool,
      groupName: this.properties.groupName,
      matchMode: this.properties.matchMode,
      trueMeansIgnore: !!this.properties.trueMeansIgnore,
      fallbackValue: !!this.properties.fallbackValue,
    });
  }

  _nfTick() {
    if (!this.graph) return;
    const inputBool = readInputBool(this);
    const signature = this._nfSignatureFromInput(inputBool);
    if (!this._nfForceApply && signature === this._nfLastSignature) return;
    this._nfLastSignature = signature;
    this._nfForceApply = false;
    applyGroupsState(this, inputBool);
  }
}

NanFengIgnoreGroup.title = NF_IGNORE_GROUP_TYPE;
NanFengIgnoreGroup.type = NF_IGNORE_GROUP_TYPE;
NanFengIgnoreGroup.collapsible = false;
NanFengIgnoreGroup.category = "南风阳平/工具";
NanFengIgnoreGroup["@groupName"] = { type: "string" };
NanFengIgnoreGroup["@matchMode"] = { type: "combo", values: ["exact", "contains", "regex"] };
NanFengIgnoreGroup["@trueMeansIgnore"] = { type: "boolean" };
NanFengIgnoreGroup["@fallbackValue"] = { type: "boolean" };

app.registerExtension({
  name: "nanfeng.IgnoreGroupByBool",
  registerCustomNodes() {
    LiteGraph.registerNodeType(NF_IGNORE_GROUP_TYPE, NanFengIgnoreGroup);
  },
});
