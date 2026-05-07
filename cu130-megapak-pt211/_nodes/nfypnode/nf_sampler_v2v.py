
import copy
import gc
import json
import os
import re
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

try:
    import comfy.sample
    import comfy.nested_tensor
    import comfy.utils
except Exception:
    comfy = None

try:
    import comfy.model_management
except Exception:
    pass

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    from nodes import NODE_CLASS_MAPPINGS as CORE_NODE_CLASS_MAPPINGS
except Exception:
    CORE_NODE_CLASS_MAPPINGS = {}

try:
    from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
except Exception:
    SamplerCustomAdvanced = None


def _normalize_node_output(result):
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    if hasattr(result, "args") and isinstance(result.args, tuple):
        return result.args
    return (result,)


_KEEP_NOISE_MASK = object()


class _NanFengEmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]

        if getattr(latent_image, "is_nested", False):
            tensors = latent_image.unbind()
            zeros = []
            for t in tensors:
                zeros.append(torch.zeros(t.shape, dtype=t.dtype, layout=t.layout, device="cpu"))
            return comfy.nested_tensor.NestedTensor(zeros)

        return torch.zeros(
            latent_image.shape,
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )


class _NanFengRandomNoise:
    def __init__(self, seed):
        self.seed = int(seed)

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)


class NanFengSamplerAdvancedV2V:
    CATEGORY = "南风阳平/采样"
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"

    PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets", "sampler")
    DEFAULT_X2_MODEL = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    SEGMENT_LATENT_SUBDIR = "latents"
    SEGMENT_CFG_DECAY_FACTORS = (1.0, 0.7, 0.5)
    FLOW_RAINBOW_COLORS = (196, 208, 220, 82, 51, 39, 93, 201)
    FLOW_ARROW_COLOR = 81
    FLOW_NOTE_COLOR = 250
    LOG_TEXT_COLOR = 151

    @classmethod
    def _preset_names(cls):
        names = []
        if os.path.isdir(cls.PRESET_DIR):
            for filename in sorted(os.listdir(cls.PRESET_DIR)):
                if filename.lower().endswith(".json"):
                    preset_name = os.path.splitext(filename)[0]
                    if preset_name not in names:
                        names.append(preset_name)
        if "自定义" not in names:
            names.insert(0, "自定义")
        return names or ["自定义"]

    @classmethod
    def _upscale_model_names(cls):
        fallback = [cls.DEFAULT_X2_MODEL]
        if folder_paths is None:
            return fallback
        try:
            names = folder_paths.get_filename_list("latent_upscale_models")
            if names:
                return names
        except Exception:
            pass
        return fallback

    @classmethod
    def INPUT_TYPES(cls):
        preset_names = cls._preset_names()
        upscale_model_names = cls._upscale_model_names()
        return {
            "required": {
                "引导器": ("GUIDER",),
                "采样器": ("SAMPLER",),
                "Latent图像": ("LATENT",),

                "预设配置": (preset_names, {"default": preset_names[0]}),
                "一采西格玛文本": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),

                "添加噪波": ("BOOLEAN", {"default": True}),
                "噪波种子": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "control_after_generate": True,
                    },
                ),

                "启动二采": ("BOOLEAN", {"default": False}),
                "二采使用放大模型": ("BOOLEAN", {"default": True}),
                "二采放大模型": (upscale_model_names, {"default": upscale_model_names[0]}),
                "二采西格玛": (
                    "STRING",
                    {
                        "default": "0.85, 0.7250, 0.4219, 0.0",
                        "multiline": False,
                    },
                ),

                "启动分段": ("BOOLEAN", {"default": False}),
                "单段秒数": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 1.0,
                        "max": 300.0,
                        "step": 0.1,
                    },
                ),
                "分段重叠秒数": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.1,
                    },
                ),
                "帧率": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.01,
                    },
                ),
                "总帧数覆盖": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                    },
                ),
                "总时长秒数覆盖": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100000.0,
                        "step": 0.1,
                    },
                ),

                "优先使用分段方案JSON": ("BOOLEAN", {"default": True}),
                "分段方案JSON": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),

                "保存分段latent": ("BOOLEAN", {"default": False}),
                "分段latent前缀": (
                    "STRING",
                    {
                        "default": "nf_v2v_seg",
                        "multiline": False,
                    },
                ),
                "官方式latent续写": ("BOOLEAN", {"default": True}),
                "续写重叠引导强度": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "续写输出线性融合": ("BOOLEAN", {"default": True}),
                "是否图生视频": ("BOOLEAN", {"default": False}),
                "按空行分段提示词": ("BOOLEAN", {"default": False}),
                "分段正向提示词文本": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "分段负向提示词文本": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
            },
            "optional": {
                "二采latent输入图像": ("IMAGE",),
                "二采引导器": ("GUIDER",),
                "二采采样器": ("SAMPLER",),
                "二采噪波": ("NOISE",),
                "西格玛": ("SIGMAS",),
                "二采VAE": ("VAE",),
                "续写VAE": ("VAE",),
                "分段提示词CLIP": ("CLIP",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    def _log(self, stage, message, level="INFO", icon="ℹ️", **kwargs):
        ts = datetime.now().strftime("%H:%M:%S")
        suffix = ""
        if kwargs:
            parts = [f"{k}={v}" for k, v in kwargs.items()]
            suffix = " | " + " | ".join(parts)
        line = f"{icon} [{ts}] [南风采样器V2V][{level}][{stage}] {message}{suffix}"
        print(self._ansi_text(line, self.LOG_TEXT_COLOR))

    def _supports_ansi_color(self):
        cached = getattr(self, "_nf_ansi_color_enabled", None)
        if cached is not None:
            return bool(cached)

        disabled = str(os.environ.get("NF_NO_COLOR", "")).strip().lower() in ("1", "true", "yes", "on")
        if disabled:
            self._nf_ansi_color_enabled = False
            return False

        try:
            is_tty = bool(sys.stdout.isatty())
        except Exception:
            is_tty = False

        if not is_tty:
            self._nf_ansi_color_enabled = False
            return False

        if os.name != "nt":
            self._nf_ansi_color_enabled = True
            return True

        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            if handle not in (0, -1):
                mode = ctypes.c_uint()
                if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                    if kernel32.SetConsoleMode(handle, int(mode.value) | 0x0004):
                        self._nf_ansi_color_enabled = True
                        return True
        except Exception:
            pass

        env = os.environ
        enabled = any((
            env.get("WT_SESSION"),
            env.get("ANSICON"),
            env.get("TERM_PROGRAM"),
            env.get("COLORTERM"),
            env.get("TERM"),
            str(env.get("ConEmuANSI", "")).upper() == "ON",
        ))
        self._nf_ansi_color_enabled = bool(enabled)
        return bool(enabled)

    def _ansi_text(self, text, color_code):
        text = str(text)
        if not self._supports_ansi_color():
            return text
        return f"\033[38;5;{int(color_code)}m{text}\033[0m"

    def _rainbow_text(self, text):
        text = str(text)
        if not text or not self._supports_ansi_color():
            return text

        out = []
        color_idx = 0
        colors = tuple(int(x) for x in self.FLOW_RAINBOW_COLORS)
        for ch in text:
            if ch.isspace():
                out.append(ch)
                continue
            color = colors[color_idx % len(colors)]
            out.append(f"\033[38;5;{color}m{ch}\033[0m")
            color_idx += 1
        return "".join(out)

    def _segment_text(self, segment_index):
        idx = max(1, int(segment_index))
        mapping = {
            1: "一段",
            2: "二段",
            3: "三段",
            4: "四段",
            5: "五段",
            6: "六段",
            7: "七段",
            8: "八段",
            9: "九段",
            10: "十段",
        }
        return mapping.get(idx, f"{idx}段")

    def _extract_segment_numbers(self, segment_label):
        text = "" if segment_label is None else str(segment_label)
        match = re.search(r"(\d+)\s*/\s*(\d+)", text)
        if not match:
            return None, None
        return int(match.group(1)), int(match.group(2))

    def _flow_stage_text(self, pass_index=1, segment_label="full", suffix="latent"):
        pass_prefix = {1: "① 采", 2: "② 采"}.get(int(pass_index), f"{int(pass_index)} 采")
        segment_idx, _ = self._extract_segment_numbers(segment_label)
        if segment_idx is not None:
            segment_text = self._segment_text(segment_idx)
        else:
            label_text = str(segment_label or "").strip().lower()
            if label_text in ("", "full", "full-no-loop", "single_pipeline", "default"):
                segment_text = "整段"
            else:
                segment_text = str(segment_label)
        if suffix:
            return f"{pass_prefix}{segment_text} {suffix}"
        return f"{pass_prefix}{segment_text}"

    def _flow_print(self, text, next_text=None, note=""):
        line = self._rainbow_text(text)
        if next_text:
            line = f"{line}{self._ansi_text(' → ', self.FLOW_ARROW_COLOR)}{self._rainbow_text(next_text)}"
        if note:
            line = f"{line}{self._ansi_text(f' {note}', self.FLOW_NOTE_COLOR)}"
        print(line)

    def _shallow_copy_latent(self, latent):
        if isinstance(latent, dict):
            return dict(latent)
        return copy.copy(latent)

    def _latent_shape_text(self, latent):
        try:
            samples = latent["samples"] if isinstance(latent, dict) else getattr(latent, "samples")
            return tuple(samples.shape)
        except Exception:
            return "unknown"

    def _scale_hw(self, h, w, factor):
        target_h = max(1, int(round(float(h) * float(factor))))
        target_w = max(1, int(round(float(w) * float(factor))))
        return target_h, target_w

    def _resize_video_samples_spatial(self, samples, factor):
        if getattr(samples, "is_nested", False):
            raise RuntimeError("当前 video latent 是 NestedTensor，暂不支持节点内空间缩放。")

        if samples.ndim == 4:
            _, _, h, w = samples.shape
            target_h, target_w = self._scale_hw(h, w, factor)
            if target_h == h and target_w == w:
                return samples, (h, w), (target_h, target_w)
            resized = F.interpolate(samples, size=(target_h, target_w), mode="bilinear", align_corners=False)
            return resized, (h, w), (target_h, target_w)

        if samples.ndim == 5:
            b, c, t, h, w = samples.shape
            target_h, target_w = self._scale_hw(h, w, factor)
            if target_h == h and target_w == w:
                return samples, (h, w), (target_h, target_w)

            flat = samples.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            flat = F.interpolate(flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            resized = flat.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()
            return resized, (h, w), (target_h, target_w)

        raise RuntimeError(f"不支持的视频 latent 维度：{tuple(samples.shape)}")

    def _resize_video_latent_spatial(self, video_latent, factor, reason):
        latent_copy = self._shallow_copy_latent(video_latent)
        if not isinstance(latent_copy, dict) or "samples" not in latent_copy:
            raise RuntimeError("video_latent 不是标准 LATENT dict，无法执行空间缩放。")

        samples = latent_copy["samples"]
        resized, before_hw, after_hw = self._resize_video_samples_spatial(samples, factor)
        latent_copy["samples"] = resized

        noise_mask = latent_copy.get("noise_mask", None)
        if torch.is_tensor(noise_mask):
            original_dtype = noise_mask.dtype
            original_device = noise_mask.device
            mask_float = noise_mask.float()
            resized_mask, _, _ = self._resize_video_samples_spatial(mask_float, factor)
            latent_copy["noise_mask"] = resized_mask.to(device=original_device, dtype=original_dtype)
        elif "noise_mask" in latent_copy and noise_mask is None:
            latent_copy.pop("noise_mask", None)

        self._log(
            "LATENT",
            f"{reason} video latent 空间缩放完成。",
            level="OK",
            icon="📐",
            factor=factor,
            before_hw=before_hw,
            after_hw=after_hw,
            tensor_shape=tuple(resized.shape),
            has_noise_mask=("noise_mask" in latent_copy),
        )
        return latent_copy

    def _release_vram(self, stage="", aggressive=False):
        mm = getattr(comfy, "model_management", None) if comfy is not None else None

        try:
            gc.collect()
        except Exception:
            pass

        if mm is not None:
            if aggressive:
                for fn_name in ("unload_all_models", "cleanup_models_gc"):
                    fn = getattr(mm, fn_name, None)
                    if callable(fn):
                        try:
                            fn()
                        except TypeError:
                            try:
                                fn([])
                            except Exception:
                                pass
                        except Exception:
                            pass

                free_memory = getattr(mm, "free_memory", None)
                if callable(free_memory) and torch.cuda.is_available():
                    try:
                        free_memory(1024 * 1024 * 1024, torch.device("cuda"), keep_loaded=[])
                    except TypeError:
                        try:
                            free_memory(1024 * 1024 * 1024, torch.device("cuda"))
                        except Exception:
                            pass
                    except Exception:
                        pass

            soft_empty_cache = getattr(mm, "soft_empty_cache", None)
            if callable(soft_empty_cache):
                try:
                    soft_empty_cache()
                except Exception:
                    pass

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

        verbose_vram = str(os.environ.get("NF_VERBOSE_VRAM", "")).strip().lower() in ("1", "true", "yes", "on")
        if stage and verbose_vram:
            mode = "强释放" if aggressive else "普通释放"
            self._log("VRAM", f"{stage}：已执行{mode}显存释放。", level="GC", icon="🧹")

    def _load_preset(self, 预设配置):
        preset_name = str(预设配置).strip()
        preset_path = os.path.join(self.PRESET_DIR, f"{preset_name}.json")
        if not os.path.isfile(preset_path):
            raise RuntimeError(f"未找到预设配置文件：{preset_path}")
        try:
            with open(preset_path, "r", encoding="utf-8") as f:
                preset = json.load(f)
        except Exception as e:
            raise RuntimeError(f"读取预设失败：{preset_path}\n{e}")
        if not isinstance(preset, dict):
            raise RuntimeError(f"预设文件内容必须是 JSON 对象：{preset_path}")
        return preset, preset_path

    def _apply_preset(self, 预设配置, 一采西格玛文本, 添加噪波, 启动二采, 二采使用放大模型, 二采放大模型, 二采西格玛):
        preset_name = str(预设配置).strip()

        if preset_name == "自定义":
            self._log("PRESET", "当前为自定义参数，跳过预设。", level="INFO", icon="🧭")
            return (
                一采西格玛文本,
                添加噪波,
                启动二采,
                二采使用放大模型,
                二采放大模型,
                二采西格玛,
                {},
            )

        preset, preset_path = self._load_preset(preset_name)
        self._log(
            "PRESET",
            "已加载预设参数。",
            level="INFO",
            icon="📦",
            preset=preset_name,
            file=os.path.basename(preset_path),
            enable_second_pass=preset.get("enable_second_pass"),
            use_second_upscale=preset.get("use_second_upscale"),
        )

        添加噪波 = bool(preset.get("add_noise", 添加噪波))
        启动二采 = bool(preset.get("enable_second_pass", 启动二采))
        二采使用放大模型 = bool(preset.get("use_second_upscale", 二采使用放大模型))
        二采放大模型 = str(preset.get("second_upscale_model", 二采放大模型))
        二采西格玛 = str(preset.get("second_sigmas_text", 二采西格玛))

        if bool(preset.get("override_first_sigmas", False)):
            一采西格玛文本 = str(preset.get("first_sigmas_text", 一采西格玛文本))

        return (
            一采西格玛文本,
            添加噪波,
            启动二采,
            二采使用放大模型,
            二采放大模型,
            二采西格玛,
            preset,
        )

    def _parse_sigmas_text(self, sigmas_text, field_name):
        if sigmas_text is None:
            raise RuntimeError(f"{field_name} 不能为空。")
        parts = [x.strip() for x in str(sigmas_text).split(",")]
        values = [float(x) for x in parts if x != ""]
        if not values:
            raise RuntimeError(f"{field_name} 解析失败，至少要有一个数值。")
        return torch.tensor(values, dtype=torch.float32)

    def _resolve_first_sigmas(self, incoming_sigmas, first_sigmas_text):
        if first_sigmas_text and str(first_sigmas_text).strip():
            return self._parse_sigmas_text(first_sigmas_text, "一采西格玛文本")
        if incoming_sigmas is not None:
            return incoming_sigmas
        raise RuntimeError("一采需要西格玛：要么填写“一采西格玛文本”，要么外接“西格玛”输入。")

    def _get_noise_obj(self, 添加噪波, 噪波种子):
        return _NanFengRandomNoise(噪波种子) if 添加噪波 else _NanFengEmptyNoise()

    def _resolve_second_pass_noise_obj(self, 添加噪波, 噪波种子, 二采噪波=None):
        if 二采噪波 is not None:
            if not hasattr(二采噪波, "generate_noise"):
                raise RuntimeError("二采噪波输入不是合法的 NOISE 对象，缺少 generate_noise 方法。")
            if not hasattr(二采噪波, "seed"):
                try:
                    二采噪波.seed = int(噪波种子)
                except Exception:
                    pass
            return 二采噪波, "external"
        return self._get_noise_obj(添加噪波, 噪波种子), ("seeded" if 添加噪波 else "empty")

    def _call_sampler_advanced(self, noise_obj, guider, sampler, sigmas, latent_image):
        if SamplerCustomAdvanced is None:
            raise RuntimeError("未能导入 comfy_extras.nodes_custom_sampler.SamplerCustomAdvanced，请确认 ComfyUI 自带高级采样节点可用。")

        sampler_node = SamplerCustomAdvanced()
        latent_copy = self._shallow_copy_latent(latent_image)

        fn_names = []
        function_name = getattr(sampler_node, "FUNCTION", None)
        if function_name:
            fn_names.append(function_name)
        fn_names.extend(["execute", "sample"])

        used = set()
        for fn_name in fn_names:
            if fn_name in used:
                continue
            used.add(fn_name)
            if hasattr(sampler_node, fn_name):
                result = getattr(sampler_node, fn_name)(
                    noise=noise_obj,
                    guider=guider,
                    sampler=sampler,
                    sigmas=sigmas,
                    latent_image=latent_copy,
                )
                return _normalize_node_output(result)

        raise RuntimeError("SamplerCustomAdvanced 未找到 execute/sample 可调用方法。")

    def _call_core_node(self, node_name, **kwargs):
        node_cls = CORE_NODE_CLASS_MAPPINGS.get(node_name)
        if node_cls is None:
            raise RuntimeError(f"未找到核心节点：{node_name}")

        node = node_cls()
        tried = []

        function_name = getattr(node, "FUNCTION", None) or getattr(node_cls, "FUNCTION", None)
        candidate_names = [function_name, "execute", "sample", "load", "load_model", "upscale", "separate", "concat"]

        for fn_name in candidate_names:
            if not fn_name or fn_name in tried:
                continue
            tried.append(fn_name)
            if hasattr(node, fn_name):
                result = getattr(node, fn_name)(**kwargs)
                return _normalize_node_output(result)

        raise RuntimeError(f"{node_name} 未找到可调用方法：{tried}")

    def _copy_guider_for_segment(self, guider):
        guider_copy = copy.copy(guider)
        if hasattr(guider, "original_conds"):
            try:
                guider_copy.original_conds = copy.deepcopy(guider.original_conds)
            except Exception:
                guider_copy.original_conds = guider.original_conds
        if hasattr(guider, "raw_conds"):
            try:
                guider_copy.raw_conds = copy.deepcopy(guider.raw_conds)
            except Exception:
                guider_copy.raw_conds = guider.raw_conds
        return guider_copy

    def _extract_raw_conds_from_guider(self, guider):
        if hasattr(guider, "raw_conds"):
            try:
                positive, negative = guider.raw_conds
                return copy.deepcopy(positive), copy.deepcopy(negative)
            except Exception:
                pass

        original_conds = getattr(guider, "original_conds", None)
        if not isinstance(original_conds, dict):
            raise RuntimeError("当前 guider 不包含 original_conds，无法注入官方式 latent guide。")
        if "negative" not in original_conds or "positive" not in original_conds:
            raise RuntimeError("当前 guider 不包含正负条件，无法注入官方式 latent guide。")

        raw_pos = original_conds["positive"]
        raw_neg = original_conds["negative"]
        positive = [[raw_pos[0]["cross_attn"], copy.deepcopy(raw_pos[0])]]
        negative = [[raw_neg[0]["cross_attn"], copy.deepcopy(raw_neg[0])]]
        return positive, negative

    def _set_guider_conds(self, guider, positive, negative):
        if hasattr(guider, "set_conds"):
            guider.set_conds(positive, negative)
        guider.raw_conds = (copy.deepcopy(positive), copy.deepcopy(negative))
        return guider

    def _get_guider_cfg(self, guider, default=1.0):
        try:
            cfg = getattr(guider, "cfg")
            if cfg is not None:
                return float(cfg)
        except Exception:
            pass
        return float(default)

    def _set_guider_cfg(self, guider, cfg_value):
        guider_copy = self._copy_guider_for_segment(guider)
        target_cfg = float(cfg_value)
        applied = False

        if hasattr(guider_copy, "set_cfg"):
            try:
                guider_copy.set_cfg(target_cfg)
                applied = True
            except Exception as e:
                self._log("CFG", f"guider.set_cfg 调用失败，改为直接写入 cfg 属性：{e}", level="WARN", icon="⚠️")

        if not applied:
            try:
                guider_copy.cfg = target_cfg
                applied = True
            except Exception:
                pass

        if not applied:
            self._log("CFG", "当前 guider 不支持 CFG 衰减写入，已保留原始 guider。", level="WARN", icon="⚠️", target_cfg=target_cfg)
            return guider_copy, self._get_guider_cfg(guider_copy, default=target_cfg)

        return guider_copy, self._get_guider_cfg(guider_copy, default=target_cfg)

    def _get_segment_cfg_decay_factor(self, segment_index):
        factors = tuple(float(x) for x in self.SEGMENT_CFG_DECAY_FACTORS)
        idx = max(1, int(segment_index))
        if idx <= len(factors):
            return factors[idx - 1]
        return factors[-1]

    def _build_segment_cfg_guider(self, guider, segment_index, segment_total, disable_decay=False, disable_reason=""):
        base_cfg = self._get_guider_cfg(guider, default=1.0)
        if disable_decay:
            decay_factor = 1.0
            target_cfg = float(base_cfg)
        else:
            decay_factor = self._get_segment_cfg_decay_factor(segment_index)
            target_cfg = max(0.0, float(base_cfg) * float(decay_factor))
        guider_copy, actual_cfg = self._set_guider_cfg(guider, target_cfg)
        if int(segment_index) == 1:
            if disable_decay:
                message = (
                    "单采分段续写，CFG 保持原值。"
                    if str(disable_reason) == "single_pass_continuity"
                    else "检测到按段提示词，CFG 保持原值。"
                )
            else:
                message = "已启用分段 CFG 衰减。"
            self._log(
                "CFG",
                message,
                level="INFO",
                icon="🎚️",
                segment=f"{int(segment_index)}/{int(segment_total)}",
                base_cfg=base_cfg,
                decay_factor=decay_factor,
                effective_cfg=actual_cfg,
            )
        return guider_copy

    def _split_segment_prompt_blocks(self, prompt_text):
        text = "" if prompt_text is None else str(prompt_text)
        if not text.strip():
            return []
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]

    def _encode_segment_prompt_text(self, clip, prompt_text, field_name):
        text = "" if prompt_text is None else str(prompt_text).strip()
        if not text:
            return None
        if clip is None:
            raise RuntimeError(f"开启空行分段提示词后，需要连接“{field_name}”对应的 CLIP（“分段提示词CLIP”）。")
        tokens = clip.tokenize(text)
        return clip.encode_from_tokens_scheduled(tokens)

    def _merge_conditioning_template(self, template_conds, replacement_conds):
        if replacement_conds is None:
            return None

        try:
            replacement_items = copy.deepcopy(replacement_conds)
        except Exception:
            replacement_items = replacement_conds

        if not template_conds:
            return replacement_items

        merged = []
        template_total = len(template_conds)
        for idx, item in enumerate(replacement_items):
            template_item = template_conds[min(idx, template_total - 1)] if template_total > 0 else None

            template_meta = {}
            if isinstance(template_item, (list, tuple)) and len(template_item) > 1 and isinstance(template_item[1], dict):
                try:
                    template_meta = copy.deepcopy(template_item[1])
                except Exception:
                    template_meta = dict(template_item[1])

            replacement_meta = {}
            if isinstance(item, (list, tuple)) and len(item) > 1 and isinstance(item[1], dict):
                try:
                    replacement_meta = copy.deepcopy(item[1])
                except Exception:
                    replacement_meta = dict(item[1])

            template_meta.update(replacement_meta)
            embedding = item[0] if isinstance(item, (list, tuple)) else item
            merged.append([embedding, template_meta])
        return merged

    def _select_segment_conditioning(self, conditioning_blocks, segment_index):
        if not conditioning_blocks:
            return None, None, False

        raw_idx = max(0, int(segment_index) - 1)
        resolved_idx = min(raw_idx, len(conditioning_blocks) - 1)
        try:
            selected = copy.deepcopy(conditioning_blocks[resolved_idx])
        except Exception:
            selected = conditioning_blocks[resolved_idx]
        return selected, resolved_idx, (resolved_idx != raw_idx)

    def _workflow_node_type(self, node):
        if not isinstance(node, dict):
            return ""
        return str(node.get("type") or node.get("class_type") or "")

    def _workflow_widget_value(self, node, index=0, default=None):
        if not isinstance(node, dict):
            return default
        widgets = node.get("widgets_values")
        if isinstance(widgets, list) and 0 <= int(index) < len(widgets):
            return widgets[int(index)]
        return default

    def _workflow_input_by_name(self, node, input_name):
        for item in node.get("inputs") or []:
            if not isinstance(item, dict):
                continue
            if item.get("name") == input_name or item.get("label") == input_name:
                return item
        return None

    def _workflow_input_link(self, node, input_name):
        item = self._workflow_input_by_name(node, input_name)
        if not isinstance(item, dict):
            return None
        link = item.get("link")
        try:
            return int(link) if link is not None else None
        except Exception:
            return None

    def _build_workflow_lookup(self, workflow):
        nodes_by_id = {}
        links_by_id = {}
        if not isinstance(workflow, dict):
            return {"nodes_by_id": nodes_by_id, "links_by_id": links_by_id}

        for node in workflow.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            try:
                node_id = int(node.get("id"))
            except Exception:
                continue
            nodes_by_id[node_id] = node

        for link in workflow.get("links") or []:
            if not isinstance(link, (list, tuple)) or len(link) < 5:
                continue
            try:
                link_id = int(link[0])
            except Exception:
                continue
            links_by_id[link_id] = link

        return {"nodes_by_id": nodes_by_id, "links_by_id": links_by_id}

    def _resolve_workflow_source_node_id(self, workflow_ctx, link_id):
        try:
            link_key = int(link_id)
        except Exception:
            return None
        link = workflow_ctx.get("links_by_id", {}).get(link_key)
        if not isinstance(link, (list, tuple)) or len(link) < 2:
            return None
        try:
            return int(link[1])
        except Exception:
            return None

    def _find_matching_set_node(self, workflow_ctx, bus_name):
        if not bus_name:
            return None
        for node in workflow_ctx.get("nodes_by_id", {}).values():
            if self._workflow_node_type(node) != "SetNode":
                continue
            if str(self._workflow_widget_value(node, 0, "")) == str(bus_name):
                return node
        return None

    def _resolve_workflow_passthrough_node(self, workflow_ctx, node_id, seen=None):
        if seen is None:
            seen = set()
        try:
            node_key = int(node_id)
        except Exception:
            return None
        if node_key in seen:
            return workflow_ctx.get("nodes_by_id", {}).get(node_key)
        seen.add(node_key)

        node = workflow_ctx.get("nodes_by_id", {}).get(node_key)
        if not isinstance(node, dict):
            return None

        node_type = self._workflow_node_type(node)
        if node_type == "GetNode":
            set_node = self._find_matching_set_node(workflow_ctx, self._workflow_widget_value(node, 0, ""))
            if set_node is None:
                return node
            return self._resolve_workflow_passthrough_node(workflow_ctx, set_node.get("id"), seen=seen)

        if node_type == "SetNode":
            for item in node.get("inputs") or []:
                if not isinstance(item, dict):
                    continue
                link = item.get("link")
                if link is None:
                    continue
                source_node_id = self._resolve_workflow_source_node_id(workflow_ctx, link)
                if source_node_id is None:
                    continue
                return self._resolve_workflow_passthrough_node(workflow_ctx, source_node_id, seen=seen)
            return node

        if node_type == "ShowText|pysssss":
            text_link = self._workflow_input_link(node, "text")
            if text_link is not None:
                source_node_id = self._resolve_workflow_source_node_id(workflow_ctx, text_link)
                if source_node_id is not None:
                    return self._resolve_workflow_passthrough_node(workflow_ctx, source_node_id, seen=seen)
        return node

    def _resolve_workflow_string_value(self, workflow_ctx, node_id, seen=None):
        if seen is None:
            seen = set()
        try:
            node_key = int(node_id)
        except Exception:
            return ""
        if node_key in seen:
            return ""
        seen.add(node_key)

        node = workflow_ctx.get("nodes_by_id", {}).get(node_key)
        if not isinstance(node, dict):
            return ""

        node_type = self._workflow_node_type(node)
        if node_type == "GetNode":
            set_node = self._find_matching_set_node(workflow_ctx, self._workflow_widget_value(node, 0, ""))
            if set_node is None:
                return ""
            return self._resolve_workflow_string_value(workflow_ctx, set_node.get("id"), seen=seen)

        if node_type == "SetNode":
            for item in node.get("inputs") or []:
                if not isinstance(item, dict):
                    continue
                link = item.get("link")
                if link is None:
                    continue
                source_node_id = self._resolve_workflow_source_node_id(workflow_ctx, link)
                if source_node_id is None:
                    continue
                return self._resolve_workflow_string_value(workflow_ctx, source_node_id, seen=seen)
            return str(self._workflow_widget_value(node, 0, "") or "")

        if node_type == "JoinStrings":
            delimiter = str(self._workflow_widget_value(node, 0, "") or "")
            parts = []
            for input_name in ("string1", "string2"):
                link = self._workflow_input_link(node, input_name)
                if link is None:
                    continue
                source_node_id = self._resolve_workflow_source_node_id(workflow_ctx, link)
                if source_node_id is None:
                    continue
                part = self._resolve_workflow_string_value(workflow_ctx, source_node_id, seen=seen.copy())
                if part:
                    parts.append(str(part))
            return delimiter.join(parts)

        if node_type == "ShowText|pysssss":
            text_link = self._workflow_input_link(node, "text")
            if text_link is not None:
                source_node_id = self._resolve_workflow_source_node_id(workflow_ctx, text_link)
                if source_node_id is not None:
                    return self._resolve_workflow_string_value(workflow_ctx, source_node_id, seen=seen)
            return str(self._workflow_widget_value(node, 0, "") or "")

        if node_type in {"CR Text", "TextBox", "Text Multiline"}:
            return str(self._workflow_widget_value(node, 0, "") or "")

        text_link = self._workflow_input_link(node, "text")
        if text_link is not None:
            source_node_id = self._resolve_workflow_source_node_id(workflow_ctx, text_link)
            if source_node_id is not None:
                text_value = self._resolve_workflow_string_value(workflow_ctx, source_node_id, seen=seen)
                if text_value:
                    return text_value

        widget_value = self._workflow_widget_value(node, 0, None)
        return str(widget_value) if isinstance(widget_value, str) else ""

    def _resolve_conditioning_prompt_text(self, workflow_ctx, node_id, side="positive", seen=None):
        if seen is None:
            seen = set()

        node = self._resolve_workflow_passthrough_node(workflow_ctx, node_id, seen=seen.copy())
        if not isinstance(node, dict):
            return ""

        try:
            node_key = int(node.get("id"))
        except Exception:
            return ""
        if node_key in seen:
            return ""
        seen.add(node_key)

        node_type = self._workflow_node_type(node)
        if node_type == "CLIPTextEncode":
            widget_text = str(self._workflow_widget_value(node, 0, "") or "")
            if widget_text.strip():
                return widget_text
            text_link = self._workflow_input_link(node, "text")
            if text_link is not None:
                source_node_id = self._resolve_workflow_source_node_id(workflow_ctx, text_link)
                if source_node_id is not None:
                    resolved_text = self._resolve_workflow_string_value(workflow_ctx, source_node_id)
                    if resolved_text:
                        return resolved_text
            return widget_text

        if node_type == "LTXVConditioning":
            input_name = "positive" if str(side) != "negative" else "negative"
            cond_link = self._workflow_input_link(node, input_name)
            if cond_link is None:
                return ""
            source_node_id = self._resolve_workflow_source_node_id(workflow_ctx, cond_link)
            if source_node_id is None:
                return ""
            return self._resolve_conditioning_prompt_text(workflow_ctx, source_node_id, side=side, seen=seen)

        if node_type == "ConditioningZeroOut":
            return ""

        cond_link = self._workflow_input_link(node, "conditioning")
        if cond_link is not None:
            source_node_id = self._resolve_workflow_source_node_id(workflow_ctx, cond_link)
            if source_node_id is not None:
                return self._resolve_conditioning_prompt_text(workflow_ctx, source_node_id, side=side, seen=seen)
        return ""

    def _extract_prompt_texts_from_workflow(self, workflow, unique_id):
        try:
            sampler_node_id = int(unique_id)
        except Exception:
            return {"positive": "", "negative": ""}

        workflow_ctx = self._build_workflow_lookup(workflow)
        sampler_node = workflow_ctx.get("nodes_by_id", {}).get(sampler_node_id)
        if not isinstance(sampler_node, dict):
            return {"positive": "", "negative": ""}

        guider_links = []
        for input_name in ("引导器", "二采引导器"):
            link_id = self._workflow_input_link(sampler_node, input_name)
            if link_id is not None:
                guider_links.append(link_id)

        result = {"positive": "", "negative": ""}
        for link_id in guider_links:
            source_node_id = self._resolve_workflow_source_node_id(workflow_ctx, link_id)
            guider_node = self._resolve_workflow_passthrough_node(workflow_ctx, source_node_id)
            if not isinstance(guider_node, dict):
                continue

            guider_type = self._workflow_node_type(guider_node)
            if "Guider" not in guider_type:
                continue

            if not result["positive"]:
                positive_link = self._workflow_input_link(guider_node, "positive")
                if positive_link is not None:
                    positive_node_id = self._resolve_workflow_source_node_id(workflow_ctx, positive_link)
                    result["positive"] = self._resolve_conditioning_prompt_text(
                        workflow_ctx,
                        positive_node_id,
                        side="positive",
                    )

            if not result["negative"]:
                negative_link = self._workflow_input_link(guider_node, "negative")
                if negative_link is not None:
                    negative_node_id = self._resolve_workflow_source_node_id(workflow_ctx, negative_link)
                    result["negative"] = self._resolve_conditioning_prompt_text(
                        workflow_ctx,
                        negative_node_id,
                        side="negative",
                    )

            if result["positive"] and result["negative"]:
                break

        return result

    def _resolve_segment_prompt_texts(
        self,
        启用空行分段提示词=False,
        分段正向提示词文本="",
        分段负向提示词文本="",
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        resolved_positive = "" if 分段正向提示词文本 is None else str(分段正向提示词文本)
        resolved_negative = "" if 分段负向提示词文本 is None else str(分段负向提示词文本)
        info = {
            "positive_from_workflow": False,
            "negative_from_workflow": False,
        }

        if bool(启用空行分段提示词):
            return resolved_positive, resolved_negative, info

        workflow = None
        if isinstance(extra_pnginfo, dict):
            workflow = extra_pnginfo.get("workflow")
        if workflow is None and isinstance(prompt, dict):
            workflow = prompt.get("workflow")
        if not isinstance(workflow, dict):
            return resolved_positive, resolved_negative, info

        extracted = self._extract_prompt_texts_from_workflow(workflow, unique_id)
        extracted_positive = str(extracted.get("positive") or "")
        extracted_negative = str(extracted.get("negative") or "")

        if not resolved_positive.strip() and extracted_positive.strip():
            resolved_positive = extracted_positive
            info["positive_from_workflow"] = True
        if not resolved_negative.strip() and extracted_negative.strip():
            resolved_negative = extracted_negative
            info["negative_from_workflow"] = True

        return resolved_positive, resolved_negative, info

    def _build_segment_prompt_context(
        self,
        segment_total,
        启用空行分段提示词=False,
        分段提示词CLIP=None,
        分段正向提示词文本="",
        分段负向提示词文本="",
    ):
        if not bool(启用空行分段提示词):
            return None

        positive_blocks = self._split_segment_prompt_blocks(分段正向提示词文本)
        negative_blocks = self._split_segment_prompt_blocks(分段负向提示词文本)
        if not positive_blocks and not negative_blocks:
            self._log(
                "PROMPT",
                "空行分段提示词已开启，但文本内容为空，当前回退到原 guider 条件。",
                level="WARN",
                icon="⚠️",
            )
            return None

        if 分段提示词CLIP is None:
            raise RuntimeError("开启“按空行分段提示词”后，需要连接“分段提示词CLIP”，用于给每段重新编码提示词。")

        context = {
            "positive_blocks": [],
            "negative_blocks": [],
        }
        for text in positive_blocks:
            context["positive_blocks"].append(self._encode_segment_prompt_text(分段提示词CLIP, text, "分段正向提示词文本"))
        for text in negative_blocks:
            context["negative_blocks"].append(self._encode_segment_prompt_text(分段提示词CLIP, text, "分段负向提示词文本"))

        self._log(
            "PROMPT",
            "已构建按空行分段的提示词条件。",
            level="INFO",
            icon="🧠",
            segment_total=int(segment_total),
            positive_segments=len(context["positive_blocks"]),
            negative_segments=len(context["negative_blocks"]),
        )

        if context["positive_blocks"] and len(context["positive_blocks"]) != int(segment_total):
            self._log(
                "PROMPT",
                "正向提示词段数与分段数不一致，超出部分会忽略，不足部分会重用最后一段。",
                level="WARN",
                icon="⚠️",
                prompt_segments=len(context["positive_blocks"]),
                segment_total=int(segment_total),
            )
        if context["negative_blocks"] and len(context["negative_blocks"]) != int(segment_total):
            self._log(
                "PROMPT",
                "负向提示词段数与分段数不一致，超出部分会忽略，不足部分会重用最后一段。",
                level="WARN",
                icon="⚠️",
                prompt_segments=len(context["negative_blocks"]),
                segment_total=int(segment_total),
            )

        return context

    def _apply_segment_prompt_conds(self, guider, segment_index, segment_total, segment_prompt_context=None, guider_role="first_pass"):
        if not segment_prompt_context:
            return guider, False

        base_positive, base_negative = self._extract_raw_conds_from_guider(guider)
        positive_cond, positive_idx, positive_reused = self._select_segment_conditioning(
            segment_prompt_context.get("positive_blocks"),
            segment_index,
        )
        negative_cond, negative_idx, negative_reused = self._select_segment_conditioning(
            segment_prompt_context.get("negative_blocks"),
            segment_index,
        )

        positive = self._merge_conditioning_template(base_positive, positive_cond) if positive_cond is not None else base_positive
        negative = self._merge_conditioning_template(base_negative, negative_cond) if negative_cond is not None else base_negative

        guider_copy = self._copy_guider_for_segment(guider)
        guider_copy = self._set_guider_conds(guider_copy, positive, negative)
        self._log(
            "PROMPT",
            "已将空行分段提示词应用到当前分段 guider。",
            level="INFO",
            icon="🧩",
            guider_role=guider_role,
            segment=f"{int(segment_index)}/{int(segment_total)}",
            positive_segment=("base" if positive_idx is None else int(positive_idx + 1)),
            negative_segment=("base" if negative_idx is None else int(negative_idx + 1)),
            positive_reused=bool(positive_reused),
            negative_reused=bool(negative_reused),
        )
        return guider_copy, True

    def _build_segment_guider(self, guider, segment_index, segment_total, segment_prompt_context=None, guider_role="first_pass", disable_cfg_decay=False, disable_cfg_decay_reason=""):
        prompt_ready_guider, prompt_overridden = self._apply_segment_prompt_conds(
            guider=guider,
            segment_index=segment_index,
            segment_total=segment_total,
            segment_prompt_context=segment_prompt_context,
            guider_role=guider_role,
        )
        disable_decay = bool(prompt_overridden or disable_cfg_decay)
        if prompt_overridden:
            disable_reason = "prompt_override"
        elif disable_cfg_decay:
            disable_reason = disable_cfg_decay_reason
        else:
            disable_reason = ""
        return self._build_segment_cfg_guider(
            prompt_ready_guider,
            segment_index,
            segment_total,
            disable_decay=disable_decay,
            disable_reason=disable_reason,
        )

    def _resolve_second_pass_route(self, 引导器, 采样器, 二采引导器=None, 二采采样器=None):
        second_pass_guider = 二采引导器 if 二采引导器 is not None else 引导器
        second_pass_sampler = 二采采样器 if 二采采样器 is not None else 采样器
        guider_source = "二采引导器" if 二采引导器 is not None else "引导器"
        sampler_source = "二采采样器" if 二采采样器 is not None else "采样器"
        return second_pass_guider, second_pass_sampler, guider_source, sampler_source

    def _resolve_continuation_vae(self, 续写VAE=None, 二采VAE=None):
        if 续写VAE is not None:
            return 续写VAE
        if 二采VAE is not None:
            return 二采VAE
        return None

    def _supports_official_style_continuation(self, guide_vae):
        if guide_vae is None:
            return False, "未连接续写VAE/二采VAE"
        if "LTXVAddLatentGuide" not in CORE_NODE_CLASS_MAPPINGS:
            return False, "未加载 LTXVAddLatentGuide 节点"
        return True, "ok"

    def _resolve_img_to_video_vae(self, 二采VAE=None, 续写VAE=None):
        return self._resolve_continuation_vae(续写VAE=续写VAE, 二采VAE=二采VAE)

    def _resolve_second_pass_img_to_video_vae(self, 二采VAE=None, 续写VAE=None):
        return self._resolve_img_to_video_vae(二采VAE=二采VAE, 续写VAE=续写VAE)

    def _apply_img_to_video_inplace(self, video_latent, i2v_image, i2v_vae, strength=1.0, stage_label="", mode="first_pass"):
        mode = "second_pass" if str(mode).lower() == "second_pass" else "first_pass"
        if mode == "second_pass":
            missing_image_message = "二采图生视频已开启，但未连接“二采latent输入图像”，当前回退到普通二采前处理。"
            success_message = "已在二采前处理插入 LTXVImgToVideoInplace。"
            stage_desc = "二采前处理"
            reason_tag = "second_pass"
        else:
            missing_image_message = "图生视频已开启，但未连接“二采latent输入图像”，当前回退到普通起始 latent。"
            success_message = "已在一采起始 latent 插入 LTXVImgToVideoInplace。"
            stage_desc = "一采起始 latent"
            reason_tag = "first_pass"

        if i2v_image is None:
            self._log(
                "I2V",
                missing_image_message,
                level="WARN",
                icon="⚠️",
                i2v_stage=(stage_label or "default"),
            )
            return video_latent

        if i2v_vae is None:
            raise RuntimeError(f"开启“是否图生视频”后，需要连接“二采VAE”或“续写VAE”，因为{stage_desc}要调用 LTXVImgToVideoInplace。")
        if "LTXVImgToVideoInplace" not in CORE_NODE_CLASS_MAPPINGS:
            raise RuntimeError("未找到核心节点：LTXVImgToVideoInplace")

        try:
            latent_for_i2v = copy.deepcopy(video_latent)
        except Exception:
            latent_for_i2v = {
                "samples": self._get_samples(video_latent).clone(),
            }
            mask = self._get_noise_mask(video_latent)
            if torch.is_tensor(mask):
                latent_for_i2v["noise_mask"] = mask.clone()

        i2v_result = self._call_core_node(
            "LTXVImgToVideoInplace",
            vae=i2v_vae,
            image=i2v_image,
            latent=latent_for_i2v,
            strength=float(strength),
            bypass=False,
        )[0]
        i2v_result = self._sanitize_latent_noise_mask(i2v_result, prefer_video=True, reason=f"{reason_tag}_img_to_video_inplace")
        self._log(
            "I2V",
            success_message,
            level="OK",
            icon="🖼️",
            i2v_stage=(stage_label or "default"),
            strength=float(strength),
            output_shape=self._latent_shape_text(i2v_result),
        )
        return i2v_result

    def _apply_second_pass_img_to_video_inplace(self, video_latent, i2v_image, i2v_vae, strength=1.0, stage_label=""):
        return self._apply_img_to_video_inplace(
            video_latent=video_latent,
            i2v_image=i2v_image,
            i2v_vae=i2v_vae,
            strength=strength,
            stage_label=stage_label,
            mode="second_pass",
        )

    def _prepare_first_pass_img_to_video_input_latent(self, latent_input, i2v_image, i2v_vae, stage_label=""):
        source_video_latent, source_audio_latent = self._try_split_av_latent(latent_input)
        has_audio = source_video_latent is not None and source_audio_latent is not None
        base_video_latent = source_video_latent if has_audio else latent_input
        seeded_video_latent = self._apply_img_to_video_inplace(
            video_latent=base_video_latent,
            i2v_image=i2v_image,
            i2v_vae=i2v_vae,
            strength=1.0,
            stage_label=stage_label,
            mode="first_pass",
        )
        if not has_audio:
            return self._sanitize_latent_noise_mask(seeded_video_latent, prefer_video=True, reason="first_pass_img_to_video_prepare")
        return self._rebuild_from_parts(seeded_video_latent, source_audio_latent)

    def _apply_official_style_video_guide(self, guider, video_latent, guide_video_latent, guide_vae, strength, segment_label=""):
        guider_copy = self._copy_guider_for_segment(guider)
        positive, negative = self._extract_raw_conds_from_guider(guider_copy)

        base_video_latent = self._sanitize_latent_noise_mask(video_latent, prefer_video=True, reason="official_style_video_guide_input")
        base_guide_video_latent = self._sanitize_latent_noise_mask(guide_video_latent, prefer_video=True, reason="official_style_video_guide_source")

        # 官方 LTXVAddLatentGuide 会在内部自己构造并 append keyframe mask。
        # 如果这里传进去的 latent 已经带了按通道展开的 noise_mask（常见为 C=128），
        # 而官方节点内部新建的是单通道 mask（常见为 C=1），底层 torch.cat(dim=2)
        # 就会因为非时间维不一致直接报：Expected size 128 but got size 1。
        # 所以官方 guide 路径里，这里主动把旧 noise_mask 清掉，只保留 latent 本体和 guide 本体。
        input_had_mask = torch.is_tensor(self._get_noise_mask(base_video_latent))
        guide_had_mask = torch.is_tensor(self._get_noise_mask(base_guide_video_latent))
        if input_had_mask or guide_had_mask:
            self._log(
                "MASK",
                "官方 guide 注入前检测到已有 noise_mask；为避免官方节点内部 keyframe mask 拼接维度冲突，已自动移除旧 mask。",
                level="WARN",
                icon="⚠️",
                segment=segment_label,
                input_mask_shape=(tuple(self._get_noise_mask(base_video_latent).shape) if input_had_mask else None),
                guide_mask_shape=(tuple(self._get_noise_mask(base_guide_video_latent).shape) if guide_had_mask else None),
            )
            base_video_latent = self._drop_noise_mask(base_video_latent)
            base_guide_video_latent = self._drop_noise_mask(base_guide_video_latent)

        try:
            positive, negative, guided_video_latent = self._call_core_node(
                "LTXVAddLatentGuide",
                vae=guide_vae,
                positive=positive,
                negative=negative,
                latent=self._shallow_copy_latent(base_video_latent),
                guiding_latent=self._shallow_copy_latent(base_guide_video_latent),
                latent_idx=0,
                strength=float(strength),
            )
        except RuntimeError as e:
            msg = str(e)
            suspicious = (
                "Expected size 128 but got size 1" in msg
                or "Sizes of tensors must match except in dimension 2" in msg
                or "append_keyframe" in msg
            )
            if not suspicious:
                raise

            self._log(
                "MASK",
                "官方 guide 注入时仍检测到疑似 noise_mask 维度冲突，已触发一次兜底重试：强制移除输入与 guide 的旧 mask 后再次注入。",
                level="WARN",
                icon="⚠️",
                segment=segment_label,
                error=msg,
            )
            positive, negative = self._extract_raw_conds_from_guider(guider_copy)
            positive, negative, guided_video_latent = self._call_core_node(
                "LTXVAddLatentGuide",
                vae=guide_vae,
                positive=positive,
                negative=negative,
                latent=self._drop_noise_mask(self._shallow_copy_latent(base_video_latent)),
                guiding_latent=self._drop_noise_mask(self._shallow_copy_latent(base_guide_video_latent)),
                latent_idx=0,
                strength=float(strength),
            )

        guided_video_latent = self._sanitize_latent_noise_mask(guided_video_latent, prefer_video=True, reason="official_style_video_guide")
        guider_copy = self._set_guider_conds(guider_copy, positive, negative)
        self._log(
            "SEG",
            "本段续写：官方 guide。",
            level="OK",
            icon="🔗",
            segment=segment_label,
            guide_strength=strength,
            overlap=self._get_temporal_length(guide_video_latent, prefer_video=True),
        )
        return guider_copy, guided_video_latent

    def _linear_overlap_blend(self, latent_a, latent_b, overlap_length, prefer_video=False):
        overlap_length = max(0, int(overlap_length))
        if latent_a is None:
            return self._shallow_copy_latent(latent_b)
        if latent_b is None:
            return self._shallow_copy_latent(latent_a)
        if overlap_length <= 0:
            return self._concat_latent_temporal([latent_a, latent_b], prefer_video=prefer_video)

        total_a = self._get_temporal_length(latent_a, prefer_video=prefer_video)
        total_b = self._get_temporal_length(latent_b, prefer_video=prefer_video)
        overlap_length = min(overlap_length, total_a, total_b)
        if overlap_length <= 0:
            return self._concat_latent_temporal([latent_a, latent_b], prefer_video=prefer_video)

        for candidate in ("LTXVLinearOverlapLatentTransition", "LinearOverlapLatentTransition"):
            try:
                if candidate in CORE_NODE_CLASS_MAPPINGS:
                    axis = self._get_temporal_axis(latent_a, prefer_video=prefer_video)
                    merged = self._call_core_node(
                        candidate,
                        samples1=latent_a,
                        samples2=latent_b,
                        overlap=overlap_length,
                        axis=axis,
                    )[0]
                    # 核心节点通常不保留 noise_mask，这里手动补回来
                    mask_a = self._get_noise_mask(latent_a)
                    mask_b = self._get_noise_mask(latent_b)
                    if torch.is_tensor(mask_a) or torch.is_tensor(mask_b):
                        merged_mask = self._linear_overlap_blend_masks(mask_a, mask_b, overlap_length, axis, merged["samples"])
                        if merged_mask is not None:
                            merged["noise_mask"] = merged_mask
                    return merged
            except Exception as e:
                self._log("SEG", f"核心线性融合节点 {candidate} 调用失败，回退到本地融合：{e}", level="WARN", icon="⚠️")

        axis = self._get_temporal_axis(latent_a, prefer_video=prefer_video)
        samples_a = self._get_samples(latent_a)
        samples_b = self._get_samples(latent_b).to(samples_a.device)

        alpha = torch.linspace(1.0, 0.0, overlap_length + 2, device=samples_a.device, dtype=samples_a.dtype)[1:-1]
        shape = [1] * samples_a.ndim
        shape[axis] = overlap_length
        alpha = alpha.reshape(shape)

        slicer = [slice(None)] * samples_a.ndim
        slicer_a_rest = slicer.copy(); slicer_a_rest[axis] = slice(None, -overlap_length)
        slicer_a_overlap = slicer.copy(); slicer_a_overlap[axis] = slice(-overlap_length, None)
        slicer_b_overlap = slicer.copy(); slicer_b_overlap[axis] = slice(0, overlap_length)
        slicer_b_rest = slicer.copy(); slicer_b_rest[axis] = slice(overlap_length, None)

        merged_samples = torch.cat([
            samples_a[tuple(slicer_a_rest)],
            alpha * samples_a[tuple(slicer_a_overlap)] + (1.0 - alpha) * samples_b[tuple(slicer_b_overlap)],
            samples_b[tuple(slicer_b_rest)],
        ], dim=axis).contiguous()

        merged_mask = self._linear_overlap_blend_masks(
            self._get_noise_mask(latent_a),
            self._get_noise_mask(latent_b),
            overlap_length,
            axis,
            merged_samples,
        )
        return self._clone_like(latent_a, merged_samples, noise_mask=merged_mask)

    def _linear_overlap_blend_masks(self, mask_a, mask_b, overlap_length, axis, reference_samples):
        mask_a = self._sanitize_noise_mask_tensor(mask_a)
        mask_b = self._sanitize_noise_mask_tensor(mask_b)
        if mask_a is None and mask_b is None:
            return None

        ref_t = int(reference_samples.shape[axis])
        axis_a = self._infer_noise_mask_temporal_axis(mask_a, ref_t) if mask_a is not None else None
        axis_b = self._infer_noise_mask_temporal_axis(mask_b, ref_t) if mask_b is not None else None

        # 静态空间 mask：不做时间混合，能对齐就直接保留一个。
        if axis_a is None and axis_b is None:
            if mask_a is not None and mask_b is not None:
                if tuple(mask_a.shape) == tuple(mask_b.shape):
                    return mask_a.to(device=reference_samples.device, dtype=reference_samples.dtype).contiguous()
                self._log("MASK", "线性融合时检测到静态 noise_mask 形状不一致，已自动移除 mask。", level="WARN", icon="⚠️")
                return None
            keep_mask = mask_a if mask_a is not None else mask_b
            return keep_mask.to(device=reference_samples.device, dtype=reference_samples.dtype).contiguous()

        if mask_a is None or mask_b is None or axis_a is None or axis_b is None or axis_a != axis_b:
            self._log("MASK", "线性融合时 noise_mask 结构不兼容，已自动移除 mask。", level="WARN", icon="⚠️")
            return None

        mask_a = mask_a.to(device=reference_samples.device, dtype=reference_samples.dtype)
        mask_b = mask_b.to(device=reference_samples.device, dtype=reference_samples.dtype)
        mask_axis = axis_a

        if not self._same_shape_except_axis(tuple(mask_a.shape), tuple(mask_b.shape), mask_axis):
            self._log("MASK", "线性融合时 noise_mask 除时间轴外形状不一致，已自动移除 mask。", level="WARN", icon="⚠️")
            return None

        overlap_length = max(0, min(int(overlap_length), int(mask_a.shape[mask_axis]), int(mask_b.shape[mask_axis])))
        if overlap_length <= 0:
            return None

        alpha = torch.linspace(1.0, 0.0, overlap_length + 2, device=reference_samples.device, dtype=reference_samples.dtype)[1:-1]
        shape = [1] * mask_a.ndim
        shape[mask_axis] = overlap_length
        alpha = alpha.reshape(shape)

        slicer = [slice(None)] * mask_a.ndim
        slicer_a_rest = slicer.copy(); slicer_a_rest[mask_axis] = slice(None, -overlap_length)
        slicer_a_overlap = slicer.copy(); slicer_a_overlap[mask_axis] = slice(-overlap_length, None)
        slicer_b_overlap = slicer.copy(); slicer_b_overlap[mask_axis] = slice(0, overlap_length)
        slicer_b_rest = slicer.copy(); slicer_b_rest[mask_axis] = slice(overlap_length, None)

        merged_mask = torch.cat([
            mask_a[tuple(slicer_a_rest)],
            alpha * mask_a[tuple(slicer_a_overlap)] + (1.0 - alpha) * mask_b[tuple(slicer_b_overlap)],
            mask_b[tuple(slicer_b_rest)],
        ], dim=mask_axis).contiguous()
        return self._sanitize_noise_mask_tensor(merged_mask)

    def _prepare_first_pass_input_for_x2_pipeline(self, latent_image):
        self._log("FLOW", "检测到“启动二采 + 启用x2放大模型”，一采前将 video latent 预缩放到 0.5。", level="INFO", icon="🧭")
        try:
            av_latent = self._shallow_copy_latent(latent_image)
            separate_out = self._call_core_node("LTXVSeparateAVLatent", av_latent=av_latent)
            video_latent, audio_latent = separate_out[:2]
            del separate_out
            del av_latent

            half_video_latent = self._resize_video_latent_spatial(video_latent, factor=0.5, reason="一采前预缩放")
            rebuilt_latent = self._call_core_node("LTXVConcatAVLatent", video_latent=half_video_latent, audio_latent=audio_latent)[0]

            self._log(
                "FLOW",
                "一采输入 AV latent 已重组完成，将以半分辨率进入一采。",
                level="OK",
                icon="✅",
                output_shape=self._latent_shape_text(rebuilt_latent),
            )
            return rebuilt_latent
        except Exception as e:
            self._log("FLOW", "AV latent 拆分预缩放失败，尝试直接对输入 latent 做 0.5 空间缩放。", level="WARN", icon="⚠️", error=str(e))
            return self._resize_video_latent_spatial(latent_image, factor=0.5, reason="一采前直接预缩放")

    def _prepare_second_pass_input_latent(self, first_pass_output, 二采使用放大模型, 二采放大模型, 二采VAE=None, stage_label="", 是否图生视频=False, 二采latent输入图像=None, 二采图生视频VAE=None, 应用二采图生视频=False):
        av_latent = self._shallow_copy_latent(first_pass_output)
        del first_pass_output
        self._release_vram("一采输出已转入二采前处理", aggressive=True)

        separate_out = self._call_core_node("LTXVSeparateAVLatent", av_latent=av_latent)
        del av_latent
        self._release_vram("AV latent 拆分完成", aggressive=True)

        video_latent = None
        audio_latent = None
        second_video_latent = None

        video_latent, audio_latent = separate_out[:2]
        del separate_out

        second_video_latent = video_latent

        if 二采使用放大模型:
            if 二采VAE is None:
                raise RuntimeError("开启“二采使用放大模型”时需要连接“二采VAE”。")

            if not 二采放大模型:
                二采放大模型 = self.DEFAULT_X2_MODEL

            self._log(
                "UPSCALE",
                "开始执行二采前 x2 latent 放大。",
                level="INFO",
                icon="🪄",
                second_stage=(stage_label or "default"),
                model=二采放大模型,
                input_shape=self._latent_shape_text(video_latent),
            )
            upscale_model = self._call_core_node("LatentUpscaleModelLoader", model_name=二采放大模型)[0]
            second_video_latent = self._call_core_node(
                "LTXVLatentUpsampler",
                samples=video_latent,
                upscale_model=upscale_model,
                vae=二采VAE,
            )[0]

            try:
                del upscale_model
            except Exception:
                pass

            self._log(
                "UPSCALE",
                "二采前 x2 latent 放大完成。",
                level="OK",
                icon="📈",
                second_stage=(stage_label or "default"),
                output_shape=self._latent_shape_text(second_video_latent),
            )
            self._release_vram("二采放大完成", aggressive=True)

        if bool(应用二采图生视频 and 是否图生视频):
            second_video_latent = self._apply_second_pass_img_to_video_inplace(
                video_latent=second_video_latent,
                i2v_image=二采latent输入图像,
                i2v_vae=二采图生视频VAE,
                strength=1.0,
                stage_label=(stage_label or "default"),
            )

        second_input_latent = self._call_core_node(
            "LTXVConcatAVLatent",
            video_latent=second_video_latent,
            audio_latent=audio_latent,
        )[0]

        try:
            del audio_latent
        except Exception:
            pass
        try:
            del video_latent
        except Exception:
            pass
        try:
            del second_video_latent
        except Exception:
            pass

        self._release_vram("二采输入 latent 组装完成", aggressive=True)
        return second_input_latent

    def _execute_second_pass_from_input(self, second_input_latent, guider, sampler, 添加噪波, 噪波种子, 二采西格玛, segment_label="", guider_source="引导器", sampler_source="采样器", 二采噪波=None):
        second_sigmas = self._parse_sigmas_text(二采西格玛, "二采西格玛")
        second_noise, _noise_source = self._resolve_second_pass_noise_obj(添加噪波, 噪波种子, 二采噪波=二采噪波)
        self._flow_print(self._flow_stage_text(2, segment_label or "full"))
        second_result = self._call_sampler_advanced(
            noise_obj=second_noise,
            guider=guider,
            sampler=sampler,
            sigmas=second_sigmas,
            latent_image=second_input_latent,
        )

        del second_noise
        del second_input_latent
        self._release_vram("二采采样完成", aggressive=True)
        return second_result

    def _run_second_pass(self, first_pass_output, guider, sampler, 添加噪波, 噪波种子, 二采使用放大模型, 二采放大模型, 二采西格玛, 二采VAE=None, 二采引导器=None, 二采采样器=None, 二采噪波=None, 是否图生视频=False, 二采latent输入图像=None, 二采图生视频VAE=None, 应用二采图生视频=False):
        second_pass_guider, second_pass_sampler, guider_source, sampler_source = self._resolve_second_pass_route(
            引导器=guider,
            采样器=sampler,
            二采引导器=二采引导器,
            二采采样器=二采采样器,
        )
        second_input_latent = self._prepare_second_pass_input_latent(
            first_pass_output=first_pass_output,
            二采使用放大模型=二采使用放大模型,
            二采放大模型=二采放大模型,
            二采VAE=二采VAE,
            stage_label="single_pipeline",
            是否图生视频=是否图生视频,
            二采latent输入图像=二采latent输入图像,
            二采图生视频VAE=二采图生视频VAE,
            应用二采图生视频=应用二采图生视频,
        )
        return self._execute_second_pass_from_input(
            second_input_latent=second_input_latent,
            guider=second_pass_guider,
            sampler=second_pass_sampler,
            添加噪波=添加噪波,
            噪波种子=噪波种子,
            二采西格玛=二采西格玛,
            segment_label="full",
            guider_source=guider_source,
            sampler_source=sampler_source,
            二采噪波=二采噪波,
        )

    def _run_delayed_segmented_second_pass(
        self,
        delayed_second_pass_segments,
        引导器,
        采样器,
        添加噪波,
        噪波种子,
        二采使用放大模型,
        二采放大模型,
        二采西格玛,
        二采VAE=None,
        二采引导器=None,
        二采采样器=None,
        二采噪波=None,
        continuation_vae=None,
        supports_official_continuation=False,
        官方式latent续写=True,
        续写重叠引导强度=0.5,
        续写输出线性融合=True,
        strict_latent_continuity=True,
        是否图生视频=False,
        二采latent输入图像=None,
        二采图生视频VAE=None,
        **kwargs,
    ):
        if not delayed_second_pass_segments:
            raise RuntimeError("没有可用于延后二采的分段一采结果。")
        segment_prompt_context = kwargs.get("segment_prompt_context", None)
        accumulated_video_output = None
        accumulated_video_denoised = None
        accumulated_audio_output = None
        accumulated_audio_denoised = None
        previous_segment_video_output = None
        previous_segment_audio_output = None
        second_pass_guider_base, second_pass_sampler, second_pass_guider_source, second_pass_sampler_source = self._resolve_second_pass_route(
            引导器=引导器,
            采样器=采样器,
            二采引导器=二采引导器,
            二采采样器=二采采样器,
        )

        total_segments = len(delayed_second_pass_segments)
        for pos, segment in enumerate(delayed_second_pass_segments, start=1):
            idx = int(segment["index"])
            overlap_v = int(segment.get("overlap_v", 0))
            overlap_a = int(segment.get("overlap_a", 0))
            has_audio = bool(segment.get("has_audio", False))

            first_pass_segment_latent = self._rebuild_from_parts(
                segment.get("first_video_output"),
                segment.get("first_audio_output"),
            )
            prepared_second_input = self._prepare_second_pass_input_latent(
                first_pass_output=first_pass_segment_latent,
                二采使用放大模型=二采使用放大模型,
                二采放大模型=二采放大模型,
                二采VAE=二采VAE,
                stage_label=f"delayed_{idx}/{total_segments}",
                是否图生视频=是否图生视频,
                二采latent输入图像=二采latent输入图像,
                二采图生视频VAE=二采图生视频VAE,
                应用二采图生视频=False,
            )

            if has_audio:
                second_video_source, second_audio_source = self._try_split_av_latent(prepared_second_input)
                if second_video_source is None:
                    raise RuntimeError("延后二采准备阶段输出不是可拆分的 AV latent，无法继续。")
            else:
                second_video_source, second_audio_source = prepared_second_input, None

            seg_source_video_len = self._get_temporal_length(second_video_source, prefer_video=True)
            seg_source_audio_len = self._get_temporal_length(second_audio_source, prefer_video=False) if second_audio_source is not None else 0
            used_official_video_continuation = False

            segment_second_pass_guider = self._build_segment_guider(second_pass_guider_base, pos, total_segments, segment_prompt_context=segment_prompt_context, guider_role="delayed_second_pass")
            if pos == 1:
                seg_video_input = second_video_source
                seg_audio_input = second_audio_source
            else:
                use_official_video_continuation = bool(
                    官方式latent续写 and supports_official_continuation and overlap_v > 0 and previous_segment_video_output is not None
                )
                if use_official_video_continuation:
                    guide_video_latent = self._tail_latent_temporal(previous_segment_video_output, overlap_v, prefer_video=True)
                    guide_video_len = self._get_temporal_length(guide_video_latent, prefer_video=True)
                    if guide_video_len <= 0:
                        raise RuntimeError(f"第 {idx} 段延后二采 guide 长度为 0，无法继续。")
                    segment_second_pass_guider, seg_video_input = self._apply_official_style_video_guide(
                        guider=segment_second_pass_guider,
                        video_latent=second_video_source,
                        guide_video_latent=guide_video_latent,
                        guide_vae=continuation_vae,
                        strength=续写重叠引导强度,
                        segment_label=f"二采{idx}/{total_segments}",
                    )
                    used_official_video_continuation = True
                else:
                    seg_video_input = self._merge_previous_context_with_source_window(
                        previous_window_output=previous_segment_video_output,
                        source_window=second_video_source,
                        overlap_length=overlap_v,
                        prefer_video=True,
                    )

                if has_audio:
                    seg_audio_input = self._merge_previous_context_with_source_window(
                        previous_window_output=previous_segment_audio_output,
                        source_window=second_audio_source,
                        overlap_length=overlap_a,
                        prefer_video=False,
                    )
                else:
                    seg_audio_input = None

            delayed_second_input = self._rebuild_from_parts(seg_video_input, seg_audio_input)
            second_result = self._execute_second_pass_from_input(
                second_input_latent=delayed_second_input,
                guider=segment_second_pass_guider,
                sampler=second_pass_sampler,
                添加噪波=添加噪波,
                噪波种子=噪波种子,
                二采西格玛=二采西格玛,
                segment_label=f"二采{idx}/{total_segments}",
                guider_source=second_pass_guider_source,
                sampler_source=second_pass_sampler_source,
                二采噪波=二采噪波,
            )

            seg_output = second_result[0]
            seg_denoised = second_result[1] if len(second_result) > 1 else second_result[0]

            if has_audio:
                seg_out_video, seg_out_audio = self._try_split_av_latent(seg_output)
                seg_den_video, seg_den_audio = self._try_split_av_latent(seg_denoised)
                if seg_out_video is None or seg_den_video is None:
                    raise RuntimeError("延后二采结果不是可拆分的 AV latent，无法继续分段拼接。")
            else:
                seg_out_video, seg_out_audio = seg_output, None
                seg_den_video, seg_den_audio = seg_denoised, None

            seg_out_video = self._match_temporal_length(
                seg_out_video,
                seg_source_video_len,
                prefer_video=True,
                keep="head",
                reason=("delayed_second_pass_official_guide_output_trim" if used_official_video_continuation else "delayed_second_pass_video_length_align"),
            )
            seg_den_video = self._match_temporal_length(
                seg_den_video,
                seg_source_video_len,
                prefer_video=True,
                keep="head",
                reason=("delayed_second_pass_official_guide_denoised_trim" if used_official_video_continuation else "delayed_second_pass_video_denoised_length_align"),
            )
            if has_audio and seg_out_audio is not None:
                seg_out_audio = self._match_temporal_length(
                    seg_out_audio,
                    seg_source_audio_len,
                    prefer_video=False,
                    keep="head",
                    reason="delayed_second_pass_audio_length_align",
                )
                seg_den_audio = self._match_temporal_length(
                    seg_den_audio,
                    seg_source_audio_len,
                    prefer_video=False,
                    keep="head",
                    reason="delayed_second_pass_audio_denoised_length_align",
                )

            if pos == 1:
                accumulated_video_output = seg_out_video
                accumulated_video_denoised = seg_den_video
                accumulated_audio_output = seg_out_audio
                accumulated_audio_denoised = seg_den_audio
            else:
                old_accumulated_video_output = accumulated_video_output
                old_accumulated_video_denoised = accumulated_video_denoised
                if (not strict_latent_continuity) and 续写输出线性融合 and overlap_v > 0:
                    accumulated_video_output = self._linear_overlap_blend(accumulated_video_output, seg_out_video, overlap_v, prefer_video=True)
                    accumulated_video_denoised = self._linear_overlap_blend(accumulated_video_denoised, seg_den_video, overlap_v, prefer_video=True)
                else:
                    seg_out_video_fresh = self._slice_latent_temporal(seg_out_video, overlap_v, self._get_temporal_length(seg_out_video, prefer_video=True), prefer_video=True)
                    seg_den_video_fresh = self._slice_latent_temporal(seg_den_video, overlap_v, self._get_temporal_length(seg_den_video, prefer_video=True), prefer_video=True)
                    accumulated_video_output = self._concat_latent_temporal([accumulated_video_output, seg_out_video_fresh], prefer_video=True)
                    accumulated_video_denoised = self._concat_latent_temporal([accumulated_video_denoised, seg_den_video_fresh], prefer_video=True)
                    for obj in (seg_out_video_fresh, seg_den_video_fresh):
                        try:
                            del obj
                        except Exception:
                            pass
                for obj in (old_accumulated_video_output, old_accumulated_video_denoised):
                    try:
                        del obj
                    except Exception:
                        pass

                if has_audio:
                    old_accumulated_audio_output = accumulated_audio_output
                    old_accumulated_audio_denoised = accumulated_audio_denoised
                    if (not strict_latent_continuity) and 续写输出线性融合 and overlap_a > 0:
                        accumulated_audio_output = self._linear_overlap_blend(accumulated_audio_output, seg_out_audio, overlap_a, prefer_video=False)
                        accumulated_audio_denoised = self._linear_overlap_blend(accumulated_audio_denoised, seg_den_audio, overlap_a, prefer_video=False)
                    else:
                        seg_out_audio_fresh = self._slice_latent_temporal(seg_out_audio, overlap_a, self._get_temporal_length(seg_out_audio, prefer_video=False), prefer_video=False)
                        seg_den_audio_fresh = self._slice_latent_temporal(seg_den_audio, overlap_a, self._get_temporal_length(seg_den_audio, prefer_video=False), prefer_video=False)
                        accumulated_audio_output = self._concat_latent_temporal([accumulated_audio_output, seg_out_audio_fresh], prefer_video=False)
                        accumulated_audio_denoised = self._concat_latent_temporal([accumulated_audio_denoised, seg_den_audio_fresh], prefer_video=False)
                        for obj in (seg_out_audio_fresh, seg_den_audio_fresh):
                            try:
                                del obj
                            except Exception:
                                pass
                    for obj in (old_accumulated_audio_output, old_accumulated_audio_denoised):
                        try:
                            del obj
                        except Exception:
                            pass

            if pos < total_segments:
                next_item = delayed_second_pass_segments[pos]
                next_overlap_v = int(next_item.get("overlap_v", 0))
                previous_segment_video_output = self._tail_latent_temporal(seg_out_video, next_overlap_v, prefer_video=True) if next_overlap_v > 0 else None
                if has_audio:
                    next_overlap_a = int(next_item.get("overlap_a", 0))
                    previous_segment_audio_output = self._tail_latent_temporal(seg_out_audio, next_overlap_a, prefer_video=False) if next_overlap_a > 0 else None
                else:
                    previous_segment_audio_output = None
                self._flow_print(
                    self._flow_stage_text(2, f"{idx}/{total_segments}"),
                    self._flow_stage_text(2, f"{idx + 1}/{total_segments}"),
                    note="续写",
                )
            else:
                previous_segment_video_output = None
                previous_segment_audio_output = None

            for obj in (
                prepared_second_input,
                second_result,
                seg_output,
                seg_denoised,
                second_video_source,
                second_audio_source,
                seg_video_input,
                seg_audio_input,
                seg_out_video,
                seg_den_video,
                seg_out_audio,
                seg_den_audio,
            ):
                try:
                    del obj
                except Exception:
                    pass
            self._release_vram(f"延后二采分段 {idx}/{total_segments} 完成", aggressive=True)

        final_output = self._rebuild_from_parts(accumulated_video_output, accumulated_audio_output)
        final_denoised = self._rebuild_from_parts(accumulated_video_denoised, accumulated_audio_denoised)
        self._log(
            "SEG",
            "延后二采所有分段处理完成，已自动拼接为最终 latent。",
            level="OK",
            icon="✅",
            final_output_shape=self._latent_shape_text(final_output),
            final_denoised_shape=self._latent_shape_text(final_denoised),
        )
        return final_output, final_denoised

    def _run_single_pipeline(self, 引导器, 采样器, latent_input, 一采西格玛文本, 添加噪波, 噪波种子, 启动二采, 二采使用放大模型, 二采放大模型, 二采西格玛, 西格玛=None, 二采引导器=None, 二采采样器=None, 二采噪波=None, 二采VAE=None, segment_label="full", 返回一采结果=False, 是否图生视频=False, 二采latent输入图像=None, 二采图生视频VAE=None, 应用起始图生视频=False, 应用二采图生视频=False):
        force_half_first_pass = bool(启动二采 and 二采使用放大模型)

        if force_half_first_pass and not 二采放大模型:
            二采放大模型 = self.DEFAULT_X2_MODEL
        if force_half_first_pass and 二采VAE is None:
            raise RuntimeError("当“启动二采 + 二采使用放大模型”启用时，必须连接“二采VAE”，因为节点会走半分辨率一采 + x2 恢复流程。")

        first_input_latent = latent_input
        if force_half_first_pass:
            first_input_latent = self._prepare_first_pass_input_for_x2_pipeline(latent_input)
        if bool(应用起始图生视频 and 是否图生视频):
            first_input_latent = self._prepare_first_pass_img_to_video_input_latent(
                latent_input=first_input_latent,
                i2v_image=二采latent输入图像,
                i2v_vae=二采图生视频VAE,
                stage_label=(segment_label or "full"),
            )

        first_input_latent = self._sanitize_latent_noise_mask(first_input_latent, prefer_video=True, reason=f"pre_sample_{segment_label}")

        first_sigmas = self._resolve_first_sigmas(西格玛, 一采西格玛文本)
        first_noise = self._get_noise_obj(添加噪波, 噪波种子)

        self._flow_print(self._flow_stage_text(1, segment_label))

        first_result = self._call_sampler_advanced(noise_obj=first_noise, guider=引导器, sampler=采样器, sigmas=first_sigmas, latent_image=first_input_latent)

        first_pass_output = first_result[0]
        first_pass_denoised = first_result[1] if len(first_result) > 1 else first_result[0]

        del first_noise
        del first_sigmas
        if first_input_latent is not latent_input:
            try:
                del first_input_latent
            except Exception:
                pass

        if not 启动二采:
            if 返回一采结果:
                return first_pass_output, first_pass_denoised, first_pass_output, first_pass_denoised
            return first_result

        self._release_vram("单段一采完成，准备进入二采", aggressive=True)
        second_result = self._run_second_pass(
            first_pass_output=first_pass_output,
            guider=引导器,
            sampler=采样器,
            添加噪波=添加噪波,
            噪波种子=噪波种子,
            二采使用放大模型=二采使用放大模型,
            二采放大模型=二采放大模型,
            二采西格玛=二采西格玛,
            二采噪波=二采噪波,
            二采VAE=二采VAE,
            二采引导器=二采引导器,
            二采采样器=二采采样器,
            是否图生视频=是否图生视频,
            二采latent输入图像=二采latent输入图像,
            二采图生视频VAE=二采图生视频VAE,
            应用二采图生视频=应用二采图生视频,
        )
        if not 返回一采结果:
            return second_result

        second_output = second_result[0]
        second_denoised = second_result[1] if len(second_result) > 1 else second_result[0]
        return second_output, second_denoised, first_pass_output, first_pass_denoised

    def _get_samples(self, latent):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise RuntimeError("LATENT 数据格式不正确，缺少 samples。")
        return latent["samples"]

    def _get_temporal_axis(self, latent, prefer_video=False):
        samples = self._get_samples(latent)
        if getattr(samples, "is_nested", False):
            raise RuntimeError("暂不支持 NestedTensor 的分段时序裁剪。")

        if samples.ndim == 5:
            return 2  # video latent: (B,C,T,H,W)
        if samples.ndim == 4:
            return 2  # audio latent: (B,C,T,D)
        if samples.ndim == 3:
            return 2
        raise RuntimeError(f"当前 latent 维度过低，无法裁剪时间维：{tuple(samples.shape)}")

    def _get_temporal_length(self, latent, prefer_video=False):
        axis = self._get_temporal_axis(latent, prefer_video=prefer_video)
        return int(self._get_samples(latent).shape[axis])

    def _get_noise_mask(self, latent):
        if isinstance(latent, dict):
            return latent.get("noise_mask", None)
        return getattr(latent, "noise_mask", None)

    def _clone_like(self, latent, samples, noise_mask=_KEEP_NOISE_MASK):
        latent_copy = self._shallow_copy_latent(latent)
        latent_copy["samples"] = samples
        if noise_mask is _KEEP_NOISE_MASK:
            return latent_copy
        if noise_mask is None:
            latent_copy.pop("noise_mask", None)
        else:
            latent_copy["noise_mask"] = noise_mask
        return latent_copy

    def _drop_noise_mask(self, latent):
        return self._clone_like(latent, self._get_samples(latent), noise_mask=None)

    def _sanitize_noise_mask_tensor(self, noise_mask):
        if not torch.is_tensor(noise_mask):
            return None
        if getattr(noise_mask, "is_nested", False):
            return None
        if noise_mask.numel() <= 0:
            return None
        if any(int(dim) <= 0 for dim in noise_mask.shape):
            return None
        if noise_mask.ndim >= 2:
            if int(noise_mask.shape[-1]) <= 0 or int(noise_mask.shape[-2]) <= 0:
                return None
        return noise_mask.contiguous()

    def _infer_noise_mask_temporal_axis(self, noise_mask, sample_temporal_length):
        noise_mask = self._sanitize_noise_mask_tensor(noise_mask)
        if noise_mask is None:
            return None

        sample_temporal_length = int(sample_temporal_length)
        if sample_temporal_length <= 0:
            return None

        candidates = []
        upper = max(0, noise_mask.ndim - 2)
        for axis in range(upper):
            try:
                if int(noise_mask.shape[axis]) == sample_temporal_length:
                    candidates.append(axis)
            except Exception:
                pass

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) >= 2 and 2 in candidates:
            return 2
        return None

    def _same_shape_except_axis(self, shape_a, shape_b, axis):
        if len(shape_a) != len(shape_b):
            return False
        for i, (a_dim, b_dim) in enumerate(zip(shape_a, shape_b)):
            if i == axis:
                continue
            if int(a_dim) != int(b_dim):
                return False
        return True

    def _sanitize_latent_noise_mask(self, latent, prefer_video=False, reason=""):
        noise_mask = self._get_noise_mask(latent)
        if noise_mask is None:
            if isinstance(latent, dict) and "noise_mask" in latent:
                return self._drop_noise_mask(latent)
            return self._shallow_copy_latent(latent)

        # AV latent 在 LTXVConcatAVLatent 之后，noise_mask 可能是 NestedTensor。
        # 这个在官方 AV 采样链路里不一定是非法数据；尤其是官方 guide 注入后，
        # 如果这里把 NestedTensor 直接当坏 mask 清掉，后面 guider 里已经写入的
        # keyframe_idxs 还在，但 denoise_mask 变成 None，最终会在 av_model._process_input
        # 里报：'NoneType' object has no attribute 'shape'。
        # 所以：NestedTensor mask 这里先保留原样，不在这个入口粗暴删掉。
        if getattr(noise_mask, "is_nested", False):
            self._log(
                "MASK",
                "检测到 AV noise_mask，已原样透传。",
                level="INFO",
                icon="🪺",
                reason=reason or "nested_noise_mask_passthrough",
            )
            return self._shallow_copy_latent(latent)

        sanitized_mask = self._sanitize_noise_mask_tensor(noise_mask)
        if sanitized_mask is None:
            self._log(
                "MASK",
                "检测到非法或空 noise_mask，已自动移除，避免底层 reshape_mask 崩溃。",
                level="WARN",
                icon="⚠️",
                reason=reason or "invalid_noise_mask",
                mask_shape=(tuple(noise_mask.shape) if torch.is_tensor(noise_mask) else type(noise_mask).__name__),
            )
            return self._drop_noise_mask(latent)

        if sanitized_mask is noise_mask:
            return self._shallow_copy_latent(latent)
        return self._clone_like(latent, self._get_samples(latent), noise_mask=sanitized_mask)

    def _build_overlap_freeze_mask(self, latent, overlap_length, prefer_video=False):
        overlap_length = max(0, int(overlap_length))
        samples = self._get_samples(latent)
        if getattr(samples, "is_nested", False):
            raise RuntimeError("NestedTensor latent 暂不支持直接构建 overlap freeze mask。")

        temporal_axis = self._get_temporal_axis(latent, prefer_video=prefer_video)
        total = int(samples.shape[temporal_axis])
        overlap_length = max(0, min(overlap_length, total))
        if overlap_length <= 0:
            return None

        mask_shape = list(samples.shape)
        if len(mask_shape) >= 2:
            mask_shape[1] = 1
        mask = torch.ones(mask_shape, device=samples.device, dtype=torch.float32)

        slicer = [slice(None)] * len(mask_shape)
        slicer[temporal_axis] = slice(0, overlap_length)
        mask[tuple(slicer)] = 0.0
        return mask.contiguous()

    def _apply_overlap_freeze_noise_mask(self, latent, overlap_length, prefer_video=False, reason=""):
        freeze_mask = self._build_overlap_freeze_mask(latent, overlap_length, prefer_video=prefer_video)
        if freeze_mask is None:
            return self._shallow_copy_latent(latent)

        existing_mask = self._get_noise_mask(latent)
        if torch.is_tensor(existing_mask):
            try:
                expanded_existing = comfy.utils.reshape_mask(existing_mask.float(), tuple(freeze_mask.shape))
                freeze_mask = freeze_mask * expanded_existing.to(device=freeze_mask.device, dtype=freeze_mask.dtype)
            except Exception as e:
                self._log(
                    "MASK",
                    "原有 noise_mask 合并失败，已改用续写锁定 mask。",
                    level="WARN",
                    icon="⚠️",
                    reason=reason or "overlap_freeze_mask_merge",
                    error=str(e),
                )

        return self._clone_like(latent, self._get_samples(latent), noise_mask=freeze_mask.contiguous())

    def _slice_latent_temporal(self, latent, start, end, prefer_video=False):
        samples = self._get_samples(latent)
        axis = self._get_temporal_axis(latent, prefer_video=prefer_video)
        total = int(samples.shape[axis])

        start = max(0, min(int(start), total))
        end = max(start, min(int(end), total))

        slicer = [slice(None)] * samples.ndim
        slicer[axis] = slice(start, end)
        new_samples = samples[tuple(slicer)].contiguous()

        noise_mask = self._get_noise_mask(latent)
        new_noise_mask = None
        if torch.is_tensor(noise_mask):
            noise_mask = self._sanitize_noise_mask_tensor(noise_mask)
            if noise_mask is not None:
                mask_axis = self._infer_noise_mask_temporal_axis(noise_mask, total)
                if mask_axis is None:
                    new_noise_mask = noise_mask
                else:
                    mask_total = int(noise_mask.shape[mask_axis])
                    mask_start = max(0, min(int(start), mask_total))
                    mask_end = max(mask_start, min(int(end), mask_total))
                    if mask_end <= mask_start:
                        self._log(
                            "MASK",
                            "noise_mask 的时间裁剪结果为空，已自动丢弃该 mask，避免后续采样崩溃。",
                            level="WARN",
                            icon="⚠️",
                            start=mask_start,
                            end=mask_end,
                            mask_shape=tuple(noise_mask.shape),
                        )
                        new_noise_mask = None
                    else:
                        mask_slicer = [slice(None)] * noise_mask.ndim
                        mask_slicer[mask_axis] = slice(mask_start, mask_end)
                        new_noise_mask = self._sanitize_noise_mask_tensor(noise_mask[tuple(mask_slicer)])
                        if new_noise_mask is None:
                            self._log(
                                "MASK",
                                "noise_mask 裁剪后为空，已自动丢弃该 mask，避免后续采样崩溃。",
                                level="WARN",
                                icon="⚠️",
                                start=mask_start,
                                end=mask_end,
                                mask_shape=tuple(noise_mask.shape),
                            )

        result = self._clone_like(latent, new_samples, noise_mask=new_noise_mask)
        return self._sanitize_latent_noise_mask(result, prefer_video=prefer_video, reason="slice_temporal")

    def _tail_latent_temporal(self, latent, length, prefer_video=False):
        total = self._get_temporal_length(latent, prefer_video=prefer_video)
        length = max(0, min(int(length), total))
        return self._slice_latent_temporal(latent, total - length, total, prefer_video=prefer_video)

    def _match_temporal_length(self, latent, target_length, prefer_video=False, keep="head", reason=""):
        total = self._get_temporal_length(latent, prefer_video=prefer_video)
        target_length = max(0, int(target_length))

        if target_length == total:
            return self._sanitize_latent_noise_mask(latent, prefer_video=prefer_video, reason=(reason or "match_equal"))

        if target_length > total:
            self._log(
                "SEG",
                "分段输出时间维比目标更短，当前不会补帧；继续保留原长度。",
                level="WARN",
                icon="⚠️",
                reason=reason or "length_mismatch",
                target_length=target_length,
                actual_length=total,
                keep=keep,
            )
            return self._sanitize_latent_noise_mask(latent, prefer_video=prefer_video, reason=(reason or "length_mismatch"))

        if keep == "tail":
            result = self._slice_latent_temporal(latent, total - target_length, total, prefer_video=prefer_video)
        else:
            result = self._slice_latent_temporal(latent, 0, target_length, prefer_video=prefer_video)

        self._log(
            "SEG",
            "已把分段输出裁回目标时间维，避免 guide 追加 token 污染后续续写。",
            level="OK",
            icon="✂️",
            reason=reason or "trim_to_target",
            target_length=target_length,
            actual_length=total,
            keep=keep,
        )
        return self._sanitize_latent_noise_mask(result, prefer_video=prefer_video, reason=(reason or "trim_to_target"))

    def _concat_latent_temporal(self, latent_list, prefer_video=False):
        latent_list = [x for x in latent_list if x is not None]
        if not latent_list:
            raise RuntimeError("没有可用于拼接的 latent。")
        if len(latent_list) == 1:
            return self._sanitize_latent_noise_mask(latent_list[0], prefer_video=prefer_video, reason="concat_single")

        base = latent_list[0]
        axis = self._get_temporal_axis(base, prefer_video=prefer_video)
        tensors = [self._get_samples(x) for x in latent_list]
        new_samples = torch.cat(tensors, dim=axis).contiguous()

        saw_any_mask = False
        all_have_mask = True
        static_masks = []
        temporal_masks = []
        temporal_axes = []

        for item in latent_list:
            raw_mask = self._get_noise_mask(item)
            if raw_mask is None:
                all_have_mask = False
                continue

            saw_any_mask = True
            item_mask = self._sanitize_noise_mask_tensor(raw_mask)
            if item_mask is None:
                all_have_mask = False
                continue

            item_mask = item_mask.to(device=new_samples.device)
            item_axis = self._infer_noise_mask_temporal_axis(item_mask, self._get_temporal_length(item, prefer_video=prefer_video))
            if item_axis is None:
                static_masks.append(item_mask)
            else:
                temporal_masks.append(item_mask)
                temporal_axes.append(item_axis)

        new_noise_mask = None
        if saw_any_mask and all_have_mask:
            if len(static_masks) == len(latent_list) and static_masks:
                first_shape = tuple(static_masks[0].shape)
                if all(tuple(mask.shape) == first_shape for mask in static_masks[1:]):
                    new_noise_mask = static_masks[0].contiguous()
                else:
                    self._log(
                        "MASK",
                        "检测到静态 noise_mask 形状不一致，拼接后将自动移除 mask，避免后续采样崩溃。",
                        level="WARN",
                        icon="⚠️",
                    )
            elif len(temporal_masks) == len(latent_list) and temporal_masks:
                first_axis = temporal_axes[0]
                first_shape = tuple(temporal_masks[0].shape)
                if all(axis_value == first_axis for axis_value in temporal_axes) and all(
                    self._same_shape_except_axis(first_shape, tuple(mask.shape), first_axis) for mask in temporal_masks[1:]
                ):
                    new_noise_mask = torch.cat(temporal_masks, dim=first_axis).contiguous()
                else:
                    self._log(
                        "MASK",
                        "检测到时间型 noise_mask 结构不一致，拼接后将自动移除 mask，避免后续采样崩溃。",
                        level="WARN",
                        icon="⚠️",
                    )
            else:
                self._log(
                    "MASK",
                    "检测到静态/时间型 noise_mask 混用，拼接后将自动移除 mask，避免后续采样崩溃。",
                    level="WARN",
                    icon="⚠️",
                )

        result = self._clone_like(base, new_samples, noise_mask=new_noise_mask)
        return self._sanitize_latent_noise_mask(result, prefer_video=prefer_video, reason="concat_temporal")

    def _merge_previous_context_with_source_window(self, previous_window_output, source_window, overlap_length, prefer_video=False):
        overlap_length = max(0, int(overlap_length))
        if overlap_length <= 0 or previous_window_output is None:
            return self._shallow_copy_latent(source_window)

        prev_tail = self._tail_latent_temporal(previous_window_output, overlap_length, prefer_video=prefer_video)
        source_total = self._get_temporal_length(source_window, prefer_video=prefer_video)
        fresh_tail = self._slice_latent_temporal(source_window, overlap_length, source_total, prefer_video=prefer_video)
        return self._concat_latent_temporal([prev_tail, fresh_tail], prefer_video=prefer_video)

    def _fraction_to_index(self, value, total, total_base):
        if total_base <= 0:
            return 0
        idx = int(round(float(value) / float(total_base) * float(total)))
        return max(0, min(total, idx))

    def _resolve_segment_total_frames(self, latent_total_units, fps, total_frames_override=0, total_seconds_override=0.0):
        if int(total_frames_override) > 0:
            return int(total_frames_override)
        if float(total_seconds_override) > 0.0:
            raw = max(1, int(round(float(total_seconds_override) * float(fps))) + 1)
            return raw
        return int(latent_total_units)

    def _map_window_to_units(self, start_value, end_value, total_units, total_base):
        total_units = int(total_units)
        total_base = max(1, int(total_base))

        start_idx = self._fraction_to_index(start_value, total_units, total_base)
        end_idx = self._fraction_to_index(end_value, total_units, total_base)

        if total_units <= 0:
            return 0, 0

        if end_idx <= start_idx:
            if start_idx >= total_units:
                start_idx = max(0, total_units - 1)
                end_idx = total_units
            else:
                end_idx = min(total_units, start_idx + 1)

        return start_idx, end_idx

    def _seconds_to_valid_frames(self, seconds, fps):
        raw = max(1, int(round(float(seconds) * float(fps))) + 1)
        return int(((raw - 1 + 7) // 8) * 8 + 1)

    def _plan_segment_windows(self, total_frames, max_frames, base_overlap_frames):
        total_frames = int(total_frames)
        max_frames = int(max_frames)
        base_overlap_frames = int(base_overlap_frames)

        if total_frames <= max_frames:
            return [(0, total_frames)]

        base_overlap_frames = max(1, min(base_overlap_frames, max_frames - 1))
        windows = []
        seen = set()
        start = 0

        while True:
            end = min(start + max_frames, total_frames)
            item = (start, end)
            if item not in seen:
                windows.append(item)
                seen.add(item)

            if end >= total_frames:
                break

            next_start = end - base_overlap_frames

            if total_frames - next_start <= max_frames:
                final_start = max(0, total_frames - max_frames)
                item = (final_start, total_frames)
                if item not in seen:
                    windows.append(item)
                    seen.add(item)
                break

            start = next_start

        return windows

    def _try_split_av_latent(self, latent):
        try:
            out = self._call_core_node("LTXVSeparateAVLatent", av_latent=latent)
            if len(out) >= 2:
                return out[0], out[1]
        except Exception:
            pass
        return None, None

    def _rebuild_from_parts(self, video_latent, audio_latent):
        if audio_latent is None:
            return self._sanitize_latent_noise_mask(video_latent, prefer_video=True, reason="rebuild_video_only")
        rebuilt = self._call_core_node("LTXVConcatAVLatent", video_latent=video_latent, audio_latent=audio_latent)[0]
        return self._sanitize_latent_noise_mask(rebuilt, prefer_video=True, reason="rebuild_av_parts")

    def _parse_plan_json(self, plan_json_text):
        text = "" if plan_json_text is None else str(plan_json_text).strip()
        if not text:
            return []

        try:
            plan = json.loads(text)
        except Exception as e:
            raise RuntimeError(f"分段方案JSON 解析失败：{e}")

        if not isinstance(plan, list):
            raise RuntimeError("分段方案JSON 必须是数组。")

        normalized = []
        for i, item in enumerate(plan, start=1):
            if not isinstance(item, dict):
                raise RuntimeError(f"分段方案第 {i} 项不是对象。")

            required_keys = [
                "段",
                "视频latent开始",
                "视频latent结束_inclusive",
                "音频token开始",
                "音频token结束_exclusive",
            ]
            for key in required_keys:
                if key not in item:
                    raise RuntimeError(f"分段方案第 {i} 项缺少字段：{key}")

            v_start = int(item["视频latent开始"])
            v_end_inclusive = int(item["视频latent结束_inclusive"])
            a_start = int(item["音频token开始"])
            a_end_exclusive = int(item["音频token结束_exclusive"])

            normalized.append({
                "index": int(item.get("段", i)),
                "frame_start": int(item.get("原始帧开始", 0)),
                "frame_end": int(item.get("原始帧结束_inclusive", 0)) + 1 if "原始帧结束_inclusive" in item else 0,
                "video_start": v_start,
                "video_end": v_end_inclusive + 1,   # 转成 exclusive
                "audio_start": a_start,
                "audio_end": a_end_exclusive,       # 本来就是 exclusive
                "meta": copy.deepcopy(item),
            })

        normalized.sort(key=lambda x: x["index"])

        previous_item = None
        for item in normalized:
            if int(item["video_end"]) <= int(item["video_start"]):
                raise RuntimeError(f"分段方案第 {item['index']} 项视频区间无效：{item['video_start']}->{item['video_end']}")
            if int(item["audio_end"]) < int(item["audio_start"]):
                raise RuntimeError(f"分段方案第 {item['index']} 项音频区间无效：{item['audio_start']}->{item['audio_end']}")

            if previous_item is not None:
                if int(item["video_start"]) < int(previous_item["video_start"]):
                    raise RuntimeError("分段方案视频起点不是单调递增，无法保证续写连续性。")
                if int(item["video_end"]) <= int(previous_item["video_end"]):
                    raise RuntimeError("分段方案视频终点不是单调递增，无法保证续写连续性。")

                overlap_v = max(0, int(previous_item["video_end"]) - int(item["video_start"]))
                curr_video_len = int(item["video_end"]) - int(item["video_start"])
                if overlap_v >= curr_video_len:
                    raise RuntimeError(
                        f"分段方案第 {item['index']} 项视频 overlap={overlap_v} 已经吃满当前段长度={curr_video_len}，会导致没有 fresh latent。"
                    )

                overlap_a = max(0, int(previous_item["audio_end"]) - int(item["audio_start"]))
                curr_audio_len = int(item["audio_end"]) - int(item["audio_start"])
                if curr_audio_len > 0 and overlap_a >= curr_audio_len:
                    raise RuntimeError(
                        f"分段方案第 {item['index']} 项音频 overlap={overlap_a} 已经吃满当前段长度={curr_audio_len}，会导致没有 fresh audio。"
                    )

            previous_item = item

        return normalized

    def _to_cpu_obj(self, obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: self._to_cpu_obj(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_cpu_obj(v) for v in obj)
        return obj

    def _sanitize_filename(self, text):
        bad = '<>:"/\\|?*'
        out = str(text)
        for ch in bad:
            out = out.replace(ch, "_")
        out = out.strip().strip(".")
        return out or "latent"

    def _get_segment_save_dir(self):
        if folder_paths is not None:
            try:
                base = folder_paths.get_output_directory()
            except Exception:
                base = os.getcwd()
        else:
            base = os.getcwd()

        save_dir = os.path.join(base, self.SEGMENT_LATENT_SUBDIR)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _save_segment_latent_pair(self, prefix, seg_index, seg_output, seg_denoised, seg_meta):
        save_dir = self._get_segment_save_dir()
        prefix = self._sanitize_filename(prefix)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_name = f"{prefix}_{stamp}_seg{int(seg_index):03d}"
        output_path = os.path.join(save_dir, f"{base_name}_output.pt")
        denoised_path = os.path.join(save_dir, f"{base_name}_denoised.pt")
        meta_path = os.path.join(save_dir, f"{base_name}_meta.json")

        torch.save(self._to_cpu_obj(seg_output), output_path)
        torch.save(self._to_cpu_obj(seg_denoised), denoised_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(seg_meta, f, ensure_ascii=False, indent=2)

        self._log(
            "SAVE",
            "已保存分段 latent。",
            level="OK",
            icon="💾",
            segment=seg_index,
            output=os.path.basename(output_path),
            denoised=os.path.basename(denoised_path),
            meta=os.path.basename(meta_path),
        )

    def _run_segment_windows(
        self,
        mapped_windows,
        引导器,
        采样器,
        Latent图像,
        一采西格玛文本,
        添加噪波,
        噪波种子,
        启动二采,
        二采使用放大模型,
        二采放大模型,
        二采西格玛,
        西格玛=None,
        二采引导器=None,
        二采采样器=None,
        二采噪波=None,
        二采VAE=None,
        续写VAE=None,
        保存分段latent=False,
        分段latent前缀="nf_v2v_seg",
        官方式latent续写=True,
        续写重叠引导强度=0.5,
        续写输出线性融合=True,
        是否图生视频=False,
        二采latent输入图像=None,
        **kwargs,
    ):
        source_video_latent, source_audio_latent = self._try_split_av_latent(Latent图像)
        has_audio = source_video_latent is not None and source_audio_latent is not None
        segment_prompt_context = kwargs.get("segment_prompt_context", None)
        if has_audio:
            self._log("SEG", "检测到 AV latent，音视频同步分段。", level="INFO", icon="🎬")
            timing_video_latent = source_video_latent
        else:
            self._log("SEG", "未检测到 AV latent，按视频分段。", level="WARN", icon="⚠️")
            timing_video_latent = Latent图像

        continuation_vae = self._resolve_continuation_vae(续写VAE=续写VAE, 二采VAE=二采VAE)
        supports_official_continuation, continuation_reason = self._supports_official_style_continuation(continuation_vae)
        if bool(官方式latent续写) and supports_official_continuation:
            self._log("SEG", "续写模式：官方 guide。", level="INFO", icon="🧩", guide_strength=续写重叠引导强度)
        elif bool(官方式latent续写):
            self._log("SEG", "续写模式：旧版拼接。", level="WARN", icon="⚠️", reason=continuation_reason)

        segment_use_delayed_segmented_second_pass = bool(启动二采 and 二采使用放大模型)
        if segment_use_delayed_segmented_second_pass:
            if 二采VAE is None:
                raise RuntimeError("分段模式下启用“二采使用放大模型”时，必须连接“二采VAE”。")
            if not 二采放大模型:
                二采放大模型 = self.DEFAULT_X2_MODEL
            self._log(
                "SEG",
                "检测到分段二采放大：先跑 0.5 一采，再逐段放大续接。",
                level="INFO",
                icon="🪜",
                upscale_model=二采放大模型,
            )

        force_single_pass_continuity = not bool(启动二采)
        strict_latent_continuity = bool(force_single_pass_continuity or (not bool(续写输出线性融合)))
        if force_single_pass_continuity:
            self._log(
                "SEG",
                "连续模式：严格连续。",
                level="INFO",
                icon="🔒",
            )
        elif strict_latent_continuity:
            self._log(
                "SEG",
                "连续模式：关闭 overlap 融合。",
                level="INFO",
                icon="🔒",
            )

        accumulated_video_output = None
        accumulated_video_denoised = None
        accumulated_audio_output = None
        accumulated_audio_denoised = None
        delayed_second_pass_segments = []

        previous_segment_video_output = None
        previous_segment_audio_output = None
        previous_video_window = None
        previous_audio_window = None

        try:
            for pos, item in enumerate(mapped_windows, start=1):
                idx = int(item["index"])
                start_v = int(item["video_start"])
                end_v = int(item["video_end"])
                start_a = int(item["audio_start"])
                end_a = int(item["audio_end"])

                seg_video_source_full = self._slice_latent_temporal(timing_video_latent, start_v, end_v, prefer_video=True)
                seg_audio_source = None
                if has_audio:
                    seg_audio_source = self._slice_latent_temporal(source_audio_latent, start_a, end_a, prefer_video=False)

                seg_source_video_len = self._get_temporal_length(seg_video_source_full, prefer_video=True)
                seg_source_audio_len = self._get_temporal_length(seg_audio_source, prefer_video=False) if seg_audio_source is not None else 0

                if segment_use_delayed_segmented_second_pass:
                    seg_video_source = self._resize_video_latent_spatial(
                        seg_video_source_full,
                        factor=0.5,
                        reason=f"分段 {idx}/{len(mapped_windows)} 一采母体预缩放",
                    )
                else:
                    seg_video_source = seg_video_source_full

                used_official_video_continuation = False
                segment_guider = self._build_segment_guider(
                    引导器,
                    pos,
                    len(mapped_windows),
                    segment_prompt_context=segment_prompt_context,
                    guider_role="first_pass",
                    disable_cfg_decay=force_single_pass_continuity,
                    disable_cfg_decay_reason="single_pass_continuity",
                )
                segment_second_pass_guider = None
                if 启动二采:
                    segment_second_pass_base = 二采引导器 if 二采引导器 is not None else 引导器
                    segment_second_pass_guider = self._build_segment_guider(segment_second_pass_base, pos, len(mapped_windows), segment_prompt_context=segment_prompt_context, guider_role="second_pass")
                segment_second_pass_sampler = 二采采样器 if 二采采样器 is not None else 采样器
                if pos == 1:
                    overlap_v = 0
                    overlap_a = 0
                    seg_video_input = seg_video_source
                    seg_audio_input = seg_audio_source
                else:
                    overlap_v = max(0, int(previous_video_window[1] - start_v))
                    overlap_a = max(0, int(previous_audio_window[1] - start_a)) if has_audio else 0

                    use_official_video_continuation = bool(
                        官方式latent续写 and supports_official_continuation and overlap_v > 0 and previous_segment_video_output is not None
                    )
                    if use_official_video_continuation:
                        guide_video_latent = self._tail_latent_temporal(previous_segment_video_output, overlap_v, prefer_video=True)
                        guide_video_len = self._get_temporal_length(guide_video_latent, prefer_video=True)
                        if guide_video_len <= 0:
                            raise RuntimeError(f"第 {idx} 段官方续写 guide 长度为 0，无法继续。")
                        segment_guider, seg_video_input = self._apply_official_style_video_guide(
                            guider=segment_guider,
                            video_latent=seg_video_source,
                            guide_video_latent=guide_video_latent,
                            guide_vae=continuation_vae,
                            strength=续写重叠引导强度,
                            segment_label=f"{idx}/{len(mapped_windows)}",
                        )
                        used_official_video_continuation = True
                    else:
                        if overlap_v > 0 and previous_segment_video_output is not None:
                            self._log(
                                "SEG",
                                "本段续写：旧版拼接。",
                                level="INFO",
                                icon="🔗",
                                segment=f"{idx}/{len(mapped_windows)}",
                                overlap=overlap_v,
                            )
                        seg_video_input = self._merge_previous_context_with_source_window(
                            previous_window_output=previous_segment_video_output,
                            source_window=seg_video_source,
                            overlap_length=overlap_v,
                            prefer_video=True,
                        )

                    if has_audio:
                        seg_audio_input = self._merge_previous_context_with_source_window(
                            previous_window_output=previous_segment_audio_output,
                            source_window=seg_audio_source,
                            overlap_length=overlap_a,
                            prefer_video=False,
                        )
                    else:
                        seg_audio_input = None

                    if strict_latent_continuity:
                        if overlap_v > 0 and seg_video_input is not None and (not used_official_video_continuation):
                            seg_video_input = self._apply_overlap_freeze_noise_mask(
                                seg_video_input,
                                overlap_v,
                                prefer_video=True,
                                reason=f"segment_video_overlap_freeze_{idx}/{len(mapped_windows)}",
                            )
                        if has_audio and overlap_a > 0 and seg_audio_input is not None:
                            seg_audio_input = self._apply_overlap_freeze_noise_mask(
                                seg_audio_input,
                                overlap_a,
                                prefer_video=False,
                                reason=f"segment_audio_overlap_freeze_{idx}/{len(mapped_windows)}",
                            )
                        if (overlap_v > 0 and (not used_official_video_continuation)) or (has_audio and overlap_a > 0):
                            self._log(
                                "SEG",
                                "续写锁定：overlap 不重采。",
                                level="INFO",
                                icon="🔒",
                                segment=f"{idx}/{len(mapped_windows)}",
                                video_overlap=overlap_v,
                                audio_overlap=overlap_a,
                            )

                seg_input_latent = self._rebuild_from_parts(seg_video_input, seg_audio_input)

                seg_result = self._run_single_pipeline(
                    引导器=segment_guider,
                    采样器=采样器,
                    latent_input=seg_input_latent,
                    一采西格玛文本=一采西格玛文本,
                    添加噪波=添加噪波,
                    噪波种子=噪波种子,
                    启动二采=(启动二采 and (not segment_use_delayed_segmented_second_pass)),
                    二采使用放大模型=二采使用放大模型,
                    二采放大模型=二采放大模型,
                    二采西格玛=二采西格玛,
                    西格玛=西格玛,
                    二采引导器=segment_second_pass_guider,
                    二采采样器=segment_second_pass_sampler,
                    二采噪波=二采噪波,
                    二采VAE=二采VAE,
                    segment_label=f"{idx}/{len(mapped_windows)}",
                    返回一采结果=True,
                    是否图生视频=是否图生视频,
                    二采latent输入图像=二采latent输入图像,
                    二采图生视频VAE=continuation_vae,
                    应用起始图生视频=bool(是否图生视频 and pos == 1),
                    应用二采图生视频=False,
                )

                seg_output = seg_result[0]
                seg_denoised = seg_result[1] if len(seg_result) > 1 else seg_result[0]
                seg_first_output = seg_result[2] if len(seg_result) > 2 else seg_output
                seg_first_denoised = seg_result[3] if len(seg_result) > 3 else seg_denoised

                if 保存分段latent:
                    seg_meta = {
                        "segment_index": idx,
                        "segment_total": len(mapped_windows),
                        "video_start": start_v,
                        "video_end_exclusive": end_v,
                        "video_overlap": overlap_v,
                        "audio_start": start_a,
                        "audio_end_exclusive": end_a,
                        "audio_overlap": overlap_a,
                        "input_shape": str(self._latent_shape_text(seg_input_latent)),
                        "output_shape": str(self._latent_shape_text(seg_output)),
                        "denoised_shape": str(self._latent_shape_text(seg_denoised)),
                        "plan_item": copy.deepcopy(item.get("meta", item)),
                    }
                    try:
                        self._save_segment_latent_pair(分段latent前缀, idx, seg_output, seg_denoised, seg_meta)
                    except Exception as save_error:
                        self._log("SAVE", f"保存分段 latent 失败，但继续采样：{save_error}", level="WARN", icon="⚠️", segment=idx)

                if has_audio:
                    seg_out_video, seg_out_audio = self._try_split_av_latent(seg_output)
                    seg_den_video, seg_den_audio = self._try_split_av_latent(seg_denoised)
                    seg_first_video, seg_first_audio = self._try_split_av_latent(seg_first_output)
                    seg_first_den_video, seg_first_den_audio = self._try_split_av_latent(seg_first_denoised)
                    if seg_out_video is None or seg_den_video is None:
                        raise RuntimeError("分段结果不是可拆分的 AV latent，无法继续 V2V 分段拼接。")
                    if seg_first_video is None or seg_first_den_video is None:
                        raise RuntimeError("分段一采结果不是可拆分的 AV latent，无法继续作为下一段续写母体。")
                else:
                    seg_out_video, seg_out_audio = seg_output, None
                    seg_den_video, seg_den_audio = seg_denoised, None
                    seg_first_video, seg_first_audio = seg_first_output, None
                    seg_first_den_video, seg_first_den_audio = seg_first_denoised, None

                seg_out_video = self._match_temporal_length(
                    seg_out_video,
                    seg_source_video_len,
                    prefer_video=True,
                    keep="head",
                    reason=("official_guide_output_trim" if used_official_video_continuation else "segment_video_length_align"),
                )
                seg_den_video = self._match_temporal_length(
                    seg_den_video,
                    seg_source_video_len,
                    prefer_video=True,
                    keep="head",
                    reason=("official_guide_denoised_trim" if used_official_video_continuation else "segment_video_denoised_length_align"),
                )
                if has_audio and seg_out_audio is not None:
                    seg_out_audio = self._match_temporal_length(
                        seg_out_audio,
                        seg_source_audio_len,
                        prefer_video=False,
                        keep="head",
                        reason="segment_audio_length_align",
                    )
                    seg_den_audio = self._match_temporal_length(
                        seg_den_audio,
                        seg_source_audio_len,
                        prefer_video=False,
                        keep="head",
                        reason="segment_audio_denoised_length_align",
                    )

                seg_first_video = self._match_temporal_length(
                    seg_first_video,
                    seg_source_video_len,
                    prefer_video=True,
                    keep="head",
                    reason=("official_guide_first_output_trim" if used_official_video_continuation else "segment_first_video_length_align"),
                )
                seg_first_den_video = self._match_temporal_length(
                    seg_first_den_video,
                    seg_source_video_len,
                    prefer_video=True,
                    keep="head",
                    reason=("official_guide_first_denoised_trim" if used_official_video_continuation else "segment_first_video_denoised_length_align"),
                )
                if has_audio and seg_first_audio is not None:
                    seg_first_audio = self._match_temporal_length(
                        seg_first_audio,
                        seg_source_audio_len,
                        prefer_video=False,
                        keep="head",
                        reason="segment_first_audio_length_align",
                    )
                    seg_first_den_audio = self._match_temporal_length(
                        seg_first_den_audio,
                        seg_source_audio_len,
                        prefer_video=False,
                        keep="head",
                        reason="segment_first_audio_denoised_length_align",
                    )

                if segment_use_delayed_segmented_second_pass:
                    delayed_second_pass_segments.append({
                        "index": idx,
                        "overlap_v": overlap_v,
                        "overlap_a": overlap_a,
                        "has_audio": has_audio,
                        "first_video_output": seg_first_video,
                        "first_audio_output": seg_first_audio,
                    })

                if pos == 1:
                    accumulated_video_output = seg_out_video
                    accumulated_video_denoised = seg_den_video
                    accumulated_audio_output = seg_out_audio
                    accumulated_audio_denoised = seg_den_audio
                else:
                    old_accumulated_video_output = accumulated_video_output
                    old_accumulated_video_denoised = accumulated_video_denoised
                    if (not strict_latent_continuity) and 续写输出线性融合 and overlap_v > 0:
                        accumulated_video_output = self._linear_overlap_blend(accumulated_video_output, seg_out_video, overlap_v, prefer_video=True)
                        accumulated_video_denoised = self._linear_overlap_blend(accumulated_video_denoised, seg_den_video, overlap_v, prefer_video=True)
                    else:
                        seg_out_video_fresh = self._slice_latent_temporal(seg_out_video, overlap_v, self._get_temporal_length(seg_out_video, prefer_video=True), prefer_video=True)
                        seg_den_video_fresh = self._slice_latent_temporal(seg_den_video, overlap_v, self._get_temporal_length(seg_den_video, prefer_video=True), prefer_video=True)
                        accumulated_video_output = self._concat_latent_temporal([accumulated_video_output, seg_out_video_fresh], prefer_video=True)
                        accumulated_video_denoised = self._concat_latent_temporal([accumulated_video_denoised, seg_den_video_fresh], prefer_video=True)
                        for obj in (seg_out_video_fresh, seg_den_video_fresh):
                            try:
                                del obj
                            except Exception:
                                pass
                    for obj in (old_accumulated_video_output, old_accumulated_video_denoised):
                        try:
                            del obj
                        except Exception:
                            pass

                    if has_audio:
                        old_accumulated_audio_output = accumulated_audio_output
                        old_accumulated_audio_denoised = accumulated_audio_denoised
                        if (not strict_latent_continuity) and 续写输出线性融合 and overlap_a > 0:
                            accumulated_audio_output = self._linear_overlap_blend(accumulated_audio_output, seg_out_audio, overlap_a, prefer_video=False)
                            accumulated_audio_denoised = self._linear_overlap_blend(accumulated_audio_denoised, seg_den_audio, overlap_a, prefer_video=False)
                        else:
                            seg_out_audio_fresh = self._slice_latent_temporal(seg_out_audio, overlap_a, self._get_temporal_length(seg_out_audio, prefer_video=False), prefer_video=False)
                            seg_den_audio_fresh = self._slice_latent_temporal(seg_den_audio, overlap_a, self._get_temporal_length(seg_den_audio, prefer_video=False), prefer_video=False)
                            accumulated_audio_output = self._concat_latent_temporal([accumulated_audio_output, seg_out_audio_fresh], prefer_video=False)
                            accumulated_audio_denoised = self._concat_latent_temporal([accumulated_audio_denoised, seg_den_audio_fresh], prefer_video=False)
                            for obj in (seg_out_audio_fresh, seg_den_audio_fresh):
                                try:
                                    del obj
                                except Exception:
                                    pass
                        for obj in (old_accumulated_audio_output, old_accumulated_audio_denoised):
                            try:
                                del obj
                            except Exception:
                                pass

                if pos < len(mapped_windows):
                    next_item = mapped_windows[pos]
                    next_video_start = int(next_item["video_start"])
                    next_overlap_v = max(0, int(end_v - next_video_start))
                    previous_segment_video_output = self._tail_latent_temporal(seg_first_video, next_overlap_v, prefer_video=True) if next_overlap_v > 0 else None

                    if has_audio:
                        next_audio_start = int(next_item["audio_start"])
                        next_overlap_a = max(0, int(end_a - next_audio_start))
                        previous_segment_audio_output = self._tail_latent_temporal(seg_first_audio, next_overlap_a, prefer_video=False) if next_overlap_a > 0 else None
                    else:
                        previous_segment_audio_output = None

                    self._flow_print(
                        self._flow_stage_text(1, f"{idx}/{len(mapped_windows)}"),
                        self._flow_stage_text(1, f"{idx + 1}/{len(mapped_windows)}"),
                        note=("续写缓存" if segment_use_delayed_segmented_second_pass else "续写"),
                    )
                else:
                    previous_segment_video_output = None
                    previous_segment_audio_output = None

                previous_video_window = (start_v, end_v)
                previous_audio_window = (start_a, end_a)

                for obj in (
                    seg_result,
                    seg_output,
                    seg_denoised,
                    seg_first_output,
                    seg_first_denoised,
                    seg_input_latent,
                    seg_video_source,
                    seg_audio_source,
                    seg_video_input,
                    seg_audio_input,
                    seg_out_video,
                    seg_den_video,
                    seg_out_audio,
                    seg_den_audio,
                    seg_first_video,
                    seg_first_den_video,
                    seg_first_audio,
                    seg_first_den_audio,
                ):
                    try:
                        del obj
                    except Exception:
                        pass

                self._release_vram(f"分段 {idx}/{len(mapped_windows)} 完成", aggressive=True)
        finally:
            for obj in (source_video_latent, source_audio_latent, timing_video_latent):
                try:
                    del obj
                except Exception:
                    pass

        final_output = self._rebuild_from_parts(accumulated_video_output, accumulated_audio_output)
        final_denoised = self._rebuild_from_parts(accumulated_video_denoised, accumulated_audio_denoised)
        self._log(
            "SEG",
            "所有分段一采处理完成，已自动拼接为阶段性 latent。",
            level="OK",
            icon="✅",
            final_output_shape=self._latent_shape_text(final_output),
            final_denoised_shape=self._latent_shape_text(final_denoised),
        )

        if segment_use_delayed_segmented_second_pass:
            self._flow_print(
                self._flow_stage_text(1, f"{len(mapped_windows)}/{len(mapped_windows)}"),
                self._flow_stage_text(2, f"1/{len(mapped_windows)}"),
                note="开始二采",
            )
            return self._run_delayed_segmented_second_pass(
                delayed_second_pass_segments=delayed_second_pass_segments,
                引导器=引导器,
                采样器=采样器,
                添加噪波=添加噪波,
                噪波种子=噪波种子,
                二采使用放大模型=二采使用放大模型,
                二采放大模型=二采放大模型,
                二采西格玛=二采西格玛,
                二采VAE=二采VAE,
                二采引导器=二采引导器,
                二采采样器=二采采样器,
                二采噪波=二采噪波,
                continuation_vae=continuation_vae,
                supports_official_continuation=supports_official_continuation,
                官方式latent续写=官方式latent续写,
                续写重叠引导强度=续写重叠引导强度,
                续写输出线性融合=续写输出线性融合,
                strict_latent_continuity=strict_latent_continuity,
                是否图生视频=是否图生视频,
                二采latent输入图像=二采latent输入图像,
                二采图生视频VAE=continuation_vae,
                segment_prompt_context=segment_prompt_context,
            )

        return final_output, final_denoised

    def _build_windows_from_auto_settings(self, Latent图像, 启动分段, 单段秒数, 分段重叠秒数, 帧率, 总帧数覆盖=0, 总时长秒数覆盖=0.0):
        if not 启动分段:
            return []

        source_video_latent, source_audio_latent = self._try_split_av_latent(Latent图像)
        has_audio = source_video_latent is not None and source_audio_latent is not None
        timing_video_latent = source_video_latent if has_audio else Latent图像

        latent_video_units = self._get_temporal_length(timing_video_latent, prefer_video=True)
        if latent_video_units <= 1:
            return []

        planning_total_frames = self._resolve_segment_total_frames(
            latent_total_units=latent_video_units,
            fps=帧率,
            total_frames_override=总帧数覆盖,
            total_seconds_override=总时长秒数覆盖,
        )
        planning_total_frames = max(1, int(planning_total_frames))
        total_seconds = max(0.0, float(planning_total_frames - 1) / float(帧率))
        if planning_total_frames <= 1 or total_seconds <= float(单段秒数):
            return []

        max_frames = min(planning_total_frames, self._seconds_to_valid_frames(单段秒数, 帧率))
        base_overlap_frames = self._seconds_to_valid_frames(max(0.0, 分段重叠秒数), 帧率)
        base_overlap_frames = max(1, min(base_overlap_frames, max_frames - 1))
        windows = self._plan_segment_windows(planning_total_frames, max_frames, base_overlap_frames)

        total_audio_steps = self._get_temporal_length(source_audio_latent, prefer_video=False) if has_audio else 0

        mapped_windows = []
        for idx, (start_f, end_f) in enumerate(windows, start=1):
            start_v, end_v = self._map_window_to_units(start_f, end_f, latent_video_units, planning_total_frames)
            start_a = end_a = 0
            if has_audio:
                start_a, end_a = self._map_window_to_units(start_f, end_f, total_audio_steps, planning_total_frames)
                if idx == len(windows):
                    end_a = total_audio_steps
            mapped_windows.append({
                "index": idx,
                "frame_start": int(start_f),
                "frame_end": int(end_f),
                "video_start": int(start_v),
                "video_end": int(end_v),
                "audio_start": int(start_a),
                "audio_end": int(end_a),
                "meta": {
                    "来源": "auto",
                    "frame_start": int(start_f),
                    "frame_end_exclusive": int(end_f),
                    "video_start": int(start_v),
                    "video_end_exclusive": int(end_v),
                    "audio_start": int(start_a),
                    "audio_end_exclusive": int(end_a),
                },
            })
        return mapped_windows

    def sample(
        self,
        引导器,
        采样器,
        Latent图像,
        预设配置,
        一采西格玛文本,
        添加噪波,
        噪波种子,
        启动二采,
        二采使用放大模型,
        二采放大模型,
        二采西格玛,
        启动分段,
        单段秒数,
        分段重叠秒数,
        帧率,
        总帧数覆盖,
        总时长秒数覆盖,
        优先使用分段方案JSON,
        分段方案JSON,
        保存分段latent,
        分段latent前缀,
        官方式latent续写,
        续写重叠引导强度,
        续写输出线性融合,
        是否图生视频,
        二采latent输入图像=None,
        二采引导器=None,
        二采采样器=None,
        二采噪波=None,
        西格玛=None,
        二采VAE=None,
        续写VAE=None,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        **kwargs,
    ):
        if comfy is None:
            raise RuntimeError("未能导入 comfy.sample / comfy.nested_tensor，请确认当前运行环境是完整的 ComfyUI。")

        启用空行分段提示词 = bool(kwargs.get("按空行分段提示词", False))
        分段正向提示词文本 = kwargs.get("分段正向提示词文本", "")
        分段负向提示词文本 = kwargs.get("分段负向提示词文本", "")
        分段提示词CLIP = kwargs.get("分段提示词CLIP", None)
        分段正向提示词文本, 分段负向提示词文本, prompt_text_info = self._resolve_segment_prompt_texts(
            启用空行分段提示词=启用空行分段提示词,
            分段正向提示词文本=分段正向提示词文本,
            分段负向提示词文本=分段负向提示词文本,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
            unique_id=unique_id,
        )
        if prompt_text_info["positive_from_workflow"] or prompt_text_info["negative_from_workflow"]:
            self._log(
                "PROMPT",
                "已读取工作流 CLIP 文本，空行即分段。",
                level="INFO",
                icon="🧷",
                positive_from_workflow=bool(prompt_text_info["positive_from_workflow"]),
                negative_from_workflow=bool(prompt_text_info["negative_from_workflow"]),
            )
        (
            一采西格玛文本,
            添加噪波,
            启动二采,
            二采使用放大模型,
            二采放大模型,
            二采西格玛,
            _preset,
        ) = self._apply_preset(
            预设配置=预设配置,
            一采西格玛文本=一采西格玛文本,
            添加噪波=添加噪波,
            启动二采=启动二采,
            二采使用放大模型=二采使用放大模型,
            二采放大模型=二采放大模型,
            二采西格玛=二采西格玛,
        )

        if bool(启动二采):
            second_pass_guider_route = "二采引导器" if 二采引导器 is not None else "引导器"
            second_pass_sampler_route = "二采采样器" if 二采采样器 is not None else "采样器"
            self._log(
                "FLOW",
                "二采路由已就绪，未接端口自动回退一采。",
                level="INFO",
                icon="🧭",
                second_pass_guider=second_pass_guider_route,
                second_pass_sampler=second_pass_sampler_route,
            )

        mapped_windows = []
        plan_windows = []
        if bool(优先使用分段方案JSON) and str(分段方案JSON).strip():
            plan_windows = self._parse_plan_json(分段方案JSON)
            mapped_windows = plan_windows
            self._log(
                "SEG",
                "已启用分段方案 JSON。",
                level="INFO",
                icon="🧩",
                segments=len(mapped_windows),
            )
        else:
            mapped_windows = self._build_windows_from_auto_settings(
                Latent图像=Latent图像,
                启动分段=启动分段,
                单段秒数=单段秒数,
                分段重叠秒数=分段重叠秒数,
                帧率=帧率,
                总帧数覆盖=总帧数覆盖,
                总时长秒数覆盖=总时长秒数覆盖,
            )
            if mapped_windows:
                self._log(
                    "SEG",
                    "未用分段方案 JSON，改用自动分段。",
                    level="INFO",
                    icon="🧭",
                    segments=len(mapped_windows),
                )
        segment_prompt_context = None
        if mapped_windows:
            segment_prompt_context = self._build_segment_prompt_context(
                segment_total=len(mapped_windows),
                启用空行分段提示词=启用空行分段提示词,
                分段提示词CLIP=分段提示词CLIP,
                分段正向提示词文本=分段正向提示词文本,
                分段负向提示词文本=分段负向提示词文本,
            )
        elif bool(启用空行分段提示词):
            self._log(
                "PROMPT",
                "当前未进入分段链路，按空行分段提示词不会生效。",
                level="WARN",
                icon="⚠️",
            )


        img_to_video_vae = self._resolve_img_to_video_vae(二采VAE=二采VAE, 续写VAE=续写VAE)
        if bool(是否图生视频):
            self._log(
                "I2V",
                "图生视频模式已开启：一采起始 latent 先吃原图；分段仅首段注入；第二段及以后继续沿用当前续写逻辑。",
                level="INFO",
                icon="🖼️",
                image_connected=(二采latent输入图像 is not None),
                vae_route=("二采VAE/续写VAE已就绪" if img_to_video_vae is not None else "未连接二采VAE/续写VAE"),
            )

        if not mapped_windows:
            self._log("SEG", "未进入内部循环，直接走单段采样。", level="INFO", icon="🧭")
            return self._run_single_pipeline(
                引导器=引导器,
                采样器=采样器,
                latent_input=Latent图像,
                一采西格玛文本=一采西格玛文本,
                添加噪波=添加噪波,
                噪波种子=噪波种子,
                启动二采=启动二采,
                二采使用放大模型=二采使用放大模型,
                二采放大模型=二采放大模型,
                二采西格玛=二采西格玛,
                西格玛=西格玛,
                二采引导器=二采引导器,
                二采采样器=二采采样器,
                二采噪波=二采噪波,
                二采VAE=二采VAE,
                segment_label="full-no-loop",
                是否图生视频=是否图生视频,
                二采latent输入图像=二采latent输入图像,
                二采图生视频VAE=img_to_video_vae,
                应用起始图生视频=bool(是否图生视频),
                应用二采图生视频=False,
            )

        return self._run_segment_windows(
            mapped_windows=mapped_windows,
            引导器=引导器,
            采样器=采样器,
            Latent图像=Latent图像,
            一采西格玛文本=一采西格玛文本,
            添加噪波=添加噪波,
            噪波种子=噪波种子,
            启动二采=启动二采,
            二采使用放大模型=二采使用放大模型,
            二采放大模型=二采放大模型,
            二采西格玛=二采西格玛,
            西格玛=西格玛,
            二采引导器=二采引导器,
            二采采样器=二采采样器,
            二采噪波=二采噪波,
            二采VAE=二采VAE,
            续写VAE=续写VAE,
            保存分段latent=保存分段latent,
            分段latent前缀=分段latent前缀,
            官方式latent续写=官方式latent续写,
            续写重叠引导强度=续写重叠引导强度,
            续写输出线性融合=续写输出线性融合,
            是否图生视频=是否图生视频,
            二采latent输入图像=二采latent输入图像,
            segment_prompt_context=segment_prompt_context,
        )


NODE_CLASS_MAPPINGS = {
    "NanFengSamplerAdvancedV2V": NanFengSamplerAdvancedV2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanFengSamplerAdvancedV2V": "南风采样器V2V",
}







