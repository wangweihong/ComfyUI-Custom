
import copy
import gc
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F

try:
    import comfy.sample
    import comfy.nested_tensor
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


class NanFengSamplerAdvanced:
    CATEGORY = "南风阳平/采样"
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"

    PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets", "sampler")
    DEFAULT_X2_MODEL = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"



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
            },
            "optional": {
                "西格玛": ("SIGMAS",),
                "二采VAE": ("VAE",),
            },
        }

    def _log(self, stage, message, level="INFO", icon="ℹ️", **kwargs):
        ts = datetime.now().strftime("%H:%M:%S")
        suffix = ""
        if kwargs:
            parts = [f"{k}={v}" for k, v in kwargs.items()]
            suffix = " | " + " | ".join(parts)
        print(f"{icon} [{ts}] [南风采样器][{level}][{stage}] {message}{suffix}")

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
            resized = F.interpolate(
                samples,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            return resized, (h, w), (target_h, target_w)

        if samples.ndim == 5:
            b, c, t, h, w = samples.shape
            target_h, target_w = self._scale_hw(h, w, factor)
            if target_h == h and target_w == w:
                return samples, (h, w), (target_h, target_w)

            flat = samples.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            flat = F.interpolate(
                flat,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
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

        self._log(
            "LATENT",
            f"{reason} video latent 空间缩放完成。",
            level="OK",
            icon="📐",
            factor=factor,
            before_hw=before_hw,
            after_hw=after_hw,
            tensor_shape=tuple(resized.shape),
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
                        free_memory(
                            1024 * 1024 * 1024,
                            torch.device("cuda"),
                            keep_loaded=[]
                        )
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

        if stage:
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

    def _apply_preset(
        self,
        预设配置,
        一采西格玛文本,
        添加噪波,
        启动二采,
        二采使用放大模型,
        二采放大模型,
        二采西格玛,
    ):
        preset_name = str(预设配置).strip()

        if preset_name == "自定义":
            self._log(
                "PRESET",
                "当前为自定义模式，保留界面手填参数；不使用代码内置预设。",
                level="INFO",
                icon="🧭",
            )
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
            "已从 JSON 文件加载预设参数。",
            level="INFO",
            icon="📦",
            preset=preset_name,
            file=preset_path,
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

    def _call_sampler_advanced(self, noise_obj, guider, sampler, sigmas, latent_image):
        if SamplerCustomAdvanced is None:
            raise RuntimeError(
                "未能导入 comfy_extras.nodes_custom_sampler.SamplerCustomAdvanced，请确认 ComfyUI 自带高级采样节点可用。"
            )

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
        candidate_names = [
            function_name,
            "execute",
            "sample",
            "load",
            "load_model",
            "upscale",
            "separate",
            "concat",
        ]

        for fn_name in candidate_names:
            if not fn_name or fn_name in tried:
                continue
            tried.append(fn_name)
            if hasattr(node, fn_name):
                result = getattr(node, fn_name)(**kwargs)
                return _normalize_node_output(result)

        raise RuntimeError(f"{node_name} 未找到可调用方法：{tried}")

    def _prepare_first_pass_input_for_x2_pipeline(self, latent_image):
        self._log(
            "FLOW",
            "检测到“启动二采 + 启用x2放大模型”，一采前将 video latent 预缩放到 0.5。",
            level="INFO",
            icon="🧭",
        )

        try:
            av_latent = self._shallow_copy_latent(latent_image)
            separate_out = self._call_core_node(
                "LTXVSeparateAVLatent",
                av_latent=av_latent,
            )
            video_latent, audio_latent = separate_out[:2]
            del separate_out
            del av_latent

            half_video_latent = self._resize_video_latent_spatial(
                video_latent,
                factor=0.5,
                reason="一采前预缩放",
            )

            rebuilt_latent = self._call_core_node(
                "LTXVConcatAVLatent",
                video_latent=half_video_latent,
                audio_latent=audio_latent,
            )[0]

            self._log(
                "FLOW",
                "一采输入 AV latent 已重组完成，将以半分辨率进入一采。",
                level="OK",
                icon="✅",
                output_shape=self._latent_shape_text(rebuilt_latent),
            )
            return rebuilt_latent

        except Exception as e:
            self._log(
                "FLOW",
                "AV latent 拆分预缩放失败，尝试直接对输入 latent 做 0.5 空间缩放。",
                level="WARN",
                icon="⚠️",
                error=str(e),
            )
            return self._resize_video_latent_spatial(
                latent_image,
                factor=0.5,
                reason="一采前直接预缩放",
            )

    def _run_second_pass(
        self,
        first_pass_output,
        guider,
        sampler,
        添加噪波,
        噪波种子,
        二采使用放大模型,
        二采放大模型,
        二采西格玛,
        二采VAE=None,
    ):
        second_sigmas = self._parse_sigmas_text(二采西格玛, "二采西格玛")

        av_latent = self._shallow_copy_latent(first_pass_output)
        del first_pass_output
        self._release_vram("一采输出已转入二采前处理", aggressive=True)

        separate_out = self._call_core_node(
            "LTXVSeparateAVLatent",
            av_latent=av_latent,
        )
        del av_latent
        self._release_vram("AV latent 拆分完成", aggressive=True)

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
                model=二采放大模型,
                input_shape=self._latent_shape_text(video_latent),
            )

            upscale_model = self._call_core_node(
                "LatentUpscaleModelLoader",
                model_name=二采放大模型,
            )[0]

            second_video_latent = self._call_core_node(
                "LTXVLatentUpsampler",
                samples=video_latent,
                upscale_model=upscale_model,
                vae=二采VAE,
            )[0]

            del upscale_model
            if second_video_latent is not video_latent:
                del video_latent

            self._log(
                "UPSCALE",
                "二采前 x2 latent 放大完成。",
                level="OK",
                icon="📈",
                output_shape=self._latent_shape_text(second_video_latent),
            )
            self._release_vram("二采放大完成", aggressive=True)

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

        second_noise = self._get_noise_obj(添加噪波, 噪波种子)
        self._log(
            "SAMPLE",
            "开始执行二采。",
            level="INFO",
            icon="🎯",
            sigma_count=len(second_sigmas),
            latent_shape=self._latent_shape_text(second_input_latent),
        )
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
        self._log("SAMPLE", "二采执行完成。", level="OK", icon="✅")
        return second_result

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
        西格玛=None,
        二采VAE=None,
    ):
        if comfy is None:
            raise RuntimeError("未能导入 comfy.sample / comfy.nested_tensor，请确认当前运行环境是完整的 ComfyUI。")

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

        force_half_first_pass = bool(启动二采 and 二采使用放大模型)

        if force_half_first_pass and not 二采放大模型:
            二采放大模型 = self.DEFAULT_X2_MODEL

        if force_half_first_pass and 二采VAE is None:
            raise RuntimeError("当“启动二采 + 二采使用放大模型”启用时，必须连接“二采VAE”，因为节点会走半分辨率一采 + x2 恢复流程。")

        first_input_latent = Latent图像
        if force_half_first_pass:
            first_input_latent = self._prepare_first_pass_input_for_x2_pipeline(Latent图像)

        first_sigmas = self._resolve_first_sigmas(西格玛, 一采西格玛文本)
        first_noise = self._get_noise_obj(添加噪波, 噪波种子)

        self._log(
            "SAMPLE",
            "开始执行一采。",
            level="INFO",
            icon="🚀",
            preset=预设配置,
            enable_second_pass=启动二采,
            use_second_upscale=二采使用放大模型,
            first_input_shape=self._latent_shape_text(first_input_latent),
            sigma_count=len(first_sigmas) if hasattr(first_sigmas, "__len__") else "unknown",
        )

        first_result = self._call_sampler_advanced(
            noise_obj=first_noise,
            guider=引导器,
            sampler=采样器,
            sigmas=first_sigmas,
            latent_image=first_input_latent,
        )

        self._log("SAMPLE", "一采执行完成。", level="OK", icon="✅")

        if not 启动二采:
            return first_result

        first_pass_output = first_result[0]

        del first_result
        del first_noise
        del first_sigmas
        if first_input_latent is not Latent图像:
            try:
                del first_input_latent
            except Exception:
                pass

        self._release_vram("一采完成，准备进入二采", aggressive=True)

        second_result = self._run_second_pass(
            first_pass_output=first_pass_output,
            guider=引导器,
            sampler=采样器,
            添加噪波=添加噪波,
            噪波种子=噪波种子,
            二采使用放大模型=二采使用放大模型,
            二采放大模型=二采放大模型,
            二采西格玛=二采西格玛,
            二采VAE=二采VAE,
        )
        return second_result


NODE_CLASS_MAPPINGS = {
    "NanFengSamplerAdvanced": NanFengSamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanFengSamplerAdvanced": "南风采样器",
}
