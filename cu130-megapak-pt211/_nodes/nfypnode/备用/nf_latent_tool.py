import copy
import json
import math

try:
    from nodes import NODE_CLASS_MAPPINGS as CORE_NODE_CLASS_MAPPINGS
except Exception:
    CORE_NODE_CLASS_MAPPINGS = {}


def _normalize_node_output(result):
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    if hasattr(result, "args") and isinstance(result.args, tuple):
        return result.args
    return (result,)


class NanFengLatentTool:
    CATEGORY = "南风阳平/工具"
    RETURN_TYPES = (
        "LATENT",  # 原始latent
        "LATENT",  # 视频latent
        "LATENT",  # 音频latent
        "STRING",  # 类型判断
        "STRING",  # 调试文本
        "STRING",  # 原始shape
        "STRING",  # 拆分状态
        "STRING",  # 分段方案JSON
        "INT",     # 原始时间维
        "INT",     # 视频时间维
        "INT",     # 音频时间维
        "INT",     # 推测原始帧数
        "INT",     # 推测总时长_毫秒
        "INT",     # 总段数
    )
    RETURN_NAMES = (
        "原始latent",
        "视频latent",
        "音频latent",
        "类型判断",
        "调试文本",
        "原始shape",
        "拆分状态",
        "分段方案JSON",
        "原始时间维",
        "视频时间维",
        "音频时间维",
        "推测原始帧数",
        "推测总时长_毫秒",
        "总段数",
    )
    FUNCTION = "inspect"

    # 这里先写死：最大单段 10 秒，小尾巴阈值 3 秒
    MAX_SEGMENT_MS_DEFAULT = 10000
    MIN_TAIL_MS_DEFAULT = 3000

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "尝试拆分AV": ("BOOLEAN", {"default": True}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "均分所有分段": ("BOOLEAN", {"default": True}),
                "打印到控制台": ("BOOLEAN", {"default": True}),
            }
        }

    def _shallow_copy_latent(self, latent):
        if isinstance(latent, dict):
            return dict(latent)
        return copy.copy(latent)

    def _get_samples(self, latent):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise RuntimeError("LATENT 数据格式不正确，缺少 samples。")
        return latent["samples"]

    def _shape_text(self, latent):
        try:
            samples = self._get_samples(latent)
            return str(tuple(samples.shape))
        except Exception:
            return "unknown"

    def _temporal_axis_guess(self, latent):
        samples = self._get_samples(latent)
        if getattr(samples, "is_nested", False):
            return None
        if samples.ndim == 5:
            return 2  # 视频 latent: (B, C, T, H, W)
        if samples.ndim == 4:
            return 2  # 音频 latent: (B, C, T, D)
        if samples.ndim == 3:
            return 2
        return None

    def _temporal_length_guess(self, latent):
        try:
            samples = self._get_samples(latent)
            axis = self._temporal_axis_guess(latent)
            if axis is None:
                return -1
            return int(samples.shape[axis])
        except Exception:
            return -1

    def _make_safe_latent(self, latent):
        return self._shallow_copy_latent(latent)

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
            "encode",
            "decode",
        ]

        for fn_name in candidate_names:
            if not fn_name or fn_name in tried:
                continue
            tried.append(fn_name)
            if hasattr(node, fn_name):
                result = getattr(node, fn_name)(**kwargs)
                return _normalize_node_output(result)

        raise RuntimeError(f"{node_name} 未找到可调用方法：{tried}")

    def _infer_video_frames_from_video_latent(self, video_latent):
        video_t = self._temporal_length_guess(video_latent)
        if video_t <= 0:
            return -1
        return int((video_t - 1) * 8 + 1)

    def _infer_total_ms_from_video_latent(self, video_latent, fps):
        frames = self._infer_video_frames_from_video_latent(video_latent)
        if frames <= 0 or fps <= 0:
            return -1
        return int(round(max(frames - 1, 0) * 1000.0 / float(fps)))

    def _build_segment_durations_ms(self, total_ms, max_segment_ms=None, min_tail_ms=None):
        max_segment_ms = int(max_segment_ms or self.MAX_SEGMENT_MS_DEFAULT)
        min_tail_ms = int(min_tail_ms or self.MIN_TAIL_MS_DEFAULT)

        if total_ms <= 0:
            return []
        if total_ms <= max_segment_ms:
            return [int(total_ms)]

        full_count = total_ms // max_segment_ms
        remainder = total_ms % max_segment_ms

        # 整除，直接全是满段
        if remainder == 0:
            return [max_segment_ms] * int(full_count)

        # 尾巴不小，直接 10 + 10 + ... + remainder
        if remainder >= min_tail_ms:
            return [max_segment_ms] * int(full_count) + [int(remainder)]

        # 这里就是你最讨厌的 11 秒 => 10 + 1 这种情况
        # 处理策略：保留前面的满段，只把“最后一个满段 + 小尾巴”平均拆成 2 段
        if full_count >= 1:
            durations = [max_segment_ms] * int(full_count - 1)
            tail_total = int(max_segment_ms + remainder)
            left = tail_total // 2
            right = tail_total - left
            durations.extend([left, right])
            return durations

        return [int(total_ms)]

    def _build_equal_segment_durations_ms(self, total_ms, max_segment_ms=None):
        max_segment_ms = int(max_segment_ms or self.MAX_SEGMENT_MS_DEFAULT)

        if total_ms <= 0:
            return []
        if total_ms <= max_segment_ms:
            return [int(total_ms)]

        segment_count = int(math.ceil(float(total_ms) / float(max_segment_ms)))
        segment_count = max(1, segment_count)

        base = int(total_ms // segment_count)
        remainder = int(total_ms % segment_count)

        durations = []
        for i in range(segment_count):
            extra = 1 if i < remainder else 0
            durations.append(int(base + extra))

        return durations

    def _durations_to_ranges_ms(self, durations_ms):
        ranges = []
        current = 0
        for d in durations_ms:
            start_ms = int(current)
            end_ms = int(current + d)
            ranges.append((start_ms, end_ms))
            current = end_ms
        return ranges

    def _ms_range_to_audio_token_slice(self, total_ms, total_tokens, start_ms, end_ms, is_last=False):
        if total_ms <= 0 or total_tokens <= 0:
            return 0, 0

        start_idx = int(round((start_ms / total_ms) * total_tokens))
        if is_last or end_ms >= total_ms:
            end_idx = total_tokens
        else:
            end_idx = int(round((end_ms / total_ms) * total_tokens))

        start_idx = max(0, min(total_tokens, start_idx))
        end_idx = max(start_idx, min(total_tokens, end_idx))
        return start_idx, end_idx

    def _ms_range_to_video_frame_range(self, total_ms, total_frames, start_ms, end_ms, is_last=False):
        # 这里按“独立成段时保留边界帧”的口径算
        # 所以相邻段会共享 1 帧，这就是你看到“总和多一帧”的主要原因
        if total_ms <= 0 or total_frames <= 0:
            return 0, 0, 0

        total_intervals = max(total_frames - 1, 0)
        start_interval = int(round((start_ms / total_ms) * total_intervals))
        if is_last or end_ms >= total_ms:
            end_interval = total_intervals
        else:
            end_interval = int(round((end_ms / total_ms) * total_intervals))

        start_interval = max(0, min(total_intervals, start_interval))
        end_interval = max(start_interval, min(total_intervals, end_interval))

        frame_start = start_interval
        frame_end = end_interval
        frame_count = int(frame_end - frame_start + 1)
        return int(frame_start), int(frame_end), int(frame_count)

    def _ms_range_to_video_latent_range(self, total_ms, video_t, start_ms, end_ms, is_last=False):
        # 这里同样保留边界 latent token，方便后面做独立分段
        if total_ms <= 0 or video_t <= 0:
            return 0, 0, 0

        total_intervals = max(video_t - 1, 0)
        start_idx = int(round((start_ms / total_ms) * total_intervals))
        if is_last or end_ms >= total_ms:
            end_idx = total_intervals
        else:
            end_idx = int(round((end_ms / total_ms) * total_intervals))

        start_idx = max(0, min(total_intervals, start_idx))
        end_idx = max(start_idx, min(total_intervals, end_idx))
        length = int(end_idx - start_idx + 1)
        return int(start_idx), int(end_idx), int(length)

    def inspect(self, latent, 尝试拆分AV=True, fps=24.0, 均分所有分段=True, 打印到控制台=True):
        original = self._shallow_copy_latent(latent)
        original_shape = self._shape_text(original)
        original_t = self._temporal_length_guess(original)

        latent_type = "普通LATENT"
        split_status = "未尝试拆分"
        video_latent = self._make_safe_latent(original)
        audio_latent = self._make_safe_latent(original)
        video_shape = "N/A"
        audio_shape = "N/A"
        video_t = -1
        audio_t = -1
        estimated_frames = -1
        estimated_total_ms = -1
        segment_count = 0
        plan_json = "[]"

        plan = []

        if 尝试拆分AV:
            try:
                out = self._call_core_node("LTXVSeparateAVLatent", av_latent=original)
                if len(out) >= 2:
                    video_latent, audio_latent = out[:2]
                    video_shape = self._shape_text(video_latent)
                    audio_shape = self._shape_text(audio_latent)
                    video_t = self._temporal_length_guess(video_latent)
                    audio_t = self._temporal_length_guess(audio_latent)
                    estimated_frames = self._infer_video_frames_from_video_latent(video_latent)
                    estimated_total_ms = self._infer_total_ms_from_video_latent(video_latent, fps)
                    latent_type = "AV_LATENT"
                    split_status = "拆分成功"

                    if audio_t > 0 and estimated_total_ms > 0:
                        if bool(均分所有分段):
                            durations_ms = self._build_equal_segment_durations_ms(estimated_total_ms)
                        else:
                            durations_ms = self._build_segment_durations_ms(estimated_total_ms)
                        ranges_ms = self._durations_to_ranges_ms(durations_ms)
                        segment_count = len(ranges_ms)

                        for idx, (start_ms, end_ms) in enumerate(ranges_ms):
                            is_last = idx == (segment_count - 1)
                            audio_start, audio_end = self._ms_range_to_audio_token_slice(
                                estimated_total_ms,
                                audio_t,
                                start_ms,
                                end_ms,
                                is_last=is_last,
                            )
                            vlat_start, vlat_end, vlat_len = self._ms_range_to_video_latent_range(
                                estimated_total_ms,
                                video_t,
                                start_ms,
                                end_ms,
                                is_last=is_last,
                            )
                            frame_start, frame_end, frame_len = self._ms_range_to_video_frame_range(
                                estimated_total_ms,
                                estimated_frames,
                                start_ms,
                                end_ms,
                                is_last=is_last,
                            )
                            plan.append({
                                "段": idx + 1,
                                "开始毫秒": int(start_ms),
                                "结束毫秒": int(end_ms),
                                "时长毫秒": int(end_ms - start_ms),
                                "音频token开始": int(audio_start),
                                "音频token结束_exclusive": int(audio_end),
                                "音频token长度": int(max(0, audio_end - audio_start)),
                                "视频latent开始": int(vlat_start),
                                "视频latent结束_inclusive": int(vlat_end),
                                "视频latent长度": int(vlat_len),
                                "原始帧开始": int(frame_start),
                                "原始帧结束_inclusive": int(frame_end),
                                "原始帧长度": int(frame_len),
                            })

                        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
                    else:
                        split_status = "拆分成功，但无法估算总时长或音频时间维"
                else:
                    split_status = "拆分节点返回结果不足"
            except Exception as e:
                split_status = f"拆分失败: {e}"
                latent_type = "普通LATENT/或非标准AV_LATENT"

        info = {
            "类型判断": latent_type,
            "拆分状态": split_status,
            "原始shape": original_shape,
            "原始时间维": original_t,
            "视频shape": video_shape,
            "视频时间维": video_t,
            "音频shape": audio_shape,
            "音频时间维": audio_t,
            "fps": float(fps),
            "均分所有分段": bool(均分所有分段),
            "单段最大毫秒": self.MAX_SEGMENT_MS_DEFAULT,
            "小尾巴重分配阈值毫秒": self.MIN_TAIL_MS_DEFAULT,
            "推测原始视频帧数": int(estimated_frames),
            "推测总时长_毫秒": int(estimated_total_ms),
            "推测总时长_秒": round((estimated_total_ms / 1000.0), 6) if estimated_total_ms > 0 else -1,
            "总段数": int(segment_count),
            "分段方案": plan,
        }
        debug_text = json.dumps(info, ensure_ascii=False, indent=2)

        if 打印到控制台:
            print("[南风latent处理] 检查结果")
            print(debug_text)

        return (
            original,
            video_latent,
            audio_latent,
            latent_type,
            debug_text,
            original_shape,
            split_status,
            plan_json,
            int(original_t),
            int(video_t),
            int(audio_t),
            int(estimated_frames),
            int(estimated_total_ms),
            int(segment_count),
        )


NODE_CLASS_MAPPINGS = {
    "NanFengLatentTool": NanFengLatentTool,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanFengLatentTool": "南风latent处理",
}
