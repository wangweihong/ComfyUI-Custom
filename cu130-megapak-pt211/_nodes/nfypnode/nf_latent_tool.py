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

    MAX_SEGMENT_MS_DEFAULT = 10000
    OVERLAP_MS_DEFAULT = 1000
    MIN_TAIL_MS_DEFAULT = 3000

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "尝试拆分AV": ("BOOLEAN", {"default": True}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "最大单段秒数": ("FLOAT", {"default": cls.MAX_SEGMENT_MS_DEFAULT / 1000.0, "min": 0.1, "max": 300.0, "step": 0.1}),
                "重合时长秒数": ("FLOAT", {"default": cls.OVERLAP_MS_DEFAULT / 1000.0, "min": 0.0, "max": 120.0, "step": 0.1}),
                "均分所有分段": ("BOOLEAN", {"default": True}),
                "小尾巴重分配阈值秒数": ("FLOAT", {"default": cls.MIN_TAIL_MS_DEFAULT / 1000.0, "min": 0.0, "max": 120.0, "step": 0.1}),
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

    def _seconds_to_ms(self, seconds, default_ms):
        if seconds is None:
            return int(default_ms)
        return int(round(max(0.0, float(seconds)) * 1000.0))

    def _split_total_evenly(self, total, parts):
        total = int(total)
        parts = max(1, int(parts))
        base = total // parts
        remainder = total % parts
        out = []
        for i in range(parts):
            extra = 1 if i < remainder else 0
            out.append(int(base + extra))
        return out

    def _validate_segment_params(self, max_segment_ms, overlap_ms, min_tail_ms):
        max_segment_ms = int(max_segment_ms)
        overlap_ms = int(overlap_ms)
        min_tail_ms = int(min_tail_ms)

        if max_segment_ms <= 0:
            raise RuntimeError("最大单段时长必须大于 0。")
        if overlap_ms < 0:
            raise RuntimeError("重合时长不能小于 0。")
        if overlap_ms >= max_segment_ms:
            raise RuntimeError(
                f"重合时长必须小于最大单段时长。当前为 overlap={overlap_ms}ms, max_segment={max_segment_ms}ms"
            )
        if min_tail_ms < 0:
            raise RuntimeError("小尾巴重分配阈值不能小于 0。")

        return max_segment_ms, overlap_ms, min_tail_ms

    def _compute_segment_count_with_overlap(self, total_ms, max_segment_ms, overlap_ms):
        total_ms = int(total_ms)
        max_segment_ms = int(max_segment_ms)
        overlap_ms = int(overlap_ms)

        if total_ms <= 0:
            return 0
        if total_ms <= max_segment_ms:
            return 1

        step_ms = max_segment_ms - overlap_ms
        return int(math.ceil(float(total_ms - overlap_ms) / float(step_ms)))

    def _build_equal_window_durations_ms(self, total_ms, max_segment_ms, overlap_ms):
        total_ms = int(total_ms)
        max_segment_ms = int(max_segment_ms)
        overlap_ms = int(overlap_ms)

        if total_ms <= 0:
            return []
        if total_ms <= max_segment_ms:
            return [int(total_ms)]

        segment_count = self._compute_segment_count_with_overlap(total_ms, max_segment_ms, overlap_ms)
        segment_count = max(1, int(segment_count))

        gross_total_ms = int(total_ms + (segment_count - 1) * overlap_ms)
        durations = self._split_total_evenly(gross_total_ms, segment_count)

        for d in durations:
            if d <= 0:
                raise RuntimeError(f"均分后的分段时长非法：{durations}")
            if d > max_segment_ms:
                raise RuntimeError(
                    f"均分后的某段时长超过最大单段时长：duration={d}ms, max_segment={max_segment_ms}ms"
                )
            if segment_count > 1 and d <= overlap_ms:
                raise RuntimeError(
                    f"均分后的某段时长不大于 overlap，无法形成有效前进步长：duration={d}ms, overlap={overlap_ms}ms"
                )

        return durations

    def _build_greedy_window_durations_ms(self, total_ms, max_segment_ms, overlap_ms, min_tail_ms):
        total_ms = int(total_ms)
        max_segment_ms = int(max_segment_ms)
        overlap_ms = int(overlap_ms)
        min_tail_ms = int(min_tail_ms)

        if total_ms <= 0:
            return []
        if total_ms <= max_segment_ms:
            return [int(total_ms)]

        durations = []
        start_ms = 0
        end_ms = min(total_ms, max_segment_ms)
        durations.append(int(end_ms - start_ms))

        while end_ms < total_ms:
            start_ms = end_ms - overlap_ms
            end_ms = min(total_ms, start_ms + max_segment_ms)
            durations.append(int(end_ms - start_ms))

        if len(durations) >= 2 and durations[-1] < min_tail_ms:
            gross_tail = int(durations[-2] + durations[-1])
            rebalanced = self._split_total_evenly(gross_tail, 2)

            can_use_rebalanced = True
            for d in rebalanced:
                if d <= 0 or d > max_segment_ms:
                    can_use_rebalanced = False
                    break
                if d <= overlap_ms:
                    can_use_rebalanced = False
                    break

            if can_use_rebalanced:
                durations = durations[:-2] + rebalanced

        return durations

    def _window_durations_to_ranges_ms(self, durations_ms, overlap_ms):
        durations_ms = [int(x) for x in durations_ms if int(x) > 0]
        overlap_ms = int(overlap_ms)

        ranges = []
        if not durations_ms:
            return ranges

        start_ms = 0
        for i, duration_ms in enumerate(durations_ms):
            end_ms = int(start_ms + duration_ms)
            ranges.append((int(start_ms), int(end_ms)))
            if i < len(durations_ms) - 1:
                start_ms = int(end_ms - overlap_ms)

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
        return int(start_idx), int(end_idx)

    def _ms_range_to_video_frame_range(self, total_ms, total_frames, start_ms, end_ms, is_last=False):
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

        frame_start = int(start_interval)
        frame_end = int(end_interval)
        frame_count = int(frame_end - frame_start + 1)
        return frame_start, frame_end, frame_count

    def _ms_range_to_video_latent_range(self, total_ms, video_t, start_ms, end_ms, is_last=False):
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

    def inspect(
        self,
        latent,
        尝试拆分AV=True,
        fps=24.0,
        最大单段秒数=10.0,
        重合时长秒数=1.0,
        均分所有分段=True,
        小尾巴重分配阈值秒数=3.0,
        打印到控制台=True,
    ):
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

        max_segment_ms = self._seconds_to_ms(最大单段秒数, self.MAX_SEGMENT_MS_DEFAULT)
        overlap_ms = self._seconds_to_ms(重合时长秒数, self.OVERLAP_MS_DEFAULT)
        min_tail_ms = self._seconds_to_ms(小尾巴重分配阈值秒数, self.MIN_TAIL_MS_DEFAULT)

        try:
            max_segment_ms, overlap_ms, min_tail_ms = self._validate_segment_params(
                max_segment_ms=max_segment_ms,
                overlap_ms=overlap_ms,
                min_tail_ms=min_tail_ms,
            )
        except Exception as e:
            split_status = f"参数非法: {e}"

        if 尝试拆分AV and not str(split_status).startswith("参数非法"):
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
                            durations_ms = self._build_equal_window_durations_ms(
                                total_ms=estimated_total_ms,
                                max_segment_ms=max_segment_ms,
                                overlap_ms=overlap_ms,
                            )
                            segment_mode = "均分所有分段（含overlap）"
                        else:
                            durations_ms = self._build_greedy_window_durations_ms(
                                total_ms=estimated_total_ms,
                                max_segment_ms=max_segment_ms,
                                overlap_ms=overlap_ms,
                                min_tail_ms=min_tail_ms,
                            )
                            segment_mode = "满段优先 + 小尾巴重分配（含overlap）"

                        ranges_ms = self._window_durations_to_ranges_ms(durations_ms, overlap_ms)
                        segment_count = len(ranges_ms)

                        if ranges_ms:
                            last_end_ms = int(ranges_ms[-1][1])
                            if last_end_ms != int(estimated_total_ms):
                                diff = int(estimated_total_ms - last_end_ms)
                                start_ms, end_ms = ranges_ms[-1]
                                ranges_ms[-1] = (int(start_ms), int(end_ms + diff))

                        for idx, (start_ms, end_ms) in enumerate(ranges_ms):
                            is_last = idx == (segment_count - 1)

                            prev_end_ms = ranges_ms[idx - 1][1] if idx > 0 else start_ms
                            next_start_ms = ranges_ms[idx + 1][0] if idx < (segment_count - 1) else end_ms

                            unique_start_ms = int(prev_end_ms if idx > 0 else start_ms)
                            unique_end_ms = int(end_ms)

                            front_overlap_ms = int(max(0, unique_start_ms - start_ms))
                            back_overlap_ms = int(max(0, end_ms - next_start_ms)) if not is_last else 0

                            audio_start, audio_end = self._ms_range_to_audio_token_slice(
                                estimated_total_ms,
                                audio_t,
                                start_ms,
                                end_ms,
                                is_last=is_last,
                            )
                            audio_unique_start, audio_unique_end = self._ms_range_to_audio_token_slice(
                                estimated_total_ms,
                                audio_t,
                                unique_start_ms,
                                unique_end_ms,
                                is_last=is_last,
                            )

                            vlat_start, vlat_end, vlat_len = self._ms_range_to_video_latent_range(
                                estimated_total_ms,
                                video_t,
                                start_ms,
                                end_ms,
                                is_last=is_last,
                            )
                            vlat_unique_start, vlat_unique_end, vlat_unique_len = self._ms_range_to_video_latent_range(
                                estimated_total_ms,
                                video_t,
                                unique_start_ms,
                                unique_end_ms,
                                is_last=is_last,
                            )

                            frame_start, frame_end, frame_len = self._ms_range_to_video_frame_range(
                                estimated_total_ms,
                                estimated_frames,
                                start_ms,
                                end_ms,
                                is_last=is_last,
                            )
                            frame_unique_start, frame_unique_end, frame_unique_len = self._ms_range_to_video_frame_range(
                                estimated_total_ms,
                                estimated_frames,
                                unique_start_ms,
                                unique_end_ms,
                                is_last=is_last,
                            )

                            plan.append({
                                "段": int(idx + 1),

                                "开始毫秒": int(start_ms),
                                "结束毫秒": int(end_ms),
                                "时长毫秒": int(end_ms - start_ms),

                                "新增开始毫秒": int(unique_start_ms),
                                "新增结束毫秒": int(unique_end_ms),
                                "新增时长毫秒": int(max(0, unique_end_ms - unique_start_ms)),

                                "前重合毫秒": int(front_overlap_ms),
                                "后重合毫秒": int(back_overlap_ms),

                                "音频token开始": int(audio_start),
                                "音频token结束_exclusive": int(audio_end),
                                "音频token长度": int(max(0, audio_end - audio_start)),

                                "音频新增token开始": int(audio_unique_start),
                                "音频新增token结束_exclusive": int(audio_unique_end),
                                "音频新增token长度": int(max(0, audio_unique_end - audio_unique_start)),

                                "视频latent开始": int(vlat_start),
                                "视频latent结束_inclusive": int(vlat_end),
                                "视频latent长度": int(vlat_len),

                                "视频latent新增开始": int(vlat_unique_start),
                                "视频latent新增结束_inclusive": int(vlat_unique_end),
                                "视频latent新增长度": int(vlat_unique_len),

                                "原始帧开始": int(frame_start),
                                "原始帧结束_inclusive": int(frame_end),
                                "原始帧长度": int(frame_len),

                                "原始帧新增开始": int(frame_unique_start),
                                "原始帧新增结束_inclusive": int(frame_unique_end),
                                "原始帧新增长度": int(frame_unique_len),
                            })

                        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
                    else:
                        split_status = "拆分成功，但无法估算总时长或音频时间维"
                        segment_mode = "N/A"
                else:
                    split_status = "拆分节点返回结果不足"
                    segment_mode = "N/A"
            except Exception as e:
                split_status = f"拆分失败: {e}"
                latent_type = "普通LATENT/或非标准AV_LATENT"
                segment_mode = "N/A"
        else:
            segment_mode = "N/A"

        info = {
            "类型判断": latent_type,
            "拆分状态": split_status,
            "原始shape": original_shape,
            "原始时间维": int(original_t),
            "视频shape": video_shape,
            "视频时间维": int(video_t),
            "音频shape": audio_shape,
            "音频时间维": int(audio_t),
            "fps": float(fps),

            "最大单段秒数": round(max_segment_ms / 1000.0, 6),
            "最大单段毫秒": int(max_segment_ms),

            "重合时长秒数": round(overlap_ms / 1000.0, 6),
            "重合时长毫秒": int(overlap_ms),

            "均分所有分段": bool(均分所有分段),

            "小尾巴重分配阈值秒数": round(min_tail_ms / 1000.0, 6),
            "小尾巴重分配阈值毫秒": int(min_tail_ms),

            "分段模式": segment_mode,

            "推测原始视频帧数": int(estimated_frames),
            "推测总时长_毫秒": int(estimated_total_ms),
            "推测总时长_秒": round((estimated_total_ms / 1000.0), 6) if estimated_total_ms > 0 else -1,

            "总段数": int(segment_count),
            "分段方案": plan,
        }
        debug_text = json.dumps(info, ensure_ascii=False, indent=2)

        if 打印到控制台:
            print(r"""
======================================================================
 
    ███╗   ██╗ ███████╗██╗   ██╗██████╗
    ████╗  ██║ ██╔════╝╚██╗ ██╔╝██╔══██╗
    ██╔██╗ ██║ █████╗   ╚████╔╝ ██████╔╝
    ██║╚██╗██║ ██╔══╝    ╚██╔╝  ██╔═══╝
    ██║ ╚████║ ██║        ██║   ██║
    ╚═╝  ╚═══╝ ╚═╝        ╚═╝   ╚═╝
 
                    南 风 阳 平 · 自 定 义 节 点
 
======================================================================
""")
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