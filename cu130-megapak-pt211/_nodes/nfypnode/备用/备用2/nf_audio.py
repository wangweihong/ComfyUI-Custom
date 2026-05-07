import json
import logging
import math
import mimetypes
import os
import wave

import folder_paths
import numpy as np
import server
from aiohttp import web

MAX_MARKERS = 64


def _strip_path(path):
    path = (path or "").strip()
    if path.startswith('"'):
        path = path[1:]
    if path.endswith('"'):
        path = path[:-1]
    return path


def _list_input_audio_files():
    input_dir = folder_paths.get_input_directory()
    if not input_dir or not os.path.isdir(input_dir):
        return []

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    discovered = []
    for root, _dirs, files in os.walk(input_dir):
        for filename in files:
            extension = os.path.splitext(filename)[1].lower()
            if extension not in audio_extensions:
                continue
            full_path = os.path.join(root, filename)
            relative_path = os.path.relpath(full_path, input_dir).replace("\\", "/")
            discovered.append(relative_path)

    return sorted(discovered)


def _resolve_audio_path(audio_file):
    audio_file = _strip_path(audio_file)
    if not audio_file:
        raise ValueError("audio_file is empty")

    if os.path.isabs(audio_file) and os.path.isfile(audio_file):
        return audio_file

    try:
        annotated = folder_paths.get_annotated_filepath(audio_file)
        if annotated and os.path.isfile(annotated):
            return annotated
    except Exception:
        pass

    input_candidate = os.path.join(folder_paths.get_input_directory(), audio_file)
    if os.path.isfile(input_candidate):
        return input_candidate

    raise ValueError(f"Audio file not found: {audio_file}")


def _read_waveform_peaks(audio_file, bins=1400):
    audio_path = _resolve_audio_path(audio_file)
    extension = os.path.splitext(audio_path)[1].lower()
    if extension != ".wav":
        raise ValueError("Waveform preview currently supports WAV files")

    with wave.open(audio_path, "rb") as wav_file:
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frame_bytes = wav_file.readframes(frame_count)

    if frame_count <= 0 or sample_rate <= 0:
        return {
            "duration": 0.0,
            "sample_rate": sample_rate,
            "peaks": [],
            "audio_path": audio_path,
        }

    if sample_width == 1:
        samples = np.frombuffer(frame_bytes, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        sign_mask = 1 << 23
        signed = (signed ^ sign_mask) - sign_mask
        samples = signed.astype(np.float32) / float(1 << 23)
    elif sample_width == 4:
        samples = np.frombuffer(frame_bytes, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channel_count > 1:
        samples = samples.reshape(-1, channel_count)
        samples = np.mean(np.abs(samples), axis=1)
    else:
        samples = np.abs(samples)

    bins = max(64, min(int(bins), 4096))
    if samples.size == 0:
        peaks = []
    else:
        edges = np.linspace(0, samples.size, num=bins + 1, dtype=np.int64)
        peaks = []
        for index in range(bins):
            start = edges[index]
            end = edges[index + 1]
            if end <= start:
                peaks.append(0.0)
                continue
            peaks.append(float(np.max(samples[start:end])))

    duration = float(frame_count) / float(sample_rate)
    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "peaks": peaks,
        "audio_path": audio_path,
    }


def _parse_keyframe_list(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        raw_values = value
    else:
        text = str(value).strip()
        if not text:
            return []
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            raw_values = parsed.get("keyframes", [])
        else:
            raw_values = parsed

    keyframes = []
    for item in raw_values:
        keyframes.append(max(0.0, float(item)))
    return keyframes


def _normalize_keyframe_list(keyframes, total_duration=None):
    normalized = []
    seen = set()
    for item in keyframes or []:
        seconds = max(0.0, float(item))
        if total_duration is not None:
            upper_bound = max(0.0, float(total_duration) - 0.001)
            seconds = min(seconds, upper_bound)
        bucket = int(round(seconds * 1000.0))
        if bucket in seen:
            continue
        seen.add(bucket)
        normalized.append(seconds)

    normalized.sort()
    return normalized[:MAX_MARKERS]


def _normalize_audio_tensor(audio):
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    return waveform, sample_rate


def _slice_audio(audio, start_frame, end_frame):
    waveform, sample_rate = _normalize_audio_tensor(audio)
    start_frame = max(0, int(start_frame))
    end_frame = max(start_frame + 1, int(end_frame))
    return {
        "waveform": waveform[..., start_frame:end_frame],
        "sample_rate": sample_rate,
    }


def _build_segments(total_duration, keyframes, skip_initial_segment, include_tail_segment):
    total_duration = max(0.0, float(total_duration))
    markers = _normalize_keyframe_list(keyframes, total_duration)

    if not markers:
        return [(0.0, total_duration)], total_duration

    segments = []
    if skip_initial_segment:
        # 第一个标记只作为音频起点，不作为第1段的结束点。
        # include_tail_segment=False: 段 = marker1->marker2, marker2->marker3, ...
        # include_tail_segment=True:  在上面基础上，再加 last_marker->total_duration。
        starts = markers[:-1]
        ends = markers[1:]
        if include_tail_segment and markers:
            starts = starts + [markers[-1]]
            ends = ends + [total_duration]
    else:
        # include_tail_segment=False: 段 = 0->marker1, marker1->marker2, ...
        # include_tail_segment=True:  在上面基础上，再加 last_marker->total_duration。
        starts = [0.0] + markers[:-1]
        ends = markers[:]
        if include_tail_segment and markers:
            starts = starts + [markers[-1]]
            ends = ends + [total_duration]

    for start_seconds, end_seconds in zip(starts, ends):
        if end_seconds <= start_seconds:
            continue
        segments.append((float(start_seconds), float(end_seconds)))

    if not segments:
        segments = [(0.0, total_duration)]

    selected_total_duration = float(sum(max(0.0, end - start) for start, end in segments))
    return segments, selected_total_duration


def _safe_int(value, default=0):
    try:
        if value is None:
            return int(default)
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return int(default)
        return int(value)
    except Exception:
        return int(default)


class NanFengAudioWaveformEditor:
    @classmethod
    def INPUT_TYPES(cls):
        audio_files = _list_input_audio_files() or [""]
        return {
            "required": {
                "audio": ("AUDIO",),
                "audio_file": (audio_files, {"default": audio_files[0]}),
                "keyframes_json": ("STRING", {"default": "[]", "multiline": False}),
                "skip_initial_segment": ("BOOLEAN", {"default": True}),
                "include_tail_segment": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "render_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = (
        "AUDIO",
        "INT",
        "INT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "INT",
        "INT",
        "INT",
        "INT",
        "STRING",
        "FLOAT",
    )
    RETURN_NAMES = (
        "音频",
        "当前分段",
        "总分段数",
        "分段时长_秒",
        "分段开始_秒",
        "分段结束_秒",
        "总选中时长_秒",
        "分段开始_毫秒",
        "分段结束_毫秒",
        "分段时长_毫秒",
        "总选中时长_毫秒",
        "调试文本",
        "总开始时长_秒",
    )
    FUNCTION = "select_segment"
    CATEGORY = "南风阳平/音频"

    @classmethod
    def VALIDATE_INPUTS(cls, audio_file="", keyframes_json="[]", **_kwargs):
        try:
            if audio_file:
                _resolve_audio_path(audio_file)
            _parse_keyframe_list(keyframes_json)
        except Exception as exc:
            return str(exc)
        return True

    def select_segment(
        self,
        audio,
        audio_file="",
        keyframes_json="[]",
        skip_initial_segment=True,
        include_tail_segment=False,
        segment_index=0,
        render_id="",
    ):
        waveform, sample_rate = _normalize_audio_tensor(audio)
        total_duration = waveform.shape[-1] / sample_rate if sample_rate else 0.0
        keyframes = _parse_keyframe_list(keyframes_json)
        segments, selected_total_duration = _build_segments(
            total_duration,
            keyframes,
            bool(skip_initial_segment),
            bool(include_tail_segment),
        )

        total_segments = max(1, len(segments))
        active_segment_index = _safe_int(segment_index, 0)
        current_segment = min(max(active_segment_index, 0), total_segments - 1)
        start_seconds, end_seconds = segments[current_segment]
        start_frame = int(math.floor(start_seconds * sample_rate))
        end_frame = int(math.ceil(end_seconds * sample_rate))
        selected_audio = _slice_audio(audio, start_frame, end_frame)

        segment_duration_seconds = max(0.0, end_seconds - start_seconds)
        segment_start_ms = int(round(start_seconds * 1000.0))
        segment_end_ms = int(round(end_seconds * 1000.0))
        segment_duration_ms = max(0, segment_end_ms - segment_start_ms)
        selected_total_duration_ms = int(round(selected_total_duration * 1000.0))

        total_start_seconds = float(segments[0][0]) if segments else 0.0

        # 秒输出保留两位小数；精确提取请使用毫秒输出。
        segment_duration_seconds_rounded = round(float(segment_duration_seconds), 2)
        start_seconds_rounded = round(float(start_seconds), 2)
        end_seconds_rounded = round(float(end_seconds), 2)
        selected_total_duration_rounded = round(float(selected_total_duration), 2)
        total_start_seconds_rounded = round(float(total_start_seconds), 2)

        debug_text = (
            f"segment={current_segment + 1}/{total_segments} | "
            f"skip_initial_segment={bool(skip_initial_segment)} | "
            f"include_tail_segment={bool(include_tail_segment)} | "
            f"segment={start_seconds:.3f}s->{end_seconds:.3f}s | "
            f"segment_ms={segment_start_ms}->{segment_end_ms} | "
            f"selected_total_duration={selected_total_duration:.3f}s | "
            f"total_start_seconds={total_start_seconds:.3f}s | "
            f"keyframes={_normalize_keyframe_list(keyframes, total_duration)} | "
            f"audio_file={audio_file or 'input audio'} | render_id={render_id or 'fresh'}"
        )
        logging.info("[南风音频] %s", debug_text)
        return (
            selected_audio,
            current_segment,
            total_segments,
            float(segment_duration_seconds_rounded),
            float(start_seconds_rounded),
            float(end_seconds_rounded),
            float(selected_total_duration_rounded),
            int(segment_start_ms),
            int(segment_end_ms),
            int(segment_duration_ms),
            int(selected_total_duration_ms),
            debug_text,
            float(total_start_seconds_rounded),
        )


async def nf_audio_file(request):
    audio_file = request.query.get("audio_file", "")
    try:
        audio_path = _resolve_audio_path(audio_file)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)

    content_type = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
    return web.FileResponse(audio_path, headers={"Content-Type": content_type})


async def nf_audio_waveform(request):
    audio_file = request.query.get("audio_file", "")
    bins = request.query.get("bins", "1400")
    try:
        payload = _read_waveform_peaks(audio_file, bins=int(bins))
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)

    payload["audio_url"] = f"/nfypnode/audio-file?audio_file={audio_file}"
    return web.json_response(payload)


_prompt_server_instance = getattr(server.PromptServer, "instance", None)
if _prompt_server_instance is not None:
    _prompt_server_instance.routes.get("/nfypnode/audio-file")(nf_audio_file)
    _prompt_server_instance.routes.get("/nfypnode/audio-waveform")(nf_audio_waveform)


NODE_CLASS_MAPPINGS = {
    "NanFengAudioWaveformEditor": NanFengAudioWaveformEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanFengAudioWaveformEditor": "南风音频",
}
