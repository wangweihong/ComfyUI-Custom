import json
import logging
import math
import mimetypes
import os
import re
import hashlib
import urllib.parse
import wave
from pathlib import Path

try:
    import av
except Exception:
    av = None
import folder_paths
import numpy as np
import torch
from aiohttp import web

try:
    import server
except Exception:
    server = None

MAX_MARKERS = 64
MODULE_DIR = Path(__file__).resolve().parent
AUDIO_PREVIEW_CACHE_DIR = MODULE_DIR / "_nf_audio_waveform_cache"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
MANUAL_AUDIO_PREFIX = "__nf_manual_audio__:"


def _strip_path(path):
    path = (path or "").strip()
    if path.startswith('"'):
        path = path[1:]
    if path.endswith('"'):
        path = path[:-1]
    return path


def _unwrap_audio_file(audio_file):
    audio_file = _strip_path(audio_file)
    if audio_file.startswith(MANUAL_AUDIO_PREFIX):
        return _strip_path(audio_file[len(MANUAL_AUDIO_PREFIX):]), True
    return audio_file, False


def _list_input_audio_files():
    input_dir = folder_paths.get_input_directory()
    if not input_dir or not os.path.isdir(input_dir):
        return []

    discovered = []
    for root, _dirs, files in os.walk(input_dir):
        for filename in files:
            extension = os.path.splitext(filename)[1].lower()
            if extension not in AUDIO_EXTENSIONS:
                continue
            full_path = os.path.join(root, filename)
            relative_path = os.path.relpath(full_path, input_dir).replace("\\", "/")
            discovered.append(relative_path)

    return sorted(discovered)


def _resolve_audio_path(audio_file):
    audio_file, _is_manual_override = _unwrap_audio_file(audio_file)
    if not audio_file:
        raise ValueError("audio_file is empty")

    expanded_path = os.path.expandvars(os.path.expanduser(audio_file))

    if os.path.isabs(expanded_path) and os.path.isfile(expanded_path):
        return expanded_path

    direct_candidate = os.path.abspath(expanded_path)
    if os.path.isfile(direct_candidate):
        return direct_candidate

    try:
        annotated = folder_paths.get_annotated_filepath(expanded_path)
        if annotated and os.path.isfile(annotated):
            return annotated
    except Exception:
        pass

    input_dir = folder_paths.get_input_directory()
    if input_dir:
        input_candidate = os.path.abspath(os.path.join(input_dir, expanded_path))
        if os.path.isfile(input_candidate):
            return input_candidate

    raise ValueError(f"Audio file not found: {audio_file}")


def _ensure_audio_preview_cache_dir():
    AUDIO_PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(AUDIO_PREVIEW_CACHE_DIR)


def _audio_frame_to_f32(frame_array, channel_count):
    samples = np.asarray(frame_array)
    if samples.ndim == 0:
        samples = samples.reshape(1, 1)
    elif samples.ndim == 1:
        if channel_count > 1:
            samples = samples.reshape(-1, channel_count).T
        else:
            samples = samples.reshape(1, -1)
    elif samples.shape[0] != channel_count:
        samples = samples.reshape(-1, channel_count).T

    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float32, copy=False)
    if samples.dtype == np.uint8:
        return (samples.astype(np.float32) - 128.0) / 128.0
    if samples.dtype == np.int16:
        return samples.astype(np.float32) / 32768.0
    if samples.dtype == np.int32:
        return samples.astype(np.float32) / float(1 << 31)

    info = np.iinfo(samples.dtype) if np.issubdtype(samples.dtype, np.integer) else None
    if info is not None:
        scale = float(max(abs(info.min), abs(info.max)))
        if scale > 0:
            return samples.astype(np.float32) / scale
    return samples.astype(np.float32)


def _load_wav_samples(audio_path):
    with wave.open(audio_path, "rb") as wav_file:
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frame_bytes = wav_file.readframes(frame_count)

    if frame_count <= 0 or sample_rate <= 0:
        return np.zeros((max(channel_count, 1), 0), dtype=np.float32), sample_rate, audio_path

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
        samples = samples.reshape(-1, channel_count).T
    else:
        samples = samples.reshape(1, -1)

    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    samples = np.clip(samples, -1.0, 1.0)
    return samples, sample_rate, audio_path


def _load_audio_samples(audio_file):
    audio_path = _resolve_audio_path(audio_file)
    if av is None:
        extension = os.path.splitext(audio_path)[1].lower()
        if extension == ".wav":
            return _load_wav_samples(audio_path)
        raise ValueError("PyAV is required to read non-WAV audio files")

    with av.open(audio_path) as audio_container:
        if not audio_container.streams.audio:
            raise ValueError("No audio stream found in the file")

        stream = audio_container.streams.audio[0]
        channel_count = int(getattr(stream, "channels", 0) or 0)
        sample_rate = int(getattr(stream.codec_context, "sample_rate", 0) or 0)

        frames = []
        for frame in audio_container.decode(streams=stream.index):
            if channel_count <= 0:
                channel_count = int(getattr(frame.layout, "nb_channels", 0) or 0) or 1
            if sample_rate <= 0:
                sample_rate = int(getattr(frame, "sample_rate", 0) or 0)

            chunk = _audio_frame_to_f32(frame.to_ndarray(), channel_count)
            if chunk.size > 0:
                frames.append(chunk)

    if sample_rate <= 0:
        raise ValueError(f"Unable to determine sample rate for: {audio_file}")

    if not frames:
        waveform = np.zeros((max(channel_count, 1), 0), dtype=np.float32)
    else:
        waveform = np.concatenate(frames, axis=1)

    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    waveform = np.clip(waveform, -1.0, 1.0)
    return waveform, sample_rate, audio_path


def _read_waveform_peaks(audio_file, bins=1400):
    waveform, sample_rate, audio_path = _load_audio_samples(audio_file)
    frame_count = int(waveform.shape[-1]) if waveform.ndim >= 2 else 0
    audio_identity = _preview_audio_hash(waveform, sample_rate) if sample_rate > 0 else ""

    if frame_count <= 0 or sample_rate <= 0:
        return {
            "duration": 0.0,
            "sample_rate": sample_rate,
            "peaks": [],
            "audio_path": audio_path,
            "audio_identity": audio_identity,
        }

    if waveform.ndim == 1:
        samples = np.abs(waveform)
    elif waveform.shape[0] > 1:
        samples = np.mean(np.abs(waveform), axis=0)
    else:
        samples = np.abs(waveform[0])

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
        "audio_identity": audio_identity,
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
    waveform, sample_rate = _get_audio_waveform_and_rate(audio)
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    return waveform, sample_rate


def _get_audio_waveform_and_rate(audio):
    if audio is None:
        raise ValueError("audio is None")

    waveform = None
    sample_rate = None

    if isinstance(audio, dict):
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
    else:
        waveform = getattr(audio, "waveform", None)
        sample_rate = getattr(audio, "sample_rate", None)

    if waveform is None or sample_rate is None:
        raise ValueError("Invalid audio input")

    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)

    if waveform.ndim == 3:
        if waveform.shape[0] == 1:
            waveform = waveform[0]
        else:
            waveform = waveform.reshape(-1, waveform.shape[-1])
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"Unsupported audio waveform shape: {tuple(waveform.shape)}")

    if waveform.shape[0] > waveform.shape[1] and waveform.shape[1] <= 8:
        waveform = waveform.transpose(0, 1)

    waveform = waveform.detach().to(dtype=torch.float32, device="cpu").contiguous()
    waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
    return waveform, int(sample_rate)


def _sanitize_cache_token(value):
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("_")
    return token or "nf_audio"


def _preview_audio_hash(waveform, sample_rate):
    hasher = hashlib.sha1()
    waveform_shape = tuple(getattr(waveform, "shape", ()) or ())
    hasher.update(str(waveform_shape).encode("utf-8"))
    hasher.update(str(sample_rate).encode("utf-8"))

    if isinstance(waveform, torch.Tensor):
        flat = waveform.reshape(-1)
        flat_size = int(flat.numel())
        if flat_size > 0:
            sample_count = min(flat_size, 32768)
            step = max(int(flat_size // sample_count), 1)
            sample = flat[::step][:sample_count].to(dtype=torch.float32).contiguous()
            hasher.update(sample.numpy().tobytes())
    else:
        flat = np.asarray(waveform, dtype=np.float32).reshape(-1)
        flat_size = int(flat.size)
        if flat_size > 0:
            sample_count = min(flat_size, 32768)
            step = max(int(flat_size // sample_count), 1)
            sample = np.ascontiguousarray(flat[::step][:sample_count], dtype=np.float32)
            hasher.update(sample.tobytes())

    return hasher.hexdigest()[:20]


def _is_preview_cache_wav_name(filename, unique_id=""):
    prefix = _sanitize_cache_token(unique_id)
    filename = os.path.basename(str(filename or "")).strip()
    if not filename.startswith(f"{prefix}_"):
        return False
    if os.path.splitext(filename)[1].lower() != ".wav":
        return False
    if filename.endswith("_edited.wav"):
        return False
    name_without_ext = filename[:-4]
    parts = name_without_ext.split("_", 1)
    if len(parts) != 2:
        return False
    hash_part = parts[1]
    return bool(re.fullmatch(r"[0-9a-f]{20}", hash_part))


def _find_cached_audio_file(unique_id=""):
    cache_dir = _ensure_audio_preview_cache_dir()
    prefix = _sanitize_cache_token(unique_id)
    prefix_start = f"{prefix}_"

    candidates = []
    try:
        for existing_name in os.listdir(cache_dir):
            if not existing_name.startswith(prefix_start):
                continue
            if not _is_preview_cache_wav_name(existing_name, unique_id=unique_id):
                continue
            existing_path = os.path.join(cache_dir, existing_name)
            if os.path.isfile(existing_path):
                try:
                    mtime = os.path.getmtime(existing_path)
                except Exception:
                    mtime = 0.0
                candidates.append((mtime, existing_path))
    except Exception:
        return ""

    if not candidates:
        return ""

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _cleanup_prefix_audio_files(unique_id="", keep_paths=None):
    cache_dir = _ensure_audio_preview_cache_dir()
    keep_paths = {os.path.abspath(path) for path in (keep_paths or []) if path}

    try:
        for existing_name in os.listdir(cache_dir):
            if not _is_preview_cache_wav_name(existing_name, unique_id=unique_id):
                continue
            existing_path = os.path.join(cache_dir, existing_name)
            if os.path.abspath(existing_path) in keep_paths:
                continue
            try:
                os.remove(existing_path)
            except Exception:
                pass
    except Exception:
        pass


def _write_preview_audio_file(audio, unique_id=""):
    waveform, sample_rate = _get_audio_waveform_and_rate(audio)
    if waveform.numel() <= 0 or sample_rate <= 0:
        return ""

    cache_dir = _ensure_audio_preview_cache_dir()
    prefix = _sanitize_cache_token(unique_id)
    audio_hash = _preview_audio_hash(waveform, sample_rate)
    output_path = os.path.join(cache_dir, f"{prefix}_{audio_hash}.wav")

    if os.path.isfile(output_path):
        return output_path

    _cleanup_prefix_audio_files(unique_id=unique_id, keep_paths=[output_path])

    pcm = (waveform.transpose(0, 1).numpy() * 32767.0).round().astype(np.int16)
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(int(waveform.shape[0]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())

    return output_path


def _edited_audio_output_path(unique_id=""):
    cache_dir = _ensure_audio_preview_cache_dir()
    prefix = _sanitize_cache_token(unique_id)
    return os.path.join(cache_dir, f"{prefix}_edited.wav")


def _save_uploaded_audio_bytes(file_bytes, unique_id=""):
    file_bytes = file_bytes or b""
    if not file_bytes:
        raise ValueError("Empty edited audio payload")

    output_path = _edited_audio_output_path(unique_id=unique_id)
    temp_path = f"{output_path}.tmp"
    with open(temp_path, "wb") as temp_file:
        temp_file.write(file_bytes)
    os.replace(temp_path, output_path)
    _cleanup_prefix_audio_files(unique_id=unique_id, keep_paths=[output_path])
    return output_path


def _load_audio_tensor_from_path(audio_path):
    waveform_np, sample_rate, resolved_path = _load_audio_samples(audio_path)
    waveform = torch.from_numpy(waveform_np).to(dtype=torch.float32, device="cpu").contiguous().unsqueeze(0)
    normalized_audio = {
        "waveform": waveform,
        "sample_rate": int(sample_rate),
    }
    return normalized_audio, waveform, int(sample_rate), resolved_path


def _resolve_audio_source(audio=None, audio_file="", edited_audio_file="", unique_id=""):
    normalized_edited_audio_file, is_manual_edited_override = _unwrap_audio_file(edited_audio_file)
    normalized_audio_file, is_manual_override = _unwrap_audio_file(audio_file)

    if normalized_edited_audio_file:
        try:
            edited_audio_candidate = edited_audio_file if is_manual_edited_override else normalized_edited_audio_file
            normalized_audio, waveform, sample_rate, audio_path = _load_audio_tensor_from_path(edited_audio_candidate)
            return normalized_audio, waveform, int(sample_rate), audio_path, "edited_audio_override"
        except Exception as exc:
            logging.warning(
                "[NanFengAudio] edited override audio missing or unreadable, fallback to other sources: %s | %s",
                normalized_edited_audio_file,
                exc,
            )

    if is_manual_override and normalized_audio_file:
        try:
            normalized_audio, waveform, sample_rate, audio_path = _load_audio_tensor_from_path(normalized_audio_file)
            return normalized_audio, waveform, int(sample_rate), audio_path, "audio_file"
        except Exception as exc:
            logging.warning(
                "[NanFengAudio] manual override audio missing or unreadable, fallback to other sources: %s | %s",
                normalized_audio_file,
                exc,
            )

    if audio is not None:
        waveform, sample_rate = _normalize_audio_tensor(audio)
        normalized_audio = {
            "waveform": waveform,
            "sample_rate": int(sample_rate),
        }
        return normalized_audio, waveform, int(sample_rate), "input audio", "input_audio"

    cached_audio_file = _find_cached_audio_file(unique_id=unique_id)
    if cached_audio_file:
        normalized_audio, waveform, sample_rate, audio_path = _load_audio_tensor_from_path(cached_audio_file)
        return normalized_audio, waveform, int(sample_rate), audio_path, "node_cached_audio"

    if normalized_audio_file:
        normalized_audio, waveform, sample_rate, audio_path = _load_audio_tensor_from_path(normalized_audio_file)
        return normalized_audio, waveform, int(sample_rate), audio_path, "audio_file"

    raise ValueError("Connect audio input or choose an audio_file first (no cached node audio found)")


def _slice_audio(audio, start_frame, end_frame):
    waveform, sample_rate = _normalize_audio_tensor(audio)
    start_frame = max(0, int(start_frame))
    end_frame = max(start_frame + 1, int(end_frame))
    return {
        "waveform": waveform[..., start_frame:end_frame],
        "sample_rate": sample_rate,
    }


def _pad_audio_with_silence(audio, prepend_seconds=0.0, append_seconds=0.0):
    waveform, sample_rate = _normalize_audio_tensor(audio)
    prepend_seconds = max(0.0, float(prepend_seconds))
    append_seconds = max(0.0, float(append_seconds))

    prepend_frames = max(0, int(round(prepend_seconds * sample_rate)))
    append_frames = max(0, int(round(append_seconds * sample_rate)))
    if prepend_frames <= 0 and append_frames <= 0:
        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }

    chunks = []
    if prepend_frames > 0:
        chunks.append(torch.zeros((*waveform.shape[:-1], prepend_frames), dtype=waveform.dtype, device=waveform.device))
    chunks.append(waveform)
    if append_frames > 0:
        chunks.append(torch.zeros((*waveform.shape[:-1], append_frames), dtype=waveform.dtype, device=waveform.device))

    padded_waveform = torch.cat(chunks, dim=-1).contiguous()
    return {
        "waveform": padded_waveform,
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


def _safe_non_negative_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return float(default)
        return max(0.0, float(value))
    except Exception:
        return float(default)


class NanFengAudioWaveformEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "vhs_path_extensions": [extension.lstrip(".") for extension in AUDIO_EXTENSIONS],
                    },
                ),
                "keyframes_json": ("STRING", {"default": "[]", "multiline": False}),
                "skip_initial_segment": ("BOOLEAN", {"default": True}),
                "include_tail_segment": ("BOOLEAN", {"default": False}),
                "前补静音秒数": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1}),
                "后补静音秒数": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "render_id": ("STRING", {"default": ""}),
                "edited_audio_file": ("STRING", {"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
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
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, audio_file="", keyframes_json="[]", **_kwargs):
        try:
            _parse_keyframe_list(keyframes_json)
        except Exception as exc:
            return str(exc)
        return True

    def select_segment(
        self,
        audio=None,
        audio_file="",
        edited_audio_file="",
        keyframes_json="[]",
        skip_initial_segment=True,
        include_tail_segment=False,
        前补静音秒数=0.0,
        后补静音秒数=0.0,
        segment_index=0,
        render_id="",
        unique_id=None,
    ):
        source_audio, waveform, sample_rate, source_reference, preview_source = _resolve_audio_source(
            audio=audio,
            audio_file=audio_file,
            edited_audio_file=edited_audio_file,
            unique_id=unique_id,
        )
        prepend_silence_seconds = _safe_non_negative_float(前补静音秒数, 0.0)
        append_silence_seconds = _safe_non_negative_float(后补静音秒数, 0.0)
        waveform, sample_rate = _normalize_audio_tensor(source_audio)
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
        source_start_seconds, source_end_seconds = segments[current_segment]
        start_frame = int(math.floor(source_start_seconds * sample_rate))
        end_frame = int(math.ceil(source_end_seconds * sample_rate))
        selected_audio = _slice_audio(source_audio, start_frame, end_frame)
        selected_audio = _pad_audio_with_silence(
            selected_audio,
            prepend_seconds=prepend_silence_seconds,
            append_seconds=append_silence_seconds,
        )
        selected_waveform, selected_sample_rate = _normalize_audio_tensor(selected_audio)

        start_seconds = float(source_start_seconds) - prepend_silence_seconds
        end_seconds = float(source_end_seconds) + append_silence_seconds
        segment_duration_seconds = (
            selected_waveform.shape[-1] / selected_sample_rate
            if selected_sample_rate
            else max(0.0, end_seconds - start_seconds)
        )
        segment_start_ms = int(round(start_seconds * 1000.0))
        segment_end_ms = int(round(end_seconds * 1000.0))
        segment_duration_ms = int(round(segment_duration_seconds * 1000.0))
        selected_total_duration = float(selected_total_duration) + (
            total_segments * (prepend_silence_seconds + append_silence_seconds)
        )
        selected_total_duration_ms = int(round(selected_total_duration * 1000.0))

        total_start_seconds = (float(segments[0][0]) - prepend_silence_seconds) if segments else -prepend_silence_seconds

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
            f"source_segment={source_start_seconds:.3f}s->{source_end_seconds:.3f}s | "
            f"output_segment={start_seconds:.3f}s->{end_seconds:.3f}s | "
            f"segment_ms={segment_start_ms}->{segment_end_ms} | "
            f"selected_total_duration={selected_total_duration:.3f}s | "
            f"total_start_seconds={total_start_seconds:.3f}s | "
            f"keyframes={_normalize_keyframe_list(keyframes, total_duration)} | "
            f"prepend_silence={prepend_silence_seconds:.3f}s | "
            f"append_silence={append_silence_seconds:.3f}s | "
            f"source={preview_source} | audio_ref={source_reference} | "
            f"audio_file_widget={audio_file or '<empty>'} | render_id={render_id or 'fresh'}"
        )
        logging.info("[南风音频] %s", debug_text)
        result = (
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

        preview_audio_file = ""
        try:
            preview_audio_file = _write_preview_audio_file(source_audio, unique_id=unique_id)
        except Exception as exc:
            logging.warning("[NanFengAudio] failed to write preview audio: %s", exc)

        if preview_audio_file:
            return {
                "ui": {
                    "nf_audio_preview": [
                        {
                            "audio_file": preview_audio_file,
                            "sample_rate": int(sample_rate),
                            "duration": float(total_duration),
                            "source": preview_source,
                        }
                    ]
                },
                "result": result,
            }

        return result


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

    payload["audio_url"] = f"/nfypnode/audio-file?audio_file={urllib.parse.quote(audio_file, safe='')}"
    return web.json_response(payload)


async def nf_audio_save_edits(request):
    try:
        post_data = await request.post()
        uploaded_audio = post_data.get("audio")
        unique_id = str(post_data.get("unique_id", "") or "")
        if uploaded_audio is None:
            raise ValueError("Missing edited audio file")

        file_bytes = uploaded_audio.file.read() if getattr(uploaded_audio, "file", None) is not None else bytes(uploaded_audio)
        saved_path = _save_uploaded_audio_bytes(file_bytes, unique_id=unique_id)
        return web.json_response(
            {
                "ok": True,
                "audio_file": saved_path,
                "audio_url": f"/nfypnode/audio-file?audio_file={urllib.parse.quote(saved_path, safe='')}",
            }
        )
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


_prompt_server_instance = getattr(server.PromptServer, "instance", None) if server is not None else None
if _prompt_server_instance is not None:
    _prompt_server_instance.routes.get("/nfypnode/audio-file")(nf_audio_file)
    _prompt_server_instance.routes.get("/nfypnode/audio-waveform")(nf_audio_waveform)
    _prompt_server_instance.routes.post("/nfypnode/audio-save-edits")(nf_audio_save_edits)


NODE_CLASS_MAPPINGS = {
    "NanFengAudioWaveformEditor": NanFengAudioWaveformEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanFengAudioWaveformEditor": "南风音频",
}
