import json
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import folder_paths
import torch

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None


CATEGORY_NAME = "南风节点/遮罩"

_INSIGHTFACE_CACHE: Dict[Tuple[int, int], Any] = {}
_OPENCV_CASCADE = None
_NVIDIA_DLLS_ADDED = False


def _ensure_runtime_dependencies():
    missing: List[str] = []
    if cv2 is None:
        missing.append("opencv-python / cv2")
    if np is None:
        missing.append("numpy")
    if missing:
        raise RuntimeError(f"缺少依赖: {', '.join(missing)}")


def _add_nvidia_dll_to_path():
    global _NVIDIA_DLLS_ADDED
    if _NVIDIA_DLLS_ADDED:
        return

    try:
        import glob
        import site

        extra_paths: List[str] = []
        for site_package in site.getsitepackages():
            pattern = os.path.join(site_package, "nvidia", "**", "bin")
            for path in glob.glob(pattern, recursive=True):
                if os.path.isdir(path):
                    extra_paths.append(path)

            torch_lib = os.path.join(site_package, "torch", "lib")
            if os.path.isdir(torch_lib):
                extra_paths.append(torch_lib)

        cuda_root = os.environ.get("CUDA_PATH")
        if cuda_root:
            cuda_bin = os.path.join(cuda_root, "bin")
            if os.path.isdir(cuda_bin):
                extra_paths.append(cuda_bin)

        if extra_paths:
            deduped_paths: List[str] = []
            seen = set()
            for path in extra_paths:
                normalized = os.path.normcase(os.path.normpath(path))
                if normalized in seen:
                    continue
                seen.add(normalized)
                deduped_paths.append(path)

            os.environ["PATH"] = os.pathsep.join(deduped_paths) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    _NVIDIA_DLLS_ADDED = True


def _pick_providers(gpu_id: int) -> List[str]:
    providers = ["CPUExecutionProvider"]
    if gpu_id is None or gpu_id < 0:
        return providers

    if ort is None:
        print("[NanFengPersonMask] onnxruntime import failed, falling back to CPUExecutionProvider.")
        return providers

    try:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            ort_version = getattr(ort, "__version__", "unknown")
            try:
                ort_device = ort.get_device()
            except Exception:
                ort_device = "unknown"

            print(
                "[NanFengPersonMask] GPU was requested "
                f"(gpu_id={gpu_id}), but onnxruntime only exposes {available}. "
                f"Falling back to CPUExecutionProvider. ort={ort_version}, device={ort_device}. "
                "This usually means the CPU-only onnxruntime package is active, "
                "or the CUDA/cuDNN DLLs required by onnxruntime-gpu could not be loaded."
            )
    except Exception:
        pass
    return providers


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _try_get_5pts(face) -> Optional["np.ndarray"]:
    if np is None:
        return None
    if hasattr(face, "kps") and face.kps is not None:
        kps = face.kps
        if isinstance(kps, np.ndarray) and kps.shape == (5, 2):
            return kps.astype(np.float32)
    return None


def _build_head_mask(
    height: int,
    width: int,
    bbox,
    kps5=None,
    up_pad: float = 0.55,
    down_pad: float = 0.0,
    side_pad: float = 0.28,
    extra_side_boost: float = 0.35,
):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    box_width = max(2, x2 - x1)
    box_height = max(2, y2 - y1)

    x1e = int(x1 - side_pad * box_width)
    x2e = int(x2 + side_pad * box_width)
    y1e = int(y1 - up_pad * box_height)
    y2e = int(y2 + down_pad * box_height)

    if kps5 is not None:
        left_eye, right_eye, nose = kps5[0], kps5[1], kps5[2]
        eye_center_x = float((left_eye[0] + right_eye[0]) * 0.5)
        delta_x = float(nose[0] - eye_center_x)
        boost = int(extra_side_boost * side_pad * box_width)
        if delta_x > 0.03 * box_width:
            x2e += boost
        elif delta_x < -0.03 * box_width:
            x1e -= boost

    x1e = _clamp(x1e, 0, width - 1)
    x2e = _clamp(x2e, 0, width - 1)
    y1e = _clamp(y1e, 0, height - 1)
    y2e = _clamp(y2e, 0, height - 1)

    mask = np.zeros((height, width), dtype=np.uint8)
    center_x = (x1e + x2e) // 2
    center_y = (y1e + y2e) // 2
    axis_x = max(1, (x2e - x1e) // 2)
    axis_y = max(1, (y2e - y1e) // 2)
    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)
    return mask


def _insightface_root() -> str:
    return os.path.join(folder_paths.models_dir, "insightface")


def _get_insightface_app(gpu_id: int, det_size: int):
    if FaceAnalysis is None:
        raise RuntimeError("当前环境没有安装 insightface")

    cache_key = (int(gpu_id), int(det_size))
    if cache_key in _INSIGHTFACE_CACHE:
        return _INSIGHTFACE_CACHE[cache_key]

    _add_nvidia_dll_to_path()

    root = _insightface_root()
    model_dir = os.path.join(root, "models", "buffalo_l")
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"未找到 buffalo_l 模型目录: {model_dir}")

    os.environ["INSIGHTFACE_HOME"] = root
    providers = _pick_providers(gpu_id)

    try:
        app = FaceAnalysis(name="buffalo_l", root=root, providers=providers)
    except TypeError:
        app = FaceAnalysis(name="buffalo_l", providers=providers)

    ctx_id = int(gpu_id) if gpu_id is not None else -1
    app.prepare(ctx_id=ctx_id, det_size=(int(det_size), int(det_size)))
    _INSIGHTFACE_CACHE[cache_key] = app
    return app


def _get_opencv_cascade():
    global _OPENCV_CASCADE
    if _OPENCV_CASCADE is not None:
        return _OPENCV_CASCADE

    if cv2 is None:
        raise RuntimeError("当前环境没有安装 cv2")

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"OpenCV 人脸级联加载失败: {cascade_path}")

    _OPENCV_CASCADE = cascade
    return _OPENCV_CASCADE


def _detect_faces_with_insightface(
    image_bgr,
    min_face: int,
    largest_only: bool,
    gpu_id: int,
    det_size: int,
):
    app = _get_insightface_app(gpu_id=gpu_id, det_size=det_size)
    faces = app.get(image_bgr)

    valid_faces: List[Dict[str, Any]] = []
    for face in faces:
        bbox = face.bbox.astype(np.float32)
        box_width = float(bbox[2] - bbox[0])
        box_height = float(bbox[3] - bbox[1])
        if box_width >= float(min_face) and box_height >= float(min_face):
            valid_faces.append(
                {
                    "bbox": bbox,
                    "kps5": _try_get_5pts(face),
                }
            )

    valid_faces.sort(
        key=lambda item: float((item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])),
        reverse=True,
    )
    if largest_only and valid_faces:
        valid_faces = valid_faces[:1]
    return valid_faces


def _detect_faces_with_opencv(image_bgr, min_face: int, largest_only: bool):
    cascade = _get_opencv_cascade()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(8, int(min_face)), max(8, int(min_face))),
    )

    valid_faces: List[Dict[str, Any]] = []
    for x, y, width, height in faces:
        valid_faces.append(
            {
                "bbox": np.array([x, y, x + width, y + height], dtype=np.float32),
                "kps5": None,
            }
        )

    valid_faces.sort(
        key=lambda item: float((item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])),
        reverse=True,
    )
    if largest_only and valid_faces:
        valid_faces = valid_faces[:1]
    return valid_faces


def _detect_faces(
    image_bgr,
    detector_backend: str,
    min_face: int,
    largest_only: bool,
    gpu_id: int,
    det_size: int,
):
    errors: List[str] = []

    if detector_backend in ("auto", "insightface"):
        try:
            return _detect_faces_with_insightface(
                image_bgr=image_bgr,
                min_face=min_face,
                largest_only=largest_only,
                gpu_id=gpu_id,
                det_size=det_size,
            ), "insightface", ""
        except Exception as exc:
            errors.append(f"insightface: {exc}")
            if detector_backend == "insightface":
                raise

    if detector_backend in ("auto", "opencv"):
        try:
            fallback_reason = "; ".join(errors)
            return _detect_faces_with_opencv(
                image_bgr=image_bgr,
                min_face=min_face,
                largest_only=largest_only,
            ), "opencv", fallback_reason
        except Exception as exc:
            errors.append(f"opencv: {exc}")

    raise RuntimeError("人脸检测器初始化失败: " + " | ".join(errors))


def _tensor_frame_to_bgr(frame: torch.Tensor):
    frame_rgb = frame.detach().cpu().clamp(0.0, 1.0).numpy()
    frame_rgb = (frame_rgb * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_tensor(image_bgr) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image_rgb.astype(np.float32) / 255.0)


class NanFengPersonMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detector_backend": (["auto", "insightface", "opencv"], {"default": "auto"}),
                "gpu_id": ("INT", {"default": 0, "min": -1, "max": 16, "step": 1}),
                "det_size": ("INT", {"default": 640, "min": 128, "max": 2048, "step": 32}),
                "min_face": ("INT", {"default": 96, "min": 1, "max": 2048, "step": 1}),
                "largest_only": ("BOOLEAN", {"default": False}),
                "up_pad": ("FLOAT", {"default": 0.55, "min": -1.0, "max": 3.0, "step": 0.01}),
                "down_pad": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 3.0, "step": 0.01}),
                "side_pad": ("FLOAT", {"default": 0.28, "min": -1.0, "max": 3.0, "step": 0.01}),
                "extra_side_boost": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 3.0, "step": 0.01}),
                "dilate": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "mask_faces"
    CATEGORY = CATEGORY_NAME

    def mask_faces(
        self,
        image,
        detector_backend,
        gpu_id,
        det_size,
        min_face,
        largest_only,
        up_pad,
        down_pad,
        side_pad,
        extra_side_boost,
        dilate,
    ):
        debug: Dict[str, Any] = {
            "node": "南风人物遮罩",
            "ok": False,
        }

        try:
            _ensure_runtime_dependencies()

            if not isinstance(image, torch.Tensor):
                raise RuntimeError("image 输入不是有效的 IMAGE Tensor")
            if image.ndim != 4 or image.shape[-1] != 3:
                raise RuntimeError(f"image 输入形状不对，期望 [T,H,W,3]，实际为 {list(image.shape)}")

            processed_frames: List[torch.Tensor] = []
            total_faces = 0
            backend_used: Optional[str] = None
            backend_note = ""

            for frame_index in range(int(image.shape[0])):
                frame_tensor = image[frame_index]
                frame_bgr = _tensor_frame_to_bgr(frame_tensor)
                frame_height, frame_width = frame_bgr.shape[:2]
                active_backend = backend_used if detector_backend == "auto" and backend_used else detector_backend

                faces, current_backend, current_note = _detect_faces(
                    image_bgr=frame_bgr,
                    detector_backend=active_backend,
                    min_face=int(min_face),
                    largest_only=bool(largest_only),
                    gpu_id=int(gpu_id),
                    det_size=int(det_size),
                )
                if backend_used is None:
                    backend_used = current_backend
                    backend_note = current_note

                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                for face in faces:
                    face_mask = _build_head_mask(
                        height=frame_height,
                        width=frame_width,
                        bbox=face["bbox"],
                        kps5=face["kps5"],
                        up_pad=float(up_pad),
                        down_pad=float(down_pad),
                        side_pad=float(side_pad),
                        extra_side_boost=float(extra_side_boost),
                    )
                    mask = cv2.bitwise_or(mask, face_mask)

                if int(dilate) > 0:
                    kernel_size = max(1, int(dilate) | 1)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    mask = cv2.dilate(mask, kernel, iterations=1)

                masked_frame = frame_bgr.copy()
                masked_frame[mask > 0] = 0
                processed_frames.append(_bgr_to_tensor(masked_frame))
                total_faces += len(faces)

            output = torch.stack(processed_frames, dim=0).to(device=image.device, dtype=image.dtype)

            debug.update(
                {
                    "ok": True,
                    "backend": backend_used,
                    "backend_note": backend_note,
                    "frame_count": int(image.shape[0]),
                    "image_shape": [int(v) for v in image.shape],
                    "faces_detected_total": int(total_faces),
                    "min_face": int(min_face),
                    "largest_only": bool(largest_only),
                }
            )
            return (output,)
        except Exception as exc:
            debug.update(
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            raise RuntimeError(json.dumps(debug, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {
    "NanFengPersonMask": NanFengPersonMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanFengPersonMask": "南风人物遮罩",
}
