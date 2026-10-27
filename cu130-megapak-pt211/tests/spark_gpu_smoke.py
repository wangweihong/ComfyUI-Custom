"""Verify core kitchen and real Spark CUDA calls coexist on an SM120 GPU."""

import argparse
from dataclasses import replace
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as functional
import comfy_kitchen
from comfy_kitchen.backends import cuda as core_cuda
from comfy_kitchen.int8_decode import Int8DecodeCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-dir", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.node_dir))
    assert torch.cuda.get_device_capability() == (12, 0), "This Spark wheel requires SM120"
    version = importlib.metadata.version("comfy-kitchen")
    assert version == "0.2.39", version
    core_extension = core_cuda._C
    registry_before = comfy_kitchen.list_backends()
    import comfyui_spark_cuda as private
    assert private._C is not core_extension
    assert callable(private.spark_attn) and hasattr(private._C, "spark_attn")
    assert comfy_kitchen.list_backends() == registry_before, "Private import changed the core registry"

    spec = importlib.util.spec_from_file_location("spark_smoke_backend", args.node_dir / "comfyui_backend.py")
    backend = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = backend
    spec.loader.exec_module(backend)
    torch.manual_seed(123)
    # The default 16-way reblock planner needs enough video blocks to exercise
    # its normal root layout; add one exact conditioning block after the video.
    video_tokens, tokens = 2048, 2112
    q, k, v = [torch.randn(1, tokens, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    indices = torch.arange(tokens, device="cuda")
    layout = backend.ComfyPackedLayout(
        permutation=indices, inverse_permutation=indices, grid=(1, 32, 64),
        video_tokens=video_tokens, sequence_length=tokens, video_positions=indices[:video_tokens],
    )
    controller = backend.ComfySparkController(backend.ComfySparkConfig(
        topk_mode="topk_ratio", topk_ratio=1.0, force_local_blocks=False,
    ))
    # Skip Python reblocking to test the shipped native kernel independently of
    # model weights; the call still goes through the real Spark node backend.
    controller.topk_only = True
    dense = backend.comfy_kitchen_spark_attention(controller, q, k, v, layout, 0)
    reference = functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    torch.testing.assert_close(dense, reference, atol=0.02, rtol=0.02)
    controller.config = replace(controller.config, topk_ratio=0.5)
    sparse = backend.comfy_kitchen_spark_attention(controller, q, k, v, layout, 0)
    torch.cuda.synchronize()
    assert sparse.shape == q.shape and torch.isfinite(sparse).all()
    assert not torch.equal(sparse, dense), "Sparse settings did not change the output"
    # Exercise the real node's reblocking and reweighted sparse mode as well.
    controller.topk_only = False
    controller.config = replace(controller.config, topk_ratio=1.0)
    full_dense = backend.comfy_kitchen_spark_attention(controller, q, k, v, layout, 1)
    torch.testing.assert_close(full_dense, reference, atol=0.02, rtol=0.02)
    controller.config = replace(controller.config, topk_ratio=0.5)
    full_sparse = backend.comfy_kitchen_spark_attention(controller, q, k, v, layout, 2)
    torch.cuda.synchronize()
    assert full_sparse.shape == q.shape and torch.isfinite(full_sparse).all()
    assert controller.counts["reblock_plan_calls"] > 0
    assert controller.counts["comfy_kitchen_calls"] == 4
    assert controller.backend == "comfy-kitchen-spark-private"
    assert core_cuda._C is core_extension
    assert comfy_kitchen.list_backends() == registry_before
    assert Int8DecodeCache is not None
    print(json.dumps({
        "core_kitchen": version,
        "core_package": comfy_kitchen.__file__,
        "private_extension": private._C.__file__,
        "core_registry_unchanged": True,
        "spark_backend": controller.backend,
        "native_calls": controller.counts["comfy_kitchen_calls"],
        "reblock_plan_calls": controller.counts["reblock_plan_calls"],
        "dense_max_error": (dense - reference).abs().max().item(),
        "sparse_mean_difference": (sparse - dense).abs().float().mean().item(),
        "full_dense_max_error": (full_dense - reference).abs().max().item(),
    }, indent=2))


if __name__ == "__main__":
    main()
