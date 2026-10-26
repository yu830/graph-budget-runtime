"""Identity from the actual imported source, installed dependencies and local model.

File hashing happens before loading weights, never in the dispatch hot path.
GPU fields are queried by the real worker. No user-written native=true flags.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

BASE_SHA = "bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2"
DEPENDENCIES = ("torch", "triton", "sglang", "sgl-kernel", "flashinfer-python",
                "flashinfer-cubin", "transformers", "compressed-tensors")
CONFIG_KEYS = ("mem_fraction_static", "max_total_tokens", "max_running_requests",
               "context_length", "page_size", "disable_radix_cache",
               "disable_overlap_schedule", "disable_cuda_graph_padding",
               "chunked_prefill_size", "sampling_backend", "random_seed",
               "kv_cache_dtype", "enable_deterministic_inference", "schedule_policy",
               "max_prefill_tokens", "stream_interval", "stream_output")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    # Small unbuffered reads avoid a reproduced DrvFS ENOMEM at 4 MiB request
    # size under host pressure. The digest still covers every original byte.
    with open(path, "rb", buffering=0) as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_digest(path):
    """Normalize only CRLF for portable Git checkouts; all source bytes remain bound."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_identity(sglang_file):
    root = Path(sglang_file).resolve().parent
    manifest = json.loads((Path(__file__).parent / "patches/manifest.json").read_text())
    actual = {name: code_digest(root / name) for name in manifest["patched_files"]}
    if actual != manifest["patched_files"]:
        raise ValueError("Loaded SGLang files differ from the pinned patch; reinstall the exact checkout")
    # Bind other runtime files as well, not merely the three modified call sites.
    tree = {str(p.relative_to(root)).replace("\\", "/"): code_digest(p)
            for p in sorted((root / "srt").rglob("*.py"))}
    if digest(tree) != manifest["srt_tree_sha256"]:
        raise ValueError("Loaded SGLang runtime tree differs from the pinned patched source")
    return {"base_commit": manifest["base_commit"], "patch_sha256": manifest["patch_sha256"],
            "srt_tree_sha256": digest(tree), "patched_files": actual}


def model_identity(model_path, revision):
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise ValueError("Use a local, immutable HF snapshot; download explicitly before launching")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("An exact 40-character model/tokenizer revision is required")
    if root.parent.name == "snapshots" and root.name != revision:
        raise ValueError("Local snapshot directory disagrees with --revision")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix in
             (".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken")]
    if not any(p.suffix in (".safetensors", ".bin") for p in files):
        raise ValueError("No model weight files in local snapshot")
    if not (root / "config.json").is_file() or not any("tokenizer" in p.name for p in files):
        raise ValueError("Model config/tokenizer files are missing")
    entries = {str(p.relative_to(root)).replace("\\", "/"): file_digest(p) for p in sorted(files)}
    return {"revision": revision, "tokenizer_revision": revision,
            "files_sha256": entries, "manifest_sha256": digest(entries)}


def collect(server_args, gpu_id=0):
    import sglang
    import torch
    props = torch.cuda.get_device_properties(gpu_id)
    gpu_row = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=driver_version,uuid", "--format=csv,noheader"], text=True
    ).strip().splitlines()[0].split(",")
    driver, gpu_uuid = (field.strip() for field in gpu_row)
    deps = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    tokenizer = server_args.tokenizer_path or server_args.model_path
    if Path(tokenizer).resolve() != Path(server_args.model_path).resolve():
        raise ValueError("v1 requires model and tokenizer from the same local snapshot")
    module_files = {p.name: code_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {
        "sglang": source_identity(sglang.__file__),
        "graphbudget_sha256": digest(module_files),
        "model": model_identity(server_args.model_path, server_args.revision),
        "gpu": {"name": props.name, "compute_capability": [props.major, props.minor],
                "total_bytes": props.total_memory, "driver": driver,
                "uuid_sha256": hashlib.sha256(gpu_uuid.encode()).hexdigest()},
        "dependencies": deps, "python": sys.version.split()[0], "cuda": torch.version.cuda,
        "dtype": server_args.dtype, "attention_backend": server_args.attention_backend,
        "decode_attention_backend": server_args.decode_attention_backend,
        "prefill_attention_backend": server_args.prefill_attention_backend,
        "tp_size": server_args.tp_size,
        "config": {key: getattr(server_args, key) for key in CONFIG_KEYS},
        "runtime_env": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_MODULE_LOADING",
            "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE",
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE",
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM",
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT")},
    }


def local_details():
    import sglang
    import torch
    return {"python_executable": sys.executable, "sglang_module": sglang.__file__,
            "torch_module": torch.__file__, "graphbudget_module": __file__,
            "pid": os.getpid()}
