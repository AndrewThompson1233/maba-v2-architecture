"""Multi-Device Hardware Backend Dispatcher and Graceful Fallback Registry.

Provides hardware detection, capability routing, environment overrides,
and fail-safe PyTorch fallback for Maba Sparse Attention (MABA-SA) kernel operations.
"""

import logging
import os
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
import warnings

import torch

from maba_sparse.kernels.common import (
    reference_compute_centroids,
    reference_dgda_prefill,
    reference_dgda_step,
    reference_index_topk,
    reference_stream_superposition,
)
from maba_sparse.kernels.cpu_dgda import (
    cpu_dgda_prefill,
    cpu_dgda_step,
)

logger = logging.getLogger(__name__)

# Registry mapping: backend -> operation_name -> callable
_KERNEL_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "reference": {
        "dgda_prefill": reference_dgda_prefill,
        "dgda_step": reference_dgda_step,
        "compute_centroids": reference_compute_centroids,
        "index_topk": reference_index_topk,
        "stream_superposition": reference_stream_superposition,
    },
    "cpu": {
        "dgda_prefill": cpu_dgda_prefill,
        "dgda_step": cpu_dgda_step,
        "compute_centroids": reference_compute_centroids,
        "index_topk": reference_index_topk,
        "stream_superposition": reference_stream_superposition,
    },
    "triton": {},
    "xla": {},
}

_WARNED_FALLBACKS: Set[str] = set()


def clear_fallback_warnings() -> None:
    """Clear deduplicated warning cache (useful for testing)."""
    _WARNED_FALLBACKS.clear()


def register_kernel(backend: str, op_name: str) -> Callable:
    """Decorator to register a hardware backend implementation for an operation."""
    def decorator(fn: Callable) -> Callable:
        if backend not in _KERNEL_REGISTRY:
            _KERNEL_REGISTRY[backend] = {}
        _KERNEL_REGISTRY[backend][op_name] = fn
        return fn
    return decorator


def get_kernel(backend: str, op_name: str) -> Optional[Callable]:
    """Retrieve registered kernel implementation for a backend and operation."""
    return _KERNEL_REGISTRY.get(backend, {}).get(op_name, None)


def is_cuda_sm75_available(device: Optional[Union[torch.device, str, int]] = None) -> bool:
    """Check if CUDA is available and target device has compute capability >= (7, 5)."""
    if not torch.cuda.is_available():
        return False
    try:
        dev_idx = 0
        if device is not None:
            if isinstance(device, int):
                dev_idx = device
            elif isinstance(device, str):
                d = torch.device(device)
                dev_idx = d.index if d.index is not None else 0
            elif isinstance(device, torch.device):
                dev_idx = device.index if device.index is not None else 0
        cap = torch.cuda.get_device_capability(dev_idx)
        return cap >= (7, 5)
    except Exception:
        return False


def is_triton_available() -> bool:
    """Check if Triton kernel compiler is installed and importable."""
    try:
        import triton  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError, Exception):
        return False


def is_xla_available(device: Optional[Union[torch.device, str]] = None) -> bool:
    """Check if Google TPU / XLA environment is active or device is XLA."""
    if device is not None:
        dev = torch.device(device) if isinstance(device, str) else device
        if dev.type == "xla":
            return True
    try:
        import torch_xla  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError, Exception):
        return False


def is_openmp_available() -> bool:
    """Check if OpenMP multiprocessing is available in PyTorch CPU runtime."""
    try:
        return torch.backends.openmp.is_available()
    except Exception:
        return False


def get_backend(device: Union[torch.device, str, None] = None) -> str:
    """Inspect runtime device target and return the active backend string.

    Possible return values: 'triton', 'cpu', 'xla', 'reference'.
    Can be overridden via environment variable MABA_BACKEND.
    """
    override = os.environ.get("MABA_BACKEND")
    if override and override.lower() != "auto":
        val = override.lower()
        if val in ("triton", "cuda", "gpu"):
            return "triton"
        if val in ("xla", "tpu"):
            return "xla"
        if val in ("cpu", "openmp"):
            return "cpu"
        if val in ("reference", "pytorch", "ref"):
            return "reference"
        raise ValueError(
            f"Unsupported MABA_BACKEND '{override}'. Supported: ['auto', 'triton', 'cpu', 'xla', 'reference', 'ref']"
        )

    if device is None:
        if torch.cuda.is_available() and is_cuda_sm75_available(0) and is_triton_available():
            return "triton"
        return "cpu"

    if isinstance(device, str):
        try:
            dev = torch.device(device)
        except Exception:
            return "reference"
    elif isinstance(device, torch.device):
        dev = device
    else:
        return "reference"

    if dev.type == "cuda":
        if torch.cuda.is_available() and is_cuda_sm75_available(dev) and is_triton_available():
            return "triton"
        return "reference"
    elif dev.type == "xla":
        if is_xla_available(dev):
            return "xla"
        return "reference"
    elif dev.type == "cpu":
        return "cpu"

    return "reference"


def _log_fallback_warning(op_name: str, backend: str, exc: Optional[Exception] = None) -> None:
    """Emit a warning when a specialized backend fails over to reference."""
    strict = os.environ.get("MABA_STRICT_BACKEND", "0").lower() in ("1", "true")
    err_detail = f" ({type(exc).__name__}: {exc})" if exc is not None else ""
    msg = (
        f"[Maba Dispatcher Fallback] Native kernel for '{op_name}' on backend '{backend}' "
        f"is unavailable or failed{err_detail}. Falling back to reference."
    )
    if strict:
        raise RuntimeError(msg)

    key = f"{backend}:{op_name}"
    # In test mode or when new error occurs, emit RuntimeWarning
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    if key not in _WARNED_FALLBACKS:
        _WARNED_FALLBACKS.add(key)
        logger.debug(msg)


def dispatch_dgda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    chunk_size: int = 16,
    initial_state: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified hardware dispatch for DGDA chunkwise recurrence prefill."""
    # Enforce device and dtype consistency
    for name, t in [("k", k), ("v", v), ("alpha", alpha), ("b", b), ("w", w)]:
        if t.device != q.device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got {q.device} and {t.device} for {name}"
            )
        if t.dtype != q.dtype:
            raise TypeError(
                f"Tensor dtypes must match, but got {q.dtype} for q and {t.dtype} for {name}"
            )
    if initial_state is not None:
        if initial_state.device != q.device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got {q.device} and "
                f"{initial_state.device} for initial_state"
            )
        if initial_state.dtype != q.dtype:
            raise TypeError(
                f"Tensor dtypes must match, but got {q.dtype} for q and {initial_state.dtype} for initial_state"
            )

    target_backend = get_backend(q.device)
    kernel_fn = get_kernel(target_backend, "dgda_prefill")

    if kernel_fn is not None and target_backend != "reference":
        try:
            return kernel_fn(
                q=q, k=k, v=v, alpha=alpha, b=b, w=w,
                chunk_size=chunk_size, initial_state=initial_state, **kwargs
            )
        except Exception as exc:
            _log_fallback_warning("dgda_prefill", target_backend, exc)
    elif target_backend != "reference" and kernel_fn is None:
        _log_fallback_warning("dgda_prefill", target_backend, None)

    ref_fn = get_kernel("reference", "dgda_prefill") or reference_dgda_prefill
    return ref_fn(
        q=q, k=k, v=v, alpha=alpha, b=b, w=w,
        chunk_size=chunk_size, initial_state=initial_state, **kwargs
    )


def dispatch_dgda_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified hardware dispatch for DGDA single-step O(1) decode recurrence."""
    for name, t in [("k", k), ("v", v), ("alpha", alpha), ("b", b), ("w", w)]:
        if t.device != q.device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got {q.device} and {t.device} for {name}"
            )
        if t.dtype != q.dtype:
            raise TypeError(
                f"Tensor dtypes must match, but got {q.dtype} for q and {t.dtype} for {name}"
            )
    if state is not None:
        if state.device != q.device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got {q.device} and {state.device} for state"
            )
        if state.dtype != q.dtype:
            raise TypeError(
                f"Tensor dtypes must match, but got {q.dtype} for q and {state.dtype} for state"
            )

    target_backend = get_backend(q.device)
    kernel_fn = get_kernel(target_backend, "dgda_step")

    if kernel_fn is not None and target_backend != "reference":
        try:
            return kernel_fn(
                q=q, k=k, v=v, alpha=alpha, b=b, w=w,
                state=state, **kwargs
            )
        except Exception as exc:
            _log_fallback_warning("dgda_step", target_backend, exc)
    elif target_backend != "reference" and kernel_fn is None:
        _log_fallback_warning("dgda_step", target_backend, None)

    ref_fn = get_kernel("reference", "dgda_step") or reference_dgda_step
    return ref_fn(
        q=q, k=k, v=v, alpha=alpha, b=b, w=w,
        state=state, **kwargs
    )


def dispatch_compute_centroids(
    k_idx: torch.Tensor,
    block_size: int = 64,
    **kwargs: Any,
) -> torch.Tensor:
    """Unified hardware dispatch for accelerated hybrid centroid pooling."""
    target_backend = get_backend(k_idx.device)
    kernel_fn = get_kernel(target_backend, "compute_centroids")

    if kernel_fn is not None and target_backend != "reference":
        try:
            return kernel_fn(k_idx=k_idx, block_size=block_size, **kwargs)
        except Exception as exc:
            _log_fallback_warning("compute_centroids", target_backend, exc)
    elif target_backend != "reference" and kernel_fn is None:
        _log_fallback_warning("compute_centroids", target_backend, None)

    ref_fn = get_kernel("reference", "compute_centroids") or reference_compute_centroids
    return ref_fn(k_idx=k_idx, block_size=block_size, **kwargs)


def dispatch_index_topk(
    q_idx: torch.Tensor,
    centroids: torch.Tensor,
    lambda_dist: float = 0.5,
    top_k: int = 32,
    block_size: int = 64,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Unified hardware dispatch for top-k block gather with distance penalty."""
    if q_idx.device != centroids.device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but got {q_idx.device} and {centroids.device}"
        )

    target_backend = get_backend(q_idx.device)
    kernel_fn = get_kernel(target_backend, "index_topk")

    if kernel_fn is not None and target_backend != "reference":
        try:
            return kernel_fn(
                q_idx=q_idx, centroids=centroids, lambda_dist=lambda_dist,
                top_k=top_k, block_size=block_size, **kwargs
            )
        except Exception as exc:
            _log_fallback_warning("index_topk", target_backend, exc)
    elif target_backend != "reference" and kernel_fn is None:
        _log_fallback_warning("index_topk", target_backend, None)

    ref_fn = get_kernel("reference", "index_topk") or reference_index_topk
    return ref_fn(
        q_idx=q_idx, centroids=centroids, lambda_dist=lambda_dist,
        top_k=top_k, block_size=block_size, **kwargs
    )


def dispatch_stream_superposition(
    o_local: torch.Tensor,
    o_sparse: torch.Tensor,
    o_hca: torch.Tensor,
    gate_logits: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Unified hardware dispatch for 3-stream output superposition."""
    device = o_local.device
    if o_sparse.device != device or o_hca.device != device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but got {device}, {o_sparse.device}, {o_hca.device}"
        )
    gate_weights = kwargs.get("gate_weights")
    gate_t = gate_logits if gate_logits is not None else gate_weights
    if gate_t is not None and gate_t.device != device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but got {device} and {gate_t.device}"
        )

    target_backend = get_backend(device)
    kernel_fn = get_kernel(target_backend, "stream_superposition")

    if kernel_fn is not None and target_backend != "reference":
        try:
            return kernel_fn(
                o_local=o_local, o_sparse=o_sparse, o_hca=o_hca,
                gate_logits=gate_logits, **kwargs
            )
        except Exception as exc:
            _log_fallback_warning("stream_superposition", target_backend, exc)
    elif target_backend != "reference" and kernel_fn is None:
        _log_fallback_warning("stream_superposition", target_backend, None)

    ref_fn = get_kernel("reference", "stream_superposition") or reference_stream_superposition
    return ref_fn(
        o_local=o_local, o_sparse=o_sparse, o_hca=o_hca,
        gate_logits=gate_logits, **kwargs
    )


__all__ = [
    "get_backend",
    "register_kernel",
    "get_kernel",
    "clear_fallback_warnings",
    "is_cuda_sm75_available",
    "is_triton_available",
    "is_xla_available",
    "is_openmp_available",
    "dispatch_dgda_prefill",
    "dispatch_dgda_step",
    "dispatch_compute_centroids",
    "dispatch_index_topk",
    "dispatch_stream_superposition",
]
