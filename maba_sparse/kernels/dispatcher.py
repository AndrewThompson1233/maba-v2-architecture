
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
    ref_compute_centroids,
    ref_dgda_prefill,
    ref_dgda_step,
    ref_index_topk,
    ref_stream_superposition,
    ref_centroids,
    ref_topk,
    ref_superposition,
)
from maba_sparse.kernels.cpu_dgda import (
    cpu_dgda_prefill,
    cpu_dgda_step,
)

logger = logging.getLogger(__name__)

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

_BACKEND_CACHE: Dict[Tuple[Optional[str], Optional[int], Optional[str]], str] = {}
_SM75_CACHE: Dict[Optional[int], bool] = {}
_TRITON_CACHE: Optional[bool] = None
_OPENMP_CACHE: Optional[bool] = None


def clear_dispatcher_cache() -> None:
    _BACKEND_CACHE.clear()
    _SM75_CACHE.clear()
    global _TRITON_CACHE, _OPENMP_CACHE
    _TRITON_CACHE = None
    _OPENMP_CACHE = None


def clear_fallback_warnings() -> None:
    _WARNED_FALLBACKS.clear()
    clear_dispatcher_cache()


def register_kernel(backend: str, op_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        if backend not in _KERNEL_REGISTRY:
            _KERNEL_REGISTRY[backend] = {}
        _KERNEL_REGISTRY[backend][op_name] = fn
        return fn
    return decorator


def get_kernel(backend: str, op_name: str) -> Optional[Callable]:
    if backend == "triton" and op_name not in _KERNEL_REGISTRY.get("triton", {}):
        if op_name in ("dgda_prefill", "dgda_step"):
            try:
                import maba_sparse.kernels.triton_dgda  # noqa: F401
            except Exception:
                pass
        elif op_name in ("compute_centroids", "index_topk", "stream_superposition"):
            try:
                import maba_sparse.kernels.triton_indexer  # noqa: F401
            except Exception:
                pass
    elif backend == "xla" and op_name not in _KERNEL_REGISTRY.get("xla", {}):
        if op_name in ("dgda_prefill", "dgda_step"):
            try:
                import maba_sparse.kernels.xla_dgda  # noqa: F401
            except Exception:
                pass
        elif op_name in ("compute_centroids", "index_topk", "stream_superposition"):
            try:
                import maba_sparse.kernels.xla_indexer  # noqa: F401
            except Exception:
                pass
    elif backend == "cpu":
        if op_name in ("compute_centroids", "index_topk", "stream_superposition"):
            reg_fn = _KERNEL_REGISTRY.get("cpu", {}).get(op_name)
            if reg_fn is None or reg_fn.__name__.startswith("ref"):
                try:
                    import maba_sparse.kernels.cpu_indexer  # noqa: F401
                except Exception:
                    pass
    return _KERNEL_REGISTRY.get(backend, {}).get(op_name, None)


def is_cuda_sm75_available(device: Optional[Union[torch.device, str, int]] = None) -> bool:
    if not torch.cuda.is_available():
        return False

    dev_idx = 0
    if device is not None:
        if isinstance(device, int):
            dev_idx = device
        elif isinstance(device, str):
            try:
                d = torch.device(device)
                dev_idx = d.index if d.index is not None else 0
            except Exception:
                return False
        elif isinstance(device, torch.device):
            dev_idx = device.index if device.index is not None else 0

    if dev_idx in _SM75_CACHE:
        return _SM75_CACHE[dev_idx]

    try:
        cap = torch.cuda.get_device_capability(dev_idx)
        res = cap >= (7, 5)
        _SM75_CACHE[dev_idx] = res
        return res
    except Exception:
        return False


def is_triton_available() -> bool:
    global _TRITON_CACHE
    if _TRITON_CACHE is not None:
        return _TRITON_CACHE
    try:
        import triton  # noqa: F401
        _TRITON_CACHE = True
        return True
    except (ImportError, ModuleNotFoundError, Exception):
        _TRITON_CACHE = False
        return False


def is_xla_available(device: Optional[Union[torch.device, str]] = None) -> bool:
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
    global _OPENMP_CACHE
    if _OPENMP_CACHE is not None:
        return _OPENMP_CACHE
    try:
        res = torch.backends.openmp.is_available()
        _OPENMP_CACHE = res
        return res
    except Exception:
        _OPENMP_CACHE = False
        return False


def get_backend(device: Union[torch.device, str, None] = None) -> str:
    dev_type: Optional[str] = None
    dev_idx: Optional[int] = None
    if device is not None:
        if isinstance(device, str):
            try:
                d = torch.device(device)
                dev_type, dev_idx = d.type, (d.index if d.index is not None else 0)
            except Exception:
                return "reference"
        elif isinstance(device, torch.device):
            dev_type, dev_idx = device.type, (device.index if device.index is not None else 0)

    env_override = os.environ.get("MABA_BACKEND")
    cache_key = (dev_type, dev_idx, env_override)
    cached = _BACKEND_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if env_override and env_override.lower() != "auto":
        val = env_override.lower()
        if val in ("triton", "cuda", "gpu"):
            res = "triton"
        elif val in ("xla", "tpu"):
            res = "xla"
        elif val in ("cpu", "openmp"):
            res = "cpu"
        elif val in ("reference", "pytorch", "ref"):
            res = "reference"
        else:
            raise ValueError(
                f"Unsupported MABA_BACKEND '{env_override}'. Supported: ['auto', 'triton', 'cpu', 'xla', 'reference', 'ref']"
            )
        _BACKEND_CACHE[cache_key] = res
        return res

    if dev_type is None:
        if torch.cuda.is_available() and is_cuda_sm75_available(0) and is_triton_available():
            res = "triton"
        else:
            res = "cpu"
    elif dev_type == "cuda":
        if torch.cuda.is_available() and is_cuda_sm75_available(dev_idx) and is_triton_available():
            res = "triton"
        else:
            res = "reference"
    elif dev_type == "xla":
        if is_xla_available(device):
            res = "xla"
        else:
            res = "reference"
    elif dev_type == "cpu":
        res = "cpu"
    else:
        res = "reference"

    _BACKEND_CACHE[cache_key] = res
    return res


def _log_fallback_warning(op_name: str, backend: str, exc: Optional[Exception] = None) -> None:
    strict = os.environ.get("MABA_STRICT_BACKEND", "0").lower() in ("1", "true")
    err_detail = f" ({type(exc).__name__}: {exc})" if exc is not None else ""
    msg = (
        f"[Maba Dispatcher Fallback] Native kernel for '{op_name}' on backend '{backend}' "
        f"is unavailable or failed{err_detail}. Falling back to reference."
    )
    if strict:
        raise RuntimeError(msg)

    key = f"{backend}:{op_name}"
    if key not in _WARNED_FALLBACKS:
        _WARNED_FALLBACKS.add(key)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        logger.debug(msg)


def dispatch_dgda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    chunk_size: int = 16,
    initial_state: Optional[torch.Tensor] = None,
    log_alpha: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha is None and log_alpha is None:
        raise ValueError("Either alpha or log_alpha must be provided")
    if b is None or w is None:
        raise ValueError("Both b and w gate tensors must be provided")

    q_dev = q.device
    q_dt = q.dtype

    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
    if q_dt not in supported_dtypes:
        raise TypeError(f"Unsupported dtype {q_dt}. Supported: {supported_dtypes}")

    tensors = [("k", k), ("v", v), ("b", b), ("w", w)]
    if alpha is not None:
        tensors.append(("alpha", alpha))
    if log_alpha is not None:
        tensors.append(("log_alpha", log_alpha))

    for name, t in tensors:
        if t.device != q_dev:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got {q_dev} and {t.device} for {name}"
            )
        if t.dtype != q_dt:
            raise TypeError(
                f"Tensor dtypes must match, but got {q_dt} for q and {t.dtype} for {name}"
            )
    if initial_state is not None:
        if initial_state.device != q_dev:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got {q_dev} and {initial_state.device} for initial_state"
            )
        if initial_state.dtype != q_dt:
            raise TypeError(
                f"Tensor dtypes must match, but got {q_dt} for q and {initial_state.dtype} for initial_state"
            )

    target_backend = get_backend(q_dev)
    kernel_fn = get_kernel(target_backend, "dgda_prefill")

    if kernel_fn is not None and target_backend != "reference":
        try:
            return kernel_fn(
                q=q, k=k, v=v, alpha=alpha, b=b, w=w,
                chunk_size=chunk_size, initial_state=initial_state,
                log_alpha=log_alpha, **kwargs
            )
        except Exception as exc:
            _log_fallback_warning("dgda_prefill", target_backend, exc)
    elif target_backend != "reference" and kernel_fn is None:
        _log_fallback_warning("dgda_prefill", target_backend, None)

    ref_fn = get_kernel("reference", "dgda_prefill") or reference_dgda_prefill
    return ref_fn(
        q=q, k=k, v=v, alpha=alpha, b=b, w=w,
        chunk_size=chunk_size, initial_state=initial_state,
        log_alpha=log_alpha, **kwargs
    )


def dispatch_dgda_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    log_alpha: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha is None and log_alpha is None:
        raise ValueError("Either alpha or log_alpha must be provided")
    if b is None or w is None:
        raise ValueError("Both b and w gate tensors must be provided")

    q_dev = q.device
    q_dt = q.dtype

    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
    if q_dt not in supported_dtypes:
        raise TypeError(f"Unsupported dtype {q_dt}. Supported: {supported_dtypes}")

    tensors = [("k", k), ("v", v), ("b", b), ("w", w)]
    if alpha is not None:
        tensors.append(("alpha", alpha))
    if log_alpha is not None:
        tensors.append(("log_alpha", log_alpha))

    for name, t in tensors:
        if t.device != q_dev:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got {q_dev} and {t.device} for {name}"
            )
        if t.dtype != q_dt:
            raise TypeError(
                f"Tensor dtypes must match, but got {q_dt} for q and {t.dtype} for {name}"
            )
    if state is not None:
        if state.device != q_dev:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got {q_dev} and {state.device} for state"
            )
        if state.dtype != q_dt:
            raise TypeError(
                f"Tensor dtypes must match, but got {q_dt} for q and {state.dtype} for state"
            )

    target_backend = get_backend(q_dev)
    kernel_fn = get_kernel(target_backend, "dgda_step")

    if kernel_fn is not None and target_backend != "reference":
        try:
            return kernel_fn(
                q=q, k=k, v=v, alpha=alpha, b=b, w=w,
                state=state, log_alpha=log_alpha, **kwargs
            )
        except Exception as exc:
            _log_fallback_warning("dgda_step", target_backend, exc)
    elif target_backend != "reference" and kernel_fn is None:
        _log_fallback_warning("dgda_step", target_backend, None)

    ref_fn = get_kernel("reference", "dgda_step") or reference_dgda_step
    return ref_fn(
        q=q, k=k, v=v, alpha=alpha, b=b, w=w,
        state=state, log_alpha=log_alpha, **kwargs
    )


def dispatch_compute_centroids(
    k_idx: torch.Tensor,
    block_size: int = 64,
    **kwargs: Any,
) -> torch.Tensor:
    from maba_sparse.kernels.common import validate_indexer_inputs
    validate_indexer_inputs(k_idx=k_idx)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

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
    if q_idx.device != centroids.device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but got {q_idx.device} and {centroids.device}"
        )
    if q_idx.dim() != 3 or centroids.dim() != 3:
        raise ValueError(f"q_idx and centroids must be 3D, got {q_idx.dim()}D and {centroids.dim()}D")
    if q_idx.shape[0] != centroids.shape[0] or q_idx.shape[2] != centroids.shape[2]:
        raise ValueError(
            f"Dimension mismatch between q_idx {list(q_idx.shape)} and centroids {list(centroids.shape)}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

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

    from maba_sparse.kernels.common import validate_superposition_inputs
    validate_superposition_inputs(o_local, o_sparse, o_hca, gate_t)

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


dgda_prefill = dispatch_dgda_prefill
dgda_step = dispatch_dgda_step
compute_centroids = dispatch_compute_centroids
index_topk = dispatch_index_topk
stream_superposition = dispatch_stream_superposition


__all__ = [
    "get_backend",
    "register_kernel",
    "get_kernel",
    "clear_fallback_warnings",
    "clear_dispatcher_cache",
    "is_cuda_sm75_available",
    "is_triton_available",
    "is_xla_available",
    "is_openmp_available",
    "dispatch_dgda_prefill",
    "dispatch_dgda_step",
    "dispatch_compute_centroids",
    "dispatch_index_topk",
    "dispatch_stream_superposition",
    "dgda_prefill",
    "dgda_step",
    "compute_centroids",
    "index_topk",
    "stream_superposition",
    "ref_dgda_prefill",
    "ref_dgda_step",
    "ref_centroids",
    "ref_topk",
    "ref_superposition",
]
