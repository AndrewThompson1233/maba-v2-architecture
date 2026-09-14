"""Maba Sparse Hardware Acceleration Kernels Package.

Exports the unified hardware dispatcher and all public operation entry points.
"""

from maba_sparse.kernels.common import (
    normalize_keys,
    ref_compute_centroids,
    ref_dgda_prefill,
    ref_dgda_sequential,
    ref_dgda_step,
    ref_index_topk,
    ref_stream_superposition,
    reference_compute_centroids,
    reference_dgda_prefill,
    reference_dgda_sequential,
    reference_dgda_step,
    reference_index_topk,
    reference_stream_superposition,
    validate_dgda_prefill_inputs,
    validate_dgda_step_inputs,
    validate_indexer_inputs,
    validate_superposition_inputs,
)
from maba_sparse.kernels.cpu_dgda import (
    cpu_dgda_prefill,
    cpu_dgda_step,
    cpu_threads,
    get_cpu_num_threads,
    set_cpu_num_threads,
)
from maba_sparse.kernels.dispatcher import (
    clear_fallback_warnings,
    dispatch_compute_centroids,
    dispatch_dgda_prefill,
    dispatch_dgda_step,
    dispatch_index_topk,
    dispatch_stream_superposition,
    get_backend,
    get_kernel,
    is_cuda_sm75_available,
    is_openmp_available,
    is_triton_available,
    is_xla_available,
    register_kernel,
)

__all__ = [
    # Hardware detection & registry
    "get_backend",
    "register_kernel",
    "get_kernel",
    "clear_fallback_warnings",
    "is_cuda_sm75_available",
    "is_triton_available",
    "is_xla_available",
    "is_openmp_available",
    # Dispatch operations
    "dispatch_dgda_prefill",
    "dispatch_dgda_step",
    "dispatch_compute_centroids",
    "dispatch_index_topk",
    "dispatch_stream_superposition",
    # Reference implementations
    "normalize_keys",
    "ref_dgda_prefill",
    "ref_dgda_step",
    "ref_dgda_sequential",
    "ref_compute_centroids",
    "ref_index_topk",
    "ref_stream_superposition",
    "reference_dgda_prefill",
    "reference_dgda_step",
    "reference_dgda_sequential",
    "reference_compute_centroids",
    "reference_index_topk",
    "reference_stream_superposition",
    "validate_dgda_prefill_inputs",
    "validate_dgda_step_inputs",
    "validate_indexer_inputs",
    "validate_superposition_inputs",
    # CPU implementations & utilities
    "cpu_dgda_prefill",
    "cpu_dgda_step",
    "get_cpu_num_threads",
    "set_cpu_num_threads",
    "cpu_threads",
]
