
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pytest  # noqa: E402
import torch  # noqa: E402
from maba_sparse.config import MabaSparseConfig  # noqa: E402
from maba_sparse.layers.dgda import DGDALayer  # noqa: E402


@pytest.fixture
def config():
    return MabaSparseConfig(
        dim=640,
        n_heads=10,
        d_head=64,
        kernel_size=4,
        chunk_size=16,
        eps=1e-6,
    )


@pytest.fixture
def small_config():
    return MabaSparseConfig(
        dim=64,
        n_heads=2,
        d_head=32,
        kernel_size=4,
        chunk_size=16,
        eps=1e-6,
    )


@pytest.fixture
def dgda_layer(small_config):
    torch.manual_seed(42)
    layer = DGDALayer(small_config)
    layer.eval()
    return layer
