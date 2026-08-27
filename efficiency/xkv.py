"""Adapters for xKV's fused selective-reconstruction CUDA operators"""

import os
import subprocess
from pathlib import Path

import torch

_LOADED = None
_OPS_DIR = Path(__file__).parent / "ops"

# Pinned upstream commit the prebuilt kernel in _OPS_DIR was compiled from.
# Used as a source-build fallback when that prebuilt .so doesn't load
_XKV_REPO_URL = "https://github.com/abdelfattah-lab/xKV.git"
_XKV_COMMIT = "05d91ecb0d698279aa4220fda5d7a5108036d692"
_BUILD_DIR = (
    Path(os.environ.get("XKV_BUILD_DIR", Path.home() / ".cache" / "xkv_kernels"))
    / _XKV_COMMIT
)


def _has_ops() -> bool:
    namespace = getattr(torch.ops, "_shadowkv", None)
    return namespace is not None and hasattr(namespace, "batch_gemm_softmax")


def _clone_xkv_source() -> Path | None:
    """Fetch the pinned xKV commit, including only the cutlass submodule"""
    src_dir = _BUILD_DIR / "src"
    csrc = src_dir / "efficiency" / "csrc"
    cutlass_include = src_dir / "3rdparty" / "cutlass" / "include"
    if csrc.is_dir() and cutlass_include.is_dir():
        return src_dir

    try:
        src_dir.parent.mkdir(parents=True, exist_ok=True)
        if not (src_dir / ".git").exists():
            subprocess.run(
                ["git", "clone", "--filter=blob:none", _XKV_REPO_URL, str(src_dir)],
                check=True, capture_output=True, timeout=600,
            )
        subprocess.run(
            ["git", "-C", str(src_dir), "checkout", _XKV_COMMIT],
            check=True, capture_output=True, timeout=60,
        )
        subprocess.run(
            [
                "git", "-C", str(src_dir), "submodule", "update", "--init",
                "--depth", "1", "--", "3rdparty/cutlass",
            ],
            check=True, capture_output=True, timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"Could not fetch kernel source ({exc}); "
              "falling back to the unfused path.")
        return None
    return src_dir


def _split_cuda_toolkit_include_dirs() -> list[str]:
    """Header dirs for CUDA libraries"""
    import importlib.util

    dirs = []
    for pkg in (
        "cusparse", "curand", "cublas", "cusolver", "cufft",
        "cuda_nvrtc", "nvjitlink", "cuda_runtime",
    ):
        spec = importlib.util.find_spec(f"nvidia.{pkg}")
        if spec and spec.submodule_search_locations:
            include = Path(spec.submodule_search_locations[0]) / "include"
            if include.is_dir():
                dirs.append(str(include))
    return dirs


def _build_from_source() -> bool:
    """Compile xKV's fused kernels for this Python/CUDA/torch build."""
    src_dir = _clone_xkv_source()
    if src_dir is None:
        return False

    csrc = src_dir / "efficiency" / "csrc"
    cutlass = src_dir / "3rdparty" / "cutlass"
    print(
        f"Prebuilt kernel unavailable, building from source "
        f"({_XKV_COMMIT[:12]}); this can take several minutes the first time..."
    )
    try:
        from torch.utils.cpp_extension import load

        load(
            name="_shadowkv",
            sources=[
                str(csrc / "main.cu"),
                str(csrc / "rope.cu"),
                str(csrc / "rope_new.cu"),
                str(csrc / "gather_copy.cu"),
                str(csrc / "batch_gather_gemm.cu"),
                str(csrc / "batch_gemm_softmax.cu"),
            ],
            extra_include_paths=[
                str(cutlass / "include"),
                str(cutlass / "examples" / "common"),
                str(cutlass / "tools" / "util" / "include"),
                str(csrc),
                *_split_cuda_toolkit_include_dirs(),
            ],
            extra_cflags=["-std=c++17", "-O3"],
            extra_cuda_cflags=[
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-O3",
                "-use_fast_math",
            ],
        )
    except Exception as exc:
        print(f"Building fused kernels failed ({exc}); "
              "falling back to the unfused path.")
        return False
    return _has_ops()


def _load_ops() -> bool:
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    if _has_ops():
        _LOADED = True
        return True

    for path in sorted(_OPS_DIR.glob("_shadowkv*.so")):
        try:
            torch.ops.load_library(str(path))
        except (OSError, RuntimeError):
            continue
        _LOADED = _has_ops()
        if _LOADED:
            return True

    if os.environ.get("XKV_NO_BUILD") != "1":
        _LOADED = _build_from_source()
        if _LOADED:
            return True

    _LOADED = False
    return False


def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def adjust_rank(
    num_rows: int,
    num_columns: int,
    target_ratio: float,
    overhead_bytes: int,
    batch_size: int,
    element_size: int,
    alignment: int = 8,
) -> int:
    """Choose an aligned factor rank after persistent prefill overhead."""
    original_bytes = batch_size * num_rows * num_columns * element_size
    bytes_per_rank = batch_size * (num_rows + num_columns) * element_size
    rank = (
        round(
            (original_bytes / target_ratio - overhead_bytes)
            / bytes_per_rank
            / alignment
        )
        * alignment
    )
    return max(alignment, min(rank, min(num_rows, num_columns)))


class FusedLandmarkScorer:
    """Reuse xKV's fused grouped-query landmark GEMM and softmax workspace."""

    def __init__(self):
        self.gemm = None
        self.softmax = None
        self.norm = None
        self.sum = None

    @staticmethod
    def supports(query: torch.Tensor, landmarks: torch.Tensor) -> bool:
        return (
            _load_ops()
            and query.is_cuda
            and query.dtype == torch.bfloat16
            and landmarks.dtype == torch.bfloat16
            and query.size(-1) == 128
            and landmarks.size(-2) % 8 == 0
        )

    def _allocate(self, query: torch.Tensor, num_landmarks: int) -> None:
        batch_size, num_heads, num_groups, _ = query.shape
        output_shape = (batch_size, num_heads, num_groups, num_landmarks)
        reduction_shape = (
            batch_size * num_heads,
            num_groups,
            (num_landmarks + 255) // 256,
        )
        if self.softmax is not None and self.softmax.shape == output_shape:
            return
        self.gemm = torch.empty(
            output_shape, device=query.device, dtype=query.dtype
        )
        self.softmax = torch.empty_like(self.gemm)
        self.norm = torch.empty(
            reduction_shape, device=query.device, dtype=torch.float32
        )
        self.sum = torch.empty_like(self.norm)

    def score(
        self,
        query: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> torch.Tensor:
        if not self.supports(query, landmarks):
            raise RuntimeError(
                "FusedLandmarkScorer.score: unsupported inputs for the "
                f"fused xKV landmark kernel (ops_loaded={_load_ops()}, "
                f"query.is_cuda={query.is_cuda}, query.dtype={query.dtype}, "
                f"landmarks.dtype={landmarks.dtype}, "
                f"head_dim={query.size(-1)} (must be 128), "
                f"num_landmarks={landmarks.size(-2)} (must be a multiple "
                "of 8)."
            )
        num_landmarks = landmarks.size(-2)
        self._allocate(query, num_landmarks)
        batch_size, num_heads, num_groups, head_dim = query.shape
        torch.ops._shadowkv.batch_gemm_softmax(
            query.contiguous(),
            landmarks.contiguous(),
            self.gemm,
            self.norm,
            self.sum,
            self.softmax,
            batch_size * num_heads,
            num_groups,
            num_landmarks,
            head_dim,
            head_dim**-0.5,
            0.0,
        )
        return self.softmax.amax(dim=2)

    @property
    def nbytes(self) -> int:
        return sum(
            _tensor_nbytes(tensor)
            for tensor in (self.gemm, self.softmax, self.norm, self.sum)
        )


class FusedKeyReconstructor:
    """Fuse selected-row reconstruction, RoPE, and reconstruction-cache reuse"""

    @staticmethod
    def available() -> bool:
        return _load_ops()

    def __init__(self):
        self.states = {}
        self.output = None

    @staticmethod
    def supports(
        shared: torch.Tensor,
        right: torch.Tensor,
        chunks: torch.Tensor,
        chunk_size: int,
    ) -> bool:
        return (
            _load_ops()
            and shared.is_cuda
            and shared.dtype == torch.bfloat16
            and right.dtype == torch.bfloat16
            and right.size(-2) == 128
            and shared.size(-1) % 8 == 0
            and chunk_size == 8
            and chunks.size(-1) in (128, 256, 512, 1024)
        )

    def _state(self, layer_idx: int, chunks: torch.Tensor, chunk_size: int):
        batch_size, num_heads, select_sets = chunks.shape
        shape = (batch_size, num_heads, select_sets)
        state = self.states.get(layer_idx)
        if state is not None and state["chunks"].shape == shape:
            return state
        device = chunks.device
        state = {
            "chunks": torch.full(shape, -1, device=device, dtype=torch.int64),
            "chunks_i32": torch.empty(shape, device=device, dtype=torch.int32),
            "offsets": torch.empty(
                batch_size * num_heads * select_sets,
                device=device,
                dtype=torch.int32,
            ),
            "counts": torch.empty(
                batch_size * num_heads,
                device=device,
                dtype=torch.int32,
            ),
            "keys": torch.empty(
                batch_size,
                num_heads,
                select_sets * chunk_size,
                128,
                device=device,
                dtype=torch.bfloat16,
            ),
        }
        self.states[layer_idx] = state
        return state

    def reorder(
        self,
        layer_idx: int,
        chunks: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        state = self._state(layer_idx, chunks, chunk_size)
        batch_size, num_heads, select_sets = chunks.shape
        torch.ops._shadowkv.reorder_keys_and_compute_offsets(
            state["chunks"],
            chunks.contiguous(),
            state["offsets"],
            state["counts"],
            batch_size,
            num_heads,
            select_sets,
        )
        return state["chunks"]

    def reconstruct(
        self,
        layer_idx: int,
        shared: torch.Tensor,
        right: torch.Tensor,
        packed_rope: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        state = self.states[layer_idx]
        chunks = state["chunks"]
        batch_size, num_heads, select_sets = chunks.shape
        sparse_budget = select_sets * chunk_size
        head_dim = right.size(-2)
        rank = shared.size(-1)
        if self.output is None or self.output.shape != state["keys"].shape:
            self.output = torch.empty_like(state["keys"])

        torch.ops._shadowkv.gather_copy_d2d_with_offsets(
            state["keys"],
            state["offsets"],
            state["counts"],
            batch_size,
            num_heads,
            sparse_budget * head_dim,
            0,
            sparse_budget * head_dim,
            select_sets,
        )
        state["chunks_i32"].copy_(chunks)
        chunk_ids = state["chunks_i32"]
        cos_sin = packed_rope
        torch.ops._shadowkv.batch_gather_gemm(
            shared.contiguous(),
            right.contiguous(),
            cos_sin,
            cos_sin,
            chunk_ids,
            self.output,
            batch_size,
            num_heads,
            shared.size(1),
            head_dim,
            rank,
            sparse_budget,
            cos_sin.size(0),
            chunk_size,
            state["counts"],
        )
        torch.ops._shadowkv.apply_rotary_pos_emb_push_cache_opt(
            self.output,
            cos_sin,
            chunk_ids,
            state["keys"],
            state["counts"],
            batch_size,
            num_heads,
            sparse_budget,
            head_dim,
            self.output.stride(0),
            self.output.stride(1),
            self.output.stride(2),
            self.output.stride(3),
            cos_sin.stride(0),
            chunk_ids.stride(0),
            chunk_ids.stride(1),
            chunk_ids.stride(2),
            state["keys"].stride(0),
            state["keys"].stride(1),
            state["keys"].stride(2),
            0,
            sparse_budget,
            head_dim // 2,
            chunk_size,
        )
        return state["keys"]

    def hit_counts(self, layer_idx: int) -> torch.Tensor:
        return self.states[layer_idx]["counts"]

    @property
    def nbytes(self) -> int:
        tensors = [self.output]
        for state in self.states.values():
            tensors.extend(state.values())
        return sum(_tensor_nbytes(tensor) for tensor in tensors)
