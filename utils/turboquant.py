from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.lloyd_max import LloydMaxCodebook


@dataclass
class CompressorParams:
    indices: torch.Tensor
    norms: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    bits: int
    idx_pad: int = 0


@dataclass
class TurboQuantFactor:
    params: CompressorParams
    bits: int


_TURBOQUANT_COMPRESSORS = {}


def get_turboquant_compressor(dim: int, bits: int, device: torch.device):
    compressor = _TURBOQUANT_COMPRESSORS.get(dim)
    if compressor is None:
        compressor = MSECompressor(dim=dim, bits=bits, device=device)
        _TURBOQUANT_COMPRESSORS[dim] = compressor
    return compressor


def quantise_factor(
    factor: torch.Tensor,
    bits: int,
) -> TurboQuantFactor:
    compressor = get_turboquant_compressor(
        dim=factor.shape[-1],
        bits=bits,
        device=factor.device,
    )
    return TurboQuantFactor(params=compressor.encode(factor), bits=bits)


def dequantise_factor(factor):
    if isinstance(factor, TurboQuantFactor):
        compressor = get_turboquant_compressor(
            dim=factor.params.shape[-1],
            bits=factor.bits,
            device=factor.params.indices.device,
        )
        return compressor.decode(factor.params)
    return factor


def factor_shape(factor) -> tuple[int, ...]:
    if isinstance(factor, TurboQuantFactor):
        return factor.params.shape
    return tuple(factor.shape)


def factor_dtype(factor) -> torch.dtype:
    if isinstance(factor, TurboQuantFactor):
        return factor.params.dtype
    return factor.dtype


def factor_nbytes(factor) -> float:
    if isinstance(factor, TurboQuantFactor):
        params = factor.params
        index_bytes = params.indices.numel()
        norm_bytes = params.norms.numel() * params.norms.element_size()
        return index_bytes + norm_bytes
    return factor.numel() * factor.element_size()


def is_quantised_factor(factor) -> bool:
    return isinstance(factor, TurboQuantFactor)


def generate_rotation_matrix(dim, seed, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(dim, dim, generator=generator)
    q, r = torch.linalg.qr(matrix)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    q = q * diag_sign.unsqueeze(0)
    return q.to(device)


class MSECompressor:
    """
    TurboQuant-style MSE coder/decoder
    """

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        seed: int = 0,
        device: str = "cpu"
    ):
        if bits < 1 or bits > 8:
            raise ValueError("MSECompressor currently supports 1 <= bits <= 8.")
        self.dim = dim
        self.bits = bits
        self.device = device

        self.rotation = generate_rotation_matrix(dim, seed + dim, device)
        self.centroids = LloydMaxCodebook(dim, bits).centroids.to(device)

    @torch.no_grad()
    def encode(self, tensor: torch.Tensor) -> CompressorParams:
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        N = tensor.numel() // self.dim

        flat = tensor.reshape(N, self.dim).float()

        # Normalise to unit sphere and store norms
        norms = flat.norm(dim=-1)
        unit = flat / (norms.unsqueeze(-1) + 1e-8)

        # Rotate and quantise
        rotated = unit @ self.rotation.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).long()

        # Bit-pack indices: pack indices_per_byte indices into each byte
        indices_per_byte = 8 // self.bits
        idx_pad = (indices_per_byte - self.dim % indices_per_byte) % indices_per_byte
        if idx_pad:
            indices = F.pad(indices, (0, idx_pad))
        n_groups = indices.shape[-1] // indices_per_byte
        idx_powers = torch.tensor(
            [2 ** (self.bits * i) for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long, device=indices.device,
        )
        idx_bytes = (indices.reshape(N, n_groups, indices_per_byte) * idx_powers).sum(-1).to(torch.uint8)

        return CompressorParams(
            indices=idx_bytes.reshape(*shape[:-1], n_groups),
            norms=norms.reshape(shape[:-1]).to(dtype=dtype),
            shape=shape,
            dtype=dtype,
            bits=self.bits,
            idx_pad=idx_pad,
        )

    @torch.no_grad()
    def decode(self, params: CompressorParams) -> torch.Tensor:
        N = params.norms.numel()
        idx_bytes = params.indices.reshape(N, -1)

        # Unpack indices
        indices_per_byte = 8 // self.bits
        mask = (1 << self.bits) - 1
        idx_shifts = torch.tensor(
            [self.bits * i for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long, device=idx_bytes.device,
        )
        indices = ((idx_bytes.long().unsqueeze(-1) >> idx_shifts) & mask).reshape(N, -1)
        if params.idx_pad:
            indices = indices[:, :self.dim]

        reconstructed = (self.centroids[indices] @ self.rotation) * params.norms.reshape(-1, 1).float()
        return reconstructed.reshape(params.shape).to(dtype=params.dtype)

    def memory_nbytes(self, params: CompressorParams) -> float:
        # .numel can be used directly to count bytes after bit-packing
        index_bytes = params.indices.numel()
        norm_bytes = params.norms.numel() * torch.finfo(params.norms.dtype).bits / 8
        return index_bytes + norm_bytes


def init_compressor(turboquant_residuals, compressor_bits, dim, device):
    if not turboquant_residuals:
        return None
    return MSECompressor(dim=dim, bits=compressor_bits, device=device)
