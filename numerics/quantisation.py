from dataclasses import dataclass

import torch
import torch.nn.functional as F

from numerics.lloyd_max import LloydMaxCodebook


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
    device = torch.device(device)
    key = (dim, bits, device)
    compressor = _TURBOQUANT_COMPRESSORS.get(key)
    if compressor is None:
        compressor = MSECompressor(dim=dim, bits=bits, device=device)
        _TURBOQUANT_COMPRESSORS[key] = compressor
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


def fold_rotation(factor, right: torch.Tensor) -> torch.Tensor:
    """Fold rotation into right factor."""
    compressor = get_turboquant_compressor(
        factor.params.shape[-1], factor.bits, right.device
    )
    return torch.matmul(compressor.rotation, right.float()).to(right.dtype)


def gather_factor_rows_unrotated(factor, positions: torch.Tensor):
    """Gather rows of a quantised factor."""
    if not is_quantised_factor(factor):
        raise TypeError(
            "gather_factor_rows_unrotated expects a quantised factor."
        )
    params = factor.params
    indices, norms = _gather_compressed_rows(params, positions)
    compressor = get_turboquant_compressor(
        params.shape[-1], factor.bits, indices.device
    )
    rows = compressor.byte_table(params.dtype)[indices.long()]
    rows = rows.reshape(*indices.shape[:-1], -1)
    dim = params.shape[-1]
    return (rows[..., :dim] if compressor.idx_pad else rows), norms


def dequantise_factor_unrotated(factor):
    """Decode a quantised factor up to, but not including, its rotation."""
    if not is_quantised_factor(factor):
        return factor
    params = factor.params
    compressor = get_turboquant_compressor(
        params.shape[-1], factor.bits, params.indices.device
    )
    rows = compressor.byte_table(params.dtype)[params.indices.long()]
    rows = rows.reshape(*params.indices.shape[:-1], -1)
    if compressor.idx_pad:
        rows = rows[..., : params.shape[-1]]
    return rows * params.norms[..., None].to(rows.dtype)


def folded_reconstruction(
    indices: torch.Tensor,
    norms: torch.Tensor,
    positions: torch.Tensor,
    byte_table: torch.Tensor,
    right: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Gather quantised rows and apply a right factor that carries rotation."""
    num_heads = positions.size(1)
    num_groups = indices.size(-1)
    gathered = (
        indices[:, None]
        .expand(-1, num_heads, -1, -1)
        .gather(2, positions[..., None].expand(-1, -1, -1, num_groups))
    )
    row_norms = norms[:, None].expand(-1, num_heads, -1).gather(2, positions)
    rows = byte_table[gathered.long()]
    rows = rows.reshape(*gathered.shape[:-1], -1)[..., :dim]
    return torch.matmul(rows, right) * row_norms[..., None].to(rows.dtype)


def _params_view(params: CompressorParams, batch_slice: slice, batch: int):
    return CompressorParams(
        indices=params.indices[batch_slice],
        norms=params.norms[batch_slice],
        shape=(batch, *params.shape[1:]),
        dtype=params.dtype,
        bits=params.bits,
        idx_pad=params.idx_pad,
    )


def select_compressed_rows(
    params: CompressorParams,
    rows: torch.Tensor,
) -> CompressorParams:
    """Row-slice an `[N, D]` compressed store, still compressed.

    Lets a caller decode only the rows it asked for instead of expanding the
    whole store to pick a few out of it.
    """
    return CompressorParams(
        indices=params.indices.index_select(0, rows),
        norms=params.norms.index_select(0, rows),
        shape=(rows.numel(), *params.shape[1:]),
        dtype=params.dtype,
        bits=params.bits,
        idx_pad=params.idx_pad,
    )


def pack_factors(factors: list):
    """Stack per-batch factors of shape `[1, ...]` into one `[B, ...]` factor.

    Returns the packed factor plus per-batch views onto its storage, so the
    batched copy is the only one that stays alive.
    """
    if is_quantised_factor(factors[0]):
        reference = factors[0].params
        packed_params = CompressorParams(
            indices=torch.stack(
                [factor.params.indices.squeeze(0) for factor in factors]
            ),
            norms=torch.stack(
                [factor.params.norms.squeeze(0) for factor in factors]
            ),
            shape=(len(factors), *reference.shape[1:]),
            dtype=reference.dtype,
            bits=reference.bits,
            idx_pad=reference.idx_pad,
        )
        packed = TurboQuantFactor(params=packed_params, bits=factors[0].bits)
        views = [
            TurboQuantFactor(
                params=_params_view(packed_params, slice(idx, idx + 1), 1),
                bits=packed.bits,
            )
            for idx in range(len(factors))
        ]
        return packed, views

    packed = torch.stack([factor.squeeze(0) for factor in factors])
    return packed, [packed[idx].unsqueeze(0) for idx in range(len(factors))]


def _gather_compressed_rows(params: CompressorParams, positions: torch.Tensor):
    """Gather the packed bytes and norms."""
    num_heads = positions.size(1)
    num_groups = params.indices.size(-1)
    indices = (
        params.indices[:, None]
        .expand(-1, num_heads, -1, -1)
        .gather(2, positions[..., None].expand(-1, -1, -1, num_groups))
    )
    norms = params.norms[:, None].expand(-1, num_heads, -1).gather(2, positions)
    return indices, norms


def gather_factor_rows(factor, positions: torch.Tensor) -> torch.Tensor:
    """Gather rows of a `[B, T, D]` factor at per-head `[B, H, S]` positions.

    Quantised factors are decoded after gathering, so only the selected rows
    are ever materialised.
    """
    batch, num_heads, selected_len = positions.shape
    if not is_quantised_factor(factor):
        return factor[:, None].expand(-1, num_heads, -1, -1).gather(
            2,
            positions[..., None].expand(-1, -1, -1, factor.size(-1)),
        )

    params = factor.params
    indices, norms = _gather_compressed_rows(params, positions)
    dim = params.shape[-1]
    compressor = get_turboquant_compressor(dim, factor.bits, indices.device)
    return compressor.decode(
        CompressorParams(
            indices=indices,
            norms=norms,
            shape=(batch, num_heads, selected_len, dim),
            dtype=params.dtype,
            bits=params.bits,
            idx_pad=params.idx_pad,
        )
    )


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
        self, dim: int, bits: int = 3, seed: int = 0, device: str = "cpu"
    ):
        if bits < 1 or bits > 8:
            raise ValueError("MSECompressor currently supports 1 <= bits <= 8.")
        self.dim = dim
        self.bits = bits
        self.device = device

        self.rotation = generate_rotation_matrix(dim, seed + dim, device)
        self.centroids = LloydMaxCodebook(dim, bits).centroids.to(device)

        self.indices_per_byte = 8 // bits
        self.mask = (1 << bits) - 1
        idx_pad = (
            self.indices_per_byte - dim % self.indices_per_byte
        ) % self.indices_per_byte
        self.idx_pad = idx_pad
        self.n_groups = (dim + idx_pad) // self.indices_per_byte
        self.idx_powers = torch.tensor(
            [2 ** (bits * i) for i in range(self.indices_per_byte - 1, -1, -1)],
            dtype=torch.long,
            device=device,
        )
        self.idx_shifts = torch.tensor(
            [bits * i for i in range(self.indices_per_byte - 1, -1, -1)],
            dtype=torch.long,
            device=device,
        )
        self._byte_tables: dict[torch.dtype, torch.Tensor] = {}

    def byte_table(self, dtype: torch.dtype) -> torch.Tensor:
        table = self._byte_tables.get(dtype)
        if table is None:
            byte_values = torch.arange(
                256, device=self.centroids.device, dtype=torch.long
            )
            codes = (byte_values[:, None] >> self.idx_shifts) & self.mask
            table = self.centroids.to(dtype)[codes]
            self._byte_tables[dtype] = table
        return table

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
        if self.idx_pad:
            indices = F.pad(indices, (0, self.idx_pad))
        idx_bytes = (
            (
                indices.reshape(N, self.n_groups, self.indices_per_byte)
                * self.idx_powers
            )
            .sum(-1)
            .to(torch.uint8)
        )

        return CompressorParams(
            indices=idx_bytes.reshape(*shape[:-1], self.n_groups),
            norms=norms.reshape(shape[:-1]).to(dtype=dtype),
            shape=shape,
            dtype=dtype,
            bits=self.bits,
            idx_pad=self.idx_pad,
        )

    @torch.no_grad()
    def decode(self, params: CompressorParams) -> torch.Tensor:
        N = params.norms.numel()
        idx_bytes = params.indices.reshape(N, -1)

        # Unpack indices
        indices = (
            (idx_bytes.long().unsqueeze(-1) >> self.idx_shifts) & self.mask
        ).reshape(N, -1)
        if params.idx_pad:
            indices = indices[:, : self.dim]

        reconstructed = (
            self.centroids[indices] @ self.rotation
        ) * params.norms.reshape(-1, 1).float()
        return reconstructed.reshape(params.shape).to(dtype=params.dtype)

    def memory_nbytes(self, params: CompressorParams) -> float:
        # .numel can be used directly to count bytes after bit-packing
        index_bytes = params.indices.numel()
        norm_bytes = (
            params.norms.numel() * torch.finfo(params.norms.dtype).bits / 8
        )
        return index_bytes + norm_bytes


def init_compressor(turboquant_residuals, compressor_bits, dim, device):
    if not turboquant_residuals:
        return None
    return MSECompressor(dim=dim, bits=compressor_bits, device=device)
