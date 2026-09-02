import torch

from numerics import quantisation


def test_turboquant_compressors_are_cached_per_device(monkeypatch):
    created = []

    class FakeCompressor:
        def __init__(self, dim, bits, device):
            self.device = device
            created.append((dim, bits, device))

    monkeypatch.setattr(quantisation, "MSECompressor", FakeCompressor)
    monkeypatch.setattr(quantisation, "_TURBOQUANT_COMPRESSORS", {})

    cpu = quantisation.get_turboquant_compressor(64, 4, "cpu")
    cuda0 = quantisation.get_turboquant_compressor(64, 4, "cuda:0")
    cuda1 = quantisation.get_turboquant_compressor(64, 4, "cuda:1")

    assert cpu is quantisation.get_turboquant_compressor(
        64, 4, torch.device("cpu")
    )
    assert cuda0 is not cuda1
    assert len(created) == 3
