import os

if os.environ.get("RUN_CUDA_TESTS") != "1":
    os.environ.setdefault("XKV_NO_BUILD", "1")
