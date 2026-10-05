"""Build and install xKV's fused CUDA kernels into ``efficiency/ops/``."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from efficiency.xkv import (
    _OPS_DIR,
    _XKV_COMMIT,
    _BUILD_DIR,
    _clone_xkv_source,
    _extension_kwargs,
)

MIN_CAPABILITY = (8, 0)


def _library_tag() -> str:
    import torch

    python = f"cp{sys.version_info.major}{sys.version_info.minor}"
    cuda = (torch.version.cuda or "none").replace(".", "")
    torch_version = torch.__version__.split("+")[0]
    return f"{python}-torch{torch_version}-cu{cuda}"


def _installed_libraries() -> list[Path]:
    return sorted(_OPS_DIR.glob("_shadowkv*.so"))


def _check_toolchain() -> list[str]:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    problems = []
    if torch.version.cuda is None:
        problems.append(
            "this torch build has no CUDA support "
            f"(torch {torch.__version__}); install a CUDA build"
        )
    if CUDA_HOME is None and shutil.which("nvcc") is None:
        problems.append(
            "nvcc was not found; install the CUDA toolkit or set CUDA_HOME"
        )
    if shutil.which("ninja") is None:
        problems.append("ninja was not found; `pip install ninja`")

    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        if not torch.cuda.is_available():
            problems.append(
                "no CUDA device is visible, so the target architecture "
                "cannot be detected"
            )
        elif torch.cuda.get_device_capability() < MIN_CAPABILITY:
            major, minor = torch.cuda.get_device_capability()
            problems.append(
                f"device capability is sm_{major}{minor}, but the fused "
                "kernels require sm_80 or newer"
            )
    return problems


def _verify_installed(library: Path) -> bool:
    """Load the installed .so in a fresh process, with JIT fallback off."""
    env = {**os.environ, "XKV_NO_BUILD": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from efficiency import FusedKeyReconstructor as R;"
            "import sys; sys.exit(0 if R.available() else 1)",
        ],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Installed kernel at {library} did not load:")
        print(result.stdout.strip() or result.stderr.strip())
    return result.returncode == 0


def install(
    build_dir: Path,
    force: bool,
) -> int:
    if not force:
        libraries = _installed_libraries()
        if libraries and _verify_installed(libraries[0]):
            print(
                f"Fused kernel already installed ({libraries[0].name}); "
                "pass --force to rebuild."
            )
            return 0

    problems = _check_toolchain()
    if problems:
        for problem in problems:
            print(f"Cannot build: {problem}")
        return 1

    src_dir = _clone_xkv_source()
    if src_dir is None:
        return 1

    if force:
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Building xKV kernels from {_XKV_COMMIT[:12]} in {build_dir}; "
        "this takes several minutes."
    )
    try:
        from torch.utils.cpp_extension import load

        load(
            name="_shadowkv",
            build_directory=str(build_dir),
            verbose=True,
            **_extension_kwargs(src_dir),
        )
    except Exception as exc:
        print(f"Build failed: {exc}")
        return 1

    built = build_dir / "_shadowkv.so"
    if not built.is_file():
        print(f"Build reported success but {built} is missing.")
        return 1

    _OPS_DIR.mkdir(parents=True, exist_ok=True)
    installed = _OPS_DIR / f"_shadowkv-{_library_tag()}.so"
    shutil.copy2(built, installed)
    print(f"Installed {installed}")

    if not _verify_installed(installed):
        installed.unlink(missing_ok=True)
        return 1
    print("Fused kernel is loadable; selective reconstruction will use it.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild and reinstall even if a working kernel is present",
    )

    parser.add_argument(
        "--build_dir",
        type=Path,
        default=_BUILD_DIR / "build",
        help="scratch directory for the ninja build",
    )
    args = parser.parse_args()

    return install(args.build_dir, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
