"""
GPU detection and VRAM management utilities.
Detects NVIDIA GPU via torch.cuda, falls back to CPU.
Selects optimal Whisper model based on available VRAM.
"""

import logging
import subprocess
import shutil
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    available: bool
    device: str  # "cuda", "cpu", or "mps"
    name: str
    vram_total_mb: int
    vram_free_mb: int
    compute_type: str  # "float16", "int8", "int8_float16", "float32"


# Cache the result so we only probe once per process lifetime
_gpu_info_cache: GPUInfo | None = None


def detect_gpu() -> GPUInfo:
    """Detect the best available compute device and return GPU info.

    The result is cached: hardware doesn't change at runtime.
    """
    global _gpu_info_cache
    if _gpu_info_cache is not None:
        return _gpu_info_cache

    _gpu_info_cache = _detect_gpu_uncached()
    return _gpu_info_cache


def _detect_gpu_uncached() -> GPUInfo:
    # Try nvidia-smi first (works without PyTorch, which is optional)
    nvidia_smi = _detect_via_nvidia_smi()
    if nvidia_smi:
        # Verify CUDA runtime libraries actually work before claiming GPU support
        if _verify_cuda_runtime():
            return nvidia_smi
        else:
            logger.warning(
                f"GPU detected ({nvidia_smi.name}) but CUDA runtime libraries are missing. "
                f"Install CUDA Toolkit 12.x or run: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
            )
            # Return CPU info but keep GPU name for display
            return GPUInfo(
                available=False,
                device="cpu",
                name=f"{nvidia_smi.name} (CUDA indisponible)",
                vram_total_mb=nvidia_smi.vram_total_mb,
                vram_free_mb=0,
                compute_type="int8",
            )

    # Try PyTorch CUDA (if installed)
    try:
        import torch
        if torch.cuda.is_available():
            device_index = 0
            name = torch.cuda.get_device_name(device_index)
            vram_total = torch.cuda.get_device_properties(device_index).total_mem
            vram_free = vram_total - torch.cuda.memory_reserved(device_index)
            vram_total_mb = int(vram_total / (1024 * 1024))
            vram_free_mb = int(vram_free / (1024 * 1024))
            logger.info(f"CUDA GPU detected via PyTorch: {name} ({vram_total_mb} MB total, {vram_free_mb} MB free)")
            return GPUInfo(
                available=True,
                device="cuda",
                name=name,
                vram_total_mb=vram_total_mb,
                vram_free_mb=vram_free_mb,
                compute_type="float16",
            )
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"PyTorch CUDA detection failed: {e}")

    # Try Apple MPS (macOS)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS device detected")
            return GPUInfo(
                available=True,
                device="mps",
                name="Apple MPS",
                vram_total_mb=0,
                vram_free_mb=0,
                compute_type="float32",
            )
    except Exception:
        pass

    # CPU fallback
    logger.info("No GPU detected, falling back to CPU")
    return GPUInfo(
        available=False,
        device="cpu",
        name="CPU",
        vram_total_mb=0,
        vram_free_mb=0,
        compute_type="int8",
    )


def _verify_cuda_runtime() -> bool:
    """Check that CUDA runtime libraries (cuBLAS, cuDNN) are actually loadable.

    ctranslate2.get_supported_compute_types('cuda') only lists types *compiled*
    into the binary — it does NOT verify that cublas64_12.dll is present.
    pip packages install DLLs into site-packages/nvidia/*/bin/, not on PATH.
    """
    _register_nvidia_dll_paths()

    # 1) Verify cuBLAS is loadable
    cublas_ok = False
    try:
        import ctypes
        import sys
        if sys.platform == "win32":
            ctypes.WinDLL("cublas64_12.dll")
        else:
            ctypes.CDLL("libcublas.so.12")
        cublas_ok = True
        logger.info("CUDA runtime verified: cublas library loaded")
    except OSError:
        pass

    if not cublas_ok:
        # Try PyTorch as last resort (bundles its own CUDA)
        try:
            import torch
            if torch.cuda.is_available():
                torch.zeros(1, device="cuda")
                logger.info("CUDA runtime verified via PyTorch")
                return True
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"PyTorch CUDA runtime check failed: {e}")

        logger.warning("CUDA runtime libraries (cublas) not found")
        return False

    return True


def _register_nvidia_dll_paths():
    """Register DLL directories from nvidia pip packages so CTranslate2 can find them.

    This must be called before any CTranslate2/faster-whisper model loading.
    Safe to call multiple times — paths are only added once.
    """
    import os
    import sys

    if sys.platform != "win32":
        # On Linux, pip packages use LD_LIBRARY_PATH or rpath — no action needed
        return

    if not hasattr(os, "add_dll_directory"):
        return

    nvidia_packages = ["nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime", "nvidia.cufft", "nvidia.curand"]
    for pkg_name in nvidia_packages:
        try:
            pkg = __import__(pkg_name, fromlist=[""])
            # Namespace packages have __file__=None, use __path__ instead
            if hasattr(pkg, "__path__") and pkg.__path__:
                pkg_dir = list(pkg.__path__)[0]
            elif pkg.__file__:
                pkg_dir = os.path.dirname(pkg.__file__)
            else:
                continue
            bin_dir = os.path.join(pkg_dir, "bin")
            lib_dir = os.path.join(pkg_dir, "lib")
            for d in (bin_dir, lib_dir):
                if os.path.isdir(d):
                    os.add_dll_directory(d)
                    logger.debug(f"Registered DLL directory: {d}")
        except ImportError:
            continue
        except Exception as e:
            logger.debug(f"Could not register {pkg_name}: {e}")


def _detect_via_nvidia_smi() -> GPUInfo | None:
    """Try detecting NVIDIA GPU via nvidia-smi command."""
    nvidia_smi_path = shutil.which("nvidia-smi")
    if not nvidia_smi_path:
        return None
    try:
        result = subprocess.run(
            [nvidia_smi_path, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                name = parts[0]
                vram_total_mb = int(parts[1])
                vram_free_mb = int(parts[2])
                logger.info(f"nvidia-smi GPU: {name} ({vram_total_mb} MB total, {vram_free_mb} MB free)")

                # Determine best compute type for this GPU
                compute_type = _safe_compute_type("cuda")

                return GPUInfo(
                    available=True,
                    device="cuda",
                    name=name,
                    vram_total_mb=vram_total_mb,
                    vram_free_mb=vram_free_mb,
                    compute_type=compute_type,
                )
    except Exception as e:
        logger.warning(f"nvidia-smi detection failed: {e}")
    return None


def _safe_compute_type(device: str) -> str:
    """Determine a compute type that is actually supported on this device."""
    if device == "cpu":
        return "int8"

    # 1) Try PyTorch to check compute capability
    try:
        import torch
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            if capability[0] >= 7:
                # Turing+ (RTX 20xx, 30xx, 40xx) — tensor cores: int8_float16 is fastest
                return "int8_float16"
            elif capability[0] >= 6:
                # Pascal (GTX 1080 Ti etc.) — no tensor cores
                return "int8"
            else:
                return "int8"
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"PyTorch CUDA capability check failed: {e}")

    # 2) Try CTranslate2 directly to verify CUDA support
    try:
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types("cuda")
        logger.info(f"CTranslate2 CUDA supported compute types: {supported}")
        for preferred in ("int8_float16", "int8", "float16", "float32"):
            if preferred in supported:
                return preferred
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"CTranslate2 CUDA check failed: {e}")

    # 3) nvidia-smi found a GPU — trust it, use int8 on CUDA (universally supported)
    logger.info("Neither PyTorch nor CTranslate2 could verify CUDA details, defaulting to int8 on CUDA")
    return "int8"


AVAILABLE_MODELS = ["large-v3", "medium", "small", "base", "tiny"]

# Approximate VRAM footprint per model in MB (int8_float16 / int8)
_MODEL_VRAM_MB = {
    "large-v3": 3200,
    "medium": 1600,
    "small": 600,
    "base": 300,
    "tiny": 150,
}

# Approximate VRAM per batch element (encoder activations + decoder KV cache + workspace)
# Scales with model size — measured empirically per model family
_BATCH_ELEMENT_MB = {
    "large-v3": 350,
    "medium": 180,
    "small": 80,
    "base": 50,
    "tiny": 30,
}


def estimate_batch_size(model_size: str, vram_free_mb: int) -> int:
    """Estimate the maximum batch_size that fits in available VRAM.

    Returns 0 for CPU (no batching).
    """
    if vram_free_mb <= 0:
        return 0

    model_vram = _MODEL_VRAM_MB.get(model_size, 3200)
    headroom = vram_free_mb - model_vram - 300  # 300 MB safety margin for OS/drivers

    if headroom <= 0:
        return 1

    element_mb = _BATCH_ELEMENT_MB.get(model_size, 350)
    batch = min(32, max(2, int(headroom / element_mb)))
    logger.info(
        f"Estimated batch_size={batch} for {model_size} "
        f"(VRAM free: {vram_free_mb} MB, model: ~{model_vram} MB, headroom: {headroom} MB)"
    )
    return batch


def select_model_size(gpu_info: GPUInfo) -> str:
    """Select the optimal Whisper model size based on available VRAM/RAM."""
    if gpu_info.device in ("cuda", "mps") and gpu_info.vram_total_mb > 0:
        vram = gpu_info.vram_free_mb if gpu_info.vram_free_mb > 0 else gpu_info.vram_total_mb
        # int8_float16 uses ~40% less VRAM than float16, so we can be more aggressive
        is_quantized = gpu_info.compute_type in ("int8", "int8_float16")
        if is_quantized:
            if vram >= 4_000:
                return "large-v3"
            elif vram >= 2_500:
                return "medium"
            elif vram >= 1_500:
                return "small"
            else:
                return "base"
        else:
            if vram >= 6_000:
                return "large-v3"
            elif vram >= 4_000:
                return "medium"
            elif vram >= 2_000:
                return "small"
            else:
                return "base"
    else:
        # CPU: check system RAM — int8 models use ~50% less RAM than float
        # large-v3 int8 ~3 GB RAM, medium int8 ~1.5 GB, small int8 ~0.5 GB
        try:
            import psutil
            ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            logger.info(f"CPU model selection: {ram_gb:.1f} GB RAM available")
            if ram_gb >= 8:
                return "large-v3"
            if ram_gb >= 5:
                return "medium"
            if ram_gb >= 2:
                return "small"
            return "base"
        except ImportError:
            return "small"


def get_system_info() -> dict:
    """Return a summary dict of system capabilities for the frontend."""
    gpu_info = detect_gpu()
    model_size = select_model_size(gpu_info)

    ffmpeg_available = shutil.which("ffmpeg") is not None

    ram_gb = 0
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass

    return {
        "gpu_available": gpu_info.available,
        "gpu_device": gpu_info.device,
        "gpu_name": gpu_info.name,
        "vram_total_mb": gpu_info.vram_total_mb,
        "vram_free_mb": gpu_info.vram_free_mb,
        "compute_type": gpu_info.compute_type,
        "selected_model": model_size,
        "available_models": AVAILABLE_MODELS,
        "ram_gb": ram_gb,
        "ffmpeg_available": ffmpeg_available,
        "cuda_warning": (
            "CUDA indisponible — installez CUDA Toolkit 12.x ou lancez : pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
            if "indisponible" in gpu_info.name else ""
        ),
    }
