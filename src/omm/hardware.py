"""Cross-platform hardware scanning for RAM/VRAM/OS detection."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass

import psutil


@dataclass
class HardwareInfo:
    os_name: str
    os_version: str
    cpu: str
    ram_total_gb: float
    ram_available_gb: float
    unified_memory: bool
    gpu_name: str | None
    vram_total_gb: float | None
    vram_free_gb: float | None


_OS_DISPLAY_NAMES = {"Darwin": "macOS"}


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _mac_cpu_brand() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return platform.processor() or "Unknown"


def _mac_chip_name() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Apple Silicon"


def _scan_nvidia_vram() -> tuple[str | None, float | None, float | None]:
    """Return (gpu_name, vram_total_gb, vram_free_gb) or (None, None, None) if unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = mem.total / (1024**3)
            free_gb = mem.free / (1024**3)
            return name, total_gb, free_gb
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None, None, None


def scan_hardware() -> HardwareInfo:
    vm = psutil.virtual_memory()
    ram_total_gb = vm.total / (1024**3)
    ram_available_gb = vm.available / (1024**3)

    raw_os_name = platform.system()
    os_name = _OS_DISPLAY_NAMES.get(raw_os_name, raw_os_name)
    os_version = platform.release()

    if _is_apple_silicon():
        cpu = _mac_cpu_brand()
        return HardwareInfo(
            os_name=os_name,
            os_version=os_version,
            cpu=cpu,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            unified_memory=True,
            gpu_name=_mac_chip_name(),
            vram_total_gb=ram_total_gb,
            vram_free_gb=ram_available_gb,
        )

    if raw_os_name == "Darwin":
        cpu = _mac_cpu_brand()
    else:
        cpu = platform.processor() or platform.machine()

    gpu_name, vram_total_gb, vram_free_gb = _scan_nvidia_vram()

    return HardwareInfo(
        os_name=os_name,
        os_version=os_version,
        cpu=cpu,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        unified_memory=False,
        gpu_name=gpu_name,
        vram_total_gb=vram_total_gb,
        vram_free_gb=vram_free_gb,
    )
