# ComfyUI Meituan Image - LongCat Image Integration
# https://github.com/meituan-longcat/LongCat-Image

import subprocess
import sys

def ensure_longcat_installed():
    """Check if longcat-image is installed, if not install it automatically."""
    try:
        import longcat_image
        print("[LongCat] longcat-image package found.")
        return True
    except ImportError:
        print("[LongCat] longcat-image not found. Installing from GitHub...")
        try:
            # Install longcat-image from GitHub
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/meituan-longcat/LongCat-Image.git@main",
                "--quiet"
            ])
            print("[LongCat] longcat-image installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[LongCat] Failed to install longcat-image: {e}")
            print("[LongCat] Please install manually: pip install git+https://github.com/meituan-longcat/LongCat-Image.git@main")
            return False

# Auto-install longcat-image on load
ensure_longcat_installed()

from .nodes import (
    MeituanLongCatLoader,
    MeituanLongCatT2I,
    MeituanLongCatEdit,
    MeituanLongCatExtension,
    comfy_entrypoint,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = [
    "MeituanLongCatLoader",
    "MeituanLongCatT2I",
    "MeituanLongCatEdit",
    "MeituanLongCatExtension",
    "comfy_entrypoint",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
