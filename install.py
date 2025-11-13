"""
Installation script for ComfyUI-CADabra
"""

import subprocess
import sys
import os
import platform


def install_system_deps():
    """Install system dependencies for Gmsh"""
    if platform.system() != "Linux":
        print("[CADabra] System dependency check skipped (non-Linux)")
        return True

    print("[CADabra] Checking system dependencies...")

    try:
        # Check if libGLU is available
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True
        )
        if "libGLU.so" in result.stdout:
            print("[CADabra] libGLU found")
            return True
    except FileNotFoundError:
        pass

    # Try to install libGLU
    print("[CADabra] Installing libglu1-mesa...")
    install_cmds = [
        ["apt-get", "update"],
        ["apt-get", "install", "-y", "libglu1-mesa"]
    ]

    for cmd in install_cmds:
        try:
            subprocess.check_call(cmd)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("[CADabra] WARNING: Could not auto-install system dependencies")
            print("[CADabra] Run manually: sudo apt-get install libglu1-mesa")
            return False

    print("[CADabra] System dependencies installed")
    return True


def install():
    """Install required dependencies"""
    print("[CADabra] Installing dependencies...")

    # Install system dependencies first
    install_system_deps()

    # Install Python packages
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_path
        ])
        print("[CADabra] Installation complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[CADabra] Installation failed: {e}")
        return False


if __name__ == "__main__":
    install()
