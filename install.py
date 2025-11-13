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
            print("[CADabra] ✓ libGLU found")
            return True
    except FileNotFoundError:
        pass

    print("[CADabra] libGLU not found, attempting installation...")

    # Check if we have root/sudo access
    is_root = os.geteuid() == 0 if hasattr(os, 'geteuid') else False

    if not is_root:
        # Try with sudo
        print("[CADabra] Checking for sudo access...")
        sudo_check = subprocess.run(["sudo", "-n", "true"], capture_output=True)
        has_sudo = sudo_check.returncode == 0

        if not has_sudo:
            print("[CADabra] ✗ No root or sudo access available")
            print("[CADabra] Run manually: sudo apt-get update && sudo apt-get install -y libglu1-mesa libxft2")
            return False

        cmd_prefix = ["sudo"]
    else:
        print("[CADabra] Running as root, installing directly...")
        cmd_prefix = []

    # Install all required system libraries for gmsh
    # libglu1-mesa: OpenGL utility library (libGLU.so.1)
    # libxft2: X11 FreeType interface library (libXft.so.2)
    # Both are required by gmsh's GUI/visualization components
    install_cmds = [
        cmd_prefix + ["apt-get", "update"],
        cmd_prefix + ["apt-get", "install", "-y", "libglu1-mesa", "libxft2"]
    ]

    for cmd in install_cmds:
        try:
            subprocess.check_call(cmd)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[CADabra] ✗ Installation failed: {e}")
            print("[CADabra] Run manually: sudo apt-get update && sudo apt-get install -y libglu1-mesa libxft2")
            return False

    # Verify installation
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True
        )
        if "libGLU.so" in result.stdout:
            print("[CADabra] ✓ libGLU successfully installed")
            return True
        else:
            print("[CADabra] ✗ Installation completed but libGLU not found in ldconfig")
            return False
    except FileNotFoundError:
        print("[CADabra] ✓ Installation completed (ldconfig not available to verify)")
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
