"""
Installation script for ComfyUI-CADabra
"""

import subprocess
import sys
import os
import platform
import shutil


def is_conda_available():
    """Check if conda executable is available in PATH"""
    return shutil.which("conda") is not None


def install_with_conda(package_name, channels=None):
    """
    Install package using conda into current environment

    Args:
        package_name: Package to install (e.g., "pythonocc-core=7.8.1")
        channels: List of conda channels to use (default: ["conda-forge"])

    Returns:
        bool: True if successful
    """
    if channels is None:
        channels = ["conda-forge"]

    conda_exe = shutil.which("conda")
    if not conda_exe:
        return False

    env_path = sys.prefix

    print(f"[CADabra] Installing {package_name} with conda...")
    print(f"[CADabra] Target environment: {env_path}")

    cmd = [
        conda_exe,
        "install",
        "-y",  # Auto yes
        "-p", env_path,  # Install to specific environment
    ]
    # Add channels
    for channel in channels:
        cmd.extend(["-c", channel])
    cmd.append(package_name)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            print(f"[CADabra] ✓ Successfully installed {package_name}")
            return True
        else:
            print(f"[CADabra] Conda installation failed:")
            print(result.stderr[:500])  # Print first 500 chars of error
            return False

    except subprocess.TimeoutExpired:
        print(f"[CADabra] Installation timed out (>10 minutes)")
        return False
    except Exception as e:
        print(f"[CADabra] Error during installation: {e}")
        return False


def fix_torchvision_libjpeg_conflict():
    """
    Detect and fix torchvision's bundled old libjpeg that conflicts with pythonocc-core.
    Linux-only fix.

    Returns:
        bool: True if conflict was detected and fixed, False otherwise
    """
    if platform.system() != "Linux":
        return False

    import glob

    print("\n[CADabra] Checking for torchvision libjpeg conflicts...")

    try:
        site_packages = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
        torchvision_libs = os.path.join(site_packages, "torchvision.libs")

        if not os.path.exists(torchvision_libs):
            print("[CADabra] ✓ No torchvision.libs directory found - no conflict possible")
            return False

        libjpeg_files = glob.glob(os.path.join(torchvision_libs, "libjpeg*.so*"))

        if not libjpeg_files:
            print("[CADabra] ✓ No bundled libjpeg found in torchvision.libs")
            return False

        print(f"[CADabra] Found {len(libjpeg_files)} bundled libjpeg file(s) in torchvision.libs")

        # Check if any of these files lack jpeg12 symbols (old version)
        conflict_detected = False
        for libjpeg_file in libjpeg_files:
            try:
                result = subprocess.run(
                    ["nm", "-D", libjpeg_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    if "jpeg12_write_raw_data" not in result.stdout:
                        conflict_detected = True
                        print(f"[CADabra] ⚠ Conflict detected: {os.path.basename(libjpeg_file)} lacks jpeg12 functions")
                        break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                conflict_detected = True
                break

        if not conflict_detected:
            print("[CADabra] ✓ torchvision's libjpeg has jpeg12 support - no conflict")
            return False

        # Fix by backing up the old libjpeg files
        print("[CADabra] Fixing library conflict...")
        fixed_count = 0
        for libjpeg_file in libjpeg_files:
            try:
                backup_file = libjpeg_file + ".backup"
                os.rename(libjpeg_file, backup_file)
                print(f"[CADabra]   Backed up: {os.path.basename(libjpeg_file)}")
                fixed_count += 1
            except Exception as e:
                print(f"[CADabra]   ✗ Failed to backup {os.path.basename(libjpeg_file)}: {e}")

        if fixed_count > 0:
            print(f"[CADabra] ✓ Fixed {fixed_count} conflicting library file(s)")
            return True
        return False

    except Exception as e:
        print(f"[CADabra] Error checking for torchvision conflict: {e}")
        return False


def install_conda_packages():
    """
    Install conda packages (pythonocc-core, cadquery, and occt-rt).
    These must be installed together with compatible versions due to shared OCCT dependency.

    Returns:
        bool: True if successful
    """
    print("\n[CADabra] ==========================================================")
    print("[CADabra] Installing CAD packages (pythonocc-core + cadquery + occt-rt)")
    print("[CADabra] ==========================================================\n")

    if not is_conda_available():
        print("[CADabra] ✗ Conda is required to install CAD packages")
        print("[CADabra] Please install Anaconda or Miniconda first")
        print("[CADabra] https://docs.conda.io/en/latest/miniconda.html")
        return False

    # Install all packages together to ensure compatible OCCT versions
    # pythonocc-core=7.8.1, cadquery=2.6.1, and occt-rt all use occt=7.8.1
    conda_exe = shutil.which("conda")
    env_path = sys.prefix

    print("[CADabra] Installing pythonocc-core=7.8.1, cadquery=2.6.1, and occt-rt...")
    print("[CADabra] (These share OCCT 7.8.1 as a common dependency)")
    print(f"[CADabra] Target environment: {env_path}")

    cmd = [
        conda_exe,
        "install",
        "-y",
        "-p", env_path,
        "-c", "pozzettiandrea",
        "-c", "cadquery",
        "-c", "conda-forge",
        "pythonocc-core=7.8.1",
        "cadquery=2.6.1",
        "occt-rt>=1.1.0",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900  # 15 minute timeout for both packages
        )

        if result.returncode == 0:
            print("[CADabra] ✓ Successfully installed pythonocc-core, cadquery, and occt-rt")
            fix_torchvision_libjpeg_conflict()
            return True
        else:
            print("[CADabra] Conda installation failed:")
            print(result.stderr[:1000])
            return False

    except subprocess.TimeoutExpired:
        print("[CADabra] Installation timed out (>15 minutes)")
        return False
    except Exception as e:
        print(f"[CADabra] Error during installation: {e}")
        return False


def install_pymesh():
    """
    Install PyMesh from pre-built wheels for mesh clipping in Point2CAD.
    Downloads from GitHub releases based on Python version.

    Returns:
        bool: True if successful
    """
    print("\n[CADabra] ==========================================================")
    print("[CADabra] Installing PyMesh for Point2CAD mesh clipping")
    print("[CADabra] ==========================================================\n")

    # Check if already installed
    try:
        import pymesh
        print(f"[CADabra] ✓ PyMesh {pymesh.__version__} already installed")
        return True
    except ImportError:
        pass

    # Only Linux wheels available for now
    if platform.system() != "Linux":
        print(f"[CADabra] ⚠ PyMesh wheels only available for Linux (current: {platform.system()})")
        print("[CADabra]   Point2CAD mesh clipping will use fallback method")
        return False

    # Determine wheel URL based on Python version
    py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    wheel_base = "https://github.com/PozzettiAndrea/PyMesh/releases/download/v0.3"
    wheel_name = f"pymesh2-0.3-cp{py_version}-cp{py_version}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    wheel_url = f"{wheel_base}/{wheel_name}"

    print(f"[CADabra] Python version: {sys.version_info.major}.{sys.version_info.minor}")
    print(f"[CADabra] Downloading: {wheel_name}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", wheel_url],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            print("[CADabra] ✓ PyMesh installed successfully")
            return True
        else:
            print(f"[CADabra] ✗ PyMesh installation failed:")
            print(result.stderr[:500])
            return False

    except subprocess.TimeoutExpired:
        print("[CADabra] ✗ PyMesh installation timed out")
        return False
    except Exception as e:
        print(f"[CADabra] ✗ PyMesh installation error: {e}")
        return False


def install_system_deps():
    """Install system dependencies for Gmsh (Linux only)"""
    if platform.system() != "Linux":
        return True

    print("[CADabra] Checking system dependencies...")

    missing_libs = []
    try:
        result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
        if "libGLU.so" not in result.stdout:
            missing_libs.append("libglu1-mesa")
        if "libXft.so" not in result.stdout:
            missing_libs.append("libxft2")
    except FileNotFoundError:
        missing_libs = ["libglu1-mesa", "libxft2"]

    if missing_libs:
        print(f"[CADabra] Missing system libraries: {', '.join(missing_libs)}")

        # Try to install automatically with sudo
        if shutil.which("sudo") and shutil.which("apt-get"):
            print("[CADabra] Attempting automatic installation...")
            try:
                cmd = ["sudo", "apt-get", "install", "-y", "--fix-missing"] + missing_libs
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    print(f"[CADabra] ✓ Installed system libraries: {', '.join(missing_libs)}")
                    return True
                else:
                    print(f"[CADabra] Automatic installation failed")
            except (subprocess.TimeoutExpired, Exception) as e:
                print(f"[CADabra] Automatic installation failed: {e}")

        print(f"[CADabra] Run manually: sudo apt-get install -y {' '.join(missing_libs)}")
        return False

    print("[CADabra] ✓ System dependencies OK")
    return True


def install():
    """Install required dependencies"""
    print("[CADabra] Installing dependencies...")

    # Check conda availability first
    if not is_conda_available():
        print("\n[CADabra] ✗ ERROR: Conda is required but not found in PATH")
        print("[CADabra] Please install Anaconda or Miniconda and activate a conda environment")
        print("[CADabra] https://docs.conda.io/en/latest/miniconda.html")
        sys.exit(1)

    # Install system dependencies (Linux only)
    install_system_deps()

    # Install pip packages
    script_dir = os.path.dirname(__file__)
    pip_requirements = os.path.join(script_dir, "requirements-pip.txt")
    if not os.path.exists(pip_requirements):
        pip_requirements = os.path.join(script_dir, "requirements.txt")

    try:
        print(f"\n[CADabra] Installing Python packages from {os.path.basename(pip_requirements)}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", pip_requirements])
        print("[CADabra] ✓ Python packages installed")
    except subprocess.CalledProcessError as e:
        print(f"[CADabra] ✗ Python package installation failed: {e}")

    # Install conda packages (pythonocc-core + cadquery + occt-rt)
    if install_conda_packages():
        print("[CADabra] ✓ CAD packages installed successfully!")
    else:
        print("[CADabra] ⚠ CAD package installation failed")
        print("[CADabra] Manual installation:")
        print("[CADabra]   conda install -c pozzettiandrea -c cadquery -c conda-forge pythonocc-core=7.8.1 cadquery=2.6.1 occt-rt")

    # Install PyMesh for Point2CAD mesh clipping
    if install_pymesh():
        print("[CADabra] ✓ PyMesh installed for Point2CAD")
    else:
        print("[CADabra] ⚠ PyMesh installation failed (Point2CAD will use fallback)")

    print("\n[CADabra] Installation complete!")
    return True


if __name__ == "__main__":
    install()
