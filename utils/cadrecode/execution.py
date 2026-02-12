"""
Safe CadQuery Code Execution
Runs generated CadQuery code in an isolated subprocess with timeout.

This isolation is important because:
1. Generated code may be invalid and crash
2. CadQuery can have memory leaks: https://github.com/CadQuery/cadquery/issues/1665
3. Complex models can take excessive time
"""

import subprocess
import tempfile
import sys
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


def execute_cadquery_code(
    code: str,
    timeout: float = 5.0,
    output_dir: Optional[str] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Safely execute CadQuery code in a subprocess with timeout.

    The generated code is expected to assign the final result to a variable 'r'.
    For example: r = w0.workplane().box(10, 10, 10)

    Args:
        code: CadQuery Python code string
        timeout: Maximum execution time in seconds
        output_dir: Directory to save output files (temp dir if None)

    Returns:
        Tuple of:
            - success: bool, whether execution succeeded
            - result: str, path to STEP file on success, error message on failure
            - stl_path: Optional[str], path to STL file if generated
    """
    # Use temp dir if no output specified
    use_temp = output_dir is None
    if use_temp:
        temp_dir = tempfile.mkdtemp(prefix='cadrecode_')
        output_dir = temp_dir
    else:
        temp_dir = None
        os.makedirs(output_dir, exist_ok=True)

    step_path = os.path.join(output_dir, 'output.step')
    stl_path = os.path.join(output_dir, 'output.stl')
    script_path = os.path.join(output_dir, 'script.py')

    # Build the execution script
    # The generated code should define 'r' as the result
    script = f'''
import sys
import cadquery as cq

try:
    # Execute the generated code
{_indent_code(code, 4)}

    # Get the result - expected to be stored in 'r'
    if 'r' not in dir():
        print("Error: Generated code did not define 'r' variable", file=sys.stderr)
        sys.exit(1)

    result = r
    if hasattr(result, 'val'):
        compound = result.val()
    else:
        compound = result

    # Export to STEP
    cq.exporters.export(compound, '{step_path}')

    # Also export to STL for preview
    try:
        vertices, faces = compound.tessellate(0.001, 0.1)
        import trimesh
        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
        mesh.export('{stl_path}')
    except Exception as e:
        print(f"STL export failed (non-fatal): {{e}}", file=sys.stderr)

    print("SUCCESS")

except Exception as e:
    import traceback
    print(f"Execution error: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''

    try:
        # Write script
        with open(script_path, 'w') as f:
            f.write(script)

        # Execute in subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            timeout=timeout,
            capture_output=True,
            text=True,
            cwd=output_dir,
        )

        if result.returncode == 0 and os.path.exists(step_path):
            stl_result = stl_path if os.path.exists(stl_path) else None
            return True, step_path, stl_result
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return False, error_msg, None

    except subprocess.TimeoutExpired:
        return False, f"Execution timed out after {timeout}s", None

    except Exception as e:
        return False, f"Subprocess error: {str(e)}", None

    finally:
        # Clean up temp dir if we created one and execution failed
        if use_temp and temp_dir and not os.path.exists(step_path):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass


def execute_cadquery_code_persistent(
    code: str,
    output_path: str,
    timeout: float = 5.0,
) -> Tuple[bool, str]:
    """
    Execute CadQuery code and save to a specific output path.

    Unlike execute_cadquery_code, this keeps the output files permanently.

    Args:
        code: CadQuery Python code string
        output_path: Path for the output STEP file
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (success, error_message or output_path)
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    success, result, _ = execute_cadquery_code(code, timeout, output_dir)

    if success:
        # Move to requested path if different
        temp_step = result
        if temp_step != output_path:
            shutil.move(temp_step, output_path)
        return True, output_path
    else:
        return False, result


def validate_cadquery_code(code: str) -> Tuple[bool, Optional[str]]:
    """
    Basic validation of generated CadQuery code.

    Checks for common issues without executing the code.

    Args:
        code: CadQuery Python code string

    Returns:
        Tuple of (is_valid, error_message if invalid)
    """
    if not code or not code.strip():
        return False, "Empty code"

    # Check for required import
    if 'cadquery' not in code and 'cq' not in code:
        return False, "Code does not appear to use CadQuery"

    # Check for result variable
    if '=' not in code:
        return False, "Code does not contain any assignments"

    # Try to compile (syntax check only)
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    return True, None


def _indent_code(code: str, spaces: int) -> str:
    """Indent all lines of code by the specified number of spaces."""
    indent = ' ' * spaces
    lines = code.split('\n')
    return '\n'.join(indent + line for line in lines)


def cleanup_temp_files(max_age_hours: int = 24):
    """
    Clean up old temporary CAD-Recode files.

    Args:
        max_age_hours: Delete files older than this many hours
    """
    import time

    temp_base = tempfile.gettempdir()
    now = time.time()
    max_age_seconds = max_age_hours * 3600

    for item in os.listdir(temp_base):
        if item.startswith('cadrecode_'):
            path = os.path.join(temp_base, item)
            try:
                if os.path.isdir(path):
                    mtime = os.path.getmtime(path)
                    if now - mtime > max_age_seconds:
                        shutil.rmtree(path)
            except:
                pass
