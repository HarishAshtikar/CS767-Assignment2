"""Sandboxed Python execution for model-requested analysis steps."""

import io
import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ALLOWED_IMPORTS = {
    "math",
    "statistics",
    "numpy",
    "np",
    "pandas",
    "pd",
    "matplotlib",
    "matplotlib.pyplot",
}


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Permit imports commonly needed for data analysis and block everything else."""
    root_name = name.split(".")[0]
    if name in ALLOWED_IMPORTS or root_name in ALLOWED_IMPORTS:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import '{name}' is not allowed in the analysis sandbox")


def execute_code(code: str, df: pd.DataFrame, outputs_dir: str) -> dict:
    """Execute agent-generated Python code in a restricted namespace."""
    os.makedirs(outputs_dir, exist_ok=True)

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    namespace = {
        "pd": pd,
        "plt": plt,
        "df": df.copy(),
        "outputs": outputs_dir,
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "list": list,
            "dict": dict,
            "str": str,
            "int": int,
            "float": float,
            "round": round,
            "sum": sum,
            "min": min,
            "max": max,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "isinstance": isinstance,
            "type": type,
            "abs": abs,
            "__import__": safe_import,
        },
    }

    original_savefig = plt.savefig
    saved_files = []

    def patched_savefig(fname, **kwargs):
        if not os.path.isabs(fname):
            fname = os.path.join(outputs_dir, os.path.basename(fname))
        saved_files.append(fname)
        original_savefig(fname, **kwargs)
        plt.close()

    namespace["plt"].savefig = patched_savefig

    error = None
    try:
        exec(code, namespace)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        namespace["plt"].savefig = original_savefig

    return {
        "stdout": output,
        "error": error,
        "saved_files": saved_files,
        "success": error is None,
    }
