"""Environment record attached to every results file (spec Section 0, honesty rule 2)."""

import datetime
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = (
    "numpy", "onnx", "onnxruntime", "soundfile", "matplotlib", "torch", "torchaudio",
)  # fmt: skip


def cpu_model() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    if sys.platform == "darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        return out.stdout.strip()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


def collect_env(ort_threads: dict[str, int | None] | None = None) -> dict[str, Any]:
    """Date, git commit, CPU, cores, OS, Python/package versions, ORT threads, exact command.

    `ort_threads` is the intra/inter-op setting of the sessions the script created
    (None means ORT's default); scripts that create no session leave it unset.
    """
    versions: dict[str, str] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "cpu_model": cpu_model(),
        "logical_cores": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
        "ort_threads": ort_threads,
        "command": "python " + " ".join(sys.orig_argv[1:]),
    }


def write_results(path: Path, payload: dict[str, Any], **env_kwargs: Any) -> None:
    """Write a results JSON with an `env` block merged in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {**payload, "env": collect_env(**env_kwargs)}
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    write_results(REPO_ROOT / "results" / "env.json", {})
