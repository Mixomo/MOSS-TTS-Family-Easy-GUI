from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SYSTEM_DLL_PREFIXES = ("api-ms-win-", "ext-ms-win-")
SYSTEM_DLLS = {
    "kernel32.dll", "user32.dll", "advapi32.dll", "shell32.dll", "ole32.dll",
    "oleaut32.dll", "ws2_32.dll", "bcrypt.dll", "ntdll.dll", "gdi32.dll",
    "secur32.dll", "crypt32.dll", "shlwapi.dll", "combase.dll", "rpcrt4.dll",
}


def dependencies(path: Path) -> list[str]:
    cp = subprocess.run(
        ["dumpbin", "/DEPENDENTS", str(path)],
        text=True, encoding="utf-8", errors="replace", capture_output=True,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"dumpbin failed for {path}:\n{cp.stdout}\n{cp.stderr}")
    names: list[str] = []
    for line in cp.stdout.splitlines():
        s = line.strip()
        if re.fullmatch(r"[^\\/:*?\"<>|]+\.dll", s, re.IGNORECASE):
            names.append(s)
    return names


def is_system(name: str) -> bool:
    low = name.lower()
    return low in SYSTEM_DLLS or low.startswith(SYSTEM_DLL_PREFIXES)


def find_existing(name: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        if not d or not d.exists():
            continue
        p = d / name
        if p.exists():
            return p
        # Windows is case-insensitive, but pathlib on non-Windows test hosts is not.
        low = name.lower()
        try:
            for c in d.iterdir():
                if c.is_file() and c.name.lower() == low:
                    return c
        except OSError:
            pass
    return None


def find_existing_recursive(name: str, roots: list[Path]) -> Path | None:
    """Find an actual DLL anywhere below the supplied roots.

    CUDA Toolkit layouts vary by major version. CUDA 13 may place runtime
    libraries in a different subdirectory than older toolkits, so do not
    assume every dependency lives directly under CUDA_PATH/bin.
    """
    low = name.lower()
    for root in roots:
        if not root or not root.exists():
            continue
        try:
            for p in root.rglob("*"):
                if p.is_file() and p.name.lower() == low:
                    return p
        except OSError:
            continue
    return None


def copy_if_external(
    name: str,
    runtime: Path,
    search_dirs: list[Path],
    recursive_roots: list[Path] | None = None,
) -> Path | None:
    existing = find_existing(name, [runtime])
    if existing:
        return existing

    src = find_existing(name, search_dirs)
    if not src and recursive_roots:
        src = find_existing_recursive(name, recursive_roots)

    if not src:
        return None

    dst = runtime / src.name
    shutil.copy2(src, dst)
    print(f"[package] + {src.name} <- {src.parent}")
    return dst


def discover_vc_redist_dirs() -> list[Path]:
    dirs: list[Path] = []
    pf86 = os.environ.get("ProgramFiles(x86)")
    if not pf86:
        return dirs
    base = Path(pf86) / "Microsoft Visual Studio" / "2022"
    for edition in ("Community", "BuildTools", "Professional", "Enterprise"):
        redist = base / edition / "VC" / "Redist" / "MSVC"
        if redist.exists():
            versions = sorted((p for p in redist.iterdir() if p.is_dir()), reverse=True)
            for v in versions:
                for candidate in (
                    v / "x64" / "Microsoft.VC143.CRT",
                    v / "x64" / "Microsoft.VC142.CRT",
                ):
                    if candidate.exists():
                        dirs.append(candidate)
    return dirs


def preload_and_validate(runtime: Path, bridge: Path, cuda_bin: Path | None) -> None:
    handles = []
    if hasattr(os, "add_dll_directory"):
        handles.append(os.add_dll_directory(str(runtime)))
        if cuda_bin and cuda_bin.exists():
            handles.append(os.add_dll_directory(str(cuda_bin)))

    ordered = [
        "ggml-base.dll", "ggml.dll", "ggml-cpu.dll", "ggml-cuda.dll", "llama.dll"
    ]
    for name in ordered:
        p = runtime / name
        if not p.exists():
            if name in {"ggml-cpu.dll", "ggml-cuda.dll"}:
                continue
            raise RuntimeError(f"Required runtime DLL is missing: {p}")
        try:
            ctypes.WinDLL(str(p))
            print(f"[validate] OK  {name}")
        except OSError as exc:
            raise RuntimeError(f"Failed while loading {name}: {exc}") from exc

    try:
        ctypes.WinDLL(str(bridge))
        print(f"[validate] OK  {bridge.name}")
    except OSError as exc:
        raise RuntimeError(f"Failed while loading {bridge.name}: {exc}") from exc

    # Keep directory handles alive through the complete load sequence.
    _ = handles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-bin", type=Path, required=True)
    ap.add_argument("--runtime-bin", type=Path, required=True)
    ap.add_argument("--bridge", type=Path, required=True)
    ap.add_argument("--cuda-bin", type=Path)
    ap.add_argument("--cuda-root", type=Path)
    ns = ap.parse_args()

    build_bin = ns.build_bin.resolve()
    runtime = ns.runtime_bin.resolve()
    bridge = ns.bridge.resolve()
    cuda_bin = ns.cuda_bin.resolve() if ns.cuda_bin else None
    cuda_root = ns.cuda_root.resolve() if ns.cuda_root else None
    runtime.mkdir(parents=True, exist_ok=True)

    if not build_bin.exists():
        raise SystemExit(f"Build output directory does not exist: {build_bin}")
    if not bridge.exists():
        raise SystemExit(f"Bridge does not exist: {bridge}")

    # Repackage cleanly from the successfully built Release directory.
    for p in runtime.glob("*.dll"):
        p.unlink()
    for p in build_bin.glob("*.dll"):
        shutil.copy2(p, runtime / p.name)
        print(f"[package] llama.cpp: {p.name}")

    required = ["llama.dll", "ggml.dll", "ggml-base.dll"]
    missing = [n for n in required if not (runtime / n).exists()]
    if missing:
        raise SystemExit(f"Missing llama.cpp DLLs after packaging: {', '.join(missing)}")

    search_dirs: list[Path] = []
    if cuda_bin and cuda_bin.exists():
        search_dirs.append(cuda_bin)
    search_dirs.extend(discover_vc_redist_dirs())

    recursive_roots: list[Path] = []
    if cuda_root and cuda_root.exists():
        recursive_roots.append(cuda_root)
        print(f"[package] CUDA dependency search root: {cuda_root}")

    # Resolve external dependencies recursively. System/UCRT API-set DLLs are left to Windows.
    queue = [p for p in runtime.glob("*.dll")] + [bridge]
    seen: set[str] = set()
    unresolved: set[str] = set()
    while queue:
        dll = queue.pop(0)
        key = str(dll).lower()
        if key in seen:
            continue
        seen.add(key)
        for dep in dependencies(dll):
            if is_system(dep):
                continue
            local = find_existing(dep, [runtime])
            if local:
                if str(local).lower() not in seen:
                    queue.append(local)
                continue
            copied = copy_if_external(dep, runtime, search_dirs, recursive_roots)
            if copied:
                queue.append(copied)
            else:
                # It may be an OS/VC runtime already available through KnownDLLs/System32.
                system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
                if find_existing(dep, [system32]):
                    continue
                unresolved.add(dep)

    if unresolved:
        print("[package] Unresolved non-system dependencies:")
        for dep in sorted(unresolved):
            print(f"  - {dep}")
        raise SystemExit(2)

    print(f"[package] Runtime DLL count: {len(list(runtime.glob('*.dll')))}")
    preload_and_validate(runtime, bridge, cuda_bin)
    print("[package] Runtime dependency closure and staged load validation passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[package] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
