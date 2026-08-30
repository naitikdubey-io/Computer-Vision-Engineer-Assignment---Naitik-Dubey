"""
Resolve ffmpeg/tesseract executables without depending on the shell's PATH
being up to date (a common issue right after installing on Windows, since
the current shell's PATH doesn't refresh until it's restarted). Checks PATH
first, then a handful of common install locations by pattern - using
environment variables and glob wildcards, never a hardcoded username or
version number, so this works on any machine rather than just the one it
was written on.
"""
import glob
import os
import shutil
from pathlib import Path

_LOCALAPPDATA = os.environ.get("LOCALAPPDATA", "")
_PROGRAMFILES = os.environ.get("ProgramFiles", r"C:\Program Files")
_PROGRAMFILES_X86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")

# glob patterns, not literal paths - covers the common install methods
# (winget, the official installers, chocolatey) without pinning a version.
_FFMPEG_PATTERNS = [
    rf"{_LOCALAPPDATA}\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin\ffmpeg.exe",
    rf"{_PROGRAMFILES}\ffmpeg\bin\ffmpeg.exe",
    r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    "/usr/local/bin/ffmpeg",
    "/opt/homebrew/bin/ffmpeg",
    "/usr/bin/ffmpeg",
]
_TESSERACT_PATTERNS = [
    rf"{_PROGRAMFILES}\Tesseract-OCR\tesseract.exe",
    rf"{_PROGRAMFILES_X86}\Tesseract-OCR\tesseract.exe",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
    "/usr/bin/tesseract",
]

_INSTALL_HINT = {
    "ffmpeg": "https://ffmpeg.org/download.html (or `winget install Gyan.FFmpeg` on Windows)",
    "tesseract": "https://github.com/UB-Mannheim/tesseract/wiki",
}


def _resolve(name, patterns):
    found = shutil.which(name)
    if found:
        return found
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        if Path(pattern).exists():  # patterns with no wildcard
            return pattern
    raise FileNotFoundError(
        f"Could not find '{name}' on PATH or in common install locations.\n"
        f"Install it from {_INSTALL_HINT[name]}, then make sure it's on PATH "
        f"(open a *new* terminal after installing - PATH doesn't refresh in "
        f"one that's already open)."
    )


def get_ffmpeg():
    return _resolve("ffmpeg", _FFMPEG_PATTERNS)


def get_tesseract():
    return _resolve("tesseract", _TESSERACT_PATTERNS)
