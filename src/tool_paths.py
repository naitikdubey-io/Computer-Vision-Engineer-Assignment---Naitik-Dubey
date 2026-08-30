"""
Resolve ffmpeg/tesseract executables without depending on the shell's PATH
being up to date (Windows terminals can cache a stale PATH across sessions).
Checks PATH first, then falls back to the known winget/installer locations.
"""
import shutil
from pathlib import Path

_FFMPEG_FALLBACKS = [
    r"C:\Users\dnait\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe",
]
_TESSERACT_FALLBACKS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
]


def _resolve(name, fallbacks):
    found = shutil.which(name)
    if found:
        return found
    for candidate in fallbacks:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find '{name}' on PATH or in known install locations: {fallbacks}"
    )


def get_ffmpeg():
    return _resolve("ffmpeg", _FFMPEG_FALLBACKS)


def get_tesseract():
    return _resolve("tesseract", _TESSERACT_FALLBACKS)
