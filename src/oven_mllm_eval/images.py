"""Image loading helpers shared by inference and the multimodal judge.

Factored out of ``scripts/run_inference.py`` so the judge (Phase 2) can load
OVEN images the same way Phase 1 does, without duplicating the logic.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from oven_mllm_eval.paths import IMAGE_EXTENSIONS

__all__ = [
    "load_pil",
    "resolve_image_path",
    "load_images",
    "resize_to_pixels",
]


def load_pil(path: str) -> Image.Image:
    """Load an image as RGB PIL, raising on missing or unreadable files."""
    if not path:
        raise ValueError("Empty image_path in example")
    p = Path(path)
    if not p.exists():
        for ext in IMAGE_EXTENSIONS:
            alt = p.with_suffix(ext)
            if alt.exists():
                p = alt
                break
        else:
            raise FileNotFoundError(f"Image not found: {p.resolve()} (cwd={Path.cwd()})")
    img = Image.open(p)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def resolve_image_path(path: str, root: Path) -> str:
    """Resolve a (possibly relative) image path and verify it exists.

    Cheap preflight (stat only, no decode) so a missing file fails the run
    immediately instead of at chunk N.  Returns the resolved path string.
    """
    if not path:
        raise ValueError("Empty image_path in example")
    p = Path(path) if Path(path).is_absolute() else root / path
    if not p.exists():
        for ext in IMAGE_EXTENSIONS:
            alt = p.with_suffix(ext)
            if alt.exists():
                return str(alt)
        raise FileNotFoundError(f"Image not found: {p.resolve()} (cwd={Path.cwd()})")
    return str(p)


def load_images(paths: list[str], max_workers: int = 16) -> list[Image.Image]:
    """Decode a batch of images in parallel.

    Called per chunk (NOT upfront for the whole dataset): PIL pixel buffers
    for 60k+ images would otherwise accumulate in host RAM as vLLM touches
    them, growing RSS monotonically until the SLURM cgroup OOM-kills the
    engine core process.  Loading per chunk keeps the working set to one
    chunk and lets the GC reclaim it after each llm.chat() call.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(load_pil, paths))


def resize_to_pixels(img: Image.Image, max_pixels: int, min_pixels: int) -> Image.Image:
    """Aspect-preserving resize so the longer side fits the pixel budget.

    Model-agnostic bound used by the multimodal judge: unlike Qwen's own
    dynamic tiling (passed via ``mm_processor_kwargs``), this is a plain
    thumbnail so it works for any VLM family (e.g. Gemma) without knowing
    that family's processor kwargs.  ``max_pixels``/``min_pixels`` are total
    pixel budgets (e.g. 512*512); images already within budget are returned
    unchanged.
    """
    w, h = img.size
    pixels = w * h
    if pixels <= max_pixels and pixels >= min_pixels:
        return img
    # Scale so total pixels ≈ target (max if over, min if under).
    target = max_pixels if pixels > max_pixels else min_pixels
    scale = (target / pixels) ** 0.5
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    # LANCZOS for downscale; the image is small enough that cost is negligible.
    return img.resize(new_size, Image.LANCZOS)
