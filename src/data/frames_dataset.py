from pathlib import Path
from typing import List, Optional

from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_frames_from_folder(folder: Path, max_frames: Optional[int] = None) -> List[Image.Image]:
    """
    Loads frames (here, images) from a folder

    Args:
        folder: Path to directory containing frame images.
        max_frames: Optional limit on number of frames.

    Returns:
        List of PIL Images in RGB.
    """
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Frames folder not found: {folder}")

    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    files = sorted(files, key=lambda p: p.name)

    if max_frames is not None:
        files = files[: max_frames]

    frames: List[Image.Image] = []
    for p in files:
        img = Image.open(p).convert("RGB")
        frames.append(img)

    if len(frames) == 0:
        raise RuntimeError(f"No image frames found in {folder} (supported: {sorted(IMG_EXTS)})")

    return frames
