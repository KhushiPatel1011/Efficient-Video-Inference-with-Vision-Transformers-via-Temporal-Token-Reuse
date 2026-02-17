from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_sorted_frames(folder: Path, max_frames: Optional[int] = None) -> List[Path]:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Frames folder not found: {folder}")

    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    files = sorted(files, key=lambda p: p.name)

    if max_frames is not None:
        files = files[:max_frames]

    if len(files) < 2:
        raise RuntimeError(f"Need at least 2 frames in {folder}, found {len(files)}")

    return files


def iter_frame_pairs(
    folder: Path,
    max_frames: Optional[int] = None,
) -> Iterator[Tuple[int, Image.Image, Image.Image]]:
    """
    Yields (t, prev_img, curr_img) where:
        prev_img = frame_{t-1}
        curr_img = frame_{t}

    Args:
        folder: directory of frames in sorted name order
        max_frames: optional limit on number of frames loaded

    Yields:
        (t_index, prev_pil_rgb, curr_pil_rgb)
    """
    files = _list_sorted_frames(folder, max_frames=max_frames)

    prev = Image.open(files[0]).convert("RGB")
    for t in range(1, len(files)):
        curr = Image.open(files[t]).convert("RGB")
        yield t, prev, curr
        prev = curr
