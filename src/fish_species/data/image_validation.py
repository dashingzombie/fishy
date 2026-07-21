from __future__ import annotations

from pathlib import Path

from PIL import Image


def resolve_path(root_dir: str | Path, path_value) -> Path:
    """Resolve a metadata path exactly as the legacy datasets do."""
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return Path(root_dir) / path


def is_valid_image(path: str | Path) -> bool:
    """Return whether Pillow can verify an image at ``path``."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    # Preserve the legacy metadata auditor: any decoder, filesystem, or Pillow
    # failure marks the row invalid without aborting metadata preparation.
    except Exception:
        return False
