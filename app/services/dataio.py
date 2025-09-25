from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def load_from_dirs(root: str | Path,
                   max_per_class: Optional[int] = None,
                   seed: int = 42) -> List[Tuple[str, str]]:
    root_p = Path(root)
    samples: List[Tuple[str, str]] = []
    if not root_p.exists():
        return samples

    label_dirs = [d for d in sorted(root_p.iterdir()) if d.is_dir()]
    rnd = random.Random(seed)

    for label_dir in label_dirs:
        imgs = [p for p in label_dir.rglob("*") if _is_image(p)]
        if not imgs:
            continue
        imgs = sorted(imgs)
        if max_per_class is not None and max_per_class > 0 and len(imgs) > max_per_class:
            rnd.shuffle(imgs)
            imgs = sorted(imgs[:max_per_class])
        for p in imgs:
            samples.append((str(p.resolve()), label_dir.name))
    return samples


def load_from_table(root: Optional[str | Path],
                    table_path: str | Path) -> List[Tuple[str, str]]:
    tp = Path(table_path)
    if not tp.exists():
        return []

    try:
        df = pd.read_csv(tp, sep=None, engine="python") 
    except Exception:
        df = pd.read_csv(tp)

    cols = {c.lower(): c for c in df.columns}
    if "path" not in cols or "label" not in cols:
        raise ValueError("CSV/TSV에는 'path','label' 컬럼이 있어야 합니다.")

    base_dir = tp.parent
    root_p = Path(root) if root else None
    out: List[Tuple[str, str]] = []

    for _, row in df.iterrows():
        lab = str(row[cols["label"]])
        raw = str(row[cols["path"]])
        rp = Path(raw)
        if rp.is_absolute():
            p = rp
        else:
            base = root_p if root_p is not None else base_dir
            p = (base / rp)
        p = p.resolve()
        if _is_image(p):
            out.append((str(p), lab))
    return out


def _sample_per_class(items: List[Tuple[str, str]],
                      max_per_class: Optional[int],
                      seed: int = 42) -> List[Tuple[str, str]]:
    if max_per_class is None or max_per_class <= 0:
        return sorted(items, key=lambda x: (x[1], x[0]))

    by: Dict[str, List[str]] = {}
    for p, lab in items:
        by.setdefault(lab, []).append(p)

    rnd = random.Random(seed)
    sampled: List[Tuple[str, str]] = []
    for lab, paths in by.items():
        paths = sorted(paths)
        if len(paths) > max_per_class:
            rnd.shuffle(paths)
            paths = sorted(paths[:max_per_class])
        sampled.extend((p, lab) for p in paths)

    return sorted(sampled, key=lambda x: (x[1], x[0]))


def load_image_items(data_root: Optional[str] = None,
                     table_path: Optional[str] = None,
                     max_per_class: Optional[int] = None,
                     seed: int = 42) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []

    if data_root:
        items.extend(load_from_dirs(data_root, max_per_class=None, seed=seed))

    if table_path:
        items.extend(load_from_table(data_root, table_path))

    seen = set()
    uniq: List[Tuple[str, str]] = []
    for p, lab in items:
        if p not in seen:
            seen.add(p)
            uniq.append((p, lab))

    return _sample_per_class(uniq, max_per_class=max_per_class, seed=seed)