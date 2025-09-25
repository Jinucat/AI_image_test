# app/services/docconvert.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import re
import shutil
import csv
import hashlib

import fitz  # PyMuPDF
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# 유틸: 안전한 파일명 / 안전 저장
# ──────────────────────────────────────────────────────────────────────────────

_INVALID = r'[<>:"/\\|?*]'

def _safe_name(stem: str, limit: int = 60) -> str:
    """Windows 불가 문자 제거 + 너무 길면 해시 덧붙여 자르기."""
    s = re.sub(_INVALID, "_", stem)
    if len(s) <= limit:
        return s
    h = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]
    return s[:limit] + "_" + h

def _save_pixmap_safe(pix: fitz.Pixmap, out_path: Path) -> None:
    """
    PyMuPDF의 pix.save()가 실패할 때를 대비해 Pillow로 우회 저장.
    너무 큰 이미지는 반으로 다운스케일.
    """
    try:
        # CMYK 등은 RGB로 변환
        if pix.n > 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        pix.save(str(out_path))
        return
    except Exception:
        pass

    has_alpha = getattr(pix, "alpha", False) is True
    mode = "RGBA" if has_alpha else ("RGB" if pix.n >= 3 else "L")
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if has_alpha:
        img = img.convert("RGB")  # PNG RGB로 저장
    # 초대형 이미지 안전장치
    if pix.width * pix.height > 12_000_000:
        img = img.resize((pix.width // 2, pix.height // 2), Image.BILINEAR)
    img.save(out_path, format="PNG")

# ──────────────────────────────────────────────────────────────────────────────
# 변환기: PDF / PPT(X)
# ──────────────────────────────────────────────────────────────────────────────

def pdf_to_images(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 200,
    max_pages: Optional[int] = None,
) -> list[Path]:
    """
    PDF → 페이지 PNG 일괄 변환(문제 페이지는 낮은 DPI로 재시도 후 스킵).
    반환: 저장된 이미지 경로 리스트
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[WARN] 열기 실패: {pdf_path} → {e}")
        return saved

    total = len(doc)
    n_pages = total if max_pages is None else min(total, max_pages)
    base = _safe_name(pdf_path.stem)
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    for i in range(n_pages):
        try:
            pix = doc[i].get_pixmap(matrix=mat, alpha=False)
        except Exception:
            # Fallback 1: 낮은 DPI + alpha=True
            try:
                pix = doc[i].get_pixmap(dpi=min(dpi, 144), alpha=True)
            except Exception:
                # Fallback 2: 더 낮은 DPI
                try:
                    pix = doc[i].get_pixmap(dpi=120, alpha=True)
                except Exception as e3:
                    print(f"[WARN] 페이지 스킵: {pdf_path.name} p{i+1} → {e3}")
                    continue

        out = out_dir / f"{base}_p{i+1:03}.png"
        try:
            _save_pixmap_safe(pix, out)
            saved.append(out)
        except Exception as e4:
            print(f"[WARN] 저장 실패 스킵: {out.name} → {e4}")
            continue

    doc.close()
    return saved


def ppt_to_images_via_powerpoint(ppt_path: Path, out_dir: Path) -> list[Path]:
    """
    Windows + PowerPoint 설치 시: PPT/PPTX → 슬라이드 PNG 일괄 내보내기.
    PowerPoint / pywin32가 없으면 예외 발생.
    """
    try:
        import win32com.client  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PowerPoint/pywin32가 없어 PPT 내보내기 불가. PPTX→PDF 후 진행하세요."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(ppt_path.resolve()), WithWindow=False)

    tmp = out_dir / f"__export_{_safe_name(ppt_path.stem)}"
    tmp.mkdir(parents=True, exist_ok=True)
    pres.Export(str(tmp.resolve()), "PNG")
    pres.Close()
    app.Quit()

    saved: list[Path] = []
    for p in sorted(tmp.glob("Slide*.PNG")):
        m = re.search(r"Slide(\d+)\.PNG$", p.name)
        idx = int(m.group(1)) if m else 0
        newp = out_dir / f"{_safe_name(ppt_path.stem)}_s{idx:03}.png"
        shutil.move(str(p), str(newp))
        saved.append(newp)

    shutil.rmtree(tmp, ignore_errors=True)
    return saved

# ──────────────────────────────────────────────────────────────────────────────
# 상위 API
# ──────────────────────────────────────────────────────────────────────────────

def _walk_docs(root: Path) -> Iterable[Path]:
    for pat in ("*.pdf", "*.PDF", "*.ppt", "*.PPT", "*.pptx", "*.PPTX"):
        yield from root.rglob(pat)

def convert_docs_to_images(
    in_root: str | Path,
    out_root: str | Path,
    dpi: int = 200,
    csv_out: str | Path | None = None,
    label_from_csv: str | Path | None = None,
    max_pages: Optional[int] = None,
) -> tuple[list[tuple[str, str]], Path]:
    """
    문서(PDF/PPT/PPTX) → PNG로 변환하고, (path,label) CSV를 선택 저장.
    - 라벨 기본: 원본 파일의 상위 폴더명
    - label_from_csv 제공 시: CSV(file,label)에서 파일명(stem) 기준으로 라벨 매핑
    반환: ([(relative_path, label)], out_root Path)
    """
    in_root = Path(in_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    label_map: dict[str, str] = {}
    if label_from_csv:
        try:
            with open(label_from_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label_map[Path(row["file"]).stem] = row["label"]
        except Exception as e:
            print(f"[WARN] label_from_csv 읽기 실패: {label_from_csv} → {e}")

    rows: list[tuple[str, str]] = []

    for doc in _walk_docs(in_root):
        label = label_map.get(doc.stem, doc.parent.name)
        label_dir = out_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        try:
            if doc.suffix.lower() == ".pdf":
                outs = pdf_to_images(doc, label_dir, dpi=dpi, max_pages=max_pages)
            else:
                outs = ppt_to_images_via_powerpoint(doc, label_dir)
        except Exception as e:
            print(f"[WARN] 변환 실패: {doc} → {e}")
            continue

        for img in outs:
            rel = img.relative_to(out_root).as_posix()
            rows.append((rel, label))

    if csv_out:
        csv_path = Path(csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["path", "label"])
                w.writerows(rows)
            print("CSV saved:", csv_path)
        except Exception as e:
            print(f"[WARN] CSV 저장 실패: {csv_path} → {e}")

    print("DONE. images:", len(rows), "at", out_root.resolve())
    return rows, out_root
