# scripts/doc_to_images.py
# CLI로 여러 폴더를 지정 가능. 예)
# python -m scripts.doc_to_images --in academic_paper\Technology academic_paper\Biology ^
#   --out academic_data\paper_001\images --dpi 160 --csv academic_data\paper_001\labels.csv --max-pages 3

from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트 import 경로 추가
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from app.services.docconvert import convert_docs_to_images  # uvicorn app.main:app 기준

def main():
    ap = argparse.ArgumentParser(description="PDF/PPT(X) → 이미지 변환 + 라벨 CSV 생성")
    ap.add_argument("--in", dest="in_roots", nargs="+", required=True,
                    help="변환할 루트 폴더(여러 개 지정 가능)")
    ap.add_argument("--out", dest="out_root", required=True,
                    help="이미지 저장 루트(라벨 폴더 자동 생성)")
    ap.add_argument("--dpi", type=int, default=200, help="PDF 렌더링 DPI (권장 144~300)")
    ap.add_argument("--csv", dest="csv_out", default=None, help="최종 (path,label) CSV 출력 위치")
    ap.add_argument("--max-pages", dest="max_pages", type=int, default=None,
                    help="문서당 최대 페이지 수 제한 (예: 3)")
    ap.add_argument("--label-from-csv", dest="label_from_csv", default=None,
                    help='CSV(file,label)로 라벨을 지정하고 싶을 때 사용 (선택)')
    args = ap.parse_args()

    all_rows: list[tuple[str, str]] = []
    out_root = args.out_root

    for in_root in args.in_roots:
        print(f"[INFO] Converting: {in_root} → {out_root}")
        rows, _ = convert_docs_to_images(
            in_root=in_root,
            out_root=out_root,
            dpi=args.dpi,
            csv_out=None,  # 폴더별 임시 CSV는 만들지 않고, 마지막에 한 번만 저장
            label_from_csv=args.label_from_csv,
            max_pages=args.max_pages,
        )
        all_rows.extend(rows)

    # 최종 CSV 저장
    if args.csv_out:
        import pandas as pd
        import os
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_rows, columns=["path", "label"])
        # 중복 제거(path 기준)
        df = df.drop_duplicates(subset=["path"])
        # 정렬(라벨, 경로)
        df = df.sort_values(["label", "path"])
        df.to_csv(args.csv_out, index=False, encoding="utf-8")
        print("CSV saved:", args.csv_out, "(rows:", len(df), ")")

    print("DONE. total images:", len(all_rows))

if __name__ == "__main__":
    main()