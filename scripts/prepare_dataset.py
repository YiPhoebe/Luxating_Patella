#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(root: Path) -> List[Path]:
    # 이미지 파일 확장자 기준으로 하위 모든 경로에서 수집
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def infer_label_from_name(path: Path) -> int:
    # 파일 경로/이름에 포함된 패턴으로 라벨 추정: 'ABN' -> 1, 'NOR' -> 0
    s = str(path).lower()
    if "abn" in s:
        return 1
    if "nor" in s:
        return 0
    # 패턴이 없을 경우 기본값: 정상(0)
    return 0


def _find_label_json(
    labels_index: Optional[Dict[str, Path]], stem: str
) -> Optional[Path]:
    """라벨 인덱스에서 파일명(stem)으로 즉시 조회."""
    if not labels_index:
        return None
    return labels_index.get(stem)


def _extract_bbox_and_label(
    obj: Dict[str, Any]
) -> Tuple[Optional[List[float]], Optional[int]]:
    """여러 흔한 스키마에서 bbox/label 추출 시도.
    - 이 프로젝트 커스텀: metadata.Disease, annotations[].points(두 점) 또는 annotations[].bbox
    - 일반: bbox | box | rect | annotations[0].bbox | shapes[0].points(LabelMe)
    반환 형식: (bbox 또는 None, label 또는 None)
    """
    # 0) label 우선: metadata.Disease 사용
    label_from_meta: Optional[int] = None
    try:
        disease = str(obj.get("metadata", {}).get("Disease", "")).lower()
        if disease:
            if "nor" in disease:
                label_from_meta = 0
            elif "abn" in disease or "abnormal" in disease:
                label_from_meta = 1
    except Exception:
        pass
    # 직접 키 시도
    for k in ("bbox", "box", "rect"):
        if k in obj and isinstance(obj[k], (list, tuple)) and len(obj[k]) == 4:
            bbox = [float(v) for v in obj[k]]
            # 라벨 추정
            label = None
            for lk in ("label", "class", "category_id"):
                if lk in obj:
                    try:
                        label = int(obj[lk])
                    except Exception:
                        label = (
                            1
                            if str(obj[lk]).lower() in ("abn", "abnormal", "1", "true")
                            else 0
                        )
                    break
            return bbox, label
    # 1) 커스텀/COCO 유사: annotations 안에서 bbox 또는 points 처리
    if (
        "annotations" in obj
        and isinstance(obj["annotations"], list)
        and obj["annotations"]
    ):
        # 우선 bbox 필드가 있으면 사용
        ann0 = obj["annotations"][0]
        if isinstance(ann0, dict):
            if (
                "bbox" in ann0
                and isinstance(ann0["bbox"], (list, tuple))
                and len(ann0["bbox"]) == 4
            ):
                bbox = [float(v) for v in ann0["bbox"]]
                return bbox, label_from_meta
            # points 두 점(좌상/우하) → [x,y,w,h] 변환
            if (
                "points" in ann0
                and isinstance(ann0["points"], list)
                and len(ann0["points"]) >= 2
            ):
                (x1, y1), (x2, y2) = ann0["points"][0], ann0["points"][1]
                x_min, y_min = float(min(x1, x2)), float(min(y1, y2))
                w, h = float(abs(x2 - x1)), float(abs(y2 - y1))
                bbox = [x_min, y_min, w, h]
                return bbox, label_from_meta
    # LabelMe 유사
    if "shapes" in obj and isinstance(obj["shapes"], list) and obj["shapes"]:
        shp = obj["shapes"][0]
        if isinstance(shp, dict):
            # points → bbox로 변환(좌상/우하 가정)
            if (
                "points" in shp
                and isinstance(shp["points"], list)
                and len(shp["points"]) >= 2
            ):
                (x1, y1), (x2, y2) = shp["points"][0], shp["points"][1]
                bbox = [
                    float(min(x1, x2)),
                    float(min(y1, y2)),
                    float(abs(x2 - x1)),
                    float(abs(y2 - y1)),
                ]
                label = None
                if "label" in shp:
                    try:
                        label = int(shp["label"])  # 숫자 가능 시
                    except Exception:
                        label = (
                            1
                            if str(shp["label"]).lower()
                            in ("abn", "abnormal", "1", "true")
                            else 0
                        )
                return bbox, label
    return None, label_from_meta


def to_records(
    files: List[Path],
    image_root: Path,
    labels_index: Optional[Dict[str, Path]] = None,
    log_prefix: str = "",
) -> List[dict]:
    recs = []
    for i, f in enumerate(files, 1):
        try:
            label = infer_label_from_name(f)
            rel = f.resolve().relative_to(image_root.resolve())
            rec: Dict[str, Any] = {
                "image_path": str(rel).replace(os.sep, "/"),
                "label": int(label),
            }
            # 라벨 JSON에서 bbox/label 병합 시도
            if labels_index:
                lj = _find_label_json(labels_index, f.stem)
                if lj:
                    try:
                        with open(lj, "r", encoding="utf-8") as jf:
                            obj = json.load(jf)
                        bbox, lbl2 = _extract_bbox_and_label(obj)
                        if lbl2 is not None:
                            rec["label"] = int(lbl2)
                        if bbox is not None:
                            rec["bbox"] = [float(v) for v in bbox]
                    except Exception:
                        pass
            recs.append(rec)
        except Exception:
            # 루트 밖이거나 접근 불가한 파일은 건너뜀
            continue
        if i % 10000 == 0:
            print(f"{log_prefix} 진행 상황: {i}개 처리", flush=True)
    return recs


def split_records(recs: List[dict], split: float) -> Tuple[List[dict], List[dict]]:
    # 무작위 셔플 후 비율에 따라 학습/검증 분할
    import random

    random.shuffle(recs)
    n_train = int(len(recs) * split)
    return recs[:n_train], recs[n_train:]


def main():
    ap = argparse.ArgumentParser(
        description="무릎 X-ray 데이터셋을 위한 학습/검증 JSON 생성 스크립트"
    )
    ap.add_argument(
        "--image-root", type=str, required=True, help="이미지 루트 디렉토리"
    )
    ap.add_argument("--out-train", type=str, required=True, help="train.json 출력 경로")
    ap.add_argument("--out-val", type=str, required=True, help="val.json 출력 경로")
    ap.add_argument(
        "--train-dir", type=str, default=None, help="(선택) 학습 이미지가 있는 디렉토리"
    )
    ap.add_argument(
        "--val-dir", type=str, default=None, help="(선택) 검증 이미지가 있는 디렉토리"
    )
    ap.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="(선택) 이미지와 매칭되는 라벨 JSON들이 있는 디렉토리(dog_id별 JSON)",
    )
    ap.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="(선택) 학습용 최대 처리 이미지 수(디버그용)",
    )
    ap.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="(선택) 검증용 최대 처리 이미지 수(디버그용)",
    )
    ap.add_argument(
        "--split", type=float, default=0.9, help="단일 루트 사용 시 학습 데이터 비율"
    )
    args = ap.parse_args()

    image_root = Path(args.image_root)
    image_root.mkdir(parents=True, exist_ok=True)

    labels_dir = Path(args.labels_dir) if args.labels_dir else None

    # 라벨 인덱스 생성(한 번만 스캔)
    labels_index: Optional[Dict[str, Path]] = None
    if labels_dir and labels_dir.exists():
        print("라벨 인덱스를 구축 중입니다...", flush=True)
        labels_index = {}
        cnt = 0
        for p in labels_dir.rglob("*.json"):
            labels_index[p.stem] = p
            cnt += 1
            if cnt % 50000 == 0:
                print(f"라벨 인덱스: {cnt}개 매핑", flush=True)
        print(f"라벨 인덱스 구축 완료: 총 {cnt}개", flush=True)

    if args.train_dir or args.val_dir:
        if not (args.train_dir and args.val_dir):
            raise SystemExit(
                "명시적 디렉토리를 사용할 경우 --train-dir 와 --val-dir 를 모두 지정해야 합니다"
            )
        train_files = collect_images(Path(args.train_dir))
        val_files = collect_images(Path(args.val_dir))
        if args.max_train:
            train_files = train_files[: args.max_train]
        if args.max_val:
            val_files = val_files[: args.max_val]
        print(
            f"학습 이미지: {len(train_files)}개, 검증 이미지: {len(val_files)}개",
            flush=True,
        )
        train_recs = to_records(
            train_files, image_root, labels_index, log_prefix="[train]"
        )
        val_recs = to_records(val_files, image_root, labels_index, log_prefix="[val]")
    else:
        files = collect_images(image_root)
        if args.max_train or args.max_val:
            print(
                "경고: 단일 루트 분할 모드에서 --max-train/--max-val은 무시됩니다",
                flush=True,
            )
        all_recs = to_records(files, image_root, labels_index)
        train_recs, val_recs = split_records(all_recs, args.split)

    out_train_path = Path(args.out_train)
    out_val_path = Path(args.out_val)
    out_train_path.parent.mkdir(parents=True, exist_ok=True)
    out_val_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_train_path, "w", encoding="utf-8") as f:
        json.dump(train_recs, f, ensure_ascii=False)
    with open(out_val_path, "w", encoding="utf-8") as f:
        json.dump(val_recs, f, ensure_ascii=False)

    print(f"train: {len(train_recs)}개, val: {len(val_recs)}개 레코드를 생성했습니다.")


if __name__ == "__main__":
    main()
