from typing import Dict, Optional, Tuple
import numpy as np
import cv2


def blur_score(gray_u8: np.ndarray) -> float:
    # 라플라시안 분산(초점/블러 지표)을 0~1 범위로 정규화
    v = cv2.Laplacian(gray_u8, cv2.CV_64F).var()
    # 휴리스틱 스케일링: 0..300 → 0..1 근사 매핑
    return float(np.clip(v / 300.0, 0.0, 1.0))


def exposure_score(gray_u8: np.ndarray) -> float:
    # 과다/과소 노출(히스토그램 양끝) 패널티 + 평균 밝도 중간값 선호
    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).flatten()
    total = hist.sum() + 1e-6
    pct_dark = hist[:5].sum() / total
    pct_bright = hist[-5:].sum() / total
    mean = gray_u8.mean() / 255.0
    penalty = pct_dark + pct_bright  # 0..1
    center_score = 1.0 - abs(mean - 0.5) * 2.0  # 이상적인 평균 ~0.5 근처
    score = np.clip(center_score * (1.0 - penalty), 0.0, 1.0)
    return float(score)


def framing_score(gray_u8: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> float:
    h, w = gray_u8.shape[:2]
    if bbox is None:
        return 0.5  # 알 수 없음(중간값 가정)
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    area_ratio = ((x2 - x1) * (y2 - y1) + 1e-6) / (w * h + 1e-6)
    # 박스가 프레임의 10%~60%를 적절히 포함하도록 유도(너무 작거나 크면 패널티)
    if area_ratio <= 0:
        return 0.0
    mid = 0.35
    score_area = np.exp(-((area_ratio - mid) ** 2) / (2 * (0.2 ** 2)))  # 0..1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    dx = abs(cx - w / 2) / (w / 2)
    dy = abs(cy - h / 2) / (h / 2)
    center_dist = np.hypot(dx, dy)  # 0..~1.4
    score_center = np.clip(1.0 - center_dist, 0.0, 1.0)  # 중심에 가까울수록 점수↑
    return float(0.6 * score_area + 0.4 * score_center)


def quality_scores(gray_u8: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
    return {
        "blur": blur_score(gray_u8),
        "exposure": exposure_score(gray_u8),
        "framing": framing_score(gray_u8, bbox),
    }


def quality_gate(gray_u8: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None,
                 thresholds: Optional[Dict[str, float]] = None) -> Dict:
    if thresholds is None:
        thresholds = {"blur": 0.25, "exposure": 0.4, "framing": 0.4}
    scores = quality_scores(gray_u8, bbox)
    passed = all(scores[k] >= thresholds.get(k, 0.0) for k in scores)
    # 가중 합산 품질 점수 요약
    quality = float(0.5 * scores["blur"] + 0.3 * scores["exposure"] + 0.2 * scores["framing"])
    return {"scores": scores, "quality": quality, "passed": passed}
