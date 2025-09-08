from typing import Dict


def decide_message(
    prob_pos: float,
    sigma_pos: float,
    quality: float,
    low_th: float = 0.2,
    high_th: float = 0.8,
    sigma_th: float = 0.06,
    quality_th: float = 0.5,
) -> Dict:
    ood_flag = quality < quality_th
    if ood_flag or (low_th < prob_pos < high_th) or (sigma_pos > sigma_th):
        return {
            "label": "uncertain",
            "message": "촬영 품질 또는 예측 불확실성이 높습니다. 재촬영을 권고합니다.",
            "ood_flag": bool(ood_flag),
        }
    if prob_pos >= high_th:
        return {
            "label": "abnormal_high_conf",
            "message": "슬개골 탈구 소견이 의심됩니다. 추가 진단을 권장합니다.",
            "ood_flag": False,
        }
    return {
        "label": "normal_high_conf",
        "message": "정상 소견으로 판단됩니다.",
        "ood_flag": False,
    }
