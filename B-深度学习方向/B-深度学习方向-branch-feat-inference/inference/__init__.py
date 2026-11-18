"""inference：异常检测推理模块（深度学习方向负责）。"""

from .detector import ANOMALY_THRESHOLD, MODEL_VERSION, WINDOW_SIZE, AnomalyDetector

__all__ = ["AnomalyDetector", "ANOMALY_THRESHOLD", "MODEL_VERSION", "WINDOW_SIZE"]
