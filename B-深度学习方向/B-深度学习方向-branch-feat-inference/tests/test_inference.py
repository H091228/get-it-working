"""inference 异常检测模块单元测试（纯规则基线，不依赖 sim / api）。"""

import pytest

from inference.detector import ANOMALY_THRESHOLD, WINDOW_SIZE, AnomalyDetector

FRAME_NORMAL = [2.0, 2.0, 2.0, 2.0, 0.0]
FRAME_TOUCH = [1.0, 2.0, 2.0, 2.0, 1.0]


def test_normal_frames_keep_prob_zero():
    detector = AnomalyDetector()
    for _ in range(WINDOW_SIZE):
        prob, label = detector.update(FRAME_NORMAL)
    assert prob == 0.0
    assert label == "normal"


def test_single_touch_stays_below_threshold():
    detector = AnomalyDetector()
    prob, label = detector.update(FRAME_TOUCH)
    assert 0.0 < prob < ANOMALY_THRESHOLD
    assert label == "normal"


def test_two_touches_trigger_anomaly():
    detector = AnomalyDetector()
    detector.update(FRAME_TOUCH)
    prob, label = detector.update(FRAME_TOUCH)
    assert prob >= ANOMALY_THRESHOLD
    assert label == "anomaly"


def test_window_slides_back_to_normal():
    """碰撞帧滑出窗口后，异常概率回落、标签恢复 normal。"""
    detector = AnomalyDetector()
    detector.update(FRAME_TOUCH)
    detector.update(FRAME_TOUCH)
    assert detector.update(FRAME_NORMAL)[1] == "anomaly"
    for _ in range(WINDOW_SIZE):
        prob, label = detector.update(FRAME_NORMAL)
    assert prob == 0.0
    assert label == "normal"


def test_window_length_is_bounded():
    """窗口超过 N 帧后旧的碰撞帧自动滑出。"""
    detector = AnomalyDetector()
    detector.update(FRAME_TOUCH)
    for _ in range(WINDOW_SIZE):
        prob, label = detector.update(FRAME_NORMAL)
    assert label == "normal"


def test_reset_clears_window():
    detector = AnomalyDetector()
    detector.update(FRAME_TOUCH)
    detector.update(FRAME_TOUCH)
    detector.reset()
    prob, label = detector.update(FRAME_NORMAL)
    assert prob == 0.0
    assert label == "normal"


def test_front_distance_below_one_without_touch_flag_is_normal():
    """契约里 touch 才是碰撞标志：距离近但 touch=0 不计碰撞。"""
    detector = AnomalyDetector()
    frame = [0.5, 2.0, 2.0, 2.0, 0.0]
    prob, label = detector.update(frame)
    assert prob == 0.0
    assert label == "normal"


def test_rejects_wrong_frame_length():
    detector = AnomalyDetector()
    with pytest.raises(ValueError):
        detector.update([1.0, 2.0, 3.0])
