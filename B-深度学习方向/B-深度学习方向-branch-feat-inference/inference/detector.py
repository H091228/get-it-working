"""异常检测推理：吃传感器读数窗口，输出 (anomaly_prob, anomaly_label)。

输入窗口与 docs/contract.md 第 3 节契约一致：
每帧为 [front, left, right, back, touch] 五个数值（touch 用 0/1 表示），
窗口长度 WINDOW_SIZE = 10，即最近 10 帧传感器读数。

当前版本为规则基线（rule-v0.1）：
- 窗口内碰撞事件（第 5 位 touch=1，由 sim 在指令被碰撞拒绝时置 1）
  的帧数越多，异常概率越高；
  prob = min(0.99, 碰撞帧数 * 0.25)；
- prob >= 0.5（即 10 帧内碰撞 2 次及以上）判定为 anomaly。

升级路径（无需改动 sim / api）：
把 update() 内部换成训练好的 PyTorch 模型推理即可——
输入仍是这个 (10, 5) 的窗口张量，输出仍是 (prob, label)，
接口与契约字段保持不变，详见 docs/model.md。
"""

from collections import deque

# 窗口长度：最近 N 帧传感器读数（与契约一致，冻结后不再改动）
WINDOW_SIZE = 10

# 每帧碰撞对异常概率的贡献
PROB_PER_TOUCH = 0.25

# 异常判定阈值：prob 达到该值即输出 label = anomaly
ANOMALY_THRESHOLD = 0.5

# 模型版本号：便于在日志与前端区分规则基线与后续训练的模型
MODEL_VERSION = "rule-v0.1"

# 每帧固定 5 个字段：front / left / right / back / touch
FRAME_FIELDS = 5


class AnomalyDetector:
    """异常检测器：维护最近 WINDOW_SIZE 帧的滑动窗口并输出判定。"""

    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self._window = deque(maxlen=window_size)

    def reset(self):
        """清空窗口（sim 的 RESET 指令会调用）。"""
        self._window.clear()

    def update(self, frame):
        """输入一帧读数，返回 (anomaly_prob, anomaly_label)。

        frame: [front, left, right, back, touch]，第 5 位为碰撞事件
        标志（1.0 = 本次指令发生碰撞，0.0 = 正常），由 sim 填入。
        """
        if len(frame) != FRAME_FIELDS:
            raise ValueError(
                f"frame must have {FRAME_FIELDS} fields "
                f"[front, left, right, back, touch], got {len(frame)}"
            )
        touch = 1.0 if float(frame[-1]) >= 0.5 else 0.0
        self._window.append(touch)
        touches = sum(self._window)
        prob = min(0.99, touches * PROB_PER_TOUCH)
        label = "anomaly" if prob >= ANOMALY_THRESHOLD else "normal"
        return float(prob), label
