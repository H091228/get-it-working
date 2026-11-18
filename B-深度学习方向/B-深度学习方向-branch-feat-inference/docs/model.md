# inference 异常检测模型说明（深度学习方向）

> 负责人：B（深度学习方向）。本模块是标准版 S2「虚拟感知」中
> 异常检测的落地：输入传感器读数窗口，输出异常概率与标签。

## 1. 在系统中的位置

```
sim（嵌入式方向）                    inference（深度学习方向，本模块）
  每处理一条指令采样一帧  ──窗口──▶  AnomalyDetector.update(frame)
  [front,left,right,back,touch]      │
                                     ▼
api（运维方向）负责组装：        (anomaly_prob, anomaly_label)
  simulator = Simulator(detector=AnomalyDetector())
                                     │
web（运维方向）◀── /state 契约字段 ──┘ 页面徽标变红展示
```

- sim 不 import inference，inference 也不 import sim：
  两边只通过「一帧 5 个数」的窗口对接，这正是 docs/contract.md 第 3 节的契约。
- 组装只发生在 api 层一行代码，B 升级模型时 A、C 的代码都不用动。

## 2. 输入输出契约（冻结）

- 输入：一帧 `[front, left, right, back, touch]`，窗口取最近 10 帧
  （`WINDOW_SIZE = 10`），构成 `(10, 5)` 的张量。
- 输出：`(anomaly_prob, anomaly_label)`，
  `anomaly_prob ∈ [0, 1]`，`label ∈ {normal, anomaly}`。
- 判定阈值：`prob >= 0.5 → anomaly`。
- `RESET` 指令会调用 `detector.reset()` 清空窗口。

## 3. 当前版本：规则基线（rule-v0.1）

```
prob = min(0.99, 窗口内碰撞事件帧数 × 0.25)
label = "anomaly"  if prob >= 0.5  else  "normal"
```

碰撞事件由 sim 填入窗口第 5 位（`1.0` = 本次指令被碰撞拒绝），
所以贴墙行驶、读取状态都不会误报，只有真的连续撞击才报警。

选择规则基线的原因：三人小组先保证「整条链路能跑、能演示」，
页面徽标、状态字段、单元测试全部就位之后，再无痛换成真模型。

演示方法：手动控制机器人朝墙/边界连续撞 2 次，
`anomaly_label` 变 `anomaly`，页面「异常检测」徽标变红。

## 4. 升级为训练模型（S2 达标路径，准确率 ≥ 0.85）

1. **采数据**：写一个小脚本轮询 `GET /state`，把每帧
   `[front,left,right,back,touch]` 落盘为 CSV；正常行驶与故意碰撞
   （异常）各采几百帧，按帧打标。
2. **训模型**：最小可用方案是逻辑回归 / 小型 MLP（PyTorch），
   输入 `(10, 5)` 展平，输出 2 分类；也可先用 sklearn 基线对比。
3. **换实现**：把 `AnomalyDetector.update()` 内部换成
   `model(window_tensor)` 推理（加载一次 `model.pt`，`torch.no_grad()`），
   返回值形式不变，`MODEL_VERSION` 改成如 `torch-v1.0`。
4. **评模型**：留出 20% 测试集，报告准确率与混淆矩阵，
   写进本文件和 README 的进度章节（标准版 S2 要求准确率 ≥ 0.85）。

## 5. 智行专项复用

阶段 4 训练完模型后，把「输入窗口 (10,5) → 展平 50 维 →
全层 50→16→2 → softmax → 概率」这张图按模块原子化要求画成
模型完整流程图，标清每步张量维度，正好覆盖智行专项
「每人一份模型完整流程图 + 张量变化标注」的要求，一次准备两用。
