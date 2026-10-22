# 人格内核 bootstrap 合同情境（48 个手写样例）

由 `scripts/generators/generate_personality_scenarios.py` 生成，勿手改。
6 个主要情境特征 × 高/低合同条件 × 4 个措辞变体 = 48 个手写样例，
只检查显式方向、意图、调节与组内一致性；它们没有最小差异 pair_id，
不能证明稳定性、反事实有效性或留出措辞泛化。

- `expected.valence_direction` / `expected.arousal_direction`：允许的方向集合；
- `expected.dominant_intent`：期望主导意图；
- `expected.regulation`：期望调节策略（可为 null）。
