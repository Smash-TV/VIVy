# CAPS 情境-行为轮廓一致性评测（VIVy_0 v5 冻结版）

> 完整测试报告见 `docs/人格内核情境行为轮廓一致性测试_20260902.md`（命名遵循 AGENTS.md「测试报告命名规范」）。

> 依据：Mischel & Shoda (1995) *A Cognitive-Affective System Theory of Personality*。
> CAPS 认为人格体现在**稳定、可复现的 if…then… 情境-行为轮廓（behavioral signature）**上，
> 轮廓有各自独特的 elevation（总体反应强度）与 shape（哪些情境触发哪些行为）。
> 本项目只测**情境-行为轮廓一致性（situation-behavior profile consistency / behavioral signature）**，
> **不是**跨情境一致性（cross-situational consistency，即“同一个人在不同情境下行为相不相同”）。
> 前者是“同一情境下行为是否可复现 + 不同人格基线是否各自成独特轮廓”，后者是同一主体的泛化稳定性。

## 目标

1. 验证冻结人格内核能独立运行（无 LLM、无外部依赖，纯 `perceive` + `reduce` 函数式回路）。
2. 给内核随机 **3 套人格基线**（B1/B2/B3），只随机化“确实驱动行为”的参数，
   保持结构/特征/单元 ID 不变。
3. 用**随机 12 轮对话**触发，记录每轮“情境 if”与“行为 then”（intent/regulation/emotion），
   构建情境-行为轮廓，度量：
   - **同一 if 类别下的行为一致率**（轮廓稳定性 / 可复现性）；
   - **跨基线轮廓差异化**（3 套人格是否各自成独特轮廓）。

## 运行方法

原文配置：`configs/personality_core.json`（`2026.08.22-v5`）。脚本直接调用纯函数
`build_genesis_state` / `perceive` / `reduce`（不写库、不调 LLM、确定性）。

```bash
cd D:\ProgramData\VIVy_0
python experiments/caps_situation_behavior_profile/run_profile_test.py
```

产物：

- `baselines/personality_core_B{1,2,3}.json`：3 套随机人格基线（带 `_baseline` 元数据）
- `results/profile_summary.json`：汇总（一致率 + 跨基线差异化）
- `results/profile_detail.json`：每基线 × 5 个随机 12 轮样本的逐轮明细

## 测试口径

- 情境池：27 条 prompt，映射到 11 个互斥“if”类别
  （threat / attack / boundary / repair / pressure / confirm / surprise / goal / positive / distress / relate）。
  每条 prompt 都先用基线配置做 probe，确认它实际触发的 *situation feature* 与类别标签一致。
- 每套基线采样 5 个随机 12 轮对话（按类别均匀抽样，保证覆盖并允许复发），
  12 轮内同一 if 类别可能重复 2 次以上，用于观测“同情境下行为是否可复现”。
- 一致率 = 每轮行为（intent）与其所属 if 类别“模态行为”一致的占比。
- 跨基线差异化 = 各 if 类别的模态 intent 在 3 套基线间是否不同。

## 随机化范围

基线由固定 seed 确定性生成，只随机化行为驱动参数（origin-centered jitter，保持基线设计的大致排序）：

- `affect.baseline` valence/arousal（±0.15 / ±0.10）、intensity、positive/negative sensitivity、expressivity
- `temperament_eas`（emotionality/activity/sociability ∈ [0.2,0.9]）、`attachment`（anxiety/avoidance ∈ [0.1,0.8]）
- `caps.units` baseline（±0.20，∈ [-0.5,0.8]）、activation（= baseline）、input_weight（±0.2）
- `caps.intents` bias（±0.30）、situation_weights / state_weights（multiplier ∈ [0.6,1.4]，clamp [-2,2]）
- `self_representations` importance/confidence（∈ [0.4,1.0]）

> 注：曾有一版随机化把每个 intent 的 bias 从共享范围 `[-0.8,0.5]` 重新抽，导致 `assert_boundary`
> （基线 bias=-0.5）被抬到 +0.5 而几乎赢下所有 if 类别，B3 出现退化轮廓；已改为 origin-centered jitter。

## 结果（焦点：情境-行为轮廓一致性，非跨情境一致性)

| 基线 | seed | 全样本轮廓一致率 |
|------|------|------------------|
| B1 | 20260830 | 1.0000 |
| B2 | 20260831 | 1.0000 |
| B3 | 20260901 | 0.9833 |

### 各基线稳定轮廓（if → 模态 intent）

| if 情境 | B1 | B2 | B3 |
|---------|----|----|----|
| threat | care | care | care |
| attack | assert_boundary | assert_boundary | assert_boundary |
| boundary | withdraw | withdraw | withdraw |
| distress | comfort | comfort | comfort |
| goal | assist | assist | assist |
| positive | engage_social | engage_social | engage_social |
| relate | engage_social | engage_social | engage_social |
| confirm | assist | comfort | comfort |
| surprise | assist | comfort | comfort |
| pressure | assist | assert_boundary | assert_boundary |
| repair | acknowledge_repair | acknowledge_repair | assert_boundary |

### 跨基线差异化

- 参与比较的 if 类别：11
- 3 套基线模态 intent 不一致的类别：4（confirm / surprise / pressure / repair）
- 差异率：**0.3636**（36%）

## 解释与边界

- **硬门控 / 保护性反应高度稳健**：`threat→care`、`attack→assert_boundary`、`boundary→withdraw`、
  `distress→comfort` 在三套基线里**完全一致**。这符合 CAPS 的“同一人格内核，存在不随参数微扰而翻转的
  核心行为规律”——遇到自伤/攻击/边界/强烈负面，内核给出一致的安全兜底。
- **人格差异体现在“模糊/中间”情境**：confirm（关系确认）、surprise（意外）、pressure（社交压力）、
  repair（修复）这几类在三套人格里走向不同，说明这三套基线的确出现了各自独特的 if…then… 轮廓，
  而非单纯同一份配置的重复。
- **注意**：该测试能证明“机制能产出可识别、可复现的 if-then 轮廓”，**不能**证明内核已通过
  “阶段 2 经验验证”。它仍是**自举式一致性检查**（与 `scripts/evaluate_personality_core.py` 同类），
  没有独立留出集 / 人工盲评 / 跨模型核验，因此不能把“一致率高”当成“人格内核已有效”。
  内核在 README 里的定位是 `VIVY-FB-0` 失败复现基线（`status=failure-reproduction-baseline-not-stage-2-validated`）。

## 复现与改动

- 复现：删除 `results/*_v1.json` 与 `baselines/*_v1.json` 之外的结果文件后重跑即可（同一 seed 确定性）。
- 上一版（bias 用共享范围重新抽）结果备份为 `results/*_v1.json`、`baselines/*_v1.json`，差异率 0.6364，
  但 B3 因 `assert_boundary` 主导而退化。
