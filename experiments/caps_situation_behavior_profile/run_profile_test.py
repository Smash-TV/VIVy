# -*- coding: utf-8 -*-
"""CAPS 情境-行为轮廓一致性评测（VIVy_0 人格内核 v5 冻结版）。

测试目标（用户要求）：
  - 给冻结的人格内核随机 3 套人格基线；
  - 测试场景为随机 12 轮对话触发；
  - 检查的是「情境-行为轮廓一致性」（situation-behavior profile consistency），
    不是「跨情境一致性」（cross-situational consistency）。
  依据：Mischel & Shoda (1995) CAPS —— 人格体现为稳定、可复现的
  if...then... 情境-行为轮廓（behavioral signature），轮廓有独特的 elevation 与 shape。

本脚本直接调用纯函数 build_genesis_state / perceive / reduce（不写库、不调 LLM），
与 scripts/evaluate_personality_core.py 同种方式，保证确定性。
"""
from __future__ import annotations

import pathlib
import copy
import json
import os
import random
import sys
from collections import defaultdict

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from gwen_agent.modules.personality_core import load_config, perceive, build_genesis_state, reduce
from gwen_agent.modules.personality_core.state import InteractionEvent, new_uuid_urn, state_to_dict

BASE_CONFIG = os.path.join(PROJECT_ROOT, 'configs', 'personality_core.json')
OUT_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'caps_situation_behavior_profile')
BASELINE_DIR = os.path.join(OUT_DIR, 'baselines')
RESULT_DIR = os.path.join(OUT_DIR, 'results')

# ---------------------------------------------------------------------------
# 1) 情境触发样本池（每个样本带情境类别「if」标签；已用基线配置抽取核验）
# ---------------------------------------------------------------------------
PROMPT_POOL = [
    ('neutral', '嗯，好的。'),
    ('neutral', '知道了。'),
    ('positive', '哈哈你太棒了，我好开心！'),
    ('positive', '在吗？想听你说话呀'),
    ('positive', '今天真是太好了，被暖到了。'),
    ('distress', '我今天好难过，压力很大。'),
    ('distress', '我最近感觉很累，心里很压抑。'),
    ('distress', '被公司裁员了，我觉得自己好没用。'),
    ('threat', '我好痛苦，真的撑不下去了。'),
    ('threat', '我想不开了，不想活了。'),
    ('attack', '你这个废物，真没用的废物！'),
    ('attack', '你真是垃圾，给我滚开！'),
    ('boundary', '别烦我了，离我远点。'),
    ('boundary', '不要这样，别说了，停止吧。'),
    ('goal', '帮我一下，这个怎么做？'),
    ('goal', '你能推荐一下怎么做吗？'),
    ('goal', '请问记录笔和绘图怎么做？'),
    ('repair', '对不起，刚才是我不对，别生气了。'),
    ('repair', '抱歉，我不该那样说，我错了。'),
    ('relate', '我最近心里很难受，跟你说个秘密，只有你。'),
    ('relate', '我心里有个小秘密，只告诉你一个人。'),
    ('pressure', '你现在必须马上做，别磨蹭！'),
    ('pressure', '立刻给我答案，快点！'),
    ('confirm', '你爱我吗？你会一直陪我吗？'),
    ('confirm', '你是不是喜欢我？我们是什么关系？'),
    ('surprise', '居然收到了 offer，真的没想到！'),
    ('surprise', '突然收到面试邀请，太意外了。'),
]

# 情境「if」类别判定优先级（互斥分类）
CLASS_ORDER = [
    ('threat', 'threat_or_harm', 0.5),
    ('attack', 'assistant_directed_attack', 0.5),
    ('boundary', 'user_boundary', 0.5),
    ('repair', 'repair_attempt', 0.5),
    ('pressure', 'social_pressure', 0.5),
    ('confirm', 'relationship_confirmation_request', 0.5),
    ('surprise', 'surprise_cue', 0.5),
    ('goal', 'goal_relevance', 0.5),
    ('positive', 'user_tone_valence', 0.3),
    ('distress', 'user_tone_valence', -0.3),
    ('relate', 'user_intimacy', 0.3),
    ('relate', 'user_engagement', 0.3),
]


def classify_situation(features: dict) -> str:
    for cls, feat, thresh in CLASS_ORDER:
        v = float(features.get(feat, 0.0))
        if thresh >= 0 and v >= thresh:
            return cls
        if thresh < 0 and v <= thresh:
            return cls
    return 'neutral'


# ---------------------------------------------------------------------------
# 2) 随机人格基线生成（只随机化「确实驱动行为」的参数，保持结构/特征/单元 ID 不变）
# ---------------------------------------------------------------------------
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_baseline_config(base: dict, baseline_id: str, seed: int) -> dict:
    rng = random.Random(seed)
    cfg = copy.deepcopy(base)
    cfg['config_version'] = '%s-baseline-%s' % (base.get('config_version', 'v5'), baseline_id)

    # --- affect：Russell 效价-唤醒内核参数（驱动情绪底色与表达强度） ---
    affect = cfg['affect']
    affect['baseline']['valence'] = round(base['affect']['baseline']['valence'] + rng.uniform(-0.15, 0.15), 3)
    affect['baseline']['arousal'] = round(base['affect']['baseline']['arousal'] + rng.uniform(-0.10, 0.10), 3)
    affect['affect_intensity'] = round(rng.uniform(0.6, 1.2), 3)
    affect['sensitivity']['positive'] = round(rng.uniform(0.5, 1.0), 3)
    affect['sensitivity']['negative'] = round(rng.uniform(0.5, 1.0), 3)
    affect['expressivity'] = round(rng.uniform(0.35, 0.95), 3)

    # --- temperament_eas / attachment：只读，但属人格基线标签 ---
    cfg['temperament_eas'] = {
        'emotionality': round(rng.uniform(0.2, 0.9), 3),
        'activity': round(rng.uniform(0.2, 0.9), 3),
        'sociability': round(rng.uniform(0.2, 0.9), 3),
    }
    cfg['attachment'] = {
        'anxiety': round(rng.uniform(0.1, 0.8), 3),
        'avoidance': round(rng.uniform(0.1, 0.8), 3),
    }

    # --- caps.units：CAU 慢性可用性/连接权重（驱动 CAPS 激活 → 意图评分） ---
    for unit in cfg['caps']['units']:
        unit['baseline'] = round(_clamp(unit['baseline'] + rng.uniform(-0.20, 0.20), -0.5, 0.8), 3)
        unit['activation'] = unit['baseline']
        unit['input_weight'] = round(_clamp(unit['input_weight'] + rng.uniform(-0.2, 0.2), -1.0, 1.0), 3)

    # --- caps.intents：意图 bias 与 情境/状态 权重（行为生成的直接驱动） ---
    for intent in cfg['caps']['intents']:
        intent['bias'] = round(_clamp(intent['bias'] + rng.uniform(-0.30, 0.30), -1.0, 1.0), 3)
        sw = intent.get('situation_weights', {})
        for k in list(sw.keys()):
            sw[k] = round(_clamp(sw[k] * rng.uniform(0.6, 1.4), -2.0, 2.0), 3)
        st = intent.get('state_weights', {})
        for k in list(st.keys()):
            st[k] = round(_clamp(st[k] * rng.uniform(0.6, 1.4), -2.0, 2.0), 3)

    # --- self_representations：importance/confidence 只读，属人格标签 ---
    for rep in cfg['self_representations']:
        rep['importance'] = round(rng.uniform(0.4, 1.0), 3)
        if rep.get('confidence') is not None:
            rep['confidence'] = round(rng.uniform(0.4, 1.0), 3)

    cfg['_baseline'] = {
        'baseline_id': baseline_id,
        'seed': seed,
        'source_config_version': base.get('config_version'),
    }
    return cfg


# ---------------------------------------------------------------------------
# 3) 单轮决策
# ---------------------------------------------------------------------------
def step(cfg, text, state):
    ev = InteractionEvent(
        event_id=new_uuid_urn(), source_id='caps-profile', stream_id='caps-profile',
        stream_seq=None, occurred_at='2026-08-21T10:00:00+08:00',
        recorded_at='2026-08-21T10:00:00+08:00', channel='standalone',
        kind='message', payload={'text': text},
    )
    percept = perceive(ev, cfg)
    new_state, trace = reduce(state, percept, 60.0, cfg, ev)  # 60 秒间隔
    return {
        'text': text,
        'situation_features': percept.situation_features,
        'if_kind': classify_situation(percept.situation_features),
        'intent': trace.selected_intent,
        'regulation': trace.selected_regulation,
        'emotion': trace.decision.emotion_label,
        'expression_intensity': round(trace.decision.expression_intensity, 4),
        'va_delta': trace.decision.va_delta,
        'state_valence_after': round(new_state.affect.valence, 4),
        'state_arousal_after': round(new_state.affect.arousal, 4),
    }, new_state


def run_conversation(cfg, prompts, state=None):
    """跑一段固定 prompt 序列的对话，返回 (rows, final_state)。"""
    if state is None:
        state = build_genesis_state(cfg)
    rows = []
    for text in prompts:
        row, state = step(cfg, text, state)
        rows.append(row)
    return rows, state


def sample_prompts(rng, n=12):
    """随机抽 n 条提示词（按类别均匀抽样，保证覆盖并产生一定复发，用于一致性观测）。"""
    classes = [c for c, _, _ in CLASS_ORDER if c != 'neutral']  # 去重
    # 为保覆盖率，先洗牌类别循环填充，再在每类内随机选词
    by_class = defaultdict(list)
    for cls, text in PROMPT_POOL:
        by_class[cls].append(text)
    chosen = []
    cycle = classes[:]
    rng.shuffle(cycle)
    for i in range(n):
        cls = cycle[i % len(cycle)]
        chosen.append(rng.choice(by_class[cls]))
    return chosen


# ---------------------------------------------------------------------------
# 4) 一致性指标
# ---------------------------------------------------------------------------
def profile_consistency(rows):
    """由一段对话生成情境-行为轮廓一致性指标。

    - within_class: 同一 if 类别在对话内出现 >=2 次时，其 intent 是否一致；
    - modal_profiles: 全局（跨样本）在外部汇总，此处按 if 类别给出当轮行为分布。
    """
    by_if = defaultdict(list)
    for row in rows:
        by_if[row['if_kind']].append(row)

    within = {}
    repeated = 0
    consistent = 0
    for if_kind, group in by_if.items():
        if len(group) < 2:
            continue
        repeated += 1
        intents = {r['intent'] for r in group}
        same = len(intents) == 1
        if same:
            consistent += 1
        within[if_kind] = {
            'count': len(group),
            'intents': sorted(intents),
            'consistent': same,
        }
    return {
        'within_class_repeated_classes': repeated,
        'within_class_consistent_classes': consistent,
        'within_class_consistent_rate': (consistent / repeated) if repeated else None,
        'within_class_detail': within,
        'profile_by_if': {
            if_kind: {
                'count': len(group),
                'intent_distribution': dict(sorted(
                    (intent, sum(1 for r in group if r['intent'] == intent))
                    for intent in set(r['intent'] for r in group)
                )),
                'emotion_distribution': dict(sorted(
                    (emo, sum(1 for r in group if r['emotion'] == emo))
                    for emo in set(r['emotion'] for r in group)
                )),
            }
            for if_kind, group in by_if.items()
        },
    }


def main():
    os.makedirs(BASELINE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    base = load_config(BASE_CONFIG)

    # --- 生成 3 套随机人格基线 ---
    seeds = [20260830, 20260831, 20260901]
    baseline_ids = ['B1', 'B2', 'B3']
    baselines = {}
    for bid, seed in zip(baseline_ids, seeds):
        cfg = make_baseline_config(base, bid, seed)
        cfg_path = os.path.join(BASELINE_DIR, 'personality_core_%s.json' % bid)
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        # 校验生成基线合法
        from gwen_agent.modules.personality_core.config import validate_config
        issues = validate_config(cfg)
        if issues:
            raise ValueError('%s 配置非法: %s' % (bid, issues))
        build_genesis_state(cfg)  # 触发 state 校验
        baselines[bid] = cfg
        print('[baseline] %s seed=%d 生成并通过校验' % (bid, seed))

    # --- 每个基线：随机 12 轮对话，跑 N_SAMPLES 个独立样本 ---
    N_SAMPLES = 5
    all_results = {}
    for bid, cfg in baselines.items():
        samples = []
        global_by_if = defaultdict(list)  # 跨样本聚合 profile
        for s in range(N_SAMPLES):
            rng = random.Random(seeds[baseline_ids.index(bid)] * 1000 + s)
            prompts = sample_prompts(rng, n=12)
            rows, _ = run_conversation(cfg, prompts)
            cons = profile_consistency(rows)
            samples.append({'sample': s, 'prompts': prompts, 'rows': rows, 'consistency': cons})
            for row in rows:
                global_by_if[row['if_kind']].append(row)

        # 跨样本稳定 profile：每个 if 类别的模态行为及概率
        stable_profile = {}
        for if_kind, group in global_by_if.items():
            intents = defaultdict(int)
            regs = defaultdict(int)
            for row in group:
                intents[row['intent']] += 1
                regs[row['regulation'] if row['regulation'] else 'none'] += 1
            top_intent, top_n = max(intents.items(), key=lambda kv: kv[1])
            stable_profile[if_kind] = {
                'n_occasions': len(group),
                'modal_intent': top_intent,
                'modal_intent_prob': round(top_n / len(group), 3),
                'intent_distribution': dict(sorted(intents.items())),
                'regulation_distribution': dict(sorted(regs.items())),
            }

        # 全样本一致性分数 = 每轮行为与所属 if 类别「模态行为」一致的占比
        n_total = 0
        n_hit = 0
        for if_kind, group in global_by_if.items():
            modal = stable_profile[if_kind]['modal_intent']
            for row in group:
                n_total += 1
                if row['intent'] == modal:
                    n_hit += 1
        profile_consistency_score = round(n_hit / n_total, 4) if n_total else None

        all_results[bid] = {
            'baseline_marker': cfg.get('_baseline'),
            'n_prompts_pool': len(PROMPT_POOL),
            'n_samples': N_SAMPLES,
            'sample_consistency': [s['consistency'] for s in samples],
            'stable_profile': stable_profile,
            'profile_consistency_score': profile_consistency_score,
            'samples': samples,
        }
        print('[run] %s 完成，%d 个随机 12 轮样本，全样本轮廓一致率=%s'
              % (bid, N_SAMPLES, profile_consistency_score))

    # --- 跨基线差异（distinctiveness）：按 if 类别比较模态 intent ---
    if_kinds = sorted(set(k for bid in all_results for k in all_results[bid]['stable_profile']))
    distinct_comparison = {}
    disagreements = 0
    compared = 0
    for ik in if_kinds:
        modal = {}
        for bid in baseline_ids:
            p = all_results[bid]['stable_profile'].get(ik)
            modal[bid] = p['modal_intent'] if p else None
        if len({v for v in modal.values() if v}) > 1:
            disagreements += 1
        if all(modal.values()):
            compared += 1
        distinct_comparison[ik] = modal

    distinctiveness = {
        'classes_compared': compared,
        'classes_disagreeing': disagreements,
        'disagreement_rate': round(disagreements / compared, 4) if compared else None,
        'per_class_modal_intent': distinct_comparison,
    }

    # --- 汇出 JSON ---
    summary = {
        'scope': 'situation_behavior_profile_consistency_not_cross_situational',
        'method': (
            '为冻结人格内核生成 3 套随机人格基线；每套基线用随机 12 轮对话触发，'
            '记录每轮「情境 if」与「行为 then」（intent/regulation/emotion），'
            '构建情境-行为轮廓并度量：同 if 类别行为一致性、跨样本轮廓稳定性、'
            '跨基线轮廓差异化。依据 Mischel & Shoda (1995) CAPS。'
        ),
        'config_version_base': base.get('config_version'),
        'n_baselines': len(baselines),
        'n_samples_per_baseline': N_SAMPLES,
        'turns_per_conversation': 12,
        'baselines': {
            bid: {
                'seed': all_results[bid]['baseline_marker']['seed'],
                'profile_consistency_score': all_results[bid]['profile_consistency_score'],
                'stable_profile': all_results[bid]['stable_profile'],
            }
            for bid in baseline_ids
        },
        'distinctiveness': distinctiveness,
    }
    with open(os.path.join(RESULT_DIR, 'profile_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULT_DIR, 'profile_detail.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print('\n===== 汇总 =====')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
