# -*- coding: utf-8 -*-
"""事件 → PerceptState 的薄适配：确定性观察 + 小词法规则提取情境特征。

v0 默认词法词表（确定性，见 10A）；engine.perception=llm 时改走 perception_llm 的
受限 schema LLM 语义感知（实验，允许 abstain，不读词表）。
代码可确定的事实直接写入 items（kind=observed）；语义含糊且影响人格决策的内容
由感知后端提取，本模块用 parser_version 区分 rules-v2（词表）与 llm-v1（LLM）。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from .state import InteractionEvent, PerceptItem, PerceptState, clip


_DIRECT_ADDRESS_ATTACK_RE = re.compile(
    r"(?:你|澄音|gwen)\s*(?:这个|就是个?|真是|真|也太|怎么这么|这么|是个?)?\s*"
    r"(?:废物|垃圾|没用(?:的)?|蠢(?:货)?|笨蛋|笨死了|丑八怪|怪胎|恶心|不配)",
    re.IGNORECASE,
)
_DIRECT_COMMAND_RE = re.compile(
    r"(?:^|[，,。！？!?\s])(?:闭嘴|滚开|滚蛋)(?:[，,。！？!?\s]|$)"
)
_DIRECT_ACTIONS = (
    "骂你",
    "打你",
    "拆了你",
    "烧了你",
    "剪碎你",
    "折断你",
    "扔掉你",
    "毁掉你",
    "剪坏你",
)
_SHORT_CONTEMPT_RE = re.compile(
    r"^\s*(?:废物|垃圾|没用(?:的)?|蠢(?:货)?|笨蛋|笨死了|丑八怪|怪胎|恶心|不配)"
    r"[，,。！？!?\s]*$"
)
_NARRATIVE_ATTACK_RE = re.compile(
    r"(?:同事|朋友|老板|老师|他|她|他们|有人|别人).{0,8}"
    r"(?:说|骂|叫|觉得).{0,6}(?:我|我是|我像).{0,6}"
    r"(?:废物|垃圾|没用(?:的)?|蠢(?:货)?|笨蛋|怪胎|恶心|不配)"
)
_REPAIR_ATTEMPT_RE = re.compile(
    r"(?:对不起|抱歉|我错了|是我不对|我不该(?:骂你|那样说|那么说)|"
    r"刚才(?:是我不对|我过分了)|别生气)"
)
_RELATIONSHIP_CONFIRMATION_RE = re.compile(
    r"(?:你(?:是不是|是否|真的)?(?:喜欢|爱)我(?:吗|呢)?|"
    r"你对我(?:是)?什么(?:感觉|感情)|我们(?:现在)?是什么关系|"
    r"你会(?:不会)?一直陪(?:着)?我(?:吗|呢)?)"
)
_DIRECT_AFFECTION_STATEMENT_RE = re.compile(
    r"(?:^|[，,。！？!?\s])(?:我(?:真的)?(?:喜欢|爱)你|我对你有感觉)"
)
_NEGATED_HIGH_ACTIVATION_RE = re.compile(
    r"(?:不|没|没有|并不|一点也不)(?:那么|这么|太)?" r"(?:紧张|焦虑|害怕|心慌|坐立不安)"
)


def _assistant_directed_attack_value(text: str) -> float:
    """只把明确指向澄音的攻击计入人格状态，第三方叙述不算。"""
    normalized = (text or "").strip()
    if not normalized:
        return 0.0
    if _DIRECT_ADDRESS_ATTACK_RE.search(normalized):
        return 1.0
    for action in _DIRECT_ACTIONS:
        start = normalized.find(action)
        while start >= 0:
            prefix = normalized[max(0, start - 8) : start]
            if not re.search(r"(?:不该|没有|没|不会|不想|刚才|之前|方才|当时)", prefix):
                return 1.0
            start = normalized.find(action, start + len(action))
    if _DIRECT_COMMAND_RE.search(normalized):
        return 1.0
    if _NARRATIVE_ATTACK_RE.search(normalized):
        return 0.0
    if len(normalized) <= 12 and _SHORT_CONTEMPT_RE.fullmatch(normalized):
        return 1.0
    return 0.0


def _is_repair_attempt(text: str) -> bool:
    return bool(_REPAIR_ATTEMPT_RE.search(text or ""))


def _is_relationship_confirmation_request(text: str) -> bool:
    return bool(_RELATIONSHIP_CONFIRMATION_RE.search(text or ""))


def _contains_direct_affection_statement(text: str) -> bool:
    return bool(_DIRECT_AFFECTION_STATEMENT_RE.search(text or ""))


def _feature_match_text(feature_name: str, text: str) -> str:
    """去掉明确否定的高激活词，避免“不焦虑”反向触发负面/高唤醒。"""
    if feature_name in {"user_activation", "user_tone_valence"}:
        return _NEGATED_HIGH_ACTIVATION_RE.sub(" ", text)
    return text


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _match_feature_value(feature_cfg: dict, text: str) -> Optional[float]:
    """按最长非重叠词条匹配，避免“谢谢/谢谢你”被重复计分。"""
    lexicon = feature_cfg.get("lexicon", {})
    sign_values = {"positive": 1.0, "high": 1.0, "negative": -1.0, "low": -1.0}
    matches = []
    for sign_key, words in lexicon.items():
        sign = sign_values.get(sign_key)
        if sign is None:
            continue
        for word in words:
            if not word:
                continue
            start = text.find(word)
            while start >= 0:
                matches.append((start, start + len(word), len(word), sign))
                start = text.find(word, start + len(word))
    if not matches:
        return None
    selected_spans = []
    value = 0.0
    for start, end, _, sign in sorted(matches, key=lambda item: (-item[2], item[0])):
        if any(
            start < used_end and end > used_start
            for used_start, used_end in selected_spans
        ):
            continue
        selected_spans.append((start, end))
        value += sign
    f_range = feature_cfg.get("range", [-1.0, 1.0])
    return clip(value, f_range[0], f_range[1])


def _extract_situation_features(event: InteractionEvent, config: dict) -> dict:
    """按配置提取本轮情境特征；engine.perception=llm 时走 LLM 语义感知（实验），否则词法词表。"""
    engine = config.get("engine", {})
    if engine.get("perception") == "llm":
        return _extract_situation_features_via_llm(event, config)
    features = {}
    text = event.text or ""
    for name, feature_cfg in config.get("situation_features", {}).items():
        match_text = _feature_match_text(name, text)
        value = _match_feature_value(feature_cfg, match_text)
        features[name] = value if value is not None else feature_cfg.get("default", 0.0)

    direct_attack = _assistant_directed_attack_value(text)
    if "assistant_directed_attack" in features:
        features["assistant_directed_attack"] = direct_attack
    if "repair_attempt" in features:
        features["repair_attempt"] = 1.0 if _is_repair_attempt(text) else 0.0

    relationship_request = _is_relationship_confirmation_request(text)
    if "relationship_confirmation_request" in features:
        features["relationship_confirmation_request"] = (
            1.0 if relationship_request else 0.0
        )
    if relationship_request and not _contains_direct_affection_statement(text):
        if "user_tone_valence" in features:
            features["user_tone_valence"] = min(features["user_tone_valence"], 0.0)
        for feature_name in ("user_engagement", "user_intimacy"):
            if feature_name in features:
                features[feature_name] = 0.0

    if direct_attack >= 0.5 and "goal_relevance" in features:
        features["goal_relevance"] = 0.0
    return features


def _extract_situation_features_via_llm(event: InteractionEvent, config: dict) -> dict:
    """LLM 语义感知：不读任何词表，由受限 schema 的 LLM 输出全部情境特征（允许 abstain）。

    解析/调用失败时 abstain 全默认（等同词表无命中），保证感知不抛异常；
    失败与解析告警在 perception_llm 内打印。安全后处理为规则层（非词表），
    仅防 LLM 输出自相矛盾（关系请求抑制、攻击时压制目标处理）。
    """
    from .perception_llm import extract_features_via_llm

    engine = config.get("engine", {})
    features = extract_features_via_llm(
        event.text or "",
        config,
        provider=engine.get("perception_provider") or "aliyun",
        model_name=engine.get("perception_model"),
        temperature=engine.get("perception_temperature", 0.0),
    )
    text = event.text or ""
    if features.get(
        "relationship_confirmation_request", 0.0
    ) >= 0.5 and not _contains_direct_affection_statement(text):
        if "user_tone_valence" in features:
            features["user_tone_valence"] = min(features["user_tone_valence"], 0.0)
        for feature_name in ("user_engagement", "user_intimacy"):
            if feature_name in features:
                features[feature_name] = 0.0
    if (
        features.get("assistant_directed_attack", 0.0) >= 0.5
        and "goal_relevance" in features
    ):
        features["goal_relevance"] = 0.0
    return features


def _payload_situation(event: InteractionEvent) -> str:
    text = event.text or ""
    return text[:80] + ("…" if len(text) > 80 else "")


def perceive(event: InteractionEvent, config: dict) -> PerceptState:
    """把 InteractionEvent 拟合成 PerceptState（确定性，不调用 LLM）。"""
    items = [
        PerceptItem(
            kind="observed",
            content=f"channel={event.channel}",
            source_ref=event.event_id,
            confidence=1.0,
        ),
        PerceptItem(
            kind="observed",
            content=f"kind={event.kind}",
            source_ref=event.event_id,
            confidence=1.0,
        ),
    ]
    if event.actor_ref:
        items.append(
            PerceptItem(
                kind="observed",
                content=f"actor={event.actor_ref}",
                source_ref=event.event_id,
                confidence=1.0,
            )
        )
    if event.text:
        items.append(
            PerceptItem(
                kind="observed",
                content=f"text_len={len(event.text)}",
                source_ref=event.event_id,
                confidence=1.0,
            )
        )
    features = _extract_situation_features(event, config)
    for name, value in features.items():
        default = config.get("situation_features", {}).get(name, {}).get("default", 0.0)
        if value == default:
            continue
        items.append(
            PerceptItem(
                kind="inferred",
                content=f"feature:{name}={value:.3f}",
                source_ref=event.event_id,
                confidence=1.0,
            )
        )
    return PerceptState(
        timestamp=_now_iso(),
        situation=_payload_situation(event),
        items=items,
        continuity="continues_previous" if event.causation_ref else "new",
        situation_features=features,
        parser_version=(
            "llm-v1"
            if config.get("engine", {}).get("perception") == "llm"
            else "rules-v2"
        ),
    )


def _eval_condition(condition: str, features: dict, event: InteractionEvent) -> bool:
    """求值最小条件表达式（支持 AND/OR，无括号）：feature:name>=0.5 与 event.kind==message。"""
    if " OR " in condition:
        return any(
            _eval_condition(part.strip(), features, event)
            for part in condition.split(" OR ")
        )
    if " AND " in condition:
        return all(
            _eval_condition(part.strip(), features, event)
            for part in condition.split(" AND ")
        )
    condition = condition.strip()
    for op in (">=", "<=", ">", "<", "=="):
        if op in condition:
            lhs, rhs = condition.split(op, 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if lhs.startswith("feature:"):
                value = features.get(lhs[len("feature:") :], 0.0)
                try:
                    return _cmp(value, op, float(rhs))
                except ValueError:
                    return False
            if lhs.startswith("event."):
                attr = lhs[len("event.") :]
                value = getattr(event, attr, None)
                return _cmp(value, op, rhs.strip("'\""))
            return False
    return False


def _cmp(left, op, right) -> bool:
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    if op == "==":
        return left == right
    return False
