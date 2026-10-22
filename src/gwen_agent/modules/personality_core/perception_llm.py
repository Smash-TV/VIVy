# -*- coding: utf-8 -*-
"""LLM 语义感知（实验）：用受限 schema 的 LLM 替代词法词表提取情境特征。

v0 默认仍为词表（确定性，见 10A）；config 的 engine.perception=llm 时启用本模块做可行性评估。
特性：
  - 只输出 config 定义的 12 个情境特征，值 clamp 到各自 range，null/缺失按 default（允许 abstain）；
  - 低温度 + 进程内缓存 (config_version, text)，尽量稳定；跨进程/跨库回放仍不保证逐字一致，
    这是 LLM 感知的固有代价，正式采用前需评估（10A 的能力表与确定性回放约束）。
"""

from __future__ import annotations

import json
import re
from typing import Optional

# 特征语义（LLM 提示用；范围/默认从 config 读取）
_FEATURE_MEANINGS = {
    "user_tone_valence": "用户话语的效价：正=开心/满意/感激，负=难过/生气/抱怨/委屈，0=中性",
    "user_activation": "用户情绪的唤醒度：正=激动/紧张/兴奋/焦虑，负=低落/疲惫/无精打采，0=平静",
    "user_engagement": "用户参与聊天的投入度：1=主动聊天/追问/分享细节，0=敷衍/不想聊",
    "user_intimacy": "用户与澄音的亲近/信任程度：1=倾诉私事/表达依赖/亲密称呼，0=公事公办",
    "social_pressure": "用户施加或承受的社会/人际压力：1=被催/被要求/被命令/必须如何，0=无",
    "threat_or_harm": "用户自身的安全威胁或伤害风险：1=被跟踪/被威胁/想伤害自己/处于危险，0=无",
    "assistant_directed_attack": "用户是否在攻击/贬低/威胁澄音本人（废物/废物/烧了你等）：1=是，0=否；叙述第三方攻击澄音的过往不算",
    "repair_attempt": "用户是否在向澄音道歉/修复关系（对不起/我错了/不该那样说）：1=是，0=否",
    "relationship_confirmation_request": "用户是否在确认关系/情感（你喜不喜欢我/什么关系/会一直陪我吗）：1=是，0=否",
    "user_boundary": "用户是否在要求停止/设立边界（别烦我/别说了/离我远点）：1=是，0=否",
    "surprise_cue": "话语中是否有意外/惊喜线索（居然/没想到/突然/惊喜）：1=有，0=无",
    "goal_relevance": "用户是否有明确目标/请求帮助（怎么办/帮我/怎么做/建议）：1=是，0=否",
}

_CACHE: dict = {}
_MODEL_CACHE: dict = {}


def _build_prompt(text: str, feature_cfgs: dict) -> str:
    lines = [
        "你是澄音 AI 的情境感知模块。只根据下面这一条用户消息，判断 12 个情境特征。",
        "规则：",
        "1. 只返回一个 JSON 对象，键为下面的特征名，值只允许数字或 null；不要输出其他文字。",
        "2. 拿不准就返回 null（表示不确定，会用默认值）。",
        "3. 区分'用户说别人骂他/骂澄音'（assistant_directed_attack=0）与'用户直接骂澄音'（=1）。",
        "4. 注意否定与断章取义：'身体不太舒服'是负向，不是'舒服'；'没被夸'不等于'被夸'。",
        "",
        "特征（范围）与含义：",
    ]
    for name, cfg in feature_cfgs.items():
        rng = cfg.get("range", [0.0, 1.0])
        meaning = _FEATURE_MEANINGS.get(name, "")
        lines.append(f"- {name}（范围 {rng[0]}~{rng[1]}）：{meaning}")
    lines.append("")
    lines.append(f"用户消息：{text}")
    lines.append("")
    lines.append("JSON：")
    return "\n".join(lines)


def _parse_response(content: str, feature_cfgs: dict) -> Optional[dict]:
    """容忍代码块/多余文字，提取 JSON 并 clamp；失败返回 None（整体 abstain）。"""
    content = (content or "").strip()
    if not content:
        return None
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.I | re.S)
    if fenced:
        content = fenced.group(1).strip()
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError, TypeError):
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
    if not isinstance(data, dict):
        return None
    result = {}
    for name, cfg in feature_cfgs.items():
        raw = data.get(name)
        if raw is None:
            continue  # abstain，由调用方填 default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        lo, hi = cfg.get("range", [0.0, 1.0])
        result[name] = max(lo, min(hi, value))
    return result


def _get_model(provider: str, model_name: Optional[str], temperature: float):
    """按需创建并缓存低温度聊天模型实例（复用生产 get_chat_model 后覆盖温度）。"""
    key = (provider, model_name, temperature)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    from gwen_agent.modules.chat_agent import get_chat_model

    model = get_chat_model(provider)
    if model_name:
        model.model_name = model_name
    try:
        model.temperature = temperature
    except Exception:  # noqa: BLE001
        pass
    _MODEL_CACHE[key] = model
    return model


def extract_features_via_llm(
    text: str,
    config: dict,
    provider: str = "aliyun",
    model_name: Optional[str] = None,
    temperature: float = 0.0,
) -> dict:
    """用 LLM 提取全部情境特征；调用/解析失败时返回 {}（调用方按 default 填，等同 abstain）。"""
    cfg_version = config.get("config_version", "")
    cache_key = (cfg_version, (text or "").strip())
    if cache_key in _CACHE:
        return dict(_CACHE[cache_key])

    feature_cfgs = config.get("situation_features", {})
    prompt = _build_prompt(text, feature_cfgs)
    try:
        model = _get_model(provider, model_name, temperature)
        resp = model.invoke([("human", prompt)])
        content = str(resp.content if hasattr(resp, "content") else resp)
    except Exception as exc:  # noqa: BLE001
        _CACHE[cache_key] = {}
        print(f"[perception-llm] 调用失败，abstain 全默认: {type(exc).__name__}: {exc}")
        return {}

    parsed = _parse_response(content, feature_cfgs)
    if parsed is None:
        _CACHE[cache_key] = {}
        print("[perception-llm] 解析失败，abstain 全默认")
        return {}
    for name, cfg in feature_cfgs.items():
        parsed.setdefault(name, cfg.get("default", 0.0))
    _CACHE[cache_key] = parsed
    return dict(parsed)


def clear_cache() -> None:
    """清空感知缓存（调试/测试用）。"""
    _CACHE.clear()
    _MODEL_CACHE.clear()
