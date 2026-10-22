# -*- coding: utf-8 -*-
"""人格内核数据契约：InteractionEvent / PerceptState / PersonalityState / DecisionTrace。

本模块只定义理论字段与工程元数据的结构、默认值和校验，不包含任何运行算法。
数值范围与参数来自 configs/personality_core.json（版本化工程配置），均为实验初值，
按 docs/personality_core/参数校准指南.md 校准，不构成心理学断言。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# 枚举常量（9/10 号文档已冻结的取值边界）
# ---------------------------------------------------------------------------
EVENT_KINDS = frozenset({"message", "tool_result", "feedback", "system_event"})
CHANNELS = frozenset({"desktop", "qq", "tool", "system"})
EVENT_STATUSES = frozenset({"active", "invalidated", "redacted"})
PERCEPT_ITEM_KINDS = frozenset({"observed", "inferred"})
SELF_REPRESENTATION_MODES = frozenset({"current", "hoped", "feared"})
CAPS_CATEGORIES = frozenset(
    {
        "encodings",
        "expectancies_and_beliefs",
        "goals_and_values",
        "competencies_and_regulatory_plans",
        "if_then_signatures",
    }
)
CAPS_SCOPES = frozenset({"global", "relationship"})
REGULATION_STRATEGIES = frozenset(
    {
        "reappraise",
        "situation_modification",
        "attention_deployment",
        "pause_set_boundary",
    }
)
INTENT_DEFAULT = "neutral"


def new_uuid_urn() -> str:
    """v0 稳定证据 ID：urn:uuid:<uuid4>（RFC 9562，不引入 UUIDv7 依赖）。"""
    return f"urn:uuid:{uuid.uuid4()}"


def is_valid_urn_uuid(value: Optional[str]) -> bool:
    if not value or not value.startswith("urn:uuid:"):
        return False
    try:
        uuid.UUID(value[len("urn:uuid:") :])
        return True
    except (ValueError, AttributeError):
        return False


def clip(value: float, low: float, high: float) -> float:
    """把数值夹到 [low, high]。"""
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# 事件（6.1 InteractionEvent）
# ---------------------------------------------------------------------------
@dataclass
class InteractionEvent:
    """可引用的输入、结果或外部事实；UUID 负责全局引用，stream_seq 负责严格重放顺序。"""

    event_id: str
    source_id: str
    stream_id: str
    occurred_at: str
    recorded_at: str
    channel: str
    kind: str
    stream_seq: Optional[int] = None
    payload: dict = field(default_factory=dict)
    schema_version: int = 1
    actor_ref: Optional[str] = None
    relationship_ref: Optional[str] = None
    session_ref: Optional[str] = None
    source_refs: list = field(default_factory=list)
    causation_ref: Optional[str] = None
    supersedes_ref: Optional[str] = None
    status: str = "active"
    outcome: Optional[dict] = None

    @property
    def text(self) -> str:
        """事件载荷中的文本；优先 payload.text，其次 content。"""
        if not isinstance(self.payload, dict):
            return str(self.payload)
        text = self.payload.get("text") or self.payload.get("content")
        return text if isinstance(text, str) else ("" if text is None else str(text))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "InteractionEvent":
        return cls(**data)


# ---------------------------------------------------------------------------
# 感知（6 世界模型 / 10 4.1 PerceptState）
# ---------------------------------------------------------------------------
@dataclass
class PerceptItem:
    """一条感知判断：observed（直接观察）与 inferred（推测）共用同一结构。"""

    kind: str
    content: str
    source_ref: str
    confidence: float = 1.0
    construct_ref: Optional[str] = None


@dataclass
class PerceptState:
    """世界模型对事件的统一感知：观察/推断/不确定性，v0 全部由确定性规则产生。"""

    timestamp: str
    situation: str
    items: list = field(default_factory=list)
    continuity: str = "new"
    uncertainty: list = field(default_factory=list)
    situation_features: dict = field(default_factory=dict)
    parser_version: str = "rules-v0"


# ---------------------------------------------------------------------------
# 人格状态（9 §3 唯一 PersonalityState）
# ---------------------------------------------------------------------------
@dataclass
class AffectState:
    """Russell 效价-唤醒：唯一可写的连续情绪内核。"""

    valence: float = 0.0
    arousal: float = 0.0


@dataclass
class TemperamentEAS:
    """EAS 气质三维；v0 只读。"""

    emotionality: float = 0.0
    activity: float = 0.0
    sociability: float = 0.0


@dataclass
class AttachmentState:
    """依恋焦虑/回避两个连续维度；v0 只读，四型按需派生不持久化。"""

    anxiety: float = 0.0
    avoidance: float = 0.0

    def attachment_type(
        self, anxiety_threshold: float, avoidance_threshold: float
    ) -> str:
        """按显式校准阈值派生四型；阈值不在状态合同中硬编码。"""
        if not 0.0 <= anxiety_threshold <= 1.0:
            raise ValueError("anxiety_threshold 必须在 [0, 1]")
        if not 0.0 <= avoidance_threshold <= 1.0:
            raise ValueError("avoidance_threshold 必须在 [0, 1]")
        low_a = self.anxiety < anxiety_threshold
        low_v = self.avoidance < avoidance_threshold
        if low_a and low_v:
            return "secure"
        if not low_a and low_v:
            return "preoccupied"
        if low_a and not low_v:
            return "dismissive"
        return "fearful"


@dataclass
class SelfRepresentation:
    """Markus 自我表示：一类记录承载当前/希望/恐惧三种语义。

    confidence 只表示主体对该自我看法的确信度，不是事实准确率或证据权重；
    hoped/feared（可能自我）不填 confidence。
    """

    representation_id: str
    mode: str
    domain: str
    content: str
    importance: float = 0.5
    confidence: Optional[float] = None
    evidence_refs: list = field(default_factory=list)
    contexts: list = field(default_factory=list)
    activate_when: list = field(default_factory=list)


@dataclass
class CapsUnit:
    """CAPS 认知-情感单元：全局模板与关系作用域共用一类记录。

    ``scope=relationship`` 且 ``subject_ref is None`` 表示全局基线模板；
    携带 ``subject_ref`` 的同 ``unit_id`` 记录只保存该关系下的激活值。
    """

    unit_id: str
    category: str
    content: str
    activation: float = 0.0
    baseline: float = 0.0
    input_weight: float = 0.0
    feature_ref: Optional[str] = None
    scope: str = "global"
    subject_ref: Optional[str] = None
    updated_at: Optional[float] = None


@dataclass
class IfThenSignature:
    """if-then 行为签名：条件行为剖面（受控情境契约），不是手写规则库。

    conditions 定义契约作用域（[{feature, min?, max?}]），全部满足才算该签名适用；
    例如“安慰”契约要显式排除威胁情境，避免与 care 签名冲突。
    """

    signature_id: str
    feature_refs: list = field(default_factory=list)
    expected_intent: Optional[str] = None
    conditions: list = field(default_factory=list)
    supporting_refs: list = field(default_factory=list)
    refuting_refs: list = field(default_factory=list)
    last_validated_at: Optional[str] = None
    config_revision: Optional[str] = None


@dataclass
class CapsState:
    """CAPS 五类单元（9 §5）；不建立每类一个 Manager。"""

    encodings: list = field(default_factory=list)
    expectancies_and_beliefs: list = field(default_factory=list)
    goals_and_values: list = field(default_factory=list)
    competencies_and_regulatory_plans: list = field(default_factory=list)
    if_then_signatures: list = field(default_factory=list)

    def all_units(self) -> list:
        return (
            list(self.encodings)
            + list(self.expectancies_and_beliefs)
            + list(self.goals_and_values)
            + list(self.competencies_and_regulatory_plans)
        )

    def unit_map(self, subject_ref: Optional[str] = None) -> dict:
        """返回全局单元，并用当前关系的同 ID 单元覆盖关系模板。"""
        units = {
            unit.unit_id: unit for unit in self.all_units() if unit.subject_ref is None
        }
        if subject_ref is not None:
            subject_key = str(subject_ref)
            units.update(
                {
                    unit.unit_id: unit
                    for unit in self.all_units()
                    if unit.subject_ref == subject_key
                }
            )
        return units


@dataclass
class PersonalityState:
    """唯一人格状态：时间尺度是字段属性，不拆快/中/慢三套状态库。"""

    affect: AffectState = field(default_factory=AffectState)
    temperament_eas: TemperamentEAS = field(default_factory=TemperamentEAS)
    attachment: AttachmentState = field(default_factory=AttachmentState)
    self_representations: list = field(default_factory=list)
    caps: CapsState = field(default_factory=CapsState)
    updated_at: float = 0.0
    config_version: Optional[str] = None


# ---------------------------------------------------------------------------
# 决策（6.2 DecisionTrace）
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    """本轮派生输出：只进 trace 用于审计，不是第二套人格状态。"""

    working_self_refs: list = field(default_factory=list)
    empathy_intensity: float = 0.0
    attention_hint: Optional[str] = None
    va_delta: dict = field(default_factory=lambda: {"valence": 0.0, "arousal": 0.0})
    expression_intensity: float = 0.0
    expression_constraints: list = field(default_factory=list)
    emotion_label: str = "neutral"


@dataclass
class DecisionTrace:
    """本轮前后状态、候选分数、门控与版本信息。"""

    trace_id: str
    event_id: Optional[str]
    state_revision_before: Optional[str]
    state_revision_after: Optional[str]
    percept_state: dict = field(default_factory=dict)
    retrieved_refs: list = field(default_factory=list)
    candidate_scores: dict = field(default_factory=dict)
    gates: list = field(default_factory=list)
    selected_intent: str = INTENT_DEFAULT
    selected_regulation: Optional[str] = None
    decision: Decision = field(default_factory=Decision)
    model_version: Optional[str] = None
    prompt_version: Optional[str] = None
    config_version: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 序列化与校验
# ---------------------------------------------------------------------------
def state_to_dict(state: PersonalityState) -> dict:
    return asdict(state)


def percept_to_dict(percept: PerceptState) -> dict:
    """序列化完整感知快照，供 trace 与冻结回放共用。"""
    return asdict(percept)


def percept_from_dict(data: dict) -> PerceptState:
    """从持久化快照恢复 PerceptState，不重新运行词法解析器。"""
    data = dict(data)
    data["items"] = [PerceptItem(**item) for item in data.get("items", [])]
    return PerceptState(**data)


def state_from_dict(data: dict) -> PersonalityState:
    data = dict(data)
    data["affect"] = AffectState(**data.get("affect", {}))
    data["temperament_eas"] = TemperamentEAS(**data.get("temperament_eas", {}))
    data["attachment"] = AttachmentState(**data.get("attachment", {}))
    data["self_representations"] = [
        SelfRepresentation(**item) for item in data.get("self_representations", [])
    ]
    caps = data.get("caps", {}) or {}
    data["caps"] = CapsState(
        encodings=[CapsUnit(**item) for item in caps.get("encodings", [])],
        expectancies_and_beliefs=[
            CapsUnit(**item) for item in caps.get("expectancies_and_beliefs", [])
        ],
        goals_and_values=[
            CapsUnit(**item) for item in caps.get("goals_and_values", [])
        ],
        competencies_and_regulatory_plans=[
            CapsUnit(**item)
            for item in caps.get("competencies_and_regulatory_plans", [])
        ],
        if_then_signatures=[
            IfThenSignature(**item) for item in caps.get("if_then_signatures", [])
        ],
    )
    return PersonalityState(**data)


def validate_state(state: PersonalityState) -> list:
    """返回不变量违规列表；空列表表示合法。"""
    issues = []
    for axis, value in (
        ("valence", state.affect.valence),
        ("arousal", state.affect.arousal),
    ):
        if not -1.0 <= value <= 1.0:
            issues.append(f"affect.{axis} 越界: {value}")
    for name, value in (
        ("emotionality", state.temperament_eas.emotionality),
        ("activity", state.temperament_eas.activity),
        ("sociability", state.temperament_eas.sociability),
        ("anxiety", state.attachment.anxiety),
        ("avoidance", state.attachment.avoidance),
    ):
        if not 0.0 <= value <= 1.0:
            issues.append(f"{name} 越界: {value}")
    seen_caps_units = set()
    for unit in state.caps.all_units():
        scoped_key = (unit.unit_id, unit.subject_ref)
        if scoped_key in seen_caps_units:
            issues.append(f"caps.{unit.unit_id} subject_ref 重复: {unit.subject_ref}")
        seen_caps_units.add(scoped_key)
        if not -1.0 <= unit.activation <= 1.0:
            issues.append(f"caps.{unit.unit_id}.activation 越界: {unit.activation}")
        if unit.scope not in CAPS_SCOPES:
            issues.append(f"caps.{unit.unit_id}.scope 非法: {unit.scope}")
        if unit.scope != "relationship" and unit.subject_ref is not None:
            issues.append(f"caps.{unit.unit_id} 非关系单元不应携带 subject_ref")
    for rep in state.self_representations:
        if rep.mode not in SELF_REPRESENTATION_MODES:
            issues.append(
                f"self_representation {rep.representation_id} mode 非法: {rep.mode}"
            )
        if rep.confidence is not None and not 0.0 <= rep.confidence <= 1.0:
            issues.append(
                f"self_representation {rep.representation_id} confidence 越界"
            )
        if not 0.0 <= rep.importance <= 1.0:
            issues.append(
                f"self_representation {rep.representation_id} importance 越界"
            )
    return issues


def assert_valid_state(state: PersonalityState) -> None:
    issues = validate_state(state)
    if issues:
        raise ValueError("人格状态不变量违规: " + "; ".join(issues))


def serialize_json(obj: Any) -> str:
    return json.dumps(
        asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj, ensure_ascii=False
    )
