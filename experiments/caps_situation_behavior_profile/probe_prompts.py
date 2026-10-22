# -*- coding: utf-8 -*-
"""使用公开副本配置探测原创角色场景的特征与意图，不写数据库。"""
import pathlib
import sys, json, os
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from gwen_agent.modules.personality_core import load_config, perceive, build_genesis_state, reduce
from gwen_agent.modules.personality_core.state import InteractionEvent, new_uuid_urn, state_to_dict

BASE = str(PROJECT_ROOT / 'configs' / 'personality_core.json')
cfg = load_config(BASE)

POOL = [
    # (class, text)
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
    ('intimacy', '我最近心里很难受，跟你说个秘密，只有你。'),
    ('intimacy', '我心里有个小秘密，只告诉你一个人。'),
    ('pressure', '你现在必须马上做，别磨蹭！'),
    ('pressure', '立刻给我答案，快点！'),
    ('confirm', '你爱我吗？你会一直陪我吗？'),
    ('confirm', '你是不是喜欢我？我们是什么关系？'),
    ('surprise', '居然收到了 offer，真的没想到！'),
    ('surprise', '突然收到面试邀请，太意外了。'),
]

def run(text):
    ev = InteractionEvent(event_id=new_uuid_urn(), source_id='probe', stream_id='probe', stream_seq=None,
        occurred_at='2026-08-21T10:00:00+08:00', recorded_at='2026-08-21T10:00:00+08:00',
        channel='system', kind='message', payload={'text': text})
    p = perceive(ev, cfg)
    st0 = build_genesis_state(cfg)
    st1, tr = reduce(st0, p, 0.0, cfg, ev)
    return p.situation_features, tr.selected_intent, tr.selected_regulation, tr.decision.emotion_label

for cls, text in POOL:
    feats, intent, reg, emo = run(text)
    active = {k: round(v,2) for k,v in feats.items() if abs(v) > 0.05}
    print('%-12s | %-30s | active=%s' % (cls, text, active))
    print('            -> intent=%s reg=%s emo=%s' % (intent, reg, emo))
