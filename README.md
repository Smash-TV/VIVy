# VIVy_0_Open

这是第一版人格内核失败基线的公开整理副本，源自作者自有的 Gwen_Agent 工程。核心研究目标尚未得到验证；它适合研究和复现实验，不是已验证的人格产品。

2026.10.22 起，公开副本使用原创角色 **澄音**，详见 [原创人设](docs/原创人设_澄音.md)。运行配置版本为 `2026.10.22-open-persona-v1`。内核计算机制保持不变，身份词表和测试输入已适配；数值仍为实验初值。

## 运行

需要 Python 3.9 或更高版本，核心无第三方运行依赖。请从源码克隆目录运行；配置与文档随源码提供，不建议脱离源码目录使用普通 wheel 安装。

```powershell
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS 使用 source .venv/bin/activate
python -m pip install -e ".[test]"
```

在克隆目录运行：

```powershell
python scripts/vivy_cli.py inspect
python scripts/vivy_cli.py preview "我今天好难过，压力很大"
python scripts/vivy_cli.py run "澄音，帮我看看这个怎么做"
```

运行时会创建空的 `data/personality_core.db`。`preview` 不提交事件；`run` 会写入本地状态。公开包不提供原始交互数据库或恢复功能。

## 验证

```powershell
python scripts/verify_snapshot.py
python scripts/evaluate_personality_core.py
python -m pytest -q
```

测试需要安装 `pytest`。48 个手写样例只是 bootstrap 合同检查，不证明人格稳定性或泛化能力。校验清单对应公开副本；数据库清单为空，不能用它证明原始证据完整。

## 公开范围

保留核心代码、理论说明、原创人设、适配后的测试与实验脚本。Git 历史、私有日志、交互数据库、旧角色配置快照、原始集成材料及旧角色实验输出不公开。历史报告只保留撤下说明，不能作为澄音的验证结果。

`src/gwen_agent/` 包名是自有工程沿用的接口标识。源代码不自动加载人设文档，也不自行生成自然语言；宿主可显式引用人设，并在获得有效回复后提交结果。

```python
from gwen_agent.modules.personality_core import PersonalityCore

core = PersonalityCore(config_path="configs/personality_core.json", db_path="data/personality_core.db")
```

[公开整理说明](docs/公开整理说明.md) · [来源声明](provenance/OWNERSHIP.md) · [理论文档](docs/personality_core/README.md)

## 许可

以 [MIT](LICENSE) 许可发布，Copyright (c) 2026 Smash-TV。来源和第三方引用见 [声明](THIRD_PARTY_NOTICES.md)。
