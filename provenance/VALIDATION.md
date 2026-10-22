# 冻结包验证记录

验证时间：2026.8.30（Asia/Shanghai）

验证环境：Windows 11，Python 3.13.5（项目原 `.venv`），pytest 8.3.4。

## 最小运行链

- `vivy_cli.py preview`：成功解析 v5 配置、生成 PerceptState、VA/CAPS 状态和 DecisionTrace，未写数据库。
- `vivy_cli.py run`：成功写入独立 smoke SQLite，事件、状态修订、决策轨迹各 1 条。
- `vivy_cli.py inspect`：SQLite `integrity_check=ok`，新库冻结回放 `replay_ok=true`、`mismatch_count=0`。
- `evaluate_personality_core.py`：48/48 bootstrap 合同命中；报告仍明确给出 `stage_gate.passed=false`。

## 自动测试

执行命令：

```powershell
D:\ProgramData\Gwen_Agent\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

结果：`75 passed in 5.16s`。

测试覆盖配置契约、确定性 reducer、bootstrap 评测、完整提交链、幂等、CAS、冻结回放、跨流顺序、旧库审计、线程并发和双进程 SQLite 写入。通过这些工程测试不代表阶段 2 人格研究门禁通过。

测试在普通 Windows 文件权限下运行。Codex 沙箱会给 `tempfile.mkdtemp()` 创建的随机子目录附加异常 ACL，因此首次沙箱内测试出现 `sqlite3.OperationalError: unable to open database file`；在沙箱外按相同代码复跑后全部通过，确认该错误不是冻结代码故障。

