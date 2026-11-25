# SEEG QC 使用说明

SEEG QC 在原采样率下检查样本完整性、局部平线、量程端点驻留、50/60 Hz 干扰、逐通道频谱和疑似波形跳变、漂移、平台与恢复。初评分反映技术风险。量程端点驻留、波形异常和跨通道重复信号是复核线索。病理活动、刺激效应和解剖位置另行评估。

## 1. 扫描记录

```bash
python -m pip install -e '.[qc]'
SEEG_QC_STORE=/path/to/private-store
python -m brainelecqc --storage-root "$SEEG_QC_STORE" qc-seeg /path/to/record.edf --output /path/to/new-scan
```

命令返回 `run_id`。扫描目录中的 `qc_result.json` 保存输入指纹、通道映射、数据流和连续段、逐窗证据、频谱、波形事件、技术初评、覆盖范围和当前复核。每次扫描得到独立运行。扫描与复核期间原输入保留为只读快照。存储和结果目录都选在 Git 工作区外。

已有记录可用 `qc-seeg --artifact-id ARTIFACT_ID --output /path/to/new-scan`。输入文件采用 [格式说明](formats.md) 中对应的读取器。多文件或专有设备格式可用 `--format-hint snapshot/relative/path=FORMAT`。

默认按 5 秒取窗，检查 50 Hz。可用 `--config /path/to/scan-config.json` 指定参数和映射。用扫描报告或 Python 的 `describe_artifact(artifact_id)` 取得以下示例所需的 ID。长记录超过资源预算时，缩短 `ranges`。每个范围都落在同一数据流的一个连续段内，样本区间采用 `[start_sample, stop_sample)`。

```json
{
  "config": {
    "line_frequency": 60,
    "window_seconds": 5,
    "ranges": [
      {"stream_id": "STREAM_ID", "segment_id": "SEGMENT_ID", "start_sample": 0, "stop_sample": 2500}
    ]
  },
  "channel_map": [
    {"channel_id": "CHANNEL_ID", "status": "confirmed", "channel_type": "seeg"}
  ]
}
```

映射中的确认值来自采集通道表。报告保留原始名称、独立通道 ID 和候选解释。不同采样率分别读取，断点保留原时钟。

## 2. 复核原始样本

从 `report.windows` 或 `report.waveform_events` 选择证据区间，按同一实际通道、连续段和范围查看电压。Python 可读取证据区间的原始物理值。

```python
from brainelecqc.config import storage_context
from brainelecqc.db import init_db
from brainelecqc.seeg import get_run
from brainelecqc.windows import samples

with storage_context("/path/to/private-store"):
    init_db()
    run = get_run("RUN_ID")
    window = run["report"]["windows"][0]
    meta, volts = samples(
        run["artifact_id"],
        window["stream_id"],
        window["segment_id"],
        [window["channel_id"]],
        window["start_sample"],
        window["stop_sample"],
        "physical",
    )
```

填写扫描生成的 `review.json`。下面是对一个波形候选对应时段的示例决定。用报告中的实际 ID 和样本位置。

```json
{
  "base_revision": 0,
  "reviewer": "审核者",
  "decisions": [
    {
      "channel_id": "CHANNEL_ID",
      "stream_id": "STREAM_ID",
      "segment_id": "SEGMENT_ID",
      "scope": "interval",
      "start_sample": 100,
      "stop_sample": 220,
      "action": "exclude",
      "reason": "检查原始波形后排除该段",
      "evidence_window_ids": ["WINDOW_ID"]
    }
  ]
}
```

- `scope` 可选 `interval`（一个连续段内）或 `channel`（整道）。区间决定优先于整道决定。
- `action` 可选 `keep`、`exclude` 或 `pending`。未决定的区间记为未知。
- `response` 可用于裁决技术问题。由对应证据窗支持的 `confirmed` 配 `exclude`，`rejected` 配 `keep`。留空时只记录样本使用决定，原技术风险评分仍保留。

```bash
python -m brainelecqc --storage-root "$SEEG_QC_STORE" review-seeg RUN_ID --review /path/to/new-scan/review.json
python -m brainelecqc --storage-root "$SEEG_QC_STORE" show-seeg RUN_ID
```

保存时提交完整决定列表。之后用 `show-seeg` 取得当前 `review.revision` 和已有决定的 `decision_id`，在下一次提交中带上这些值。Python 的 `get_audit(run_id)` 返回已保存版本及复核者。通过 `brainelecqc.seeg.run_seeg_qc(artifact_id)` 也可从已导入记录开始扫描。

## 3. 冻结并导出

```bash
python -m brainelecqc --storage-root "$SEEG_QC_STORE" freeze-seeg RUN_ID --base-revision 1 --reviewer "审核者"
python -m brainelecqc --storage-root "$SEEG_QC_STORE" export-seeg RUN_ID --output /path/to/new-seeg-qc.zip
```

`--base-revision` 填要冻结的当前版本。冻结时复核输入指纹并校验 ZIP 读回。ZIP 包含 `report.html`、结构化结果、通道表、决定与证据窗表、波形事件及带成员 SHA-256 的 `manifest.json`。报告显示技术初评、复评、覆盖范围和未知时长。后续意见从一次新扫描开始。

SEEG 算法与复核规则沿用 Cortrix（VISVB）的基础技术 QC。见 [Cortrix 接入](cortrix-seeg-qc.md) 和 [Apache 2.0 许可](licenses/Apache-2.0.md)。
