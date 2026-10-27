# EEG 模块与 Linxi 使用

**可以打包接入 Linxi，现有 `BrainElecQCProcessor` 已接通 EEG 自动 QC。** Python 核心负责检测、评分和报告；Linxi 负责提供输入、运行任务并取得结果。人工审核保存与复评分列在后续接入中。

## 模块与函数

| 模块 | 现有函数 | 输出 | 人工复核 |
| --- | --- | --- | --- |
| 文件导入与读取 | [`import_recording()`](../src/brainelecqc/recordings.py)、[`read_eeg_input()`](../src/brainelecqc/eeg/input.py) | 记录身份、电压数组、通道、采样率、原始时钟、辅助导联 | 核对输入文件、通道类型、参考和分析范围 |
| 头文件检查 | [`inspect_input_header()`](../src/brainelecqc/eeg/header_qc.py) | 格式、通道表、采样率、单位标定、长度及时间轴检查 | 对照采集与导出信息 |
| 坏道检测 | [`compute_bad_channels()`](../src/brainelecqc/eeg/bad_channels.py) | 稳健幅度、高频噪声、低相关及具体区间 | 对照原始波形、同组通道和频谱 |
| 连续性检查 | [`compute_continuity_qc()`](../src/brainelecqc/eeg/continuity.py) | 近零恒值、标定端点平台、其他恒值和原始边界 | 确认起止、恢复和使用范围 |
| 工频与频谱 | [`compute_spectral_qc()`](../src/brainelecqc/eeg/spectral.py) | 50/60 Hz、PSD、背景拟合、α 峰、功率与占比 | 结合记录状态检查谱形、峰频及异常通道 |
| ECG / EOG | [`compute_event_qc()`](../src/brainelecqc/eeg/events.py) | 事件位置、锁时波形、重复性 | 核对辅助导联检出与 EEG 同步影响 |
| 肌电相关高频 | [`compute_muscle_qc()`](../src/brainelecqc/eeg/muscle.py) | 宽带高频、EMG 同步、候选与有效范围 | 对照 EEG、EMG 与采集带宽 |
| 整数初评 | [`compute_quality_score()`](../src/brainelecqc/eeg/scoring.py) | 分项分数、扣分理由、证据和检测覆盖 | 填写调整分数与理由 |
| 图与报告 | [`make_paper_eeg_figures()`](../src/brainelecqc/plotting/paper_eeg.py)、[`render_automatic_report()`](../src/brainelecqc/eeg/automatic_report.py)、[`render_score_review()`](../src/brainelecqc/eeg/score_report.py) | 图对象、主报告、评分表和最终评价栏 | 逐项看证据，填写最终使用意见 |
| 整体运行与导出 | [`run_eeg_qc()`](../src/brainelecqc/eeg/pipeline.py)、[`export_eeg_qc()`](../src/brainelecqc/eeg/export.py) | 完整结果包、报告和文件引用 | 确认输入、运行版本与报告对应 |

**示例：** T8 长时间高频异常进入坏道复核；F8 56–57 s 低相关进入局部区间复核。两项都能回到原始波形和 PSD。

各检测函数可独立调用，计算与绘图分开。坏道、连续性、频谱、事件和肌电函数共用四个基本输入：`data_volts`（通道×样本、V）、`sfreq`（Hz）、`channel_ids`、`channel_names`。通过 `start_sample`、`start_seconds` 保留原始位置，通过 `config` 保存方法参数；辅助导联与 EEG 使用同一时间轴。坏道比较组按实际参考填写。

整体入口组织检测依赖、评分与导出；例如将连续性区间交给肌电模块作为排除范围。独立调用时由调用者传入相同上下文。

## 运行第一版处理器

插件在 Linxi 的 `quality` 阶段注册，依赖 Linxi 1.0.1、Python 3.12+。在仓库根目录安装：

```bash
python -m pip install -e '.[qc,figures,readers]' -e plugins/linxi
```

将配置存为仓库外的 `eeg_qc.yml`。示例选取前四个 EEG 通道和前 60 秒，通道索引及比较组按实际采集填写：

```yaml
name: eeg_qc
linxi_plugin: [linxi_brainelecqc]
steps:
  - stage: quality
    processor_name: BrainElecQCProcessor
    params:
      qc_config:
        selection:
          start_seconds: 0
          duration_seconds: 60
          eeg_indices: [0, 1, 2, 3]
        method_config:
          comparison_groups: [reference_A, reference_A, reference_A, reference_A]
      figures: true
```

```python
from datetime import datetime, timezone
from pathlib import Path
from linxi.fabric.linxi_context import LinxiContext
from linxi.fabric.task_pipeline import PipelineDefinition
from linxi.fabric.task_runner import TaskRunner

context = LinxiContext(file_create_date=datetime.now(timezone.utc))
context.run_state.input_path = "/path/to/record.edf"
context.run_state.work_dir = Path("/path/to/new-task")
pipeline = PipelineDefinition.from_file(Path("/path/to/eeg_qc.yml"))
TaskRunner(pipeline).run(context, skip_dry_run=False, allow_errors=False)
print(context.get_metric("brainelecqc")["report_path"])
```

工作目录放在仓库与输入目录之外。Linxi 1.0.1 在 macOS 上先导入 Linxi，再导入 NumPy/MNE/QC，以上示例遵循此顺序。

## 打包与宿主交付

核心 `brainelecqc` 与插件 `linxi-brainelecqc` 各构建一个 wheel，安装到同一宿主环境。核心保留独立 Python/CLI 用法；插件只转换参数、任务状态与结果引用。

```bash
python -m pip wheel '.[qc,figures,readers]' --no-deps --wheel-dir /path/to/wheels
python -m pip wheel ./plugins/linxi --no-deps --wheel-dir /path/to/wheels
```

| 接点 | 当前行为 |
| --- | --- |
| `processor.py` | 注册 `BrainElecQCProcessor`，接收配置并处理任务 |
| `adapter.py: process_file()` | 调用导入、`run_eeg_qc()`、`export_eeg_qc()` |
| `context.get_metric("brainelecqc")` | 返回 `report_path`、`bundle_dir`、评分、覆盖、图与清单路径 |
| `context.get_metric("brainelecqc_status")` | 返回导入、分析、导出进度及完成/失败/取消状态 |
| `review_status / review_id` | 当前为 `pending / None`，人工保存接口待接入 |

输入副本位于 `work_dir/brainelecqc/store`，各次运行位于 `work_dir/brainelecqc/runs/<run_id>`。相同输入复用记录身份时核对内容指纹，每次分析保留独立结果目录。

## 单独调用与结果文件

`read_eeg_input()` 和 `run_eeg_qc()` 支持 `stream_id`、`segment_id`、`start_seconds`、`duration_seconds`、`eeg_indices`、`reference_indices`。一次选择一个连续段，分析数组上限为 256 MiB；长文件的分段调度与文件级汇总见 [实施计划](../overallplan.md)。

`run_eeg_qc()` 返回原始电压、辅助导联、通道、特征、方法结果与输入清单；`compute_quality_score(result)` 汇总评分；`export_eeg_qc(result, output_dir)` 导出报告及结果包。最短文件调用见 [README](../README.md#快速开始)。

| 文件 | 用途 |
| --- | --- |
| `review_report.md`、`score_report.md`、PNG | 人类阅读、证据定位和分项核对 |
| `quality_score.json`、`quality_overview.tsv` | 自动分数、规则版本、已评权重与核心指标 |
| `score_review.tsv`、`score_evidence.tsv` | 人工填写表、原始位置及扣分证据 |
| `paper_methods.json`、各方法 TSV | PSD、谱峰、坏道、工频、连续性、事件及肌电测量 |
| `manifest.json`、`channels.tsv` | 输入指纹、版本、通道顺序、时间范围与文件校验 |
| `review_signals.npz`、`keep_mask.npy`、`retained_intervals.tsv` | 原始信号、保留掩码与区间 |

当前人工表用于填写与保存复核记录。表格回读、调整后得分、审核版本和下游发布处理器是下一阶段交付。

## 开发与验收入口

```bash
python -m pytest tests -q
python -m pytest plugins/linxi/tests -q
```

真实方法对照运行 `scripts/validate_eeg_papers.py`；宿主任务验收运行 `plugins/linxi/validate_real_eeg.py`。验收核对原始电压、通道和时间、指标数值、报告链接与文件校验值，结果保存在仓库外。发布 wheel 时对本次构建运行宿主验收。
