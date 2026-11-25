# BrainElecQC

面向 **EEG 和 SEEG** 的信号质量检查 Python 项目。EEG 支持 Linxi 插件和图文报告。SEEG 支持独立命令行，用于技术筛查、按样本复核和冻结导出。

流程：**读取记录 → 自动检测与初评 → 人工复核 → 审核结果 → 下游分析**。

当前入口完成单连续段 EEG 自动 QC：检查头文件、坏道、短平线与平台、工频与频谱、ECG/EOG 事件关系和肌电相关高频，生成整数初评、波形证据和复核报告。文件报告与多段组织见 [QC 设计](docs/qc-design.md)。

## 快速开始

在仓库根目录、Python 3.10+ 环境运行：

```bash
python -m pip install -e '.[qc,figures]'
python -m brainelecqc qc-eeg /path/to/record.edf --eeg-channels 0,1,2,3 --duration 60 --output /path/to/qc-result
```

示例选择原文件前四个 EEG 通道的 60 秒数据；按实际通道索引、文件和仓库外的新输出目录修改。输入副本默认保存在 `~/.local/share/brainelecqc`，可用 `python -m brainelecqc list` 查看记录。

常用选项：

| 需求 | 参数 |
| --- | --- |
| 指定 EEG 通道 | `--eeg-channels 0,1,2,3`，使用原文件索引并保持此顺序 |
| 指定计算范围 | `--start 0 --duration 120`，一次选择一个连续段 |
| 放大波形窗口 | `--plot-start 32 --plot-duration 14`，独立于计算范围 |
| 设置比较组和方法参数 | `--config /path/to/config.json` |

EDF/BDF、BrainVision 使用包内读取器；BrainVision 导入包含 `.vhdr/.vmrk/.eeg` 的完整目录。FIF、EEGLAB 等扩展读取安装 `python -m pip install -e '.[readers]'`。逐格式入口与校验范围见 [格式说明](docs/formats.md)。

## SEEG 扫描与复核

```bash
python -m pip install -e '.[qc]'
python -m brainelecqc --storage-root /path/to/private-store qc-seeg /path/to/record.edf --output /path/to/seeg-scan
```

打开 `qc_result.json` 查看通道映射、逐窗指标、频谱、波形候选和技术初评。填写同目录的 `review.json`，用 `review-seeg` 保存，用 `freeze-seeg` 冻结当前版本，再用 `export-seeg` 导出含报告和校验清单的 ZIP。命令及按原始样本查看证据的方法见 [SEEG QC 使用说明](docs/seeg-qc.md)。

## EEG 看图与复核

打开结果目录中的 **`review_report.md`**，先看质量总结中的异常通道占比、异常时间占比和整数初评，按右侧“人工复核”列查看证据，最后填写文件评价。图片下的链接打开 **`score_report.md`**，其中列出分项评分、扣分标准和人工调整栏。`quality_overview.tsv` 保存同一份核心指标表。

图中分别回答这些问题，报告列出对应的通道、时间与分数：

| 图 | 回答什么 |
| --- | --- |
| `03_psd_alpha` | 哪些频率突出？关键通道的 α 功率（µV²）和 α 占比是多少？ |
| `08_alpha_time` | 补充查看：定位 α 能量变化的时段，结合状态标记与原始波形核对。 |
| `03_all_channels_psd` | 全部 EEG 通道中，哪些频谱明显不同？ |
| `02_bad_channels` 及其分类页 | B 宽频 PSD、C 原始波形：高频噪声、近零平线与量程平台在哪里？ |
| `04_line_noise` | 50 Hz 还是 60 Hz？哪些通道、哪些窗口命中？ |
| `05_ecg`、`06_eog` | 一组上下波形：上方 EEG、下方辅助导联，核对几个连续事件处的同步变化；单事件用方形图。标题标明锁时事件，轴标签标明导联。 |
| `10_muscle` | 原始 EEG、EMG 与高频包络是否同步？ |

自动初评采用整数百分制，可评项目按固定权重折算，检测覆盖和未评分原因分别列出。人工核对、调整已有初评。图中 `-Ref` 表示原文件的采集参考，结果保留原始名称和参考分组。

[异常占比与整数分档](docs/eeg-methods.md#6-自动整数评分与人工复核)：1.3% 短平线得 9/10，2/64 坏道得 18/20；完整头文件独立计 10 分。坏道图用红色系显示全部自动坏道，并配同参考组对照。

EEG 自动结果保留全部样本并标记为待审核，审核保存与下游发布按 [实施计划](overallplan.md) 接入。计算与参数见 `methods.md`；程序读取 `quality_score.json`、`manifest.json` 和通道、时间表。图片输出 PNG。

重绘已有结果：

```bash
python scripts/figure_eeg_qc.py /path/to/qc-result /path/to/new-figures
```

同一受试者有睁眼、闭眼记录时，可将两份结果中的 O1/Oz/O2 放到同一张对照图：

```bash
python scripts/figure_alpha_comparison.py /path/to/eyes-open-result /path/to/eyes-closed-result /path/to/new-comparison
```

## 按需阅读

| 要做什么 | 入口 |
| --- | --- |
| 给同事安装试用版 | [交付包使用说明](docs/distribution.md) |
| 看模块函数、单独调用、把结果交给下游 | [QC 模块与接口](docs/portability.md) |
| 在 Linxi 中运行 | [安装、YAML 与最短调用](docs/portability.md#运行第一版处理器)，使用 Python 3.12+ |
| 理解每文件报告、EEG 检查项和人工复核 | [QC 设计](docs/qc-design.md) |
| 运行 SEEG 技术筛查、保存复核和导出冻结版 | [SEEG QC 使用说明](docs/seeg-qc.md) |
| 核对公式、论文、阈值与适用条件 | [EEG 方法说明](docs/eeg-methods.md) |
| 查看后续工作和验收方式 | [overallplan](overallplan.md) |
| 开发、运行测试、检查提交内容 | [开发规范](CONTRIBUTING.md)、[验收入口](docs/portability.md#开发与验收入口) |

Git 保存代码、通用文档和测试信号生成程序；真实记录、测试数据、标注、报告、图和日志保存在仓库外。
