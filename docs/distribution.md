# EEG QC 同事试用说明

这是 BrainElecQC 0.1.0 的 EEG 交付包（2026-10-27）。输入 EEG 文件，输出自动初评、定位证据和人工复核报告。先看示例报告，再按下方步骤运行自己的文件。

## 现在能做什么

| 模块 | 主要输出 | 人工复核 |
| --- | --- | --- |
| 导入、读取与头文件 | 通道、单位、采样率、时间、标定和完整性 | 核对采集与导出信息 |
| 坏道与连续性 | 幅度、高频、低相关、短平线与平台候选 | 确认整道或局部区间 |
| 工频与频谱 | 50/60 Hz、PSD、背景拟合、α 峰及能量 | 结合波形与记录状态 |
| ECG / EOG / EMG | 事件锁时、同步影响与具体范围 | 对照辅助导联 |
| 自动初评与报告 | 整数分数、扣分理由、证据图、人工填写栏 | 填写调整理由与最终评价 |
| Python / Linxi | 可独立调用的函数、命令行和 Linxi 自动任务 | 打开本次输入对应的报告 |

当前计算入口每次分析一个连续段。多段自动汇总、人工复评分回读与审核提交列入下一版；本版可先完成自动报告与人工填写。函数明细见 [模块与接口](portability.md)。

## 交付包目录

| 内容 | 用途 |
| --- | --- |
| `README.html` / `README.md` | 这份使用说明 |
| `wheels/` | EEG 核心包、可选 Linxi 插件 |
| `requirements-core.txt` / `requirements-linxi.txt` | 固定主要依赖的安装清单 |
| `examples/` | QC 参数、Linxi YAML 和运行脚本 |
| `example-report/` | 公开 EEG 的本版报告示例 |
| `source/` | 对应源码、详细文档与测试 |
| `validation.json` / `SHA256SUMS.txt` | 本次验证结果与文件校验值 |

## 1. 安装 EEG 核心

本次实测环境为 macOS Apple Silicon、Python 3.12.13。准备 **Python 3.12**，解压后在交付包根目录打开终端。安装时联网下载依赖。

macOS / Linux：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m pip check
```

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.venv\Scripts\python.exe -m pip check
```

下文的 `python` 指这个环境的解释器；Windows 对应 `.venv\Scripts\python.exe`。只做独立 EEG QC 时，安装到这里即可。

## 2. 运行自己的文件

示例分析原文件前四个 EEG 通道的 60 秒数据；请把通道索引换成实际 EEG 通道：

```bash
python -m brainelecqc qc-eeg "/path/to/record.edf" --eeg-channels 0,1,2,3 --duration 60 --output "../qc-results/record-01"
```

把输入换成实际文件，输出选一个新目录。可用 `--start 100 --duration 60` 查看其他时段；`--eeg-channels 0,1,2,3` 指定原文件通道索引。参数文件通过 `--config examples/qc_config.json` 传入。

EDF/BDF、BrainVision 使用包内读取器；BrainVision 输入包含配套文件的目录。FIF/EEGLAB 等通过 MNE 读取。逐格式状态见 [格式说明](formats.md)。

通道比较按实际采集参考分组，配置字段为 `method_config.comparison_groups`，列表顺序与所选 EEG 通道一致。缺少辅助导联或比较条件时，报告保留未评分原因。

## 3. 阅读与填写报告

1. 打开结果目录的 `review_report.md`：先看初评、评分覆盖和质量总结。
2. 按表格右侧“人工复核”列查看原始波形、频谱及辅助导联。
3. 在 `score_review.tsv` 填写分项调整分数与理由，在报告末尾填写最终评价、使用范围、审核人与日期。

`score_report.md` 列出评分标准；`quality_score.json` 和 `manifest.json` 供程序读取。PNG 可直接打开，Markdown 报告用支持 Markdown 的编辑器预览。

## 4. 可选：接入 Linxi

使用 **Linxi 1.0.1**，在同一虚拟环境、交付包根目录运行：

```bash
python -m pip install -r requirements-linxi.txt
python examples/run_linxi.py "/path/to/record.edf" "../qc-results/linxi-record-01"
```

默认使用 `examples/linxi_eeg_qc.yml`，示例选择前四个通道和前 60 秒；按采集信息修改通道与比较组后再运行。处理器注册在 `quality` 阶段，结束后打印报告路径。脚本先导入 Linxi，再载入计算库。

本次 Linxi 启动有可选排序器 `mountainsort5` 缺失提示，EEG 步骤正常完成；该提示与 EEG QC 运行无关。

## 进一步使用

模块调用与结果字段见 [模块与接口](portability.md)，公式和评分见 [方法说明](eeg-methods.md)，后续开发见 [实施计划](../overallplan.md)。交付测试详情保存在包内 `validation.json`。
