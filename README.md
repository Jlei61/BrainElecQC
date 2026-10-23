# BrainElecQC

面向 **EEG、SEEG、ECoG** 的独立电生理 QC 项目：保真读入 → 自动候选 → 人工复核 → 可追溯交付。

当前版本完成 **S0–S2 的基础迁移**：稳定 recording 身份、不可覆盖输入快照、分段原始波形读取、明确的单位/通道/时钟合同、派生版本与二进制传输。**尚未完成生产级检测、人工复核工作台和正式 QC 发布闭环。** 图形脚本的阈值只是验收演示，不能当经过验证的检测器。

本项目不依赖原平台数据库或目录。输入只有电生理文件和显式元数据；没有 ICA、CT/MRI 联动或依赖电极坐标的网络分析。Neuropixels 本轮不做。相对电极名称不等于已知空间邻接。

## 先看这些

- [overallplan.md](overallplan.md)：完整阶段、当前进度、复用/新增内容、逐项验收。
- [QC 设计](docs/qc-design.md)：三种模态检查什么、怎么人工复核、界面交互和下游合同。
- [格式支持矩阵](docs/formats.md)：正式边界、测试等级与待验证格式。
- [可迁移性](docs/portability.md)：独立核心与未来 Linxi 适配边界。
- [开发与提交规范](CONTRIBUTING.md)：数据隔离、验收和提交署名要求。

## 安装与运行

需要 Python 3.10+；Node.js 仅用于独立二进制协议测试。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[api,readers,test,figures]'
git config core.hooksPath .githooks

# 数据目录必须独立；默认也在代码仓库外。
export BRAINELECQC_STORAGE_ROOT="$HOME/.local/share/brainelecqc"
python -m brainelecqc import /path/to/record.edf --modality eeg --label recording-01
python -m brainelecqc list
python -m brainelecqc serve --port 8840
```

服务默认只绑定 `127.0.0.1`，本版本没有远程多用户鉴权；它是本地研究工具。BrainVision 等多文件格式要一起导入入口和伴随文件，也可以导入完整目录。模糊后缀必须声明厂商，如单文件 `--format-hint nihon`，或成套目录 `--format-hint session/record.eeg=nihon`。

Python 直接调用不要求启动 Flask：

```python
from brainelecqc.db import init_db
from brainelecqc.recordings import import_recording
from brainelecqc.signals import describe_artifact

init_db()
record, reused = import_recording(
    {"modality": "eeg", "label": "recording-01"}, paths=["/path/to/record.edf"]
)
catalog = describe_artifact(record["input_artifact_id"])
```

HTTP 前缀 `/api/v1/ephys`：

| 路径 | 用途 |
| --- | --- |
| `POST /recordings` | multipart 的 `metadata` JSON + `files`；禁止请求服务器任意路径 |
| `GET /recordings`、`GET /recordings/{id}` | 列表与稳定记录身份 |
| `POST /recordings/{id}/verify` | 输入指纹检查；不代表科学 QC 通过 |
| `GET /artifacts/{id}/signals`、`/events` | 流、通道、分段和事件元数据 |
| `GET /artifacts/{id}/samples`、`/overview` | 精确样本窗口或明确区分的 min/max 概览 |

样本窗口显式提供 `stream_id`、`segment_id`、有序 `channel_id`、半开区间 `start_sample/stop_sample`、`representation=physical|digital`。原始数据不隐式处理；概览不能替代检测输入。`web/ephysBinary.js` 是独立 JavaScript 解码器，检查身份、形状、单位和顺序；它不是完整播放器 UI。

## 验收与图

```bash
python -m pytest -q
npm test
python scripts/figure_acceptance.py --output "$HOME/brainelecqc-evidence"
python scripts/check_public_tree.py --staged
git diff --check
```

图形脚本输出流程框图及三模态的合成三阶段对照 PNG/SVG/PDF。真实数据通过仓库外 `--manifest` 指定，结构见脚本 `real_case`；图和路径清单均不得提交。所有图保持同一窗口、通道、物理单位和比例。真实数据尚未人工审核时第三栏保持“待审核”；合成图的第三栏明确是模拟决定。

**合成夹具**是测试时临时生成、数值和异常位置已知的小信号。例如注入一条平线、工频或已知时间缺口，检查程序是否保留/识别它。它能够发现程序错误，不能替代真实数据、专家标注或临床准确率验证。

公开代码只保留生成夹具的代码。真实文件、测试文件、波形图、派生数据、数据库、日志与私人路径都留在仓库外。
