# 文件格式与读取方式

成套文件一起建立输入指纹；读取保留通道名称、顺序、采样率、单位、事件和分段时间。

| 输入 | 读取方式 | 校验情况与后续工作 |
| --- | --- | --- |
| EDF/EDF+、BDF/BDF+ | 包内按字节解码；数字量与物理量；混合采样率分流；TAL 分段与事件 | 已知值测试、本地真实三模态对照；异常厂商 TAL 保留原字节、恢复解释与告警 |
| BrainVision `.vhdr/.vmrk/.eeg` | 从 `.vhdr` 进入；INT16/UINT16/FLOAT32、multiplexed/vectorized；按索引保留重复名称和逐通道单位 | 已知值与分段测试、公开 ECoG 数值对照；完整导入伴随文件 |
| FIF / FIF.GZ | MNE 按需读取连续记录，保留 first_samp、通道与校准 | 生成数据数值回读；扩展项为采集断点映射与更多真实样例 |
| EEGLAB `.set/.fdt` | MNE；连续记录、小型未压缩 MAT-v5 头、外置 float32 FDT | 生成数据回读；扩展项为其他头结构与 boundary 分段 |
| Nicolet `.data/.head` | MNE 按需读取配对文件 | 已知值回读；继续增加真实厂商样例 |
| Neuroscan `.cnt` | 显式 `neuroscan_cnt` 格式提示 | 已接读取入口；下一步是厂商真实数值对照 |
| Nihon Kohden `.EEG` | 显式 `nihon` 入口及伴随文件，连续单块 | 已接读取入口；下一步是厂商真实样例与多块时间映射 |
| Nicolet/Nervus `.e` | 解码开发项 | 获取真实样例与同次官方导出，核对极性、逐流采样率、全部分段、事件及变体 |
| 内部派生 `.npy` | NumPy 窗口读取 | 配套 V 单位、父指纹、处理、通道映射与时钟描述 |

MNE 接口提供校准后的物理样本；包内 EDF/BDF 读取器同时提供原始数字量。格式共用统一的信号访问接口，调用方式见 [模块与函数](portability.md#模块与函数)。

独立厂商文件可使用 `--format-hint nihon`；成套目录使用 `--format-hint session/record.eeg=nihon`，将格式声明绑定到具体文件。

## 尼高丽 .e 的接入

[FieldTrip Nicolet 说明](https://www.fieldtriptoolbox.org/getting_started/eeg/nicolet/)介绍了不同年代的文件变体；[MNE Nicolet](https://mne.tools/stable/generated/mne.io.read_raw_nicolet.html)对应 `.data/.head` 导出。

`.e` 的候选实现参考 [eeg2edf](https://github.com/BellaNes-Systems/eeg2edf)低层按范围读取接口。接入检查包括文件索引和 payload 长度、逐通道样本数、分段时间精度、极性和事件；所有通道保留自身采样率。通过官方导出逐窗口对照后纳入格式矩阵。

## 头文件完整性检查

`inspect_input_header(path)` 从原文件检查技术字段，输出逐项状态、整数得分和文件指纹。EEG 自动流程保存到 `paper_methods.json` 的 `header`，并放在报告开头的质量总结及头文件检查表中。

| 核对内容 | EDF / BDF 的字段与检查 | 作用 |
| --- | --- | --- |
| 格式与头长度 | 版本签名、通道数、头字节数；头长应为 256 × (通道数 + 1) | 确认解码方式与信号区起点 |
| 通道表 | 原始名称、源索引、信号通道与注释通道 | 保留通道顺序和身份 |
| 采样率 | 每记录样本数 ÷ 每记录持续秒数，逐通道计算 | 核对频率轴、事件和时长；混合采样率逐流保留 |
| 单位与标定 | 物理单位、physical min/max、digital min/max | 将整数样本转换成正确物理幅值 |
| 数据长度 | 记录数、记录时长、每道样本数与文件字节数相符 | 发现截断、缺记录和错误长度声明 |
| 时间轴 | 开始日期/时间；EDF+/BDF+ 的注释计时与连续/分段标志 | 保留停采、续采与事件时间 |
| 采集背景 | 硬件滤波、传感器、参考、事件说明及匿名身份状态 | 为频谱和人工解释提供背景 |

前六类技术检查合计 10 分，细分权重见 [评分表](eeg-methods.md#6-自动整数评分与人工复核)。身份字段只保存是否填写；采集背景按实际填写状态记录。格式依据：[EDF 规范](https://www.edfplus.info/specs/edf.html)、[BioSemi BDF 规范](https://www.biosemi.com/faq/file_format.htm)。

| 厂商文件 | 头信息与配套文件核对 | 当前检查入口 |
| --- | --- | --- |
| 尼高丽 Nicolet/Nervus `.e` | 文件版本、目录索引与数据块长度、逐流采样率、通道映射与标定、分段起止和事件索引 | 待真实版本样例接入低层解码；报告保留“格式待确认” |
| Nihon Kohden / NK 文件组 | `.EEG` 信号块；`.21E` 通道/电极信息；`.PNT` 记录元数据；`.LOG` 注释，以及各文件一致性 | 现有 `nihon` 读取入口；逐字段头质控待厂商样例校验 |
| 单独标为 `.nkt` 的文件 | 先确认文件签名、导出软件和版本，再绑定对应头结构 | 保留“格式待确认”及空评分 |

Nicolet `.e` 格式有年代差异，按实际版本处理。[FieldTrip 格式说明](https://www.fieldtriptoolbox.org/getting_started/eeg/nicolet/)；NK 配套文件的职责见 [MNE 官方读取说明](https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html#nihon-kohden-eeg-data-eeg-21e-pnt-log)。

## 真实样例来源

| 来源 | 用途 |
| --- | --- |
| [HMC sleep staging v1.1](https://physionet.org/content/hmc-sleep-staging/1.1/)，CC BY 4.0 | EEG 与同步 ECG/EMG/EOG；睡眠阶段作为状态背景，污染区间另行人工标注 |
| [TUH EEG Artifact Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) | 注册申请后使用伪迹标注记录 |
| [EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet) | 处理后片段与半合成对照，用于算法校准 |

下载文件、许可副本、版本、hash 和人工标签保存在仓库外；Git 保存来源链接与生成、校验程序。
