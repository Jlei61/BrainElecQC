# 读取格式与验收等级

读入采用显式格式注册表。成套文件一起建立输入指纹；名称、顺序、采样率、单位、事件和分段都属于读取合同。下表为本版本边界，不表示支持每个厂商的全部变体。

| 输入 | 当前实现 | 验收证据 / 限制 |
| --- | --- | --- |
| EDF/EDF+、BDF/BDF+ | 直接按字节解码；数字量与物理量；混合 Fs 分流；TAL 分段/事件 | 生成反例 + 本地真实三模态数值对照。未知物理量保留原单位；异常厂商 TAL 留原字节、替代解释和告警，不把恢复结果当确认真值 |
| BrainVision `.vhdr/.vmrk/.eeg` | 直接 NumPy 窗口；INT16/UINT16/FLOAT32，multiplexed/vectorized；按索引保留重复名和逐通道单位 | 生成已知值、分段和反例；公开 ECoG 样例单独做真实对照。必须从 `.vhdr` 进入，检查伴随路径；缺少可信分段时钟不能强拼 |
| FIF / FIF.GZ | MNE 惰性读取，保留 first_samp、通道与校准 | 生成数据数值回读；不宣称全部临床变体。遇到 BAD_ACQ_SKIP 等未适配断点明确拒绝 |
| EEGLAB `.set/.fdt` | MNE；连续、小型未压缩 MAT-v5 头 + 外置 float32 FDT | 生成数据回读；嵌入大数组、压缩头、epochs、boundary 等本轮不支持，避免偷偷全量加载或拼段 |
| Nicolet `.data/.head` | MNE 惰性读取；完整配对文件 | 已知值回读；它与尼高丽 `.e` 是不同格式 |
| Neuroscan `.cnt` | 显式 `neuroscan_cnt` 适配入口 | 待厂商真实数值样例；`.cnt` 也可能指其他厂家，不自动猜 |
| Nihon Kohden `.EEG` | 显式 `nihon` 入口及伴随文件 | 待厂商真实样例；多块时间未经验证时拒绝拼接；不是尼高丽 `.e` |
| Nicolet/Nervus `.e` | **已识别但未开放解码** | 待真实样例和官方导出对照，需核对极性、逐流 Fs、分段、事件及变体；不能通过改后缀绕过 |
| `.npy` | 仅内部派生信号 | 必须带明确 V 单位、父指纹、处理、通道映射和时钟描述；不是任意 NPY 原始上传适配 |

可选 MNE 入口只提供 `physical`；不存在可证明的 ADC 数字码时拒绝 `digital`。不为统一格式而重采样、丢慢通道、改参考或改极性。

## 为什么 `.e` 暂不直接接现成转换器

[FieldTrip 的 Nicolet 说明](https://www.fieldtriptoolbox.org/getting_started/eeg/nicolet/)描述了不同年代的变体及当前读入限制；[MNE Nicolet](https://mne.tools/stable/generated/mne.io.read_raw_nicolet.html)只处理 `.data/.head`。

候选 [eeg2edf](https://github.com/BellaNes-Systems/eeg2edf)的低层范围读取值得复用，但高层路径包含通道筛选、慢通道上采样、分段拼接和极性约定。本项目不会直接采用这些默认处理。接入前须限制索引/元数据大小、逐通道核对 payload 与分段样本总数，保留原始时间精度和解码证据，再用真实 `.e` 做逐窗口对照。这是明确的下一步，不能写成当前已经完成。

## 后续样例来源

- [OpenNeuro ds003688 v1.0.7](https://github.com/OpenNeuroDatasets/ds003688/blob/1.0.7/dataset_description.json)：CC0，SEEG/ECoG 与通道 good/bad/noisy 标签；只需要信号及通道表。空间放置导致的坏道不用于声称纯信号检测准确率。
- [HMC sleep staging v1.1](https://physionet.org/content/hmc-sleep-staging/1.1/)：CC BY 4.0，EEG + 同步 ECG/EMG/EOG；官方标签是睡眠阶段，不是伪迹真值。用于参考证据展示，仍需人工标注是否污染 EEG。
- [TUH EEG Artifact Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)：伪迹标注资源，需注册申请，尚未获取。
- [EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet)：处理后片段/半合成对照，不能替代连续原始临床验证。

下载文件、许可说明副本、版本、hash 和人工标签均保存在仓库外的验收目录。代码仓库只保留公开来源链接和生成/校验程序。
