# 可迁移性与未来 Linxi 接入

现在是普通 Python 包，不建立插件框架。Reader 层只接文件、稳定 source_id 和显式 format hint，返回可序列化目录及有序窗口；不知道原平台的 subject、影像、任务或前端状态。

## 已隔离的边界

| 层 | 当前职责 | 迁移约束 |
| --- | --- | --- |
| readers | 格式识别、伴随文件、保真数据和时间 | 可单独导入；NumPy 核心、MNE 可选；不创建 DB 或启动服务 |
| recordings/storage | 独立身份、快照、指纹、事务 | 独立 SQLite、相对存储路径；无需原项目数据库；整体移动存储后身份不变 |
| signals/windows/derived | 目录、精确样本、概览、父版本和时钟 | 明确 source/stream/segment/channel；原始与派生不能混淆 |
| CLI / HTTP | 输入参数和序列化 | 本地路径只在可信 Python/CLI；HTTP 仅上传；没有远程服务认证 |
| JS decoder | EPHY 二进制验证与解码 | 不依赖 UI、Three.js、全局配置或原平台 fetch 封装 |

当前配置通过独立环境变量或 Flask 应用上下文解析，仍不是完全显式的多租户 StorageContext。真正嵌入其他系统时需对配置、日志、取消与并发生命周期做适配；本轮的“可迁移”指无原平台运行时依赖、可移动存储、独立安装和数值合同通过，不等于已验证所有宿主。

## Linxi 核查与后续动作

已检查 [Linxi PluginTemplate](https://gitee.com/lgl-linshu/Linxi-PluginTemplate) 的文档、processor 和测试。模板通过 YAML `linxi_plugin` 导入模块，由装饰器注册 processor；处理器继承 `DefaultProcessor`，`_process(context)` 返回更新 context。

后续新建一个很薄的宿主适配包：将 context 的文件、记录身份和明确参数转成 QC 调用，返回 artifact、QC revision、掩码和来源。不把核心模块注册到宿主；仅适配包的导入执行宿主注册。无需 probe 或位置时不引入相关模板。

待实施验收：宿主 API/版本锁定、缺少字段错误、宿主取消、幂等重跑、临时目录生命周期、结果回读、插件卸载后独立核心仍可用。本轮没有安装 Linxi，也没有声称插件验收完成。模板未见明确 LICENSE，因此只参考公开接口，不复制模板源码。

## double check 的判据

1. 在新目录安装包，不设置原平台路径仍可运行测试。
2. 原平台源码和未提交文件 hash 不变；源数据只读。
3. 两个独立 store 不互串；误指其他应用的数据库明确拒绝。
4. 复制整个 store 到新目录后 recording/artifact/hash、通道顺序和样本值不变。
5. 网络传输解码器拒绝错记录、错段、错顺序、错单位/形状和截断数据。
6. 完整暂存树、提交对象和作者信息检查不包含数据、私有路径或自动化署名。
