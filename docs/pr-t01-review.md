# 修复模型采集并验证全部合成模型的参考可运行性

## 背景与目标

基础安装提前导入 torchvision，缺少 PyTorch 时连合成采集、报告和比较入口都无法使用。采集配置、随机种子、缓存及 external-data 处理也不可靠。上一阶段完整尝试 566 个原始规格时，只有 233 项通过 ONNX 检查，333 项因规格或输出描述问题失败。

本 PR 将采集基础和模型可运行性修复合并，保留全部原有用例标识，形成 548 个参考端可运行正例和 18 个明确负例，避免把测试数据问题误报成设备不支持。

## 设计思路与主要改动

- 按需加载模型来源，统一配置优先级，落实筛选、离线、路径和 dry-run；缺少已选来源依赖时明确失败。
- 按用例派生随机种子，完整复制、计量并校验 ONNX/external-data，在独立代次验证后原子发布缓存索引；增量采集保留其他记录。
- 修复 opset 18 的 Squeeze/Unsqueeze 输入、Clip 常量类型、MatMul/MLP 维度、Attention 转置和缩放、Slice 输入顺序；按形状推导生成输出描述，保留 Split 全部输出。
- 明确声明 10 个非法 Squeeze、6 个非法池化、2 个非法 Split，验证具体拒绝原因，单独统计并排除设备测试和算子抽取。
- 每个正例必须通过 ONNX full_check 和 ONNX Runtime CPU 执行，检查全部输出的数量、形状、dtype 及有限值，记录输入与逐输出摘要。参考失败不能发布成功缓存。
- 采集器和合成生成器版本分别升级为 2/3；参考依赖、输入内容和负例声明参与缓存身份，旧的仅结构检查证据不直接复用。
- 基础 CI 覆盖 Python 3.9/3.12 的完整正负例和全量离线缓存复验，保存 XML 与 catalog；不将此 CI 当作 InfiniTensor 设备验收。

## 验证与效果

```bash
python -m pip install -e '.[test]'
python -m pytest tests/unit tests/operators -q
infinitensor-test-collect --sources synthetic_full --offline --data-dir .validation/models
infinitensor-test-collect --sources synthetic_full --offline --data-dir .validation/models
```

全新依赖环境的真实 pytest 输出：

```text
........................................................................ [ 78%]
....................                                                     [100%]
92 passed in 13.42s
```

- Windows / Python 3.12.14；ONNX 1.23.2、ORT CPU 1.31.0、NumPy 2.5.4。既有 ONNX 1.17.0、ORT 1.20.1、NumPy 2.3.5 环境也为 92 项通过。
- 没有安装 torch、torchvision、timm、huggingface_hub 或 InfiniTensor。
- 首次真实 CLI 采集：total=566、generated=566、positive_validated=548、negative_rejected=18、failed=0、skipped=0，退出 0。
- 再次真实 CLI 采集：cached=566、generated=0，548 个正例和 18 个负例重新验证成功，退出 0。
- 回归测试覆盖全部规格的顺序独立性、完整 external-data、缓存损坏/证据失效、NPZ 大整数及错误输入、参考异常、第二输出数量/shape/dtype 错误、NaN/Inf，以及负例不进入模型设备分母或算子抽取。

## 范围与限制

548 个正例通过的是 ONNX 检查与 ONNX Runtime CPU 参考执行，不代表 548 个模型都在 InfiniTensor 上正确执行。HardSwish、HardSigmoid、AveragePool 等上游问题没有修改，也没有放宽容差。

本 PR 尚未交付命名 InputProfile、设备端多输入/dtype/profile 执行、严格数值比较、设备子进程重做、算子忠实重放、CPU/L20 基线和统一回归门禁。未进行真实 torchvision 导出、在线来源下载或 InfiniTensor CPU/GPU 重测；GitHub 托管 CI 结果需要创建 PR 后确认。

缓存消费者读取 catalog 路径；已发布代次暂不自动回收，采集采用单写入者。完整使用说明见同目录 pr-t01-collection.md。
