# 修复模型采集的来源加载、可复现性和完整包缓存

## 背景与目标

基础安装会提前导入 torchvision，导致没有 PyTorch 时连合成采集、报告及比较入口都无法使用。采集还存在 CLI 缺少导入、筛选配置未完整生效、随机权重受发现顺序影响，以及缓存遗漏 external-data、复用时不校验内容等问题。

本 PR 恢复独立于设备运行时的采集基础，为后续修正用例和建立严格数值门禁提供可复现、可校验的模型包。

## 设计思路与主要改动

- 按需加载来源，为 torchvision、timm 和 Hugging Face 提供独立安装扩展；明确选择缺依赖来源时返回配置/环境错误。
- 统一 CLI、环境变量和 YAML 配置优先级，移除机器固定路径；应用 suite 的来源、类别、标签、名称、数量、大小和种子筛选，支持离线与不写目录的 dry-run。
- 由全局种子和用例身份派生随机种子，隔离工厂调用的 RNG 状态；全量、分块及不同发现/调用顺序生成相同模型。
- 在独立代次目录中生成或复制完整 ONNX 包，按主文件及全部 external-data 计算大小和 SHA256，通过 ONNX full_check 后原子发布缓存索引。重建失败保留旧包，增量采集保留其他 catalog 记录。
- 缓存记录生成器版本、定义摘要及验证证据；catalog 使用 schema_version=2，仍可读取旧 catalog。
- 合成模型按 opset 18 生成 IR 8，避免最新版 ONNX 默认 IR 超出 ONNX Runtime 支持范围；生成器版本升级为 2，使旧缓存失效。
- 新增基础安装 CI，覆盖 Python 3.9/3.12 的单元测试与离线缓存流程，PR 触发目标包含 master 和 main。

## 验证与效果

在不安装 torch、torchvision、timm、huggingface_hub 或 InfiniTensor 的环境执行：

```bash
python -m pip install -e '.[test]'
python -m pip check
python -m pytest tests/unit tests/operators -q
```

全新安装环境的真实 pytest 输出：

```text
...................................................................      [100%]
67 passed in 7.04s
```

- Windows / Python 3.12.14；全新安装为 ONNX 1.23.2、ONNX Runtime CPU 1.31.0、NumPy 2.5.4，pip check 通过。既有 ONNX 1.17.0、ONNX Runtime 1.20.1、NumPy 2.3.5 环境也为 67 项通过。
- 回归测试覆盖全部 566 个规格的字节级顺序独立性、缺依赖提示、配置优先级、完整 external-data、总大小限制、缓存损坏重建、失败发布回滚、旧 catalog 和增量采集；包含实际参考执行。
- 全新环境通过真实 console entrypoint 离线采集 5 个 Relu，重复执行 5/5 命中缓存，再采集 4 个 Add 后 catalog 保留 9 项；9 个模型全部由 ONNX Runtime CPU 执行成功。
- 完整原始集合 566 项全部尝试：233 项通过 ONNX full_check，333 项明确失败，CLI 按预期退出 1，没有跳过或环境错误。没有修改原有算子、维度、属性或输出描述来获得绿色结果。

## 范围与限制

本 PR 提升的是基础安装可用性、采集可复现性、离线配置和缓存完整性。ONNX full_check 不代表 InfiniTensor 数值正确；333 项原有规格错误仍需后续用例修复，不能计为设备不支持。548 个正例和 18 个负例的整理、InputProfile、设备隔离执行及严格数值门禁尚未包含。

尚未进行在线 Hugging Face/ONNX Zoo 下载、真实 torchvision 导出或 InfiniTensor CPU/GPU 设备验收；下载策略使用受控用例验证。GitHub 托管 CI 的结果需在创建 PR 后确认。

消费者应读取 catalog 中的模型路径。已发布缓存代次暂不自动清理，会保留磁盘占用；采集按单写入者运行，并发 catalog 合并不在本 PR 范围内。
