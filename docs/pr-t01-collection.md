# 模型采集与参考有效性修复

本 PR 在同一个分支合并采集基础与模型可运行性修复，只修改 InfiniTensor-Test。原有 566 个用例标识全部保留，集合整理为 548 个正例和 18 个明确负例。模型输入、节点和输出描述修复后，模型摘要会改变，不能沿用旧摘要建立的设备基线。

## 能力提升

| 方面 | 原始版本 | 修改后 |
|---|---|---|
| 基础安装 | 提前导入 torchvision，无 PyTorch 无法使用部分入口 | 来源按需加载，合成采集、既有报告和比较无需模型扩展库 |
| 采集配置 | suite 筛选不完整，包含机器固定路径 | CLI > 环境变量 > YAML；筛选、种子、相对路径、离线及 dry-run 生效 |
| 可复现性 | 全局 RNG 受发现顺序影响 | 每用例和参数独立种子，工厂恢复调用者 RNG，参考输入按用例及输入名生成 |
| 模型规格 | opset、dtype、维度和输出描述错误混在设备失败里 | 修复 Squeeze/Unsqueeze、Clip、MatMul/MLP、Attention、Slice，输出由形状推导产生，保留 Split 全部输出 |
| 参考验证 | 未作为采集完成条件 | 每个正例通过 ONNX full_check 和 ONNX Runtime CPU 执行，检查全部输出的数量、形状、dtype 和有限值 |
| 非法用例 | 与待测正例混用 | 10 个 Squeeze、6 个池化、2 个 Split 显式声明拒绝原因，按原因验证，单独统计 |
| 完整包缓存 | 主文件存在即复用，external-data 可能丢失或被邻居清理 | 完整包摘要、大小、版本及验证证据；独立代次验证后原子发布，失败保留旧包 |
| 增量采集 | 过滤采集可能覆盖其他记录 | 按来源家族及名称合并 catalog，保留未选中记录 |

正例的参考执行通过不代表 InfiniTensor 正确支持该模型。本 PR 不修改上游内核，不交付严格设备数值门禁。命名 InputProfile、设备端多输入/dtype/profile 执行、算子忠实重放和 CPU/L20 基线由后续改动交付。

## 安装与完整复验

```bash
python -m pip install -e '.[test]'
python -m pytest tests/unit tests/operators -q

# 不需要 PyTorch、模型来源扩展库或 InfiniTensor。
infinitensor-test-collect --list-sources
infinitensor-test-collect --sources synthetic_full --offline --data-dir .validation/models
# total=566, positive_validated=548, negative_rejected=18, failed=0，退出 0。

# 重复执行：cached=566, generated=0；缓存命中仍会重新验证内容和参考结果。
infinitensor-test-collect --sources synthetic_full --offline --data-dir .validation/models

# 仅预览筛选，不调用工厂或创建 data_dir。
infinitensor-test-collect --suite synthetic --categories activation --tags synthetic --name-glob 'synth_Relu*' --limit 2 --offline --data-dir .validation/preview --dry-run
```

`added` 是成功存储的记录总数，包括正例和按预期拒绝的负例；支持率只能使用正例。`positive_validated` 与 `negative_rejected` 分开统计，失败按 `errors_by_stage` 显示为模型无效、输入配置错误、参考失败、负例原因不匹配或其他采集失败。失败详情保存在 catalog 的 `collection_error` 和 `collection_error_code`。

已选集合全部符合验证条件且至少存储一项时退出 0；用例失败、空集合或全部被排除时退出 1；来源、配置或环境初始化错误时退出 2。仅选择负例且全部按声明原因拒绝也属于有效验证，退出 0。

可选来源分别通过 `pip install -e '.[torchvision]'`、`'.[timm]'`、`'.[huggingface]'` 或 `'.[full]'` 安装。选择缺依赖来源时明确退出 2，不静默返回空集合。

`--config` 加载 YAML；采集优先级为 CLI > `ITEST_*` 环境变量 > YAML > suite 对应默认值 > 内置默认值。`data_dir` 相对调用目录，指定的相对 models_dir 相对 data_dir，绝对路径保持不变。`--output` 是 `--models-dir` 的别名。

`--offline` 约束 Hugging Face 主文件及 sidecar、ONNX Zoo 的下载；缓存缺失时明确失败。torchvision/timm 使用随机权重，检测模型的 backbone 也不下载权重。pip 安装是否联网由 pip 控制。

## 输入和输出验证

参考端从 ONNX 读取实际输入名称、dtype 和形状，排除 initializer。静态浮点输入按全局种子、用例标识和输入名生成 0.1 到 0.9 的正值，避免 Log/Sqrt/Reciprocal/Pow 的非法定义域。每个输出都按 ONNX 声明检查数量、形状、dtype 和 NaN/Inf，并记录摘要；整数不转换为 FP32。

通过 Python API 采集具有整数、BOOL、动态形状或额外取值约束输入的本地模型时，必须设置 `spec.metadata['input_file']` 为显式 `.npz` 文件。NPZ 的键必须覆盖且仅覆盖所有运行输入，shape 和 dtype 必须匹配；相同符号维度必须一致。文件相对调用目录解析，内容摘要参与缓存身份。不会随机伪造 token、索引，或把未知维度替换为 1。此处仅交付采集参考验证，尚未把这些输入能力接入设备 runner。

负例由 ModelSpec 的 `expected_rejection` 和 `rejection_reason` 明确声明，并验证实际语义原因；不把任意异常当成负例通过。负例文件保留用于审查，catalog.filter 默认排除负例；模型 runner 的整套统计和算子采集也排除负例。直接调用 runner.run_one 时，负例在加载设备运行时或创建子进程前返回 skip，此保护不计入正常整套运行的设备分母。

MatMul 保留原有用例名称作为历史标识，实际输入维度以 ONNX 和 input_shape 为准。合成模型固定 opset 18，并推导所需 IR 8；正例输出由 strict shape inference 生成，而不是复制输入形状。

## 缓存和兼容

```text
models/
  catalog.json                         # schema_version=2，可读取旧 catalog
  synthetic/<安全名称>-<名称摘要>/
    current.json                       # 原子指向验证完成的代次
    package-<唯一标识>/
      manifest.json                    # 版本、缓存键、完整包摘要和验证记录
      files/<主 ONNX 原文件名>
      files/<external-data 相对文件>
```

调用方使用 catalog 的 path，不拼接旧版文件位置。大小限制按主 ONNX 和所有 external-data 的总和计算。采集器版本升级为 2、合成生成器版本升级为 3，旧结构检查缓存不能直接当作参考验证成功；缓存身份还包含 ONNX/NumPy/ORT 版本、用例定义、种子、负例声明和显式输入文件摘要。

失败发布不会覆盖旧有效包或删除其他模型的数据。已发布代次暂不自动清理，采集按单写入者运行。

## 验证与范围

两个 Python 3.12 依赖环境均通过 92 项单元及既有算子框架测试。测试包括完整 548/18 验证、566 个规格的字节级顺序独立性、NPZ 大整数、第二输出错误、NaN/Inf、参考异常、缓存证据失效、失败回滚及负例不进入设备分母。真实 CLI 完整采集与缓存复验见 [PR 验证记录](pr-t01-review.md)。

Collection foundation CI 在 master/main 的 PR 中执行基础安装、完整正负例测试和两次全量离线采集，保存测试 XML 和 catalog。它不构建 InfiniTensor，也不代表已完成 CPU/GPU 设备验收。
