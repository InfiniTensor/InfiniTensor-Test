# T01：模型接口、来源配置、确定性采集与缓存

本 PR 从 `2fc2654` 独立整理，只修改 InfiniTensor-Test 的采集基础。原有 566 个合成规格的名称、算子、维度与属性保持原样；随机常量改为按用例派生种子。T02 的规格修正、InputProfile 与参考验证，T03 的设备隔离与严格判定，T04 的全量门禁均未合入本分支。

## 能力提升

| 方面 | 原始版本 | T01 |
|---|---|---|
| 基础安装 | 包导入会导入 torchvision；无 PyTorch 无法使用采集/报告/比较 | 合成采集、既有报告及比较可在没有模型来源扩展库的环境运行 |
| 来源选择 | 全量导入；timm 缺失时静默返回空集；插件标识与来源家族混用 | 按需加载；明确缺依赖；支持插件 ID 与来源家族筛选 |
| CLI | console entrypoint 缺少 json 导入；suite 仅应用来源 | 配置、类别、标签、名称、数量、大小和种子生效；dry-run 不写数据目录 |
| 路径 | 带固定机器路径；自定义相对子目录会丢失 | data_dir 相对调用目录；相对子目录保持原值；绝对路径保留 |
| 可复现性 | 发现时使用全局 RNG，来源和用例顺序改变权重 | 全局种子 + 用例/参数身份生成独立 RNG；工厂调用恢复调用者 RNG 状态 |
| 缓存 | 按主 ONNX 文件是否存在复用；未检查内容或生成器版本 | 版本、定义摘要、完整包 SHA256、ONNX 检查结果共同确定复用 |
| external-data | 复制主文件；共享目录 glob 清理可能误删权重 | 所有引用文件完整复制、完整计大小；互不清理；缺失/损坏明确失败或重建 |
| 发布与恢复 | 直接覆盖目标文件 | 独立包验证完成后原子切换 current.json；重导出失败保留旧包 |
| 持续采集 | 重建 catalog 会丢失其他来源的记录 | 按来源家族和名称更新选中项，保留其他已采集记录 |

本 PR 不新增 InfiniTensor 内核，不改变设备通过率判定，也不把 ONNX 检查成功称作数值正确。缓存中的 `validation.reference` 明确为 `not_run`。报告/比较沿用原有结果语义；严格 v2 数值门禁由 T03/T04 交付。

合成模型按 opset 18 推导其所需的 IR 版本（IR 8），避免依赖 ONNX 包的最新默认 IR 而超出参考运行时支持范围。此变更不修改算子、形状或属性；合成生成器版本升级为 2，使旧缓存失效。

## 安装和复验

```bash
python -m pip install -e '.[test]'
python -m pytest tests/unit tests/operators -q

# 不需要 PyTorch、torchvision、timm、Hugging Face 或 InfiniTensor。
infinitensor-test-collect --list-sources
infinitensor-test-collect --sources synthetic_activation --name-glob 'synth_Relu*' --offline --data-dir .validation/t01
# 再次执行：5 项 cached，0 项 generated。
infinitensor-test-collect --sources synthetic_activation --name-glob 'synth_Relu*' --offline --data-dir .validation/t01

# 新增 4 项 Add，保留已有 5 项 Relu，catalog 合计 9 项。
infinitensor-test-collect --sources synthetic_elementwise --name-glob 'synth_Add*' --offline --data-dir .validation/t01

# 仅预览，不创建 data_dir 或调用模型工厂。
infinitensor-test-collect --suite synthetic --categories activation --tags synthetic --name-glob 'synth_Relu*' --limit 2 --offline --data-dir .validation/preview --dry-run
```

可选来源分别使用 `pip install -e '.[torchvision]'`、`'.[timm]'`、`'.[huggingface]'`，或保留原来的 `'.[full]'`。明确选择缺少依赖的来源时退出 2，不静默跳过。

`--config` 加载 YAML，优先级为 CLI > `ITEST_*` 环境变量 > YAML > suite 中的对应默认值 > 内置默认值。`--offline` 等价于设置 `network_enabled=false`，约束 Hugging Face 的主模型与 sidecar 下载，以及 ONNX Zoo 缓存未命中的下载。torchvision/timm 继续使用随机权重；检测模型的 backbone 权重也不下载。包安装本身的联网由 pip 管理。

`--output` 是 `--models-dir` 的别名。相对 models_dir（包括该别名）相对于 data_dir，绝对路径保持不变。采集 limit 限制筛选后的尝试数量；大小限制按 ONNX 与所有引用的 external-data 文件总和计算，不按模型名称猜测大小。warmup/iterations 属于运行接口，T01 不新增这些执行能力。

采集退出码：至少成功采集一项且没有失败为 0；用例失败、空集合或全部被排除为 1；配置或已选来源依赖错误为 2。大小超限作为显式排除统计，不冒充导入/设备失败。

## 缓存和兼容

```text
models/
  catalog.json                     # schema_version=2；可读旧 catalog
  synthetic/<安全名称>-<名称摘要>/
    current.json                   # 原子发布，只指向验证完成的代次
    package-<唯一标识>/
      manifest.json                # 生成器版本、缓存键、完整包摘要、ONNX 检查
      files/
        <主 ONNX 原文件名>
        <external-data 相对文件>
```

调用方应使用 catalog 的 `spec.path`，不要拼接旧版 `<source>/<name>.onnx` 路径。旧 catalog 的路径仍可读取；没有版本和摘要证据的旧缓存不自动复用，也不会批量删除。已发布代次保留以保护正在读取旧路径的任务；本 PR 不提供垃圾回收或并发 catalog 写入协议，采集按单写入者运行。

生成器或导出策略修改时必须增加相应版本。合成规格另有包含节点、常量和输入输出描述的定义摘要；发现顺序和来源别名不参与模型身份。external-data 遵循 [ONNX 模型包规则](https://onnx.ai/onnx/repo-docs/ExternalData.html)，下载来源使用同一离线策略。

## 本分支的测试证据

单元与既有算子框架测试覆盖：无可选模型库时采集/报告/比较、缺依赖提示、suite 与 CLI 优先级、全部 566 个规格的字节级顺序独立性、工厂 RNG 隔离、缓存复用与失效、完整 external-data 与参考端执行、邻居文件保留、失败发布回滚、旧 catalog 读取和增量采集。

新增 `Collection foundation` 工作流在 main/master 的 PR 上检查基础安装和离线采集。它不构建设备运行时，不将这项 CI 描述为 CPU/GPU 模型验收。GitHub 托管运行结果需要实际推送后确认。

完整原始合成集合会暴露旧规格的 schema/输出形状问题；这些问题的修复属于 T02。全量采集诊断的退出 1 应保留，不能通过禁用 full_check 或修改规格来获得本 PR 的绿色结果。详见同目录的 [PR 描述与验收结果](pr-t01-review.md)。
