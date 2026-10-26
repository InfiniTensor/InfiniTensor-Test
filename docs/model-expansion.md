# 真实网络模型扩充与复验

本文记录首轮提交 c7430c4 的 1347 项基线。当前统一清单已扩充到 1409 项、16 种任务；安装与最新复验入口见[多任务扩充说明](task-expansion.md)。下述首轮复验命令应与 c7430c4 版本的清单配套使用。

本功能将真实网络架构配置与合成用例分别统计。通过数量仅包含成功导出、通过 ONNX 检查及 ONNX Runtime 执行、且全部输出与原始 PyTorch 一致的网络。随机初始化的参数用于编译器验证，不代表模型任务精度或训练结果。

## 接入来源

| 来源 | 用途 | 来源与许可 |
|---|---|---|
| timm | CNN、视觉 Transformer、MLP 等分类网络；按官方默认输入尺寸导出，启用 exportable 配置，排除权重标签别名 | [上游仓库](https://github.com/huggingface/pytorch-image-models)，Apache-2.0 |
| PyTorchCV | ImageNet/CIFAR/SVHN 等网络及分割、检测、姿态候选；从模型实际 in_size 和第一层卷积读取输入 | [上游仓库](https://github.com/osmr/pytorchcv)，MIT |
| Segmentation Models PyTorch | 官方解码器与编码器构成的完整分割网络，使用 encoder_weights=None | [上游仓库](https://github.com/qubvel-org/segmentation_models.pytorch)，MIT |

SMP 的组合是候选配置，不代表每个编码器都兼容每个解码器。只有实际验证成功的组合才进入清单。分类网络的预训练权重标签、同一网络重复导出、不同随机种子及合成用例不能用来凑真实模型数量。

[ONNX Model Zoo 官方](https://github.com/onnx/models)已停止旧 Git LFS 模型下载，现有模型迁移到 [Hugging Face 镜像](https://huggingface.co/onnxmodelzoo)。本轮没有将其不同 opset/框架副本直接累加进目标数量；带 token、索引和动态控制输入的来源还需要显式输入合同，不能随机伪造输入。原有 onnx_zoo/huggingface 小型来源保留，但不作为本轮数量证明。

## 本轮实测结果

最终固定清单包含 **1347 个去重的真实网络配置**，严格复验 **1347/1347、0 失败、0 排除、0 重复、退出 0**。原有 548 个合成正例和 18 个负例另计；有效正例合计 1895，但真实网络指标始终单列。

| 来源 | 固定清单中的独立通过项 |
|---|---:|
| timm | 615 |
| PyTorchCV | 477 |
| Segmentation Models PyTorch | 255 |
| 合计 | 1347 |

任务分布为分类 1082、分割 260、检测 4、姿态 1。部分分类项来自同一家族不同深度/宽度/分类头的官方具名配置，不应把 1347 解读成 1347 个独立模型家族。

2595 个候选的最终状态为：1487 个通过全部参考检查，去除 140 个重复后为 1347；871 个按参数/模型包预算排除；237 个失败，全部保留诊断。其中 74 个导出数值不一致、50 个参考执行失败、2 个模型检查失败，另外 111 个为构建/导出等失败（包含缺少部分候选专用依赖）。这些失败不是 InfiniTensor 设备失败。

PSPNet 的合法输入修复恢复了 32 个候选，通过去重新增 30 项，没有更改模型结构或容差。完整统计和固定依赖版本见 [机器可读验收摘要](model-expansion-results.json)。

## 固定环境

本轮批量导出在 Linux / Python 3.12 的独立 CPU 环境进行，依赖见 configs/constraints-networks.txt。使用 A100 服务器的 CPU 资源导出和参考校验，没有将这项验证描述为 InfiniTensor/A100 设备支持率。

```bash
python3.12 -m venv .venv-networks
source .venv-networks/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[networks,test]' -c configs/constraints-networks.txt
python -m pip check
```

基础安装仍不需要这些来源库。也可以分别安装 `.[pytorchcv]`、`.[segmentation]` 或 `.[timm]`。选择缺依赖来源时明确失败。

## 全量候选筛选

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
infinitensor-test-expand --sources pytorchcv segmentation_models timm \
  --data-dir data/real-networks --source-config configs/network-sources.yaml \
  --offline --max-size-mb 160 --minimum-real-models 1001 \
  --jobs 4 --batch-size 8 --timeout 180
```

来源发现只生成具名规格，不下载预训练权重。每个进程按批次复用依赖导入；每个用例仍独立落盘，超时或崩溃后回收该 worker 并继续剩余用例。默认两个 worker；并发数应根据 CPU、内存和磁盘调整。

若模型支持 meta 构建，先估算参数字节数，避免实际分配明显超过预算的模型；不支持时正常构建后检查。`parameter_size_limit` 是参数预算排除，`size_limit` 是完整 ONNX 包大小排除，两者都不等于设备不支持。160 MiB 是本轮资源策略，用户可显式更改；来源配置中的 `max_size_mb` 与全局预算取较小值。

真实模型导出使用库的网络实现，固定随机种子，不下载检查点、不训练。原始 PyTorch 与 ONNX 采用同一确定性输入，全部输出检查数量、shape、dtype 和有限值；浮点逐元素要求 `abs(actual-reference) <= 1e-3 + 1e-3*abs(reference)`，整数精确相等。所有差异保留为失败，不放宽容差。

SMP 普通分割输入为 256×256。PSPNet 使用合法的 384×384，使 stride 8 特征图为 48×48，可被金字塔池化的 1/2/3/6 各档整除；不修改网络结构或替换算子。

## 固定清单严格复验

```bash
infinitensor-test-expand --manifest catalogs/real-networks.json \
  --data-dir data/real-networks --offline --max-size-mb 160 \
  --minimum-real-models 1001 --jobs 4 --batch-size 8 --timeout 180
```

固定清单只包含已通过的独立真实网络配置，具有来源、名称和结构摘要。模型发生结构变化、必测项失败/被排除/漏执行、去重后数量不足或执行不完整，都不能通过严格复验，包括使用 `--limit` 只执行清单子集的情况。验收不会自动缩减或改写输入清单；生成的子集另存为 validated-models.json。输入清单不得放在模型缓存或报告输出目录中，探索产生的 accepted-models.json 除外，避免输出覆盖输入。

`--resume` 仅在请求的源码/配置签名相同时复用已完成任务记录，并检查成功模型包摘要。它适用于恢复一次被中断的探索；正式复验不使用该参数。采集器仍可复用模型包及输入摘要校验通过的参考验证缓存，本轮最终 1347 项严格复验采用此方式；首次候选筛选逐项执行了导出与 PyTorch/ONNX 比较。需要重新导出和计算参考结果时增加 `--force`，该参数优先于 `--resume`。清单版本更新需要显式审查，不会自动修改仓库中的清单。

若希望保留探索报告、同时复用模型缓存，可将复验输出放到另一个 data_dir，并设置绝对路径 `ITEST_MODELS_DIR` 指向原缓存目录。

## 数量与去重

同时使用具名架构身份和 ONNX 结构指纹去重，采用传递合并，选择稳定代表项。结构指纹忽略节点/张量名称和浮点参数值，保留算子、属性、opset、连线、参数形状/dtype、整数控制常量及输入输出描述。因此同一个网络换权重或随机种子不会提高计数。它不是任意两个图数学等价的证明；清单表示通过上述规则去重的具名网络配置，不是 1000 个独立研究模型家族。

输出目录包括：

- models/catalog.json：保留增量采集记录，包含模型与参考验证证据。
- expansion/inventory.json：唯一通过项、重复项、排除项、来源和任务统计。
- expansion/results.json：本次每个候选的执行状态。
- expansion/accepted-models.json：探索产生的候选通过清单；严格复验写 validated-models.json。
- expansion/tasks/：逐用例请求/结果。
- expansion/batches/：worker 日志、活动状态和崩溃/超时诊断。
- expansion/progress.json：明确标记任务是否完整结束。

退出码：完整完成、达到真实模型最低数量且没有失败为 0；候选失败、超时/崩溃、空集合、数量不足或固定清单未全部通过为 1；配置或来源环境错误为 2。探索阶段有失败时退出 1 是诊断结果，不应当用通过项数量覆盖失败状态。

## CI 与设备边界

新增来源 CI 固定依赖，执行基础测试以及三个来源各一个真实网络的导出和 PyTorch/ONNX 比较，上传 JSON/XML/worker 日志。本轮本地 116 项测试通过，服务器同一套测试 116 项通过，三来源冒烟 3/3 通过；GitHub 托管 CI 尚待 PR 触发。1000+ 全量复验在提供的服务器执行，不强制普通 GitHub runner 保存数十 GiB 模型权重。

本功能增加可供 InfiniTensor 测试的有效网络集合，不保证这些网络都已被 InfiniTensor 支持。设备执行隔离、全部输出数值门禁与 CPU/GPU 支持基线仍按后续计划交付。
