# 多任务真实网络扩充

本文记录 e9660ba 的 1409 项通用基线。新增 13 项飞桨网络采用独立固定清单和环境，联合去重后共 1422 项、21 种任务，见[飞桨模型接入说明](paddle-models.md)。

统一清单从 1347 项扩充为 **1409 个去重真实网络配置、16 种任务**。本轮专门筛选非图像分类/语义分割网络：69 个候选中 62 个通过 ONNX 检查、ONNX Runtime 执行和全部输出 PyTorch/ONNX 比较，7 个失败如实保留。原有 548 个合成正例和 18 个负例另计。

## 新增任务和来源

| 任务 | 新增通过 | 网络实例 | 上游实现 |
|---|---:|---|---|
| 超分辨率 | 27 | CARN、EDSR、NinaSR、RCAN、RDN 的 x2/x3/x4 配置 | [torchSR](https://github.com/Coloquinte/torchSR)，MIT |
| 语音识别 | 2 | Wav2Letter、DeepSpeech | [torchaudio](https://github.com/pytorch/audio)，BSD-2-Clause |
| 音频分离 | 1 | Conv-TasNet | torchaudio |
| 视频动作识别 | 5 | R3D、MC3、R(2+1)D、S3D、Swin3D | [Torchvision](https://github.com/pytorch/vision)，BSD-3-Clause |
| 双图光流 | 2 | RAFT Small / Large，保留全部迭代输出 | Torchvision |
| 目标检测 | 5 | Faster R-CNN 两种骨干、RetinaNet、FCOS、DETR | Torchvision / [Transformers](https://github.com/huggingface/transformers)，Apache-2.0 |
| 文本分类 | 4 | BERT、DistilBERT、ELECTRA、ALBERT 的序列分类头 | Transformers |
| 序列标注 | 4 | 上述四种架构的 token 分类头 | Transformers |
| 抽取式问答 | 4 | 上述四种架构，比较 start/end 两个输出 | Transformers |
| 语言建模 | 5 | 四种 masked-LM 头以及 GPT-2 causal-LM | Transformers |
| 文本生成前向 | 1 | T5 Small 编码器—解码器 | Transformers |
| 单目深度估计 | 1 | DPT | Transformers |
| 时间序列预测前向 | 1 | Time Series Transformer | Transformers |
| 合计 | 62 | 13 种新增覆盖任务，其中目标检测原已存在 | 4 个来源插件 |

累计还保留图像分类 1076、语义分割 266、姿态 1；检测累计 9。图像分类数量仍占多数，因此单独提供 [62 项多任务清单](../catalogs/diverse-networks.json)，其 `required_tasks` 要求 13 种任务全部出现。数量达标但任务缺失时仍退出失败；统一清单同样要求全部 16 种任务。

网络使用上游实现及随机初始化参数，不下载权重、不训练。Transformers 使用库默认配置；ALBERT 使用 base-v2 的 768 隐藏维度、12 头、3072 中间维度。DETR 使用库内 ResNet-50，不下载骨干权重。没有通过缩减层数或用骨干替代完整网络来取得通过。

文本生成验证的是固定长度前向：GPT-2 关闭 KV cache、显式传入合法的 4D 加性因果 mask，T5 同时传入 encoder/decoder token 并比较 logits 和 encoder hidden state。它们不代表完整 tokenizer、采样循环或 KV-cache 解码服务。时间序列网络验证 teacher-forced 前向的三个 Student-t 分布参数、encoder hidden state、loc、scale、static features，共七个输出，不宣称验证了随机生成循环或预测精度。

## 输入和有效性

新来源声明输入名称、具体 shape、dtype、显式 token/控制值或浮点范围。采集器生成独立 `.npz`，原始 PyTorch 与 ORT 使用同一份数组；导出、参考和缓存保留输入摘要。浮点值由全局种子及用例/输入名称决定。整数 token 不从随机浮点数转换，动态形状不被替换成 1。

支持多输入、INT64/BOOL 输入、嵌套多输出及逐输出命名指标。整数/布尔输出精确比较，浮点使用 `abs(actual-reference) <= 1e-3 + 1e-3*abs(reference)`，shape/dtype 不同或 NaN/Inf 均失败。针对固定调用，用实际 PyTorch 输出补齐静态输出描述，避免 exporter 错误复用符号维度；不修改 ONNX 计算节点。ORT 检查和全部输出比较仍必须通过。

严格清单增加逐用例资源预算及输入摘要门禁，同时保留结构摘要、去重、完整性和数量门禁。来源插件 ID 随清单保存，避免将 torchvision_tasks 错误重放为原有分类来源。

## 安装与复验

完整复验环境为 Python 3.12、CPU torch/torchvision/torchaudio 2.9.1/0.24.1/2.9.1、torchSR 1.0.4、Transformers 4.57.1，完整约束见 configs/constraints-diverse.txt。保持 CPU 三个 PyTorch 包版本匹配。

```bash
python3.12 -m venv .venv-networks
source .venv-networks/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[networks,diverse,test]' -c configs/constraints-diverse.txt
python -m pip check

# 只验收 62 项多任务网络，必须覆盖全部 13 种任务
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
infinitensor-test-expand --manifest catalogs/diverse-networks.json \
  --data-dir data/real-networks --offline --max-size-mb 768 \
  --minimum-real-models 62 --jobs 4 --batch-size 4 --timeout 300

# 完整 1409 项及全部 16 种任务
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
infinitensor-test-expand --manifest catalogs/real-networks.json \
  --data-dir data/real-networks --offline --max-size-mb 768 \
  --minimum-real-models 1001 --jobs 4 --batch-size 8 --timeout 300

# 扩大候选，包含已知失败；诊断完成后因候选失败退出 1
infinitensor-test-expand --sources torchsr torchaudio transformers torchvision_tasks \
  --source-config configs/diverse-sources.yaml --data-dir data/real-networks \
  --offline --max-size-mb 768 --minimum-real-models 62 --timeout 300

# 明确选择真实来源集成测试；缺依赖会失败
python -m pytest tests/integration/test_diverse_sources.py -q
```

旧 1347 项预算仍为每项 160 MiB；新 62 项上限为每项 768 MiB，以容纳标准文本和深度网络。最终预算取 CLI、来源配置及清单单项预算的最小值。提高资源预算没有改变比较容差或任务通过标准。

正式复验不用 `--resume`；采集器可以复用内容校验通过的模型缓存并重新执行 ONNX/ORT 检查，复用相同包及输入的 PyTorch 比较证据。增加 `--force` 可重新导出并计算全部参考结果。首次候选筛选和 CI 冒烟均实际导出并比较输出。

## 保留的失败

| 候选 | 最终失败阶段 | 实际问题 |
|---|---|---|
| PatchTSMixer | invalid_model | 导出图 MatMul 维度冲突 |
| PatchTST | reference_failure | PyTorch 输出 (1,24,1)，ORT 输出 (1,24,32) |
| MViT v1 B | reference_failure | 导出 If/Squeeze 路径运行时形状冲突 |
| SSD300、SSDLite320 | export_mismatch | PyTorch/ONNX 数值不一致 |
| Mask R-CNN、Keypoint R-CNN | export_mismatch | PyTorch/ONNX 数值不一致 |

这些是采集/导出/参考侧失败，不计入通过清单，也不标为 InfiniTensor 不支持。完整诊断见 [机器可读结果](task-expansion-results.json)。实例分割和新增姿态候选目前没有新增通过项。

## InfiniTensor 能力边界

筛选以完整 ONNX 图、确定输入、CPU 参考执行和 PyTorch/ONNX 数值一致性为准，不要求 InfiniTensor 目前已经实现所有算子。新网络覆盖 Conv/ConvTranspose、MatMul、归一化、Softmax、Gather/索引、Resize、GridSample、ROI/NMS 等计算需求，可用于发现并推动后端缺口。

本轮没有构建或执行 InfiniTensor，也没有验证 A100 GPU 设备支持率。实际支持仍需要检查图中算子、dtype、形状和组合语义并运行设备比较。随机参数通过不代表模型任务精度；固定输入通过不代表所有动态形状或数据相关分支均已验证。

现有 `runner/engine.py` 尚未接入这里保存的输入 NPZ，仍只写入第一个 FP32 输入、读取第一个输出，且参考失败时可能关闭比较后继续执行。因此本 PR 的 1422 项采集/导出通过数不能作为设备正确执行数，也不能用旧执行器的 `OK` 证明新增多输入、INT64 或多输出模型已验证正确。后续执行器修复应优先接入 NPZ、全部输入输出比较，并将参考失败作为明确失败处理。
