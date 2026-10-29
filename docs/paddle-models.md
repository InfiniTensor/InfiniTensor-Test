# 飞桨模型仓库接入与复验

新增 13 个通过完整参考检查的真实 Paddle 网络，联合原有 1409 项去重后为 **1422 个网络配置、21 种任务**。两套清单分别验收，避免强制用户在同一环境安装两个框架；548 个合成正例和 18 个负例另计。

## 本轮引入结果

| 来源 | 通过网络 | 新增覆盖 |
|---|---|---|
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | DB MobileNetV3、DB ResNet50-vd；CRNN MobileNetV3、CRNN ResNet34-vd、SVTR Tiny；MobileNetV3 文字方向；TableAttention、SLANet | 文字检测 2、识别 3、方向 1、表格结构 2，共 8 项 |
| [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) / Matting | PP-LiteSeg STDC1、BiSeNet V2；MODNet MobileNetV2、MODNet ResNet50-vd、PP-Matting HRNet18 | 语义分割 2、图像抠图 3，共 5 项 |
| [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) | PicoDet、TinyPose 两个候选完成接口接入，但当前独立环境缺少 imgaug，尚未完成完整参考验证 | 2 项 environment_blocked，不计入可用清单 |

17 个候选中 13 个通过，2 个数值失败、2 个环境阻塞，没有重复或资源排除。EAST 和 PSE 的 Paddle/ONNX 数值差异保留为失败；PaddleDetection 的依赖阻塞不被标为设备不支持。完整模型、固定输入和失败原因见 [机器可读结果](paddle-models-results.json)。

本轮新增的是原有集合没有覆盖的文字检测、文字识别、文字方向、表格结构和图像抠图五种任务。OCR 网络的输出为原始检测图、识别概率、方向概率或表格结构/位置张量，不包括 Python/OpenCV 文字框拼接、字典解码和业务流水线；抠图输出为 alpha，而非语义类别标签。

## 真实网络与导出边界

使用官方配置构建完整推理网络，参数随机初始化，不下载检查点、不训练。仅关闭 pretrained/weights 加载；未减少层数、通道数或表格最大长度，未用骨干替换完整网络。字符/表格输出通道由上游字典和后处理配置推导，与官方导出逻辑一致。不同静态输入是调用配置，不作为额外模型重复计数。

固定源码：

- PaddleOCR：`3f32858ce3fb95ba04a8202000e432bcc50a6016`（release/2.9）。
- PaddleSeg：`3c4db66de1d9d59d0628ed87590b6308a2f4aa2a`（release/2.10）。
- PaddleDetection 探索：`f049a01a94ccd5addf780641da3ffff276b60083`（release/2.8.1）。

固定配置用于获得可复现的转换链路，不宣称覆盖飞桨最新全部网络。上游均保留 Apache-2.0 来源信息。模型二进制和上游源码不提交到本仓库。

Paddle2ONNX 的静态转换可能修改 Layer 内部状态。参考输出在转换前、同一显式输入下计算并复制保存，再与 ORT 全部输出比较；这恢复了 SLANet，避免将转换后的 Layer 错当原始参考。FLOAT 使用逐元素 `abs(actual-reference) <= 1e-3 + 1e-3*abs(reference)`，INT/BOOL 精确比较，数量、shape、dtype 或有限值不符均失败。没有降低容差。

为避免 PaddleSeg/Matting 的注册表互相影响，每个 Paddle 模型使用新的子进程，即使设置了较大的 `--batch-size`。超时、崩溃、日志、原子缓存、完整性和退出码规则与已有模型相同。Paddle 结果必须包含自己的 `paddle_reference` 证据，不能借用 PyTorch 比较证据；缓存及恢复签名包含 Paddle/Paddle2ONNX 版本。

## 独立 CPU 环境

推荐本轮实测的 Linux / Python 3.12。基础安装保持不依赖 Paddle；明确选择 Paddle 来源后，缺少依赖或固定源码会报错。

```bash
python3.12 -m venv .venv-paddle
source .venv-paddle/bin/activate
python -m pip install cmake==3.31.10
export PATH="$(python -c 'import cmake; print(cmake.CMAKE_BIN_DIR)'):$PATH"
python -m pip install -e '.[paddle,test]' -c configs/constraints-paddle.txt
python -m pip check

# 显式联网准备固定源码；默认只准备已验证的 OCR 和 Seg。
infinitensor-test-paddle-setup --directory data/paddle-repos
export ITEST_PADDLE_REPOS_DIR="$PWD/data/paddle-repos"

# 13 项独立真实网络，要求全部六种任务均出现。
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
infinitensor-test-expand --manifest catalogs/paddle-networks.json \
  --data-dir data/paddle-networks --offline --max-size-mb 768 \
  --minimum-real-models 13 --jobs 3 --timeout 300

# 四任务冷缓存冒烟及实际框架测试，缺依赖不通过 skip 变绿。
infinitensor-test-expand --manifest configs/paddle-source-smoke.json \
  --data-dir .validation/paddle-smoke --offline --max-size-mb 768 \
  --minimum-real-models 4 --jobs 2 --timeout 300
python -m pytest tests/integration/test_paddle_exports.py -q
```

准备入口使用稀疏检出和固定提交；已有目录只检查，不 reset、不覆盖。正式采集会拒绝版本不符或修改过的上游源码。`--offline` 不自动克隆源码、不下载权重；准备代码与导出运行分开。

主要版本：Paddle 3.0.0、Paddle2ONNX 2.0.1、NumPy 1.26.4、ONNX 1.17.0、ORT 1.20.1、protobuf 4.25.3。Python 3.12 环境显式安装 setuptools；无桌面服务器使用 headless OpenCV。完整约束见 configs/constraints-paddle.txt。

正式复验不用 `--resume`。采集器重新检查模型包摘要及 ONNX/ORT 结果，并复用同一包和输入的原框架比较证据；增加 `--force` 可重新导出和计算全部参考输出。固定清单还检查结构、输入摘要、全部任务和执行完整性。

## 探索失败与已知限制

```bash
# 全量探索包含已知失败，因此诊断结束后退出 1。
infinitensor-test-expand --sources paddleocr paddleseg \
  --data-dir data/paddle-networks --offline --max-size-mb 768 \
  --minimum-real-models 13 --jobs 3 --timeout 300
```

PaddleDetection 插件仍属探索能力，可显式准备其源码并选择来源。当前环境未安装其 imgaug 训练/导入依赖，两个候选均保留 environment_blocked；尚未证明解决该依赖后就能完整导出。后续应建立适合该分支的独立环境，继续验证 PicoDet、PP-YOLOE、TinyPose，而非计入本轮通过数。

另有一个不计入真实网络的 INT64/BOOL 探针：大整数原样传递验证精度和全部输出检查；固定 Paddle/Paddle2ONNX 组合对大整数算术出现不一致时，必须被判为 export_mismatch。不能把 INT64 转成 FP32/FP64 来掩盖差异。

本轮没有构建或执行 InfiniTensor，也未在 A100 GPU 上运行设备测试。ONNX/参考通过说明模型已具备进入设备测试的条件，不代表 InfiniTensor 已实现其全部算子、dtype、形状和组合语义。随机参数验证也不代表 OCR/抠图任务精度。

## 后续飞桨来源优先级

| 仓库 | 候选价值 | 当前状态 |
|---|---|---|
| [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) | TTS 声学模型和 vocoder，可补充 FastSpeech2/HiFiGAN 等方向；官方有 ONNX 使用文档 | 下一批优先，未计数 |
| [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN) | 图像修复、图像转换、生成及视频处理 | 下一批优先，需逐项验证，未计数 |
| [PaddleTS](https://github.com/PaddlePaddle/PaddleTS) | 时序预测、异常检测和表示学习 | 候选，需明确多输入合同，未计数 |
| [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) | 语言模型及文本任务的不同实现 | 候选，需处理导出和 cache/控制输入，未计数 |
| [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) | 分类和视觉识别实现 | 分类已覆盖较多，优先级低于新任务，未计数 |
