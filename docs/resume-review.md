# PR #2 恢复逻辑与类别筛选审核修复

审核基线：815d3b2。本次修复三项 P2 问题，范围为采集/导出框架，不修改 InfiniTensor 执行器。

- `--resume` 接受成功项之前，同时核对 ONNX/external-data 包摘要和实际输入数组摘要。输入读取沿用参考验证的名称、shape、dtype 和有限值检查，数组摘要必须等于保存的 `validation.inputs_sha256`。NPZ 丢失、损坏、改值或证据缺失时不能直接沿用旧成功状态，交回正常 worker。合同输入可重建，外部非法输入须明确失败；等值数组更换 NPZ 压缩方式仍可复用。
- `timeout` / `crash` 默认重新执行，调大超时可保留已验证成功项并重试未完成项。相同签名的已完成 `failed` / `excluded` 诊断保留原有复用策略；强制重试使用 `--force` 或不使用 `--resume`。
- PyTorchCV 的姿态和语音网络分别使用现有的 `POSE_ESTIMATION`、`SPEECH_RECOGNITION` 类别，确保 catalog/collect/run 筛选与任务统计一致。已有 catalog 需重新采集以刷新类别；模型结构、任务清单和 1422 项统计均不变。

## 回归验证

新增 19 项定向测试，在修复前为 18 失败、1 通过；修复后全部通过。测试使用真实 Gather ONNX、ORT、采集器和受控子进程复现输入破坏、输入重建和恢复后的退出码/清单统计，不需要 Torch 即可运行。超时/崩溃重试保留完成项，9 个姿态/语音模块的发现与类别筛选均被覆盖。

基础安装环境完整测试：

```bash
python -m pytest tests/unit tests/operators -q
```

实际结果：`188 passed, 1 skipped in 70.94s`。唯一跳过项是缺少可选 Torch 时的真实 Torch 导出测试；新增恢复与类别测试均执行。

服务器固定 PyTorch 多任务环境：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/unit tests/operators tests/integration/test_diverse_sources.py -q
```

服务器日志连续末尾原文：

```text

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
194 passed, 28 warnings in 35.03s
```

真实 PyTorchCV 0.0.74 adapter 另核对：`lwopenpose2d_mobilenet_cmupan_coco` 可由姿态类别选中，`jasper5x3` 和 `quartznet5x5_en_ls` 可由语音类别选中；发现阶段不构建网络。

本机缺少 torchSR、torchaudio、Transformers，首次显式运行它们的集成检查时明确报依赖错误，随后在服务器固定依赖环境完成上述全部验证。本轮没有重新导出全部 1422 项，也没有进行 InfiniTensor 设备测试。

## 设备集成边界

目前 `runner/engine.py` 只写首个 FP32 输入、读取首个输出，且参考失败可能关闭比较继续执行。1422 项表示采集、导出和原框架/ONNX 比较通过，不代表主执行器已经验证设备正确；旧 `OK` 不能证明新增多输入、多输出和 INT64 合同已被正确消费。后续应优先接入 NPZ、全部输入输出比较和参考失败的明确失败状态。
