# InfiniTensor-Test

本分支交付 T01：来源按需加载、确定性采集和完整 ONNX 包缓存。
请先阅读[能力提升及复验命令](docs/pr-t01-collection.md)与[PR 审查说明](docs/pr-t01-review.md)。
设备执行、全部输出数值比较和严格门禁由后续 PR 交付。

> [InfiniTensor](https://github.com/InfiniTensor/InfiniTensor) 深度学习编译器的集成测试套件。

**语言 / Languages:** [English](README.md) · [简体中文](README.zh-CN.md)

## 概述

InfiniTensor-Test 是针对 [InfiniTensor](https://github.com/InfiniTensor/InfiniTensor)
深度学习编译器的独立集成测试套件，与 InfiniTensor 源码内随附的单元测试
为互补关系。

InfiniTensor 项目自带的单元测试覆盖其自身代码路径；本套件则面向"已编译
完成的 InfiniTensor 构建产物"，在真实与合成的 ONNX 模型上对其编译、运行与
数值正确性进行端到端验证。

本套件提供以下能力：

- **模型覆盖测试**：在 **2,000+** 个真实与合成 ONNX 模型上对 InfiniTensor
  进行编译与推理验证
- **正确性比对**：将 InfiniTensor 的输出与 **onnxruntime** 的输出进行数值
  对比，判定算子实现的正确性
- **延迟基准**：对代表性模型记录推理耗时，用于性能回归分析
- **对比报告**：以清晰的 Markdown / HTML 形式呈现两次运行之间的差异，
  可直接用于 Pull Request 描述
- **算子级测试**（新增）：将单个 ONNX 算子于多个 backend（InfiniTensor CUDA、
  InfiniTensor CPU、onnxruntime）上分别执行并交叉比对，明确给出当前构建所
  支持的算子集合及其正确性

本套件在设计上具有以下特性：

- **模块化**：模型来源（torchvision / timm / synthetic / huggingface / onnx zoo）
  与检查类型（正确性 / 覆盖率 / 性能）解耦，可任意组合
- **参数化**：`pytest` 根据目录中的模型自动发现并生成测试
- **可复现**：使用固定的随机种子与确定性配置
- **不污染代码仓库**：模型为运行时生成，**不**纳入版本控制，存放于
  1 TB 数据盘

---

## 前置条件

本套件以 [InfiniTensor](https://github.com/InfiniTensor/InfiniTensor) 编译产物为
运行依赖。**使用本套件前，须先在本机完成 InfiniTensor 的编译。**

### 1. 编译 InfiniTensor

```bash
git clone https://github.com/InfiniTensor/InfiniTensor.git
cd InfiniTensor

# 设置 CUDA 路径
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# 配置并编译。请将目标 GPU 架构加入 CUDA_ARCHITECTURES：
#   80 = A100,  86 = RTX 30xx,  89 = L4 / L20,  90 = H100
cmake -B build -S . -DCUDA_ARCHITECTURES="80;89" -DBACKEND=cuda
cmake --build build -j$(nproc)
```

编译完成后应生成：

- `pyinfinitensor/src/pyinfinitensor/backend.cpython-*.so`（C 扩展模块）
- `pyinfinitensor/src/pyinfinitensor/onnx.py` 中定义的 `OnnxStub` 类

### 2. 验证编译产物

```bash
PYINF_SRC="$(pwd)/pyinfinitensor/src"
python3 -c "
import sys; sys.path.insert(0, '${PYINF_SRC}')
import onnx, backend
from pyinfinitensor.onnx import OnnxStub
print('OK:', backend.__file__)
"
```

若输出形如 `OK: .../backend.cpython-312-x86_64-linux-gnu.so`，则编译产物可用。
若发生异常，通常原因为 `LD_LIBRARY_PATH` 或 `CUDA_HOME` 未正确设置。

### 3. 配置 InfiniTensor-Test

```bash
cp configs/local.yaml.example config.local.yaml
$EDITOR config.local.yaml
```

关键配置项如下：

```yaml
# 数据盘路径（用于存放模型与运行结果，建议至少 20 GB 可用空间）
data_dir: /home/<user>/minimax-work/data

# 已编译的 pyinfinitensor 所在目录
infinitensor_python_path: /home/<user>/InfiniTensor/pyinfinitensor/src
```

若 `data_dir` 已设置但未显式指定 `models_dir`、`results_dir` 或 `reports_dir`，
本套件将自动以其作为子目录锚定（`models`、`results`、`reports`）。

### 4. 从 Python 侧验证配置

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from infinitensor_test.config import load_settings
from infinitensor_test.runner.engine import TestEngine
s = load_settings()
engine = TestEngine(s, backend='cuda')   # 若无 GPU 可使用 'cpu'
print('Engine ready:', engine.settings.infinitensor_python_path)
"
```

若配置正确，命令将正常退出。异常信息中通常会指明具体缺失的导入或路径错误，
据此修正 `config.local.yaml` 后重试。

### InfiniTensor 尚未编译的过渡方案

本套件的**算子级**测试（详见 [docs/operator-testing.md](docs/operator-testing.md)）
在 InfiniTensor 未编译时仍可部分运行：以 `onnxruntime` 作为参考实现，验证算子
规格抽取与运行流程的正确性。InfiniTensor 相关列将统一显示为 `UNSUPPORTED`，
直至完成编译后重新运行。

**模型级**测试（`infinitensor-test-run`）须在 InfiniTensor 编译完成后方可使用，
否则所有测试将以 `compile_fail` 状态结束。

---

## 快速开始

> 完成上文 [前置条件](#前置条件) 全部步骤后继续。

```bash
# 1. 安装本套件（editable 模式）
pip install -e ".[full]"

# 2. 采集 smoke 套件模型（约 10 个 torchvision 分类器，< 30 秒）
infinitensor-test-collect --suite smoke

# 3. 运行 smoke 套件
infinitensor-test-run --suite smoke

# 4. 生成报告
infinitensor-test-report data/results/smoke_*.json -f html
```

运行完整 ~2,000 模型基准：

```bash
./scripts/run_all_suites.sh data/results/batch_$(date +%Y%m%d)
```

算子级测试的使用方法见 [docs/operator-testing.md](docs/operator-testing.md)。

---

## 架构

```
                ┌──────────────────────────┐
                │   Model Sources          │  ← torchvision, timm, hf, onnx_zoo, synthetic
                │   （插件）               │
                └────────────┬─────────────┘
                             │ ModelSpec
                             ▼
                ┌──────────────────────────┐
                │   Collector              │  ← 下载 / 导出 / 复制 ONNX 至
                │   （幂等）               │     data/models/
                └────────────┬─────────────┘
                             │ ModelCatalog
                             ▼
                ┌──────────────────────────┐
                │   TestEngine             │  ← load → compile → run → 比对
                │   （全仓库唯一导入       │     （全仓库唯一导入
                │    InfiniTensor 的模块） │      InfiniTensor 的模块）
                └────────────┬─────────────┘
                             │ TestResult[]
                             ▼
                ┌──────────────────────────┐
                │   Reports                │  ← Markdown + HTML + 对比
                └──────────────────────────┘
```

设计要点：

- **来源以插件形式注册**：新增 torchvision / timm / huggingface 等来源
  通常仅需 30 行代码
- **目录持久化**：模型采集结果保存于 `catalog.json`，无需重复采集
- **InfiniTensor 隔离导入**：`TestEngine` 是仓库内唯一导入 `pyinfinitensor` 与
  C++ `backend` 模块的位置。其余部分与 InfiniTensor 解耦，可独立测试
- **pytest 为可选依赖**：所有功能均可通过 CLI 使用，pytest 仅用于
  自动化测试与持续集成

---

## 测试套件

套件为来源过滤器、模型数量上限与检查类型的命名组合。

| 套件           | 说明                                              | 耗时     |
|----------------|---------------------------------------------------|----------|
| `smoke`        | 10 个 torchvision 分类器，用于冒烟测试             | < 30s    |
| `torchvision`  | 全部约 40 个 torchvision 分类器                    | ~1 min   |
| `timm_small`   | 全部小尺寸 timm 模型（≤30 MB）                     | ~3 min   |
| `synthetic`    | 全部约 1,700 个合成算子覆盖模型                    | ~3 min   |
| `correctness`  | 50 个小模型，与 onnxruntime 输出比对               | ~1 min   |
| `full`         | 全量集合：torchvision + timm + synthetic + hf + zoo | ~10 min  |
| `ci`           | 面向 CI 场景的小型正确性子集                       | < 60s    |

新增套件请编辑 `src/infinitensor_test/utils/suites.py`。

---

## 命令行工具

### `infinitensor-test-collect`

建立模型目录。模型于运行时生成并缓存至 `data/models/<source>/<name>.onnx`，
**不**纳入版本控制。

```bash
# 采集所有已注册来源
infinitensor-test-collect

# 指定来源
infinitensor-test-collect --sources synthetic_full torchvision

# 仅采集 torchvision，用于快速验证
infinitensor-test-collect --sources torchvision

# 强制重新导出
infinitensor-test-collect --sources synthetic_full --force

# 仅生成目录索引，不实际导出模型
infinitensor-test-collect --sources synthetic_full --dry-run

# 列出所有可用来源
infinitensor-test-collect --list-sources
```

### `infinitensor-test-run`

执行测试套件。每次运行生成一份 JSON 结果文件。

```bash
# 运行命名套件
infinitensor-test-run --suite smoke

# 自定义来源与数量
infinitensor-test-run --sources torchvision timm_small --limit 50

# 启用正确性比对
infinitensor-test-run --suite correctness --check-correctness

# 性能测试：每个模型执行 3 次
infinitensor-test-run --sources torchvision --iterations 3 --warmup 1
```

### `infinitensor-test-report`

将结果文件渲染为 Markdown 与 / 或 HTML 报告。

```bash
infinitensor-test-report data/results/full_abc123.json -f both
# → data/reports/full_abc123/report.html
# → data/reports/full_abc123/report.md
```

### `infinitensor-test-compare`

比较两份结果文件，用于 Pull Request 评审。当无回归时，命令以退出码 0
结束。

```bash
infinitensor-test-compare main.json pr.json
infinitensor-test-compare main.json pr.json -o report.md
```

---

## 新增模型来源

"来源"为可生成 `ModelSpec` 列表的函数。新增来源的流程为三步：

**1. 实现来源函数**于 `src/infinitensor_test/models/sources/`：

```python
# my_source.py
from infinitensor_test.models.spec import ModelSpec, ModelSource, ModelCategory
from infinitensor_test.models.registry import register_source

@register_source("my_source")
def my_source(params: dict) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="my_model_v1",
            source=ModelSource.MY,
            category=ModelCategory.CLASSIFICATION,
            factory=lambda: build_my_model(),
            input_shape=[1, 3, 224, 224],
        ),
    ]
```

**2. 于 `src/infinitensor_test/models/sources/__init__.py` 中导入该模块**（注册以
   导入副作用的形式完成）。

**3. 启用**：

```bash
infinitensor-test-collect --sources my_source
```

---

## 新增测试

测试以标准 `pytest` 函数形式编写。`tests/conftest.py` 中提供 `catalog` 与
`engine` 等 fixture：

```python
def test_my_special_model(engine, catalog_torchvision):
    """验证 resnet18 输出张量形状为 (1, 1000)。"""
    spec = next(s for s in catalog_torchvision if s.name == "tv_resnet18")
    r = engine.run_one(spec)
    assert r.status == TestStatus.OK
    assert r.output_shape == [1, 1000]
```

可通过 pytest markers（`correctness`、`coverage`、`performance`、`regression`、
`slow`、`ci`）选择测试子集：

```bash
pytest -m "not slow"        # 跳过慢速测试
pytest -m correctness      # 仅执行正确性测试
pytest -m "performance"    # 仅执行性能测试
```

---

## 算子级测试

模型级测试运行**完整计算图**。其结果可表示为"X% 的模型通过"，但无法
精确指出**哪一个算子**未通过。算子级测试弥补这一不足：对每个 ONNX 算子
（Conv、MatMul、Relu 等），分别于各已配置 backend 上执行一次，并将输出
与 `onnxruntime` 参考值进行数值比对。

**算子级测试并非单元测试**。其运行对象为真实的 `OnnxStub`、真实的 C++
`backend.so` 与真实的 `onnxruntime`。其目的在于将 InfiniTensor 构建的
**算子完备性与正确性**以可量化的方式呈现——当 InfiniTensor 被移植至新
GPU 平台时，可立即获得各算子在该平台上的支持情况。

```bash
# 1. 自模型目录抽取算子规格（约 80 类算子、7,000+ 条规格）
infinitensor-test-ops-collect --max-per-op 20

# 2. 于 cuda + cpu + onnxruntime 上执行全部规格
infinitensor-test-ops-run

# 3. 仅检查指定算子族
infinitensor-test-ops-run --op Conv MatMul --limit 50

# 4. 生成算子覆盖率矩阵报告
infinitensor-test-ops-report data/results/ops_<id>.json
```

报告示例：

```
| Op     | specs | cuda-ok | cpu-ok | or-ok | cuda-vs-or | cpu-vs-or |
|--------|-------|---------|--------|-------|------------|-----------|
| Conv   |    20 |     17  |    20  |   20  |  17 pass   | 20 pass   |
| Tile   |     3 |      3  |     3  |    3  |   0 pass   |  0 pass   |  ← 当前为占位实现
| Expand |     1 |      0  |     0  |    1  |   0 pass   |  0 pass   |  ← C++ 实现尚未提供
```

完整说明见 [docs/operator-testing.md](docs/operator-testing.md)。

---

## 报告与回归跟踪

每次执行 `infinitensor-test-run` 于 `data/results/` 下生成一份 JSON 文件。
`report` 命令行工具将 JSON 渲染为 Markdown / HTML 摘要。`compare` 命令行
工具接收两份 JSON 作为输入，输出二者之间的差异。

Pull Request 评审的标准流程为：

```bash
# 1. 于 main 分支生成基线
git checkout main
infinitensor-test-collect --sources synthetic_full  # 若尚未建立
infinitensor-test-run --suite full --backend cuda > data/results/baseline.json

# 2. 创建分支并实施修改
git checkout -b my-feature

# 3. 修改完成后重新运行
infinitensor-test-run --suite full --backend cuda > data/results/pr.json

# 4. 比较两份结果
infinitensor-test-compare data/results/baseline.json data/results/pr.json
```

`compare` 的输出可粘贴至 Pull Request 描述中。

---

## 面向 InfiniTensor 的 Pull Request 工作流

向 InfiniTensor 提交 Pull Request 时，建议遵循以下流程：

1. 选取一项聚焦的修改（编译修复、运行时修复、算子桩实现等），避免在单次
   提交中混入无关变更
2. 执行 `infinitensor-test-run --suite full > data/results/before.json`
3. 应用修改
4. 执行 `infinitensor-test-run --suite full > data/results/after.json`
5. 执行 `infinitensor-test-compare before.json after.json`，确认仅相关的
   模型状态发生变化
6. 提交修改（单一关注点）以及对比报告

此流程保证每个 Pull Request 具备清晰的修改边界与可验证的证据。

---

## 项目结构

```
InfiniTensor-Test/
├── src/infinitensor_test/
│   ├── config.py                # Settings、路径解析、env 覆盖
│   ├── models/
│   │   ├── spec.py              # ModelSpec, ModelCategory, ModelSource
│   │   ├── catalog.py           # ModelCatalog：持久化、可过滤
│   │   ├── registry.py          # @register_source 装饰器
│   │   ├── collector.py         # 下载 / 导出 ONNX 至本地
│   │   └── sources/             # 各来源独立文件
│   │       ├── torchvision_source.py
│   │       ├── timm_source.py
│   │       ├── synthetic_source.py
│   │       ├── huggingface_source.py
│   │       └── onnx_zoo_source.py
│   ├── runner/
│   │   ├── engine.py            # TestEngine（仓库内唯一导入 InfiniTensor）
│   │   ├── result.py            # TestResult, TestStatus, RunSummary
│   │   └── subprocess_runner.py # 子进程隔离版本，避免 CUDA 内存堆积
│   ├── operators/               # 算子级测试框架
│   │   ├── spec.py              # OperatorSpec, InputSpec, BackendOutput
│   │   ├── generator.py         # 自 ONNX 模型抽取算子规格
│   │   ├── catalog.py           # 算子目录
│   │   ├── backends.py          # Backend 基类及 3 个内置 backend
│   │   ├── runner.py            # OperatorTestRunner（跨 backend 比对）
│   │   ├── report.py            # 算子覆盖率矩阵报告
│   │   └── scripts/             # 3 个命令行工具
│   ├── report/
│   │   ├── markdown_report.py
│   │   ├── html_report.py
│   │   └── comparator.py        # ResultComparator
│   ├── utils/
│   │   └── suites.py            # 命名套件配置
│   └── scripts/                 # 命令行工具入口
│       ├── collect.py           # infinitensor-test-collect
│       ├── run.py               # infinitensor-test-run
│       ├── report.py            # infinitensor-test-report
│       └── compare.py           # infinitensor-test-compare
├── tests/
│   ├── conftest.py              # pytest fixtures
│   ├── unit/                    # 不依赖 InfiniTensor
│   ├── operators/               # 算子框架的健全性测试
│   │   └── test_framework.py
│   ├── integration/             # 依赖 InfiniTensor
│   │   ├── test_correctness.py
│   │   ├── test_coverage.py
│   │   └── test_op_subset.py
│   ├── performance/
│   │   └── test_performance.py
│   └── regression/
│       ├── test_regression.py
│       └── goldens/             # 基线计数
├── examples/                    # 独立示例脚本
├── configs/                     # 默认配置模板
├── scripts/                     # shell 辅助脚本
├── docs/                        # 设计文档
│   ├── operator-testing.md      # 算子级测试
│   ├── pr-sequences.md          # InfiniTensor Pull Request 拆分策略
│   └── ...
└── .github/workflows/           # 持续集成配置
```

---

## 故障排查

### "Cannot import InfiniTensor"

`config.local.yaml` 中 `infinitensor_python_path` 指向不正确。该字段应
指向包含 `pyinfinitensor/` 的目录，使得以下导入可成功执行：

```python
import pyinfinitensor
from pyinfinitensor.onnx import OnnxStub
import backend
```

参见上文 [前置条件 - 验证编译产物](#2-验证编译产物) 一节。

### 磁盘空间不足

模型存放于 `data_dir`（默认 1 TB 数据盘）下。若数据盘未挂载，应将
`data_dir` 指向具备至少 20 GB 可用空间的目录。可同步下调
`max_model_size_bytes` 以缩减采集范围。

### "Most synthetic models fail with KeyError: y"

此为 InfiniTensor 测试基础设施中的一处真实缺陷，已记录在案。本仓库的
合成模型在构造时令 `graph.output[0].name` 与节点实际输出一致，从而规避
此问题。

### "onnxruntime doesn't support op X"

部分算子（如 `LpPool`、`ReduceL1/L2`）在 onnxruntime CPU 与 InfiniTensor
之间的行为存在差异。相关算子的合成源在 `pyinfinitensor/onnx.py` 中
使用占位实现，正确性测试将自动跳过此类算子。

### 算子测试中各列均为 0 通过

通常表示 `infinitensor_python_path` 指向不正确，或编译产物 `.so` 文件
未位于所指定目录的子目录中。参见上文 [前置条件 - 验证编译产物](#2-验证编译产物)
一节。

### 测试结果不稳定

cait、coat、gcvit 等模型在多次运行间可能呈现非确定性失败，其原因为
GPU 状态污染——`TestEngine` 于同一进程中顺序运行多个模型，而 CUDA
不释放已分配内存。`SubprocessTestRunner`
（`src/infinitensor_test/runner/subprocess_runner.py`）为每个模型启动
独立的 Python 进程，可彻底消除该问题。`infinitensor-test-run` 所使用的
`SubprocessTestEngine` 默认采用此模式。

---

## 许可证

本项目以 MIT 许可证发布。
