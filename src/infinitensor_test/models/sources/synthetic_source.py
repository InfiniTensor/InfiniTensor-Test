"""Synthetic ONNX model source.

These are hand-crafted ONNX models that exhaustively cover operator
combinations. They are designed to:

- exercise every ONNX op at least once
- vary shape, axis, and keepdims parameters
- combine ops into small templates (MLP, CNN block, ResBlock, ...)

Synthetic models are written directly to ONNX (not via torch.onnx.export)
which makes them stable across torch versions and avoids dynamic-control-
flow issues in real torch models.

There are several "block" sources so they can be selected independently:

  - synthetic_shape      (Reshape, Flatten, Transpose, ...)
  - synthetic_elementwise (Add, Sub, Mul, Div, ...)
  - synthetic_matmul     (MatMul, Gemm)
  - synthetic_activation (Relu, Sigmoid, Tanh, Softmax, ...)
  - synthetic_pool       (MaxPool, AveragePool, Global*, LpPool)
  - synthetic_reduction  (ReduceMean/Sum/Max/.../L1/L2/LogSum* + Mean)
  - synthetic_slice      (Slice, Gather, Concat, Split)
  - synthetic_norm       (LayerNorm, BatchNorm, InstanceNorm)
  - synthetic_template   (VGG-block, ResNet-block, Inception, MLP, U-Net, Attention)
  - synthetic_full       (all of the above)
"""

from __future__ import annotations

import itertools
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

from infinitensor_test.models.spec import (
    ModelSpec,
    ModelCategory,
    ModelSource,
)
from infinitensor_test.models.registry import register_source, SOURCE_SEED
from infinitensor_test.artifacts import digest, stable_seed


def _random(shape, case_id, parameter):
    """One RNG per case/parameter; aliases and discovery order have no effect."""
    identity = f"synthetic:synth_{case_id}:{parameter}"
    return np.random.default_rng(stable_seed(SOURCE_SEED.get(), identity)).random(shape)


# ---------------------------------------------------------------------------
# ONNX building helpers
# ---------------------------------------------------------------------------

def _save_model(name: str, nodes, inits, in_shape, out_shape) -> onnx.ModelProto:
    """Build a model with the last node's output renamed to 'y' (so that
    the OnnxStub in InfiniTensor can find tensors['y'] when the graph
    declares it as an output)."""
    nodes, inits = deepcopy(nodes), deepcopy(inits)
    if nodes:
        nodes[-1].output[0] = "y"
    graph = helper.make_graph(
        nodes,
        f"g_{name}",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in nodes[-1].output],
        inits,
    )
    # Use the IR required by our opset instead of the installed ONNX release's
    # newest IR, which may not yet be supported by ONNX Runtime.
    model = helper.make_model_gen_version(graph, opset_imports=[helper.make_opsetid("", 18)])
    return model


def _t(arr, name):
    """Make a numpy initializer."""
    return numpy_helper.from_array(arr.astype(np.float32), name)


def _i(arr, name):
    """int64 initializer."""
    return numpy_helper.from_array(arr.astype(np.int64), name)


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


# ---------------------------------------------------------------------------
# Block 1: Shape ops
# ---------------------------------------------------------------------------

def _shape_specs() -> List[ModelSpec]:
    out: List[ModelSpec] = []
    # Reshape (element-count preserving) — name includes the shape so each spec is unique
    for in_s in [(1, 3, 224, 224), (1, 8, 8, 8), (1, 16, 16), (1, 8, 8)]:
        e = _numel(in_s)
        for out_s in [[1, e], [e, 1], [1, 1, e], [1, e, 1, 1], [1, 2, e // 2], [1, e // 4, 4]]:
            if _numel(out_s) != e:
                continue
            inits = [_i(np.array(out_s), "s")]
            nodes = [helper.make_node("Reshape", ["x", "s"], ["y"])]
            name = f"reshape_{in_s}_to_{out_s}"
            out.append(_spec(name, in_s, out_s, inits, nodes, ModelCategory.SHAPE_OPS))
    # Flatten
    for in_s in [(1, 3, 8, 8), (1, 16, 16, 16), (1, 8, 8, 8, 8)]:
        for axis in [1, 2, 3]:
            nodes = [helper.make_node("Flatten", ["x"], ["y"], axis=axis)]
            out.append(_spec(f"flatten_{in_s}_a{axis}", in_s, None, [], nodes, ModelCategory.SHAPE_OPS))
    # Transpose
    for in_s in [(1, 3, 8, 8), (1, 8, 8, 16)]:
        for perm in [[0, 2, 3, 1], [0, 1, 3, 2], [3, 2, 1, 0]]:
            nodes = [helper.make_node("Transpose", ["x"], ["y"], perm=perm)]
            out.append(_spec(f"transpose_{in_s}_p{perm}", in_s, None, [], nodes, ModelCategory.SHAPE_OPS))
    # Squeeze / Unsqueeze
    for in_s in [(1, 1, 8, 8), (1, 3, 1, 8), (1, 8, 1, 1)]:
        for axes in [[0], [1], [2], [3], [0, 1], [1, 3]]:
            nodes = [helper.make_node("Squeeze", ["x", "axes"], ["y"])]
            reason = "squeeze_nonunit_axis" if any(in_s[a] != 1 for a in axes) else None
            out.append(_spec(f"squeeze_{in_s}_a{axes}", in_s, None,
                             [_i(np.array(axes), "axes")], nodes, ModelCategory.SHAPE_OPS, reason))
    for in_s in [(1, 8, 8), (8, 8, 8), (3, 8)]:
        for axes in [[0], [1], [2], [0, 1], [0, 2]]:
            nodes = [helper.make_node("Unsqueeze", ["x", "axes"], ["y"])]
            out.append(_spec(f"unsqueeze_{in_s}_a{axes}", in_s, None,
                             [_i(np.array(axes), "axes")], nodes, ModelCategory.SHAPE_OPS))
    # Identity
    for in_s in [(1, 8, 8), (1, 3, 224, 224), (8,)]:
        nodes = [helper.make_node("Identity", ["x"], ["y"])]
        out.append(_spec(f"identity_{in_s}", in_s, None, [], nodes, ModelCategory.SHAPE_OPS))
    return out


# ---------------------------------------------------------------------------
# Block 2: Elementwise
# ---------------------------------------------------------------------------

def _elementwise_specs() -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for op in ["Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Mean"]:
        for in_s in [(1, 8, 8), (1, 16, 16), (1, 32, 32), (1, 64, 64)]:
            c = (_random(in_s, f"{op}_{in_s}", "c") * 0.1 + 0.5).astype(np.float32)
            inits = [_t(c, "c")]
            nodes = [helper.make_node(op, ["x", "c"], ["y"])]
            out.append(_spec(f"{op}_{in_s}", in_s, None, inits, nodes, ModelCategory.ELEMENTWISE))
    for op in ["Neg", "Abs", "Reciprocal", "Sqrt", "Exp", "Log"]:
        for in_s in [(1, 8, 8), (1, 16, 16), (1, 32, 32)]:
            nodes = [helper.make_node(op, ["x"], ["y"])]
            out.append(_spec(f"{op}_{in_s}", in_s, None, [], nodes, ModelCategory.ELEMENTWISE))
    for op in ["Sin", "Cos", "Tan", "Sinh", "Cosh"]:
        for in_s in [(1, 8, 8), (1, 16, 16)]:
            nodes = [helper.make_node(op, ["x"], ["y"])]
            out.append(_spec(f"{op}_{in_s}", in_s, None, [], nodes, ModelCategory.ELEMENTWISE))
    return out


# ---------------------------------------------------------------------------
# Block 3: MatMul / Gemm
# ---------------------------------------------------------------------------

def _matmul_specs() -> List[ModelSpec]:
    out: List[ModelSpec] = []
    shapes = [
        (8, 8, 8), (8, 16, 8), (16, 8, 16), (4, 8, 16),
        (1, 8, 1), (1, 1, 8), (32, 32, 32), (64, 64, 64),
        (128, 128, 128), (16, 32, 16), (8, 8, 16),
    ]
    for m, k, n_dim in shapes:
        case = f"matmul_{m}x{k}x{n_dim}_in_{(1, m)}"
        w = (_random([k, n_dim], case, "w") * 0.1).astype(np.float32)
        b = (_random(n_dim, case, "b") * 0.1).astype(np.float32)
        inits = [_t(w, "w"), _t(b, "b")]
        nodes = [
            helper.make_node("MatMul", ["x", "w"], ["mm"]),
            helper.make_node("Add", ["mm", "b"], ["y"]),
        ]
        # Preserve the legacy case ID; the authoritative dimensions are input_shape.
        out.append(_spec(case, [m, k], [m, n_dim], inits, nodes, ModelCategory.MLP))
    # Gemm with various alpha/beta/transB
    for alpha, beta, transB in [(1.0, 1.0, 0), (0.5, 0.5, 1), (2.0, 0.0, 0), (1.0, 1.0, 1)]:
        w = (_random([8, 8], f"gemm_a{alpha}_b{beta}_t{transB}", "w") * 0.1).astype(np.float32)
        b = np.zeros(8, dtype=np.float32)
        inits = [_t(w, "w"), _t(b, "b")]
        nodes = [helper.make_node("Gemm", ["x", "w", "b"], ["y"],
                                   alpha=alpha, beta=beta, transB=transB)]
        out.append(_spec(f"gemm_a{alpha}_b{beta}_t{transB}", [1, 8], None, inits, nodes, ModelCategory.MLP))
    return out


# ---------------------------------------------------------------------------
# Block 4: Activations
# ---------------------------------------------------------------------------

def _activation_specs() -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for op in ["Relu", "Sigmoid", "Tanh", "LeakyRelu", "Elu", "HardSwish", "HardSigmoid",
               "Softsign", "Softplus", "ThresholdedRelu"]:
        for in_s in [(1, 8, 8), (1, 16, 16), (1, 32, 32), (1, 3, 224, 224), (8,)]:
            attrs = {}
            if op == "LeakyRelu":
                attrs["alpha"] = 0.1
            elif op == "Elu":
                attrs["alpha"] = 1.0
            elif op == "HardSigmoid":
                attrs["alpha"] = 0.2
                attrs["beta"] = 0.5
            elif op == "ThresholdedRelu":
                attrs["alpha"] = 1.0
            nodes = [helper.make_node(op, ["x"], ["y"], **attrs)]
            out.append(_spec(f"{op}_{in_s}", in_s, None, [], nodes, ModelCategory.ACTIVATION))
    for op in ["Softmax", "LogSoftmax"]:
        for axis in [0, 1, 2, -1, -2]:
            for in_s in [(1, 8, 8), (1, 16, 16), (1, 3, 8)]:
                nodes = [helper.make_node(op, ["x"], ["y"], axis=axis)]
                out.append(_spec(f"{op}_{in_s}_a{axis}", in_s, None, [], nodes, ModelCategory.ACTIVATION))
    for in_s in [(1, 8, 8), (1, 3, 224, 224)]:
        inits = [_t(np.array(0.0, dtype=np.float32), "min"),
                 _t(np.array(6.0, dtype=np.float32), "max")]
        nodes = [helper.make_node("Clip", ["x", "min", "max"], ["y"])]
        out.append(_spec(f"clip_{in_s}", in_s, None, inits, nodes, ModelCategory.ACTIVATION))
    return out


# ---------------------------------------------------------------------------
# Block 5: Pool
# ---------------------------------------------------------------------------

def _pool_specs() -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for op in ["MaxPool", "AveragePool"]:
        for k in [2, 3, 5]:
            for s in [1, 2, 3]:
                for p in [0, 1, 2]:
                    nodes = [helper.make_node(op, ["x"], ["y"],
                                               kernel_shape=[k, k], strides=[s, s],
                                               pads=[p, p, p, p])]
                    out.append(_spec(f"{op}_k{k}_s{s}_p{p}",
                                      [1, 8, 16, 16], None, [], nodes, ModelCategory.POOLING,
                                      "pool_padding_out_of_range" if p >= k else None))
    for op in ["GlobalAveragePool", "GlobalMaxPool"]:
        for in_s in [(1, 4, 8, 8), (1, 8, 16, 16), (1, 16, 32, 32)]:
            nodes = [helper.make_node(op, ["x"], ["y"])]
            out.append(_spec(f"{op}_{in_s}", in_s, None, [], nodes, ModelCategory.POOLING))
    return out


# ---------------------------------------------------------------------------
# Block 6: Reduction
# ---------------------------------------------------------------------------

def _reduction_specs() -> List[ModelSpec]:
    out: List[ModelSpec] = []
    reduce_ops = ["ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd",
                  "ReduceSumSquare", "ReduceL1", "ReduceL2",
                  "ReduceLogSum", "ReduceLogSumExp"]
    axes_list = [[0], [1], [2], [3], [-1], [-2], [0, 1], [1, 2], [2, 3], [-2, -1]]
    for op in reduce_ops:
        for axes in axes_list:
            for keepdims in [0, 1]:
                inits = [_i(np.array(axes, dtype=np.int64), "ax")]
                nodes = [helper.make_node(op, ["x", "ax"], ["y"], keepdims=keepdims)]
                out.append(_spec(f"{op}_a{axes}_k{keepdims}_s{(1, 4, 8, 8)}",
                                  [1, 4, 8, 8], None, inits, nodes, ModelCategory.REDUCTION))
    return out


# ---------------------------------------------------------------------------
# Block 7: Slice / Gather / Concat / Split
# ---------------------------------------------------------------------------

def _slice_specs() -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for in_s in [(1, 8, 16), (1, 4, 16, 16), (1, 16, 16, 16)]:
        axes = list(range(len(in_s)))
        starts = [0] * len(in_s)
        ends = [max(1, s // 2) for s in in_s]
        inits = [_i(np.array(axes, dtype=np.int64), "ax"),
                 _i(np.array(starts, dtype=np.int64), "st"),
                 _i(np.array(ends, dtype=np.int64), "en"),
                 _i(np.array([1] * len(in_s), dtype=np.int64), "stp")]
        nodes = [helper.make_node("Slice", ["x", "st", "en", "ax", "stp"], ["y"])]
        out.append(_spec(f"slice_{in_s}", in_s, None, inits, nodes, ModelCategory.SHAPE_OPS))
    for in_s in [(8, 8), (16, 8), (4, 8, 8)]:
        idx = np.array([0, 1, 2], dtype=np.int64)
        inits = [_i(idx, "i")]
        nodes = [helper.make_node("Gather", ["x", "i"], ["y"], axis=0)]
        out.append(_spec(f"gather_{in_s}", in_s, None, inits, nodes, ModelCategory.SHAPE_OPS))
    for axis in [0, 1, 2, 3]:
        for in_s in [(1, 4, 8, 8), (1, 8, 8, 8), (1, 16, 16, 16)]:
            c1 = (_random(in_s, f"concat_a{axis}_{in_s}", "c1") * 0.1).astype(np.float32)
            c2 = (_random(in_s, f"concat_a{axis}_{in_s}", "c2") * 0.1).astype(np.float32)
            inits = [_t(c1, "c1"), _t(c2, "c2")]
            nodes = [helper.make_node("Concat", ["c1", "c2"], ["y"], axis=axis)]
            out.append(_spec(f"concat_a{axis}_{in_s}", in_s, None, inits, nodes, ModelCategory.SHAPE_OPS))
    for axis in [0, 1, 2, 3]:
        for in_s in [(1, 8, 8, 8), (1, 4, 4, 4)]:
            nodes = [helper.make_node("Split", ["x"], ["y", "y2"], axis=axis, num_outputs=2)]
            out.append(_spec(f"split_a{axis}_{in_s}", in_s, None, [], nodes, ModelCategory.SHAPE_OPS,
                             "split_outputs_exceed_dimension" if in_s[axis] < 2 else None))
    return out


# ---------------------------------------------------------------------------
# Block 8: Templates (VGG, ResBlock, Inception, MLP, U-Net, Attention)
# ---------------------------------------------------------------------------

def _template_specs() -> List[ModelSpec]:
    out: List[ModelSpec] = []
    # VGG block
    for c1, c2 in [(4, 8), (8, 16), (16, 32), (32, 64)]:
        w1 = (_random([c1, 3, 3, 3], f"vggblock_{c1}_{c2}", "w1") * 0.1).astype(np.float32)
        b1 = np.zeros(c1, dtype=np.float32)
        w2 = (_random([c2, c1, 3, 3], f"vggblock_{c1}_{c2}", "w2") * 0.1).astype(np.float32)
        b2 = np.zeros(c2, dtype=np.float32)
        inits = [_t(w1, "w1"), _t(b1, "b1"), _t(w2, "w2"), _t(b2, "b2")]
        nodes = [
            helper.make_node("Conv", ["x", "w1", "b1"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Relu", ["c1"], ["r1"]),
            helper.make_node("Conv", ["r1", "w2", "b2"], ["c2"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Relu", ["c2"], ["r2"]),
            helper.make_node("MaxPool", ["r2"], ["y"], kernel_shape=[2, 2], strides=[2, 2]),
        ]
        out.append(_spec(f"vggblock_{c1}_{c2}", [1, 3, 16, 16], [1, c2, 8, 8],
                          inits, nodes, ModelCategory.CNN_BLOCK))
    # ResBlock
    for c in [4, 8, 16, 32]:
        w1 = (_random([c, c, 3, 3], f"resblock_c{c}", "w1") * 0.05).astype(np.float32)
        b1 = np.zeros(c, dtype=np.float32)
        w2 = (_random([c, c, 3, 3], f"resblock_c{c}", "w2") * 0.05).astype(np.float32)
        b2 = np.zeros(c, dtype=np.float32)
        s = (_random(c, f"resblock_c{c}", "scale") * 0.1 + 0.5).astype(np.float32)
        bo = np.zeros(c, dtype=np.float32)
        m = np.zeros(c, dtype=np.float32)
        v = np.ones(c, dtype=np.float32)
        inits = [
            _t(w1, "w1"), _t(b1, "b1"), _t(w2, "w2"), _t(b2, "b2"),
            _t(s, "s1"), _t(bo, "bo1"), _t(m, "m1"), _t(v, "v1"),
            _t(s, "s2"), _t(bo, "bo2"), _t(m, "m2"), _t(v, "v2"),
        ]
        nodes = [
            helper.make_node("Conv", ["x", "w1", "b1"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("BatchNormalization", ["c1", "s1", "bo1", "m1", "v1"], ["bn1"], epsilon=1e-5),
            helper.make_node("Relu", ["bn1"], ["r1"]),
            helper.make_node("Conv", ["r1", "w2", "b2"], ["c2"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("BatchNormalization", ["c2", "s2", "bo2", "m2", "v2"], ["bn2"], epsilon=1e-5),
            helper.make_node("Add", ["bn2", "x"], ["add"]),
            helper.make_node("Relu", ["add"], ["y"]),
        ]
        out.append(_spec(f"resblock_c{c}", [1, c, 8, 8], None, inits, nodes, ModelCategory.CNN_BLOCK))
    # Inception
    for c in [4, 8, 16]:
        w1 = (_random([c, c, 1, 1], f"inception_c{c}", "w1") * 0.1).astype(np.float32)
        w3 = (_random([c, c, 3, 3], f"inception_c{c}", "w3") * 0.1).astype(np.float32)
        w5 = (_random([c, c, 5, 5], f"inception_c{c}", "w5") * 0.1).astype(np.float32)
        b = np.zeros(c, dtype=np.float32)
        inits = [_t(w1, "w1"), _t(b, "b1"), _t(w3, "w3"), _t(b, "b3"),
                 _t(w5, "w5"), _t(b, "b5")]
        nodes = [
            helper.make_node("Conv", ["x", "w1", "b1"], ["o1"], kernel_shape=[1, 1]),
            helper.make_node("Relu", ["o1"], ["r1"]),
            helper.make_node("Conv", ["x", "w3", "b3"], ["o3"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Relu", ["o3"], ["r3"]),
            helper.make_node("Conv", ["x", "w5", "b5"], ["o5"], kernel_shape=[5, 5], pads=[2, 2, 2, 2]),
            helper.make_node("Relu", ["o5"], ["r5"]),
            helper.make_node("MaxPool", ["x"], ["mp"], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]),
            helper.make_node("Concat", ["r1", "r3", "r5", "mp"], ["y"], axis=1),
        ]
        out.append(_spec(f"inception_c{c}", [1, c, 8, 8], [1, 4 * c, 8, 8],
                          inits, nodes, ModelCategory.CNN_BLOCK))
    # MLP (varied depth/dim)
    for depth, dim in [(1, 8), (2, 16), (3, 32), (4, 64), (2, 8), (3, 16), (4, 32), (5, 64)]:
        inits = []
        nodes = []
        cur = "x"
        for i in range(depth):
            w = (_random([dim if i > 0 else 8, dim], f"mlp_d{depth}_d{dim}", f"w{i}") * 0.1).astype(np.float32)
            b = np.zeros(dim, dtype=np.float32)
            inits += [_t(w, f"w{i}"), _t(b, f"b{i}")]
            nodes.append(helper.make_node("MatMul", [cur, f"w{i}"], [f"m{i}"]))
            nodes.append(helper.make_node("Add", [f"m{i}", f"b{i}"], [f"a{i}"]))
            nodes.append(helper.make_node("Relu", [f"a{i}"], [f"r{i}"]))
            cur = f"r{i}"
        out.append(_spec(f"mlp_d{depth}_d{dim}", [1, 8], None, inits, nodes, ModelCategory.MLP))
    # Attention block
    for d in [16, 32, 64]:
        wq = (_random([d, d], f"attn_d{d}", "wq") * 0.05).astype(np.float32)
        wk = (_random([d, d], f"attn_d{d}", "wk") * 0.05).astype(np.float32)
        wv = (_random([d, d], f"attn_d{d}", "wv") * 0.05).astype(np.float32)
        wo = (_random([d, d], f"attn_d{d}", "wo") * 0.05).astype(np.float32)
        b = np.zeros(d, dtype=np.float32)
        inits = [_t(np.array(1.0 / np.sqrt(d)), "scale"),
                 _t(wq, "wq"), _t(b, "bq"),
                 _t(wk, "wk"), _t(b, "bk"),
                 _t(wv, "wv"), _t(b, "bv"),
                 _t(wo, "wo"), _t(b, "bo")]
        nodes = [
            helper.make_node("MatMul", ["x", "wq"], ["q"]),
            helper.make_node("Add", ["q", "bq"], ["qa"]),
            helper.make_node("MatMul", ["x", "wk"], ["k"]),
            helper.make_node("Add", ["k", "bk"], ["ka"]),
            helper.make_node("MatMul", ["x", "wv"], ["v"]),
            helper.make_node("Add", ["v", "bv"], ["va"]),
            helper.make_node("Transpose", ["ka"], ["kt"], perm=[0, 2, 1]),
            helper.make_node("MatMul", ["qa", "kt"], ["raw_scores"]),
            helper.make_node("Mul", ["raw_scores", "scale"], ["scores"]),
            helper.make_node("Softmax", ["scores"], ["attn"], axis=-1),
            helper.make_node("MatMul", ["attn", "va"], ["ctx"]),
            helper.make_node("MatMul", ["ctx", "wo"], ["co"]),
            helper.make_node("Add", ["co", "bo"], ["y"]),
        ]
        out.append(_spec(f"attn_d{d}", [1, 4, d], None, inits, nodes, ModelCategory.TRANSFORMER))
    return out


# ---------------------------------------------------------------------------
# Block 9: Norm
# ---------------------------------------------------------------------------

def _norm_specs() -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for c in [4, 8, 16, 32]:
        # LayerNorm: enumerate shape variants explicitly so each spec is unique
        ln_shapes = [
            (0, [1, c, 8]),
            (1, [1, 8, c]),
            (2, [1, c, c]),
            (3, [c, c]),
        ]
        for variant_idx, shape in ln_shapes:
            scale = _random([shape[-1]], f"layernorm_c{c}_v{variant_idx}", "scale").astype(np.float32)
            bias = np.zeros([shape[-1]], dtype=np.float32)
            inits = [_t(scale, "s"), _t(bias, "b")]
            nodes = [helper.make_node("LayerNormalization", ["x", "s", "b"], ["y"],
                                       axis=-1, epsilon=1e-5)]
            out.append(_spec(f"layernorm_c{c}_v{variant_idx}", shape, None, inits, nodes, ModelCategory.NORM))
        # BatchNorm
        for variant_idx, shape in enumerate([[1, c, 8, 8], [1, c, 16, 16]]):
            scale = (_random(c, f"batchnorm_c{c}_v{variant_idx}", "scale") * 0.1 + 0.5).astype(np.float32)
            bias = np.zeros(c, dtype=np.float32)
            mean = np.zeros(c, dtype=np.float32)
            var = np.ones(c, dtype=np.float32)
            inits = [_t(scale, "s"), _t(bias, "b"), _t(mean, "m"), _t(var, "v")]
            nodes = [helper.make_node("BatchNormalization", ["x", "s", "b", "m", "v"], ["y"],
                                       epsilon=1e-5, momentum=0.9)]
            out.append(_spec(f"batchnorm_c{c}_v{variant_idx}", shape, None, inits, nodes, ModelCategory.NORM))
    for c in [4, 8]:
        scale = _random(c, f"instnorm_c{c}_v0", "scale").astype(np.float32)
        bias = np.zeros(c, dtype=np.float32)
        inits = [_t(scale, "s"), _t(bias, "b")]
        nodes = [helper.make_node("InstanceNormalization", ["x", "s", "b"], ["y"], epsilon=1e-5)]
        out.append(_spec(f"instnorm_c{c}_v0", [1, c, 8, 8], None, inits, nodes, ModelCategory.NORM))
    return out


# ---------------------------------------------------------------------------
# Spec dispatch
# ---------------------------------------------------------------------------

def _spec(name: str, in_shape, out_shape, inits, nodes, category: ModelCategory,
          rejection_reason=None) -> ModelSpec:
    """Wrap a synthetic ONNX construction into a ModelSpec.

    The factory closes over the parameters; the collector calls it and writes
    the resulting ONNX to disk.
    """
    fname = f"synth_{name}"
    in_shape = list(in_shape) if in_shape is not None else None
    out_shape = list(out_shape) if out_shape is not None else in_shape

    def factory():
        m = _save_model(name, nodes, inits, in_shape, out_shape)
        if not rejection_reason:
            m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
        return ("onnx", m)

    return ModelSpec(
        name=fname,
        source=ModelSource.SYNTHETIC,
        category=category,
        factory=factory,
        input_shape=in_shape,
        tags=["synthetic", category.value],
        size_hint_bytes=0,
        expected_rejection=bool(rejection_reason),
        rejection_reason=rejection_reason,
        metadata={
            "generator_version": 3,
            "definition_sha256": digest({
                "inputs": in_shape, "outputs": out_shape,
                "nodes": [hashlib.sha256(n.SerializeToString()).hexdigest() for n in nodes],
                "constants": [hashlib.sha256(t.SerializeToString()).hexdigest() for t in inits],
            }),
        },
    )


# Each block exposed as its own source so tests can opt into a subset.
@register_source("synthetic_shape")
def synthetic_shape_specs(params: dict) -> List[ModelSpec]:
    return _shape_specs()


@register_source("synthetic_elementwise")
def synthetic_elementwise_specs(params: dict) -> List[ModelSpec]:
    return _elementwise_specs()


@register_source("synthetic_matmul")
def synthetic_matmul_specs(params: dict) -> List[ModelSpec]:
    return _matmul_specs()


@register_source("synthetic_activation")
def synthetic_activation_specs(params: dict) -> List[ModelSpec]:
    return _activation_specs()


@register_source("synthetic_pool")
def synthetic_pool_specs(params: dict) -> List[ModelSpec]:
    return _pool_specs()


@register_source("synthetic_reduction")
def synthetic_reduction_specs(params: dict) -> List[ModelSpec]:
    return _reduction_specs()


@register_source("synthetic_slice")
def synthetic_slice_specs(params: dict) -> List[ModelSpec]:
    return _slice_specs()


@register_source("synthetic_template")
def synthetic_template_specs(params: dict) -> List[ModelSpec]:
    return _template_specs()


@register_source("synthetic_norm")
def synthetic_norm_specs(params: dict) -> List[ModelSpec]:
    return _norm_specs()


@register_source("synthetic_full")
def synthetic_full_specs(params: dict) -> List[ModelSpec]:
    """All synthetic blocks combined."""
    return (
        _shape_specs()
        + _elementwise_specs()
        + _matmul_specs()
        + _activation_specs()
        + _pool_specs()
        + _reduction_specs()
        + _slice_specs()
        + _template_specs()
        + _norm_specs()
    )
