"""Per-operator tests.

These tests pin down the behavior of *individual* operators. They run
small synthetic models that exercise a single op and check the result.

This is useful for:
- quickly pinpointing a regression in a specific op
- documenting the expected behavior of each op
- having targeted tests for ops that are commonly broken in compilers
"""

from __future__ import annotations

import pytest
import numpy as np

from infinitensor_test.runner.result import TestStatus


pytestmark = [pytest.mark.integration, pytest.mark.correctness]


# ----- Shape ops -----

def test_reshape_preserves_count(engine, catalog_synthetic):
    """Reshape should preserve the total number of elements."""
    specs = [s for s in catalog_synthetic if "reshape" in s.name]
    assert specs, "no reshape specs"
    r = engine.run_one(specs[0])
    assert r.status == TestStatus.OK, r.error_message
    if r.input_shape and r.output_shape:
        in_elems = 1
        for d in r.input_shape: in_elems *= d
        out_elems = 1
        for d in r.output_shape: out_elems *= d
        assert in_elems == out_elems, \
            f"reshape changed element count: {r.input_shape} -> {r.output_shape}"


def test_transpose_preserves_count(engine, catalog_synthetic):
    """Transpose should preserve the total number of elements."""
    specs = [s for s in catalog_synthetic if "transpose" in s.name]
    assert specs, "no transpose specs"
    r = engine.run_one(specs[0])
    assert r.status == TestStatus.OK, r.error_message
    if r.input_shape and r.output_shape:
        assert sorted(r.input_shape) == sorted(r.output_shape)


# ----- Activations -----

@pytest.mark.parametrize("op", ["relu", "sigmoid", "tanh", "softmax"])
def test_activation_runs(engine, catalog_synthetic, op):
    specs = [s for s in catalog_synthetic if op in s.name]
    assert specs
    r = engine.run_one(specs[0])
    assert r.status == TestStatus.OK, r.error_message


# ----- Reductions -----

@pytest.mark.parametrize("op", [
    "reducemean", "reducesum", "reducemax", "reducemin",
    "reducesumsquare", "reducel1", "reducel2",
])
def test_reduction_runs(engine, catalog_synthetic, op):
    specs = [s for s in catalog_synthetic if op in s.name]
    assert specs, f"no {op} specs"
    r = engine.run_one(specs[0])
    assert r.status == TestStatus.OK, r.error_message


# ----- MatMul / Gemm -----

def test_matmul_runs(engine, catalog_synthetic):
    specs = [s for s in catalog_synthetic if "matmul" in s.name]
    assert specs
    r = engine.run_one(specs[0])
    assert r.status == TestStatus.OK, r.error_message


def test_gemm_runs(engine, catalog_synthetic):
    specs = [s for s in catalog_synthetic if "gemm" in s.name]
    assert specs
    r = engine.run_one(specs[0])
    assert r.status == TestStatus.OK, r.error_message


# ----- Norm layers -----

def test_layernorm_runs(engine, catalog_synthetic):
    specs = [s for s in catalog_synthetic if "layernorm" in s.name]
    assert specs
    r = engine.run_one(specs[0])
    assert r.status == TestStatus.OK, r.error_message


def test_batchnorm_runs(engine, catalog_synthetic):
    specs = [s for s in catalog_synthetic if "batchnorm" in s.name]
    assert specs
    r = engine.run_one(specs[0])
    assert r.status == TestStatus.OK, r.error_message


# ----- Pool -----

def test_maxpool_runs(engine, catalog_synthetic):
    specs = [s for s in catalog_synthetic if "maxpool" in s.name]
    assert specs
    r = engine.run_one(specs[0])
    assert r.status == TestStatus.OK, r.error_message


def test_avgpool_runs(engine, catalog_synthetic):
    specs = [s for s in catalog_synthetic if "averagepool" in s.name]
    assert specs
    r = engine.run_one(specs[0])
    assert r.status == TestStatus.OK, r.error_message


def test_global_avgpool_runs(engine, catalog_synthetic):
    specs = [s for s in catalog_synthetic if "global_avgpool" in s.name or "globalaveragepool" in s.name]
    assert specs
    r = engine.run_one(specs[0])
    assert r.status == TestStatus.OK, r.error_message
