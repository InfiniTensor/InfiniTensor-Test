"""Auditable real-network inventory; weights and tensor names do not inflate counts."""

from collections import Counter
import hashlib
import re

import onnx
from onnx import helper, numpy_helper

from infinitensor_test.artifacts import digest


def architecture_digest(path):
    model = onnx.load(str(path))

    def tensor(t, constant=False):
        info = {"dtype": t.data_type, "shape": list(t.dims)}
        if constant or t.data_type not in (1, 10, 11, 16):
            value = numpy_helper.to_array(t)
            info["value"] = hashlib.sha256(value.tobytes()).hexdigest()
        return info

    symbols = {}
    def value_info(info):
        tensor_type = info.type.tensor_type
        shape = []
        for d in tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape.append(d.dim_value)
            elif d.dim_param:
                shape.append(symbols.setdefault(d.dim_param, f"d{len(symbols)}"))
            else:
                shape.append(None)
        return {"dtype": tensor_type.elem_type, "shape": shape}

    def graph(g, outer=None):
        names = dict(outer or {})
        constants = {t.name: t for t in g.initializer}
        inputs = [v for v in g.input if v.name not in constants]
        for i, v in enumerate(inputs):
            names[v.name] = f"input{i}"
        initializer_info, nodes = {}, []
        def identify(name):
            if not name:
                return ""
            if name not in names and name in constants:
                names[name] = f"constant{len(initializer_info)}"
                initializer_info[names[name]] = tensor(constants[name])
            return names.get(name, f"capture:{name}")
        for index, node in enumerate(g.node):
            args = [identify(n) for n in node.input]
            attrs = {}
            for a in sorted(node.attribute, key=lambda a: a.name):
                value = helper.get_attribute_value(a)
                if isinstance(value, onnx.TensorProto):
                    value = tensor(value, constant=True)
                elif isinstance(value, onnx.GraphProto):
                    value = graph(value, names)
                elif isinstance(value, bytes):
                    value = value.hex()
                attrs[a.name] = value
            for i, name in enumerate(node.output):
                names[name] = f"node{index}:{i}"
            nodes.append({"op": node.op_type, "domain": node.domain,
                          "inputs": args, "outputs": len(node.output), "attrs": attrs})
        outputs = [{"value": identify(v.name), **value_info(v)} for v in g.output]
        return {"inputs": [value_info(v) for v in inputs], "nodes": nodes,
                "constants": initializer_info, "outputs": outputs}
    return digest({"opsets": sorted((o.domain, o.version) for o in model.opset_import),
                   "graph": graph(model.graph)})


def real_inventory(specs):
    rows, candidates, unique, duplicate, excluded = [], [], [], [], []
    seen_names, seen_graphs, parents = {}, {}, {}
    seen_cases = set()
    def representative(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    for spec in sorted(specs, key=lambda s: (s.source.value, s.name)):
        if spec.source.value == "synthetic" or spec.expected_rejection:
            continue
        row = {"case_id": spec.short_id(), "source": spec.source.value,
               "source_id": spec.source_id or spec.source.value,
               "architecture": spec.metadata.get("architecture"),
               "task": spec.metadata.get("task", spec.category.value),
               "path": str(spec.path or ""), "skip_reason": spec.skip_reason,
               "error": spec.metadata.get("collection_error"),
               "validation": spec.metadata.get("validation", {})}
        rows.append(row)
        if (spec.metadata.get("model_kind") != "real_network" or not spec.path or not spec.path.is_file() or spec.is_skipped
                or row["validation"].get("reference") != "passed"
                or (spec.metadata.get("export_backend") == "torch" and
                    row["validation"].get("torch_reference", {}).get("state") != "passed")
                or (spec.metadata.get("export_backend") == "paddle" and
                    row["validation"].get("paddle_reference", {}).get("state") != "passed")):
            row["state"] = "excluded"
            excluded.append(row)
            continue
        if row["case_id"] in seen_cases:
            row.update(state="duplicate", duplicate_of=row["case_id"])
            duplicate.append(row)
            continue
        seen_cases.add(row["case_id"])
        name = re.sub(r"[^a-z0-9/]", "", spec.metadata.get("architecture_id", spec.short_id()).lower())
        sha = spec.metadata.get("architecture_sha256") or architecture_digest(spec.path)
        spec.metadata["architecture_sha256"] = row["architecture_sha256"] = sha
        current = row["case_id"]
        representative(current)
        for original in (seen_names.get(name), seen_graphs.get(sha)):
            if original:
                a, b = representative(current), representative(original)
                parents[max(a, b)] = min(a, b)
        seen_names.setdefault(name, current)
        seen_graphs.setdefault(sha, current)
        candidates.append(row)
    for row in candidates:
        original = representative(row["case_id"])
        if original == row["case_id"]:
            row["state"] = "validated_unique"
            unique.append(row)
        else:
            row.update(state="duplicate", duplicate_of=original)
            duplicate.append(row)
    return {"schema_version": 1, "real_candidates": len(rows), "validated_unique": len(unique),
            "duplicates": len(duplicate), "excluded": len(excluded),
            "by_source": dict(Counter(r["source"] for r in unique)),
            "by_task": dict(Counter(r["task"] for r in unique)), "models": rows}
