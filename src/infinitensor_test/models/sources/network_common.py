"""Shared discovery policies for real, randomly initialized network architectures."""

import fnmatch
from functools import lru_cache
from importlib.metadata import version

version = lru_cache()(version)


def create_bounded_model(builder, kwargs, limit=None):
    """Avoid allocating large parameter tensors when a library supports meta construction."""
    import random
    import numpy as np
    import torch
    from infinitensor_test.models.registry import SourceSizeLimitError
    size = None
    if limit:
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]), torch.device("meta"):
                prototype = builder(**kwargs)
                size = sum(v.numel() * v.element_size() for v in prototype.state_dict().values())
                del prototype
        except Exception:
            # Not every third-party builder supports meta tensors. Actual package
            # size and reference execution are still checked after normal construction.
            pass
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        if size is not None and size > limit:
            raise SourceSizeLimitError(size)
    return builder(**kwargs)


def select_names(names, params):
    include, exclude = params.get("include", []), params.get("exclude", [])
    if isinstance(include, str) or isinstance(exclude, str):
        raise ValueError("source include/exclude must be lists of glob patterns")
    return [name for name in sorted(set(names))
            if (not include or any(fnmatch.fnmatchcase(name, p) for p in include))
            and not any(fnmatch.fnmatchcase(name, p) for p in exclude)]


def provenance(package, repository, architecture, task):
    return {"model_kind": "real_network", "architecture": architecture,
            "export_backend": "torch",
            "generator_version": 3,
            "architecture_id": f"{task}/{architecture.lower()}",
            "upstream_repository": repository,
            "upstream_license": {"timm": "Apache-2.0", "transformers": "Apache-2.0",
                                 "torchaudio": "BSD-2-Clause", "torchvision": "BSD-3-Clause"}.get(package, "MIT"),
            "source_library_version": version(package), "weights": "random",
            "task": task}
