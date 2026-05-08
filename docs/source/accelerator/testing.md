# Reusing Built-in Test Cases

This page explains how an out-of-tree accelerator backend can reuse PyTorch's built-in tests instead of maintaining a separate test suite.

The goal is to maximize upstream test reuse and keep backend-specific customization minimal.

This guide covers four common mechanisms:

1. Run tests gated by device decorators such as `@onlyCUDA` or `@onlyOn`.
2. Restrict @ops tests to a supported operator subset.
3. Customize `OpInfo`-based tests for unsupported ops, expected failures, or relaxed precision.
4. Exclude unsupported test classes or test methods.

## When to use which mechanism

| Requirement | Mechanism |
| ----------------- | --- |
| Run tests restricted to CUDA or another built-in device | `bypass_device_restrictions` |
| Restrict `@ops` tests to supported operators only | `set_test_configs(op_allowlist=...)` |
| Customize behavior of `@ops` tests | `set_test_configs(op_overrides=...)` |
| exclude tests at the test-class or test-method level | `set_test_configs(test_exclusions=...)` |

> Notice: `set_test_configs(...)` replaces the entire test configuration.If multiple fields are needed (`op_allowlist`, `op_overrides`, `test_exclusions`), pass them in the same call.

## 1. Bypass device restrictions

Some upstream tests are decorated with `@onlyCUDA` or `@onlyOn(["cuda"])`, even though the actual test logic is backend-agnostic.

Set `bypass_device_restrictions = True` on the instantiated test class to let those tests run on your backend.

```python
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from test_prims import TestPrims

class TestPrimsOpenReg(TestPrims):
    bypass_device_restrictions = True

instantiate_device_type_tests(TestPrimsOpenReg, globals(), only_for="openreg")
```

## 2. Filter `@ops` tests with `op_allowlist`

If your backend supports only a subset of operators, use `op_allowlist` to generate `@ops` test variants only for supported ops.

`op_allowlist` matches against `OpInfo.full_name`.

```python
from torch.testing._internal.common_device_type import PrivateUse1TestBase

PrivateUse1TestBase.set_test_configs(
    op_allowlist=(
        "add.Tensor",
        "sub.Tensor",
    )
)
```
Use this when you want to reduce operator coverage up front instead of generating tests for every op in the `@ops(...)` list.

## 3. Override `OpInfo`-based tests

`OpInfo` drives a large portion of PyTorch operator testing.

If an operator is unsupported, expected to fail, or requires relaxed precision on your backend, use backend-specific overrides instead of modifying the upstream `op_db`.

```python
import unittest

import torch
from torch.testing._internal.common_device_type import (
    PrivateUse1TestBase,
    precisionOverride,
)
from torch.testing._internal.opinfo.core import DecorateInfo

PrivateUse1TestBase.set_test_configs(
    op_overrides={
        "add.Tensor": [
            DecorateInfo(
                precisionOverride({torch.float32: 1e-2, torch.float16: 1e-1})
            )
        ],
    }
)
```

If both `op_allowlist` and `op_overrides` are set, `op_allowlist` is applied first and `op_overrides` is applied only to the ops that remain.

## 4. Exclude unsupported tests

Use `set_test_configs(test_exclusions=...)` when you want to exclude tests by generic test class name or test method name during `instantiate_device_type_tests`.

```python
from torch.testing._internal.common_device_type import PrivateUse1TestBase

PrivateUse1TestBase.set_test_configs(
    test_exclusions={
        "TestTensorIndexing": [
            "test_index_put",
            "test_gather",
        ],
        "TestCUDAGraphs": "*",
    }
)
```

## Summary

A typical backend bring-up workflow is:

1. Reuse as many upstream tests as possible.
2. Use `op_allowlist` to start from a smaller supported operator set if needed.
3. Add targeted `OpInfo` overrides for known operator gaps.
4. Exclude only tests that are genuinely unsupported.

This approach keeps backend-specific customization small, explicit, and maintainable while maximizing upstream coverage reuse.

For a complete runnable example, see [test_testing.py](https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_testing.py).
