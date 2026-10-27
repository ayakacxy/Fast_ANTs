# 测试

[English](README.md) · [简体中文](README.zh-CN.md) · [开发指南](../docs/zh-CN/development.md)

| 目录 | 测试内容 |
| --- | --- |
| `core/` | 坐标、变换、影像 I/O、共用类型和质量计算 |
| `cpu_fp64/` | FP64 度量、优化器、形变场和配准 |
| `cpu_fp32/` | FP32 算子、数值稳定性、工作区和主机端控制 |
| `metal/` | Metal GPU 算子和配准组件 |
| `cuda/` | CUDA 算子、配准、状态保留和最终 QA |
| `contracts/` | CLI 参数、基准测试元数据、日志和模拟调度 |
| `fixtures/` | 共用的小型参考输入和元数据 |

在项目根目录运行：

```sh
python -B scripts/maintenance/run_tests.py --suite contracts
python -B scripts/maintenance/run_tests.py --suite cpu-fp32 --pattern test_stable32.py
python -B scripts/maintenance/run_tests.py --suite cpu --list
```

默认 `cpu` 包含基础功能、接口、FP64 和 FP32 测试。`--pattern` 选择文件名，`--list` 打印测试名称而不执行测试。数值测试首次运行时可能触发 Numba 编译。

Metal 测试需要支持的 Apple GPU，并通过参数显式选择：

```sh
python -B scripts/maintenance/run_tests.py --suite metal
```

CUDA 测试需要 CuPy 和可用的 NVIDIA GPU：

```sh
python -B scripts/maintenance/run_tests.py --suite cuda
```

`--suite all` 同时运行 CPU 和 GPU 测试。配准质量指标和 ANTs 对比方法见[测试与质量评估](../docs/zh-CN/validation.md)。
