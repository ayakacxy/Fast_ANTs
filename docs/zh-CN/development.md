# 开发指南

[English](../en/development.md) · [简体中文](development.md) · [文档目录](index.md)

## 环境配置

```bash
python -m pip install -e '.[test,build]'
# Optional Metal dependencies:
python -m pip install -e '.[metal]'
```

项目使用 NumPy 管理数组，NiBabel 读写影像，Numba 编译 CPU 算子，MLX 调度自定义 Metal kernel。conda 环境见[安装说明](installation.md)，模块划分见[代码结构](architecture.md)。

CUDA 是下一个计划支持的 GPU 后端。其运行时和设备算子将独立实现，复用图像几何、阶段配置及主机端优化控制，具体步骤见 [CUDA 开发计划](roadmap.md)。

## 添加或修改算子

将实现放在对应后端和配准阶段中。共用的图像类型、配置和主机端计算放在 `core` 或 `pipeline`。明确指定数组的数据类型，并说明数组布局、单位、插值方法及图像外采样的处理方式。

CPU 算子应使用 Numba 的 nopython 模式编译。将互不依赖的任务并行执行，并采用数值稳定的规约顺序。Metal 实现应尽量让中间数组保留在 GPU 上，合并相关操作并复用缓冲区，回传数据以少量控制参数和最终结果为主。已有实现见[优化方法](optimizations.md)。

工作区由一次配准或一个金字塔层级管理。返回数组不能与后续迭代会覆盖的缓冲区混用，回调中保留的数据也需要独立处理。CPU 模块不应依赖 MLX；新增日志应直接使用已计算的数值。

## 测试

通过测试运行器选择分组或单个文件：

```bash
python -B scripts/maintenance/run_tests.py --suite contracts
python -B scripts/maintenance/run_tests.py --suite cpu-fp64 --pattern test_fields.py
python -B scripts/maintenance/run_tests.py --suite cpu-fp32 --pattern test_gaussian_sections32.py
python -B scripts/maintenance/run_tests.py --suite cpu --list
```

默认 `cpu` 包含基础功能、接口、FP64 和 FP32 测试。在支持 Metal 的机器上显式选择 GPU 测试：

```bash
python -B scripts/maintenance/run_tests.py --suite metal --pattern test_metal_cc_window32.py
```

新增数值测试时，根据算子的特点覆盖图像边界、常量图像、无效数值、各向异性间距、斜切网格，以及大坐标原点下的小位移。先用小型参考计算检查算子，再测试完整配准。修改优化器、规约或形变场更新时，结合[配准质量指标](validation.md)检查结果。

## 性能分析

[验证工具](../../validation/README.zh-CN.md)支持算子分析、完整流程计时和已保存结果的质量评估。对比时固定输入、参数、精度和线程数；重复测量前预热 Numba 和 Metal kernel，GPU 计时结束前需要同步。分析单个阶段时，单独统计图像读写时间。

按改动范围选择测试。文档修改需要检查 Markdown 格式和链接，插值或形变场复合的修改则需要数值测试。

## 文档与打包

中英文文档同步维护，命令、默认值、文件名和数学符号保持一致。运行日志和 CLI 消息使用英文。

独立公式使用 `math` 代码围栏，行内公式使用 `$...$`，并与相邻中文文字或标点留一个空格。正体标签和函数名使用 `\mathrm{...}`。

```bash
npx --yes markdownlint-cli2@0.23.2
python -B scripts/maintenance/check_markdown_links.py
python -m build
```

Markdown 格式检查需要 Node.js。`.markdownlint-cli2.jsonc` 定义项目文档在编辑器和命令行中使用的规则，排除参考源码和构建产物，不限制行长，并允许 README 使用 `h1` 和 `div` 居中排版。

检查 wheel 是否包含 `common.metal` 和预设 JSON。影像数据、生成结果、参考源码和缓存不进入发行包。保留 [NOTICE](../../licenses/NOTICE.zh-CN.md) 及上游许可证中的适用声明。
