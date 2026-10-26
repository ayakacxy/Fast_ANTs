# 验证工具

[English](README.md) · [简体中文](README.zh-CN.md) · [测试与质量评估](../docs/zh-CN/validation.md) · [优化方法](../docs/zh-CN/optimizations.md)

本目录包含数值算子检查、ANTs 参考结果生成、已保存变换评估和运行时间测量工具。日常图像配准使用 `fast-ants register`，不需要这些工具。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| [benchmarks/](benchmarks/) | 当前五后端计时、源码／开关对比及专项算子计时 |
| [quality/](quality/) | 已保存图像、形变场评估及参考结果对比 |
| [operators/](operators/) | 独立前缀和、规约及串行 watchdog 检查；日常算子测试位于 `tests/` |
| [reference_tools/](reference_tools/) | 数据准备和官方参考结果生成 |
| [maintenance/](maintenance/) | 依赖及实现检查 |
| [common/](common/) | 共用路径、进程辅助函数和工具索引 |
| [configs/](configs/) | 当前参考配方、质量策略和小案例矩阵 |
| [sources/](sources/) | 输入来源和校验记录 |
| `reports/` | 本地实测输出；预热产生的大体积影像不保留 |

[工具索引](common/tool-index.json)记录脚本名与模块路径。新命令建议使用模块路径；根目录的 `measure_fp32_pipeline.py`、`measure_fp64_cpu.py` 和 `assess_saved_registration.py` 也提供命令行入口。

## 查看工具参数

在项目根目录运行：

```bash
python -m validation.benchmarks.measure_current_native --help
python -m validation.benchmarks.measure_fp32_pipeline --help
python -m validation.benchmarks.measure_fp64_cpu --help
python -m validation.quality.assess_saved_registration --help
```

当前原始分辨率工具按顺序运行 CPU FP64、ANTs float0、CPU FP32、Metal FP32 和 ANTs float1，使用当前 `src`，记录源码哈希，不保存冻结的旧实现副本。FP32 流程工具可对比 CPU FP32、Metal 和 ANTs float1；FP64 工具对比 CPU FP64 与 ANTs float0。已保存结果的评估工具直接读取变换和图像，不重复拟合。

也可以按脚本名选择工具：

```bash
python -B validation/legacy.py --list
python -B validation/legacy.py verify_float1_alignment --help
```

索引只列出维护中的命令。源码对比工具接收调用者提供的源码目录，本目录不再保存旧实验实现副本。日常算子开发使用已分组的测试。

## 质量评估

在同一物理网格上比较图像，分别检查正向场与逆向场。常用指标包括 NCC、归一化均方误差、地标 TRE、Jacobian 行列式和逆一致性。计算方法见[测试与质量评估](../docs/zh-CN/validation.md)。

与 ANTs 对比时，CPU FP64 选择 `--float 0`，CPU FP32 和 Metal 选择 `--float 1`，并保持图像、阶段参数、插值、采样方法和输入几何一致。[质量评估设置](configs/registration_quality_policy.json)说明了相关选项。

## 计时

记录影像尺寸、配准参数、后端、精度、线程数、软件版本、预热次数和重复运行次数。GPU 任务完成后需要同步，并等待所有结果文件写入完成。分别说明计时是否包含进程启动、编译、影像 I/O 和最终质量统计。

Numba 和 Metal 首次运行时会编译 kernel，编译耗时应与重复执行耗时分开统计。对比任务顺序运行，保持相近的系统负载和供电状态。

## 数据

[cases.json](cases.json)、[输入信息](sources/input-inventory.json)、[下载校验值](sources/downloaded-files.json)和[数据来源](sources/public-data.json)记录了开发使用的三份 IXI 图像。不同分辨率存放在 `data/public`、`data/p11-half` 和 `data/prepared`。影像数据和生成结果不包含在 Python 软件包中。

模板和图谱资源保存在本地 `templates/` 目录中，分别遵循原始来源的许可。

## 运行测试

```bash
python -B scripts/maintenance/run_tests.py --suite contracts
python -B scripts/maintenance/run_tests.py --suite cpu-fp64 --pattern test_fields.py
python -B scripts/maintenance/run_tests.py --suite cpu-fp32 --pattern test_gaussian_sections32.py
```

使用 `--list` 查看选中的测试。GPU 测试需要 Apple GPU，并通过 `--suite metal` 显式选择。全部分组见[测试目录](../tests/README.zh-CN.md)。
