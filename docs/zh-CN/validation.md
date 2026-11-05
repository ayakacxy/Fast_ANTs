# 测试与质量评估

[English](../en/validation.md) · [简体中文](validation.md) · [文档](index.md)

以下原始 MRI 结果使用同一对 256×256×150 IXI 图像，在计时结束后由统一的 CPU FP64 采样器重新应用保存的变换。前景为固定图像中强度大于最小值加上强度范围 5% 的 3,236,832 个体素。四个地标来自非专家的粗略标注。配准前前景 NCC 为 0.171354，平均地标误差为 21.4895 mm。

所有结果的重采样图像和位移场均为有限值，几何与固定网格一致。正逆场的 Jacobian 行列式均为有限正值，非正体素数为零。Windows 和 Linux 的 NCC 与平均地标误差取两次测量的中位数，macOS 使用单次测量值。表中的 Jacobian 最小值取全部运行的正、逆场最小值；逆一致性 RMS 取全部运行、两个复合方向中的最大值。

## Windows 原始 MRI 结果

[16 线程 Windows 测量](performance.md#windows-cuda-与-cpu16-线程)中五个引擎各运行两次。

| 引擎 | 前景 NCC | 平均地标误差 / mm | 正逆 Jacobian 最小值 | 最大逆一致性 RMS / mm |
| --- | ---: | ---: | ---: | ---: |
| CUDA FP32 | 0.664377 | 1.4347 | 0.161298 | 0.023652 |
| CPU FP32 | 0.664385 | 1.4345 | 0.161414 | 0.023654 |
| CPU FP64 | 0.664417 | 1.4345 | 0.161804 | 0.023478 |
| ANTs FP32 | 0.664393 | 1.4346 | 0.161264 | 0.023664 |
| ANTs FP64 | 0.663937 | 1.4336 | 0.163237 | 0.023333 |

逐次指标与来源记录见 [Windows 摘要](../assets/benchmarks/windows-native-16t.json)。

## Linux 原始 MRI 结果

[32 线程 Linux 测量](performance.md#linux-cuda-与-cpu32-线程)中五个引擎各运行两次。

| 引擎 | 前景 NCC | 平均地标误差 / mm | 正逆 Jacobian 最小值 | 最大逆一致性 RMS / mm |
| --- | ---: | ---: | ---: | ---: |
| CUDA FP32 | 0.663688 | 1.4324 | 0.161487 | 0.023656 |
| CPU FP32 | 0.664355 | 1.4309 | 0.160964 | 0.023635 |
| CPU FP64 | 0.664417 | 1.4345 | 0.161804 | 0.023478 |
| ANTs FP32 | 0.663660 | 1.4293 | 0.161125 | 0.023659 |
| ANTs FP64 | 0.663937 | 1.4336 | 0.163237 | 0.023333 |

逐次指标与来源记录见 [Linux 摘要](../assets/benchmarks/linux-native-32t.json)。

## macOS 原始 MRI 结果

[10 线程 macOS 测量](performance.md#macos-metal-与-cpu10-线程)中五个引擎各运行一次，使用 v0.1.0。

| 引擎 | 前景 NCC | 平均地标误差 / mm | 正逆 Jacobian 最小值 | 最大逆一致性 RMS / mm |
| --- | ---: | ---: | ---: | ---: |
| Metal FP32 | 0.662625 | 1.4299 | 0.165859 | 0.022969 |
| CPU FP32 | 0.664474 | 1.4382 | 0.160806 | 0.023699 |
| CPU FP64 | 0.664417 | 1.4345 | 0.161804 | 0.023478 |
| ANTs FP32 | 0.664369 | 1.4315 | 0.160959 | 0.023638 |
| ANTs FP64 | 0.664417 | 1.4345 | 0.161804 | 0.023478 |

Metal 地标误差的 RMS 和最大值分别为 1.8645 和 3.3925 mm，ANTs FP32 分别为 1.8650 和 3.3914 mm。前景归一化均方误差分别为 0.013558 和 0.013509。

Metal 正向和逆向 Jacobian 最小值分别为 0.165859 和 0.179500。每个方向的逆一致性统计均使用全部 9,830,400 个网格点：

| 复合顺序 | Metal RMS / mm | ANTs FP32 RMS / mm | Metal P99 / mm | ANTs FP32 P99 / mm | Metal 最大值 / mm | ANTs FP32 最大值 / mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正向场作用于逆向场之后 | 0.022969 | 0.023638 | 0.083221 | 0.085717 | 0.392295 | 0.409191 |
| 逆向场作用于正向场之后 | 0.020985 | 0.021935 | 0.077979 | 0.081582 | 0.478370 | 0.489421 |

逐点误差、两个复合方向的指标与来源记录见 [macOS 数据](../assets/benchmarks/v0.1.0-native.json)。

## 测试分组

测试覆盖图像几何与 I/O、数值算子、优化器和完整配准流程。CPU FP64 与 ANTs `--float 0` 对照，CPU FP32、Metal 和 CUDA 与 `--float 1` 对照。

```bash
python -B scripts/maintenance/run_tests.py --suite core
python -B scripts/maintenance/run_tests.py --suite cpu-fp64
python -B scripts/maintenance/run_tests.py --suite cpu-fp32
```

Metal 测试需要 Apple GPU：

```bash
python -B scripts/maintenance/run_tests.py --suite metal
```

CUDA 测试需要 NVIDIA GPU 和 CuPy：

```bash
python -B scripts/maintenance/run_tests.py --suite cuda
```

小范围改动可通过 `--pattern` 选择相关文件。分组说明见[测试目录](../../tests/README.zh-CN.md)。

## 数值测试

算子测试使用小数组和已知变换，检查插值、物理梯度、平滑、局部统计、直方图导数、位移场复合和求逆。测试点应包含图像内部与边界；常量图像、近常量区域、大坐标原点和各向异性网格有助于发现数值问题。

完整配准可使用具有已知位移的合成变换。正向和逆向映射都要检查，同时确认单位和复合顺序。跨后端比较数值误差及最终对齐效果时，应考虑浮点规约顺序的差异。

## 图像与地标对齐

在同一固定图像网格、同一前景区域上评估不同实现。已保存结果的评估工具计算以下指标：

| 指标 | 含义 |
| --- | --- |
| 归一化互相关（NCC） | 重采样后的强度相似性，越大表示正相关性越强 |
| 归一化均方误差 | 强度归一化后的残差，越小表示越接近 |
| 目标配准误差（TRE） | 对应地标经变换后的距离，单位为毫米 |

设 $F$ 为固定图像， $W$ 为配准后重采样到固定网格的移动图像，前景 NCC 为：

```math
\mathrm{NCC}_{\Omega} =
\frac{\sum_{i\in\Omega}(F_i-\bar{F})(W_i-\bar{W})}
{\sqrt{\sum_{i\in\Omega}(F_i-\bar{F})^2}\,
 \sqrt{\sum_{i\in\Omega}(W_i-\bar{W})^2}}.
```

$\bar{F}$ 和 $\bar{W}$ 均在同一固定图像前景区域内计算：

```math
\Omega = \{i\mid F_i \gt F_{\min}+0.05(F_{\max}-F_{\min})\}.
```

NCC 的取值范围为 −1 至 1，越接近 1 表示强度正相关性越强。[评估工具](../../validation/quality/assess_saved_registration.py)先按各输入图像的强度范围进行线性归一化，该处理不改变 NCC；分母为零时记为 `null`。这里的前景 NCC 在整个前景区域计算，用于结果评估；SyN 优化使用局部 CC。

设固定图像地标为 `p_i`、对应移动图像地标为 `q_i`，固定到移动的映射为 `T`，则

```math
\mathrm{TRE}_{\mathrm{RMS}}=\sqrt{\frac{1}{N}\sum_i\|T(p_i)-q_i\|^2}.
```

图像相似性与地标误差反映不同方面的对齐情况。报告结果时，说明图像统计所用区域及地标标注来源。

## 位移场质量

位移场 `u` 的空间 Jacobian 为 `J=I+∇u`。检查行列式的最小值，以及行列式非正的体素数量。内置质量统计分别评估非线性的正向场和逆向场。

逆一致性通过复合两侧映射后的残差计算：

```math
r(x)=\phi^{-1}(\phi(x))-x.
```

两个复合方向都需要检查，在复合采样有效的位置统计 RMS、百分位数或最大位移，单位为毫米。同时检查位移场、重采样图像和统计结果是否为有限值。结合 `quality.json` 或独立评估报告，对配准图像进行目视检查。

## 评估数据

评估示例使用三份 IXI T1 图像：IXI002-Guys-0828、IXI012-HH-1211 和 IXI013-HH-1212，尺寸为 256×256×150。[输入信息](../../validation/sources/input-inventory.json)、[下载校验值](../../validation/sources/downloaded-files.json)和[数据来源](../../validation/sources/public-data.json)记录文件及预处理方法。

原始 IXI 数据使用 CC BY-SA 3.0。Zenodo 再发布页面对新增材料标注 CC BY 4.0；MRI 文件应遵循原始数据集的许可。图像数据不包含在 Python 软件包中。

## 评估工具

[验证目录](../../validation/README.zh-CN.md)包含参考结果生成与已保存结果评估工具。使用前查看其参数：

```bash
python -m validation.quality.assess_saved_registration --help
python -m validation.quality.run_float1_quality_matrix --help
```

与 ANTs 对比时，保持图像、几何信息、预处理、阶段参数、插值、采样方法和精度一致，并随结果保存配置及软件版本。[质量评估配置](../../validation/configs/registration_quality_policy.json)列出了评估脚本所用检查项。

## CUDA 检查

`tests/cuda` 包含 CUDA 算子与完整配准测试，在 NVIDIA 硬件上设置 `FAST_ANTS_TEST_CUDA=1` 启用。检查范围包括图像采样、滤波、Mattes 控制、场运算、逆场停止、状态保留、保存变换重应用和原生 FP64 最终质量检查。
