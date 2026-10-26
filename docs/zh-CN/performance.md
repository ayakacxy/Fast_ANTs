# 性能测试

[English](../en/performance.md) · [简体中文](performance.md) · [文档](index.md)

## 原始分辨率配准

Fast ANTs v0.1.0 使用 `brain-syn` 预设，将 IXI012-HH-1211 配准到 IXI002-Guys-0828。两幅图像均为 256×256×150 体素。

| 后端 | 完整进程耗时 / s |
| --- | ---: |
| Metal FP32 | 15.34 |
| CPU FP32 | 42.57 |
| CPU FP64 | 60.44 |

各后端在新进程中使用已有编译缓存，测量一次。计时包含进程启动、模块导入、图像读入、配准、适用后端的 GPU 同步与下载、输出质量检查和文件写出。

CPU FP32 记录到 40 次 Numba 磁盘缓存命中和 1 次未命中，编译耗时已计入结果。实际运行时间会随编译缓存、温度、后台负载和收敛所需的迭代次数变化。

[测试数据](../assets/benchmarks/v0.1.0-native.json)包含完整耗时、输入校验值、参数、依赖版本、源码哈希和输出校验值。具体实现见[优化说明](optimizations.md)，配准结果的评价方法见[质量评估](validation.md)。

## ANTs 参考耗时

相同输入、配准参数和线程数下，ANTs 2.6.5 的单次运行时间如下：

| 精度 | ANTs 选项 | 完整进程耗时 / s |
| --- | --- | ---: |
| FP64 | `--float 0` | 505.85 |
| FP32 | `--float 1` | 507.48 |

ANTs 参考运行没有单独执行完整预热。Fast ANTs 的计时包含内置输出质量检查，ANTs 不执行这项额外计算。比较耗时时还应考虑编译缓存、温度和后台负载。

## 测试环境

- Apple M5（Mac17,3），10 个逻辑 CPU 核心；macOS 27.0.1 arm64。
- Python 3.11.17、NumPy 2.4.6、Numba 0.68.0、NiBabel 5.4.2、MLX 0.32.4。
- 使用包内 `brain-syn` 参数、10 线程、随机种子 20261022；图像几何为 `itk_scanner`，强度缩放为 `itk`。
- 每个 Fast ANTs 后端使用新进程和已有编译缓存运行一次；SIMD 内核通过小规模算子调用预先编译。
- 各引擎顺序运行。报告记录电源状态，后台负载和温度没有固定。

外部计时工具自身的启动时间不计入结果。

## 配准参数

| 阶段 | 度量 | 最大迭代次数 | 降采样因子 | 图像平滑 / vox |
| --- | --- | --- | --- | --- |
| Rigid | Mattes MI，32 bins | 1000×500×250×100 | 8×4×2×1 | 3×2×1×0 |
| Affine | Mattes MI，32 bins | 1000×500×250×100 | 8×4×2×1 | 3×2×1×0 |
| SyN | 局部 CC，半径 2 | 100×70×50×20 | 8×4×2×1 | 3×2×1×0 |

使用稠密采样。SyN 步长为 0.2、更新场方差为 3、总场方差为 0；收敛阈值为 1e-6，窗口为 10。各阶段可在达到收敛条件后提前停止。此参数配置不使用 winsorization 和直方图匹配。

## 复现方法

按[验证工具说明](../../validation/README.zh-CN.md)准备输入图像。在项目根目录运行以下命令，使用已有缓存对 CPU FP32 测量一次，输出目录需要是新目录：

```bash
python -B validation/benchmarks/measure_fp32_pipeline.py \
  --source src --config src/fast_ants/presets/brain-syn.json \
  --case ixi002_to_ixi012 --resolution native --threads 10 \
  --engine cpu --warmups 0 --repeats 1 \
  --output validation/reports/native-cpu-fp32
```

测试 Metal 时，将后端设为 `--engine metal`，并使用另一个输出目录。内核首次调用所需的编译会计入耗时；GPU 测量需要可用的 Metal 设备。

[计时工具](../../validation/README.zh-CN.md)还支持 CPU FP64、官方 ANTs、多次测量和性能分析，配置、日志及输出校验值保存在指定目录。
