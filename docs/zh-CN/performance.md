# 性能测试

[English](../en/performance.md) · [简体中文](performance.md) · [文档](index.md)

所有测试均使用 `brain-syn` 预设，在 256×256×150 分辨率下将 IXI012-HH-1211 配准到 IXI002-Guys-0828。各平台采用相同的[配准参数](#配准参数)；加速比与同一硬件上相同精度的官方 ANTs 比较：FP32 对应 `--float 1`，FP64 对应 `--float 0`。

计时包含进程启动、模块导入、输入读取、刚体／仿射／SyN 配准、适用后端的 GPU 同步与下载，以及完整写出。Fast ANTs 还包含内置质量检查；官方 ANTs 使用原生写出设置，不执行这项额外计算。外部计时器自身启动不计入。质量指标在计时之外通过同一 CPU FP64 采样器重新应用已保存变换计算，详见[测试与质量评估](validation.md)。

所有测量均在新进程中依次运行。Windows 和 Linux 取两次测量的中位数：Fast ANTs 各后端先完整预热一次，两轮正式测量使用相反顺序；macOS 各后端使用已有编译缓存测量一次。官方 ANTs 没有单独执行完整预热。温度和后台负载未受控，逐次耗时、完整软件版本和测量记录见各平台的数据链接。

## Windows CUDA 与 CPU，16 线程

Fast ANTs v0.3.0，Windows 11；Intel Core i5-14400F（10 核心 / 16 逻辑 CPU）、32 GB 内存和 RTX 4060 Ti（8 GB 显存）。Numba 和 ITK 使用 16 线程，BLAS 和 OMP 限制为 1；CUDA 运行库 13.2、CuPy 14.2.0、原生 Windows ANTs 2.6.5。

| 后端 | 完整流程耗时 / s | 加速比 |
| --- | ---: | ---: |
| CUDA FP32 | 7.43 | 70.54× |
| CPU FP32 | 76.29 | 6.87× |
| CPU FP64 | 93.88 | 5.71× |
| ANTs FP32 | 524.23 | — |
| ANTs FP64 | 536.38 | — |

另一次 CUDA 单次测量使用空的 CuPy、CUDA 驱动和 Numba 编译缓存目录，耗时 **27.26 s**；文件系统缓存和 GPU 状态未重置。Fast ANTs 使用 Python gzip level 0 写出，每个文件一个写入线程。

[Windows 测试数据](../assets/benchmarks/windows-native-16t.json) · [Windows 质量结果](validation.md#windows-原始-mri-结果)

## Linux CUDA 与 CPU，32 线程

Fast ANTs v0.2.0，Linux x86-64；双路 Xeon Gold 6230（40 核心 / 80 逻辑 CPU）和 RTX 2080 Ti（11 GiB 显存）。配准使用 32 线程，BLAS 和 OMP 限制为 1；CuPy 14.2.0、ANTs 2.6.5.dev1。

| 后端 | 完整流程耗时 / s | 加速比 |
| --- | ---: | ---: |
| CUDA FP32 | 8.24 | 40.6× |
| CPU FP32 | 51.79 | 6.46× |
| CPU FP64 | 85.58 | 4.63× |
| ANTs FP32 | 334.32 | — |
| ANTs FP64 | 396.34 | — |

每次测量前先读取输入文件以预热文件缓存。Fast ANTs 使用 gzip level 0 写出，可选 pigz 写出器每个文件使用两个工作线程。正式测量前后采样的 CPU 忙碌比例为 0.30%–2.52%。

[Linux 测试数据](../assets/benchmarks/linux-native-32t.json) · [Linux 质量结果](validation.md#linux-原始-mri-结果)

## macOS Metal 与 CPU，10 线程

Fast ANTs v0.1.0，macOS 27.0.1 arm64；Apple M5（Mac17,3，10 个逻辑 CPU 核心）。配准使用 10 线程；MLX 0.32.4、ANTs 2.6.5。

| 后端 | 完整流程耗时 / s | 加速比 |
| --- | ---: | ---: |
| Metal FP32 | 15.34 | 33.1× |
| CPU FP32 | 42.57 | 11.9× |
| CPU FP64 | 60.44 | 8.37× |
| ANTs FP32 | 507.48 | — |
| ANTs FP64 | 505.85 | — |

SIMD 内核通过小规模算子调用预先编译。CPU FP32 记录到 40 次 Numba 磁盘缓存命中和 1 次未命中，相应编译耗时已计入结果。图像读取使用 `itk_scanner` 几何和 `itk` 强度缩放。

[macOS 测试数据](../assets/benchmarks/v0.1.0-native.json) · [macOS 质量结果](validation.md#macos-原始-mri-结果)

## 配准参数

| 阶段 | 度量 | 最大迭代次数 | 降采样因子 | 图像平滑 / vox |
| --- | --- | --- | --- | --- |
| Rigid | Mattes MI，32 bins | 1000×500×250×100 | 8×4×2×1 | 3×2×1×0 |
| Affine | Mattes MI，32 bins | 1000×500×250×100 | 8×4×2×1 | 3×2×1×0 |
| SyN | 局部 CC，半径 2 | 100×70×50×20 | 8×4×2×1 | 3×2×1×0 |

使用稠密采样，随机种子为 20261022。SyN 步长为 0.2、更新场方差为 3、总场方差为 0；收敛阈值为 1e-6，窗口为 10，各阶段可在收敛后提前停止。不使用 winsorization 和直方图匹配。

## 复现方法

按[验证工具说明](../../validation/README.zh-CN.md)准备输入。在项目根目录运行以下命令，使用已有缓存对 CPU FP32 测量一次，输出目录需要是新目录：

```bash
python -B validation/benchmarks/measure_fp32_pipeline.py \
  --source src --config src/fast_ants/presets/brain-syn.json \
  --case ixi002_to_ixi012 --resolution native --threads 10 \
  --engine cpu --warmups 0 --repeats 1 \
  --output validation/reports/native-cpu-fp32
```

测试 Metal 时，在支持的 Apple GPU 上使用 `--engine metal`，并选择另一个输出目录。内核首次调用所需的编译计入耗时。[计时工具](../../validation/README.zh-CN.md)还支持 CPU FP64、官方 ANTs、多次测量和性能分析。

### Windows 复现方法

完成 [Windows 安装](installation.md#windows-powershell)后，在项目根目录运行以下命令。将 `--ants-registration` 设为实际的 ANTs 路径，按本机配置选择线程数，并使用新的输出目录。

```powershell
.\.venv\Scripts\python.exe -B -m validation.reference_tools.fetch_public_data
.\.venv\Scripts\python.exe -B -m validation.benchmarks.measure_cuda_native `
  --source src --data validation/data/public `
  --ants-registration .\setup\windows\reference\ants-2.6.5\bin\antsRegistration.exe `
  --threads 16 --repeats 2 --include-fp64 --skip-official-warmup `
  --timeout 3600 --output validation/reports/windows-native-16t
.\.venv\Scripts\python.exe -B -m validation.benchmarks.measure_cuda_native `
  --assess-report validation/reports/windows-native-16t/report.json `
  --threads 16 --output validation/reports/windows-native-16t-quality
```

性能测试分别记录空缓存和预热后缓存的运行结果，保留输出供最后一步评估已保存变换。
