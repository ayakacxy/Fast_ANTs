# 性能测试

[English](../en/performance.md) · [简体中文](performance.md) · [文档](index.md)

## Windows CUDA 与 CPU，16 线程

Fast ANTs v0.3.0 使用与下文一致的 `brain-syn` 参数，在 256×256×150 分辨率下将 IXI012-HH-1211 配准到 IXI002-Guys-0828。原生 Windows 对照使用 16 个 CPU 线程，各后端与相同精度的官方 ANTs 比较。

| 后端 | 第一次 / s | 第二次 / s | 中位数 / s | 同精度 ANTs / 本后端 |
| --- | ---: | ---: | ---: | ---: |
| CUDA FP32 | 7.49 | 7.38 | 7.43 | 70.54× |
| CPU FP32 | 77.38 | 75.20 | 76.29 | 6.87× |
| CPU FP64 | 99.72 | 88.05 | 93.88 | 5.71× |
| ANTs FP32 | 522.11 | 526.36 | 524.23 | — |
| ANTs FP64 | 535.93 | 536.83 | 536.38 | — |

每次运行均启动新进程。Fast ANTs 各后端在正式计时前完整运行一次以准备编译缓存，两轮正式测量使用相反顺序。官方 ANTs 使用编译好的 C++ 程序，没有单独执行完整预热。表中列出两次测量和中位数。另一次使用空的任务专用 CuPy、CUDA 驱动和 Numba 缓存的 CUDA 单次运行耗时为 **27.26 s**；文件系统缓存和 GPU 状态没有重置。

计时包含启动、导入模块、读取输入、刚体／仿射／SyN 配准、适用后端的 GPU 同步与下载，以及完整写出。Fast ANTs 还包含内置 QA，写出使用 Python gzip level 0，每个文件一个写入线程；官方 ANTs 保留原生写出设置，不运行额外 QA。双方均不扣除任何耗时。计时器自身启动不计入，Windows CPU 时间和峰值工作集通过目标进程句柄读取。

硬件为 Intel Core i5-14400F，10 个物理核心、16 个逻辑 CPU，32 GB 内存及 8 GB 显存的 RTX 4060 Ti；系统为 Windows 11 专业工作站版，内部版本 26200。依赖为 Python 3.11.14、NumPy 2.3.0、Numba 0.64.0、llvmlite 0.46.0、NiBabel 5.3.3、tqdm 4.70.0、CuPy 14.2.0（`cupy-cuda13x`），CUDA 运行库为 13.2，NVIDIA 驱动为 610.60。官方 ANTs 为 Windows 2.6.5-gfdce4d2 可执行程序。Numba 和 ITK 使用 16 线程，BLAS 和 OMP 限制为 1，温度和后台负载没有固定。

[Windows 测试数据](../assets/benchmarks/windows-native-16t.json)记录参数、源码与输入输出校验值、冷启动和预热耗时、两轮正式结果及独立质量检查。十份正式变换均在计时之外通过同一 CPU FP64 采样器重新应用，详见 [Windows 质量结果](validation.md#windows-原始-mri-结果)。

## Linux CUDA 与 CPU，32 线程

CPU 和 CUDA 后端使用 `brain-syn`，在 256×256×150 原始分辨率下将 IXI012-HH-1211 配准到 IXI002-Guys-0828，线程数为 32。v0.2.0 使用相同的配准算法和参数。[测试数据](../assets/benchmarks/linux-native-32t.json)记录了完整配置和源码校验值。

| 后端 | 第一次 / s | 第二次 / s | 中位数 / s | 同精度 ANTs / 本后端 |
| --- | ---: | ---: | ---: | ---: |
| CUDA FP32 | 8.43 | 8.04 | 8.24 | 40.60× |
| CPU FP32 | 51.14 | 52.43 | 51.79 | 6.46× |
| CPU FP64 | 84.18 | 86.99 | 85.58 | 4.63× |
| ANTs FP32 (`--float 1`) | 333.73 | 334.90 | 334.32 | — |
| ANTs FP64 (`--float 0`) | 395.52 | 397.15 | 396.34 | — |

每次测量都启动一个新进程，并使用已缓存的输入和编译结果。Fast ANTs 的每个后端先完整运行一次以准备缓存，该次不计入结果；官方 ANTs 使用编译好的 C++ 程序，无需 JIT 编译预热。各引擎依次运行，第二次使用相反顺序。中位数只反映这两次测量。

计时包括程序启动、加载模块、读取图像、刚体／仿射／SyN 配准、适用后端的 GPU 同步与数据下载，以及结果写出；Fast ANTs 的计时还包括输出质量检查。Fast ANTs 写出仿射变换、正逆场、重采样图像和诊断文件，gzip 级别为 0，使用 pigz 时每个文件有两个压缩线程。官方 ANTs 使用自身默认的写出设置，不执行 Fast ANTs 的额外质量检查。计时工具自身的启动时间不计入，两边也没有扣除任何耗时。

硬件为双路 Xeon Gold 6230，共 40 个物理核心、80 个逻辑 CPU，以及 11 GiB 显存的 RTX 2080 Ti，系统为 Linux x86-64。配准使用 32 线程，BLAS 和 OMP 环境变量限制为 1。依赖为 Python 3.11.17、NumPy 2.4.6、Numba 0.68.0、NiBabel 5.4.2、CuPy 14.2.0，NVIDIA 驱动为 570.133.07。官方 ANTs 可执行程序为 2.6.5.dev1-g89fa5be。正式运行前后采样的 CPU 忙碌比例为 0.30% 至 2.52%。

[Linux 测试数据](../assets/benchmarks/linux-native-32t.json)记录每次耗时、输入和输出的校验值、配准参数、软硬件版本及质量检查结果。质量检查在计时结束后进行，所有引擎都使用同一套 CPU FP64 采样方法，详见[质量评估](validation.md)。Linux 和 Apple M5 的数据分别反映各自平台的表现。

## Apple M5 发布版本测量

### 原始分辨率配准

Fast ANTs v0.1.0 使用 `brain-syn` 预设，将 IXI012-HH-1211 配准到 IXI002-Guys-0828。两幅图像均为 256×256×150 体素。

| 后端 | 完整进程耗时 / s | 前景 NCC | 平均地标误差 / mm |
| --- | ---: | ---: | ---: |
| Metal FP32 | 15.34 | 0.662625 | 1.4299 |
| CPU FP32 | 42.57 | 0.664474 | 1.4382 |
| CPU FP64 | 60.44 | 0.664417 | 1.4345 |

各后端在新进程中使用已有编译缓存，测量一次。计时包含进程启动、模块导入、图像读入、配准、适用后端的 GPU 同步与下载、输出质量检查和文件写出。

CPU FP32 记录到 40 次 Numba 磁盘缓存命中和 1 次未命中，编译耗时已计入结果。实际运行时间会随编译缓存、温度、后台负载和收敛所需的迭代次数变化。

[测试数据](../assets/benchmarks/v0.1.0-native.json)包含完整耗时、输入校验值、参数、依赖版本、源码哈希和输出校验值。具体实现见[优化说明](optimizations.md)，配准结果的评价方法见[质量评估](validation.md)。

质量列使用与计时输出 SHA-256 一致的已保存结果，在计时之外通过同一 CPU FP64 采样器重新应用变换。前景 NCC 使用 3,236,832 个体素，平均地标误差使用四组粗略对应点。Metal 正逆场的 Jacobian 均为正，两个复合方向的逆一致性 RMS 分别为 0.02297 和 0.02099 mm。指标定义和双向统计见 [Apple M5 质量结果](validation.md#apple-m5-原始-mri-结果)。

### ANTs 参考耗时

相同输入、配准参数和线程数下，ANTs 2.6.5 的单次运行时间如下：

| 精度 | ANTs 选项 | 完整进程耗时 / s | 前景 NCC | 平均地标误差 / mm |
| --- | --- | ---: | ---: | ---: |
| FP64 | `--float 0` | 505.85 | 0.664417 | 1.4345 |
| FP32 | `--float 1` | 507.48 | 0.664369 | 1.4315 |

ANTs 参考运行没有单独执行完整预热。Fast ANTs 的计时包含内置输出质量检查，ANTs 不执行这项额外计算。比较耗时时还应考虑编译缓存、温度和后台负载。

### 测试环境

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

### Windows 复现方法

使用 [Windows 安装说明](installation.md#windows-powershell)中的独立环境和原生 ANTs 程序。在项目根目录下载固定的公开输入，并选择新的输出目录：

```powershell
python -B -m validation.reference_tools.fetch_public_data
python -B -m validation.benchmarks.measure_cuda_native --source src --data validation/data/public --ants-registration .\setup\windows\reference\ants-2.6.5\bin\antsRegistration.exe --threads 16 --repeats 2 --include-fp64 --skip-official-warmup --timeout 3600 --output validation/reports/windows-native-16t
python -B -m validation.benchmarks.measure_cuda_native --assess-report validation/reports/windows-native-16t/report.json --threads 16 --output validation/reports/windows-native-16t-quality
```

性能测试命令在输出目录中分别创建冷启动缓存和预热共享缓存，保留全部输出。评估命令核验运行记录与校验值，再检查已保存图像和正逆场，不重复配准。ANTs 路径应替换为实际的 `antsRegistration.exe` 位置，线程数按本机配置选择。
