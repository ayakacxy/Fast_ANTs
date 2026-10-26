# 性能测试

[English](../en/performance.md) · [简体中文](performance.md) · [文档](index.md)

## 原始分辨率配准

测试日期为 2026-10-26，固定图像为 IXI002-Guys-0828，移动图像为 IXI012-HH-1211，两者大小均为 256×256×150 体素。各后端在新进程中复用已有缓存，仅运行一次，没有另做完整预热。

| 后端 | 完整进程耗时 / s |
| --- | ---: |
| Metal FP32 | 15.34 |
| CPU FP32 | 42.57 |
| CPU FP64 | 60.44 |

CPU FP32 在 [SIMD 与工作区优化](optimizations.md)后测量；CPU FP64 和 Metal 没有改动，保留当天稍早的缓存测量。后台负载和温度没有固定，这些是单次结果，不是稳定的平均耗时。

CPU FP32 本次记录到 40 次 Numba 磁盘缓存命中和 1 次未命中，该项编译已计入 42.57 秒。相对优化前的缓存单次实测 59.79 秒，耗时缩短 28.8%，加速 1.40 倍。仿射矩阵、正向形变场、逆向形变场和配准后图像的文件哈希均与优化前相同；这说明现有结果得到保留，不代表已经证明与官方 ANTs 等价。

[缓存测量结果](../assets/benchmarks/native-cached-20261026.json)记录完整耗时、源码哈希、缓存计数及输出对比。本次检查没有新增官方 ANTs 测量。

## 对应精度的 ANTs 对照

以下 v0.1.0 对照在 2026-10-25 使用同一图像对及 ANTs 2.6.5 测量，保留当次配对结果；不把上方更新的缓存耗时代入该组加速倍数。

| 后端 | 耗时 / s | ANTs 精度 | ANTs 耗时 / s | 加速倍数 |
| --- | ---: | --- | ---: | ---: |
| CPU FP64 | 59.27 | `--float 0` | 505.85 | 8.53× |
| CPU FP32 | 92.01 | `--float 1` | 507.48 | 5.52× |
| Metal FP32 | 21.36 | `--float 1` | 507.48 | 23.76× |

加速倍数为 ANTs 耗时除以对应后端耗时，每个引擎正式测量一次。

## 测试环境

- Apple M5（Mac17,3），10 个逻辑 CPU 核心；macOS 27.0.1 arm64。
- Python 3.11.17、NumPy 2.4.6、Numba 0.68.0、NiBabel 5.4.2、MLX 0.32.4。
- 使用包内 `brain-syn` 参数、10 线程、随机种子 20261022；图像几何为 `itk_scanner`，强度缩放为 `itk`。
- 2026-10-26 的检查保留已有缓存，各后端只运行一次，不做完整预热；CPU FP32 测量前以小算子测试编译新增 SIMD 内核。
- 2026-10-25 的 ANTs 对照中，每个 Python 后端在正式运行前完整预热一次；ANTs 不单独进行完整预热。所有引擎串行运行，输入图像已由前序 Python 运行读入。
- 报告记录了电源状态，后台负载和温度没有固定。

计时包含进程启动、模块导入、图像读入、配准、适用后端的 GPU 同步与下载、输出质量检查和完整写出。Python 后端执行内置质量检查，ANTs 不执行这部分额外计算。外部计时器的启动时间不计入结果。

## 配准参数

| 阶段 | 度量 | 最大迭代次数 | 降采样因子 | 图像平滑 / vox |
| --- | --- | --- | --- | --- |
| Rigid | Mattes MI，32 bins | 1000×500×250×100 | 8×4×2×1 | 3×2×1×0 |
| Affine | Mattes MI，32 bins | 1000×500×250×100 | 8×4×2×1 | 3×2×1×0 |
| SyN | 局部 CC，半径 2 | 100×70×50×20 | 8×4×2×1 | 3×2×1×0 |

使用稠密采样。SyN 步长为 0.2、更新场方差为 3、总场方差为 0；收敛阈值为 1e-6，窗口为 10。各阶段可在达到收敛条件后提前停止。此参数配置不使用 winsorization 和直方图匹配。

## 复现方法

在环境中安装官方 ANTs，并按[验证工具说明](../../validation/README.zh-CN.md)准备输入图像。在项目根目录运行，输出目录需要是新目录：

```bash
python -B validation/benchmarks/measure_current_native.py \
  --source src --config src/fast_ants/presets/brain-syn.json \
  --threads 10 --output validation/reports/native-local
```

脚本依次测量 CPU FP64、ANTs float0、CPU FP32、Metal FP32 和 ANTs float1，需要可用的 Metal 设备。预热图像及多余中间输出在记录校验值后删除，每个正式运行保留四个主要输出文件。

[ANTs 对照结果](../assets/benchmarks/native-20261025.json)包含完整耗时、参数、依赖版本、输入校验值和运行源码哈希。完整本地报告保存在 `validation/reports/current-native-20261025/`、`validation/reports/cached-single-20261026/` 和 `validation/reports/cpu32-opt-20261026/`。复测产生的重复图像在记录哈希后删除，原始对照输出仍保留在本地。实现细节见[优化说明](optimizations.md)，质量指标见[测试与质量评估](validation.md)。
