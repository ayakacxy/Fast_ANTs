# 性能测试

[English](../en/performance.md) · [简体中文](performance.md) · [文档](index.md)

## 原始分辨率配准

测试日期为 2026-10-25，固定图像为 IXI002-Guys-0828，移动图像为 IXI012-HH-1211，两者大小均为 256×256×150 体素。下表比较当前实现与对应精度的 ANTs 2.6.5。

| 后端 | 耗时 / s | ANTs 精度 | ANTs 耗时 / s | 加速倍数 |
| --- | ---: | --- | ---: | ---: |
| CPU FP64 | 59.27 | `--float 0` | 505.85 | 8.53× |
| CPU FP32 | 92.01 | `--float 1` | 507.48 | 5.52× |
| Metal FP32 | 21.36 | `--float 1` | 507.48 | 23.76× |

加速倍数为 ANTs 耗时除以对应后端耗时。每个引擎正式测量一次。本轮 CPU FP32 慢于 CPU FP64；仅凭 float32 存储不能推断完整流程的运行时间。

## 测试环境

- Apple M5（Mac17,3），10 个逻辑 CPU 核心；macOS 27.0.1 arm64。
- Python 3.11.17、NumPy 2.4.6、Numba 0.68.0、NiBabel 5.4.2、MLX 0.32.4。
- 使用包内 `brain-syn` 参数、10 线程、随机种子 20261022；图像几何为 `itk_scanner`，强度缩放为 `itk`。
- 每个 Python 后端在正式运行前完整预热一次；ANTs 不单独进行完整预热。所有引擎串行运行，输入图像已由前序 Python 运行读入。
- 接 AC 电源，电池 80% 且未充电；未固定后台负载和温度。

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

[机器可读结果](../assets/benchmarks/native-20261025.json)包含完整耗时、参数、依赖版本、输入校验值和运行源码哈希。完整本地日志及输出保存在 `validation/reports/current-native-20261025/`。实现细节见[优化说明](optimizations.md)，质量指标见[测试与质量评估](validation.md)。
