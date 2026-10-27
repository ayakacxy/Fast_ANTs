# 使用说明

[English](../en/usage.md) · [简体中文](usage.md) · [文档](index.md)

## 命令行配准

将移动图像配准到固定图像网格：

```bash
fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration-metal \
  --backend metal --precision fp32 \
  --preset brain-syn --threads 10 --verbose
```

使用 `--backend cpu --precision fp64` 或 `--backend cpu --precision fp32` 选择 CPU 后端。`python -m fast_ants` 调用相同的命令行接口。

使用 NVIDIA CUDA 时，激活 [CUDA 环境](installation.md#cuda-环境)并运行：

```bash
fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration-cuda --backend cuda --precision fp32 \
  --preset brain-syn --threads 10 --verbose
```

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--backend` | `cpu` | CPU、Metal 或 CUDA 后端 |
| `--precision` | CPU 为 `fp64`，Metal／CUDA 为 `fp32` | 计算精度 |
| `--preset` | `brain-syn` | 包内配准预设 |
| `--config` | — | 完整的自定义 JSON 配置 |
| `--threads` | 预设值：10 | CPU 线程数 |
| `--geometry-policy` | `itk_scanner` | 输入图像几何读取方式 |
| `--scaling-policy` | `itk` | 输入体素强度缩放方式 |
| `--verbose`、`-v` | 开启 | 输出英文进度和诊断日志 |
| `--quiet`、`-q` | 关闭 | 关闭进度和详细诊断日志 |
| `--log-every` | 1 | GPU 迭代诊断日志间隔 |

输出目录需要是新目录，线程数应按机器配置设置。Metal 和 CUDA 支持 FP32；CPU 支持两种精度。预设使用随机种子 20261022。

默认 scanner 几何选项要求图像具有 scanner-anatomical sform。使用标准 NiBabel NIfTI 几何和强度缩放时，添加：

```bash
--geometry-policy nifti --scaling-policy nifti
```

详见[图像几何](inputs-outputs.md)。

## 配准参数

`brain-syn` 使用四层分辨率：刚体和仿射阶段采用 Mattes 互信息，随后进行局部 CC 的 SyN 配准。迭代次数、平滑、采样和收敛参数见[配置说明](../../configs/README.zh-CN.md)。

使用其他参数时，复制[预设 JSON](../../src/fast_ants/presets/brain-syn.json)，修改后传入文件路径：

```bash
fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration-custom --backend cpu --precision fp64 \
  --config my-recipe.json --threads 10
```

`--config` 替换完整预设，与 `--preset` 互斥。命令行流程使用密集采样、线性插值和无掩膜图像，关闭直方图匹配、winsorization 及逆向重采样强度图像输出。配置中包含不支持的值时会报错。

## Python 接口

```python
from fast_ants import load_images, register, register32, register_metal, register_cuda

fixed, moving = load_images(
    ("fixed.nii.gz", "moving.nii.gz"),
    geometry_policy="itk_scanner", scaling_policy="itk",
)
result = register(fixed, moving, output_directory="registration-fp64")
# result = register32(fixed, moving, output_directory="registration-fp32")
# result = register_metal(fixed, moving, output_directory="registration-metal",
#                         verbose=True, log_every=1)
# result = register_cuda(fixed, moving, output_directory="registration-cuda",
#                        verbose=True, log_every=1)
warped = result.warped
pull = result.fixed_to_moving
```

Python 接口默认使用两层分辨率。通过 `LinearStage` 和 `SyNStage` 对象设置配准参数。[完整预设示例](../../examples/register_api.py)使用了与命令行相同的各阶段参数。

CPU FP64 的 `register` 接受 `initial_transform`，用于固定仿射变换并运行非线性阶段。Metal 和 CUDA 接口接受 `verbose` 和 `log_every`，其中 `verbose` 默认为 `False`；CPU 阶段日志由命令行程序输出。

### 公共函数与类型

| 分类 | 名称 |
| --- | --- |
| 图像与变换 | `Grid`, `Image`, `RegistrationResult`, `AffineTransform`, `CompositeTransform`, `DisplacementField` |
| 图像读写 | `load_image`, `load_images`, `save_image` |
| 变换读写 | `load_affine`, `save_affine`, `load_field`, `save_field`, `load_ants_transform_list` |
| 采样与重采样 | `apply_image`, `sample_image`, `transform_to_field` |
| 配准 | `LinearStage`, `SyNStage`, `register_linear`, `register`, `register32`, `register_metal`, `register_cuda` |

变换方向和数组布局见[输入与输出](inputs-outputs.md)。

## 日志

所有运行日志均为英文。命令行默认开启进度和详细日志；`--verbose` 继续可用，`--quiet` 同时关闭终端和 `console.log` 中的进度及详细诊断记录，错误信息仍会保留。Metal 和 CUDA 输出阶段、金字塔层级、迭代度量值、收敛值、耗时和停止原因；CPU 命令行显示加载和配准进度，配准阶段包含质量检查及写出。

Metal 和 CUDA 在交互式终端中为每个迭代金字塔层级显示 `tqdm` 进度条，显示 `metric`、`conv` 和 `valid`。迭代记录提供学习率、gamma 和线搜索评估次数时，还会显示 `lr`、`gamma` 和 `ls_eval`。结束时显示停止原因；提前收敛时保留实际迭代数与迭代预算。软件包依赖中已包含 `tqdm`。重定向输出使用不带进度条控制字符的可读文本；`console.log` 保留诊断记录，不保存终端动画。

迭代表包含以下字段：

```text
DIAGNOSTIC,Iteration,metricValue,convergenceValue,ITERATION_TIME_INDEX,SINCE_LAST,validPoints,learningRate,gamma,lineSearchEvaluations
```

`ITERATION_TIME_INDEX` 为当前层累计耗时，`SINCE_LAST` 为距上一次迭代的时间。SyN 可同时显示固定图像和移动图像的有效点数。`--log-every N` 记录 GPU 的第一次、每隔 N 次及最后一次迭代，交互式进度条仍随每次迭代回调更新。

## Gzip 输出

兼容布局的标量／场 NIfTI 输出默认使用无损 gzip level 0，以减少浮点数组写出耗时。
解压后的头和体素字节不变，文件可能更大。可显式选择压缩级别：

```bash
FAST_ANTS_GZIP_LEVEL=1 fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration-compressed --backend cuda --precision fp32 \
  --preset brain-syn --threads 10
```

`FAST_ANTS_GZIP_LEVEL` 接受 0 到 9 的整数，默认 0。
`FAST_ANTS_GZIP_BACKEND=auto` 在可用时选择每文件两个工作线程的可选 pigz；`gzip`
强制标准库路径，`pigz` 要求存在对应程序。两种流式实现均遵循级别设置并保留相同
解压字节，不支持的图像布局沿用 NiBabel 回退。CLI 在 `run.json` 中记录实际实现、
版本、级别和工作线程数。该设置适用于所有使用共享写出器的后端。

## 输出与错误处理

命令行程序将变换、重采样图像、配置、迭代记录和质量检查结果写入输出目录。`run.json` 记录输入文件、配置哈希、后端、精度和线程设置；`console.log` 保存终端输出。详见[输出文件](inputs-outputs.md)。

参数错误、文件缺失、不支持的图像几何或 GPU 设备不可用时，程序会停止并输出错误信息，不会静默选择 CPU 执行。失败运行的日志保留在输出目录中，重新运行时请使用新目录。
