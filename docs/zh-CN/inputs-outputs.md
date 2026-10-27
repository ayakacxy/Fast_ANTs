# 输入与输出

[English](../en/inputs-outputs.md) · [简体中文](inputs-outputs.md) · [文档](index.md)

## 图像格式与坐标

输入为体素值有限的三维标量 NIfTI 图像。Fast ANTs 使用以下数组和坐标约定：

| 数据 | 布局或单位 |
| --- | --- |
| 标量图像数组 | `[z,y,x]` |
| 索引和物理坐标向量 | `[x,y,z]` |
| 物理坐标 | LPS，毫米 |
| 位移数组 | `[3,z,y,x]`，LPS 分量，单位为毫米 |

加载时将 NIfTI RAS+ 坐标转换为 LPS。空间单位为米或微米时换算为毫米；缺失单位时按毫米处理，并记录到图像元数据中。图像网格可以倾斜且各向异性，方向轴需保持正交；不支持剪切网格。

对于索引向量 `i`、原点 `o`、方向矩阵 `D` 和体素间距矩阵 `S`，物理坐标为

```math
x=o+DSi.
```

## 读取选项

`load_image` 和 `load_images` 默认使用 `geometry_policy="nifti"` 与 `scaling_policy="nifti"`；命令行默认使用 `itk_scanner` 与 `itk`。

| 设置 | 行为 |
| --- | --- |
| `geometry_policy="nifti"` | 使用 NiBabel 选择的仿射矩阵，依次考虑带编码的 sform、qform 和默认矩阵 |
| `geometry_policy="itk_scanner"` | 要求 scanner-anatomical sform，归一化方向列向量并使用头信息中的体素间距 |
| `scaling_policy="nifti"` | 使用标准 NiBabel 强度转换 |
| `scaling_policy="itk"` | 使用 ITK 兼容标量转换，受支持的带缩放整数输入包含 FP32 中间转换 |

比较不同运行时应使用相同的读取选项。读取标签图像时，可通过合适的整数 `dtype` 保留标签类型。

## 变换方向

`result.fixed_to_moving` 将固定图像的物理坐标映射到移动图像的采样位置：

```math
W(x)=M(T(x)),\qquad T:\Omega_F\rightarrow\Omega_M.
```

因此，`warped.nii.gz` 是在固定图像网格上重采样的移动图像。将移动空间中的点映射到固定空间时，使用逆变换 `result.moving_to_fixed`。

设非线性映射为 `phi`，仿射变换为 `A`，完整采样映射为 `T = A ◦ phi`。`CompositeTransform((first, second))` 先执行 `first`，再执行 `second`；求逆时反转顺序并对两者分别求逆。逆向位移场通过迭代计算。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `affine.mat` | 仿射矩阵和中心参数 |
| `warp.nii.gz` | 固定空间网格上的正向非线性位移场 |
| `inverse-warp.nii.gz` | 逆向非线性位移场 |
| `warped.nii.gz` | 固定图像网格上的移动图像重采样结果 |
| `configuration.json` | 配准参数、后端、精度和坐标设置 |
| `history.json` | 阶段、层级、迭代值和停止原因 |
| `quality.json` | 输出质量指标 |
| `console.log` | 命令行输出及启用的运行日志 |
| `run.json` | 命令行输入文件、哈希和运行设置 |

`console.log` 和 `run.json` 由命令行程序写出。Python 接口在指定 `output_directory` 后保存图像、变换和 JSON 结果；否则只返回内存中的结果对象。

布局兼容的标量和位移场 NIfTI 默认使用无损 gzip level 0。可将 `FAST_ANTS_GZIP_LEVEL` 设为 0 到 9 选择压缩级别，解压后的头和数组保持不变。可选 pigz 写出器和回退行为见[使用说明](usage.md)。

仿射读写支持读取器实现的 ANTs MAT-v4 和 ITK 文本格式。形变场 NIfTI 存储采用 `[x,y,z,1,3]`，读取器也接受 `[x,y,z,3]`。尽管 NIfTI 仿射矩阵使用 RAS 坐标，位移向量分量仍为 LPS 毫米。

## 应用已保存的变换

```python
from fast_ants import load_image, load_ants_transform_list, apply_image, save_image

moving = load_image("moving.nii.gz")
fixed = load_image("fixed.nii.gz")
pull = load_ants_transform_list(["registration/warp.nii.gz", "registration/affine.mat"])
warped = apply_image(moving, fixed.grid, pull, interpolation="Linear")
save_image(warped, "reapplied.nii.gz")
```

标量强度图像使用 `Linear`，整数标签使用 `NearestNeighbor`。保存和重采样时，应将位移场与其网格信息一起处理。变换复合公式见[算法](algorithms.md)。
