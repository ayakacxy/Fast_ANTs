# 配准参数

[English](README.md) · [简体中文](README.zh-CN.md) · [使用说明](../docs/zh-CN/usage.md)

包内 [`brain-syn`](../src/fast_ants/presets/brain-syn.json) 预设采用密集采样和四层分辨率。

| 阶段 | 度量 | 最大迭代次数 | 缩小因子 | 平滑标准差 / 体素 |
| --- | --- | --- | --- | --- |
| 刚体 | Mattes 互信息，32 bins | 1000, 500, 250, 100 | 8, 4, 2, 1 | 3, 2, 1, 0 |
| 仿射 | Mattes 互信息，32 bins | 1000, 500, 250, 100 | 8, 4, 2, 1 | 3, 2, 1, 0 |
| SyN | 局部 CC，半径 2 | 100, 70, 50, 20 | 8, 4, 2, 1 | 3, 2, 1, 0 |

刚体和仿射步长为 0.1。SyN 使用 `SyN[0.2,3,0]`，分别表示步长 0.2、更新场方差 3 和总场方差 0。收敛阈值为 1e-6，窗口为 10，随机种子为 20261022。各层达到收敛条件后可提前停止；缩小因子会根据物理体素间距调整。

运行预设：

```bash
fast-ants register fixed.nii.gz moving.nii.gz \
  --output registration --preset brain-syn
```

## 自定义参数

将预设 JSON 复制到其他文件，修改支持的参数：

```bash
fast-ants register fixed.nii.gz moving.nii.gz \
  --output registration-custom --config my-recipe.json
```

`--config` 替换整个预设，与 `--preset` 互斥。每个阶段的 `iterations`、`shrink_factors` 和 `smoothing_sigmas` 需要长度一致。图像平滑参数表示标准差，SyN 的更新场和总场平滑参数表示方差。

命令行支持 Rigid/MI → Affine/MI → SyN/CC，采用密集采样、线性插值和无掩膜图像，关闭直方图匹配及 winsorization。Python 接口可通过阶段对象直接设置参数。各参数含义见[算法](../docs/zh-CN/algorithms.md)。
