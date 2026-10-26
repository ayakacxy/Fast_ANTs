# 配准算法

[English](../en/algorithms.md) · [简体中文](algorithms.md) · [文档](index.md)

Fast ANTs 先使用 Mattes 互信息进行刚体和仿射配准，再使用局部归一化互相关进行位移场形式的 greedy SyN 配准。本文说明这些阶段执行的数学运算，CPU 与 Metal 的计算优化见[实现优化](optimizations.md)。

## 图像采样与坐标

记固定图像为 $F:\Omega_F\to\mathbb R$，移动图像为 $M:\Omega_M\to\mathbb R$。配准求解采样映射，并在固定图像网格上计算输出：

```math
W(x)=M(T(x)),\qquad T:\Omega_F\rightarrow\Omega_M.
```

因此，`fixed_to_moving` 将固定空间中的点映射到移动图像的采样位置，输出是移动图像在固定网格上的重采样结果。

全部物理坐标和位移分量使用 LPS 毫米。数组顺序为 `[z,y,x]`，索引与点向量顺序为 `[x,y,z]`。设原点为 $o$、方向矩阵为 $D$、间距矩阵为 $S=\mathrm{diag}(s_x,s_y,s_z)$、索引为 $i$：

```math
x=o+DSi,\qquad i=(DS)^{-1}(x-o).
```

NIfTI 的 RAS 坐标通过 $C=\mathrm{diag}(-1,-1,1)$ 转换： $x_{\rm LPS}=Cx_{\rm RAS}$。物理向量分量使用相同转换。文件格式与变换顺序见[输入输出](inputs-outputs.md)。

对连续索引 $q=n+t$，其中 $n=\lfloor q\rfloor$、 $t\in[0,1)^3$，三线性插值为：

```math
M(q)=\sum_{b\in\{0,1\}^3}M[n+b]\prod_{d=1}^3w_{b_d}(t_d),
\qquad w_0(t)=1-t,\quad w_1(t)=t.
```

采样会检查图像有效范围，并使用指定的越界填充值。度量使用变换位置处的有限差分图像梯度，索引梯度与物理坐标梯度满足：

```math
\nabla_xM=(DS)^{-\mathsf T}\nabla_iM.
```

## 刚体与仿射变换

设中心为 $c$、平移为 $t$、旋转矩阵为 $R$、仿射矩阵为 $B$：

```math
T_R(x)=R(\theta)(x-c)+c+t,\qquad
T_A(x)=B(x-c)+c+t.
```

刚体模型有 6 个参数：3 个欧拉角和 3 个平移量，采用 ITK Euler3D 的 $R=R_zR_xR_y$ 顺序。仿射模型有 12 个参数：9 个矩阵元素和 3 个平移量，其齐次矩阵中的偏移为 $b=c+t-Bc$。

每个线性阶段在已有累计变换之前优化一个增量变换。若已有变换为 $A_0$，则当前映射为 $T_p=A_0\circ T_{\rm local,p}$。

对相似性损失 $\mathcal D$，参数导数按链式法则在有效样本集合 $\mathcal S$ 上累加：

```math
\frac{\partial\mathcal D}{\partial p_j}
=\sum_{x\in\mathcal S}\frac{\partial\mathcal D}{\partial W_p(x)}
\nabla M(T_p(x))^{\mathsf T}
\frac{\partial T_p(x)}{\partial p_j}.
```

对带中心的仿射变换：

```math
\frac{\partial T_a}{\partial B_{ab}}=x_b-c_b,\qquad
\frac{\partial T_a}{\partial t_b}=\delta_{ab}.
```

旋转导数由旋转矩阵解析计算。强度矩初始化对齐两幅图像的质心；`initialization_feature` 也支持几何中心或原点初始化。

对应实现：[线性模型](../../src/fast_ants/pipeline/models64.py)、[FP32 线性模型](../../src/fast_ants/pipeline/models32.py)和 [CPU 线性配准](../../src/fast_ants/backends/cpu/fp64/linear.py)。

## Mattes 互信息

损失取互信息的负值：

```math
\mathcal D_{\rm MI}=-\sum_{a,b}p_{ab}\log\frac{p_{ab}}{p_a^Fp_b^M},
\qquad p_a^F=\sum_b p_{ab},\quad p_b^M=\sum_a p_{ab}.
```

首先把强度转换到直方图分箱坐标。设分箱数为 $B$、强度范围为 $[I_{\min},I_{\max}]$，则箱宽和强度坐标为：

```math
h=\frac{I_{\max}-I_{\min}}{B-4},\qquad
\widetilde I=\frac{I}{h}-\left(\frac{I_{\min}}h-2\right).
```

固定图像使用零阶 B 样条核 $\beta_0$，移动图像使用三阶 B 样条核：

```math
\beta_3(t)=
\begin{cases}
\frac23-t^2+\frac12|t|^3,&|t|<1,\\
\frac16(2-|t|)^3,&1\le|t|<2,\\
0,&|t|\ge2.
\end{cases}
```

对 $N$ 个有效样本，联合分布估计为：

```math
p_{ab}(p)\approx\frac1N\sum_{x\in\mathcal S}
\beta_0(a-\widetilde F(x))
\beta_3(b-\widetilde M(T_p(x))).
```

实现会处理分箱范围并归一化概率，然后计算对数。概率为零的项不参与求和。

在有效样本集合不变、移动图像箱宽为 $h_M$ 时，概率导数为：

```math
\frac{\partial p_{ab}}{\partial p_j}
\approx-\frac1{Nh_M}\sum_{x\in\mathcal S}
\beta_0(a-\widetilde F(x))\beta_3'(b-\widetilde M(T_p(x)))
\nabla M(T_p(x))^{\mathsf T}\frac{\partial T_p(x)}{\partial p_j}.
```

归一化后，损失导数可写为：

```math
\frac{\partial\mathcal D_{\rm MI}}{\partial p_j}
=-\sum_{a,b}\frac{\partial p_{ab}}{\partial p_j}
\log\frac{p_{ab}}{p_a^Fp_b^M}.
```

固定端零阶分箱和图像有效范围检查会在样本跨越分箱或图像边缘时引入不连续变化。参数导数描述的是当前样本选择下的平滑部分。

对应实现：[CPU FP64 Mattes](../../src/fast_ants/backends/cpu/fp64/metrics/mattes.py)、[CPU FP32 Mattes](../../src/fast_ants/backends/cpu/fp32/metrics/mattes.py)和 [Metal Mattes](../../src/fast_ants/backends/metal/metrics/mattes/metric.py)。

## 线性参数优化

角度、矩阵系数和平移量对图像位置的影响不同。优化器通过参数扰动 $\delta$ 和虚拟网格的 8 个角点 $\mathcal C$，估计每个参数的尺度：

```math
s_j=\left[
\frac{\max_{x\in\mathcal C}\|T_{p+\delta e_j}(x)-T_p(x)\|}{\delta}
\right]^2.
```

零尺度用最小正尺度替代；所有尺度均为零时使用单位尺度。记度量返回的下降向量为 $q_k=-\nabla_p\mathcal D(p_k)$，缩放后的向量和 Polak–Ribiere 共轭搜索方向为：

```math
d_{k,j}=\frac{q_{k,j}}{s_j},\qquad
\gamma_k=\frac{(d_k-d_{k-1})^{\mathsf T}d_k}{d_{k-1}^{\mathsf T}d_{k-1}},
\qquad c_k=d_k+\gamma_kc_{k-1}.
```

首次迭代使用 $\gamma_0=0$。分母过小或 $\gamma_k$ 超出 $[0,5]$ 时，设 $\gamma_k=0$ 重新开始搜索。初始学习率根据第一个搜索方向引起的最大物理位移缩放。随后沿下式做黄金分割搜索：

```math
p(\alpha)=p_k+\alpha c_k.
```

初始区间为 $[0,\alpha_k,2\alpha_k]$。试探位置只需计算度量值，导数在每次优化迭代开始时计算。

### 窗口收敛判据

记度量值为 $E_i$，累计绝对能量为 $A_k=\sum_{i=0}^k|E_i|$。取得长度为 $w$ 的窗口后，用两个线性 B 样条控制值拟合归一化样本 $E_i/A_k$。对窗口坐标 $t_i=i/(w-1)$，令基函数权重为 $b_{i0}=1-t_i$、 $b_{i1}=t_i$：

```math
\omega_j=\sum_i b_{ij}^2,\qquad
\Delta_j=\sum_i\frac{E_i}{A_k}\frac{b_{ij}^3}{b_{i0}^2+b_{i1}^2},
\qquad v_j=\frac{\Delta_j}{\omega_j},\qquad
\kappa_k=-(v_1-v_0).
```

实现中将上端点略微移入 B 样条支持域。 $\kappa_k$ 达到设置的阈值或迭代次数达到上限时停止。能量全部为零时返回零；窗口未满时继续迭代。

对应实现：[FP64 优化器](../../src/fast_ants/pipeline/optimizer64.py)、[FP32 优化器](../../src/fast_ants/pipeline/optimizer32.py)和[窗口收敛计算](../../src/fast_ants/pipeline/convergence.py)。

## 局部归一化互相关

对邻域 $\mathcal N_x$，记有效样本的均值为 $\mu_F,\mu_W$，定义：

```math
A_x=\sum_{y\in\mathcal N_x}(F(y)-\mu_F)(W(y)-\mu_W),
\qquad
B_x=\sum_{y\in\mathcal N_x}(F(y)-\mu_F)^2,\quad
C_x=\sum_{y\in\mathcal N_x}(W(y)-\mu_W)^2.
```

局部得分和记录的损失为：

```math
\mathrm{CC}_x=\frac{A_x^2}{B_xC_x},\qquad
\mathcal D_{\rm CC}=-\frac1{N_c}\sum_{x\ {\rm valid}}\mathrm{CC}_x.
```

其中 $N_c$ 为有效中心数。平方相关系数度量局部线性相关的强度，灰度关系相反时也可能得到较高得分。

ITK 形式的移动图像中心更新为：

```math
g_W(x)=\frac{2A_x}{B_xC_x}
\left[F(x)-\mu_F-\frac{A_x}{C_x}(W(x)-\mu_W)\right]\nabla W(x).
```

这是 SyN 使用的中心局部更新。若对所有重叠窗口的损失之和求完整导数，还需累加每个包含该体素的窗口贡献，两者表达式不同。

图像边缘处的窗口截断到有效邻域，无效中心跳过。局部方差退化时，得分初值为 1、更新为零；只有相应方差检查通过时才计算比值和更新。这些检查使用当前浮点精度对应的阈值。

FP32 统计采用中心化求和与稳定合并。对两个互不重叠的样本组，设样本数为 $n_a,n_b$、均值为 $\mu_a,\mu_b$、离均差平方和为 $V_a,V_b$，Chan 合并公式为：

```math
\delta=\mu_b-\mu_a,\qquad
\mu=\mu_a+\delta\frac{n_b}{n_a+n_b},\qquad
V=V_a+V_b+\delta^2\frac{n_an_b}{n_a+n_b}.
```

协方差合并将 $\delta^2$ 替换为两个变量均值差的乘积，从而避免低对比窗口中两个接近的原始矩相减。

对应实现：[CPU FP64 CC](../../src/fast_ants/backends/cpu/fp64/metrics/cc.py)、[CPU FP32 队列统计](../../src/fast_ants/backends/cpu/fp32/metrics/cc_queue.py)和 [Metal 窗口统计](../../src/fast_ants/backends/metal/metrics/cc/window.py)。

## Greedy SyN

SyN 维护两侧正向映射及其逆场：

```math
P_F=\mathrm{Id}+u_F,\qquad P_M=\mathrm{Id}+u_M,
\qquad P_F^{-1},\quad P_M^{-1}.
```

四个场都位于当前虚拟网格。设累计线性变换为 $A$，当前金字塔图像在公共域中重采样为：

```math
\widehat F(z)=F_\ell(P_F^{-1}(z)),\qquad
\widehat M(z)=M_\ell(A(P_M^{-1}(z))).
```

每次迭代执行以下操作：

1. 按固定／移动两种角色顺序计算局部 CC。第一次调用的更新用于 $P_F$，角色互换后的更新用于 $P_M$。
2. 按更新场方差 $q_u$ 平滑两侧更新。
3. 分别使用全场统一的间距缩放最大范数归一化更新。
4. 将更新与对应正向场复合，再按总场方差 $q_t$ 平滑。
5. 估计各侧逆场，再对逆场求逆以修正对应正向场。
6. 将两次度量值的平均值传给窗口收敛计算。

### 场平滑与步长

位移场滤波使用可分离离散高斯核，其系数正比于：

```math
k_n(q)=e^{-q}I_n(q).
```

其中 $I_n$ 为第一类修正 Bessel 函数， $q$ 为体素单位下的方差。有限核截断后重新归一化。图像金字塔平滑使用的则是 `vox` 或 `mm` 单位下的标准差。

当 $q>0$ 时，各分量依次沿三个轴卷积，执行小方差混合，并将最外层网格向量设为零。网格内部为：

```math
\mathcal S_q(g)=w(q)(k_q*_xk_q*_yk_q*_zg)+(1-w(q))g,\qquad
w(q)=
\begin{cases}
1-q/0.5,&0\lt q\lt 0.5,\\
1,&q\ge0.5.
\end{cases}
```

$\mathcal S_0$ 保持场不变。设平滑更新为 $\widetilde g=\mathcal S_{q_u}(g)$、学习率为 $\eta$、间距为 $s_d$：

```math
m=\max_z\sqrt{\sum_{d=1}^3\left(\frac{\widetilde g_d(z)}{s_d}\right)^2},
\qquad
\Delta u(z)=
\begin{cases}(\eta/m)\widetilde g(z),&m>0,\\
\eta\widetilde g(z),&m=0.\end{cases}
```

同一更新场中的所有向量使用相同的缩放系数。

### 形变复合

对 $U=\mathrm{Id}+u$、 $V=\mathrm{Id}+v$：

```math
(U\circ V)(x)=x+v(x)+u(x+v(x)).
```

SyN 两侧更新为：

```math
P_F^{\rm tmp}=\mathcal S_{q_t}\!\left((\mathrm{Id}+\Delta u_F)\circ P_F\right),
\qquad
P_M^{\rm tmp}=\mathcal S_{q_t}\!\left((\mathrm{Id}+\Delta u_M)\circ P_M\right).
```

这里的平滑作用于复合映射的位移部分。复合时，外层位移场在内层映射变换后的位置插值。

### 逆位移场

对 $U=\mathrm{Id}+u$，其逆映射 $V=\mathrm{Id}+v$ 满足 $r_n(x)=0$，其中：

```math
r_n(x)=v_n(x)+u(x+v_n(x)).
```

令 $e_n(x)=\sqrt{\sum_d(r_{n,d}(x)/s_d)^2}$、 $M_n=\max_xe_n(x)$，受限残差更新为：

```math
f_n(x)=\min\left(1,\frac{\epsilon_nM_n}{e_n(x)}\right),\qquad
v_{n+1}(x)=v_n(x)-\epsilon_nf_n(x)r_n(x).
```

其中 $e_n=0$ 时取 $f_n=1$。首次松弛系数为 $\epsilon_0=0.75$，后续迭代为 $0.5$。启用边界约束时，边缘向量设为零。

默认求逆最多迭代 20 次。每次更新后检查更新前计算的残差统计：最大误差不超过 0.1 **或**平均误差不超过 0.001 时停止，误差采用上述间距缩放范数。

每侧先对临时正向场求逆，再对逆场求逆以修正正向场，因此每次外部 SyN 迭代有四次逆场求解，初始值来自当前场状态。

### 最终变换

完整的非线性映射与图像采样映射为：

```math
\phi=P_M^{-1}\circ P_F,\qquad
\phi^{-1}=P_F^{-1}\circ P_M,\qquad T=A\circ\phi.
```

最终使用 $T$ 在原始固定网格上采样原始移动图像。仿射变换与非线性场分开保存。

对应实现：[CPU SyN](../../src/fast_ants/backends/cpu/fp64/syn.py)、[CPU 场运算](../../src/fast_ants/backends/cpu/fp64/fields.py)、[高斯核系数](../../src/fast_ants/pipeline/gaussian.py)和 [Metal SyN](../../src/fast_ants/backends/metal/syn.py)。

## 多分辨率处理

算法从粗虚拟网格逐层推进到细网格。每层依次平滑强度图像，根据缩小因子构造中心对齐的虚拟网格，再优化线性变换或执行 SyN 更新。进入下一层前，在物理空间重采样四个 SyN 场。

平滑后的度量图像与虚拟网格是不同对象，缩小虚拟网格不等于直接对输入强度数组做切片。物理间距策略会针对各向异性图像调整各轴因子。各层之间的位移始终以毫米表示，改变网格间距不改变向量数值的单位。

对应实现：[虚拟网格](../../src/fast_ants/pipeline/pyramid.py)、[CPU 金字塔准备](../../src/fast_ants/backends/cpu/fp64/preprocessing.py)和[阶段配置](../../src/fast_ants/core/config.py)。

## 浮点计算

CPU FP64 以双精度进行图像和场计算，部分 ANTs 参数转换和图像滤波中间结果保留 FP32 转换。CPU FP32 与 Metal 使用普通 FP32 体素计算；小型几何矩阵、高斯系数、收敛计算和输出质量评估按实现需要使用主机端精度。

FP32 稳定性处理包括中心化 CC 统计、补偿或成对累加、小位移的局部坐标计算，以及递归高斯滤波的二阶节实现。二阶节使用相邻样本差分驱动，在保持常量响应的同时避免大递推项相减，不使用软件扩展浮点表示。

质量评估包括图像相似性、可用地标、Jacobian 行列式和逆一致性。对 $\phi=\mathrm{Id}+u$：

```math
J_\phi(x)=\det(I+\nabla u(x)),\qquad
e_{\rm inv}(x)=\|\phi^{-1}(\phi(x))-x\|.
```

工具使用方法见[测试与质量评估](validation.md)。

## 算法参考

实现遵循标准 ANTs／ITK 配准流程。源码参考为 ANTs 2.6.5，以及使用 ITK 5.4.7 的固定 ANTs 开发版本：

- [ANTs 配准辅助实现](https://github.com/ANTsX/ANTs/blob/38144917d78c20144f4a70e4a6e03c059aaefef8/Examples/itkantsRegistrationHelper.hxx)：阶段设置与度量选择。
- [ITK Mattes MI](https://github.com/InsightSoftwareConsortium/ITK/blob/4c49df934c316ff87dbbd971b89e63df6bbd1836/Modules/Registration/Metricsv4/include/itkMattesMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader.hxx)：Parzen 概率估计与导数。
- [ITK 邻域 CC](https://github.com/InsightSoftwareConsortium/ITK/blob/4c49df934c316ff87dbbd971b89e63df6bbd1836/Modules/Registration/Metricsv4/include/itkANTSNeighborhoodCorrelationImageToImageMetricv4GetValueAndDerivativeThreader.hxx)：局部统计与更新。
- [ITK SyN](https://github.com/InsightSoftwareConsortium/ITK/blob/4c49df934c316ff87dbbd971b89e63df6bbd1836/Modules/Registration/RegistrationMethodsv4/include/itkSyNImageRegistrationMethod.hxx)与[逆场求解](https://github.com/InsightSoftwareConsortium/ITK/blob/4c49df934c316ff87dbbd971b89e63df6bbd1836/Modules/Filtering/DisplacementField/include/itkInvertDisplacementFieldImageFilter.hxx)。
- Avants 等，*Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain*。[DOI: 10.1016/j.media.2007.06.004](https://doi.org/10.1016/j.media.2007.06.004)。
