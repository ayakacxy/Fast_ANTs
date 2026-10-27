# Registration algorithms

[English](algorithms.md) · [简体中文](../zh-CN/algorithms.md) · [Documentation](index.md)

Fast ANTs uses Mattes mutual information for rigid and affine registration, followed by displacement-field greedy SyN with local normalized cross-correlation. This page describes the mathematical operations performed by these stages. CPU, Metal and CUDA execution details are covered in [implementation optimizations](optimizations.md).

## Image sampling and coordinates

Let $F:\Omega_F\to\mathbb R$ be the fixed image and $M:\Omega_M\to\mathbb R$ the moving image. Registration computes a sampling map and evaluates the output on the fixed grid:

```math
W(x)=M(T(x)),\qquad T:\Omega_F\rightarrow\Omega_M.
```

Thus `fixed_to_moving` maps a fixed-space point to a sampling position in the moving image. The output is the moving image resampled onto the fixed grid.

All physical coordinates and displacement components use LPS millimetres. Arrays use `[z,y,x]`; index and point vectors use `[x,y,z]`. For origin $o$, direction matrix $D$, spacing matrix $S=\mathrm{diag}(s_x,s_y,s_z)$, and index $i$:

```math
x=o+DSi,\qquad i=(DS)^{-1}(x-o).
```

NIfTI RAS coordinates are converted with $C=\mathrm{diag}(-1,-1,1)$: $x_{\rm LPS}=Cx_{\rm RAS}$. The same conversion applies to physical vector components. See [inputs and outputs](inputs-outputs.md) for file formats and transform order.

For continuous index $q=n+t$, where $n=\lfloor q\rfloor$ and $t\in[0,1)^3$, trilinear interpolation is

```math
M(q)=\sum_{b\in\{0,1\}^3}M[n+b]\prod_{d=1}^3w_{b_d}(t_d),
\qquad w_0(t)=1-t,\quad w_1(t)=t.
```

Sampling checks image support and applies the configured outside value. The metric uses a finite-difference image gradient at the transformed location. Index and physical gradients satisfy

```math
\nabla_xM=(DS)^{-\mathsf T}\nabla_iM.
```

## Rigid and affine transforms

With centre $c$, translation $t$, rotation $R$, and affine matrix $B$:

```math
T_R(x)=R(\theta)(x-c)+c+t,\qquad
T_A(x)=B(x-c)+c+t.
```

The rigid model has 6 parameters: 3 Euler angles and 3 translations. It uses the ITK Euler3D order $R=R_zR_xR_y$. The affine model has 12 parameters: 9 matrix entries and 3 translations. Its homogeneous offset is $b=c+t-Bc$.

Each linear stage optimizes an incremental transform before the accumulated transform from the preceding stage. If that accumulated transform is $A_0$, the current map is $T_p=A_0\circ T_{\rm local,p}$.

For similarity loss $\mathcal D$, the parameter derivative follows the chain rule over valid samples $\mathcal S$:

```math
\frac{\partial\mathcal D}{\partial p_j}
=\sum_{x\in\mathcal S}\frac{\partial\mathcal D}{\partial W_p(x)}
\nabla M(T_p(x))^{\mathsf T}
\frac{\partial T_p(x)}{\partial p_j}.
```

For a centred affine transform:

```math
\frac{\partial T_a}{\partial B_{ab}}=x_b-c_b,\qquad
\frac{\partial T_a}{\partial t_b}=\delta_{ab}.
```

Rotation derivatives are computed analytically from the rotation matrices. Intensity-moment initialization aligns image centres of mass; `initialization_feature` also supports geometric-centre or origin initialization.

Implementation: [linear models](../../src/fast_ants/pipeline/models64.py), [FP32 linear models](../../src/fast_ants/pipeline/models32.py), and [CPU linear registration](../../src/fast_ants/backends/cpu/fp64/linear.py).

## Mattes mutual information

The loss is negative mutual information:

```math
\mathcal D_{\rm MI}=-\sum_{a,b}p_{ab}\log\frac{p_{ab}}{p_a^Fp_b^M},
\qquad p_a^F=\sum_b p_{ab},\quad p_b^M=\sum_a p_{ab}.
```

Intensities are mapped to histogram-bin coordinates. With $B$ bins and range $[I_{\min},I_{\max}]$, the bin width and coordinate are

```math
h=\frac{I_{\max}-I_{\min}}{B-4},\qquad
\widetilde I=\frac{I}{h}-\left(\frac{I_{\min}}h-2\right).
```

The fixed image uses a zero-order B-spline kernel $\beta_0$; the moving image uses the cubic B-spline kernel

```math
\beta_3(t)=
\begin{cases}
\frac23-t^2+\frac12|t|^3,&|t|<1,\\
\frac16(2-|t|)^3,&1\le|t|<2,\\
0,&|t|\ge2.
\end{cases}
```

For $N$ valid samples, the joint distribution is estimated by

```math
p_{ab}(p)\approx\frac1N\sum_{x\in\mathcal S}
\beta_0(a-\widetilde F(x))
\beta_3(b-\widetilde M(T_p(x))).
```

The implementation applies bin limits and probability normalization before evaluating logarithms. Zero-probability entries do not contribute to the sum.

For a fixed valid-sample set and moving bin width $h_M$, the PDF derivative is

```math
\frac{\partial p_{ab}}{\partial p_j}
\approx-\frac1{Nh_M}\sum_{x\in\mathcal S}
\beta_0(a-\widetilde F(x))\beta_3'(b-\widetilde M(T_p(x)))
\nabla M(T_p(x))^{\mathsf T}\frac{\partial T_p(x)}{\partial p_j}.
```

After normalization, the loss derivative can be written as

```math
\frac{\partial\mathcal D_{\rm MI}}{\partial p_j}
=-\sum_{a,b}\frac{\partial p_{ab}}{\partial p_j}
\log\frac{p_{ab}}{p_a^Fp_b^M}.
```

The zero-order fixed bin and image-support checks introduce discontinuities when a sample crosses a bin or image edge. Parameter derivatives describe the smooth part for the current sample selection.

Implementation: [CPU FP64 Mattes](../../src/fast_ants/backends/cpu/fp64/metrics/mattes.py), [CPU FP32 Mattes](../../src/fast_ants/backends/cpu/fp32/metrics/mattes.py), [Metal Mattes](../../src/fast_ants/backends/metal/metrics/mattes/metric.py), and [CUDA Mattes](../../src/fast_ants/backends/cuda/metrics/mattes/metric.py).

## Linear optimization

Angles, matrix entries, and translations have different physical effects. The optimizer estimates a scale for each parameter using perturbation $\delta$ and the 8 corners $\mathcal C$ of the virtual grid:

```math
s_j=\left[
\frac{\max_{x\in\mathcal C}\|T_{p+\delta e_j}(x)-T_p(x)\|}{\delta}
\right]^2.
```

A zero scale is replaced by the smallest positive scale; if all scales vanish, unit scales are used. Let $q_k=-\nabla_p\mathcal D(p_k)$ be the metric's descent vector. The scaled vector and Polak–Ribiere search direction are

```math
d_{k,j}=\frac{q_{k,j}}{s_j},\qquad
\gamma_k=\frac{(d_k-d_{k-1})^{\mathsf T}d_k}{d_{k-1}^{\mathsf T}d_{k-1}},
\qquad c_k=d_k+\gamma_kc_{k-1}.
```

The first direction uses $\gamma_0=0$. A very small denominator, or $\gamma_k$ outside $[0,5]$, restarts the search with $\gamma_k=0$. The initial learning rate is scaled by the maximum physical displacement caused by the first direction. Golden-section search evaluates the loss along

```math
p(\alpha)=p_k+\alpha c_k.
```

with initial bracket $[0,\alpha_k,2\alpha_k]$. Trial positions need only the metric value; derivatives are computed at the start of an optimizer iteration.

### Windowed convergence

Let $E_i$ be the metric value and $A_k=\sum_{i=0}^k|E_i|$. Once a window of $w$ samples is available, its normalized samples $E_i/A_k$ are fitted with two linear B-spline control values. For window coordinate $t_i=i/(w-1)$, use basis weights $b_{i0}=1-t_i$, $b_{i1}=t_i$:

```math
\omega_j=\sum_i b_{ij}^2,\qquad
\Delta_j=\sum_i\frac{E_i}{A_k}\frac{b_{ij}^3}{b_{i0}^2+b_{i1}^2},
\qquad v_j=\frac{\Delta_j}{\omega_j},\qquad
\kappa_k=-(v_1-v_0).
```

The upper endpoint is offset slightly inside the B-spline support in the implementation. Iteration stops when $\kappa_k$ reaches the configured threshold or the iteration limit is reached. All-zero energy returns zero; an incomplete window continues iterating.

Implementation: [FP64 optimizer](../../src/fast_ants/pipeline/optimizer64.py), [FP32 optimizer](../../src/fast_ants/pipeline/optimizer32.py), and [window convergence](../../src/fast_ants/pipeline/convergence.py).

## Local normalized cross-correlation

For neighbourhood $\mathcal N_x$, let $\mu_F,\mu_W$ be the means of valid samples and define

```math
A_x=\sum_{y\in\mathcal N_x}(F(y)-\mu_F)(W(y)-\mu_W),
\qquad
B_x=\sum_{y\in\mathcal N_x}(F(y)-\mu_F)^2,\quad
C_x=\sum_{y\in\mathcal N_x}(W(y)-\mu_W)^2.
```

The local score and reported loss are

```math
\mathrm{CC}_x=\frac{A_x^2}{B_xC_x},\qquad
\mathcal D_{\rm CC}=-\frac1{N_c}\sum_{x\ {\rm valid}}\mathrm{CC}_x.
```

where $N_c$ is the number of valid centres. Squared correlation measures the magnitude of local linear correlation, including a reversed intensity relationship.

The ITK-style moving update at the centre is

```math
g_W(x)=\frac{2A_x}{B_xC_x}
\left[F(x)-\mu_F-\frac{A_x}{C_x}(W(x)-\mu_W)\right]\nabla W(x).
```

This is the centre-local update used by SyN. The full derivative of a sum of overlapping window losses would additionally sum contributions from every window containing the voxel; it is a different expression.

At image edges, windows are clipped to valid neighbours. Invalid centres are skipped. For degenerate variance, the score starts at 1 and the update is zero; the ratio and update are evaluated only when their respective variance checks pass. Those checks use the selected floating-point precision.

FP32 statistics use centred sums and stable merging. For two disjoint groups with sizes $n_a,n_b$, means $\mu_a,\mu_b$, and centred squared sums $V_a,V_b$, Chan merging gives

```math
\delta=\mu_b-\mu_a,\qquad
\mu=\mu_a+\delta\frac{n_b}{n_a+n_b},\qquad
V=V_a+V_b+\delta^2\frac{n_an_b}{n_a+n_b}.
```

The covariance merge uses the product of the two mean differences in place of $\delta^2$. This avoids subtracting nearly equal raw moments in low-contrast windows.

Implementation: [CPU FP64 CC](../../src/fast_ants/backends/cpu/fp64/metrics/cc.py), [CPU FP32 queue statistics](../../src/fast_ants/backends/cpu/fp32/metrics/cc_queue.py), [Metal window statistics](../../src/fast_ants/backends/metal/metrics/cc/window.py), and [CUDA CC](../../src/fast_ants/backends/cuda/metrics/cc/operators.py).

## Greedy SyN

SyN maintains two forward maps and their inverse fields:

```math
P_F=\mathrm{Id}+u_F,\qquad P_M=\mathrm{Id}+u_M,
\qquad P_F^{-1},\quad P_M^{-1}.
```

All four fields use the current virtual grid. With accumulated linear transform $A$, the pyramid images are sampled into a common domain:

```math
\widehat F(z)=F_\ell(P_F^{-1}(z)),\qquad
\widehat M(z)=M_\ell(A(P_M^{-1}(z))).
```

An iteration performs these operations:

1. Evaluate local CC with fixed/moving roles in both orders. The first update belongs to $P_F$; the reversed call produces the update for $P_M$.
2. Smooth both updates with update variance $q_u$.
3. Normalize each smoothed update with a global spacing-scaled maximum.
4. Compose updates with the corresponding forward fields and apply total-field smoothing with variance $q_t$.
5. Estimate each inverse, then invert it again to correct the corresponding forward field.
6. Pass the average of the two metric values to windowed convergence.

### Field smoothing and step size

The displacement-field filter uses separable discrete Gaussian coefficients proportional to

```math
k_n(q)=e^{-q}I_n(q).
```

where $I_n$ is the modified Bessel function and $q$ is a variance in voxel units. The finite kernel is normalized after truncation. Image-pyramid smoothing instead takes a standard deviation in `vox` or `mm`.

For $q>0$, each component is convolved along the three axes, a small-variance blend is applied, and outer boundary vectors are zeroed. Inside the grid,

```math
\mathcal S_q(g)=w(q)(k_q*_xk_q*_yk_q*_zg)+(1-w(q))g,\qquad
w(q)=
\begin{cases}
1-q/0.5,&0\lt q\lt 0.5,\\
1,&q\ge0.5.
\end{cases}
```

$\mathcal S_0$ leaves the field unchanged. For smoothed update $\widetilde g=\mathcal S_{q_u}(g)$, learning rate $\eta$, and spacing $s_d$:

```math
m=\max_z\sqrt{\sum_{d=1}^3\left(\frac{\widetilde g_d(z)}{s_d}\right)^2},
\qquad
\Delta u(z)=
\begin{cases}(\eta/m)\widetilde g(z),&m>0,\\
\eta\widetilde g(z),&m=0.\end{cases}
```

The same scale is applied to every vector in that update field.

### Composition

For $U=\mathrm{Id}+u$ and $V=\mathrm{Id}+v$:

```math
(U\circ V)(x)=x+v(x)+u(x+v(x)).
```

Each SyN side is updated as

```math
P_F^{\rm tmp}=\mathcal S_{q_t}\!\left((\mathrm{Id}+\Delta u_F)\circ P_F\right),
\qquad
P_M^{\rm tmp}=\mathcal S_{q_t}\!\left((\mathrm{Id}+\Delta u_M)\circ P_M\right).
```

Here smoothing acts on the displacement part of the composed map. Composition samples the outer displacement at the location transformed by the inner map.

### Inverse displacement fields

For $U=\mathrm{Id}+u$, the inverse $V=\mathrm{Id}+v$ satisfies $r_n(x)=0$, where

```math
r_n(x)=v_n(x)+u(x+v_n(x)).
```

Let $e_n(x)=\sqrt{\sum_d(r_{n,d}(x)/s_d)^2}$ and $M_n=\max_xe_n(x)$. The bounded residual update is

```math
f_n(x)=\min\left(1,\frac{\epsilon_nM_n}{e_n(x)}\right),\qquad
v_{n+1}(x)=v_n(x)-\epsilon_nf_n(x)r_n(x).
```

with $f_n=1$ when $e_n=0$. The first relaxation factor is $\epsilon_0=0.75$; subsequent iterations use $0.5$. Boundary vectors are zeroed when boundary enforcement is enabled.

The default solver uses at most 20 iterations. After each update it checks the residual statistics computed before that update: maximum error at most 0.1 **or** mean error at most 0.001 stops the solve. Errors use the spacing-scaled norm above.

Each side computes an inverse of its temporary forward field, then an inverse of that inverse to correct the forward field. There are four inverse solves per outer SyN iteration, initialized from the current field state.

### Final transform

The complete nonlinear and image-sampling maps are

```math
\phi=P_M^{-1}\circ P_F,\qquad
\phi^{-1}=P_F^{-1}\circ P_M,\qquad T=A\circ\phi.
```

The output samples the original moving image with $T$ on the original fixed grid. The affine and nonlinear fields are stored separately.

Implementation: [CPU SyN](../../src/fast_ants/backends/cpu/fp64/syn.py), [CPU field operations](../../src/fast_ants/backends/cpu/fp64/fields.py), [Gaussian coefficients](../../src/fast_ants/pipeline/gaussian.py), [Metal SyN](../../src/fast_ants/backends/metal/syn.py), and [CUDA SyN](../../src/fast_ants/backends/cuda/syn.py).

## Multiresolution processing

The algorithm advances from a coarse virtual grid to a fine one. At each level it smooths intensity images, constructs a centred virtual grid with the shrink factors, and fits the linear transform or performs SyN updates. The four SyN fields are resampled in physical space before advancing to the next level.

Smoothed metric images and the virtual grid are separate objects. Shrinking the virtual grid does not imply slicing the input intensity array. The physical-spacing policy adjusts axis-specific factors for anisotropic images. Displacements remain in millimetres between levels, so a change in grid spacing does not rescale vector values.

Implementation: [virtual grids](../../src/fast_ants/pipeline/pyramid.py), [CPU level preparation](../../src/fast_ants/backends/cpu/fp64/preprocessing.py), and [stage configuration](../../src/fast_ants/core/config.py).

## Floating-point calculations

CPU FP64 uses double precision for image and field calculations; selected ANTs parameter conversions and image-filter intermediates retain FP32 conversions. CPU FP32, Metal and CUDA use ordinary FP32 voxel calculations. Small geometry matrices, Gaussian coefficient construction, and convergence calculations use host precision as required by the implementation. Metal output QA uses CPU arithmetic; CUDA output QA promotes field values to native FP64 on the device.

FP32 stability uses centred CC statistics, compensated or pairwise accumulation, local coordinates for small displacements, and second-order sections for recursive Gaussian filtering. The sections operate on adjacent sample differences and preserve the constant response without subtracting large recurrence terms. No software-expanded floating-point representation is used.

Quality assessment includes image similarity, landmarks where available, Jacobian determinants, and inverse consistency. For $\phi=\mathrm{Id}+u$:

```math
J_\phi(x)=\det(I+\nabla u(x)),\qquad
e_{\rm inv}(x)=\|\phi^{-1}(\phi(x))-x\|.
```

See [testing and quality assessment](validation.md) for available tools.

## Algorithm references

The implementation follows the standard ANTs/ITK registration path. Source references are ANTs 2.6.5 and the pinned ANTs development source using ITK 5.4.7:

- [ANTs registration helper](https://github.com/ANTsX/ANTs/blob/38144917d78c20144f4a70e4a6e03c059aaefef8/Examples/itkantsRegistrationHelper.hxx): stage setup and metric selection.
- [ITK Mattes MI](https://github.com/InsightSoftwareConsortium/ITK/blob/4c49df934c316ff87dbbd971b89e63df6bbd1836/Modules/Registration/Metricsv4/include/itkMattesMutualInformationImageToImageMetricv4GetValueAndDerivativeThreader.hxx): Parzen estimation and derivatives.
- [ITK neighbourhood CC](https://github.com/InsightSoftwareConsortium/ITK/blob/4c49df934c316ff87dbbd971b89e63df6bbd1836/Modules/Registration/Metricsv4/include/itkANTSNeighborhoodCorrelationImageToImageMetricv4GetValueAndDerivativeThreader.hxx): local statistics and updates.
- [ITK SyN](https://github.com/InsightSoftwareConsortium/ITK/blob/4c49df934c316ff87dbbd971b89e63df6bbd1836/Modules/Registration/RegistrationMethodsv4/include/itkSyNImageRegistrationMethod.hxx) and [inverse-field solver](https://github.com/InsightSoftwareConsortium/ITK/blob/4c49df934c316ff87dbbd971b89e63df6bbd1836/Modules/Filtering/DisplacementField/include/itkInvertDisplacementFieldImageFilter.hxx).
- Avants et al., *Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain*. [DOI: 10.1016/j.media.2007.06.004](https://doi.org/10.1016/j.media.2007.06.004).
