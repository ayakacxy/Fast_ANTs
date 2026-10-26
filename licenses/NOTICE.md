# Third-party notices

[English](NOTICE.md) · [简体中文](NOTICE.zh-CN.md)

## Fast ANTs

Fast ANTs is distributed under the GNU General Public License version 3
only (`GPL-3.0-only`); the license text is in [LICENSE](../LICENSE).
The upstream portions described below retain their original copyright,
license terms and attribution. Their Apache-2.0 license texts are included
alongside the project license.

## ITK

Portions of Fast ANTs are modified translations of ITK source code,
Copyright NumFOCUS, licensed under the Apache License 2.0. The license text
is included in [LICENSE-ITK](LICENSE-ITK).

The main source reference is ITK commit
`4c49df934c316ff87dbbd971b89e63df6bbd1836` (5.4.7). Reference files and
checksums are listed in the [ITK source manifest](../docs/assets/provenance/itk-v5.4.7-sources.json).

| Component | ITK source | Python implementation |
| --- | --- | --- |
| Gaussian filtering | `itkRecursiveGaussianImageFilter.hxx`, `itkRecursiveSeparableImageFilter.hxx`, `itkGaussianOperator.hxx` | `pipeline/gaussian.py`, CPU filtering modules, Metal `filtering.py` and `gaussian_sections.py` |
| Mattes mutual information | Mattes metric and value/derivative threaders | CPU and Metal `metrics/mattes`, `pipeline/mattes32.py`, `pipeline/mattes64.py` |
| Local cross-correlation | `itkANTSNeighborhoodCorrelationImageToImageMetricv4GetValueAndDerivativeThreader.hxx` | CPU and Metal `metrics/cc` |
| Field composition and inversion | `itkComposeDisplacementFieldsImageFilter.hxx`, `itkInvertDisplacementFieldImageFilter.hxx` | CPU and Metal field modules |
| Standard greedy SyN | `itkSyNImageRegistrationMethod.hxx` | CPU and Metal `syn.py`, `core/state.py` |
| Optimization | Conjugate-gradient line search, physical-shift parameter scales, window convergence | `pipeline/optimizer32.py`, `pipeline/optimizer64.py`, `pipeline/convergence.py` |
| Linear transforms | Euler3DTransform, MatrixOffsetTransformBase, CompositeTransform | `core/transforms.py`, `pipeline/models32.py`, `pipeline/models64.py` |
| Initialization and image geometry | ImageMomentsCalculator, ImageBase, NiftiImageIO | CPU linear initialization, `core/types.py`, `io/images.py`, `io/transforms.py` |

Implementation paths are relative to `src/fast_ants`.

The translations replace C++ templates, image pipelines and iterators with
Python array interfaces, Numba loops and custom Metal kernels. Modifications
include parallel execution, fused sampling and metric evaluation, centred
local statistics, streaming derivative histograms, reusable buffers, local
coordinate arithmetic and asynchronous submission of independent GPU work.
FP32 recursive Gaussian filtering uses two second-order sections with
adjacent-pixel differences. Scalar FMA uses native hardware instructions.
Selected CPU FP32 sampling, histogram and neighbourhood operations use
four-lane LLVM vectors, reusable value workspaces and cached zero-background
gradient support.

## ANTs

Image preprocessing follows ANTs `PreprocessImage` and ITK histogram endpoint
and intensity-window calculations. The upstream Apache License 2.0 text is
included in [LICENSE-ANTs](LICENSE-ANTs), copied from the pinned ANTs
`COPYING.txt`. The reference commit is
`38144917d78c20144f4a70e4a6e03c059aaefef8`.

ANTs and ITK reference files retain their original copyright and license notices.

## Small-matrix routines

`core/linalg.py` specializes the LINPACK DSVDC bidiagonal SVD/QR algorithm
and its BLAS scaled norms and Givens rotations to 3×3 FP64 geometry.
The references are unmodified Netlib routines included in ITK 5.4.1;
source URLs and checksums are retained in the
[numerical source manifest](../docs/assets/provenance/itk-v5.4.1-numerical-sources.json).

The Python/Numba implementation uses zero-based arrays, specialized scalar
QR operations and explicit hardware FMA. VNL is used only by separate
reference tools, not the registration runtime.

## Data and dependencies

MRI and template source information is maintained under `validation/sources`
and `templates`. Their redistribution conditions are independent of the
software licenses above.

NumPy, Numba, NiBabel and optional MLX retain their respective licenses.
