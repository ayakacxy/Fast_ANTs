"""Metric degeneracy thresholds for the official ANTs ``--float 1`` target.

ANTs' registration helper instantiates both CC and Mattes with
``TInternalComputationValueType=float``. ITK's CC threader uses that type as
``LocalRealType``; Mattes uses it as ``PDFValueType``. Their epsilon branches
therefore use float epsilon even when our bounded host control uses FP64.

These are metric branch thresholds, not inverse-field stopping tolerances,
error tolerances, or a claim of identical floating-point evaluation order.
"""

import numpy as np


# ITK ANTSNeighborhoodCorrelation...Threader.hxx, ComputeMovingTransformDerivative:
# abs(ff * mm) > epsilon for the score; ff > epsilon and mm > epsilon for updates.
FLOAT1_CC_EPS = np.float32(np.finfo(np.float32).eps)

# ITK MattesMutualInformationImageToImageMetricv4.hxx, ComputeResults:
# joint sum < epsilon is invalid; each contributing normalized PDF must be > epsilon.
FLOAT1_PDF_EPS = np.float32(np.finfo(np.float32).eps)

# Generate the exact same representable threshold in both Metal CC kernels.
FLOAT1_CC_EPS_METAL = f"{float(FLOAT1_CC_EPS):.24g}f"
