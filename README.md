# 1D-CNN-Assisted SERS Quantification — Python

Python implementation of the four-block 1D convolutional regression network in the [MATLAB toolbox](https://github.com/ZihanLiao-ZJU/1D-CNN-Assisted-SERS-Quantification).

The code accompanies:

> Guo, L., Liao, Z., & Yang, T. (2026). Tailored deep learning–assisted in situ SERS: Overcoming surface irregularities-induced large signal variation on biological tissues. *Analytical Chemistry, 98*(25), 18645–18659. [https://doi.org/10.1021/acs.analchem.6c01156](https://doi.org/10.1021/acs.analchem.6c01156)

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Input data format](#input-data-format)
- [Network architecture](#network-architecture)
- [Training settings](#training-settings)
- [Outputs](#outputs)
- [Citation](#citation)
- [License](#license)

## Overview

The two source files correspond to `drive_CNN.m` and `RegressionPlt.m`.

No other repository is needed. Experimental spectra and pretrained weights are not included.

## Requirements

- Python 3.10 or later, NumPy, SciPy, Matplotlib and PyTorch.

Install from this repository directory:

```bash
python -m pip install -r requirements.txt
```

CPU is the default. An appropriate CUDA-enabled PyTorch installation can be used with `--device cuda`.

## Repository structure

```text
.
├── drive_CNN.py          # MAT-file loading, network, training, prediction and command-line entry.
├── RegressionPlt.py      # regression metrics and calibration/test plots.
├── requirements.txt      # dependencies.
├── LICENSE               # MIT license.
└── README.md
```

## Quick start

With your pre-split `data_CNN.mat` file:

```bash
python drive_CNN.py --data data_CNN.mat
```

A self-contained installation demonstration is available without external data:

```bash
python drive_CNN.py --demo --epochs 3 --output demo_results
```

**The demonstration generates synthetic spectra, not the paper's experimental data.** It is intended only to exercise loading-equivalent shapes, training and evaluation. Default real-data training uses 500 epochs; `--epochs`, `--batch-size`, `--seed`, `--threads`, `--output` and `--no-plots` can be specified. Use `python drive_CNN.py --help` for all options.

## Input data format

Use a MATLAB v5/v7 MAT-file (not the HDF5-based v7.3 format). It must contain `X_tra`, `X_vad`, `X_tst` with shape **spectral points × samples**, and `Y_tra`, `Y_vad`, `Y_tst` with one target per sample. Row and column target vectors are accepted. Arrays must be real and finite; the three sets must have equal spectral length, at least 16 points. Optional scalar counts `N_dim`, `N_sam`, `N_tra`, `N_vad`, `N_tst` are checked against the arrays.

The supplied train/validation/test split is retained. No automatic normalization, resampling or splitting is performed. PyTorch tensors use sample × channel × spectral-position layout, with one input channel. Split by independent specimen/measurement group before producing the MAT-file when spectra are repeated measurements of the same specimen.

## Network architecture

Four blocks use convolution → batch normalization → ReLU → same-padding stride-2 max pooling (output length `ceil(input length/2)`). Kernel sizes are `[57, 37, 11, 3]` and channel counts `[8, 16, 32, 32]`. Global average pooling is followed by a fully connected 8-feature layer, ReLU and a scalar linear output.

## Training settings

Training uses mean squared error, Adam (`lr=0.001`, betas `0.9/0.999`, epsilon `1e-8`), batch size 32, reshuffling each epoch, 500 epochs and a learning-rate factor of 0.5 after every 500 epochs. L2 strength is `1e-4` for convolution/linear weights and batch-normalization scale/offset; convolution/linear biases are not penalized. Weights use Glorot-normal initialization, biases zero, batch-normalization epsilon `1e-5` and moving-statistic update weight 0.1. Validation is evaluated at the end of each epoch, never used for gradient updates; the test set is used only after training.

By default, the model with the lowest validation MSE is returned, corresponding to `trainnet`'s automatic output selection with validation data in MATLAB R2024a and later. Use `--output-network last` for the last-iteration selection used by older versions. The seed controls the Python run; random-number streams and training kernels are framework-specific.

## Outputs

The output directory contains `model.pt`, `predictions.npz`, `metrics.json` and, unless disabled, `calibration.png` and `test.png`. Calibration combines training and validation, matching the MATLAB workflow; it is **not** an independent generalization estimate. Use test-set metrics for held-out evaluation.

`RegressionPlt` returns five two-element arrays (calibration, test): Pearson correlation, R², bias, RMSEP and LOD. LOD is three times the sample standard deviation of blank predictions (`target == 0`), with no slope division. Missing blanks and undefined correlation/R² are recorded as JSON `null`. Test plots include an OLS line and pointwise 95% confidence/prediction intervals where the regression is identifiable.

To use a saved Python model for new spectra with the same spectral axis and preprocessing:

```python
import numpy as np
import torch
from drive_CNN import CNN, predict

checkpoint = torch.load('results_CNN/model.pt', map_location='cpu', weights_only=True)
model = CNN()
model.load_state_dict(checkpoint['state_dict'])
# X_new: [spectral points, samples]
assert X_new.shape[0] == checkpoint['spectral_length']
x = torch.from_numpy(np.ascontiguousarray(X_new.T[:, None, :], dtype=np.float32))
y_predicted = predict(model, x)
```

## Citation

If you use this toolbox in your research, please cite:

> Guo, L., Liao, Z., & Yang, T. (2026). Tailored deep learning–assisted in situ SERS: Overcoming surface irregularities-induced large signal variation on biological tissues. *Analytical Chemistry, 98*(25), 18645–18659. [https://doi.org/10.1021/acs.analchem.6c01156](https://doi.org/10.1021/acs.analchem.6c01156)

```bibtex
@article{guo2026tailored,
  title={Tailored Deep Learning--Assisted In Situ SERS: Overcoming Surface Irregularities-Induced Large Signal Variation on Biological Tissues},
  author={Guo, Ling and Liao, Zihan and Yang, Tianxi},
  journal={Analytical Chemistry},
  volume={98}, number={25}, pages={18645--18659}, year={2026},
  doi={10.1021/acs.analchem.6c01156}
}
```

## License

Licensed under the [MIT License](LICENSE).
