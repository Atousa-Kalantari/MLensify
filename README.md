# Microlensify

**Deep-learning microlensing classifier — works with any light curve from any telescope**

Microlensify is a deep learning model that detects microlensing events in light curves. It works with light curves from any telescope, either via URLs (e.g., MAST FITS files) or your own data files.

Microlensify splits your light curve into chunks and downsamples them to a fixed length for prediction. Predictions are made for each chunk individually and also for the whole light curve.

---

## Features

- Works with any telescope light curve.
- Handles both URL-based FITS files and local files.
- Automatically splits light curves into chunks for robust predictions.
- Provides latent space representations for advanced analysis or reconstruction.
- Predicts event duration (4FWHM) and probability of microlensing.

---

## Installation

```bash
pip install Microlensify
```

## Usage
```bash
Microlensify <input_file> <compute_stats> <n_cores>
```

## Arguments

### `<input_file>`

- **URL input:** a file containing URLs of light curves and their flux & time column names.  
- **Local files:** a file containing paths to your files and their flux & time column names.

### `<compute_stats>`

- `yes` — compute flux statistics (min, max, median, std, std/(max-min)) from raw flux.  
- `no` — flux is already normalized (like TESS QLP pipeline flux), so the model uses fixed values from the training set for these statistics.

### `<n_cores>`

- Number of CPU cores to use for parallel processing.
