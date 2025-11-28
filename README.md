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

## Input File Format

- Tab-separated file.

- **Column 1:** time  
- **Column 2:** flux

- **URL input:** list the URLs and the flux/time column names.  
- **Local files:** list file paths and the flux/time column names.

- Example input files are provided in the `example` folder.

## Output

All results are saved to `prediction_results.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `Source` | URL or file path of the light curve |
| `Class` | 1 if probability > 0.99, else 0 |
| `Probability` | Model probability of microlensing |
| `Real_4FWHM_days` | Predicted 4FWHM duration (trustworthy if light curve has 940 points over 27.4 days) |
| `Latent_Space` | 20 latent space values from the model (can be used for reconstruction) |
| `Points` | Number of points in the chunk |
| `Chunk_Description` | Description of the chunk used |

---

## Citation

If you use Microlensify in your research, please cite it appropriately.

---

## Contact

For questions or support, please open an issue or contact **atousakalantari99@gmail.com**.

