# Microlensify/core.py
import os
import csv
import numpy as np
import random as rn
import pandas as pd
from astropy.table import Table
from pathlib import Path
import tensorflow as tf
import joblib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import tempfile
import urllib.request
import warnings

from .model import Sampling, CVAE

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ========================== AUTO-DOWNLOAD ==========================
ASSETS_DIR = Path.home() / ".microlensify_assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
RELEASE_TAG = "v1.0"
BASE_URL = f"https://github.com/Atousa-Kalantari/Microlensify/releases/download/{RELEASE_TAG}"
FILES = [
    "Microlensify_Model.keras",
    "scaler_4fwhm.pkl",
    "scaler_std_div_diff.pkl",
    "scaler_max_flux.pkl",
    "scaler_min_flux.pkl",
    "scaler_median_flux.pkl",
    "scaler_std_flux.pkl",
]
MODEL_PATH = ASSETS_DIR / "Microlensify_Model.keras"

def ensure_assets():
    from urllib.request import urlopen
    def download_with_progress(url, dest_path):
        if dest_path.exists():
            return
        print(f"Downloading {dest_path.name} ... ", end="", flush=True)
        with urlopen(url) as response, open(dest_path, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', 0))
            downloaded = 0
            block_size = 1024 * 1024
            while True:
                data = response.read(block_size)
                if not data:
                    break
                out_file.write(data)
                downloaded += len(data)
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    print(f"\rDownloading {dest_path.name} ... {downloaded//(1024*1024)}/{total_size//(1024*1024)} MB ({percent:.1f}%)", end="", flush=True)
        print(" done")

    missing = False
    for fname in FILES:
        fpath = ASSETS_DIR / fname
        if not fpath.exists():
            if not missing:
                print("First run — downloading model and scalers...")
                missing = True
            download_with_progress(f"{BASE_URL}/{fname}", fpath)
    if missing:
        print("Download complete! Future runs are offline.\n")

# ========================== CONSTANTS ==========================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)
TESS_SECTOR_DAYS = 27.4

FIXED_STATS = {
    "max": 2379.638043809108,
    "min": 1470.406889942004,
    "median": 1663.0377499789672,
    "std": 89.51542107215877,
}

# ========================== LOAD MODEL ==========================
print("Loading Microlensify model and scalers (10–30 sec first time)...")
ensure_assets()

cvae = tf.keras.models.load_model(MODEL_PATH, custom_objects={'Sampling': Sampling, 'CVAE': CVAE})
encoder = cvae.encoder
decoder = cvae.decoder

scaler_fwhm = joblib.load(ASSETS_DIR / "scaler_4fwhm.pkl")
scaler_std_div_diff = joblib.load(ASSETS_DIR / "scaler_std_div_diff.pkl")
scaler_max = joblib.load(ASSETS_DIR / "scaler_max_flux.pkl")
scaler_min = joblib.load(ASSETS_DIR / "scaler_min_flux.pkl")
scaler_median = joblib.load(ASSETS_DIR / "scaler_median_flux.pkl")
scaler_std = joblib.load(ASSETS_DIR / "scaler_std_flux.pkl")
print("Model ready!\n")

# ========================== HELPERS ==========================
def adjust_to_940_points(arr):
    arr = np.array(arr, dtype=float)
    n = len(arr)
    if n == 940:
        return arr.copy()
    elif n > 940:
        drop_idx = set(rn.sample(range(n), n - 940))
        return np.array([arr[i] for i in range(n) if i not in drop_idx])
    else:
        min_val = np.min(arr)
        rng = np.random.default_rng(SEED)
        noise = 0.001 * min_val * rng.standard_normal(940 - n)
        return np.concatenate([arr, min_val + noise])

def safe_log10(x):
    return np.log10(np.clip(x, 1e-10, None))

# ========================== PREDICTION ==========================
def predict_on_chunk(chunk_flux, chunk_time, description, source, compute_stats_flag):
    if compute_stats_flag == "yes":
        fmax = np.max(chunk_flux)
        fmin = np.min(chunk_flux)
        fdiff = fmax - fmin + 1e-12
        fmed = np.median(chunk_flux)
        fstd = np.std(chunk_flux)
    else:
        fmax = FIXED_STATS["max"]
        fmin = FIXED_STATS["min"]
        fdiff = fmax - fmin
        fmed = FIXED_STATS["median"]
        fstd = FIXED_STATS["std"]

    norm_max = scaler_max.transform([[safe_log10(fmax)]])[0,0]
    norm_min = scaler_min.transform([[safe_log10(fmin)]])[0,0]
    norm_std = scaler_std.transform([[safe_log10(fstd)]])[0,0]
    norm_median = scaler_median.transform([[safe_log10(fmed)]])[0,0]
    norm_std_div_diff = scaler_std_div_diff.transform([[safe_log10(fstd / fdiff)]])[0,0]

    scalar_test = np.array([norm_max, norm_min, norm_std, norm_median, norm_std_div_diff]).reshape(1, 5)
    normflux = (chunk_flux - np.min(chunk_flux)) / (np.max(chunk_flux) - np.min(chunk_flux) + 1e-12)
    x_test = adjust_to_940_points(normflux).reshape(1, 940, 1)

    z_mean, _, z_sampled, class_pred = encoder.predict([x_test, scalar_test], verbose=0)
    decoder.predict(z_sampled, verbose=0)

    prob = float(class_pred[0][0])
    y_pred = int(prob > 0.99)
    pred_norm_fwhm = z_mean[0, -2]
    pred_fwhm_model_units = 10 ** scaler_fwhm.inverse_transform([[pred_norm_fwhm]])[0,0]
    time_span_days = max(chunk_time[-1] - chunk_time[0], 1e-6)
    real_4fwhm_days = pred_fwhm_model_units * (time_span_days / TESS_SECTOR_DAYS)

    latent_str = '"' + ",".join([f"{v:.6f}" for v in z_mean.flatten()]) + '"'
    return [source, y_pred, f"{prob:.6f}", f"{real_4fwhm_days:.3f}", latent_str, len(chunk_flux), description]

# ========================== PROCESS ONE SOURCE ==========================
def process_source(args):
    source, flux_col, time_col, compute_stats_flag = args
    results = []
    tmp_file = None
    try:
        if source.startswith("http"):
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.fits')
            urllib.request.urlretrieve(source, tmp_file.name)
            filepath = tmp_file.name
        else:
            filepath = source

        if filepath.lower().endswith(('.fits', '.fits.gz', '.fit')):
            data = Table.read(filepath, format='fits').to_pandas()
        else:
            try:
                df = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None)
                data = pd.DataFrame({'TIME': df.iloc[:, 0], 'SAP_FLUX': df.iloc[:, 1], 'QUALITY': 0})
            except Exception:
                data = np.loadtxt(filepath)
                data = pd.DataFrame({'TIME': data[:, 0], 'SAP_FLUX': data[:, 1], 'QUALITY': 0})

        data = data[data['QUALITY'] == 0]
        flux_col = flux_col if flux_col in data.columns else 'SAP_FLUX'
        time_col = time_col if time_col in data.columns else 'TIME'

        if flux_col not in data.columns or time_col not in data.columns:
            return [[source, 0, "0.0", "0.0", "", 0, "bad_columns"]]

        flux = data[flux_col].values.astype(float)
        time_full = data[time_col].values.astype(float)

        if source.startswith("http"):
            valid = np.isfinite(flux) & np.isfinite(time_full) & (flux > 0)
        else:
            valid = np.isfinite(flux) & np.isfinite(time_full)
        flux = flux[valid]
        time_full = time_full[valid]

        N = len(flux)
        if N < 500:
            return [[source, 0, "0.0", "0.0", "", N, "too_short"]]

        # Sliding + end-anchored windows
        for target_size in range(1000, max(N//2 + 1, 1001), 1000):
            step = max(1, target_size // 1000)
            start = 0
            while start + target_size <= N:
                f_chunk = flux[start:start + target_size]
                t_chunk = time_full[start:start + target_size]
                f_ds = f_chunk[::step]
                t_ds = t_chunk[::step]
                f_940 = adjust_to_940_points(f_ds)
                t_940 = np.linspace(t_ds[0], t_ds[-1], 940)
                results.append(predict_on_chunk(f_940, t_940,
                    f"win{target_size}_step{step}_seg{start//target_size}", source, compute_stats_flag))
                start += target_size

            # End-anchored
            f_end = flux[-target_size:]
            t_end = time_full[-target_size:]
            f_ds = f_end[::step]
            t_ds = t_end[::step]
            f_940 = adjust_to_940_points(f_ds)
            t_940 = np.linspace(t_ds[0], t_ds[-1], 940)
            results.append(predict_on_chunk(f_940, t_940, f"win{target_size}_end", source, compute_stats_flag))

        # Full downsampled
        step = max(1, N // 1000)
        f_940 = adjust_to_940_points(flux[::step])
        t_940 = np.linspace(time_full[0], time_full[-1], 940)
        results.append(predict_on_chunk(f_940, t_940, f"full_downsampled_step{step}", source, compute_stats_flag))

        # Last 1000 points
        if N >= 1000:
            f_last = flux[-1000:]
            f_940 = adjust_to_940_points(f_last)
            results.append(predict_on_chunk(f_940, time_full[-1000:][[0, -1]], "last1000_fixed", source, compute_stats_flag))

        return results

    except Exception as e:
        return [[source, 0, "0.0", "0.0", f"ERROR: {str(e)}", 0, "exception"]]
    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            try:
                os.unlink(tmp_file.name)
            except:
                pass

# ========================== CLI ENTRY POINT ==========================
def run_prediction(list_file: str, compute_stats_flag: str = "yes", num_cores: int = 8):
    compute_stats_flag = compute_stats_flag.lower()
    if compute_stats_flag not in ["yes", "no"]:
        print("compute_stats_flag must be 'yes' or 'no'")
        return

    num_cores = max(1, int(num_cores))

    tasks = []
    with open(list_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            source, flux_col, time_col = parts[0], parts[1], parts[2]
            tasks.append((source, flux_col, time_col, compute_stats_flag))

    if not tasks:
        print("No light curves found in list file.")
        return

    print(f"Loaded {len(tasks)} light curves | stats={compute_stats_flag} | threads={num_cores}")
    print("Predicting...\n")

    with open("prediction_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Class", "Probability", "Real_4FWHM_days", "Latent_Space", "Points", "Chunk_Description"])

        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            for results in tqdm(executor.map(process_source, tasks),
                                total=len(tasks), desc="Predicting", unit="source", colour="cyan"):
                for row in results:
                    writer.writerow(row)

    print("\nAll done! → prediction_results.csv")
