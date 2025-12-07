# Microlensify/core.py
import os
import csv
import sys
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

# ========================== AUTO-DOWNLOAD FROM GITHUB RELEASE ==========================
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
    """Download model and scalers with live progress bar"""
    from urllib.request import urlopen
    def download_with_progress(url, dest_path):
        if dest_path.exists():
            return
        print(f" • Downloading {dest_path.name} ... ", end="", flush=True)
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
                    mb_done = downloaded / (1024*1024)
                    mb_total = total_size / (1024*1024)
                    print(f"\r • Downloading {dest_path.name} ... {mb_done:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)
                else:
                    print(f"\r • Downloading {dest_path.name} ... {downloaded//(1024*1024)} MB", end="", flush=True)
        print(" done")

    missing = False
    for fname in FILES:
        fpath = ASSETS_DIR / fname
        if not fpath.exists():
            if not missing:
                print("First run detected — downloading model and scalers from GitHub Release...")
                missing = True
            url = f"{BASE_URL}/{fname}"
            download_with_progress(url, fpath)
    if missing:
        print("All assets downloaded! Future runs will be instant and offline.\n")

# ========================== SEEDS & CONSTANTS ==========================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)
TESS_SECTOR_DAYS = 27.4

# Fixed training statistics (used when compute_stats="no")
FIXED_STATS = {
    "max": 2379.638043809108,
    "min": 1470.406889942004,
    "median": 1663.0377499789672,
    "std": 89.51542107215877,
}

# ========================== LOAD MODEL & SCALERS GLOBALLY ==========================
print("Loading Microlensify model and scalers (this may take 10–30 seconds)...")
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

print("Model and scalers loaded successfully!\n")

# ========================== HELPERS ==========================
def adjust_to_940_points(arr):
    arr = np.array(arr, dtype=float)
    n = len(arr)
    if n == 940:
        return arr.copy()
    elif n > 940:
        n_remove = n - 940
        drop_idx = set(rn.sample(range(n), n_remove))
        return np.array([arr[i] for i in range(n) if i not in drop_idx])
    else:
        n_pad = 940 - n
        min_val = np.min(arr)
        rng = np.random.default_rng(SEED)
        noise = 0.001 * min_val * rng.standard_normal(n_pad)
        padding = min_val + noise
        return np.concatenate([arr, padding])

def safe_log10(x):
    return np.log10(np.clip(x, 1e-10, None))

# ========================== CORE PREDICTION ==========================
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
    decoder.predict(z_sampled, verbose=0)  # Keep graph warm

    prob = float(class_pred[0][0])
    y_pred = int(prob > 0.99)
    pred_norm_fwhm = z_mean[0, -2]
    pred_fwhm_model_units = 10 ** scaler_fwhm.inverse_transform([[pred_norm_fwhm]])[0,0]
    time
