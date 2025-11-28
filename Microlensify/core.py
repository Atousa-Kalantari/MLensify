#Microlensify/core.py
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
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    "MLensify_Model.keras",
    "scaler_4fwhm.pkl",
    "scaler_std_div_diff.pkl",
    "scaler_max_flux.pkl",
    "scaler_min_flux.pkl",
    "scaler_median_flux.pkl",
    "scaler_std_flux.pkl",
]

MODEL_PATH = ASSETS_DIR / "MLensify_Model.keras"

def ensure_assets():
    """Download model and scalers with live progress bar"""
    import sys
    from urllib.request import urlopen

    def download_with_progress(url, dest_path):
        if dest_path.exists():
            return
        print(f"   • Downloading {dest_path.name} ... ", end="", flush=True)
        with urlopen(url) as response, open(dest_path, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB chunks
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
                    print(f"\r   • Downloading {dest_path.name} ... {mb_done:.1f} / {mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)
                else:
                    print(f"\r   • Downloading {dest_path.name} ... {downloaded//(1024*1024)} MB", end="", flush=True)
        print(" done")

    missing = False
    for fname in FILES:
        fpath = ASSETS_DIR / fname
        if not fpath.exists():
            if not missing:
                print("First run detected — downloading model and scalers from GitHub Release (one-time only)...")
                missing = True
            url = f"{BASE_URL}/{fname}"
            download_with_progress(url, fpath)
    if missing:
        print("All files downloaded! Future runs will be instant and work offline.\n")

# ========================== SEEDS & PATHS ==========================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)
TESS_SECTOR_DAYS = 27.4

# ========================== GLOBALS ==========================
encoder = None
decoder = None
scaler_fwhm = scaler_std_div_diff = scaler_max = scaler_min = scaler_median = scaler_std = None

def init_worker():
    global encoder, decoder
    global scaler_fwhm, scaler_std_div_diff, scaler_max, scaler_min, scaler_median, scaler_std

    if encoder is not None:
        return  

    print("Loading Microlensify model and scalers...")
    ensure_assets()  

    cvae = tf.keras.models.load_model(MODEL_PATH, custom_objects={'Sampling': Sampling, 'CVAE': CVAE})
    encoder, decoder = cvae.encoder, cvae.decoder

    scaler_fwhm = joblib.load(ASSETS_DIR / "scaler_4fwhm.pkl")
    scaler_std_div_diff = joblib.load(ASSETS_DIR / "scaler_std_div_diff.pkl")
    scaler_max = joblib.load(ASSETS_DIR / "scaler_max_flux.pkl")
    scaler_min = joblib.load(ASSETS_DIR / "scaler_min_flux.pkl")
    scaler_median = joblib.load(ASSETS_DIR / "scaler_median_flux.pkl")
    scaler_std = joblib.load(ASSETS_DIR / "scaler_std_flux.pkl")

    print("Model and scalers loaded successfully!\n")


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

def predict_on_chunk(chunk_flux, chunk_time, description, source, compute_stats_flag):
    if compute_stats_flag == "yes":
        fmax = np.max(chunk_flux)
        fmin = np.min(chunk_flux)
        fdiff = fmax - fmin + 1e-12
        fmed = np.median(chunk_flux)
        fstd = np.std(chunk_flux)
    else:
        fmax = 396.817703
        fmin = 189.911636
        fdiff = fmax - fmin
        fmed = 310.554672
        fstd = 7.381589

    norm_max = scaler_max.transform([[safe_log10(fmax)]])[0,0]
    norm_min = scaler_min.transform([[safe_log10(fmin)]])[0,0]
    norm_std = scaler_std.transform([[safe_log10(fstd)]])[0,0]
    norm_median = scaler_median.transform([[safe_log10(fmed)]])[0,0]
    norm_std_div_diff = scaler_std_div_diff.transform([[safe_log10(fstd / fdiff)]])[0,0]

    scalar_test = np.array([norm_max, norm_min, norm_std, norm_median, norm_std_div_diff]).reshape(1, 5)

    normflux = (chunk_flux - np.min(chunk_flux)) / (np.max(chunk_flux) - np.min(chunk_flux))
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

# ========================== process_source  =============================
def process_source(source, flux_col, time_col, compute_stats_flag):
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
            except:
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

        target_sizes = list(range(1000, N + 1, 1000))
        for target_size in target_sizes:
            step = max(1, target_size // 1000)
            start = 0
            while start + target_size <= N:
                t_chunk = time_full[start:start + target_size]
                f_chunk = flux[start:start + target_size]
                f_ds = f_chunk[::step] if step > 1 else f_chunk
                t_ds = t_chunk[::step] if step > 1 else t_chunk
                f_940 = adjust_to_940_points(f_ds)
                t_940 = np.linspace(t_ds[0], t_ds[-1], 940)
                desc = f"win{target_size}_step{step}_seg{start//target_size}"
                results.append(predict_on_chunk(f_940, t_940, desc, source, compute_stats_flag))
                start += target_size

            # end window
            t_end = time_full[-target_size:]
            f_end = flux[-target_size:]
            f_ds = f_end[::step] if step > 1 else f_end
            t_ds = t_end[::step] if step > 1 else t_end
            f_940 = adjust_to_940_points(f_ds)
            t_940 = np.linspace(t_ds[0], t_ds[-1], 940)
            results.append(predict_on_chunk(f_940, t_940, f"win{target_size}_end", source, compute_stats_flag))

        # full light curve downsampled
        full_step = max(1, N // 1000)
        f_ds = flux[::full_step]
        t_ds = time_full[::full_step]
        f_940 = adjust_to_940_points(f_ds)
        t_940 = np.linspace(t_ds[0], t_ds[-1], 940)
        results.append(predict_on_chunk(f_940, t_940, f"full_downsampled_step{full_step}", source, compute_stats_flag))

        return results

    except Exception as e:
        return [[source, 0, "0.0", "0.0", f"ERROR: {str(e)}", 0, "exception"]]
    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)

# ========================== BATCH & MAIN ==========================
def process_batch(batch):
    results = []
    for args in batch:
        results.extend(process_source(*args))
    return results

def run_prediction(list_file: str, compute_stats_flag: str, num_cores: int):
    compute_stats_flag = compute_stats_flag.lower()
    if compute_stats_flag not in ["yes", "no"]:
        print("Second argument must be 'yes' or 'no'")
        sys.exit(1)

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

    print(f"Loaded {len(tasks)} light curves | compute_stats = {compute_stats_flag} | using {num_cores} cores")

    output_file = "prediction_results.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Class", "Probability", "Real_4FWHM_days", "Latent_Space", "Points", "Chunk_Description"])

        with ProcessPoolExecutor(max_workers=num_cores, initializer=init_worker) as executor:
            batches = [tasks[i:i+4] for i in range(0, len(tasks), 4)]
            futures = [executor.submit(process_batch, b) for b in batches]
            completed = 0
            for future in as_completed(futures):
                for row in future.result():
                    writer.writerow(row)
                completed += 1
                print(f"Progress: {completed}/{len(batches)} batches completed")

    print(f"\nAll done! Results saved to → {output_file}")
