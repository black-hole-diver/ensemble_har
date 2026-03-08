import numpy as np

def extract_physics(window: np.ndarray) -> np.ndarray:
    """
    Extracts 72 physics features from a single 2-second sensor window.
    Expected window shape: (N_samples, 9) -> Accel(x,y,z), Gyro(x,y,z), Mag(x,y,z).
    Time domain, magnitudes, jerk, frequency domain (FFT).
    """
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    max_v = np.max(window, axis=0)
    min_v = np.min(window, axis=0)

    mag_u = np.sqrt(np.sum(window[:, 0:3]**2, axis=1))
    mag_g = np.sqrt(np.sum(window[:, 3:6]**2, axis=1))
    mag_v = np.sqrt(np.sum(window[:, 6:9]**2, axis=1))
    mag_stats = [
        np.mean(mag_u), np.std(mag_u), np.max(mag_u),
        np.mean(mag_g), np.std(mag_g), np.max(mag_g),
        np.mean(mag_v), np.std(mag_v), np.max(mag_v)
    ]

    jerk = np.diff(window[:, 0:3], axis=0)
    j_stats = np.concatenate([np.mean(jerk, axis=0), np.std(jerk, axis=0), np.max(jerk, axis=0)])

    fft_v = np.abs(np.fft.rfft(window, axis=0))
    f_stats = np.concatenate([np.mean(fft_v, axis=0), np.max(fft_v, axis=0)])

    return np.concatenate([mean, std, max_v, min_v, mag_stats, j_stats, f_stats])

def extract_batch_features(windows) -> np.ndarray:
    """
    Processes a list or array of windows and returns a 2D feature matrix.
    """
    return np.array([extract_physics(w) for w in windows])