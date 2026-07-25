import numpy as np

def generate_lfm(
        sample_rate: float,
        duration: float,
        start_freq: float,
        stop_freq: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_rate <= 0:
        raise ValueError("Sample rate must be greater than 0")

    if duration <= 0:
        raise ValueError("Duration must be greater than 0")

    dt = 1 / sample_rate
    time_s = np.arange(0, duration, dt)

    chirp_rate = (stop_freq - start_freq ) / duration

    phase_rad = 2 * np.pi * (start_freq * time_s + 0.5 * chirp_rate *time_s**2)
    tx = np.exp(1j * phase_rad)

    return time_s, tx


