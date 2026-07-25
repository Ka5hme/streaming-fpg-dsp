import numpy as np

def gaussian_noise(
        signal: np.ndarray,
        snr: float,
        rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:

    if signal.size == 0:
        raise ValueError("No signal provided")

    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(np.abs(signal) **2)

    if signal_power == 0:
        raise ValueError("No signal provided")

    snr_linear = 10 ** (snr / 10)

    noise_power = signal_power / snr_linear

    noise_std = np.sqrt(noise_power / 2)

    noise = noise_std * (
        rng.standard_normal(signal.shape)
        + 1j * rng.standard_normal(signal.shape)
    )

    n_signal = signal + noise

    return n_signal, noise