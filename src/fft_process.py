import numpy as np

from constants import c
from target import Target

def matched_filter_fft(
        rx:np.ndarray,
        tx:np.ndarray,
        sample_rate_Hz:float,
) -> tuple[np.ndarray, np.ndarray, int, float]:

    if rx.size == 0:
        raise ValueError("Receive signal cannot be empty.")

    if tx.size == 0:
        raise ValueError("Transmit signal cannot be empty.")

    if sample_rate_Hz <= 0:
        raise ValueError("Sample rate must be greater than 0.")

    output_length = len(rx) + len(tx)

    matched_filter = np.conj(tx[::-1])


