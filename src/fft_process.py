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

    output_length = len(rx) + len(tx) - 1

    # reverses waveform and takes complex conjugate
    matched_filter = np.conj(tx[::-1])

    rx_fft = np.fft.fft(rx, n = output_length)

    # rx signal into freq domain
    matched_filter_fft_values = np.fft.fft(matched_filter, n = output_length)

    # Multiplication in the frequency domain is equivalent to convolution in the time domain
    compressed_fft =(rx_fft * matched_filter_fft_values)

    # post pulse compression back to time domain
    compressed_signal = np.fft.ifft(compressed_fft)
    compressed_magnitude = np.abs(compressed_signal)

    peak_index = int(np.argmax(compressed_magnitude))
    estimated_delay_samples = (peak_index - (len(tx) - 1))
    estimated_delay_seconds = (estimated_delay_samples / sample_rate_Hz)
    estimated_range = (estimated_delay_seconds * c / 2)

    return compressed_signal, compressed_magnitude, estimated_delay_samples, estimated_range


