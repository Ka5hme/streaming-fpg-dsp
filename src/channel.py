import numpy as np
import matplotlib.pyplot as plt

from constants import c
from target import Target

def simulate_target_return(
        tx: np.ndarray,
        target: Target,
        sample_rate_Hz: float,
) -> np.ndarray:

    if sample_rate_Hz < 0:
        raise ValueError("Sample rate must be greater than 0")

    target_delay_seconds  = (2 * target.range_m / c)
    delay_samples = round(target_delay_seconds * sample_rate_Hz)
    target_amplitude = np.sqrt(target.rcs) / target.range_m**2

    rx = np.zeros(len(tx) + delay_samples, dtype=complex)
    rx[delay_samples:delay_samples+len(tx)] = target_amplitude * tx

    print("Target range:", target.range_m, "m")
    print("Propagation delay:", target_delay_seconds, "s")
    print("Delay samples:", delay_samples)
    print("Target amplitude:", target_amplitude)

    return rx

