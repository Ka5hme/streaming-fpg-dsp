import numpy as np
import matplotlib.pyplot as plt

from signal_gen import generate_lfm
from target import Target
from channel import simulate_target_return
from noise_gen import gaussian_noise

fs = 10_000_000 # samples freq in samples per sec | 0.001 secs
T = 100e-6      # signal duration in seconds
f0 = 100_000    # start freq in Hz
f1 = 1_000_000  # ending freq in Hz
snr = 10.0      # signal to noise ratio in db
rng = np.random.default_rng(seed=42)

tx_time, tx = generate_lfm(sample_rate = fs, duration = T, start_freq = f0, stop_freq = f1)

# Call target function
target = Target(range_m = 200, rcs = 1.0, velocity = 0.0)

# Signal rx pre noise
rx_pre = simulate_target_return(tx = tx, target = target, sample_rate_Hz = fs)

# Signal rx with noise added
rx, noise = gaussian_noise(signal = rx_pre, snr = snr, rng = rng)

measured_signal_power = np.mean(np.abs(rx_pre) ** 2)
measured_noise_power = np.mean(np.abs(noise) ** 2)

measured_snr_linear = (
    measured_signal_power / measured_noise_power
)

measured_snr = 10 * np.log10(measured_snr_linear)

tx_i_signal = np.real(tx) # I: real component of tx
tx_q_signal = np.imag(tx) # Q: imaginary component of tx

rx_i_signal = np.real(rx) # I: real component of tx
rx_q_signal = np.imag(rx) # Q: imaginary component of tx

rx_time = np.arange(len(rx)) / fs

# Print basic information
print("Transmit samples:", len(tx))
print("Receive samples:", len(rx))
print("Inputted SNR:", snr, "dB")
print("Actual SNR:", measured_snr, "dB")