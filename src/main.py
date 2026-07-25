import numpy as np
import matplotlib.pyplot as plt

from constants import c
from signal_gen import generate_lfm
from target import Target
from channel import simulate_target_return
from noise_gen import gaussian_noise
from fft_process import matched_filter_fft

fs = 50_000_000 # samples freq in samples per sec | 0.001 secs
T = 100e-6      # signal duration in seconds
f0 = 1_000_000    # start freq in Hz
f1 = 11_000_000  # ending freq in Hz
snr = 100.0      # signal to noise ratio in db
rng = np.random.default_rng(seed=42)

tx_time, tx = generate_lfm(sample_rate = fs, duration = T, start_freq = f0, stop_freq = f1)

# Call target function
target = Target(range_m = 50000, rcs = 1.0, velocity = 0.0)

# Signal rx pre noise
rx_pre = simulate_target_return(tx = tx, target = target, sample_rate_Hz = fs)

# Signal rx with noise added
rx, noise = gaussian_noise(signal = rx_pre, snr = snr, rng = rng)

rx_time = np.arange(len(rx)) / fs

signal_power = np.mean(np.abs(rx_pre) ** 2)
noise_power = np.mean(np.abs(noise) ** 2)

measured_snr = 10 * np.log10(signal_power / noise_power)

(
    compressed_signal,
    compressed_magnitude,
    estimated_delay_samples,
    estimated_range_m
) = matched_filter_fft(
    rx=rx,
    tx=tx,
    sample_rate_Hz=fs
)

correlation_indices = (
    np.arange(len(compressed_magnitude))
    - (len(tx) - 1)
)

range_axis_m = (
    correlation_indices
    * c
    / (2 * fs)
)

valid_ranges = range_axis_m >= 0

normalized_magnitude = (
    compressed_magnitude
    / np.max(compressed_magnitude)
)

magnitude_db = 20 * np.log10(
    normalized_magnitude + 1e-12
)

tx_i_signal = np.real(tx) # I: real component of tx
tx_q_signal = np.imag(tx) # Q: imaginary component of tx

rx_i_signal = np.real(rx) # I: real component of tx
rx_q_signal = np.imag(rx) # Q: imaginary component of tx

# Print basic information
print("Transmit samples:", len(tx))
print("Receive samples:", len(rx))
print("Inputted SNR:", snr, "dB")
print("Actual SNR:", measured_snr, "dB")

# =====================================================
# Create range axis
# =====================================================

# The matched-filter output uses full convolution.
# Subtract len(tx) - 1 to convert output indices into
# physical delay samples.
lag_samples = (
    np.arange(len(compressed_signal))
    - (len(tx) - 1)
)

# Convert round-trip delay samples into one-way range.
range_axis_m = (
    lag_samples
    * c
    / (2 * fs)
)

# Negative lags do not represent valid target ranges.
valid_ranges = range_axis_m >= 0

valid_range_axis_m = range_axis_m[valid_ranges]
valid_compressed_signal = compressed_signal[valid_ranges]


# =====================================================
# Convert matched-filter output to power in dB
# =====================================================

range_profile_power = (
    np.abs(valid_compressed_signal) ** 2
)

max_power = np.max(range_profile_power)

# A very small value prevents log10(0).
range_profile_db = 10 * np.log10(
    range_profile_power / max_power + 1e-12
)


# =====================================================
# Print results
# =====================================================

print()
print("========== Simulation Results ==========")
print(f"Transmit samples       : {len(tx)}")
print(f"Receive samples        : {len(rx)}")
print(f"Requested input SNR    : {snr:.2f} dB")
print(f"Measured input SNR     : {measured_snr:.2f} dB")
print(f"Estimated delay        : {estimated_delay_samples} samples")
print(f"True target range      : {target.range_m:.2f} m")
print(f"Estimated target range : {estimated_range_m:.2f} m")
print("========================================")


# =====================================================
# Plot transmitted I/Q chirp
# =====================================================

plt.figure(figsize=(10, 5))

plt.plot(
    tx_time * 1e6,
    tx_i_signal,
    label="Transmit I"
)

plt.plot(
    tx_time * 1e6,
    tx_q_signal,
    label="Transmit Q",
    alpha=0.7
)

plt.xlabel("Time (microseconds)")
plt.ylabel("Amplitude")
plt.title("Transmit LFM I/Q Signal")
plt.grid(True)
plt.legend()

plt.show()


# =====================================================
# Plot noisy received I/Q signal
# =====================================================

plt.figure(figsize=(10, 5))

plt.plot(
    rx_time * 1e6,
    rx_i_signal,
    label="Received I",
    linewidth=0.8
)

plt.plot(
    rx_time * 1e6,
    rx_q_signal,
    label="Received Q",
    linewidth=0.8,
    alpha=0.7
)

plt.xlabel("Time (microseconds)")
plt.ylabel("Amplitude")
plt.title(f"Noisy Received I/Q Signal — {snr:.1f} dB Input SNR")
plt.grid(True)
plt.legend()

plt.show()


# =====================================================
# Plot radar range profile
# =====================================================

plt.figure(figsize=(11, 6))

plt.plot(
    valid_range_axis_m,
    range_profile_db,
    linewidth=0.6,
    label="Range profile"
)

# Mark the strongest detected target.
detected_index = int(
    np.argmax(range_profile_db)
)

detected_range_m = (
    valid_range_axis_m[detected_index]
)

detected_power_db = (
    range_profile_db[detected_index]
)

plt.plot(
    detected_range_m,
    detected_power_db,
    marker="o",
    markersize=8,
    markerfacecolor="none",
    markeredgewidth=2,
    label=f"Detected target: {detected_range_m:.1f} m"
)

plt.xlabel("Range (m)")
plt.ylabel("Normalized Power (dB)")
plt.title("Matched-Filter Radar Range Profile")
plt.xlim(0, 60_000)
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()
