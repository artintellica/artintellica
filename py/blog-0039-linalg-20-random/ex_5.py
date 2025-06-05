import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for signal generation
N = 1024  # Number of points in the signal
t = np.linspace(0, 1, N)  # Time vector from 0 to 1 second
fs = N  # Sampling frequency (since time range is 1 second, fs = N Hz)

# Generate a 1D signal: sum of two sine waves with noise
f1 = 50  # Frequency of first sine wave (50 Hz)
f2 = 120  # Frequency of second sine wave (120 Hz)
signal = 1.0 * np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
noise = 0.2 * np.random.randn(N)  # Add Gaussian noise with amplitude 0.2
signal_with_noise = signal + noise
print("Signal Length:", len(signal_with_noise))

# Apply Fast Fourier Transform (FFT) to transform to frequency domain
fft_result = np.fft.fft(signal_with_noise)
frequencies = np.fft.fftfreq(N, 1 / fs)  # Frequency axis in Hz

# Compute the magnitude spectrum (absolute value of FFT result)
magnitude_spectrum = np.abs(fft_result)
# Only plot the positive frequencies (up to Nyquist frequency, fs/2)
positive_freq_idx = frequencies > 0
frequencies_positive = frequencies[positive_freq_idx]
magnitude_positive = magnitude_spectrum[positive_freq_idx]

# Plotting
plt.figure(figsize=(12, 8))

# Plot original signal in time domain
plt.subplot(2, 1, 1)
plt.plot(t, signal_with_noise, "b-", label="Signal with Noise")
plt.title("Original Signal (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Plot frequency spectrum
plt.subplot(2, 1, 2)
plt.plot(frequencies_positive, magnitude_positive, "r-", label="Magnitude Spectrum")
plt.title("Frequency Spectrum (Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.legend()
# Limit x-axis to a reasonable range to focus on the main frequencies
plt.xlim(0, 200)

plt.tight_layout()
plt.show()
