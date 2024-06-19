from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import scipy

fs_rate, signal = wavfile.read("resources/440hz.wav")
print("Frequency sampling: ", fs_rate)


channels = len(signal.shape)
print("Channels: ", channels)

if channels == 2:
    signal = signal.sum(axis=1) / 2

N = signal.shape[0]
print("Complete Samplings N: ", N)

secs = N / float(fs_rate)
print("secs: ", secs)

Ts = 1.0 / fs_rate  # sampling interval in time
print("Timestep between samples Ts: ", Ts)

t = np.arange(0, secs, Ts)  # time vector as scipy arange field / numpy.ndarray
FFT = abs(scipy.fft.fft(signal))
FFT_side = FFT[range(int(N / 2))]  # one side FFT range

freqs = scipy.fftpack.fftfreq(signal.size, t[1] - t[0])
fft_freqs = np.array(freqs)
freqs_side = freqs[range(int(N / 2))]  # one side frequency range
fft_freqs_side = np.array(freqs_side)

idx = np.argmax(np.abs(FFT))
freq = freqs_side[idx]
print(freq)

FFT_side = FFT_side[0:20000]
freqs_side = freqs_side[0:20000]

p1 = plt.plot(t, signal, "g") # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

p3 = plt.plot(freqs_side, abs(FFT_side), "b")  # plotting the positive fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count single-sided')
plt.show()


