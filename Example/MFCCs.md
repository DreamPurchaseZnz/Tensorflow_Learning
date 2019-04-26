import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import numpy as np


# Setup
sample_rate, signal = scipy.io.wavfile.read('102632.syn.wav')
signal = signal[0:int(3.5 * sample_rate)]

fig = plt.figure(figsize=(10, 3), dpi=300, facecolor='white')
ax = fig.add_subplot(111)
ax.plot(signal, '-b', ms=0.01, linewidth=0.1)
ax.legend(["waveform"], fontsize=10)
ax.set_ylabel('Amplitude',fontsize=9)
ax.set_xlabel('Time(s)', fontsize=9)
plt.tight_layout()
plt.savefig("signal.png")
plt.close()


# Pre-Emphasis
pre_emphasis = 0.97
emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

fig = plt.figure(figsize=(10, 3), dpi=300, facecolor='white')
ax = fig.add_subplot(111)
ax.plot(emphasized_signal, '-b', ms=0.01, linewidth=0.1)
ax.legend(["waveform"], fontsize=10)
ax.set_ylabel('Amplitude',fontsize=9)
ax.set_xlabel('Time(s)', fontsize=9)
plt.tight_layout()
plt.savefig("emphasized-signal.png")
plt.close()

# framing
frame_size = 0.025
frame_stride = 0.01
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = numpy.zeros((pad_signal_length - signal_length))
pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + \
          numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(numpy.int32, copy=False)]

# Window
N = 200
f = []
for i in range(N):
    f.append(0.54-0.46 * np.cos(2*np.pi*i/(N-1)))
f = np.array(f)

fig = plt.figure(figsize=(4, 4), dpi=300, facecolor='white')
ax = fig.add_subplot(111)
ax.plot(np.arange(N), f, '-b', ms=0.01, linewidth=1)
ax.legend(["Windowing"], fontsize=10)
ax.set_ylabel('Intensity',fontsize=9)
ax.set_xlabel('Time(s)', fontsize=9)
plt.tight_layout()
plt.savefig("windowing.png")
plt.close()


frames *= numpy.hamming(frame_length)

fig = plt.figure(figsize=(4, 4), dpi=300, facecolor='white')
ax = fig.add_subplot(111)
ax.plot(frames[1], '-b', ms=0.01, linewidth=1)
ax.legend(["Windowing"], fontsize=10)
ax.set_ylabel('Amplitude',fontsize=9)
ax.set_xlabel('Time(s)', fontsize=9)
plt.tight_layout()
plt.savefig("windowing_1.png")
plt.close()
print(np.shape(frames))

# Fourier-Transform and Power Spectrum
NFFT = 512
mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
print(np.shape(mag_frames))

fig = plt.figure(figsize=(4, 4), dpi=300, facecolor='white')
ax = fig.add_subplot(111)
ax.plot(mag_frames[1], '-b', ms=0.01, linewidth=1)
ax.legend(["Windowing"], fontsize=10)
ax.set_ylabel('magnitude',fontsize=9)
ax.set_xlabel('Time(s)', fontsize=9)
plt.tight_layout()
plt.savefig("Window_magnitude.png")
plt.close()


pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
print(np.shape(pow_frames))

fig = plt.figure(figsize=(4, 4), dpi=300, facecolor='white')
ax = fig.add_subplot(111)
ax.plot(pow_frames[1], '-b', ms=0.01, linewidth=1)
ax.legend(["Windowing"], fontsize=10)
ax.set_ylabel('Power spectrum',fontsize=9)
ax.set_xlabel('Time(s)', fontsize=9)
plt.tight_layout()
plt.savefig("Window_power.png")
plt.close()


# Filter Banks

N = int(2e4)
m = []
for f in range(N):
    m.append(2595*np.log10(1+f/700))
m = np.array(m)

fig = plt.figure(figsize=(4, 4), dpi=300, facecolor='white')
ax = fig.add_subplot(111)
ax.plot(np.arange(N), m, '-b', ms=0.01, linewidth=1)
ax.legend(["Frequency-Mel"], fontsize=10)
ax.set_ylabel('Mel', fontsize=9)
ax.set_xlabel('Frequency', fontsize=9)
plt.tight_layout()
plt.savefig("Frequency-mel.png")
plt.close()


nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
print(np.shape(hz_points))
bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)
print(np.shape((NFFT + 1) * hz_points))

fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])


fig = plt.figure(figsize=(4, 4), dpi=300, facecolor='white')
ax = fig.add_subplot(111)
ax.plot(fbank[2], '-b', ms=0.01, linewidth=1)
ax.legend(["Filterbank"], fontsize=10)
ax.set_ylabel('Amplitude', fontsize=9)
ax.set_xlabel('Frequency', fontsize=9)
plt.tight_layout()
plt.savefig("Frequency-Amplitude-F.png")
plt.close()


filter_banks = numpy.dot(pow_frames, fbank.T)
filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * numpy.log10(filter_banks)  # dB

fig = plt.figure(figsize=(4, 4), dpi=300, facecolor='white')
ax = fig.add_subplot(111)
ax.plot(filter_banks[1], '-b', ms=0.01, linewidth=1)
ax.legend(["Filter banks"], fontsize=10)
ax.set_ylabel('Mel', fontsize=9)
ax.set_xlabel('Times', fontsize=9)
plt.tight_layout()
plt.savefig("Filter_banks.png")
plt.close()

print(np.shape(filter_banks))
print(np.shape(pow_frames))

fig = plt.figure(figsize=(8, 2), dpi=300, facecolor='white')
ax = fig.add_subplot(111)
ax.imshow(filter_banks.T, cmap="jet")
ax.set_ylabel('Frequency(KHZ)', fontsize=9)
ax.set_xlabel('Times', fontsize=9)
plt.tight_layout()
plt.savefig("Spectrogram.png")
plt.close()


# Mel-frequency Cepstral Coefficients (MFCCs)
num_ceps = 12
cep_lifter = 20 # ?

mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

(nframes, ncoeff) = mfcc.shape
n = numpy.arange(ncoeff)
lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
mfcc *= lift  #*

fig = plt.figure(figsize=(8, 1), dpi=300, facecolor='white')
ax = fig.add_subplot(111)
ax.imshow(mfcc.T, cmap="jet")
ax.set_ylabel('MFCCs', fontsize=9)
ax.set_xlabel('Times', fontsize=9)
plt.tight_layout()
plt.savefig("MFCCs.png")
plt.close()


# Mean Normalization
filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
np.shape(filter_banks)

fig = plt.figure(figsize=(8, 2), dpi=300, facecolor='white')
ax = fig.add_subplot(111)
ax.imshow(filter_banks.T, cmap="jet")
ax.set_ylabel('MFCCs', fontsize=9)
ax.set_xlabel('Times', fontsize=9)
plt.tight_layout()
plt.savefig("Spectrogram_mean.png")
plt.close()










