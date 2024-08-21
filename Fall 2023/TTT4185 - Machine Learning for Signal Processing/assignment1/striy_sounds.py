import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def striy_sounds():

    duration = 0.5  # Duration in seconds
    fs = 44100  # Sample rate in Hz

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    s_sound = 0.5 * np.sin(2 * np.pi * 5000 * t)

    t_sound = np.zeros_like(t)
    t_sound[int(0.1 * fs)] = 1.0
    t_sound[int(0.15 * fs)] = -1.0

    r_sound = 0.5 * np.sin(2 * np.pi * 400 * t)

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    f0 = 300
    iy_sound = np.sin(2 * np.pi * f0 * t)

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(s_sound), ref=np.max), y_axis='log', x_axis='time')
    # plt.title('/s/ Sound Spectrogram')
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()

    plt.figure(figsize=(10, max([len(s_sound), len(t_sound), len(r_sound), len(iy_sound)]) / fs * 50))

    plt.subplot(4, 1, 1)
    plt.plot(t, s_sound, color='g')
    plt.title('/s/ Sound Amplitude')

    plt.subplot(4, 1, 2)
    plt.plot(t, t_sound, color='g')
    plt.title('/t/ Sound Amplitude')

    plt.subplot(4, 1, 3)
    plt.plot(t, r_sound, color='g')
    plt.title('/r/ Sound Amplitude')

    plt.subplot(4, 1, 4)
    plt.plot(t, iy_sound, color='g')
    plt.title('/iy/ Sound Amplitude')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(s_sound), ref=np.max), y_axis='log', x_axis='time')
    plt.title('/s/ Sound Spectrogram')

    plt.subplot(2, 2, 2)
    plt.plot(t, t_sound, color='g')
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(t_sound), ref=np.max), y_axis='log', x_axis='time')
    plt.title('/t/ Sound Spectrogram')

    plt.subplot(2, 2, 3)
    plt.plot(t, r_sound, color='g')
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(r_sound), ref=np.max), y_axis='log', x_axis='time')
    plt.title('/r/ Sound Spectrogram')

    plt.subplot(2, 2, 4)
    plt.plot(t, iy_sound, color='g')
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(iy_sound), ref=np.max), y_axis='log', x_axis='time')
    plt.title('/iy/ Sound Spectrogram')

    plt.tight_layout()
    plt.show()