import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def make_spectrogram(audio_file, image_filepath):
    """
    Generate a MEL spectrogram from a WAV file and save it as a PNG image.

    Parameters:
    - audio_file (str): The path to the input WAV file.
    - image_filepath (str): The desired path to save the generated MEL spectrogram image.

    Returns:
    None
    """
    # Load audio file
    y, sr = librosa.load(audio_file, sr=22050)

    # Create mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Convert power spectrogram to dB scale (log scale)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot mel-spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.title('Mel-Spectrogram')
    plt.tight_layout()

    # Save as PNG file
    plt.savefig(image_filepath)
