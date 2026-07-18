
import sys

import librosa
import numpy as np
import soundfile as sf
import tritonclient.http as httpclient

TARGET_SR = 16000


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    this is for resampling audio to the target sample rate (default 16kHz) using librosa.
    """
    audio = audio.astype(np.float32)
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def transcribe_array(
    audio: np.ndarray,
    model_name: str = "mbaza_asr",
    url: str = "localhost:8000",
) -> str:

    client = httpclient.InferenceServerClient(url=url)
    audio = audio.astype(np.float32)

    inputs = [httpclient.InferInput("AUDIO", audio.shape, "FP32")]
    inputs[0].set_data_from_numpy(audio)

    outputs = [httpclient.InferRequestedOutput("TRANSCRIPTION")]

    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    result = response.as_numpy("TRANSCRIPTION")
    text = result[0].decode("utf-8")

    print(text)
    return text


def transcribe(
    audio_path: str,
    model_name: str = "mbaza_asr",
    url: str = "localhost:8000",
) -> str:
    """Read a WAV/FLAC/etc file from disk, resample if needed, and send it."""
    audio, sr = sf.read(audio_path, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono

    audio = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SR)

    return transcribe_array(audio, model_name=model_name, url=url)


if __name__ == "__main__":
    # if len(sys.argv) not in (2, 3):
    #     print("Usage: python client.py path/to/audio.wav [model_name]")
    #     sys.exit(1)

    audio_path = sys.argv[1]
    # model_name = sys.argv[2] if len(sys.argv) == 3 else "mbaza_asr_nemo"
    transcribe(audio_path)