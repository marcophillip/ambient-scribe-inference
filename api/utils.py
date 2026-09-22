
import sys

import librosa
import numpy as np
import soundfile as sf
import tritonclient.http as httpclient
import io
from pydub import AudioSegment
import numpy as np

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
    model_name: str,
    url: str,
    language: str | None = None,
) -> tuple[str, str]:
    """Send a waveform to Triton and return (transcription, detected_language).

    `language` is only meaningful for the prompt-conditioned nemotron model:
    a locale like "en-US"/"fr-FR", or "auto" to let it detect and tag.
    detected_language comes back empty for models that do not report one.
    """

    client = httpclient.InferenceServerClient(url=url)
    audio = audio.astype(np.float32)

    audio_input = httpclient.InferInput("AUDIO", audio.shape, "FP32")
    audio_input.set_data_from_numpy(audio)
    inputs = [audio_input]

    if language:
        lang_array = np.array([language], dtype=object)
        lang_input = httpclient.InferInput("LANGUAGE", [1], "BYTES")
        lang_input.set_data_from_numpy(lang_array)
        inputs.append(lang_input)

    # LANGUAGE_DETECTED only exists on the nemotron model -- ask for it only
    # when we also sent a language prompt, so mbaza_asr keeps working.
    output_names = ["TRANSCRIPTION"] + (["LANGUAGE_DETECTED"] if language else [])
    outputs = [httpclient.InferRequestedOutput(name) for name in output_names]

    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    text = response.as_numpy("TRANSCRIPTION")[0].decode("utf-8")

    detected = ""
    if language:
        result = response.as_numpy("LANGUAGE_DETECTED")
        if result is not None:
            detected = result[0].decode("utf-8")

    return text, detected


def blob_bytes_to_array(blob_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode an arbitrary-format audio blob (webm, ogg, mp3, wav, ...)
    into a mono float32 numpy array at target_sr, using ffmpeg under the hood."""
    audio = AudioSegment.from_file(io.BytesIO(blob_bytes))
    audio = audio.set_channels(1).set_frame_rate(target_sr)

    samples = np.array(audio.get_array_of_samples())
    # pydub gives int16 samples -- normalize to float32 [-1, 1]
    audio_float32 = samples.astype(np.float32) / 32768.0

    return audio_float32


def transcribe(
    audio_path: str,
    model_name: str = "mbaza_asr",
    url: str = "localhost:8000",
) -> tuple[str, str]:
    """Read a WAV/FLAC/etc file from disk, resample if needed, and send it."""
    audio, sr = sf.read(audio_path, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono

    audio = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SR)

    return transcribe_array(audio, model_name=model_name, url=url)


#### for debugging stuff
# if __name__ == "__main__":
#     audio_path = sys.argv[1]
#     transcribe(audio_path)