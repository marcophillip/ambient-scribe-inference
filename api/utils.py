
import sys

import librosa
import numpy as np
import soundfile as sf
import tritonclient.http as httpclient
import io
from pydub import AudioSegment
import numpy as np
from openai import OpenAI
import os
from prompts import CLINICAL_SYSTEM_PROMPT, NO_CLINICAL

TARGET_SR = 16000
LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemma-3-4b-it")


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
    language: str = "auto",
) -> tuple[str, str]:
    """Send a waveform to the nemotron model on Triton and return
    (transcription, detected_language).

    `language` is a locale like "en-US"/"fr-FR", or "auto" to let the model
    detect it. detected_language is empty if the model emitted no language tag.
    """

    client = httpclient.InferenceServerClient(url=url)
    audio = audio.astype(np.float32)

    audio_input = httpclient.InferInput("AUDIO", audio.shape, "FP32")
    audio_input.set_data_from_numpy(audio)

    lang_input = httpclient.InferInput("LANGUAGE", [1], "BYTES")
    lang_input.set_data_from_numpy(np.array([language], dtype=object))

    outputs = [httpclient.InferRequestedOutput(name) for name in ("TRANSCRIPTION", "LANGUAGE_DETECTED")]

    response = client.infer(model_name=model_name, inputs=[audio_input, lang_input], outputs=outputs)
    text = response.as_numpy("TRANSCRIPTION")[0].decode("utf-8")

    result = response.as_numpy("LANGUAGE_DETECTED")
    detected = result[0].decode("utf-8") if result is not None else ""

    return text, detected


def blob_bytes_to_array(blob_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode an arbitrary-format audio blob (webm, ogg, mp3, wav, ...)
    into a mono float32 numpy array at target_sr, using ffmpeg under the hood."""
    audio = AudioSegment.from_file(io.BytesIO(blob_bytes))
    audio = audio.set_channels(1).set_frame_rate(target_sr)

    samples = np.array(audio.get_array_of_samples())
    # sample width depends on the source: wav/flac decode to int16, but
    # browser webm/opus decodes to int32 -- normalize by the actual width
    full_scale = float(1 << (8 * audio.sample_width - 1))
    audio_float32 = samples.astype(np.float32) / full_scale

    return audio_float32


def transcribe(
    audio_path: str,
    model_name: str = "nemotron_asr",
    url: str = "localhost:8003",
    language: str = "auto",
) -> tuple[str, str]:
    """Read a WAV/FLAC/etc file from disk, resample if needed, and send it."""
    audio, sr = sf.read(audio_path, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1) 

    audio = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SR)

    return transcribe_array(audio, model_name=model_name, url=url, language=language)



def structure_notes(
        raw_text: str,
        llm_url: str
):

    """Extract the clinical content of a consultation transcript as a markdown note."""
    if not raw_text.strip():
        return NO_CLINICAL

    client = OpenAI(
        base_url=llm_url,
        api_key="not-needed",
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n{raw_text}"},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    return (response.choices[0].message.content or "").strip()


#### for debugging stuff
# if __name__ == "__main__":
#     audio_path = sys.argv[1]
#     transcribe(audio_path)