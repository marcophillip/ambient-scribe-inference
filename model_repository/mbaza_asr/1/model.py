import os
import tempfile

import numpy as np
import soundfile as sf
import triton_python_backend_utils as pb_utils
import nemo.collections.asr as nemo_asr

MODEL_ID = "DigitalUmuganda/Mbaza-ASR-Afrivoice-660h"

# If a local .nemo checkpoint is present (e.g. mbaza_asr.nemo from your
# earlier ONNX export step, or any other .nemo file dropped in here), use
# it instead of downloading from the Hugging Face Hub on every cold start.
MODEL_DIR = os.path.join(os.path.dirname(__file__), "nemo_model")
LOCAL_NEMO_PATH = os.path.join(MODEL_DIR, "model.nemo")


class TritonPythonModel:
    def initialize(self, args):
        if os.path.isfile(LOCAL_NEMO_PATH):
            print(f"Loading local checkpoint from {LOCAL_NEMO_PATH} ...")
            self.model = nemo_asr.models.ASRModel.restore_from(
                LOCAL_NEMO_PATH, map_location="cpu"
            )
        else:
            print(f"No local checkpoint found, downloading {MODEL_ID} from the Hub ...")
            self.model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)

        self.model.eval()

    def execute(self, requests):
        responses = []

        for request in requests:
            audio_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO")
            audio_array = audio_tensor.as_numpy().astype(np.float32).flatten()

            # transcribe() takes file paths, not in-memory arrays -- its
            # accepted input types have varied across nemo_toolkit
            # versions, so a temp WAV file is the most stable option.
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                sf.write(tmp.name, audio_array, samplerate=16000)
                hypotheses = self.model.transcribe([tmp.name])

            # Return shape has varied across nemo_toolkit versions --
            # handle both a plain list[str] and a list of Hypothesis-like
            # objects.
            first = hypotheses[0]
            transcription = first if isinstance(first, str) else getattr(first, "text", str(first))

            out_tensor = pb_utils.Tensor(
                "TRANSCRIPTION",
                np.array([transcription], dtype=object),
            )
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_tensor])
            )

        return responses

    def finalize(self):
        self.model = None