import json
import os
import re

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import nemo.collections.asr as nemo_asr
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


LOCAL_NEMO_PATH = os.path.join(os.path.dirname(__file__), "model.nemo")
NEMOTRON_ASR_HUGGINGFACE = os.getenv(
    "NEMOTRON_ASR", "nvidia/nemotron-3.5-asr-streaming-0.6b"
)

# Language-ID prompt used when a request does not carry a LANGUAGE input.
# "auto" lets the model detect the language and emit a <xx-XX> tag.
DEFAULT_TARGET_LANG = os.getenv("NEMOTRON_TARGET_LANG", "auto")

# att_context_size = [left_frames, right_frames] in 80ms frames. The right
# value picks the streaming chunk size: 0/1/3/6/13 -> 80/160/320/560/1120ms.
# 13 is the largest lookahead and the best-WER setting -- the right pick for
# whole-utterance requests, where the extra 1.12s of latency costs nothing.
ATT_CONTEXT_SIZE = json.loads(os.getenv("NEMOTRON_ATT_CONTEXT_SIZE", "[56, 13]"))

# The model appends the detected language after the terminal punctuation.
LANG_TAG_RE = re.compile(r"\s*<([a-z]{2}-[A-Z]{2})>\s*$")


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        instance_group = self.model_config.get("instance_group", [{}])[0]
        instance_kind = instance_group.get("kind", "KIND_CPU")

        if instance_kind == "KIND_GPU" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        if os.path.isfile(LOCAL_NEMO_PATH):
            print(f"Loading local checkpoint from {LOCAL_NEMO_PATH} ...")
            self.model = nemo_asr.models.ASRModel.restore_from(
                LOCAL_NEMO_PATH, map_location=self.device
            )
        else:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                print("HF_TOKEN not set -- relying on anonymous access")
            else:
                try:
                    HfApi().whoami(token=hf_token)
                except HfHubHTTPError:
                    raise RuntimeError("HF_TOKEN is invalid or expired")

            print("No local checkpoint found, downloading from huggingface ...")
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                NEMOTRON_ASR_HUGGINGFACE
            )
            self.model.to(self.device)

            print(f"Saving checkpoint to {LOCAL_NEMO_PATH} for next time ...")
            self.model.save_to(LOCAL_NEMO_PATH)

        self.model.to(self.device)

        # Cache-aware streaming encoder: fix the lookahead once at load time.
        if hasattr(self.model.encoder, "set_default_att_context_size"):
            self.model.encoder.set_default_att_context_size(
                att_context_size=ATT_CONTEXT_SIZE
            )

        # Keep the <xx-XX> tag in the decoded text so we can report the
        # detected language, and strip it from TRANSCRIPTION ourselves.
        if hasattr(self.model.decoding, "set_strip_lang_tags"):
            self.model.decoding.set_strip_lang_tags(False)

        self.model.eval()

        # language -> prompt index, e.g. {"en-US": 0, "fr-FR": 8, "auto": 101}
        self.prompt_dict = dict(self.model.cfg.model_defaults.prompt_dictionary)

    @torch.no_grad()
    def _transcribe(self, audio_array: np.ndarray, target_lang: str) -> str:
        """Run the encoder + RNNT decoder directly with an explicit prompt index.

        model.transcribe() is not used: its Lhotse dataset derives the prompt
        from a random "unified" mode (crashing with "Unknown prompt key: 'None'"
        about half the time) and overrides target_lang when it does succeed.
        """
        if target_lang not in self.prompt_dict:
            raise ValueError(
                f"Unsupported LANGUAGE '{target_lang}'. "
                f"Use 'auto' or one of: {sorted(k for k in self.prompt_dict if k != 'auto')}"
            )

        signal = torch.from_numpy(audio_array).unsqueeze(0).to(self.device)
        signal_len = torch.tensor([signal.shape[1]], dtype=torch.long, device=self.device)
        prompt_indices = torch.tensor(
            [self.prompt_dict[target_lang]], dtype=torch.long, device=self.device
        )

        encoded, encoded_len = self.model.forward(
            input_signal=signal, input_signal_length=signal_len, prompt_indices=prompt_indices
        )
        hypotheses = self.model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
        )

        # Return shape has varied across nemo_toolkit versions -- older ones
        # return a (best, all) tuple, items may be str or Hypothesis objects.
        if isinstance(hypotheses, tuple):
            hypotheses = hypotheses[0]
        first = hypotheses[0]
        return first if isinstance(first, str) else getattr(first, "text", str(first))

    def execute(self, requests):
        responses = []

        for request in requests:
            audio_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO")
            audio_array = audio_tensor.as_numpy().astype(np.float32).flatten()

            # LANGUAGE is optional -- fall back to the configured default.
            lang_tensor = pb_utils.get_input_tensor_by_name(request, "LANGUAGE")
            if lang_tensor is None:
                target_lang = DEFAULT_TARGET_LANG
            else:
                raw = lang_tensor.as_numpy().flatten()[0]
                target_lang = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                target_lang = target_lang.strip() or DEFAULT_TARGET_LANG

            try:
                transcription = self._transcribe(audio_array, target_lang)
            except ValueError as e:
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e))))
                continue

            match = LANG_TAG_RE.search(transcription)
            if match:
                detected_lang = match.group(1)
                transcription = LANG_TAG_RE.sub("", transcription)
            else:
                detected_lang = "" if target_lang == "auto" else target_lang

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "TRANSCRIPTION", np.array([transcription], dtype=object)
                        ),
                        pb_utils.Tensor(
                            "LANGUAGE_DETECTED", np.array([detected_lang], dtype=object)
                        ),
                    ]
                )
            )

        return responses

    def finalize(self):
        self.model = None
