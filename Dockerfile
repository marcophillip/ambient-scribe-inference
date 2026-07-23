FROM nvcr.io/nvidia/tritonserver:26.05-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    protobuf-compiler \
    git \
    && rm -rf /var/lib/apt/lists/*



# Install nemo_toolkit and other dependencies
RUN pip install --no-cache-dir \
    "nemo_toolkit[asr]" \
    "optimum[onnxruntime]" \
    transformers \
    soundfile \
    librosa \
    accelerate \
    torch

WORKDIR /models