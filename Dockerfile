FROM nvcr.io/nvidia/tritonserver:26.05-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    protobuf-compiler \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir Cython packaging
RUN pip install --no-cache-dir \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main" \
    "transformers>=5.13.0" \
    soundfile \
    librosa \
    accelerate \
    torch \
    huggingface_hub


WORKDIR /models