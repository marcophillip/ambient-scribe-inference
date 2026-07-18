FROM nvcr.io/nvidia/tritonserver:26.05-py3
# FROM ambient-scribe:latest

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     cmake \
#     build-essential \
#     protobuf-compiler \
#     git \
#     && rm -rf /var/lib/apt/lists/*


# RUN pip install --no-cache-dir --force-reinstall --no-deps scipy      "optimum[onnxruntime-gpu]" \

# Install nemo_toolkit and other dependencies
RUN pip install --no-cache-dir \
    "nemo_toolkit[asr]" \
    transformers \
    soundfile \
    librosa \
    accelerate \
    onnxruntime

WORKDIR /models