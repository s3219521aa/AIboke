# AI 中文双人播客生成系统
#
# 模型不烘进镜像，运行时以 -v 挂载（符合赛题「支持挂载模型」）：
#   docker run --gpus all -v /host/models:/models \
#       -v /host/out:/out -e MODELS_ROOT=/models aiboke \
#       python scripts/generate.py --input /case.json --output-dir /out
#
# 注意：llama.cpp 需在此镜像内或构建阶段编译（见 scripts/build_llamacpp.sh），
# 因为要针对 A100 的 sm_80 编译 CUDA 内核。

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    CUDA_ARCH=80

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg git cmake build-essential ninja-build \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    python -m pip install --upgrade pip

WORKDIR /app

# 先装依赖，利用层缓存。
# 必须带 [gpu]：封面用 cover_runner 以 sys.executable 起子进程调 diffusers/
# torch，transformers 后备也要 torch/transformers——它们得装在同一个解释器
# 环境里，否则封面只能退化成纯色兜底图。
COPY pyproject.toml ./
RUN pip install -e ".[server,gpu]"

# 编译两份 llama.cpp（upstream 用于文稿，OpenMOSS fork 用于语音）
COPY scripts/build_llamacpp.sh /app/scripts/build_llamacpp.sh
RUN bash /app/scripts/build_llamacpp.sh /opt/llama.cpp

# 源码
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
RUN pip install -e .

ENV PATH="/opt/llama.cpp/upstream/build-cuda/bin:/opt/llama.cpp/openmoss/build-cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/llama.cpp/upstream/build-cuda/bin:/opt/llama.cpp/openmoss/build-cuda/bin" \
    MODELS_ROOT=/models \
    PYTHONPATH=/app/src

VOLUME ["/models", "/out"]

# 默认进入交互 shell；评测方按约定传入实际命令
CMD ["bash"]
