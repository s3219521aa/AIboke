#!/usr/bin/env bash
# 编译两份 llama.cpp：
#   1. upstream（用于 Qwen3.8-27B 文稿生成）
#   2. OpenMOSS fork（moss-tts-firstclass 分支，用于 MOSS-TTSD 语音合成）
#
# 为什么要两份：MOSS-TTS 的 first-class 推理路径位于 OpenMOSS 的 fork 中，
# 该 fork 是否已 rebase 到支持 Qwen3.8 的上游 master 无法事先确认，
# 因此分开构建互不影响。
#
# 关键：必须同时更新二进制与 .so。只替换 llama-server 而保留旧的
# libggml-cuda.so 会导致 Qwen3.8 的 DeltaNet 层静默产出乱码。
#
# 用法：bash scripts/build_llamacpp.sh [安装根目录]   # 默认 /opt/llama.cpp

set -euo pipefail

PREFIX="${1:-/opt/llama.cpp}"
JOBS="$(nproc)"
# A100 是 sm_80，不要照抄消费级卡的 86/89
CUDA_ARCH="${CUDA_ARCH:-80}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

build() {
    local name="$1" url="$2" branch="$3" srcdir="$4"
    log "构建 ${name}（arch=${CUDA_ARCH}）"
    if [ ! -d "${srcdir}/.git" ]; then
        git clone --branch "${branch}" "${url}" "${srcdir}"
    else
        git -C "${srcdir}" fetch --all --tags
        git -C "${srcdir}" checkout "${branch}"
        git -C "${srcdir}" pull --ff-only
    fi
    # 确保不是浅克隆——浅克隆下 git pull 会谎报 Already up to date
    if [ -f "${srcdir}/.git/shallow" ]; then
        git -C "${srcdir}" fetch --unshallow
    fi

    cmake -S "${srcdir}" -B "${srcdir}/build-cuda" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
    cmake --build "${srcdir}/build-cuda" --config Release -j "${JOBS}"
}

mkdir -p "${PREFIX}"

# 1) upstream：文稿生成
build "llama.cpp (upstream)" \
      "https://github.com/ggml-org/llama.cpp" "master" \
      "${PREFIX}/upstream"

# 2) OpenMOSS fork：MOSS-TTSD 语音合成
build "llama.cpp (OpenMOSS, moss-tts-firstclass)" \
      "https://github.com/OpenMOSS/llama.cpp" "moss-tts-firstclass" \
      "${PREFIX}/openmoss"

cat <<EOF

构建完成。

请记录 upstream 的版本号，必须 >= b10450：
  ${PREFIX}/upstream/build-cuda/bin/llama-server --version

自检：确认二进制与 .so 来自同一次构建（否则可能静默乱码）
  ldd ${PREFIX}/upstream/build-cuda/bin/llama-server | grep ggml
  echo \$LD_LIBRARY_PATH

若 MOSS-TTSD 的 GGUF 不是 first-class 格式，还需从完整权重转换
（注意：模型目录是位置参数，该脚本没有 --model-gguf 这个标志）：
  python ${PREFIX}/openmoss/convert_hf_to_gguf.py <MOSS-TTSD 完整权重目录> \\
      --outfile <输出.gguf> --outtype f16
EOF
