#!/usr/bin/env bash
# 模型拉取：优先 ModelScope（国内直连），失败则回退 hf-mirror。
#
# 用法：
#   bash scripts/download_models.sh [目标目录]     # 默认 /models
#
# 预计磁盘占用（2026-09 实测仓库体积，全部 Apache-2.0）：
#   1) Qwen3.8-27B-GGUF      只取 Q5_K_XL ≈ 20.9GB。该仓库含 25 个量化版本
#                            与 BF16 目录，整仓 >400GB，故必须用 --include
#                            过滤——不加过滤会撑爆挂载盘。
#   2) MOSS-TTSD             设了 HF_TOKEN：gated 仓库整仓（ModelScope 镜像
#                            68GB，其中 first_class/ 与词表另有约 25GB）；
#                            未设 HF_TOKEN：完整权重（transformers 后备）
#                            ≈ 16.7GB。
#   3) MOSS-Audio-Tokenizer-ONNX ≈ 14.2GB（ONNX fp32 权重）。
#   4) Z-Image-Turbo         ≈ 32.9GB（仓库只提供 bf16/fp32 权重，无 FP8）。
#
# 合计约 75GB（未设 HF_TOKEN 时）到 135GB（设了 HF_TOKEN 时）。
# 设计文档里 42-43GB 的估算是按「量化后权重」算的，实际仓库体积更大，
# 挂载盘请按 150GB 预留。若需从完整权重转换 MOSS-TTSD 的 first-class
# GGUF，再额外留约 17GB 的转换空间。

set -euo pipefail

MODELS_ROOT="${1:-/models}"
mkdir -p "${MODELS_ROOT}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

ensure_tools() {
    if ! have modelscope && ! have hf; then
        log "安装下载工具（modelscope + huggingface_hub）"
        pip install -U modelscope "huggingface_hub[cli]"
    fi
}

# 优先 ModelScope，失败回退 hf-mirror
#
#   fetch <repo> <dest> [--ms-repo <ModelScope id>] [--include <glob>]
#
# --ms-repo：两站的仓库 id 不同时用（如 OpenMOSS-Team/... 只存在于 HF）
# --include：只拉匹配的文件。用于避开「整仓含几十个量化版本」的仓库；
#            两个 CLI 的 --include 语义一致，任一失败都会走 else 分支回退。
fetch() {
    local repo="$1" dest="$2"; shift 2
    local ms_repo="${repo}" include=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --ms-repo) ms_repo="$2"; shift 2 ;;
            --include) include="$2"; shift 2 ;;
            *) echo "fetch: 未知参数 $1" >&2; return 2 ;;
        esac
    done

    if [ -d "${dest}" ] && [ -n "$(ls -A "${dest}" 2>/dev/null)" ]; then
        log "已存在，跳过：${dest}"
        return 0
    fi

    local -a extra=()
    [ -n "${include}" ] && extra=(--include "${include}")

    log "拉取 ${repo}"
    if have modelscope && modelscope download --model "${ms_repo}" \
            --local_dir "${dest}" "${extra[@]}"; then
        return 0
    fi
    log "ModelScope 失败，回退 hf-mirror"
    hf download "${repo}" --local-dir "${dest}" "${extra[@]}"
}

ensure_tools

# --- 1. 文稿模型：Qwen3.8-27B GGUF Q5_K_XL（约 20.9GB）---
# 选 Q5_K_XL 而非 Q8_0：留出显存给 TTS 与封面，使四个模型可同时常驻。
# --include 是必须的：整仓含 25 个量化版本（>400GB），只取要用的那个。
fetch "unsloth/Qwen3.8-27B-GGUF" "${MODELS_ROOT}/Qwen3.8-27B-GGUF" \
      --include '*Q5_K_XL*'

# --- 2. 语音模型：MOSS-TTSD（GGUF 整仓约 68GB / 完整权重约 17GB）---
# 注意：HF 上的 OpenMOSS-Team/MOSS-TTS-GGUF 为 gated 仓库，需先在网页
# 接受条款并配置 HF_TOKEN；ModelScope / GitCode 有同样的镜像。
if [ -n "${HF_TOKEN:-}" ]; then
    hf download "OpenMOSS-Team/MOSS-TTS-GGUF" --local-dir "${MODELS_ROOT}/MOSS-TTSD-GGUF" || \
        log "MOSS-TTS-GGUF 拉取失败——该仓库为 gated，请先接受条款或改用 ModelScope 镜像"
else
    log "未设置 HF_TOKEN，尝试 ModelScope 镜像"
    fetch "openmoss/MOSS-TTSD-v1.0" "${MODELS_ROOT}/MOSS-TTSD-GGUF" || \
        log "MOSS-TTSD GGUF 未获取，transformers 后端将回退到完整权重"
fi

# --- 3. 音频编解码器（约 14GB）---
# HF 上的 id 是 OpenMOSS-Team/...，ModelScope 上叫 openmoss/...，显式指定别名，
# 否则 ModelScope 一步必然 404，白白落到 hf-mirror。
fetch "OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX" \
      "${MODELS_ROOT}/MOSS-Audio-Tokenizer-ONNX" \
      --ms-repo "openmoss/MOSS-Audio-Tokenizer-ONNX"

# --- 4. 封面模型：Z-Image-Turbo（约 33GB）---
fetch "Tongyi-MAI/Z-Image-Turbo" "${MODELS_ROOT}/Z-Image-Turbo"

log "完成。目录结构："
find "${MODELS_ROOT}" -maxdepth 1 -mindepth 1 -type d -printf '  %p\n' | sort
cat <<EOF

请据此核对 configs/default.yaml 中的路径：
  tts.model_path   -> ${MODELS_ROOT}/MOSS-TTSD-GGUF
  cover.model_path -> ${MODELS_ROOT}/Z-Image-Turbo
EOF
