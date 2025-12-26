#!/usr/bin/env bash
set -euo pipefail

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118  https_proxy=http://sys-proxy-rd-relay.byted.org:8118  no_proxy=code.byted.org


# zezhou: you need 4.56.2 for gpt-oss.
pip install transformers==4.56.2
pip install bitsandbytes==0.45.5
pip install accelerate
MAX_JOBS=64 python -m pip -v install flash-attn --no-build-isolation

# Megatron-fsdp
# check `https://pypi.org/project/megatron-fsdp/0.1.0rc3/#description`
MAX_JOBS=64 python -m pip -v install transformer-engine[pytorch] --no-build-isolation
pip install "megatron-core[mlm]"
MAX_JOBS=64 python -m pip -v install "git+https://github.com/NVIDIA/Megatron-LM.git@4cf968cd26cc0e8cfcb65eebac6e3a60b220699d"
pip install megatron-fsdp

pip install transformers==4.56.2
