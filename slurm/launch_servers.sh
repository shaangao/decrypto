# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash


models=(
    # "llama3.2_1B:meta-llama/Llama-3.2-1B-Instruct"
    "llama3.1_8B:meta-llama/Meta-Llama-3.1-8B-Instruct"
    "llama3.1_70B:meta-llama/Meta-Llama-3.1-70B-Instruct"
#     "deepseek_r1_32B":"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    # "qwen2_1.5B:Qwen/Qwen2-1.5B"
)

# how many gpus for each model
ngpus=4

# Count the number of vllm models
EXPECTED_MODELS=${#models[@]}

# Launch vllm models
SERVER_JIDS=""
for model in "${models[@]}"; do
  # Extract the model key and name
  model_key=$(echo $model | cut -d ':' -f 1)
  model_name=$(echo $model | cut -d ':' -f 2)

  # generate a random port between 8000 and 8999
  port=$((8000 + RANDOM % 1000))

  jid=$(
    sbatch \
    --parsable \
    --gpus-per-node=$ngpus \
    --time=12:00:00 \
    --cpus-per-task=32 \
    --nodes=1 \
    --mem-per-cpu=16G \
    --constraint="a100|h100" \
    --ntasks=1 \
    --output "logs/vllm/${model_key}-%j.out" \
    --error "logs/vllm/${model_key}-%j.err" \
    --partition=general \
    --job-name "${model_key}:${port}" \
    --export=ALL,HF_HOME=/net/projects2/ycleong/sg/strategy-rl/tmp/hf_cache,HUGGINGFACE_HUB_CACHE=/net/projects2/ycleong/sg/strategy-rl/tmp/hf_cache \
    --wrap "vllm serve $model_name --port $port --enable-prefix-caching --tensor-parallel-size $ngpus --trust-remote-code --disable-sliding-window --disable-log-stats"
  )

  echo "Submitted $model_key with Job ID: $jid"

  if [ -z "$SERVER_JIDS" ]; then
    SERVER_JIDS=$jid
  else
    SERVER_JIDS="${SERVER_JIDS}:${jid}"
  fi
  
done


# Submit the run_exp.sbatch job, waiting for ALL servers to *start* executing
echo "Submitting evaluation job..."
sbatch --dependency=after:$SERVER_JIDS --export=ALL,SERVER_JIDS=$SERVER_JIDS,EXPECTED_MODELS=$EXPECTED_MODELS slurm/run_exp.sbatch
