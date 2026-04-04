#!/bin/bash
#SBATCH --job-name=decrypto
#SBATCH --nodes=2
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16G
#SBATCH --constraint="a100|h100"
#SBATCH --partition=general
#SBATCH --output=logs/paper/fig4_all_%j.out
#SBATCH --error=logs/paper/fig4_all_%j.err
#SBATCH --export=ALL,HF_HOME=/net/projects2/ycleong/sg/strategy-rl/tmp/hf_cache,HUGGINGFACE_HUB_CACHE=/net/projects2/ycleong/sg/strategy-rl/tmp/hf_cache


# TODO: slurm.ping_servers and get_models_from_slurm don't work for single job submission.

models=(
    "llama3.2_1B:meta-llama/Llama-3.2-1B"
    # "llama3.1_8B:meta-llama/Meta-Llama-3.1-8B-Instruct"
    # "llama3.1_70B:meta-llama/Meta-Llama-3.1-70B-Instruct"
#     "deepseek_r1_32B":"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "qwen2_1.5B:Qwen/Qwen2-1.5B"
)

# Count the number of vllm models
EXPECTED_MODELS=${#models[@]}

cd /net/projects2/ycleong/sg/strategy-rl/decrypto


# 1. Get the hostnames of the allocated nodes
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))


# 2. Launch models in the background on specific nodes
# for model in "${models[@]}"; do
for i in "${!models[@]}"; do

    model=${models[i]}

    # Extract the model key and name
    model_key=$(echo $model | cut -d ':' -f 1)
    model_name=$(echo $model | cut -d ':' -f 2)

    srun \
    --nodes=1 \
    --ntasks=1 \
    --nodelist=${NODES[i]} \
    --exact \
    vllm serve $model_name --enable-prefix-caching --tensor-parallel-size $SLURM_GPUS_PER_NODE --trust-remote-code --disable-sliding-window --disable-log-stats &

done


# 3. Wait for the model weights to load into VRAM (takes ~1-3 mins)
echo "Waiting for vLLM models to load into VRAM..."
while true; do
  # ping_servers.py prints "Reply time: X sec" for every healthy server
  READY_COUNT=$(python -m slurm.ping_servers | grep -c "Reply time")
  if [ "$READY_COUNT" -ge "$EXPECTED_MODELS" ]; then
      echo "All $EXPECTED_MODELS servers are ready!"
      break
  fi
  echo "$READY_COUNT out of $EXPECTED_MODELS servers ready. Sleeping 30s..."
  sleep 30
done


# 4. Run the experiments
python run.py --config-path=config/paper --config-name=figure_4_tom_piaget_test +get_models_from_slurm=true
python run.py --config-path=config/paper --config-name=figure_4_tom_gopnik_test +get_models_from_slurm=true
echo "Experiments finished."
