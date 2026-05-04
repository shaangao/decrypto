# run benchmark with vllm
- set the pairs of model you'd like to evaluate in yaml config files (e.g., `/decrypto/config/paper/figure_4_tom_gopnik.yaml`).
- set the models you'd like to lauch with vllm in the `models` variable and `ngpus` needed for each model in `/decrypto/slurm/launch_servers.sh`. 
- `conda activate decrypto` and run `/decrypto/slurm/launch_servers.sh`. it will automatically submit a dependency job `/decrypto/slurm/run_exp.sbatch` after submitting lauch_server jobs. benchmark takes significant time to run. so it's recommended to only run one experiment in each `/decrypto/slurm/run_exp.sbatch` submission.
- analyze figure 4 results: `/decrypto/analysis/figure_4.ipynb`

# serve models from MARSHAL checkpoints
- convert megatron format MARSHAL checkpoint to huggingface format: `/MARSHAL/scripts/model_convert_apptainer.sh`
- serve with vllm. example:
```
vllm serve /net/projects2/ycleong/sg/strategy-rl/MARSHAL/results/hf_models/selfplay/hanabi_selfplay \
    --port 8000 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-sliding-window \
    --disable-log-stats
```
- configure decrypto:
-- `/decrypto/src/utils/server.py`
-- experiment config files, e.g., `/decrypto/config/paper/figure_4_tom_gopnik.yaml`
-- `/decrypto/slurm/launch_servers.sh`
- run decrypto benchmark like normal