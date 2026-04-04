import os
import glob
import json
import argparse
import pandas as pd
import numpy as np

# A list of models commonly found in your config files to help parse directory names.
KNOWN_MODELS = [
    "llama3.1_70B", "llama3.1_8B", "gpt-4o", "claude4.6", "claude3.7", 
    "o1", "qwen2_1.5B", "deepseek_r1_32B", "GloVe", "Word2Vec", 
    "baseline-GloVe", "baseline-Word2Vec", "alice"
]

def parse_model_string(model_string, known_models=KNOWN_MODELS):
    """
    Given a string like 'llama3.1_8B_llama3.1_8B_llama3.1_70B', 
    greedily match known models to extract encoder, decoder, interceptor.
    """
    models = []
    current_s = model_string
    for _ in range(3):
        # Check longest names first
        for m in sorted(known_models, key=len, reverse=True):
            if current_s.startswith(m):
                models.append(m)
                current_s = current_s[len(m):]
                if current_s.startswith('_'):
                    current_s = current_s[1:]
                break
    if len(models) == 3 and not current_s:
        return models
    else:
        # Fallback if unknown models were used
        print(f"Warning: Could not cleanly parse '{model_string}'. Storing as raw chunks.")
        return model_string.split("_")[:3]

def main(results_dir):
    summary_data = []

    # Recursively find all stats.json files
    stats_files = glob.glob(os.path.join(results_dir, "**", "stats.json"), recursive=True)
    
    if not stats_files:
        print(f"No stats.json files found in {results_dir}")
        return

    print(f"Found {len(stats_files)} experiments. Processing...")

    for stats_file in stats_files:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            
        parts = stats_file.split(os.sep)
        exp_dir = os.path.dirname(stats_file)
        
        # Structure handles paths with and without 'env_seedX' folder
        has_env_seed = "env_seed" in parts[-2]
        
        if has_env_seed:
            env_seed = int(parts[-2].replace("env_seed", ""))
            model_part = parts[-3]
            model_seed = int(parts[-4].replace("model_seed", ""))
        else:
            env_seed = None
            model_part = parts[-2]
            model_seed = int(parts[-3].replace("model_seed", ""))
            
        # Parse modes and ToM from folder name
        tom_mode = None
        if "_ToM" in model_part:
            model_part, tom_part = model_part.rsplit("_ToM", 1)
            tom_mode = int(tom_part)
            
        tokens = model_part.split("_")
        system_modes = tokens[-1]  # e.g., '000'
        modes = tokens[-2]         # e.g., '000'
        models_str = "_".join(tokens[:-2])
        
        enc_model, dec_model, int_model = parse_model_string(models_str)

        # Average turns aren't stored in `stats.json`, so we must recompute them
        # from the individual `stats_episode_*.json` histories inside the run folder.
        all_intercept_turns = []
        all_miscomm_turns = []
        
        episode_stats_files = glob.glob(os.path.join(exp_dir, "episode_*", "stats_episode_*.json"))
        for ep_file in episode_stats_files:
            with open(ep_file, 'r') as epf:
                ep_data = json.load(epf)
                all_intercept_turns.extend(ep_data.get("intercept_turns", []))
                all_miscomm_turns.extend(ep_data.get("miscomm_turns", []))
                
        avg_intercept_turn = np.mean(all_intercept_turns) if all_intercept_turns else 0.0
        avg_miscomm_turn = np.mean(all_miscomm_turns) if all_miscomm_turns else 0.0

        # Build row
        summary = {
            "env_seed": env_seed,
            "encoder": enc_model,
            "encoder_mode": modes[0],
            "encoder_system_mode": system_modes[0],
            "decoder": dec_model,
            "decoder_mode": modes[1],
            "decoder_system_mode": system_modes[1],
            "interceptor": int_model,
            "interceptor_mode": modes[2],
            "interceptor_system_mode": system_modes[2],
            "intercepts": stats.get("intercept", 0),
            "miscomms": stats.get("miscomm", 0),
            "both": stats.get("both", 0),
            "survived": stats.get("survived", 0),
            "avg_turns_per_episode": stats.get("avg_turns_per_episode", 0),
            "avg_intercept_turn": avg_intercept_turn,
            "avg_miscomm_turn": avg_miscomm_turn,
            "avg_changed_hint": stats.get("avg_changed_hint", 0),
            "avg_encoder_attempts": stats.get("avg_encoder_attempts", 0),
            "avg_decoder_attempts": stats.get("avg_decoder_attempts", 0),
            "avg_interceptor_attempts": stats.get("avg_interceptor_attempts", 0),
            "total_encoder_fails": stats.get("total_encoder_fails", 0),
            "total_decoder_fails": stats.get("total_decoder_fails", 0),
            "total_interceptor_fails": stats.get("total_interceptor_fails", 0),
        }
        
        # Optional: Theory of Mind overrides
        if tom_mode is not None:
            summary["encoder_tom_mode"] = tom_mode
            
        # Optional: Piaget overrides
        for k in ["avg_predicted_success", "avg_predicted_success_on_intercept", "avg_prediction_is_code"]:
            if k in stats:
                summary[k] = stats[k]
                
        # Optional: Gopnik overrides
        for k, v in stats.items():
            if "gopnik" in k:
                summary[k] = v

        summary_data.append(summary)

    # Save to experiment_summary.csv in the given directory
    df = pd.DataFrame(summary_data)
    summary_file = os.path.join(results_dir, "experiment_summary.csv")
    df.to_csv(summary_file, index=False)
    print(f"Success! Summary saved as '{summary_file}'.")

if __name__ == "__main__":
    """
    example usage:
    python analysis/csv_summary_working.py /net/projects2/ycleong/sg/strategy-rl/decrypto/results/figure_4_tom_piaget_all
    """
    parser = argparse.ArgumentParser(description="Generate experiment_summary.csv from experiment directories")
    parser.add_argument("results_dir", help="Path to the results directory (e.g., results/figure_4_tom_piaget_all)")
    args = parser.parse_args()
    
    main(args.results_dir)