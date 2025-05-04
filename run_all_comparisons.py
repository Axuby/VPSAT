import os
import glob
from pathlib import Path
from angular_ablation import compare_models

def run_all_comparisons():
    """Run compare_models for all experiment folders in ablation_results."""
   
    base_dir = "ablation_results/"
    
    # Find all experiment directories (they should contain checkpoint folders)
    experiment_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if this looks like an experiment directory
            # (contains model checkpoints or has parameter value subdirectories)
            checkpoint_pattern = os.path.join(item_path, "*", "checkpoint_best.pth.tar")
            if glob.glob(checkpoint_pattern):
                experiment_dirs.append(item)
    
    print(f"Found {len(experiment_dirs)} experiment directories: {experiment_dirs}")
    
    # Process each experiment directory
    for exp_dir in experiment_dirs:
        print(f"\n{'='*50}\nProcessing experiment: {exp_dir}\n{'='*50}")
        
        # Find all parameter value directories within this experiment
        full_exp_dir = os.path.join(base_dir, exp_dir)
        param_dirs = []
        
        for item in os.listdir(full_exp_dir):
            item_path = os.path.join(full_exp_dir, item)
            if os.path.isdir(item_path):
                # Check if this directory contains a model checkpoint
                checkpoint_path = os.path.join(item_path, "checkpoint_best.pth.tar")
                if os.path.exists(checkpoint_path):
                    param_dirs.append(item)
        
        if not param_dirs:
            print(f"No parameter value directories found in {exp_dir}, skipping...")
            continue
        
        print(f"Found {len(param_dirs)} parameter values: {param_dirs}")
        
        # Create model configs for comparison
        model_configs = {}
        for param_dir in param_dirs:
            # Extract parameter value from directory name
            param_value = param_dir.split('_')[-1]  # Assumes naming like "num_layers_4"
            
            # Paths to checkpoint and config
            checkpoint_path = os.path.join(base_dir, exp_dir, param_dir, "checkpoint_best.pth.tar")
            config_path = os.path.join(base_dir, exp_dir, param_dir, "config.yaml")
            
            # Only include if both files exist
            if os.path.exists(checkpoint_path) and os.path.exists(config_path):
                model_configs[param_value] = (checkpoint_path, config_path)
            else:
                print(f"Missing checkpoint or config for {param_dir}, skipping...")
        
        if not model_configs:
            print(f"No valid model configurations found for {exp_dir}, skipping...")
            continue
        
        # Create output directory
        output_dir = f"evaluation_results/{exp_dir}/model_comparison"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run compare_models
        try:
            compare_models(model_configs, output_dir)
            print(f"Successfully compared models for {exp_dir}")
        except Exception as e:
            print(f"Error comparing models for {exp_dir}: {e}")

if __name__ == "__main__":
    run_all_comparisons()