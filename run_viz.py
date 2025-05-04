import numpy as np
import os
import json
from viz import AdvancedVisualizer

# Initialize visualizer
visualizer = AdvancedVisualizer(results_dir='evaluation_results')

# Example: Creating a grid search visualization for num_layers and num_heads
def create_grid_search_plot():
    # Parameters and their values
    layer_values = [2, 4, 6, 8]
    head_values = [4, 8, 16]
    
    # Create empty results grid
    results_grid = np.zeros((len(layer_values), len(head_values)))
    results_grid.fill(np.nan)  # Fill with NaN to indicate missing data
    
    # For each combination, try to find the result
    for i, layers in enumerate(layer_values):
        for j, heads in enumerate(head_values):
            # Look for a directory with this combination
            stats_file = f"evaluation_results/model_transformer_num_layers/model_comparison/{layers}/stats.json"
            
            # If the exact combination exists, use its result
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    results_grid[i, j] = stats['mean_error']
    
    # Plot the grid search results (only for combinations that exist)
    visualizer.plot_parameter_grid_search(
        'transformer.num_layers', 
        'transformer.num_heads', 
        results_grid,
        metric='mean_error',
        title='Effect of Transformer Layers and Heads on Angular Error'
    )



def create_radar_chart():
    # Collect best values for each parameter
    best_configs = {}
    
    # Find best value for each parameter type
    parameter_experiments = {
        'num_layers': 'model_transformer_num_layers',
        'num_heads': 'model_transformer_num_heads',
        'feature_extractor': 'model_feature_extractor_type',
        'patch_size': 'model_patch_size',
        'vp_head': 'model_vp_head_type',
        'loss_type': 'training_loss_type'
    }
    
    for param_name, experiment in parameter_experiments.items():
        comparison_file = f"evaluation_results/{experiment}/model_comparison/comparison.json"
        
        if os.path.exists(comparison_file):
            with open(comparison_file, 'r') as f:
                comparison = json.load(f)
            
            # Find best model (lowest mean error)
            best_model = None
            best_error = float('inf')
            
            for model_name, stats in comparison.items():
                if 'mean_error' in stats and stats['mean_error'] < best_error:
                    best_error = stats['mean_error']
                    best_model = model_name
            
            if best_model:
                # Convert string model name to its parameter value
                best_configs[param_name] = best_model
                best_configs[f"{param_name}_error"] = best_error
    
    # Create radar chart with best parameter values
    visualizer.create_parameter_radar_chart(
        best_configs,
        title="Best Configuration from Ablation Studies"
    )


def find_best_models():
    best_models = {}
    results_dir = 'evaluation_results'
    
    # List all experiment directories
    experiments = [d for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d))]
    
    for experiment in experiments:
        # Look for model_comparison directory
        comparison_dir = os.path.join(results_dir, experiment, 'model_comparison')
        if not os.path.exists(comparison_dir):
            continue
            
        # Look for comparison.json which contains all model results
        comparison_file = os.path.join(comparison_dir, 'comparison.json')
        if not os.path.exists(comparison_file):
            continue
            
        # Load comparison results
        with open(comparison_file, 'r') as f:
            comparison = json.load(f)
            
        # Find best model (lowest mean error)
        best_model = None
        best_error = float('inf')
        
        for model_name, stats in comparison.items():
            if 'mean_error' in stats and stats['mean_error'] < best_error:
                best_error = stats['mean_error']
                best_model = model_name
                
        if best_model:
            error_path = os.path.join(comparison_dir, best_model, 'angular_errors.npy')
            if os.path.exists(error_path):
                best_models[f'best_{experiment}'] = error_path
    
    return best_models

def last_run():
    experiments = [
        'model_transformer_num_layers',
        'model_feature_extractor_type',
        'training_loss_type',
        'model_vp_head_type',
        'model_patch_size',
        'model_transformer_num_heads'
    ]

    for experiment in experiments:
        # Strip 'model_' prefix and convert underscores to dots for parameter path
        if experiment.startswith('model_'):
            param_path = experiment[6:].replace('_', '.')
        else:
            param_path = experiment.replace('_', '.')
        
        print(f"Creating visualizations for: {experiment}")
        
        try:
            # Create parameter comparison plot
            visualizer.plot_parameter_comparison(param_path)
            
            # Create convergence comparison plot
            visualizer.plot_convergence_comparison(param_path)
        except Exception as e:
            print(f"Error creating visualizations for {experiment}: {str(e)}")


# Use the function to get the actual best models
# best_models = find_best_models()
# print("Best models found:", best_models)


# print("Running grid search plot")
# create_grid_search_plot()


# Best models
best_models = {
    'Transformer Layers (2)': 'evaluation_results/model_transformer_num_layers/model_comparison/2/angular_errors.npy',
    'Feature Extractor (ResNet)': 'evaluation_results/model_feature_extractor_type/model_comparison/resnet/angular_errors.npy',
    'Loss Type (MSE)': 'evaluation_results/training_loss_type/model_comparison/mse/angular_errors.npy',
    'VP Head (Hybrid)': 'evaluation_results/model_vp_head_type/model_comparison/hybrid/angular_errors.npy',
    'Patch Size (32)': 'evaluation_results/model_patch_size/model_comparison/32/angular_errors.npy',
    'Transformer Heads (4)': 'evaluation_results/model_transformer_num_heads/model_comparison/4/angular_errors.npy'
}

# Plot angular error comparison
visualizer.plot_angular_error_comparison(best_models)

# Create individual error distributions
for model_name, error_path in best_models.items():
    try:
        # Load the angular errors
        errors = np.load(error_path)
        
        # Create error distribution visualization
        visualizer.plot_angular_error_distribution(
            errors, 
            experiment_name=model_name
        )
    except Exception as e:
        print(f"Error creating plot for {model_name}: {str(e)}")


# print("radar chart")
# # create_radar_chart()


# print("Last run")
# last_run()


