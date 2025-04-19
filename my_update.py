import os
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

from models.config import C, M
from models.vpsat import VpSatNet, multi_vp_loss
from data.datasets import WireframeDataset
from utils.config_loader import load_config
from models.trainer import Trainer


class AblationStudy:
    def __init__(self, base_config_path='./config/model_config.yaml', results_dir='./ablation_results'):
        """
        Initialize the ablation study manager.

        Args:
            base_config_path: Path to the base YAML configuration file
            results_dir: Directory to save results
        """
        self.base_config_path = base_config_path
        self.base_config = load_config(base_config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = defaultdict(list)

    def modify_config(self, param_path, value):
        """
        Modify a specific parameter in the configuration.

        Args:
            param_path: Dot-separated path to the parameter (e.g., 'model.transformer.num_layers')
            value: The new value to set

        Returns:
            Modified config dictionary
        """
        config = deepcopy(self.base_config)
        keys = param_path.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
        return config

    def run_experiment(self, experiment_name, param_path, values, epochs=15):
        """
        Run an ablation study by varying a parameter.

        Args:
            experiment_name: Name of the experiment
            param_path: Parameter path to modify
            values: List of values to test
            epochs: Number of epochs to train each model variant

        Returns:
            Dictionary with results
        """
        print(f"Running ablation study: {experiment_name} - Testing {param_path}")
        results = []

        for value in tqdm(values, desc=f"Testing {param_path}"):
            config = self.modify_config(param_path, value)

            # Update configuration
            C.update(config)
            M.update(C.model)

            # Prepare result directory
            exp_dir = self.results_dir / experiment_name / f"{param_path.replace('.', '_')}_{value}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Save configuration
            with open(exp_dir / 'config.yaml', 'w') as f:
                yaml.dump(config, f)

            # Set up model paths
            C.io.model_save_dir = str(exp_dir)
            C.io.logdir = str(exp_dir / "logs")

            # Set up data loaders
            device = torch.device(C.training.device if torch.cuda.is_available() else "cpu")
            Dataset = WireframeDataset
            kwargs = {"batch_size": C.training.batch_size, "num_workers": C.io.num_workers, "pin_memory": True}

            train_loader = torch.utils.data.DataLoader(
                Dataset(C.io.datadir, split="train"), shuffle=True, **kwargs
            )
            val_loader = torch.utils.data.DataLoader(
                Dataset(C.io.datadir, split="valid"), shuffle=False, **kwargs
            )

            # Set up model and optimizer
            model = VpSatNet(C).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=C.optim.lr)

            # Train the model
            trainer = Trainer(model, optimizer, train_loader, val_loader, C)
            trainer.train(epochs)

            # Get results
            result = {
                'param': param_path,
                'value': value,
                'train_loss': trainer.train_losses[-1],
                'val_loss': trainer.val_losses[-1],
                'best_val_loss': trainer.best_val_loss,
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses
            }

            # Save results
            with open(exp_dir / 'results.json', 'w') as f:
                json.dump(result, f)

            results.append(result)

        # Save combined results
        self.results[experiment_name] = results
        with open(self.results_dir / f"{experiment_name}_results.json", 'w') as f:
            json.dump(results, f)

        return results

    def plot_results(self, experiment_name, metric='best_val_loss', ylabel=None, title=None):
        """
        Plot the results of an ablation study.

        Args:
            experiment_name: Name of the experiment
            metric: Metric to plot ('train_loss', 'val_loss', or 'best_val_loss')
            ylabel: Label for y-axis
            title: Plot title

        Returns:
            Matplotlib figure
        """
        results = self.results.get(experiment_name)
        if not results:
            # Try to load from file
            try:
                with open(self.results_dir / f"{experiment_name}_results.json", 'r') as f:
                    results = json.load(f)
                self.results[experiment_name] = results
            except:
                print(f"No results found for experiment {experiment_name}")
                return None

        # Extract parameter path from first result
        param_path = results[0]['param']
        param_name = param_path.split('.')[-1]

        # Extract values and metrics
        values = [r['value'] for r in results]
        metrics = [r[metric] for r in results]

        # Create figure
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot
        plt.plot(values, metrics, 'o-', linewidth=2, markersize=8)

        # Set labels and title
        plt.xlabel(param_name)
        plt.ylabel(ylabel or metric.replace('_', ' ').title())
        plt.title(title or f"Effect of {param_name} on {metric.replace('_', ' ').title()}")

        # Save figure
        fig_path = self.results_dir / f"{experiment_name}_{param_name}_{metric}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        return plt.gcf()

    def plot_loss_curves(self, experiment_name, value_idx=0):
        """
        Plot training and validation loss curves for a specific experiment configuration.

        Args:
            experiment_name: Name of the experiment
            value_idx: Index of the value to plot

        Returns:
            Matplotlib figure
        """
        results = self.results.get(experiment_name)
        if not results:
            # Try to load from file
            try:
                with open(self.results_dir / f"{experiment_name}_results.json", 'r') as f:
                    results = json.load(f)
                self.results[experiment_name] = results
            except:
                print(f"No results found for experiment {experiment_name}")
                return None

        result = results[value_idx]
        param_path = result['param']
        param_name = param_path.split('.')[-1]
        value = result['value']

        # Create figure
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot
        epochs = range(1, len(result['train_losses']) + 1)
        plt.plot(epochs, result['train_losses'], 'b-', linewidth=2, label='Training Loss')
        plt.plot(epochs, result['val_losses'], 'r-', linewidth=2, label='Validation Loss')

        # Set labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"Training and Validation Loss for {param_name}={value}")
        plt.legend()

        # Save figure
        fig_path = self.results_dir / f"{experiment_name}_{param_name}_{value}_loss_curves.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        return plt.gcf()

    def plot_angular_error_histogram(self, angles, experiment_name, param_value):
        """
        Plot histogram of angular errors.

        Args:
            angles: List of angular errors in degrees
            experiment_name: Name of the experiment
            param_value: Value of the parameter for this experiment

        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        bins = np.linspace(0, 90, 19)  # 0 to 90 degrees in 5-degree bins

        plt.hist(angles, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
        plt.axvline(np.median(angles), color='r', linestyle='dashed', linewidth=2,
                    label=f'Median: {np.median(angles):.2f}°')

        # Add mean line
        plt.axvline(np.mean(angles), color='g', linestyle='dashed', linewidth=2,
                    label=f'Mean: {np.mean(angles):.2f}°')

        plt.xlabel('Angular Error (degrees)')
        plt.ylabel('Count')
        plt.title(f'Distribution of Angular Errors - {experiment_name} = {param_value}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save figure
        fig_path = self.results_dir / f"{experiment_name}_{param_value}_angular_errors.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        return plt.gcf()

    def plot_comparative_bar_chart(self, experiment_names, param_values, angular_errors):
        """
        Plot a comparative bar chart for angular errors across different experiments.

        Args:
            experiment_names: List of experiment names
            param_values: List of parameter values
            angular_errors: List of lists of angular errors

        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")

        # Calculate mean and median errors
        mean_errors = [np.mean(errors) for errors in angular_errors]
        median_errors = [np.median(errors) for errors in angular_errors]

        # Create x positions for bars
        x = np.arange(len(experiment_names))
        width = 0.35

        # Plot bars
        plt.bar(x - width / 2, mean_errors, width, label='Mean Error', color='steelblue')
        plt.bar(x + width / 2, median_errors, width, label='Median Error', color='lightcoral')

        # Add text on bars
        for i, v in enumerate(mean_errors):
            plt.text(i - width / 2, v + 0.5, f'{v:.2f}°', ha='center')

        for i, v in enumerate(median_errors):
            plt.text(i + width / 2, v + 0.5, f'{v:.2f}°', ha='center')

        # Set labels and title
        plt.xlabel('Experiment')
        plt.ylabel('Angular Error (degrees)')
        plt.title('Comparison of Angular Errors Across Experiments')
        plt.xticks(x, [f'{name}={val}' for name, val in zip(experiment_names, param_values)], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save figure
        plt.tight_layout()
        fig_path = self.results_dir / "comparative_angular_errors.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        return plt.gcf()

    def plot_heatmap(self, results_2d, x_param, y_param, x_values, y_values, metric='best_val_loss'):
        """
        Plot a heatmap for two-dimensional parameter search.

        Args:
            results_2d: 2D array of results
            x_param: Name of parameter on x-axis
            y_param: Name of parameter on y-axis
            x_values: List of values for x-axis parameter
            y_values: List of values for y-axis parameter
            metric: Metric to visualize

        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 8))
        sns.set_style("white")

        # Create heatmap
        ax = sns.heatmap(results_2d, annot=True, fmt=".3f", cmap="YlGnBu",
                         xticklabels=x_values, yticklabels=y_values)

        # Set labels and title
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.title(f"{metric.replace('_', ' ').title()} for Different {x_param} and {y_param} Values")

        # Save figure
        fig_path = self.results_dir / f"heatmap_{x_param}_{y_param}_{metric}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        return plt.gcf()


def run_single_parameter_ablation(param_path, values, base_config_path='./config/model_config.yaml',
                                  results_dir='./ablation_results', epochs=15):
    """
    Convenience function to run a single parameter ablation study.

    Args:
        param_path: Parameter path to modify (e.g., 'model.transformer.num_layers')
        values: List of values to test
        base_config_path: Path to the base configuration file
        results_dir: Directory to save results
        epochs: Number of epochs to train each model variant

    Returns:
        Results of the ablation study
    """
    study = AblationStudy(base_config_path, results_dir)
    experiment_name = param_path.replace('.', '_')
    results = study.run_experiment(experiment_name, param_path, values, epochs)

    # Plot results
    study.plot_results(experiment_name)

    # Plot loss curves for the best configuration
    best_idx = np.argmin([r['best_val_loss'] for r in results])
    study.plot_loss_curves(experiment_name, best_idx)

    return results


def evaluate_angular_errors(model_path, config_path, output_dir):
    """
    Evaluate a trained model and compute angular errors.

    Args:
        model_path: Path to the trained model checkpoint
        config_path: Path to the configuration file
        output_dir: Directory to save evaluation results

    Returns:
        List of angular errors
    """
    # Load configuration
    config = load_config(config_path)
    C.update(config)
    M.update(C.model)

    # Load the model
    device = torch.device(C.training.device if torch.cuda.is_available() else "cpu")
    model = VpSatNet(C).to(device)

    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Set up data loader
    Dataset = WireframeDataset
    kwargs = {"batch_size": C.training.batch_size, "num_workers": C.io.num_workers, "pin_memory": True}
    val_loader = torch.utils.data.DataLoader(Dataset(C.io.datadir, split="test"), shuffle=False, **kwargs)

    # Collect angular errors
    angular_errors = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            original_image_size = images[0].shape[1:]

            # Preprocess images
            from utils.helper_functions import preprocess_batch, extract_patches, to_pixel, adjust_vanishing_points

            images = preprocess_batch(images, C.model.input_image_size)
            patches = extract_patches(images, C.model.patch_size)
            b, n, c, h, w = patches.shape
            patches = patches.view(b * n, c, h, w).to(device)

            n_patches = images[0].size()[1] * images[0].size()[2] // (C.model.patch_size ** 2)
            labels_vpts = labels["vpts"].unsqueeze(1).repeat(1, n_patches, 1, 1).view(-1, 3, 3)
            vpts_2d = to_pixel(labels_vpts, C.io.focal_length, original_image_size[0])
            vpts_2d = adjust_vanishing_points(vpts_2d, original_image_size, C.model.input_image_size).to(device)

            # Forward pass
            outputs = model(patches)

            # Compute angular errors
            outputs_norm = torch.nn.functional.normalize(outputs, p=2, dim=-1)
            vpts_2d_norm = torch.nn.functional.normalize(vpts_2d, p=2, dim=-1)

            cosine_sim = torch.sum(outputs_norm * vpts_2d_norm, dim=-1)
            angles = torch.acos(cosine_sim.clamp(-1.0, 1.0)) * (180.0 / np.pi)

            angular_errors.extend(angles.cpu().numpy().flatten().tolist())

    # Save angular errors
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'angular_errors.npy'), np.array(angular_errors))

    # Compute statistics
    mean_error = np.mean(angular_errors)
    median_error = np.median(angular_errors)
    std_error = np.std(angular_errors)

    # Save statistics
    stats = {
        'mean_error': float(mean_error),
        'median_error': float(median_error),
        'std_error': float(std_error),
    }

    with open(os.path.join(output_dir, 'error_stats.json'), 'w') as f:
        json.dump(stats, f)

    print(f"Mean Angular Error: {mean_error:.2f}°")
    print(f"Median Angular Error: {median_error:.2f}°")
    print(f"Standard Deviation: {std_error:.2f}°")

    return angular_errors


def run_comprehensive_ablation():
    """
    Run a comprehensive ablation study on multiple parameters.
    """
    # Create ablation study object
    study = AblationStudy()

    # Define parameters to ablate
    ablation_params = [
        {
            'param_path': 'model.transformer.num_layers',
            'values': [2, 4, 6, 8]
        },
        {
            'param_path': 'model.transformer.num_heads',
            'values': [4, 8, 12]
        },
        {
            'param_path': 'model.transformer.d_model',
            'values': [128, 256, 512]
        },
        {
            'param_path': 'model.patch_size',
            'values': [16, 32, 64]
        },
        {
            'param_path': 'model.feature_extractor_type',
            'values': ['cnn', 'resnet']
        },
        {
            'param_path': 'model.vp_head_type',
            'values': ['simple', 'hybrid']
        },
        {
            'param_path': 'training.loss_type',
            'values': ['mse', 'cosine', 'combined']
        }
    ]

    # Run experiments
    for param_config in ablation_params:
        param_path = param_config['param_path']
        values = param_config['values']
        experiment_name = param_path.replace('.', '_')

        # Run experiment
        study.run_experiment(experiment_name, param_path, values, epochs=15)

        # Plot results
        study.plot_results(experiment_name)

    # Compare best configurations
    best_configs = []
    for param_config in ablation_params:
        param_path = param_config['param_path']
        experiment_name = param_path.replace('.', '_')

        # Load results
        with open(study.results_dir / f"{experiment_name}_results.json", 'r') as f:
            results = json.load(f)

        # Find best configuration
        best_idx = np.argmin([r['best_val_loss'] for r in results])
        best_configs.append(results[best_idx])

    # Save best configurations
    with open(study.results_dir / "best_configs.json", 'w') as f:
        json.dump(best_configs, f)

    return best_configs


# Example usage

    # Run a simple ablation study on number of transformer layers
    
if __name__ == '__main__':
    run_single_parameter_ablation(
        param_path='model.vp_head_type',
        values=['simple', 'hybrid'],
        epochs=15
    )
    # run_single_parameter_ablation(
    #     param_path='model.use_fpn',
    #     values=[True, False],
    #     epochs=15
    # )
    # run_single_parameter_ablation(
    #     param_path='model.feature_extractor_type', 
    #     values=['cnn', 'resnet'], 
    #     epochs=5
    # )

# run_single_parameter_ablation(
#     param_path='model.transformer.num_layers',
#     values=[2, 4, 6, 8],
#     epochs=15
# )

    # More comprehensive ablation
    # run_comprehensive_ablation()