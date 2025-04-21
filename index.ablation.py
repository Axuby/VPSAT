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
from my_update import AblationStudy


def run_single_parameter_ablation(param_path, values, base_config_path='./config/model_config.yaml',
                                  results_dir='./ablation_results', epochs=15):
    """
    function to run a single parameter ablation study.

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

    study.plot_results(experiment_name)

    # plot loss curves for the best configuration
    best_idx = np.argmin([r['best_val_loss'] for r in results])
    study.plot_loss_curves(experiment_name, best_idx)

    return results


def evaluate_angular_errors(model_path, config_path, output_dir):
    """
    evaluates a trained model and compute angular errors.
    returns: List of angular errors
    """
    
    config = load_config(config_path)
    C.update(config)
    M.update(C.model)

    
    device = torch.device(C.training.device if torch.cuda.is_available() else "cpu")
    model = VpSatNet(C).to(device)

    # load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    Dataset = WireframeDataset
    kwargs = {"batch_size": C.training.batch_size, "num_workers": C.io.num_workers, "pin_memory": True}
    val_loader = torch.utils.data.DataLoader(Dataset(C.io.datadir, split="test"), shuffle=False, **kwargs)

    angular_errors = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            original_image_size = images[0].shape[1:]

        
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
    runs a comprehensive ablation study on multiple parameters.
    """

    study = AblationStudy()

    # parameters to ablate
    ablation_params = [
        #        {
        #            'param_path': 'model.transformer.num_layers',
        #            'values': [2, 4, 6, 8]
        #        },
        #        {
        #            'param_path': 'model.transformer.num_heads',
        #            'values': [4, 8, 16]
        #        },
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
    for param_config in ablation_params:
        param_path = param_config['param_path']
        values = param_config['values']
        experiment_name = param_path.replace('.', '_')

        study.run_experiment(experiment_name, param_path, values, epochs=15)

        study.plot_results(experiment_name)

    # this compares best configs
    best_configs = []
    for param_config in ablation_params:
        param_path = param_config['param_path']
        experiment_name = param_path.replace('.', '_')

    
        with open(study.results_dir / f"{experiment_name}_results.json", 'r') as f:
            results = json.load(f)

       
        best_idx = np.argmin([r['best_val_loss'] for r in results])
        best_configs.append(results[best_idx])

    with open(study.results_dir / "best_configs.json", 'w') as f:
        json.dump(best_configs, f)

    return best_configs



# Run a simple ablation study on number of transformer layers

if __name__ == '__main__':
    run_comprehensive_ablation()
    # run_single_parameter_ablation(
    #     param_path='model.vp_head_type',
    #     values=['simple', 'hybrid'],
    #     epochs=15
    # )
    # run_single_parameter_ablation(
    #     param_path='model.use_fpn',
    #     values=[True, False],
    #     epochs=15
    # )
    # run_single_parameter_ablation(
    #     param_path='model.feature_extractor_type', 
    #     values=['cnn', 'resnet'], 
    #     epochs=15
    # )

# run_single_parameter_ablation(
#     param_path='model.transformer.num_layers',
#     values=[2, 4, 6, 8],
#     epochs=15
# )

# More comprehensive ablation
#run_comprehensive_ablation





# Evaluate ResNet model
# evaluate_model(
#     model_path="checkpoint/resnet_model/checkpoint_best.pth.tar",
#     config_path="config/resnet_config.yaml",
#     output_dir="evaluation_results/resnet",
#     visualize=True
# )

# Then you can use the visualizer to compare them

# from .viz import AdvancedVisualizer
# visualizer = AdvancedVisualizer(results_dir='evaluation_results')
# visualizer.plot_angular_error_comparison({
#     'CNN': 'evaluation_results/cnn/angular_errors.npy',
#     'ResNet': 'evaluation_results/resnet/angular_errors.npy'
# })
