import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import json
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap

from models.config import C, M
from models.vpsat import VpSatNet
from data.datasets import WireframeDataset
from utils.config_loader import load_config
from utils.helper_functions import preprocess_batch, extract_patches, to_pixel, adjust_vanishing_points




def compute_angular_errors(predictions, targets):
    """
    Compute angular errors between predicted and ground truth vanishing points.
    
    Args:
        predictions (torch.Tensor): Predicted VPs, shape [batch_size, 3, 2/3].
        targets (torch.Tensor): Ground-truth VPs, shape [batch_size, 3, 2/3].
        
    Returns:
        torch.Tensor: Angular errors in degrees, shape [batch_size, 3].
    """
    # Print warning if shapes don't match
    if predictions.shape[0] != targets.shape[0]:
        print(f"Warning: Batch size mismatch! Predictions: {predictions.shape}, Targets: {targets.shape}")
        
        # If targets are expanded for patches, take only the first batch_size entries
        if targets.shape[0] > predictions.shape[0]:
            targets = targets[:predictions.shape[0]]
        
        print(f"Adjusted shapes - Predictions: {predictions.shape}, Targets: {targets.shape}")
    
    # Normalize vectors to unit length (add a small z component if 2D)
    if predictions.shape[-1] == 2:
        # Convert 2D to 3D by adding a z component of 1
        pred_3d = torch.cat([predictions, torch.ones_like(predictions[:,:,:1])], dim=-1)
        targ_3d = torch.cat([targets, torch.ones_like(targets[:,:,:1])], dim=-1)
    else:
        pred_3d = predictions
        targ_3d = targets
    
    pred_3d = torch.nn.functional.normalize(pred_3d, p=2, dim=-1)
    targ_3d = torch.nn.functional.normalize(targ_3d, p=2, dim=-1)
    
    # Compute cosine similarity
    cosine_sim = torch.sum(pred_3d * targ_3d, dim=-1)  # Shape: [batch_size, 3]
    
    # Clamp values to avoid numerical issues with arccos
    cosine_sim = cosine_sim.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    
    # Convert to angles in degrees
    angles = torch.acos(cosine_sim) * (180.0 / np.pi)
    
    return angles
# def compute_angular_errors(predictions, targets):
#     """
#     Compute angular errors between predicted and ground truth vanishing points.

#     Args:
#         predictions (torch.Tensor): Predicted VPs, shape [batch_size, 3, 3].
#         targets (torch.Tensor): Ground-truth VPs, shape [batch_size, 3, 3].

#     Returns:
#         torch.Tensor: Angular errors in degrees, shape [batch_size, 3].
#     """
#     # Check if shapes match and handle if needed
#     if predictions.shape[0] != targets.shape[0]:
#         print(f"Warning: Batch size mismatch! Predictions: {predictions.shape}, Targets: {targets.shape}")
#         # Match sizes (use smaller batch size)
#         min_batch_size = min(predictions.shape[0], targets.shape[0])
#         predictions = predictions[:min_batch_size]
#         targets = targets[:min_batch_size]
#         print(f"Adjusted shapes - Predictions: {predictions.shape}, Targets: {targets.shape}")
    
#     # Normalize vectors to unit length
#     predictions = torch.nn.functional.normalize(predictions, p=2, dim=-1)
#     targets = torch.nn.functional.normalize(targets, p=2, dim=-1)

#     # Compute cosine similarity (removed duplicate line)
#     cosine_sim = torch.sum(predictions * targets, dim=-1)

#     # Clamp values to avoid numerical issues with arccos
#     cosine_sim = cosine_sim.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

#     # Convert to angles in degrees
#     angles = torch.acos(cosine_sim) * (180.0 / np.pi)

#     return angles


def evaluate_model_flexible(model_path, config_path, output_dir, visualize=True):
    """
    Evaluate a trained model with flexible architecture handling.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # First load the checkpoint to determine architecture
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Analyze checkpoint keys to determine architecture
    state_dict_keys = checkpoint['model_state_dict'].keys()
    
    # Detect feature extractor type
    if any('feature_extractor.backbone' in key for key in state_dict_keys):
        feature_extractor_type = 'resnet'
    elif any('feature_extractor.layer' in key for key in state_dict_keys):
        feature_extractor_type = 'cnn'
    else:
        feature_extractor_type = 'simple'
    
    # Detect VP head type
    if any('vp_head.attention' in key for key in state_dict_keys):
        vp_head_type = 'hybrid'
    elif any('vp_head.projection' in key for key in state_dict_keys):
        vp_head_type = 'simple'
    else:
        vp_head_type = 'simple'  # Default
    
    # Load and update the configuration
    config = load_config(config_path)
    if 'model' not in config:
        config['model'] = {}
    
    config['model']['feature_extractor_type'] = feature_extractor_type
    config['model']['vp_head_type'] = vp_head_type
    
    # Now initialize model with the correct architecture
    C.update(config)
    M.update(C.model)
    
    # Set up device
    device = torch.device(C.training.device if torch.cuda.is_available() else "cpu")
    
    # Initialize model with the matched architecture
    model = VpSatNet(C).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()




def evaluate_model(model_path, config_path, output_dir, visualize=True):
    """
    Evaluates a trained VPD model with architecture detection.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Load base config as a plain dictionary, not a Box object
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure required sections exist with default values
    if 'model' not in config:
        config['model'] = {}
    if 'io' not in config:
        config['io'] = {}
    if 'training' not in config:
        config['training'] = {}
    if 'optim' not in config:
        config['optim'] = {}
    
    # Set default values
    config['training']['device'] = config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    config['training']['batch_size'] = config['training'].get('batch_size', 16)
    config['io']['num_workers'] = config['io'].get('num_workers', 2)
    config['io']['focal_length'] = config['io'].get('focal_length', 2.1875)
    config['io']['num_vpts'] = config['io'].get('num_vpts', 3)
    config['io']['datadir'] = config['io'].get('datadir', 'data/su3/')
    
    # Detect feature extractor type
    if any('feature_extractor.conv1' in key for key in state_dict):
        config['model']['feature_extractor_type'] = 'simple'
    elif any('feature_extractor.layer1.0' in key and 'feature_extractor.layer1.0.conv1' not in key for key in state_dict):
        config['model']['feature_extractor_type'] = 'cnn'
    elif any('feature_extractor.backbone' in key for key in state_dict):
        config['model']['feature_extractor_type'] = 'resnet'
    
    # Detect VP head type
    if any('vp_head.projection' in key for key in state_dict):
        config['model']['vp_head_type'] = 'simple'
    elif any('vp_head.attention' in key for key in state_dict):
        config['model']['vp_head_type'] = 'hybrid'
    
    # Create a local copy of the C and M objects with our config
    from models.config import C as BaseC, M as BaseM
    local_C = BaseC.copy()
    local_M = BaseM.copy()
    local_C.update(config)
    local_M.update(config['model'])
    
    # Create model with the local config
    device = torch.device(local_C.training.device if torch.cuda.is_available() else "cpu")
    model = VpSatNet(local_C).to(device)
    
    # Load state dict
    try:
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model with matching architecture")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Try partial loading
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                        if k in model_dict and model_dict[k].shape == v.shape}
        if len(pretrained_dict) > 0:
            print(f"Performing partial loading: {len(pretrained_dict)}/{len(model_dict)} parameters")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
        else:
            print("No compatible parameters found. Using random initialization.")
            return None
    
    # Set model to evaluation mode
    model.eval()

    # Load test dataset
    dataset = WireframeDataset(local_C.io.datadir, split="test")
    kwargs = {"batch_size": local_C.training.batch_size, "num_workers": 2, "pin_memory": True}  # Reduced workers for stability
    test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, **kwargs)

    # Initialize arrays for results
    all_errors = []
    all_predictions = []
    all_targets = []

    # Rest of your evaluation code (unchanged)
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            original_image_size = images[0].shape[1:]
            
            # Process images
            images = preprocess_batch(images, local_C.model.input_image_size)
            patches = extract_patches(images, local_C.model.patch_size)
            b, n, c, h, w = patches.shape
            patches = patches.view(b * n, c, h, w).to(device)
            
            # Use image-level targets
            vpts_3d = labels["vpts"].to(device)
            vpts_2d = to_pixel(vpts_3d, local_C.io.focal_length, original_image_size[0])
            vpts_2d = adjust_vanishing_points(vpts_2d, original_image_size, local_C.model.input_image_size)
            
            # Forward pass
            outputs = model(patches)
            
            # Compute errors
            errors = compute_angular_errors(outputs, vpts_2d)

            # Store results
            all_errors.append(errors.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(vpts_2d.cpu().numpy())

    # Process and save results (unchanged)
    all_errors = np.concatenate(all_errors, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    flattened_errors = all_errors.flatten()

    stats = {
        'mean_error': float(np.mean(flattened_errors)),
        'median_error': float(np.median(flattened_errors)),
        'std_error': float(np.std(flattened_errors)),
        'min_error': float(np.min(flattened_errors)),
        'max_error': float(np.max(flattened_errors)),
        'p95_error': float(np.percentile(flattened_errors, 95)),
        'p99_error': float(np.percentile(flattened_errors, 99)),
        'success_rate_5deg': float(np.mean(flattened_errors < 5.0)),
        'success_rate_10deg': float(np.mean(flattened_errors < 10.0)),
        'success_rate_20deg': float(np.mean(flattened_errors < 20.0)),
    }

    # Save results
    np.save(os.path.join(output_dir, 'angular_errors.npy'), all_errors)
    np.save(os.path.join(output_dir, 'predictions.npy'), all_predictions)
    np.save(os.path.join(output_dir, 'targets.npy'), all_targets)

    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    # Print statistics
    print("\nEvaluation Results:")
    print(f"Mean Angular Error: {stats['mean_error']:.2f}°")
    print(f"Median Angular Error: {stats['median_error']:.2f}°")
    print(f"Standard Deviation: {stats['std_error']:.2f}°")
    print(f"95th Percentile Error: {stats['p95_error']:.2f}°")
    print(f"Success Rate (<5°): {stats['success_rate_5deg'] * 100:.2f}%")
    print(f"Success Rate (<10°): {stats['success_rate_10deg'] * 100:.2f}%")
    print(f"Success Rate (<20°): {stats['success_rate_20deg'] * 100:.2f}%")

    # Create visualizations
    if visualize:
        create_error_visualizations(all_errors, output_dir)

    return stats

# def evaluate_model(model_path, config_path, output_dir, visualize=True):
#     """
#     Evaluate a trained vanishing point detection model.
#     """
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # First determine model architecture from the checkpoint
#     checkpoint = torch.load(model_path, map_location='cpu')
#     checkpoint_keys = checkpoint['model_state_dict'].keys()
    
#     # Load base config
#     config = load_config(config_path)
    
#     # Check for feature extractor type
#     if any('feature_extractor.backbone' in k for k in checkpoint_keys):
#         config['model']['feature_extractor_type'] = 'resnet'
#     elif any('feature_extractor.layer' in k for k in checkpoint_keys):
#         config['model']['feature_extractor_type'] = 'cnn'
    
#     # Check for VP head type
#     if any('vp_head.projection' in k for k in checkpoint_keys):
#         config['model']['vp_head_type'] = 'simple'
#     elif any('vp_head.attention' in k for k in checkpoint_keys):
#         config['model']['vp_head_type'] = 'hybrid'
    
#     # Check for transformer num_layers by counting unique layer indices
#     if 'model' not in config:
#         config['model'] = {}
#     if 'transformer' not in config['model']:
#         config['model']['transformer'] = {}
    
#     # Update configuration
#     C.update(config)
#     M.update(C.model)
    
#     # Set up device
#     device = torch.device(C.training.device if torch.cuda.is_available() else "cpu")
    
#     # Initialize model with the matching architecture
#     model = VpSatNet(C).to(device)
    
#     # Now the model should match the checkpoint architecture
#     try:
#         model.load_state_dict(checkpoint["model_state_dict"])
#         print(f"Successfully loaded all parameters from {model_path}")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None
    
#     model.eval()
    
#     # Set up data loader
#     dataset = WireframeDataset(C.io.datadir, split="test")
#     kwargs = {"batch_size": C.training.batch_size, "num_workers": 2, "pin_memory": True}
#     test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, **kwargs)
    
#     print(f"Number of test samples: {len(dataset)}")
    
#     # Initialize arrays for results
#     all_errors = []
#     all_predictions = []
#     all_targets = []
    
#     # Evaluate model
#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, desc="Evaluating"):
#             original_image_size = images[0].shape[1:]
            
#             # Preprocess images
#             images = preprocess_batch(images, C.model.input_image_size)
#             patches = extract_patches(images, C.model.patch_size)
#             b, n, c, h, w = patches.shape
#             patches = patches.view(b * n, c, h, w).to(device)
            
#             # Get ground truth vanishing points (don't repeat for each patch)
#             vpts_3d = labels["vpts"].to(device)
#             vpts_2d = to_pixel(vpts_3d, C.io.focal_length, original_image_size[0])
#             vpts_2d = adjust_vanishing_points(vpts_2d, original_image_size, C.model.input_image_size)
            
#             # Forward pass
#             outputs = model(patches)
            
#             # Compute angular errors
#             errors = compute_angular_errors(outputs, vpts_2d)
            
#             # Store results
#             all_errors.append(errors.cpu().numpy())
#             all_predictions.append(outputs.cpu().numpy())
#             all_targets.append(vpts_2d.cpu().numpy())

#     # Flatten errors for overall statistics
#     flattened_errors = all_errors.flatten()

#     # Compute statistics
#     stats = {
#         'mean_error': float(np.mean(flattened_errors)),
#         'median_error': float(np.median(flattened_errors)),
#         'std_error': float(np.std(flattened_errors)),
#         'min_error': float(np.min(flattened_errors)),
#         'max_error': float(np.max(flattened_errors)),
#         'p95_error': float(np.percentile(flattened_errors, 95)),
#         'p99_error': float(np.percentile(flattened_errors, 99)),
#         'success_rate_5deg': float(np.mean(flattened_errors < 5.0)),
#         'success_rate_10deg': float(np.mean(flattened_errors < 10.0)),
#         'success_rate_20deg': float(np.mean(flattened_errors < 20.0)),
#     }

#     # Save results
#     np.save(os.path.join(output_dir, 'angular_errors.npy'), all_errors)
#     np.save(os.path.join(output_dir, 'predictions.npy'), all_predictions)
#     np.save(os.path.join(output_dir, 'targets.npy'), all_targets)

#     with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
#         json.dump(stats, f, indent=4)

#     # Print statistics
#     print("\nEvaluation Results:")
#     print(f"Mean Angular Error: {stats['mean_error']:.2f}°")
#     print(f"Median Angular Error: {stats['median_error']:.2f}°")
#     print(f"Standard Deviation: {stats['std_error']:.2f}°")
#     print(f"95th Percentile Error: {stats['p95_error']:.2f}°")
#     print(f"Success Rate (<5°): {stats['success_rate_5deg'] * 100:.2f}%")
#     print(f"Success Rate (<10°): {stats['success_rate_10deg'] * 100:.2f}%")
#     print(f"Success Rate (<20°): {stats['success_rate_20deg'] * 100:.2f}%")

#     # Create visualizations
#     if visualize:
#         create_error_visualizations(all_errors, output_dir)

#     return stats


def create_error_visualizations(errors, output_dir):
    """
    Create visualizations of angular errors.

    Args:
        errors (numpy.ndarray): Angular errors array [N, 3] for x, y, z directions
        output_dir (str): Directory to save visualizations
    """
    # Flatten errors for overall statistics
    flattened_errors = errors.flatten()

    # Create directory for figures
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Set up visualization style
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("viridis", 4)

    # 1. Create histogram of all errors
    plt.figure(figsize=(12, 8))
    bins = np.linspace(0, min(90, np.max(flattened_errors) * 1.2), 36)
    hist = plt.hist(flattened_errors, bins=bins, alpha=0.7, color=palette[0], edgecolor='black')

    # Add vertical lines for statistics
    plt.axvline(np.mean(flattened_errors), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(flattened_errors):.2f}°')
    plt.axvline(np.median(flattened_errors), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(flattened_errors):.2f}°')
    plt.axvline(np.percentile(flattened_errors, 95), color='blue', linestyle='--', linewidth=2,
                label=f'95th Percentile: {np.percentile(flattened_errors, 95):.2f}°')

    plt.xlabel('Angular Error (degrees)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Distribution of Angular Errors', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(fig_dir, 'error_histogram.png'), dpi=300, bbox_inches='tight')

    # 2. Create CDF plot
    plt.figure(figsize=(12, 8))
    sorted_errors = np.sort(flattened_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

    plt.plot(sorted_errors, cumulative, color=palette[2], linewidth=3)

    # Add reference thresholds
    thresholds = [5, 10, 20, 30]
    for threshold in thresholds:
        success_rate = np.mean(flattened_errors < threshold) * 100
        plt.axvline(threshold, color='gray', linestyle='--', alpha=0.7)
        plt.text(threshold + 1, 10, f'{threshold}°: {success_rate:.1f}%', fontsize=12)

    plt.xlabel('Angular Error (degrees)', fontsize=14)
    plt.ylabel('Cumulative Percentage (%)', fontsize=14)
    plt.title('Cumulative Distribution of Angular Errors', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, min(90, np.max(flattened_errors) * 1.2))

    plt.savefig(os.path.join(fig_dir, 'error_cdf.png'), dpi=300, bbox_inches='tight')

    # 3. Create box plots for different VP directions
    plt.figure(figsize=(12, 8))

    vp_data = [errors[:, i] for i in range(errors.shape[1])]
    labels = ['Vanishing Point 1', 'Vanishing Point 2', 'Vanishing Point 3']

    box = plt.boxplot(vp_data, labels=labels, patch_artist=True, notch=True, showfliers=False)

    # Customize box appearance
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(palette[i])

    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(box[element], color='black')

    # Add individual points with jitter
    for i, data in enumerate(vp_data):
        # Sample points if too many
        sample_size = min(200, len(data))
        sample_idx = np.random.choice(len(data), sample_size, replace=False)
        sample_data = data[sample_idx]

        # Add jitter
        x = np.random.normal(i + 1, 0.05, size=sample_size)
        plt.scatter(x, sample_data, alpha=0.4, s=20, c=palette[i], edgecolor='none')

    plt.xlabel('Vanishing Point Direction', fontsize=14)
    plt.ylabel('Angular Error (degrees)', fontsize=14)
    plt.title('Angular Errors by Vanishing Point Direction', fontsize=16, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)

    plt.savefig(os.path.join(fig_dir, 'error_by_direction.png'), dpi=300, bbox_inches='tight')

    # 4. Create a 2D histogram/heatmap of errors
    plt.figure(figsize=(10, 8))

    # Compute 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        errors[:, 0], errors[:, 1],
        bins=[np.linspace(0, 50, 25), np.linspace(0, 50, 25)]
    )

    # Create heatmap
    plt.imshow(heatmap.T, cmap='viridis', origin='lower',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               aspect='auto')

    plt.colorbar(label='Count')
    plt.xlabel('VP1 Angular Error (degrees)', fontsize=14)
    plt.ylabel('VP2 Angular Error (degrees)', fontsize=14)
    plt.title('2D Distribution of Angular Errors', fontsize=16, fontweight='bold')

    plt.savefig(os.path.join(fig_dir, 'error_2d_heatmap.png'), dpi=300, bbox_inches='tight')


# def compare_models(model_configs, output_dir):
#     """
#     Compare multiple model configurations.

#     Args:
#         model_configs (dict): Dictionary mapping model names to (model_path, config_path)
#         output_dir (str): Directory to save comparison results
#     """
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)

#     # Evaluate each model
#     results = {}
#     all_errors = {}

#     for model_name, (model_path, config_path) in model_configs.items():
#         print(f"\nEvaluating model: {model_name}")

#         # Create subdirectory for this model
#         model_dir = os.path.join(output_dir, model_name)
#         os.makedirs(model_dir, exist_ok=True)

#         # Evaluate model
#         stats = evaluate_model(model_path, config_path, model_dir)
#         results[model_name] = stats

#         # Load errors for comparison
#         all_errors[model_name] = np.load(os.path.join(model_dir, 'angular_errors.npy')).flatten()

#     # Save combined results
#     with open(os.path.join(output_dir, 'comparison.json'), 'w') as f:
#         json.dump(results, f, indent=4)

#     # Create comparison visualizations
#     create_comparison_visualizations(results, all_errors, output_dir)

#     return results

def ensure_config_has_required_fields(config):
    """Make sure config has all required fields with default values."""
    if 'model' not in config:
        config['model'] = {}
    if 'io' not in config:
        config['io'] = {}
    if 'training' not in config:
        config['training'] = {}
    if 'optim' not in config:
        config['optim'] = {}
    
    # Add default values
    config['training']['device'] = config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    config['training']['batch_size'] = config['training'].get('batch_size', 16)
    
    config['io']['num_workers'] = config['io'].get('num_workers', 2)
    config['io']['focal_length'] = config['io'].get('focal_length', 2.1875)
    config['io']['num_vpts'] = config['io'].get('num_vpts', 3)
    config['io']['datadir'] = config['io'].get('datadir', 'data/su3/')
    
    return config


def compare_models(model_configs, output_dir):
    """
    Compare multiple model configurations with robust error handling.

    Args:
        model_configs (dict): Dictionary mapping model names to (model_path, config_path)
        output_dir (str): Directory to save comparison results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate each model
    results = {}
    all_errors = {}

    for model_name, (model_path, config_path) in model_configs.items():
        print(f"\nEvaluating model: {model_name}")

        # Create subdirectory for this model
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        try:
            # Load and validate config
            config = load_config(config_path)
            config = ensure_config_has_required_fields(config)
            
            # Save the validated config
            validated_config_path = os.path.join(output_dir, f"{model_name}_validated_config.yaml")
            with open(validated_config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Evaluate model with validated config
            stats = evaluate_model(model_path, validated_config_path, model_dir)
            results[model_name] = stats

            # Load errors for comparison
            error_path = os.path.join(model_dir, 'angular_errors.npy')
            if os.path.exists(error_path):
                all_errors[model_name] = np.load(error_path).flatten()
            else:
                print(f"Warning: Angular error file not found at {error_path}")
        except Exception as e:
            print(f"Error evaluating model {model_name}: {str(e)}")

    # Save combined results if we have any
    if results:
        with open(os.path.join(output_dir, 'comparison.json'), 'w') as f:
            json.dump(results, f, indent=4)

    # Create comparison visualizations if we have at least two successful evaluations
    if len(all_errors) >= 2:
        try:
            create_comparison_visualizations(results, all_errors, output_dir)
        except Exception as e:
            print(f"Error creating comparison visualizations: {str(e)}")
    elif all_errors:
        print(f"Only one model evaluated successfully, skipping comparison visualizations")
    else:
        print(f"No models evaluated successfully, cannot create comparison visualizations")

    return results


def create_comparison_visualizations(results, all_errors, output_dir):
    """
    Create visualizations comparing multiple models.

    Args:
        results (dict): Dictionary with evaluation statistics for each model
        all_errors (dict): Dictionary with error arrays for each model
        output_dir (str): Directory to save visualizations
    """
    # Create directory for figures
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Set up visualization style
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("viridis", len(results))

    # 1. Bar chart comparing key metrics
    plt.figure(figsize=(14, 10))

    # Prepare data
    model_names = list(results.keys())
    metrics = ['mean_error', 'median_error', 'p95_error']
    metric_names = ['Mean Error', 'Median Error', '95th Percentile']

    x = np.arange(len(model_names))
    width = 0.25

    # Create grouped bar chart
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        plt.bar(x + i * width - width, values, width, label=metric_names[i],
                color=palette[i % len(palette)], edgecolor='black')

        # Add value labels
        for j, value in enumerate(values):
            plt.text(x[j] + i * width - width, value + 0.5, f'{value:.2f}°',
                     ha='center', va='bottom', fontsize=10)

    plt.xlabel('Model Configuration', fontsize=14)
    plt.ylabel('Angular Error (degrees)', fontsize=14)
    plt.title('Comparison of Angular Error Metrics', fontsize=16, fontweight='bold')
    plt.xticks(x, model_names, rotation=30, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'metric_comparison.png'), dpi=300, bbox_inches='tight')

    # 2. Success rate comparison
    plt.figure(figsize=(14, 10))

    # Prepare data
    thresholds = ['success_rate_5deg', 'success_rate_10deg', 'success_rate_20deg']
    threshold_names = ['< 5°', '< 10°', '< 20°']

    # Create grouped bar chart
    for i, threshold in enumerate(thresholds):
        values = [results[model][threshold] * 100 for model in model_names]
        plt.bar(x + i * width - width, values, width, label=threshold_names[i],
                color=palette[i % len(palette)], edgecolor='black')

        # Add value labels
        for j, value in enumerate(values):
            plt.text(x[j] + i * width - width, value + 1, f'{value:.1f}%',
                     ha='center', va='bottom', fontsize=10)

    plt.xlabel('Model Configuration', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.title('Comparison of Success Rates', fontsize=16, fontweight='bold')
    plt.xticks(x, model_names, rotation=30, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'success_rate_comparison.png'), dpi=300, bbox_inches='tight')

    # 3. CDF comparison
    plt.figure(figsize=(14, 10))

    # Plot CDF for each model
    for i, (model_name, errors) in enumerate(all_errors.items()):
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        plt.plot(sorted_errors, cumulative, linewidth=3, label=model_name, color=palette[i])

    # Add reference thresholds
    thresholds = [5, 10, 20]
    for threshold in thresholds:
        plt.axvline(threshold, color='gray', linestyle='--', alpha=0.5)
        plt.text(threshold + 0.5, 20, f'{threshold}°', fontsize=12, ha='left')

    plt.xlabel('Angular Error (degrees)', fontsize=14)
    plt.ylabel('Cumulative Percentage (%)', fontsize=14)
    plt.title('Cumulative Error Distribution Comparison', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'cdf_comparison.png'), dpi=300, bbox_inches='tight')

    # 4. Box plot comparison
    plt.figure(figsize=(14, 10))

    # Prepare data for box plot
    data = [errors for errors in all_errors.values()]

    # Create box plot
    box = plt.boxplot(data, labels=model_names, patch_artist=True, notch=True, showfliers=False)

    # Customize box appearance
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(palette[i])

    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(box[element], color='black')

    # Add violin plots
    violin_parts = plt.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(palette[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.3)

    plt.xlabel('Model Configuration', fontsize=14)
    plt.ylabel('Angular Error (degrees)', fontsize=14)
    plt.title('Distribution of Angular Errors Across Models', fontsize=16, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'boxplot_comparison.png'), dpi=300, bbox_inches='tight')


def compare_parameters(param_name, param_values, config_path, model_paths, output_dir):
    """
    Compare performance across different values of a specific parameter.

    Args:
        param_name (str): Name of the parameter being varied
        param_values (list): List of parameter values
        config_path (str): Base configuration path
        model_paths (list): List of model checkpoint paths
        output_dir (str): Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load base configuration
    base_config = load_config(config_path)

    # Initialize dictionaries for results
    stats = {}
    errors = {}

    # Evaluate models for each parameter value
    for value, model_path in zip(param_values, model_paths):
        print(f"\nEvaluating {param_name} = {value}")

        # Create parameter-specific config
        config = base_config.copy()

        # Update config with parameter value
        keys = param_name.split('.')
        current = config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value

        # Create subdirectory for this parameter value
        param_dir = os.path.join(output_dir, f"{param_name.replace('.', '_')}_{value}")
        os.makedirs(param_dir, exist_ok=True)

        # Save config
        with open(os.path.join(param_dir, 'config.yaml'), 'w') as f:
            json.dump(config, f, indent=4)

        # Evaluate model
        stats[value] = evaluate_model(model_path, config_path, param_dir)

        # Load errors for comparison
        errors[value] = np.load(os.path.join(param_dir, 'angular_errors.npy')).flatten()

    # Create visualizations
    create_parameter_comparison_visualizations(param_name, param_values, stats, errors, output_dir)

    return stats





def create_parameter_comparison_visualizations(param_name, param_values, stats, errors, output_dir):
    """
    Create visualizations comparing different values of a parameter.

    Args:
        param_name (str): Name of the parameter
        param_values (list): List of parameter values
        stats (dict): Dictionary of statistics for each parameter value
        errors (dict): Dictionary of error arrays for each parameter value
        output_dir (str): Directory to save visualizations
    """
    # Create directory for figures
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Set up visualization style
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("viridis", len(param_values))

    # Format parameter name for display
    param_display = param_name.split('.')[-1].replace('_', ' ').title()

    # 1. Line plot of key metrics
    plt.figure(figsize=(14, 8))

    # Prepare data
    metrics = ['mean_error', 'median_error', 'p95_error']
    metric_names = ['Mean Error', 'Median Error', '95th Percentile']

    # Create line plot
    for i, metric in enumerate(metrics):
        values = [stats[value][metric] for value in param_values]
        plt.plot(param_values, values, 'o-', linewidth=2, markersize=8, label=metric_names[i], color=palette[i])

        # Add value labels
        for j, value in enumerate(values):
            plt.text(param_values[j], value + 0.2, f'{value:.2f}°', ha='center', va='bottom', fontsize=10)

    plt.xlabel(param_display, fontsize=14)
    plt.ylabel('Angular Error (degrees)', fontsize=14)
    plt.title(f'Effect of {param_display} on Angular Error', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'{param_name.replace(".", "_")}_metrics.png'), dpi=300, bbox_inches='tight')

    # 2. Line plot of success rates
    plt.figure(figsize=(14, 8))

    # Prepare data
    thresholds = ['success_rate_5deg', 'success_rate_10deg', 'success_rate_20deg']
    threshold_names = ['< 5°', '< 10°', '< 20°']

    # Create line plot
    for i, threshold in enumerate(thresholds):
        values = [stats[value][threshold] * 100 for value in param_values]
        plt.plot(param_values, values, 'o-', linewidth=2, markersize=8, label=threshold_names[i], color=palette[i])

        # Add value labels
        for j, value in enumerate(values):
            plt.text(param_values[j], value + 0.5, f'{value:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.xlabel(param_display, fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.title(f'Effect of {param_display} on Success Rate', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'{param_name.replace(".", "_")}_success_rates.png'), dpi=300,
                bbox_inches='tight')

    # 3. CDF comparison
    plt.figure(figsize=(14, 8))

    # Plot CDF for each parameter value
    for i, value in enumerate(param_values):
        sorted_errors = np.sort(errors[value])
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        plt.plot(sorted_errors, cumulative, linewidth=2, label=f'{param_display} = {value}', color=palette[i])

    plt.xlabel('Angular Error (degrees)', fontsize=14)
    plt.ylabel('Cumulative Percentage (%)', fontsize=14)
    plt.title(f'Effect of {param_display} on Error Distribution', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'{param_name.replace(".", "_")}_cdf.png'), dpi=300, bbox_inches='tight')

    # 4. Box plot comparison
    plt.figure(figsize=(14, 8))

    # Prepare data for box plot
    data = [errors[value] for value in param_values]

    # Create box plot
    box = plt.boxplot(data, labels=[str(v) for v in param_values], patch_artist=True, notch=True, showfliers=False)

    # Customize box appearance
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(palette[i])

    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(box[element], color='black')

    plt.xlabel(param_display, fontsize=14)
    plt.ylabel('Angular Error (degrees)', fontsize=14)
    plt.title(f'Distribution of Angular Errors by {param_display}', fontsize=16, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'{param_name.replace(".", "_")}_boxplot.png'), dpi=300, bbox_inches='tight')








if __name__ == "__main__":

    # evaluate_model(
    #     model_path="checkpoint_1/baseline/best_model_epoch_5.pth",
    #     config_path="config/model_config.yaml",
    #     output_dir="evaluation_results/baseline",
    #     visualize=True
    # )
    # evaluate_model(
    #     model_path="ablation_results/model_feature_extractor_type/model_feature_extractor_type_resnet/checkpoint_best.pth.tar",
    #     config_path="ablation_results/model_feature_extractor_type/model_feature_extractor_type_resnet/config.yaml",
    #     output_dir="evaluation_results/model_feature_extractor_type_resnet",
    #     visualize=True
    # )
    # evaluate_model(
    #     model_path="ablation_results/model_feature_extractor_type/model_feature_extractor_type_cnn/checkpoint_best.pth.tar",
    #     config_path="ablation_results/model_feature_extractor_type/model_feature_extractor_type_cnn/config.yaml",
    #     output_dir="evaluation_results/model_feature_extractor_type_cnn",
    #     visualize=True
    # )
    # evaluate_model(
    #     model_path="ablation_results/model_vp_head_type/model_vp_head_type_simple/checkpoint_best.pth.tar",
    #     config_path="ablation_results/model_vp_head_type/model_vp_head_type_simple/config.yaml",
    #     output_dir="evaluation_results/model_vp_head_type/model_vp_head_type_simple",
    #     visualize=True
    # )
    import sys
    import yaml
    
    if len(sys.argv) < 4:
        print("Usage: python angular_error_evaluation.py <model_path> <config_path> <output_dir>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    config_path = sys.argv[2]
    output_dir = sys.argv[3]
    evaluate_model(model_path, config_path, output_dir)

    # model_configs = {
    #     "num_layer_2": ("ablation_results/s_2/ablation_results/model_transformer_num_layers/model_transformer_num_layers_2/checkpoint_best.pth.tar", "ablation_results/s_2/ablation_results/model_transformer_num_layers/model_transformer_num_layers_2/config.yaml"),
    #     "num_layer_4": ("ablation_results/s_2/ablation_results/model_transformer_num_layers/model_transformer_num_layers_4/checkpoint_best.pth.tar", "ablation_results/s_2/ablation_results/model_transformer_num_layers/model_transformer_num_layers_4/config.yaml"),
    #     "num_layer_6": ("ablation_results/s_2/ablation_results/model_transformer_num_layers/model_transformer_num_layers_6/checkpoint_best.pth.tar", "ablation_results/s_2/ablation_results/model_transformer_num_layers/model_transformer_num_layers_6/config.yaml"),
    #     "num_layer_8": ("ablation_results/s_2/ablation_results/model_transformer_num_layers/model_transformer_num_layers_8/checkpoint_best.pth.tar", "ablation_results/s_2/ablation_results/model_transformer_num_layers/model_transformer_num_layers_8/config.yaml"),
    #     # "Transformer-6L": ("checkpoint/transformer_6l/checkpoint_best.pth.tar", "config/transformer_6l.yaml"),
    #     # "Transformer-4L": ("checkpoint/transformer_4l/checkpoint_best.pth.tar", "config/transformer_4l.yaml"),
    # }

    # compare_models(model_configs, "evaluation_results/model_transformer_num_layers/model_comparison")

    # evaluate_model(
    #     model_path="ablation_results/training_loss_type/training_loss_type_mse/checkpoint_latest.pth.tar",
    #     config_path="ablation_results/training_loss_type/training_loss_type_mse/config.yaml",
    #     output_dir="evaluation_results/training_loss_type/training_loss_type_simple_mse",
    #     visualize=True
    # )
    # evaluate_model(
    #     model_path="ablation_results/training_loss_type/training_loss_type_cosine/checkpoint_latest.pth.tar",
    #     config_path="ablation_results/training_loss_type/training_loss_type_cosine/config.yaml",
    #     output_dir="evaluation_results/training_loss_type/training_loss_type_simple_cosine",
    #     visualize=True
    # )

    # evaluate_model(
    #     model_path="ablation_results/training_loss_type/training_loss_type_combined/checkpoint_latest.pth.tar",
    #     config_path="ablation_results/training_loss_type/training_loss_type_combined/config.yaml",
    #     output_dir="evaluation_results/training_loss_type/training_loss_type_simple_combined",
    #     visualize=True
    # )
    #evaluate_model(
    #    model_path="ablation_results/model_vp_head_type/model_vp_head_type_hybrid/checkpoint_best.pth.tar",
    #    config_path="ablation_results/model_vp_head_type/model_vp_head_type_hybrid/config.yaml",
    #    output_dir="evaluation_results/model_vp_head_type_hybrid",
    #    visualize=True
    #    )

    #evaluate_model(
    #    model_path="ablation_results/model_vp_head_type/model_vp_head_type_simple/checkpoint_best.pth.tar",
    #    config_path="ablation_results/model_vp_head_type/model_vp_head_type_simple/config.yaml",
    #    output_dir="evaluation_results/model_vp_head_type_simple",
    #    visualize=True
    #)

    # model_configs = {
    #     "ResNet": ("ablation_results/model_feature_extractor_type/model_feature_extractor_type_resnet/checkpoint_best.pth.tar", "ablation_results/model_feature_extractor_type/model_feature_extractor_type_resnet/config.yaml"),
    #     "CNN": ("ablation_results/model_feature_extractor_type/model_feature_extractor_type_cnn/checkpoint_best.pth.tar", "ablation_results/model_feature_extractor_type/model_feature_extractor_type_cnn/config.yaml"),
    #     # "Transformer-6L": ("checkpoint/transformer_6l/checkpoint_best.pth.tar", "config/transformer_6l.yaml"),
    #     # "Transformer-4L": ("checkpoint/transformer_4l/checkpoint_best.pth.tar", "config/transformer_4l.yaml"),
    # }
    #
    # compare_models(model_configs, "evaluation_results/model_comparison")

    # # 3. Compare different values of a parameter
    # compare_parameters(
    #     param_name="model.transformer.num_layers",
    #     param_values=[2, 4, 6, 8],
    #     config_path="config/model_config.yaml",
    #     model_paths=[
    #         "checkpoint/layers_2/checkpoint_best.pth.tar",
    #         "checkpoint/layers_4/checkpoint_best.pth.tar",
    #         "checkpoint/layers_6/checkpoint_best.pth.tar",
    #         "checkpoint/layers_8/checkpoint_best.pth.tar",
    #     ],
    #     output_dir="evaluation_results/num_layers_comparison"
    # )
