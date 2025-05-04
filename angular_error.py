import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import json
import yaml

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


def evaluate_model(model_path, config_path, output_dir, visualize=True):
    """
    evaluates a trained vpd model.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load configuration
    try:
        # First load as plain dictionary to check
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
            
        # Then use your config loader
        config = load_config(config_path)
        
        # Check and update if needed
        if hasattr(config, 'update'):
            config.update(raw_config)
            
        # Ensure required sections exist
        if not hasattr(config, 'model'):
            config.model = raw_config.get('model', {})
        if not hasattr(config, 'io'):
            config.io = raw_config.get('io', {})
        if not hasattr(config, 'training'):
            config.training = raw_config.get('training', {})
            
        # Update global config
        C.update(config)
        M.update(config.model if hasattr(config, 'model') else raw_config.get('model', {}))
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        # Fallback to using raw dictionary
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Set up minimal required config
        if 'model' not in config:
            print("Warning: 'model' section missing from config")
            config['model'] = {}
        if 'io' not in config:
            config['io'] = {}
        if 'training' not in config:
            config['training'] = {}

    # Set up device
    device = torch.device(config.get('training', {}).get('device', 'cuda') if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

   
    model = VpSatNet(config).to(device)
    print("Model initialized")

    checkpoint = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {str(e)}")
        print("Attempting partial loading...")
        # Try partial loading
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                           if k in model_dict and model_dict[k].shape == v.shape}
        if len(pretrained_dict) > 0:
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
        else:
            print("No compatible parameters found. Using random initialization.")
            return None
    
    model.eval()

    # Create test dataset
    datadir = config.get('io', {}).get('datadir', 'data/su3/')
    batch_size = config.get('training', {}).get('batch_size', 16)
    num_workers = min(config.get('io', {}).get('num_workers', 6), 2)  # Limit workers for stability
    
    dataset = WireframeDataset(datadir, split="test")
    print(f"Number of test samples: {len(dataset)}")
    
    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
    test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, **kwargs)

    # Initialize arrays for results
    all_errors = []
    all_predictions = []
    all_targets = []

    # Evaluate model
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            original_image_size = images[0].shape[1:]
            
            # Process images
            input_image_size = config.get('model', {}).get('input_image_size', 256)
            patch_size = config.get('model', {}).get('patch_size', 32)
            
            images = preprocess_batch(images, input_image_size)
            patches = extract_patches(images, patch_size)
            b, n, c, h, w = patches.shape
            patches = patches.view(b * n, c, h, w).to(device)
            
            # IMPORTANT: Do NOT repeat targets for each patch
            # Use image-level targets
            vpts_3d = labels["vpts"].to(device)  # [batch_size, 3, 3]
            focal_length = config.get('io', {}).get('focal_length', 2.1875)
            vpts_2d = to_pixel(vpts_3d, focal_length, original_image_size[0])
            vpts_2d = adjust_vanishing_points(vpts_2d, original_image_size, input_image_size)
            
            # Forward pass
            outputs = model(patches)  # [batch_size, 3, 2]
            
            # Compute errors with correctly aligned targets
            errors = compute_angular_errors(outputs, vpts_2d)

            # Store results
            all_errors.append(errors.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(vpts_2d.cpu().numpy())

    # Concatenate results
    all_errors = np.concatenate(all_errors, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Flatten errors for overall statistics
    flattened_errors = all_errors.flatten()

    # Compute statistics
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
    
    # 1. Create histogram of all errors
    plt.figure(figsize=(12, 8))
    bins = np.linspace(0, min(90, np.max(flattened_errors) * 1.2), 36)
    hist = plt.hist(flattened_errors, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add vertical lines for statistics
    plt.axvline(np.mean(flattened_errors), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(flattened_errors):.2f}°')
    plt.axvline(np.median(flattened_errors), color='green', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(flattened_errors):.2f}°')
    plt.axvline(np.percentile(flattened_errors, 95), color='purple', linestyle='--', linewidth=2, 
               label=f'95th Percentile: {np.percentile(flattened_errors, 95):.2f}°')
    
    plt.xlabel('Angular Error (degrees)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Distribution of Angular Errors', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(fig_dir, 'error_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create CDF plot
    plt.figure(figsize=(12, 8))
    sorted_errors = np.sort(flattened_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    
    plt.plot(sorted_errors, cumulative, color='steelblue', linewidth=3)
    
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
    plt.close()
    
    # 3. Create box plots for different VP directions
    plt.figure(figsize=(12, 8))
    
    vp_data = [errors[:, i] for i in range(errors.shape[1])]
    labels = ['Vanishing Point 1', 'Vanishing Point 2', 'Vanishing Point 3']
    
    box = plt.boxplot(vp_data, labels=labels, patch_artist=True, notch=True, showfliers=False)
    
    # Customize box appearance
    colors = ['#5DA5DA', '#FAA43A', '#60BD68']
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(colors[i])
    
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(box[element], color='black')
    
    # Add individual points with jitter
    for i, data in enumerate(vp_data):
        # Sample points if too many
        sample_size = min(200, len(data))
        sample_idx = np.random.choice(len(data), sample_size, replace=False)
        sample_data = data[sample_idx]
        
        # Add jitter
        x = np.random.normal(i+1, 0.05, size=sample_size)
        plt.scatter(x, sample_data, alpha=0.4, s=20, color=colors[i], edgecolor='none')
    
    plt.xlabel('Vanishing Point Direction', fontsize=14)
    plt.ylabel('Angular Error (degrees)', fontsize=14)
    plt.title('Angular Errors by Vanishing Point Direction', fontsize=16, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(fig_dir, 'error_by_direction.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import sys
    import yaml
    
    if len(sys.argv) < 4:
        print("Usage: python angular_error_evaluation.py <model_path> <config_path> <output_dir>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    config_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    evaluate_model(model_path, config_path, output_dir)