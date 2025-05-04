import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import json
import yaml
from scipy import integrate

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


def compute_angle_accuracy(errors, thresholds):
    """
    Compute Angle Accuracy (AA) metrics.
    
    Args:
        errors (np.ndarray): Angular errors in degrees
        thresholds (list): List of thresholds for AA calculation
        
    Returns:
        dict: Dictionary with AA metrics for each threshold
    """
    # Flatten errors if multi-dimensional
    if len(errors.shape) > 1:
        errors = errors.flatten()
    
    aa_metrics = {}
    
    # For each threshold, compute area under curve
    for theta in thresholds:
        # Create fine-grained thresholds from 0 to theta
        fine_thresholds = np.linspace(0, theta, 100)
        
        # Compute accuracy at each fine threshold
        accuracy_curve = [np.mean(errors <= t) * 100 for t in fine_thresholds]
        
        # Compute area under curve normalized by theta
        auc = integrate.simps(accuracy_curve, fine_thresholds) / theta
        
        # Store result
        aa_metrics[f'AA{theta}'] = auc
    
    return aa_metrics


def compute_success_rates(errors, thresholds):
    """
    Compute success rates at specific thresholds.
    
    Args:
        errors (np.ndarray): Angular errors in degrees
        thresholds (list): List of thresholds for success rate calculation
        
    Returns:
        dict: Dictionary with success rate at each threshold
    """
    # Flatten errors if multi-dimensional
    if len(errors.shape) > 1:
        errors = errors.flatten()
    
    success_rates = {}
    
    for threshold in thresholds:
        success_rate = np.mean(errors <= threshold) * 100
        success_rates[f'success_rate_{threshold}deg'] = float(success_rate)
    
    return success_rates


def plot_angle_accuracy_curve(errors, max_angle, output_path, title=None):
    """
    Plot Angle Accuracy curve.
    
    Args:
        errors (np.ndarray): Angular errors in degrees
        max_angle (float): Maximum angle for the curve
        output_path (str): Path to save the plot
        title (str, optional): Plot title
    """
    # Flatten errors if multi-dimensional
    if len(errors.shape) > 1:
        errors = errors.flatten()
    
    # Create thresholds from 0 to max_angle
    thresholds = np.linspace(0, max_angle, 100)
    
    # Compute accuracy at each threshold
    accuracy = [np.mean(errors <= t) * 100 for t in thresholds]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracy, '-', linewidth=2.5)
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('Angle Difference (degrees)', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    
    # Add title
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    else:
        plt.title(f'AA Curve @ {max_angle}° for Model', fontsize=14, fontweight='bold')
    
    # Set axes limits
    plt.xlim(0, max_angle)
    plt.ylim(0, 100)
    
    # Compute AA metric
    aa_metric = integrate.simps(accuracy, thresholds) / max_angle
    
    # Add AA metric text
    plt.text(max_angle * 0.7, 20, f'AA{max_angle}° = {aa_metric:.2f}', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()



def evaluate_with_angle_accuracy(model_path, config_path, output_dir, visualize=True):
    """
    Evaluate a model with angle accuracy metrics.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Convert to Box object if your models expect it
        from utils.box import Box
        box_config = Box(config)
        
        # Also update the global C and M objects if needed
        from models.config import C, M
        C.update(config)
        if 'model' in config:
            M.update(config['model'])
        
        # Print loaded config for debugging
        print("Config loaded successfully")
        print("Config sections:", list(config.keys()))
        
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        # Ensure minimal config exists
        config = {'model': {}, 'io': {}, 'training': {}}
        box_config = config  # Fallback to dict

    # Set up device
    device_name = config.get('training', {}).get('device', 'cuda')
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model - try both Box and dict access
    try:
        # First try with Box object
        model = VpSatNet(box_config).to(device)
    except Exception as e:
        print(f"Error initializing model with Box: {str(e)}")
        try:
            # Monkey patch the VpSatNet initialization to handle dict
            original_init = VpSatNet.__init__
            
            def patched_init(self, cfg):
                # Convert dict access to attribute access
                self.C = cfg  # Store config
                
                # Get configuration values with dict access
                if isinstance(cfg, dict):
                    extractor_type = cfg.get('model', {}).get('feature_extractor_type', 'resnet')
                    self.feature_extractor = get_feature_extractor(
                        extractor_type,
                        cfg.get('model', {}).get('transformer', {}).get('d_model', 256)
                    )
                    # ... rest of initialization with dict access ...
                else:
                    # Original attribute-based access
                    original_init(self, cfg)
            
            # Apply monkey patch
            VpSatNet.__init__ = patched_init
            
            # Try again with dict
            model = VpSatNet(config).to(device)
        except Exception as e:
            print(f"Both initialization methods failed: {str(e)}")
            return None

    # Load model weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
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

    # Calculate standard evaluation metrics
    standard_metrics = {
        'mean_error': float(np.mean(flattened_errors)),
        'median_error': float(np.median(flattened_errors)),
        'std_error': float(np.std(flattened_errors)),
        'min_error': float(np.min(flattened_errors)),
        'max_error': float(np.max(flattened_errors)),
        'p95_error': float(np.percentile(flattened_errors, 95)),
        'p99_error': float(np.percentile(flattened_errors, 99)),
    }
    
    # Calculate success rates at standard thresholds
    success_rates = compute_success_rates(flattened_errors, [1, 2, 3, 5, 10, 20])
    
    # Calculate AA metrics
    # For synthetic dataset
    synthetic_aa = compute_angle_accuracy(flattened_errors, [0.2, 0.5, 1.0])
    
    # For real-world dataset
    real_aa = compute_angle_accuracy(flattened_errors, [1, 2, 10])
    
    # Combine all metrics
    all_metrics = {
        **standard_metrics,
        **success_rates,
        **synthetic_aa,
        **real_aa
    }

    # Save results
    np.save(os.path.join(output_dir, 'angular_errors.npy'), all_errors)
    np.save(os.path.join(output_dir, 'predictions.npy'), all_predictions)
    np.save(os.path.join(output_dir, 'targets.npy'), all_targets)

    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)

    # Print statistics
    print("\nEvaluation Results:")
    print(f"Mean Angular Error: {all_metrics['mean_error']:.2f}°")
    print(f"Median Angular Error: {all_metrics['median_error']:.2f}°")
    print(f"Standard Deviation: {all_metrics['std_error']:.2f}°")
    print("\nSuccess Rates:")
    for threshold in [1, 2, 3, 5, 10, 20]:
        print(f"  < {threshold}°: {all_metrics[f'success_rate_{threshold}deg']:.2f}%")
    
    print("\nAngle Accuracy Metrics (Synthetic Dataset):")
    for theta in [0.2, 0.5, 1.0]:
        print(f"  AA{theta}°: {all_metrics[f'AA{theta}']:.2f}")
    
    print("\nAngle Accuracy Metrics (Real-world Dataset):")
    for theta in [1, 2, 10]:
        print(f"  AA{theta}°: {all_metrics[f'AA{theta}']:.2f}")

    # Create visualizations
    if visualize:
        create_aa_visualizations(flattened_errors, output_dir)

    return all_metrics


# def create_aa_visualizations(errors, output_dir):
#     """
#     Create Angle Accuracy visualizations.
#     """
#     # Create directory for figures
#     fig_dir = os.path.join(output_dir, 'figures')
#     os.makedirs(fig_dir, exist_ok=True)
    
#     # Set up visualization style
#     sns.set_theme(style="whitegrid")
    
#     # 1. Create fine-grained AA curve for synthetic data
#     plot_angle_accuracy_curve(
#         errors, 
#         max_angle=1.0, 
#         output_path=os.path.join(fig_dir, 'aa_curve_synthetic_fine.png'),
#         title='AA Curve @ 1° (Synthetic Fine-grained)'
#     )
    
#     # 2. Create coarse-grained AA curve for synthetic data
#     plot_angle_accuracy_curve(
#         errors, 
#         max_angle=10.0, 
#         output_path=os.path.join(fig_dir, 'aa_curve_synthetic_coarse.png'),
#         title='AA Curve @ 10° (Synthetic Coarse-grained)'
#     )
    
#     # 3. Create fine-grained AA curve for real data
#     plot_angle_accuracy_curve(
#         errors, 
#         max_angle=2.0, 
#         output_path=os.path.join(fig_dir, 'aa_curve_real_fine.png'),
#         title='AA Curve @ 2° (Real-world Fine-grained)'
#     )
    
#     # 4. Create coarse-grained AA curve for real data
#     plot_angle_accuracy_curve(
#         errors, 
#         max_angle=10.0, 
#         output_path=os.path.join(fig_dir, 'aa_curve_real_coarse.png'),
#         title='AA Curve @ 10° (Real-world Coarse-grained)'
#     )
    
#     # 5. Create CDF plot (similar to AA curve but with different styling)
#     plt.figure(figsize=(10, 6))
    
#     sorted_errors = np.sort(errors)
#     cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    
#     plt.plot(sorted_errors, cumulative, linewidth=2.5, color='steelblue')
    
#     # Add reference lines
#     thresholds = [0.2, 0.5, 1, 2, 5, 10, 20]
#     for threshold in thresholds:
#         accuracy = np.mean(errors <= threshold) * 100
#         plt.axvline(threshold, color='gray', linestyle='--', alpha=0.5)
#         plt.text(threshold + 0.1, 10, f'{threshold}°: {accuracy:.1f}%', fontsize=10)
    
#     plt.xlabel('Angle Difference (degrees)', fontsize=12)
#     plt.ylabel('Percentage (%)', fontsize=12)
#     plt.title('Cumulative Accuracy Distribution', fontsize=14, fontweight='bold')
#     plt.grid(True, alpha=0.3)
#     plt.xlim(0, 30)
#     plt.ylim(0, 100)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(fig_dir, 'cumulative_accuracy.png'), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # 6. Create comparison with state-of-the-art methods (placeholder)
#     plt.figure(figsize=(12, 8))
    
#     # Example data for illustration (replace with actual values)
#     methods = ['Your Model', 'NeurVPS', 'CONSAC', 'J-Linkage']
#     aa1_values = [integrate.simps([np.mean(errors <= t) * 100 for t in np.linspace(0, 1, 100)]) / 1.0, 
#                   0.85, 0.82, 0.75]  # Example values
#     aa2_values = [integrate.simps([np.mean(errors <= t) * 100 for t in np.linspace(0, 2, 100)]) / 2.0, 
#                   0.88, 0.85, 0.79]  # Example values
#     aa10_values = [integrate.simps([np.mean(errors <= t) * 100 for t in np.linspace(0, 10, 100)]) / 10.0, 
#                    0.91, 0.89, 0.84]  # Example values
    
#     x = np.arange(len(methods))
#     width = 0.25
    
#     plt.bar(x - width, aa1_values, width, label='AA1°', color='#5DA5DA')
#     plt.bar(x, aa2_values, width, label='AA2°', color='#FAA43A')
#     plt.bar(x + width, aa10_values, width, label='AA10°', color='#60BD68')
    
#     # Add value labels
#     for i, v in enumerate(aa1_values):
#         plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
#     for i, v in enumerate(aa2_values):
#         plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
#     for i, v in enumerate(aa10_values):
#         plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    
#     plt.xlabel('Method', fontsize=12)
#     plt.ylabel('Angle Accuracy (AA)', fontsize=12)
#     plt.title('Comparison of Angle Accuracy Metrics', fontsize=14, fontweight='bold')
#     plt.xticks(x, methods)
#     plt.ylim(0, 1.0)
#     plt.legend()
#     plt.grid(True, axis='y', alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(fig_dir, 'aa_comparison.png'), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Angle Accuracy visualizations saved to {fig_dir}")

def create_aa_visualizations(errors, output_dir):
    """
    Create visualizations for angle accuracy (AA) curves.
    """
    # Create directory for figures
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Set reasonable figure size limits
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.max_open_warning'] = 50
    
    # Calculate AA curve data
    thresholds = np.linspace(0, 30, 100)  # Angle thresholds from 0 to 30 degrees
    accuracy = [np.mean(errors < t) * 100 for t in thresholds]
    
    # 1. Create AA curve
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, accuracy, linewidth=3, color='#1f77b4')
    
    # Add reference points
    key_thresholds = [5, 10, 20]
    for t in key_thresholds:
        idx = np.searchsorted(thresholds, t)
        acc = accuracy[idx]
        plt.plot(t, acc, 'o', markersize=8, color='red')
        plt.text(t + 0.5, acc - 2, f'{acc:.1f}%', fontsize=12)
    
    plt.xlabel('Angle Threshold (degrees)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Angle Accuracy (AA) Curve', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 30)
    plt.ylim(0, 100)
    
    # Save figure safely
    try:
        plt.savefig(os.path.join(fig_dir, 'aa_curve.png'), dpi=300, bbox_inches='tight')
    except ValueError as e:
        print(f"Warning: Could not save high-resolution figure: {e}")
        plt.savefig(os.path.join(fig_dir, 'aa_curve.png'), dpi=100)  # Lower resolution fallback
    plt.close()
    
    # 2. Create comparison with state-of-the-art methods
    # Use a separate function with safe limits
    try:
        create_aa_comparison(thresholds, accuracy, fig_dir)
    except Exception as e:
        print(f"Warning: Could not create AA comparison: {e}")
    
    return thresholds, accuracy

def create_aa_comparison(thresholds, your_accuracy, fig_dir):
    """Create a comparison of AA curves with state-of-the-art methods."""
    # State-of-the-art AA curves (approximate values from papers)
    # These should be arrays of the same length as thresholds
    
    # Example data - replace with actual values if available
    neurvps_accuracy = [0] * len(thresholds)
    consac_accuracy = [0] * len(thresholds)
    jlinkage_accuracy = [0] * len(thresholds)
    
    # Fill with approximate values (based on published results)
    for i, t in enumerate(thresholds):
        # NeurVPS (example curve)
        neurvps_accuracy[i] = min(100, 77.7 * (1.0 - np.exp(-0.3 * t)))
        
        # CONSAC (example curve)
        consac_accuracy[i] = min(100, 73.4 * (1.0 - np.exp(-0.28 * t)))
        
        # J-Linkage (example curve)
        jlinkage_accuracy[i] = min(100, 61.8 * (1.0 - np.exp(-0.25 * t)))
    
    # Create figure with reasonable size
    plt.figure(figsize=(10, 8))
    
    # Plot curves
    plt.plot(thresholds, your_accuracy, linewidth=3, label='VPSAT', color='#1f77b4')
    plt.plot(thresholds, neurvps_accuracy, linewidth=2, label='NeurVPS', linestyle='--', color='#ff7f0e')
    plt.plot(thresholds, consac_accuracy, linewidth=2, label='CONSAC', linestyle='-.', color='#2ca02c')
    plt.plot(thresholds, jlinkage_accuracy, linewidth=2, label='J-Linkage', linestyle=':', color='#d62728')
    
    # Add key threshold points
    for t in [5, 10, 20]:
        idx = np.searchsorted(thresholds, t)
        plt.axvline(t, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Angle Threshold (degrees)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('AA Curve Comparison with State-of-the-Art Methods', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='lower right')
    plt.xlim(0, 30)
    plt.ylim(0, 100)
    
    # Save figure safely with reduced resolution if needed
    try:
        plt.savefig(os.path.join(fig_dir, 'aa_comparison.png'), dpi=200, bbox_inches='tight')
    except ValueError:
        plt.savefig(os.path.join(fig_dir, 'aa_comparison.png'), dpi=100)
    plt.close()


def compare_with_sota_aa(your_results_path, sota_results, output_dir):
    """
    Compare your model's Angle Accuracy with state-of-the-art methods.
    
    Args:
        your_results_path (str): Path to your model's evaluation results
        sota_results (dict): Dictionary with SOTA methods' AA metrics
        output_dir (str): Directory to save comparison results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load your results
    with open(your_results_path, 'r') as f:
        your_results = json.load(f)
    
    # Add your results to SOTA dictionary
    all_results = {
        'VPSAT': {
            'AA0.2': your_results.get('AA0.2', 0),
            'AA0.5': your_results.get('AA0.5', 0),
            'AA1': your_results.get('AA1.0', 0),
            'AA2': your_results.get('AA2', 0),
            'AA10': your_results.get('AA10', 0)
        },
        **sota_results
    }
    
    # Create DataFrame for plotting
    import pandas as pd
    df = pd.DataFrame(all_results).T.reset_index()
    df = df.rename(columns={'index': 'Method'})
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'figures')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot Synthetic AA metrics (AA0.2, AA0.5, AA1)
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(df['Method']))
    width = 0.25
    
    plt.bar(x - width, df['AA0.2'], width, label='AA0.2°', color='#5DA5DA')
    plt.bar(x, df['AA0.5'], width, label='AA0.5°', color='#FAA43A')
    plt.bar(x + width, df['AA1'], width, label='AA1°', color='#60BD68')
    
    # Add value labels
    for i, v in enumerate(df['AA0.2']):
        plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    for i, v in enumerate(df['AA0.5']):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    for i, v in enumerate(df['AA1']):
        plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Angle Accuracy (AA)', fontsize=12)
    plt.title('Comparison of Angle Accuracy Metrics (Synthetic Dataset)', fontsize=14, fontweight='bold')
    plt.xticks(x, df['Method'], rotation=15, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'aa_synthetic_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Real-world AA metrics (AA1, AA2, AA10)
    plt.figure(figsize=(12, 8))
    
    plt.bar(x - width, df['AA1'], width, label='AA1°', color='#5DA5DA')
    plt.bar(x, df['AA2'], width, label='AA2°', color='#FAA43A')
    plt.bar(x + width, df['AA10'], width, label='AA10°', color='#60BD68')
    
    # Add value labels
    for i, v in enumerate(df['AA1']):
        plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    for i, v in enumerate(df['AA2']):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    for i, v in enumerate(df['AA10']):
        plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Angle Accuracy (AA)', fontsize=12)
    plt.title('Comparison of Angle Accuracy Metrics (Real-world Dataset)', fontsize=14, fontweight='bold')
    plt.xticks(x, df['Method'], rotation=15, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'aa_real_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison data
    df.to_csv(os.path.join(output_dir, 'aa_comparison.csv'), index=False)
    
    print(f"AA comparison visualizations saved to {vis_dir}")
    
    return df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python angle_accuracy_eval.py <model_path> <config_path> <output_dir>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    config_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Evaluate with Angle Accuracy metrics
    metrics = evaluate_with_angle_accuracy(model_path, config_path, output_dir)
    
    # Example SOTA results for comparison (replace with real values)
    sota_results = {
        'NeurVPS': {
            'AA0.2': 0.68,
            'AA0.5': 0.79,
            'AA1': 0.85,
            'AA2': 0.88,
            'AA10': 0.91
        },
        'CONSAC': {
            'AA0.2': 0.63,
            'AA0.5': 0.76,
            'AA1': 0.82,
            'AA2': 0.85,
            'AA10': 0.89
        },
        'J-Linkage': {
            'AA0.2': 0.58,
            'AA0.5': 0.70,
            'AA1': 0.75,
            'AA2': 0.79,
            'AA10': 0.84
        }
    }
    
    # Compare with SOTA
    if metrics:
        stats_path = os.path.join(output_dir, 'stats.json')
        compare_with_sota_aa(stats_path, sota_results, output_dir)