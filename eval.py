import os
import torch
from tqdm import tqdm
import torch.nn as nn

from models.config import C, M
from data.datasets import WireframeDataset
from models.vpsat import VpSatNet
from utils.config_loader import load_config
from utils.helper_functions import preprocess_batch, extract_patches, to_pixel, adjust_vanishing_points
# from scripts.lsd.get_lsd_lines import get_lsd_lines

def evaluate():
    # Load configuration
    config = load_config('./config/model_config.yaml')
    C.update(config)
    M.update(C.model)

    # Set up device
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = C.training.device
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print(f"GPU in use: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available")

    # Set up data loader
    datadir = C.io.datadir
    batch_size = C.training.batch_size
    num_workers = C.io.num_workers
    Dataset = WireframeDataset

    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    val_loader = torch.utils.data.DataLoader(
        Dataset(datadir, split="test"), shuffle=False, **kwargs
    )

    # Initialize the model
    model = VpSatNet(C).to(C.training.device)
    model.eval()  # Set the model to evaluation mode

    # Load the trained model weights
    model_checkpoint = C.io.model_checkpoint
    if os.path.exists(model_checkpoint):
        checkpoint = torch.load(model_checkpoint, map_location=C.training.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model checkpoint from {model_checkpoint}")
    else:
        print(f"Model checkpoint not found at {model_checkpoint}")
        return

    # Initialize metrics
    total_loss = 0.0
    num_samples = 0
    metric_results = []  # Collect per-sample metrics (e.g., angular errors)

    # Start evaluation
    with torch.no_grad():  # No gradients needed for evaluation
        with tqdm(total=len(val_loader), desc="Evaluating", unit="batch") as pbar:
            for images, labels in val_loader:
                org_img_size = images[0].shape[1:]
                # Preprocess images
                images = preprocess_batch(images, C.model.input_image_size)
                image_patches = extract_patches(images, C.model.patch_size)  # Shape: [batch_size, num_patches, 3, PATCH_SIZE, PATCH_SIZE]
                
                # Reshape patches for the model
                batch_size, num_patches, channels, patch_h, patch_w = image_patches.size()
                image_patches = image_patches.view(batch_size * num_patches, channels, patch_h, patch_w)
                num_patches_per_image = images[0].size()[1] * images[0].size()[2] // (C.model.patch_size ** 2)

                # Preprocess labels (vanishing points)
                expanded_labels = labels["vpts"].unsqueeze(1).repeat(1, num_patches_per_image, 1, 1)
                expanded_labels = expanded_labels.view(-1, 3, 3)  # Shape: [4096, 3, 3]
                vpts_2d = to_pixel(expanded_labels, focal_length=C.io.focal_length, image_size=org_img_size[0])
                vpts_2d = adjust_vanishing_points(vpts_2d, org_img_size, C.model.input_image_size)

                # Move inputs to the device
                image_patches = image_patches.to(C.training.device)
                vpts_2d = vpts_2d.to(C.training.device)

                # Forward pass
                outputs = model(image_patches, vpts=vpts_2d, line_segments=None)

                # Compute loss
                loss = nn.functional.mse_loss(outputs, vpts_2d)
                total_loss += loss.item() * batch_size
                num_samples += batch_size

                # Compute per-sample metrics (e.g., angular error)
                angular_errors = compute_angular_error(outputs, vpts_2d)  # Implement this function
                metric_results.extend(angular_errors)

                # Update progress bar
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

    # Compute overall metrics
    avg_loss = total_loss / num_samples
    avg_angular_error = sum(metric_results) / len(metric_results)

    print(f"Evaluation Complete: Avg Loss = {avg_loss:.4f}, Avg Angular Error = {avg_angular_error:.4f} degrees")

def compute_angular_error(predictions, targets):
    """
    Compute angular errors between predicted and target vanishing points.

    Args:
        predictions (torch.Tensor): Predicted vanishing points, shape [N, 3, 3].
        targets (torch.Tensor): Ground-truth vanishing points, shape [N, 3, 3].

    Returns:
        List[float]: Angular errors (in degrees) per vanishing point.
    """
    # Normalize predictions and targets
    predictions = nn.functional.normalize(predictions, p=2, dim=-1)
    targets = nn.functional.normalize(targets, p=2, dim=-1)

    # Compute cosine similarity
    cosine_sim = torch.sum(predictions * targets, dim=-1)  # Shape: [N, 3]
    angular_error = torch.acos(cosine_sim.clamp(-1.0, 1.0))  # Shape: [N, 3]

    # Convert to degrees and return per-sample errors
    return angular_error.mean(dim=-1).cpu().tolist()

if __name__ == '__main__':
    evaluate()
