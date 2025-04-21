import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as TF

def extract_patches(images, patch_size):
    """
    Extract non-overlapping patches from images.

    Args:
        images (torch.Tensor): Input tensor of shape [batch_size, 3, H, W].
        patch_size (int): Size of each patch (patch_size x patch_size).

    Returns:
        patches (torch.Tensor): Output tensor of patches, 
                                shape [batch_size, num_patches, 3, patch_size, patch_size].
    """
    batch_size, channels, height, width = images.size()
    assert height % patch_size == 0 and width % patch_size == 0, \
        "Image dimensions must be divisible by the patch size."

    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_patches = num_patches_h * num_patches_w

    # Use unfold to split the image into patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # Rearrange to get [batch_size, num_patches, 3, patch_size, patch_size]
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, num_patches, channels, patch_size, patch_size)

    return patches


def to_pixel(vpts_3d, focal_length=1.0, image_size=512):
    """
    Converts 3D vanishing points into 2D image-plane pixel coordinates.

    Args:
        vpts_3d (torch.Tensor): Tensor of shape [batch_size, num_vpts, 3].
        focal_length (float): Camera focal length (default: 1.0).
        image_size (int): Size of the square image (default: 512).

    Returns:
        vpts_2d (torch.Tensor): Tensor of shape [batch_size, num_vpts, 2].
    """
    # Ensure the 3rd dimension (z) is non-zero to avoid division by zero
    z = vpts_3d[..., 2].clamp(min=1e-5)  # Shape: [batch_size, num_vpts]

    # Compute pixel coordinates
    x = (vpts_3d[..., 0] / z) * focal_length * (image_size / 2) + (image_size / 2)
    y = -(vpts_3d[..., 1] / z) * focal_length * (image_size / 2) + (image_size / 2)

    # Combine into a single tensor
    vpts_2d = torch.stack([x, y], dim=-1)  # Shape: [batch_size, num_vpts, 2]
    return vpts_2d


def adjust_vanishing_points(vpts, original_size, resized_size):
    """
    Adjust vanishing points to match the resized image dimensions.

    Args:
        vpts (torch.Tensor): Vanishing points of shape [batch_size, num_vpts, 2].
        original_size (tuple): (H_original, W_original).
        resized_size (tuple): (H_resized, W_resized).

    Returns:
        torch.Tensor: Adjusted vanishing points of shape [batch_size, num_vpts, 2].
    """
    H_original, W_original = original_size
    H_resized, W_resized = resized_size, resized_size

    scale_x = W_resized / W_original
    scale_y = H_resized / H_original

    vpts_resized = vpts.clone()
    vpts_resized[..., 0] *= scale_x  # Scale x-coordinates
    vpts_resized[..., 1] *= scale_y  # Scale y-coordinates

    return vpts_resized


def preprocess_image(image, resized_size):
    """
    Preprocess an individual image by resizing and normalizing it.

    Args:
        image (torch.Tensor): Input image tensor of shape [C, H, W].
        target_size (tuple): Target size (H, W).

    Returns:
        torch.Tensor: Preprocessed image tensor of shape [C, target_H, target_W].
    """
    # Convert torch.Tensor to PIL Image if needed
    if isinstance(image, torch.Tensor):
        # Convert from torch.Tensor (C, H, W) to NumPy array (H, W, C) for PIL compatibility
        image = image.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray((image * 255).astype('uint8'))  # Convert to uint8 for PIL

    # Define transform
    transform = transforms.Compose([
        transforms.Resize(resized_size),
        transforms.ToTensor(),  # Convert back to torch.Tensor
    ])

    # Apply transform
    image_resized = transform(image)
    return image_resized


def preprocess_batch(images, resized_size):
    """
    Preprocess a batch of images.

    Args:
        images (torch.Tensor): Batch of images with shape [batch_size, C, H, W].
        target_size (tuple): Target size (H, W).

    Returns:
        torch.Tensor: Preprocessed batch of images with shape [batch_size, C, target_H, target_W].
    """
    # Process each image in the batch
    preprocessed_images = [
        preprocess_image(image, resized_size) for image in images
    ]

    # Stack processed images back into a batch
    return torch.stack(preprocessed_images)


def compute_accuracy(predictions, targets, threshold=0.9):
    """
    Compute accuracy based on cosine similarity.
    Args:
        predictions (torch.Tensor): Predicted vanishing points, shape [N, 3].
        targets (torch.Tensor): Ground-truth vanishing points, shape [N, 3].
        threshold (float): Cosine similarity threshold to consider as correct.
    Returns:
        int, int: Number of correct predictions and total predictions.
    """
    # Normalize predictions and targets
    predictions = predictions / torch.linalg.norm(predictions, dim=-1, keepdim=True)
    targets = targets / torch.linalg.norm(targets, dim=-1, keepdim=True)

    # Compute cosine similarity
    cosine_sim = torch.sum(predictions * targets, dim=-1)  # Shape: [N]
    
    # Count correct predictions based on the threshold
    correct = torch.sum(cosine_sim > threshold).item()
    total = predictions.size(0)
    
    return correct, total


def compute_angular_accuracy(predictions, targets, threshold=5.0):
    """
    Compute accuracy based on angular deviation.
    Args:
        predictions (torch.Tensor): Predicted vanishing points, shape [N, 3].
        targets (torch.Tensor): Ground-truth vanishing points, shape [N, 3].
        threshold (float): Angular deviation threshold (in degrees).
    Returns:
        float: Accuracy as a percentage.
    """
    # Normalize predictions and targets
    predictions = predictions / torch.linalg.norm(predictions, dim=-1, keepdim=True)
    targets = targets / torch.linalg.norm(targets, dim=-1, keepdim=True)

    # Compute angular deviation in degrees
    cosine_sim = torch.sum(predictions * targets, dim=-1).clamp(-1, 1)
    angular_deviation = torch.acos(cosine_sim) * (180 / torch.pi)  # Convert to degrees
    
    # Count correct predictions based on the threshold
    correct = torch.sum(angular_deviation < threshold).item()
    total = predictions.size(0)

    return correct / total * 100  # Return accuracy in percentage


def plot_image_with_vps(image_tensor, gt_vp, pred_vp, title="VP Pixel Location for Prediction vs GT", show_all=True, save_path=None):
    """
    Plots an image with its vanishing points.

    There are a few things to note:
        You might not see an image but just a blank white canvas and the plotted VP.
        This is because the VP might be too far away from the image and image has become significantly smaller.
        If you set show_all to True, you will see all the VPs, but very unlikely to be useful.
        If you set show_all to False, you will see only the first VP. 
        If you do not see the image, just for the next iteration and you might see the image in which the VP is closer to the image.

    
    Args:
        image_tensor (torch.Tensor): shape [3, H, W], values should be in [0, 1].
        gt_vp (torch.Tensor): Ground-truth vanishing point (2D) [2].
        pred_vp (torch.Tensor): Predicted vanishing point (2D) [2].
        title (str): title for the plot.
        show_all (bool): if True, plot all VPs; else only the first one.
        save_path (str or None): path to save the image, if provided.
    """
    img = image_tensor.detach().cpu()
    if img.max() > 1.0:
        img = img / 255.0

    image_np = img.permute(1, 2, 0).numpy()  # [H, W, 3]

    # --- Plot ---
    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)
    plt.title(title)

    colors_gt = ['g', 'b', 'r']
    colors_pred = ['r', 'g', 'b']
    num_vpts = gt_vp.shape[0] if show_all else 1

    for i in range(num_vpts):
        vp1 = gt_vp.detach().cpu().numpy()
        plt.scatter(vp1[0], vp1[1], color=colors_gt[i % len(colors_gt)], marker='o', s=50)
        plt.text(vp1[0]+5, vp1[1]-5, f"GT VP{i+1}", color=colors_gt[i % len(colors_gt)], fontsize=9)

        vp2 = pred_vp.detach().cpu().numpy()
        plt.scatter(vp2[0], vp2[1], color=colors_pred[i % len(colors_pred)], marker='*', s=50)
        plt.text(vp2[0]+5, vp2[1]-5, f"Pred VP{i+1}", color=colors_pred[i % len(colors_pred)], fontsize=9)

    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    else:
        plt.show()



def plot_vp_direction_arrows(image_tensor, gt_vp, pred_vp, save_path=None, title="VP Direction Vectors for Prediction vs GT", arrow_length=100):
    """
    Plot arrows starting from image center toward ground-truth and predicted VP directions.

    To avoid the problem of image not being visible due to the VP being too far away,
    we visualize the direction vectors instead of the actual VPs from the image center.
    The arrows tell the direction of the VPs. It is guranteed that the image and the arrows will be visible.

    Args:
        image_tensor (torch.Tensor): Image tensor [3, H, W], already resized.
        gt_vp (torch.Tensor): Ground-truth vanishing point (2D) [2].
        pred_vp (torch.Tensor): Predicted vanishing point (2D) [2].
        save_path (str or None): If provided, save the figure here.
        title (str): Title for the plot.
        arrow_length (int): Length of the arrows in pixels.
    """
    # Prepare image
    image = image_tensor.detach().cpu()
    if image.max() > 1.0:
        image = image / 255.0
    image = image.permute(1, 2, 0).numpy()  # [H, W, 3]

    width, height, _ = image.shape
    center_x, center_y = width / 2, height / 2

    # Convert VPs to numpy
    gt_vp = gt_vp.detach().cpu().numpy()
    pred_vp = pred_vp.detach().cpu().numpy()

    # Compute direction vectors (unit vectors)
    gt_direction = gt_vp - np.array([center_x, center_y])
    gt_direction = gt_direction / (np.linalg.norm(gt_direction) + 1e-6)  # Normalize

    pred_direction = pred_vp - np.array([center_x, center_y])
    pred_direction = pred_direction / (np.linalg.norm(pred_direction) + 1e-6)

    # Scale to fixed arrow length
    gt_arrow = gt_direction * arrow_length
    pred_arrow = pred_direction * arrow_length

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

    # Green: GT, Red: Prediction
    plt.arrow(center_x, center_y, gt_arrow[0], gt_arrow[1],
              head_width=5, head_length=8, fc='green', ec='green', label='GT VP')

    plt.arrow(center_x, center_y, pred_arrow[0], pred_arrow[1],
              head_width=5, head_length=8, fc='red', ec='red', label='Pred VP')

    # Scatter actual VPs. 
    # plt.scatter([], [], c='lime', marker='-', s=50, label='GT VP')
    # plt.scatter([], [], c='red', marker='-', s=50, label='Pred VP')

    plt.legend(loc='lower right')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()
    else:
        plt.show()


