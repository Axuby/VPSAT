import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

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

