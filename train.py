import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models.config import C, M
from data.datasets import WireframeDataset
from models.vpsat import VpSatNet, multi_vp_loss
from utils.config_loader import load_config
from utils.helper_functions import *

# from scripts.lsd.get_lsd_lines import get_lsd_lines


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    """
    Save model checkpoint.
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        train_loss (float): The training loss at this epoch.
        val_loss (float): The validation loss at this epoch.
        path (str): The file path to save the checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    } 
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")


def test(model, test_loader):
    print("Testing the model on the test set...")
    model.eval()
    test_loss = 0.0
    correct_cosine = 0
    total_samples = 0
    total_angular_accuracy = 0.0

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Testing", unit="batch") as pbar:
            for images, labels in test_loader:
                # Preprocess data
                images = preprocess_batch(images, C.model.input_image_size)
                image_patches = extract_patches(images, C.model.patch_size)
                batch_size, num_patches, channels, patch_h, patch_w = image_patches.size()
                image_patches = image_patches.view(batch_size * num_patches, channels, patch_h, patch_w)

                num_patches_per_image = images[0].size()[1] * images[0].size()[2] // (C.model.patch_size ** 2)
                expanded_labels = labels["vpts"].unsqueeze(1).repeat(1, num_patches_per_image, 1, 1)
                expanded_labels = expanded_labels.view(-1, 3, 3)
                vpts_2d = to_pixel(expanded_labels, focal_length=C.io.focal_length, image_size=images[0].shape[1])
                vpts_2d = adjust_vanishing_points(vpts_2d, images[0].shape[1:], C.model.input_image_size)

                image_patches = image_patches.to(C.training.device)
                vpts_2d = vpts_2d.to(C.training.device)

                # Get predictions
                outputs = model(image_patches, vpts=None, line_segments=None)
                loss = multi_vp_loss(outputs, vpts_2d)
                test_loss += loss.item()

                # Compute accuracy
                batch_correct, batch_total = compute_accuracy(outputs, vpts_2d)
                correct_cosine += batch_correct
                total_samples += batch_total

                # Compute angular accuracy
                batch_angular_accuracy = compute_angular_accuracy(outputs, vpts_2d)
                total_angular_accuracy += batch_angular_accuracy

                pbar.set_postfix(test_loss=loss.item(), batch_accuracy=(batch_correct / batch_total) * 100)
                pbar.update(1)

    avg_test_loss = test_loss / len(test_loader)
    avg_cosine_accuracy = (correct_cosine / total_samples) * 100
    avg_angular_accuracy = total_angular_accuracy / len(test_loader)

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy (Cosine Similarity): {avg_cosine_accuracy:.2f}%")
    print(f"Test Angular Accuracy (<5 degrees): {avg_angular_accuracy:.2f}%")

    return avg_test_loss, avg_cosine_accuracy, avg_angular_accuracy



def train():
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

    # Set up data loaders
    batch_size = C.training.batch_size
    num_workers = C.io.num_workers
    datadir = C.io.datadir
    Dataset = WireframeDataset
    
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    train_loader = torch.utils.data.DataLoader(
        Dataset(datadir, split="train"), shuffle=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        Dataset(datadir, split="valid"), shuffle=False, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        Dataset(datadir, split="test"), shuffle=False, **kwargs
    )

    # Initialize the model
    model = VpSatNet(C).to(C.training.device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=C.optim.lr)
    # criterion = nn.MSELoss()  # L2 loss for vanishing point prediction
    best_val_loss = float("inf")  # Track the best validation loss

    train_losses = []
    val_losses = []
    torch.cuda.empty_cache()

    # Training loop
    for epoch in range(C.training.epochs):
        model.train()
        running_loss = 0.0

        # Initialize progress bar for the epoch
        with tqdm(total=len(train_loader), desc=f"Training Epoch [{epoch+1}/{C.training.epochs}]", unit="batch") as pbar:
            for images, labels in train_loader:
                
                # Add line segment detection here
                # line_segments = get_lsd_lines(images)
                # line_segments = line_segments.to(C.training.device)
                
                # Get images ready
                images = preprocess_batch(images, C.model.input_image_size)
                image_patches = extract_patches(images, C.model.patch_size)  # Shape: [batch_size, num_patches, 3, PATCH_SIZE, PATCH_SIZE]
                 # Reshape patches for the model (e.g., flatten spatial dimensions if needed)
                # [batch_size * num_patches, 3, PATCH_SIZE, PATCH_SIZE] or other forms
                batch_size, num_patches, channels, patch_h, patch_w = image_patches.size()
                image_patches = image_patches.view(batch_size * num_patches, channels, patch_h, patch_w)
                # Expand labels to match the number of patches per image
                num_patches_per_image = images[0].size()[1] * images[0].size()[2] // (C.model.patch_size ** 2)


                # Get labels ready
                expanded_labels = labels["vpts"].unsqueeze(1).repeat(1, num_patches_per_image, 1, 1)
                # Flatten labels to align with the flattened patch representation
                expanded_labels = expanded_labels.view(-1, 3, 3)  # Shape: [4096, 3, 3]
                # Convert 3D vanishing points to 2D
                vpts_2d = to_pixel(expanded_labels, focal_length=C.io.focal_length, image_size=images[0].shape[1])  # Shape: [batch_size, num_vpts, 2]
                # Preprocess the vpts
                vpts_2d = adjust_vanishing_points(vpts_2d, images[0].shape[1:], C.model.input_image_size)

                # Put patches and vpts on the device
                image_patches = image_patches.to(C.training.device)
                vpts_2d = vpts_2d.to(C.training.device)
                
                optimizer.zero_grad()
                outputs = model(image_patches, mode="train")

                # Calculate losses
                # vp_loss = multi_vp_loss(outputs, vpts_2d)

                # Define Cosine Similarity
                cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
                # Compute cosine similarity along the last dimension (2D vectors)
                similarity = cosine_similarity(outputs, vpts_2d)
                # Convert to loss (1 - similarity) and take the mean
                vp_loss = 1 - similarity.mean()
                
                total_loss = vp_loss 

                total_loss.backward()
                optimizer.step()

                # Update running loss and pr ogress bar
                running_loss += total_loss.item()

                pbar.set_postfix(loss=total_loss.item())
                pbar.update(1)

        # Calculate average losses for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f"Epoch [{epoch+1}/{C.training.epochs}], Training Loss: {avg_train_loss:.4f}")

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with tqdm(total=len(val_loader), desc=f"Validation Epoch[{epoch+1}/{C.training.epochs}]", unit="batch") as qbar:
            with torch.no_grad():
                for images, labels in val_loader:
                    images = preprocess_batch(images, C.model.input_image_size)
                    image_patches = extract_patches(images, C.model.patch_size)
                    batch_size, num_patches, channels, patch_h, patch_w = image_patches.size()
                    image_patches = image_patches.view(batch_size * num_patches, channels, patch_h, patch_w)

                    # Preprocess labels
                    num_patches_per_image = images[0].size()[1] * images[0].size()[2] // (C.model.patch_size ** 2)
                    expanded_labels = labels["vpts"].unsqueeze(1).repeat(1, num_patches_per_image, 1, 1)
                    expanded_labels = expanded_labels.view(-1, 3, 3)  # Shape: [batch_size * num_patches, 3, 3]
                    vpts_2d = to_pixel(expanded_labels, focal_length=C.io.focal_length, image_size=images[0].shape[1])
                    vpts_2d = adjust_vanishing_points(vpts_2d, images[0].shape[1:], C.model.input_image_size)

                    # Move data to the device
                    image_patches = image_patches.to(C.training.device)
                    vpts_2d = vpts_2d.to(C.training.device)

                    # Forward pass without teacher offsets
                    outputs = model(image_patches, vpts=None, line_segments=None)

                    # Define Cosine Similarity
                    cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
                    # Compute cosine similarity along the last dimension (2D vectors)
                    similarity = cosine_similarity(outputs, vpts_2d)
                    # Convert to loss (1 - similarity) and take the mean
                    vp_loss = 1 - similarity.mean()
                    # loss = multi_vp_loss(outputs, vpts_2d)

                    val_loss += vp_loss.item()

                    qbar.set_postfix(val_loss=vp_loss.item())
                    qbar.update(1)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{C.training.epochs}], Validation Loss: {avg_val_loss:.4f}")

        # Save the model if it has the best validation loss so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(C.io.model_save_dir, f"best_model_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch+1, avg_train_loss, avg_val_loss, checkpoint_path)

    print("Training Complete")

    # Plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Progression")
    plt.legend()
    plt.grid()
    
    # Save the plot to a file
    plt.savefig("training_validation_loss.png", dpi=300, bbox_inches="tight")
    
    plt.show()

    # After training, evaluate on the test set
    test_loss, test_predictions, test_ground_truths = test(model, test_loader)
    print(f"Final Test Loss: {test_loss:.4f}")

    # Save test results if needed
    torch.save({"predictions": test_predictions, "ground_truths": test_ground_truths},
               os.path.join(C.io.model_save_dir, "test_results.pth"))
    print("Test results saved.")
    

if __name__ == '__main__':
    train()