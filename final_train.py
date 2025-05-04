# final_train.py
import torch
import torch.optim as optim
import os
import yaml
from models.config import C, M
from models.vpsat import VpSatNet
from data.datasets import WireframeDataset
from utils.config_loader import load_config
from models.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

def train_final_model():
    # Load best configuration
    config = load_config('./config/best_model_config.yaml')    
    C.update(config)
    M.update(C.model)

    # Print training configuration
    print("\n=== Training Final Model with Best Configuration ===")
    print(f"Feature Extractor: {C.model.feature_extractor_type}")
    print(f"Transformer Layers: {C.model.transformer.num_layers}")
    print(f"Transformer Heads: {C.model.transformer.num_heads}")
    print(f"VP Head Type: {C.model.vp_head_type}")
    print(f"Patch Size: {C.model.patch_size}")
    print(f"Loss Type: {C.training.loss_type}")
    print(f"Epochs: {C.training.epochs}")
    print(f"Batch Size: {C.training.batch_size}")
    print(f"Learning Rate: {C.optim.lr}")
    print("="*50 + "\n")

    # Set up device and data loaders
    device = torch.device(C.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    Dataset = WireframeDataset
    kwargs = {"batch_size": C.training.batch_size, "num_workers": C.io.num_workers, "pin_memory": True}
    
    # Print dataset information
    print("Loading datasets...")
    train_dataset = Dataset(C.io.datadir, split="train")
    val_dataset = Dataset(C.io.datadir, split="valid")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, **kwargs
    )

    # Initialize model with best configuration
    print("Initializing model...")
    model = VpSatNet(C).to(device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=C.optim.lr,
        weight_decay=C.optim.weight_decay,
        amsgrad=C.optim.amsgrad
    )
    
    # Create output directories
    os.makedirs(C.io.model_save_dir, exist_ok=True)
    os.makedirs(C.io.logdir, exist_ok=True)
    
    # Save configuration used for training
    with open(os.path.join(C.io.model_save_dir, 'final_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create trainer
    trainer = Trainer(model, optimizer, train_loader, val_loader, C)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Add scheduler to trainer
    # If your Trainer class doesn't have a scheduler attribute,
    # you'll need to modify the train method below to use the scheduler
    if hasattr(trainer, "scheduler"):
        trainer.scheduler = scheduler
    
    # Train for specified number of epochs
    print(f"Starting training for {C.training.epochs} epochs...")
    
    # If your Trainer doesn't support a scheduler, you can extend its train method:
    original_train = trainer.train
    
    def train_with_scheduler(epochs):
        for epoch in range(1, epochs + 1):
            # Run original epoch training
            avg_train_loss = trainer.train_epoch()
            avg_val_loss = trainer.validate()
            
            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)
            
            # Save checkpoint, etc. (already handled in original train method)
            trainer.train_losses.append(avg_train_loss)
            trainer.val_losses.append(avg_val_loss)
            
            # Log to tensorboard
            trainer.writer.add_scalar("Loss/train", avg_train_loss, epoch)
            trainer.writer.add_scalar("Loss/val", avg_val_loss, epoch)
            
            # Check if this is the best model
            is_best = avg_val_loss < trainer.best_val_loss
            if is_best:
                trainer.best_val_loss = avg_val_loss
            
            # Save checkpoint
            trainer.save_checkpoint(epoch, avg_train_loss, avg_val_loss, is_best)
            
            # Print progress
            print(f"Epoch {epoch}/{epochs}: "
                  f"Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}, "
                  f"LR = {optimizer.param_groups[0]['lr']:.6f}")
    
    # Use the appropriate training method based on whether scheduler is supported
    if hasattr(trainer, "scheduler"):
        trainer.train(C.training.epochs)
    else:
        train_with_scheduler(C.training.epochs)
    
    # Plot training curves
    trainer.plot_losses()
    
    print("Final model training complete!")
    
    # Return path to best checkpoint
    return os.path.join(C.io.model_save_dir, "checkpoint_best.pth.tar")

if __name__ == '__main__':
    final_model_path = train_final_model()
    print(f"Best model saved at: {final_model_path}")
    print("\nTo evaluate the model on the test set, run:")
    print(f"python -m angular_error_evaluation {final_model_path} config/best_model_config.yaml checkpoint/best_model")