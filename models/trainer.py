# trainer.py
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
from utils.helper_functions import *


class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.C = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        self.outdir = config.io.model_save_dir
        self.logdir = config.io.logdir
        loss_type = config.training.get('loss_type', 'mse')  # Default to 'mse' if not present
        self.loss_function = get_loss_function(loss_type)
        self.loss_function = get_loss_function(config.training.loss_type)
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.outdir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.logdir, "tensorboard"))
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

        logger.add(os.path.join(self.logdir, "training.log"), rotation="1 MB")
        logger.info("Trainer initialized")


    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        path_latest = os.path.join(self.outdir, "checkpoint_latest.pth.tar")
        torch.save(checkpoint, path_latest)
        logger.info(f"Checkpoint saved at {path_latest}")
        if is_best:
            path_best = os.path.join(self.outdir, "checkpoint_best.pth.tar")
            torch.save(checkpoint, path_best)
            logger.info(f"Best checkpoint updated at {path_best}")


    def _write_metrics(self, epoch, train_loss, val_loss):
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/val", val_loss, epoch)

    def _run_epoch(self, loader, mode="train"):
        is_train = mode == "train"
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        epoch_loss = 0.0
        context = torch.enable_grad() if is_train else torch.no_grad()

        with context:
            for images, labels in tqdm(loader, desc=mode.capitalize(), unit="batch"):
                original_image_size = images[0].shape[1:]
                images = preprocess_batch(images, self.C.model.input_image_size)
                patches = extract_patches(images, self.C.model.patch_size)
                b, n, c, h, w = patches.shape
                patches = patches.view(b * n, c, h, w).to(self.device)

                # n_patches = images[0].size()[1] * images[0].size()[2] // (self.C.model.patch_size ** 2)
                # labels = labels["vpts"].unsqueeze(1).repeat(1, n_patches, 1, 1).view(-1, 3, 3)
                # vpts_2d = to_pixel(labels, self.C.io.focal_length, original_image_size[0])
                # vpts_2d = adjust_vanishing_points(vpts_2d, original_image_size, self.C.model.input_image_size).to(self.device)
                # Don't repeat labels for each patch - keep them at image level
                vpts_3d = labels["vpts"].to(self.device)  # Shape: [batch_size, 3, 3]
                vpts_2d = to_pixel(vpts_3d, self.C.io.focal_length, original_image_size[0])
                vpts_2d = adjust_vanishing_points(vpts_2d, original_image_size, self.C.model.input_image_size)

                if is_train:
                    self.optimizer.zero_grad()

                outputs = self.model(patches)
                # similarity = self.cosine_similarity(outputs, vpts_2d)
                # loss = 1 - similarity.mean()
                loss = self.loss_function(outputs, vpts_2d)

                if is_train:
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()

        return epoch_loss / len(loader)

    def validate(self):
        return self._run_epoch(self.val_loader, mode="validate")
    

    def train_epoch(self):
        return self._run_epoch(self.train_loader, mode="train")
    

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            start = timer()
            avg_train_loss = self.train_epoch()
            self.train_losses.append(avg_train_loss)
            logger.info(f"Epoch {epoch}: Training Loss = {avg_train_loss:.4f}")

            avg_val_loss = self.validate()
            self.val_losses.append(avg_val_loss)
            logger.info(f"Epoch {epoch}: Validation Loss = {avg_val_loss:.4f}")

            self._write_metrics(epoch, avg_train_loss, avg_val_loss)

            is_best = avg_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = avg_val_loss

            self.save_checkpoint(epoch, avg_train_loss, avg_val_loss, is_best)
            logger.info(f"Epoch {epoch} completed in {timer() - start:.2f} seconds")

        self.plot_losses()
        logger.info("Training complete")

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.outdir, "loss_plot.png"))
