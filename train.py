# train.py
import torch
import torch.optim as optim

from models.config import C, M
from models.vpsat import VpSatNet, multi_vp_loss
from data.datasets import WireframeDataset
from utils.config_loader import load_config
from models.trainer import Trainer


def main():
    config = load_config('./config/model_config.yaml')    
    C.update(config)
    M.update(C.model)

    device = torch.device(C.training.device if torch.cuda.is_available() else "cpu")
    Dataset = WireframeDataset
    kwargs = {"batch_size": C.training.batch_size, "num_workers": C.io.num_workers, "pin_memory": True}
    
    train_loader = torch.utils.data.DataLoader(Dataset(C.io.datadir, split="train"), shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(Dataset(C.io.datadir, split="valid"), shuffle=False, **kwargs)

    model = VpSatNet(C).to(device)
    optimizer = optim.Adam(model.parameters(), lr=C.optim.lr)

    trainer = Trainer(model, optimizer, train_loader, val_loader, C)
    trainer.train(C.training.epochs)


if __name__ == '__main__':
    main()
